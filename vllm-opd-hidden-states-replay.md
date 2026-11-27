# vLLM 社区 OPD Hidden States Replay 实现调研

> **调研范围（本版已收窄）**：只收 **hidden state 的捕获 / 导出 / 重放**（capture → export → replay）这条线，以及直接决定它能否落地的接口、交付、兼容性讨论。
> **明确排除**：OPD 的 top-K 打分链路（`prompt_logprob_token_ids` 那一套）、scoring 的前缀复用、logprob 计数类 PR。它们只是"同一件事的另一种载体"，不是 hidden states replay 本体的实现，见 §1.2。
> 数据快照：2026-11-25（`merged_at` 逐个核对）
> 代码依据：本机 `vllm-latest/` 检出，commit `10e8e921b8`（2026-10-19）

---

## 0. TL;DR

1. **hidden states replay 在 vLLM 里叫 hidden-state capture / export，没有叫 "replay" 的正式术语**；"replay" 发生在 trainer 侧（用 rollout 时捕获的 hidden 重放分布）。
2. **这一条线目前 0 个实现合入 main**。相关 5 个 PR 全部 open/closed-unmerged，唯一已合入的 hidden 导出能力是 `method=extract_hidden_states`（面向 EAGLE-3/DFlash drafter 训练，按"层 id"取 hidden，**不是**按 token 位置窗口取，见 §8）。
3. **主推实现是 [#57736](https://github.com/vllm-project/vllm/pull/57736)**（vx120，open，`needs-rebase`）：runner 级按**绝对 token 位置窗口**捕获 lm_head 输入行，复用 AuxOutput Connector 交付，同时覆盖 rollout 生成窗口与 OPD teacher 的 prompt 行。
4. **RFC [#65056](https://github.com/vllm-project/vllm/issues/65056) 已宣布不再另起实现，复用 #57736 的 input producer**；它自己收缩为"teacher 侧契约 + 校验方法 + 交付/生命周期要求"。
5. **三条实现路线并存且未收敛**：runner 级窗口捕获（#57736）、KV-connector 级窗口导出（#62411）、typed token-output 通道（#59543）。三者互相重叠，社区共识是"应对齐而不是各合一套"。
6. **最大阻塞不是代码量，是三个没被裁决的问题**：API 形状（`aux_outputs` vs `hidden_state_window`）、**全引擎强制同步调度**（实测 −21.3% 吞吐）、以及**重建契约只对 Qwen3 成立**。

---

## 1. 范围与术语

### 1.1 什么算 hidden states replay

本文只收满足下面三条的提案：

1. 从模型内部**取 hidden state**（不是 logprob、不是 logits、不是 routing）；
2. 取的范围是**按 token 位置**（窗口）或**按层**的**子集**（而非全序列 dump）；
3. 目的是**交给外部训练侧使用**（rollout 采集 / teacher 打分 / 特征导出）。

### 1.2 明确排除（它们属于 OPD 打分链路，不是 hidden 本体）

| 编号 | 为什么排除 |
|---|---|
| [#56860](https://github.com/vllm-project/vllm/issues/56860) per-position target token logprobs | 只传 **K 个 token 的分数**，不涉及 hidden |
| [#54335](https://github.com/vllm-project/vllm/pull/54335) fixed-token prefill scoring（已合入） | 打分实现的 M1，同上 |
| [#60960](https://github.com/vllm-project/vllm/pull/60960)（已合入）、[#57269](https://github.com/vllm-project/vllm/pull/57269)（已合入） | scoring 的前缀缓存边界与 connector 命中语义 |
| [#60366](https://github.com/vllm-project/vllm/issues/60366) bounded prefix reuse for scoring | 同上 |
| [#68813](https://github.com/vllm-project/vllm/issues/68813)、[#66177](https://github.com/vllm-project/vllm/pull/66177)、[#68769](https://github.com/vllm-project/vllm/pull/68769) 等 aux hidden state bugfix | 修的是 **drafter 训练的 aux hidden 层捕获**，与按位置窗口的 replay 无关 |

这些内容属于 OPD 的 top-K 打分链路，是当前唯一能直接用的替代方案，但不在本文讨论范围，故不展开。

### 1.3 "replay" 一词在 vLLM 的三种含义（别混）

| 术语 | 出处 | 含义 | 与本文关系 |
|---|---|---|---|
| hidden-state capture / export | #56916 / #57736 / #62411 | 按请求/窗口捕获 lm_head 输入行并导出 | **本文对象** |
| MoE routing replay（R3） | docs `training/routed_experts` | 回放每 token 的专家选择 | 同族 artifact，机制可复用（AuxOutput），但不是 hidden |
| distribution replay（sampling mask） | docs `training/sampling_mask` | 回放采样 support set | 同族，但不是 hidden |
| decoder replay | #62651 等 | DeepSeek-V4.1 的行裁剪与重算 | 同名不同物，无关 |

---

## 2. 现状：0 个实现合入

| PR | 路线 | 状态（2026-11-25 核对） |
|---|---|---|
| [#57736](https://github.com/vllm-project/vllm/pull/57736) | runner 级窗口捕获（主推） | **open**，`merged_at: null`，`needs-rebase`，13 评论，最后更新 2026-11-21 |
| [#62411](https://github.com/vllm-project/vllm/pull/62411) | KV-connector 级窗口导出 | **open draft**，`merged_at: null`，`mergeable: dirty` |
| [#59543](https://github.com/vllm-project/vllm/pull/59543) | typed token-output 通道 | **open**，`merged_at: null`，`mergeable_state: unstable` |
| [#57185](https://github.com/vllm-project/vllm/pull/57185) | 只返回最后一个 prompt 位置 | **open**，`merged_at: null`，`needs-rebase` |
| [#57663](https://github.com/vllm-project/vllm/pull/57663) | 只加窗口字段（前置） | **closed，未合并** |

旁证：`docs/features/hidden_states.md`、`docs/training/hidden_states.md` 在 main 上均为 404；`docs/training/` 下只有 scoring / R3 / sampling mask 三篇。

**唯一已合入的 hidden 相关能力**是 `method=extract_hidden_states`（drafter 训练用），见 §8——它不能替代本文要的东西。

---

## 3. 治理：这件事挂在谁下面

### 3.1 OPD roadmap 的 work item G（原文）

[#62175 Roadmap Q4 2026 OPD Teacher Serving](https://github.com/vllm-project/vllm/issues/62175) 里，hidden-state 是**最后一个、条件触发**的工作项：

> **G. Hidden-state output** — Starts with a consumer that needs teacher hidden states.
> - [ ] Identify the consumer, layers, positions, normalization and precision, and compare existing hidden-state output paths.
> - [ ] Fill teacher-side gaps in chunked prefill or output delivery, with lifecycle and memory validation.
>
> G is a separate training input from candidate scoring; large outputs use D's delivery path. [#65056] coordinates teacher requirements around [#57736], which includes prompt rows.

读法：**hidden 被定位为"consumer 驱动"的独立输入**，不与打分混，且必须复用交付路径（D）与 AuxOutput 底座，不允许新开一条传输。

### 3.2 交付底座已就绪（这是 hidden 能"只写捕获"的前提）

| PR | 状态 | 对 hidden 的意义 |
|---|---|---|
| [#45635](https://github.com/vllm-project/vllm/pull/45635) AuxOutput Connector（block-keyed 存储 + 发布） | **已合入 2026-09-21** | hidden 行可以挂进去，不用自建生命周期 |
| [#62056](https://github.com/vllm-project/vllm/pull/62056) R3 支持 P/D 分离 | **已合入 2026-11-16** | 跨 prefill/decode 实例的 artifact 交付已验证 |
| [#45221](https://github.com/vllm-project/vllm/issues/45221) AuxOutput tracker | open | 后续 by-reference（Mooncake）交付的入口 |

所以 #57736 的写法是"捕获 + 复用既有 connector"，而 #62411 是"把捕获做进 connector 本身"——两者都在蹭这套底座。

---

## 4. RFC

### 4.1 [#56916](https://github.com/vllm-project/vllm/issues/56916) — Request-Scoped Hidden-State Capture Windows（vx120，2026-09-15，open）

**这是 hidden-state 捕获的设计原点**，也是 #57736 的 RFC。动机：应用只要**一小段**生成 token 的 hidden，全序列捕获会浪费显存带宽、D2H 拷贝与序列化。

提议 API（示意；最终 #57736 改成了 `aux_outputs`）：

```yaml
hidden_state_capture:
  mode: fixed
  coordinate: response      # 默认 response-relative
  start: 1000               # 含
  end: 2000                 # 不含
  layer_ids: [1, 8, 16]
```

坐标：response offset `r` 对应"预测 response token `r` 的那个 hidden 位置"，即绝对位置 `prompt_len - 1 + r`；另提供 `absolute`。

每请求状态机：

```
DISABLED → ARMED → CAPTURING → FINISHED
```

设计要求（原文提炼）：

- 捕获元数据在**请求接纳时**生成，随调度传播到 worker；**不得**放进全局开关或公共 sampling params 的不受支持字段；
- **不得每步注册/移除 PyTorch hook**，应在正常执行路径里用 per-request 选择控制导出；
- 只有落在窗口内的行可以 gather / 跨层打包 / D2H / 返回；`ARMED`、`FINISHED` 不得保留张量；
- 返回 `hidden_positions`（权威，允许是窗口子集）、`hidden_window_start/end`、`layer_ids`、`request_id`；
- 不得为捕获打开全局 `output_hidden_states=True`，也不得悄悄退回 eager；不支持就**明确报 capability error**。

观测指标：`vllm/hidden_capture_requests`、`..._rows_planned`、`..._rows_captured`、`..._bytes_copied`、`..._copy_ms`。

性能预算：同模型/bs/seqlen/投机配置下，**窗口外 decode 吞吐中位数不得比关闭捕获低 3% 以上**。

评论区有一条有价值的实现建议（HEMU-VERMA）：复用 `gpu_model_runner.py` 里 `_get_prompt_logprobs_dict` 已有的机制（按 `num_computed_tokens` 做 per-request 切片 + 在 request 对象挂 in-progress buffer），并给出 `SamplingParams.hidden_state_window: tuple[int,int]|None` 的备选 API 与 18 个 interval-math 单测。作者回复：窗口保持 response-relative，首版不做 `PoolingParams`，由 #57736 承接端到端实现。

### 4.2 [#65056](https://github.com/vllm-project/vllm/issues/65056) — Selected input-row hidden-state export for full-vocabulary OPD teachers（NolenLiang，2026-11-02，open）

**这是"teacher 侧契约"的 RFC**，也是理解取舍最好的一篇。它已经放弃自研：

> **Coordination update, 2026-11-03.** #57736 at `dfc9982` now implements absolute input-row capture for OPD teachers, bounded prefix hits, HTTP output, and a Qwen3 reconstruction contract. ... **We will reuse that input producer rather than start a parallel implementation of the same feature.**

它保留下来的核心内容：

**(a) 为什么是 hidden 而不是 logits / top-K**

| 每个选中位置的 payload（Qwen3-14B，H=5120，V=151936） | 原始大小 | 全词表信息 |
|---|---:|---|
| 采样 token 的 logprob | 4 B | 否 |
| top-64 logprobs + IDs | ≈512 B | 否（尾部丢失） |
| **BF16 final hidden** | **10 KiB** | **是**（需配套完整 head 变换） |
| BF16 / FP32 logits | 296.75 / 593.5 KiB | 是 |

hidden/logits 压缩比 **29.675× / 59.35×**。原文明确："这是务实的接口选择，不声称不存在更小的无损压缩或 model-specific 表示"。

**(b) 位置与捕获点**

- 行 `p` 预测 token `p+1`；prompt 长 `P`、response 长 `R` → teacher 行区间 `[P-1, P+R-1)`；rollout 的 response offset `[a,b)` 映射到 `[P-1+a, P-1+b)`。
- 捕获点是 **lm_head 的输入**，包含 final norm 及之前的 model-specific collapse/projector；**排除** pooler 的 L2 normalization。
- 重建必须应用完整 head 语义（bias、scale、softcap、effective vocab）。
- 结果需记录 positions/segments、native dtype、shape、capture point、head/model 身份。

**(c) 重建契约的模型限制**

目前仅 **标准、未量化 Qwen3（无 LoRA、无自定义 projection override）**。原文："Qwen 的测量不能证明 DeepSeek、hybrid、量化 head 或其他 TP 配置"。其他模型可以只返回 hidden、不承诺重建契约。

**(d) prefix cache 策略（有争议）**

- 每个需要的行都必须被计算；最大前缀命中 = 第一个被选中行（受 block 对齐影响）。
- #57736 已把 prompt capture 纳入 `Request.get_prefix_cache_hit_limit()` 并支持跨 non-sampling chunk 携带输入行；**缺行要 fail 而不是静默返回部分张量**。
- RFC 提议的"审计态"策略是**默认 skip reads**，只在已验证的 BI 配置里显式开有界复用，且"BI flag 本身不得自动开启 reads 或替代 row limit"。
- 并明确指出这与 #57736 当前行为不同（hidden-only 捕获在 APC 开启时可自动读有界前缀，无 BI 硬门控），**需要作者/maintainer 拍板**。

**(e) 交付与生命周期**

- 内部复用 vLLM 的 tensor-aware MessagePack + auxiliary buffer，**不要新开 hidden 专用 IPC**；#57736 目前是 JSON/Base64。
- bulk 后续建议 octet-stream（沿用 pooling `bytes` 约定，native dtype + 显式字节序 + 有界元数据）；不要把全位置或可选 token id 塞进无界响应头。
- 每请求 cap + 进程/阶段级聚合上限；GPU 源在 D2H 完成前存活；CPU/IPC/HTTP backing buffer 在消费者释放前存活；abort/preemption 抑制陈旧 attempt 但**不释放** in-flight 存储；每请求最多发布一个逻辑结果；ASGI send 完成 ≠ trainer 收到。
- by-reference 交付（AuxOutput/Mooncake）是独立后续项。

**(f) 两道独立审计**

1. 部署路径等价门：相同精度与完整 logits 语义下，重建 vs engine 原生结果；
2. 公共 FP32 诊断：把两侧 hidden 投影到同一参考，调查 backbone/cache 差异；记录 operand/output dtype、backend、TF32、effective vocab、metric 精度。

**(g) SC（behavior replay）的额外要求**：可复用结果契约，但还需要 generating behavior model/head 版本 + 实际 sampling 变换/支撑集——**固定 teacher 的 identity 不等于 behavior replay**。

---

## 5. PR

### 5.1 [#57736](https://github.com/vllm-project/vllm/pull/57736) — runner 级窗口捕获（**主推**，open）

标题：*feat: capture selected hidden-state rows for rollout windows and teacher scoring*

**架构**：`lm_head` 输入行 → per-request 绝对窗口选择 → AuxOutput Connector 交付 → Python frontend 汇总 → HTTP/离线返回。

**API**（已从 RFC 的 `extra_args["hidden_state_capture"]` 演进为 `aux_outputs`）：

```python
# 1) rollout 生成窗口：行 p 预测 token p+1
SamplingParams(aux_outputs=("hidden_states",),
               hidden_states_start=start, hidden_states_end=end)

# 2) teacher 打分 y|x：多采的那个 token 不算
SamplingParams(max_tokens=1, aux_outputs=("prompt_hidden_states",),
               hidden_states_start=len(x)-1,
               hidden_states_end=len(x)+len(y)-1)
```

引擎开关：`--enable-return-hidden-states --no-async-scheduling`。

**实现要点**：

- 生成行区间 `[P-1, P+O-1)`，prompt 行区间 `[0, P-1)`，可同时请求；`CompletionOutput` 分别携带张量与 start（生成用 `weight_versions`，prompt 用 `prompt_hidden_states_weight_versions`）；空交集返回 `[0, hidden_size]`。
- GPU 行在 output-copy stream 上**异步**拷贝；connector 只有**一个** scheduler 抽取入口；prompt chunk 走内部 IPC，无 worker carry 或 scheduler payload buffer。
- frontend 用 request-local `HiddenStatesAccumulator` 累积（没请求就不创建），**preemption replay 时保留第一次计算结果**，完成时才交付。
- 缺行只让**该请求**失败（`code="hidden_states_unavailable"`），同 batch 其他请求继续；新增 `RequestOutput.error` 承载结构化错误。
- HTTP 用 base64 `.npy`，BF16 以 uint16 bits + 显式 dtype；head 元数据经 `collective_rpc("get_hidden_states_metadata")` 与 `GET /inference/v1/hidden_states_metadata`。
- prompt 捕获默认 `skip_reading_prefix_cache=True`；显式复用只限首个被请求位置之前的行；拒绝 fast-prefill。

**性能数据**（RTX 5090 32GB，TP1，BF16，sync engine）：

a) OPD teacher 打分（Qwen3-4B，4 请求共享 512 token prompt，各 2048 行 = 8192 行）

| 路径 | teacher batch | 返回 payload |
|---|---:|---:|
| 不捕获（参考） | 373.64 ms | — |
| **prompt hidden** | **402.38 ms（+7.69%）** | **40 MiB** |
| 自动 `prompt_logprobs=128` | 1670.78 ms | 25.14 MiB JSON |
| 128 个指定候选 | 428.33 ms | 4 MiB + ids |

→ hidden 的 Python 侧结果延迟是自动 top-128 的 **1/4.15**；序列化单项：hidden `.npy` 20.4 ms vs top-128 JSON 1522.9 ms。消费侧 head-only H2D 2.29 ms，chunked BF16 投影 + FP32 log-softmax 45.53 ms。

b) rollout 窗口（Qwen3-4B + DSpark block7，bs=8，1056 token，只取 final hidden，窗口 `[1000,1032)`）

| 模式 | batch 时间 |
|---|---:|
| 允许捕获但请求不要求 | 4450.92 ms |
| 全量生成捕获 | 4614.40 ms |
| **32 行窗口** | **4473.26 ms** |
| 窗口从未到达 | 4464.01 ms |

→ 窗口 vs 全量 **+3.16% 吞吐**；窗口 vs 不捕获 +0.50% 延迟；payload 41.25 MiB → **1.25 MiB**；窗口行与全量捕获同位置**逐位相等**。

c) Qwen3-8B 交叉验证（untied `lm_head`）：hidden 669.59 vs 参考 638.79 ms（+4.82%），payload 64 MiB；重建 max abs diff = 0、mean KL/TV = 0。

d) **同步调度的价（最重要的一条）**：同引擎、Qwen3-4B、CUDA graph、bs=32、128 输出 token —— **3962.60 tok/s（async） vs 3116.72 tok/s（sync）**，吞吐 **−21.3%**、batch 延迟 **+27.1%**。

**已知限制**：只支持 exact unquantized `Qwen3ForCausalLM` logits 路径；CUDA Model Runner V2；TP 只验到 1；无 LoRA/PP/CP/adaptive verification；**engine 级必须同步调度**；API 只在 Python frontend（Rust 不暴露）；store/Mooncake 流式交付、async scheduling、drafter 辅助层都是后续项。

### 5.2 [#62411](https://github.com/vllm-project/vllm/pull/62411) — KV-connector 级窗口导出（open **draft**）

作者自己定位为"与 #57736 对照的另一种窗口化捕获"。改动全在 connector 侧：

- `hidden_state_window.py`：窗口解析/校验，坐标 `abs = P - 1 + r`；
- `hidden_state_cache.py`：**masked store**，窗口外的行不写 cache；
- `ExampleHiddenStatesConnector`：只导出窗口行、写 layout 元数据（layer ids / 窗口边界 / 是否含 final layer）、**有界 pinned memory 分批 D2H**（大窗口回退 pageable）、取消/失败清理；
- 配置校验：V1 runner、PP=1、无 async scheduling、prefix caching 关闭、CUDA。

| 维度 | #62411（connector 级） | #57736（runner 级） |
|---|---|---|
| 与真实投机 drafter 共存 | **否**（占用 `method=extract_hidden_states`） | 能 |
| 核心 scheduler/engine 改动 | 无（connector + 小 hook） | scheduler、request、protocol、output processor |
| 显存 | 仍按全序列分配 KV cache，只窗口化写入/导出 | 无持久分配 |
| 导出 | 经既有 connector 生命周期落文件 | 进程内结果直接返回 |
| 适用 | prefill 式特征导出 | 跑真实 drafter 的 rollout |

限制：保留全序列 cache 分配；V1 only；async/prefix cache 必须关；**尚未与 #57736 做端到端对比**。测试 57 unit passed（RTX 3080 Ti）。

### 5.3 [#59543](https://github.com/vllm-project/vllm/pull/59543) — typed token-output 通道（open，实现 RFC #59542）

- API：`SamplingParams.return_last_hidden_states` → `CompletionOutput.last_hidden_states`；引擎需 `--enable-return-last-hidden-states`。
- 语义：行 `i` = **该生成 token 的 `compute_logits` 真正收到的张量**（sampling 后 gather，不重算），`[len(token_ids), hidden_size]`。
- 传输：新增 `vllm/v1/token_outputs.py` **命名 entry 通道**（first entry 就是 last hidden），为 #62145/#62272 的 named readouts 铺路。
- 规模：**237 行库代码 / 12 文件**；MRV2 only；启动时**拒绝投机解码、PP、CP**。
- 默认零开销：flag 关闭时不持状态；只有请求的请求付一次 gather + output-copy stream 上 non-blocking D2H。
- 实测（较早传输、RTX 4090 / RTX PRO 6000、Qwen3.6-35B-A3B-FP8、1400 条线上单问）：35.0 ms → 35.2 ms；替代的"保留 logit 列"workaround 是 112.9 ms。
- 与 #57736 差异：**无窗口、拒绝投机解码、无 prompt 行、无 HTTP surface**。

### 5.4 [#57185](https://github.com/vllm-project/vllm/pull/57185) — 只返回最后一个 prompt 位置（open）

- API：`kv_transfer_params.return_last_hidden_state` → `kv_transfer_params.last_hidden_state`，带位置/层/表示元数据。
- 复用现有 target forward 与 MRV2 输出同步，无额外前向 / pooling / L2 norm / connector / RPC。
- 限制：必须 1 个输出 token、`n=1`、non-streaming；仅 native Qwen2/Qwen3.5 + V2；允许 TP，拒绝 PP/CP、speculation、KV connectors、beam search、resumable inputs、Responses API。
- 定位：最窄的"final prompt row"契约，但**有 Python/Rust HTTP 交付**。

### 5.5 [#57663](https://github.com/vllm-project/vllm/pull/57663) — closed，未合并

只加 `SamplingParams` 窗口字段与位置数学，**不含 worker 侧抽取与 GPU 验证**。被 #57736 取代，其 interval-math 测试被建议复用。

### 5.6 相邻但不同

| 编号 | 关系 |
|---|---|
| [#48743](https://github.com/vllm-project/vllm/pull/48743) | 通过 extraction connector 返回 hidden（#59543 正文归类为另一条路），不是按窗口捕获 |
| [#62145](https://github.com/vllm-project/vllm/issues/62145) / [#62272](https://github.com/vllm-project/vllm/pull/62272) | "命名 readout" 通道，hidden 只是其中一个 entry；#59543 的通道是它的前置 |
| [speculators#335](https://github.com/vllm-project/speculators/issues/335) | 讨论复用 `extract_hidden_states` 做在线 drafter 训练，见 §8 |

---

## 6. 三条实现路线对比

| 维度 | #57736 runner 级 | #62411 connector 级 | #59543 typed channel |
|---|---|---|---|
| 捕获位置 | runner 内，lm_head 输入行 | connector 的 hidden_state_cache（masked store） | sample 后按 `logits_indices` gather |
| 窗口 | 绝对位置窗口（生成 + prompt 两类） | response 窗口（start/end/min_rows/layer layout） | **无窗口**，每生成 token 一行 |
| 坐标系 | 生成 `[P-1,P+O-1)`；prompt `[0,P-1)` | `abs = P-1+r` | token 索引 |
| 投机解码 | 支持（按 acceptance 裁剪） | **不兼容** | **启动拒绝** |
| engine 改动面 | scheduler + request + protocol + output processor | 仅 connector + 小 hook | 新 token_outputs 模块 + runner/scheduler/processor |
| async scheduling | **必须关闭**（engine-wide） | 必须关闭 | 未强制（flag off 零开销，MRV2 only） |
| prefix cache | 支持有界复用（默认 skip reads） | 必须关闭 | 不涉及 |
| 显存 | 无持久分配 | 仍分配全序列 KV cache | 仅 D2H 缓冲 |
| 交付 | AuxOutput Connector → 进程内 / HTTP base64 npy | 既有 file connector 生命周期 | Msgpack 通道 → final output |
| 代码量 | 大（跨 scheduler/前端/HTTP） | 中（1858+, 16 files, draft） | **小（953+, 20 files；库 237 行）** |
| 状态 | open、主推、needs-rebase | open draft、dirty | open、unstable |
| 适合 | 真实 rollout + teacher 打分 + 全词表还原 | prefill 式特征导出 | 只要"每 token 最后一个 hidden"的轻量消费方 |

**结论**：功能上只有 #57736 覆盖 OPD 完整需求（rollout 窗口 + teacher prompt 行 + 全词表重建 + 投机解码），但它的**同步调度是全局税**；#59543 在"轻量可维护"上更优，社区正在把它变成统一 named-readout 通道。

---

## 7. 设计争议（hidden 专属）

### 7.1 API 形状三选一

- `SamplingParams.hidden_state_interval` / `hidden_state_window`（#56916 评论、#57663）
- `extra_args["hidden_state_capture"] = {...}`（#56916 / #65056 早期文本）
- `aux_outputs=("hidden_states",) + hidden_states_start/end`（#57736 现形态）

#65056 直言"RFC 文本仍用旧的 `extra_args` 形式，而 #57736 的接口是 `aux_outputs`"，需要 maintainer 决定。

### 7.2 坐标系

#56916 定 `response` 为默认（对 rollout 友好）；#65056 的 OPD 用例显式用 `absolute`；#57736 实际只做 absolute，靠 `len(x)-1` 手算。两边都需要，最终要给"双坐标 + 明确校验"。

### 7.3 同步调度税（最现实的阻塞）

`--no-async-scheduling` 是 #57736 的硬前提，且是 **engine 全局**（包括不请求 hidden 的请求、窗口外的 token），实测 −21.3% 吞吐 / +27.1% 延迟。影响：teacher 要么接受损失，要么与 RL rollout 分池部署；也解释了为什么 roadmap 把 hidden（G）排在打分/交付（A–D）之后，并要求"consumer 驱动"再启动。

### 7.4 模型覆盖：只有 Qwen3 的重建契约

"重建 = 逐位相等"只在 unquantized `Qwen3ForCausalLM`（tied 与 untied 各验一个）成立；#65056 明确 Qwen 数据不能外推到 DeepSeek / hybrid / 量化 head / 其他 TP。其他模型只能"返回 hidden，不承诺契约"，等于把验证成本转移给 trainer。

### 7.5 Prefix cache 读取策略不一致

#57736 现状：hidden-only 捕获在 APC 开启时**自动**读有界前缀（无 BI 硬门控）；加 prompt-logprob 字段才走 scoring 的 skip 默认。
#65056 提议：**默认 skip**，只在已验证 BI 配置显式开复用，且 BI flag 不得自动开启 reads。
两者差异被明确标为"需要作者/maintainer 决策"，并提醒"未测配置不能靠 BI flag 获得通用 uncached-equivalence 保证"。

### 7.6 交付与序列化

- #57736 现用 HTTP JSON + base64 `.npy`；8192 行 40 MiB 量级下 Base64 膨胀与 CPU 开销会放大。
- #65056 主张内部走 tensor-aware MessagePack，bulk 走 octet-stream，**不要新开 hidden 专用 IPC**；by-reference 交给 AuxOutput/Mooncake。
- `HiddenStatesAccumulator` 在 frontend 保活整段窗口，内存上限/背压仍是待验证项（"ASGI send 完成 ≠ trainer 收到"）。

### 7.7 重叠 PR 未收敛

#57736 与 #59543 在"生成行传输"重叠；#57185 在"final prompt row"重叠；#62411 是同一作者的另一条对照路线。共识是"对齐而非各合一套"，但目前**没有 merged 的收敛结果**。

---

## 8. 与已合入 hidden 能力的边界：`extract_hidden_states`

vLLM 已有一个**已合入**的 hidden 导出路径：`method=extract_hidden_states`（`docs/features/speculative_decoding/extract_hidden_states/`），给 EAGLE-3/DFlash 等 drafter 训练取 aux hidden。

**它为什么不能替代 OPD 要的 replay**：

| 维度 | `extract_hidden_states` | OPD 需要的 |
|---|---|---|
| 选什么 | 按**层 id**（`aux_hidden_state_layers`） | 按 **token 位置窗口** |
| 何时 | prefill/生成全程，按层 dump | 只要指定窗口内的行 |
| 谁消费 | drafter 训练（离线数据生成） | trainer 的 KL |
| 与 drafter 共存 | 它就是 drafter 采集路径 | rollout 时要能同时跑真实 drafter（#62411 就卡在这） |
| 位置语义 | 层 id 捕获"进入该层的 hidden"（id k = layer k-1 的输出） | 行 p 预测 token p+1，需要精确位置 |

这也是 #62411 的硬伤：它构建在 `extract_hidden_states` 上，所以**无法与真实投机 drafter 共存**。

---

## 9. 落地建议（只针对 hidden 路线）

1. **今天就要用**：没有可用实现。要么自己打 #57736 的 patch（接受同步调度、Qwen3-only），要么退回 top-K 打分（有偏）——见另一份文档。
2. **要打 #57736 的 patch，先确认四件事**：
   - teacher 是不是 Qwen3 家族（否则重建契约无效）；
   - 能否接受 `--no-async-scheduling`（把 −21% 算进容量）；
   - trainer 侧能否复现 head 语义（final norm + lm_head + bias/scale/softcap），并跑通"重建 vs 原生 logits max abs diff == 0"；
   - 交付走 HTTP inline 还是等 AuxOutput 的 by-reference（大数据量时 Base64 会痛）。
3. **只想拿 rollout 窗口特征**（probe / reward / 一致性分析）：优先等 #59543 形态或 #62145 named readout，开销最小。
4. **纯 prefill 特征导出**（不跑 drafter）：#62411 的 connector 方案改动面最小，但 async/prefix cache 都要关。
5. **跨节点大批量**：不要自写 socket 传 hidden，等 AuxOutput/Mooncake 的 by-reference（#45221 / #64380），否则就是社区明确反对的"新开 hidden 专用传输层"。
6. **要跟踪的信号**：#57736 摘掉 `needs-rebase`、API 定为 `aux_outputs`、`docs/training/hidden_states.md` 出现在 main——三者任一发生，说明这条线要动了。

---

## 10. 时间线（只列 hidden 相关）

| 时间 | 事件 |
|---|---|
| 2026-09-15 | RFC #56916（窗口化 hidden 捕获，vx120） |
| 2026-09-16 | PR #57185（final prompt hidden） |
| 2026-09-20 | PR #57736（主推实现） |
| 2026-09-21 | （依赖）AuxOutput #45635 合入 |
| 2026-10-01 | PR #59543 / RFC #59542（typed token-output 通道） |
| 2026-10-17 | roadmap #62175 立项，hidden 为 work item G |
| 2026-10-19 | PR #62411（connector 级，draft） |
| 2026-11-02 | RFC #65056（teacher 侧 hidden 契约） |
| 2026-11-03 | #65056 宣布复用 #57736，不再另起实现 |
| 2026-11-16 | （依赖）R3 + P/D 的 #62056 合入 |
| 2026-11-16 | #65056 feedback 期结束（正文承诺） |
| 2026-11-21 | #57736 最近一次更新（open，13 评论，needs-rebase） |

---

## 11. 链接总表

**RFC**
- [#56916 RFC: Windowed Hidden-State Collection for vLLM Rollouts](https://github.com/vllm-project/vllm/issues/56916)
- [#65056 RFC: Selected input-row hidden-state export for full-vocabulary OPD teachers](https://github.com/vllm-project/vllm/issues/65056)
- [#59542 RFC: last hidden states](https://github.com/vllm-project/vllm/pull/59542)

**PR**
- [#57736 capture selected hidden-state rows for rollout windows and teacher scoring](https://github.com/vllm-project/vllm/pull/57736)
- [#62411 Windowed hidden-state export for ExampleHiddenStatesConnector](https://github.com/vllm-project/vllm/pull/62411)
- [#59543 Opt-in per-request return of the last hidden states (MRV2)](https://github.com/vllm-project/vllm/pull/59543)
- [#57185 Return the final prompt hidden state from generation](https://github.com/vllm-project/vllm/pull/57185)
- [#57663（closed，未合并）](https://github.com/vllm-project/vllm/pull/57663)

**相邻 / 依赖**
- [#62175 Roadmap Q4 2026 OPD Teacher Serving（work item G）](https://github.com/vllm-project/vllm/issues/62175)
- [#45221 AuxOutput Connector tracker](https://github.com/vllm-project/vllm/issues/45221)
- [#48743](https://github.com/vllm-project/vllm/pull/48743)、[#62145](https://github.com/vllm-project/vllm/issues/62145) / [#62272](https://github.com/vllm-project/vllm/pull/62272)
- [speculators#335](https://github.com/vllm-project/speculators/issues/335)

**文档**
- [Hidden State Extraction（已合入的 drafter 路径）](https://docs.vllm.ai/en/latest/features/speculative_decoding/extract_hidden_states/)

---

## 12. 本版相对上一版的删减说明

本版相对上一版（曾把 OPD 打分链路也混进来）删掉的内容：

| 删除项 | 原因 |
|---|---|
| RFC #56860 per-position token logprobs 详解 | 打分链路，不涉及 hidden |
| PR #54335 / #60960 / #57269 / #60366 | scoring 与前缀复用，属另一条载体 |
| "已合入的相邻能力"整表（R3 / sampling mask / scoring） | 只保留与 hidden 交付相关的 #45635 / #62056，移到 §3.2 |
| aux hidden state 的 speculator bugfix（#68813 / #66177 / #68769 等） | 修的是 drafter 层捕获，不是窗口 replay |
| #48305 anchor RFC 与 vLLM × RL roadmap 其它条目 | 与 hidden 无直接实现关系，仅保留 OPD roadmap 的 work item G |
