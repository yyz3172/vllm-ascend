## DynamicKV（vLLM / vLLM-Ascend，PD 场景）说明

本文是一份面向工程落地的 **DynamicKV** 说明；算法与开源仓库 `code/DynamicKV/` 中 **跨层预算**（`update_and_reset_budget_per_kv_head` 一类）及 **per-head 打分 / 并集 indices** 等思路对齐。需区分：`impl=attn` 下还存在与参考实现相近的 **「前向内、每隔若干层刷新预算」** 的节奏；**默认 `impl=offload`** 则在 prefill 步后 **一次性** 对全层做跨层预算，**没有**「每 4 层」的前向周期。当前适配目标模型：Mistral-7B-Instruct-v0.2。

---

## 1. 功能目的与总体策略

### 1.1 目的：压缩的是什么？为什么要压缩？

大模型推理时，注意力需要读写 **KV Cache（Key/Value）**。当 prompt 很长（例如 20K tokens），KV Cache 会占用大量显存/显存带宽，并且在 PD 场景下还会带来：

- **Prefill → Decode 的 KV 传输代价**（传多少 blocks、耗时多少）
- **Decode 侧每步 attention 读取 KV 的代价**

DynamicKV 的核心思想是：**不必保留所有历史 token 的 KV**。对较旧 token 做"保留重要的少量 token"，并始终保留最近一段窗口（window）以维持局部一致性。

补充：在 PD 场景里，"压缩"有两层含义：

- **少算（compute-side）**：decode 每层用自己的 `kv_len`（更短的 `context_lens`）做 attention，减少算力/带宽。
- **少传 + 少占物理块（memory/transfer-side）**：prefill → decode 只传输压缩后需要的 KV blocks，并让 decode 侧只分配这些 blocks（`kv_cache_usage` 随之下降）。

### 1.2 压缩策略：怎么决定"重要 token"？

以"最后的 window_size 个 query"（最近 token）为查询，计算它们对历史 keys 的注意力权重；把历史 token 按"被注意力关注的总量"排序，取 top-k：

- **旧 token（old）**：按重要性保留 top-k
- **新 token（cur/window）**：最近 window_size 全保留

最终每层保留的 token 数量为：

- \(kv\_len\_layer = old\_budget\_layer + window\_size\)

#### 1.2.1 Per-Head Scores + 并集 Indices（保留所有 head 的选择）

由于 vLLM 的 Paged KV Cache 是 token-level 布局（同一 slot 的所有 head 来自同一 token），无法实现开源的 per-head 压缩（每个 head 保留不同的 token 集合）。

为确保不丢失任何 head 认为重要的 token，本实现采用**并集策略**：

1. **Per-head scores**：每个 KV head 独立计算 token importance，得到 `[Hkv, old_len]` 的 scores
2. **Per-head top-k**：每个 head 独立选出 `budget_size` 个最重要的 token
3. **并集 indices**：取所有 head 选择的 token **并集**，确保任意 head 认为重要的 token 都被保留

```
indices_per_head = scores_old.topk(budget_size, dim=-1).indices  # [Hkv, budget_size]
union_indices = torch.unique(indices_per_head.flatten())         # 并集
# 按重要性排序，用于后续跨层截断
indices = sort_by_combined_importance(union_indices)
```

**与开源的对比**：
- 开源：每个 head 各保留 `budget_size` 个（可以不同），总 token 数 = `budget_size`/head
- 本实现：所有 head 共享并集，总 token 数 = `|union|`（介于 `budget_size` 和 `budget_size × Hkv` 之间）

**indices 顺序**：
- 中间态按**重要性顺序**存储，截断 `indices[:budget]` 保留最重要的
- 最终输出通过 `cap_keep_indices_chronological` 恢复**时间顺序**，用于 KV 写回

### 1.3 分层的关键：跨层预算；周期重分配仅 `impl=attn`

分层策略不要求每层保留相同的 old token 数量。两种 **`impl`** 的**事实**如下。

**共性（两种 impl 一致）**

- 每层先得到一份候选的重要性分布（scores / indices，见 §1.2.1）。
- 跨层「每层保留多少 old token」由 **`update_and_reset_budget_per_kv_head`**（及同一套 `radio_max` / `radio_min`、`prompt_kv_len_budget` 等配置）决定；最终在 prefill 侧得到各层的 `old_budget`（再结合 window 做 keep / 写回或导出元数据）。

**差异（必须按配置区分阅读）**

| 项目 | **`impl=attn`**（`attention_v1.py`） | **`impl=offload`（默认）**（`dynamic_kv_offload.py` + `model_runner_v1.py`） |
|------|--------------------------------------|--------------------------------------------------------------------------|
| 跨层预算调用时机 | 在 **attention 前向**内：除最后一层外，每当 **`layer_idx % 4 == 3`** 时，用 **当前已累积的若干层** scores 调一次 `update_and_reset_budget_per_kv_head`，并可能对 **已算层** 的 indices 做截断；**最后一层**再基于全层 scores 做预算（或沿用周期阶段写入的缓存，以代码为准）。 | **prefill 整段 forward 结束之后**：各层 `q_last` 与 KV 已由 hook 与普通 attention 写满，再 **对全层 `num_layers` 只调一次** `update_and_reset_budget_per_kv_head`。 |
| 「每 4 层」周期重分配 | **有**（仅该路径）。 | **无**。 |
| 与开源「前向里逐层改 KV」的类比 | 更接近（仍在步内、随层推进）。 | 不等价：压缩发生在 **步后**，跨层预算只有 **一轮**。 |

---

## 2. 接口与配置参数说明（无新 API，新增配置）

DynamicKV 通过 vLLM 的 `additional_config["dynamic_kv"]` 下发（由 vLLM-Ascend 解析）。

配置入口：`code/vllm-ascend/vllm_ascend/ascend_config.py`

### 2.1 配置项

- **enabled**：是否启用 DynamicKV（默认 `false`）
- **impl**：在 vLLM-Ascend 里 **把压缩算法挂在哪里执行**（默认 **`offload`**）
  - **`offload`（默认）**：prefill 时 attention 仍走常规路径（`reshape_and_cache` 等）把 KV 写满；**该步结束后**在 worker / `model_runner` 里做一次后处理：通过 Q 相关模块的 forward hook 捕获各层 `q_last`，再调用 `run_offload_rewrite_and_build_updates` 做选 token、跨层预算、前缀写回与 PD 元数据组装。**不经过** `attention_v1.py` 里那套「prefill forward 内逐层累积 + 每 4 层重分配」分支。启动后日志里若出现 **`[DynamicKV][offload] installed ...`**，即表示当前为 offload。
  - **`attn`**：压缩逻辑挂在 **`attention_v1.py` 的 Python attention forward** 中：在 `PrefillNoCache` 或 ChunkedPrefill **最后一块**内逐层累计各层 `q_last`/scores/indices，并在 forward 内触发跨层预算与写回。文档 **§3.2** 里描述的「每 4 层」跨层重分配与沿途截断 indices，**当前主要对应此路径**。适合与开源「在模型前向里直接操作 KV」的读法对齐；与默认 offload 相比，对图捕获、eager 等环境更敏感。
- **model_types**：允许启用的 HF `model_type` 列表（默认 `["mistral"]`）
- **window_size**：窗口大小 window（默认 `16`）
- **prompt_kv_len_budget**：每层 KV 长度的预算目标（默认 `512`，通常约等于 \(old + window\_size\) 的目标值）。**不是硬上限**：个别层可能 \(kv\_len > prompt\_kv\_len\_budget\)，硬上限仍为 prompt_len
- **pooling**：对 **旧 token 段**上的一维 importance 分数做空间平滑（沿 token 下标轴，步长 1，两侧对称 padding 为 `kernel_size // 2`）。取值含义：
  - **`none`**：不平滑，各位置的分数直接进入后续 top-k / 并集逻辑。
  - **`avgpool`**（默认）：`avg_pool1d`，邻域内取平均，可抑制单 token 尖峰、让局部区间的重要性更平滑。
  - **`maxpool`**：`max_pool1d`，邻域内取最大，更偏向「只要邻域里有一处高响应就抬高该位置」。
  - 当 `pooling == "none"` 或 `kernel_size <= 1` 时，实现上等价于不做 pooling。
- **kernel_size**：pooling 的卷积核长度（默认 `7`）；仅在 `pooling` 为 `avgpool` / `maxpool` 且 `kernel_size > 1` 时生效。
- **radio_max**（默认 `10.0`）：与实现内部 **`base = max(prompt_kv_len_budget - window_size, 0)`**（`DynamicKVConfig.base`，非单独 YAML 键）相乘，得到 **`cand_old = min(int(radio_max × base), old_len)`**，其中 **`old_len`** 为当前 prompt 下旧段长度 **`prompt_len - window_size`**。该值在 **`impl=offload`**（`dynamic_kv_offload.py`）与 **`impl=attn`**（`attention_v1.py`）中均用作：**(1)** per-head 对旧段打分时 **`topk` 的候选规模上界**（及并集路径上的候选池）；**(2)** 传入 **`update_and_reset_budget_per_kv_head`** 的 **`budget_size`**，参与跨层比例与归一化。**不宜**把 `radio_max` 理解成「每层最终最多保留 `radio_max×base` 个 old token」：全层 old 总量仍由跨层函数里 **`need_fill_kv = base × num_layers`** 等与 `dynamic_kv.py` 实现一致的逻辑收束；较大 `radio_max` 主要是 **扩大候选集**，使跨层分配基于更密的分数（offload 源码注释：允许候选池大于 `prompt_kv_len_budget` 对应量级，**最终每层预算仍由 `update_and_reset_budget*` 给出**）。YAML：`additional_config["dynamic_kv"]["radio_max"]` → `dynamic_kv_radio_max`。
- **radio_min**（默认 `0.1`）：在 **`update_and_reset_budget`** / **`update_and_reset_budget_per_kv_head`**（`dynamic_kv.py`）中定义 **`min_budget = int(radio_min × base)`**，对跨层比例分配得到的每层 old 预算做 **下界**（`max(min_budget, …)`），避免某层被压到过低；再通过最后一层加减 **`diff`** 等对全层总和纠偏。更贴近「**跨层重分配后、每层 old 预算的下限系数**」。YAML 中为 `additional_config["dynamic_kv"]["radio_min"]`（与 `radio_max` 同理），由 `ascend_config.py` 读入为 `dynamic_kv_radio_min`。
- **validation_mode**：验证模式，用于验证 token 选择策略的有效性（默认 `none`）
  - `none`：正常压缩模式
  - `mask`：prefill 不落盘压缩；decode 用完整 KV + `atten_mask`（`npu_fusion_attention`），被掩蔽位置不参与注意力
  - `zero`：在 **prefill 节点**对 paged KV 中不重要 token 的 **K/V 物理置零**，PD 传输置零后的 KV；decode 走 **常规 `_npu_paged_attention`**，不依赖 `per_layer_important_indices` / `_VALIDATION_MASKS`

#### 2.1.1 `impl=offload`：何时不做 pack、何时全长元数据、何时整 request 跳过

以下均指 **`run_offload_rewrite_and_build_updates`**（`vllm_ascend/worker/dynamic_kv_offload.py`）。**`C`** 即传入的 **`prompt_kv_len_budget`**，**`L`** 为该 request 的 prompt 长度（`seq_lens` 项）。

**整批无更新（返回 `{}`）**

- `req_ids` 为空、`num_layers <= 0`，或 **`C <= 0`**（预算无效）。
- 从 **`kv_caches`** 解析不到任何合法 4D K/V（**`layer_items` 为空**）。

**单 request：全长前缀元数据（`_full_prefix_update`，不 pack 物理 KV）**

- **`L <= C`**：整段 prompt 不超过预算，不做跨层压缩（与模块 docstring 一致）。
- **`(L - C) < min_rewrite_delta`**：实现里 **`min_rewrite_delta` 写死为 `1`**（**非 YAML**，当前不改）。对 **整数 `L,C`**，在 **`L > C`** 时 **`L - C >= 1`**，故该条件与 **`L <= C`** 叠加后 **实际等价于仅 `L <= C` 触发**；源码注释提到「增量过小跳过」及 **128 与 block_size**，与 **当前常数 `1`** 不一致，以代码为准；后续对齐见待办 **dynkv-min-rewrite-delta**。
- **无任何 `q_last` 捕获**（`q_last_store[rid]` 为空）且 **`L > C`**：打 **WARNING**，写入 **full prefix**；若环境变量 **`VLLM_ASCEND_DYNKV_STRICT`** 为 `1` / `true` / `yes`，则 **抛 `RuntimeError`**（见同文件模块 docstring）。
- **`validation_mode` 为 `mask` / `zero`**：仍算预算与重要位置，但 **不做** `none` 下的前缀 pack；对 **mask** 写 **`per_layer_important_indices`** 等，对 **zero** 在 paged KV 上置零不重要 slot；随后 **`_full_prefix_update(rid, L)`**（全长 `per_layer_kv_lens` / keep 语义），再 **`continue`**。
- **`_validate_payload_or_fallback` 校验失败**（已写入的 `per_layer_kv_lens` / `per_layer_keep_indices` 与 `L` 或非空 indices 规则不一致）：覆盖为该 request 的 **full prefix**，保证元数据自洽。

**单 request：不写 `updates[rid]`（`continue`，与「无 Q 仍 full prefix」不同）**

- **`L <= 0`**：无效长度，直接跳过。
- **某层缺少 `q_last`** 导致 **`per_layer_scores_old` 不完整**：打 WARNING「**incomplete Q capture**」，**本 request 不进入 `updates`**（非 `_full_prefix_update`）。与「整包无 Q」仍给全长 PD 元数据的行为 **不一致**，排查 PD 缺 `dynamic_kv` 时需注意。

### 2.2 与 PD（Mooncake）的关系

PD 场景下，prefill 完成后会通过 `kv_transfer_params` 把 DynamicKV 的结果带给 decode：

- **per_layer_kv_lens**：长度为 num_layers 的列表，每层一个 kv_len；**PD decode 侧按层有效长度应以此为准**（尤其 **`impl=offload`** 物理 pack 之后，见 §3.3）。
- **per_layer_keep_indices**：长度为 num_layers 的列表，每层在 **原始 prompt token 下标空间** 中的保留下标列表（可 JSON 序列化）。**`impl=offload`** 在 **`validation_mode=none`** 且完成 **packed 前缀写回** 后，实现上常 **每层为 `[]`**：KV 已按时间序压入 cache 前缀 slot `0..kv_len-1`，再传原始坐标易与 pack 语义冲突，故省略 indices、**仅依赖 `per_layer_kv_lens`**。未压缩回退（全长前缀）或 **`impl=attn`** 导出时可为 **`0..L-1`** 等非空列表，便于与 `kv_len` 交叉校验。
- **per_layer_important_indices**（可选）：`validation_mode` 为 **`mask`** 时由 prefill 写入，每层为「重要 token」的下标列表（稀疏）；decode 据此恢复 attention mask（PD 下 decode 与 prefill 不同进程，不能共享 `_VALIDATION_MASKS`）。**`zero`** 不写入该字段（KV 已在 prefill 置零）

为支持 **少传 + 少占物理块**，在 `kv_transfer_params.dynamic_kv` 下新增可选字段（DynamicKV 关闭时不会出现）：

- **per_layer_num_prompt_blocks**：长度为 num_layers 的列表，每层压缩 KV 需要的 blocks 数（\(ceil(kv\_len / block\_size)\)）
- **per_layer_remote_block_ids**：长度为 num_layers 的二维列表；第 \(L\) 层为「该层要参与 PD 的远端 block id 序列」。实现上常由 **同一组「按逻辑 prompt 前缀排列的物理块 id 列表」** 再按每层 `per_layer_num_prompt_blocks[L]` 取 **前若干个 id** 得到（各层共享前缀，尾部层块数可更少）。**前提**：底层 id 列表的顺序必须是 **逻辑上的 prompt 从前到后**（见下节「块序」），而不能默认等于调度器里任意 `computed_block_ids` 的枚举顺序。
- **prefix_remote_block_ids**（可选，**当前仅 `impl=offload`** 由 worker 写入）：长度 `n_transfer = ceil(max(L for L in per_layer_kv_lens if L > 0) / block_size)` 的一维列表（与 `model_runner_v1.py` 中计算一致），元素为 **该 request 的 `block_table` 行** 中 **前 `n_transfer` 个槽位**对应的物理 block id（与 paged KV 中「压缩 KV 写在连续前缀 slots」一致）。用于纠正「分配器给出的 block id 列表顺序 ≠ 逻辑前缀序」时，对 `remote_block_ids` 盲目取前缀会 **拉错块、decode KV 损坏** 的问题。

#### 2.2.1 块序为何重要；offload 下如何收缩传输

DynamicKV 将每层有效 KV 写回 **各层 paged 布局的连续前缀**（从 slot 0 起长度 `kv_len`）。PD 要传的应是 **覆盖这些前缀槽位** 的物理块，且 **block id 在列表中的顺序须与逻辑 token 序一致**。

- **可靠顺序**：每个 request 的 **`block_table` 一行**（page table）按 slot 下标排列，对应 **prompt 从首 token 往后的块映射**；取该行前 \(n_{transfer}\) 个非负 id，即得到与压缩前缀对齐的 **`prefix_remote_block_ids`**。实现：`vllm_ascend/worker/model_runner_v1.py`（offload 后处理挂载 `kv_transfer_params_updates` 时写入 `dynamic_kv.prefix_remote_block_ids`；注释说明勿用分配器序的 `block_ids[:n]`）。
- **连接器侧（Mooncake）**：`vllm_ascend/distributed/mooncake_connector.py` 在合并 `request_finished` 时 **保留** worker 带来的 `prefix_remote_block_ids`。随后在 **`impl=offload`** 且即将发起传输时：**若存在 `prefix_remote_block_ids`**，则以其作为本次 **`remote_block_ids`（`send_block_ids`）** 并据此重算 `num_prompt_blocks` 与 **`per_layer_num_prompt_blocks` / `per_layer_remote_block_ids`**（每层对同一前缀列表再截前 `nb` 个块）。日志标签含 **`[DynamicKV][PD] offload shrink transfer(block_table)`**。**若无**该字段，则退化为对当前 `send_block_ids` 做 **`[:n_transfer]`** 的 legacy 收缩并打 **WARNING**（`allocator_fallback`）——块序可能与逻辑前缀不一致，存在风险。
- **调度侧一段实现注意**：在仅根据 `computed_block_ids` 构造一版 `per_layer_remote_block_ids` 时，**不会**在此处改短顶层 `computed_block_ids` 本身，以避免误用非逻辑序列表的前缀（见 `mooncake_connector.py` 内注释）。
- **`impl=attn`**：当前代码路径 **不写入** `prefix_remote_block_ids`；PD 元数据与顶层 `remote_block_ids` 的组合行为与 offload **不完全相同**，排查传输与对齐问题时需按 **`impl`** 区分。

代理（proxy）会把 prefill 返回的 `kv_transfer_params` 原样传给 decode 请求。

---

## 3. 算法/流程（人话版）
**阅读约定（重要）**

- **§3.1、§3.2 仅描述 `impl=attn`**：代码在 `attention_v1.py` 的 Ascend attention forward 内，与 **ChunkedPrefill 的最后一块** 或 **PrefillNoCache** 共用同一套「逐层累积 →（可选）每 4 层周期预算 → 最后一层写回/导出」分支。
- **`impl=offload`（默认）**：不在 §3.1–3.2 展开；其事实流程为：prefill 内常规写 KV → 步后 `run_offload_rewrite_and_build_updates`（见 §2.1、`dynamic_kv_offload.py`、`model_runner_v1.py`），**无 §3.2 步骤 4 的「每 4 层 + 沿途截断 indices」**。

### 3.1 PrefillNoCache（非 chunked）— 仅 `impl=attn`

触发条件：**`impl=attn`**，且 Prefill、此前没有 cache（`PrefillNoCache`），DynamicKV 开启、模型类型允许。

流程与 **§3.2 ChunkedPrefill 的最后一块** 在 **attention 内嵌分支** 上相同：每层先 `reshape_and_cache`，在最后一层前累积各层 `q_last`/scores/indices，最后一层对全 prompt 做 gather、跨层预算与按层写回或导出（与 `validation_mode` 有关）。

实现位置：`attention_v1.py` 中与 ChunkedPrefill **最后一块** 共用同一套逻辑（非 chunked 时不再要求 `dynamic_kv_is_last_chunk`）。

**`impl=offload` 时**：本小节不适用；PrefillNoCache 下 KV 仍由常规 attention 写满，压缩与跨层预算发生在 **该 prefill 步结束后的 offload 后处理**。

### 3.2 ChunkedPrefill 最后一块 — 仅 `impl=attn`（含「每 4 层」与沿途截断）

本节描述 **`impl=attn`** 下，`attention_v1.py` 在 **chunked prefill 且最后一块**（`dynamic_kv_is_last_chunk=True`）时的行为。

触发条件：**`impl=attn`**，chunked prefill 且 `dynamic_kv_is_last_chunk=True`（以及 DynamicKV、模型类型等前置条件满足）。

流程：

1. 先把本 chunk 的 KV 正常写入 paged cache（保证 cache 完整）
2. 从每层的 paged cache 中 gather 出「全 prompt」的 K/V（按 request）
3. 对每层、每个 request：计算旧 token 段的 per-head scores / indices（top-k 候选）
4. **仅本 impl**：在 forward 内，当层索引满足 **每 4 层一次**（实现上为 `layer_idx % 4 == 3` 且尚未到最后一层）时，对 **当前已出现的层集合** 调用一次 **`update_and_reset_budget_per_kv_head`**，并 **可能对已累积层的 indices 做截断**，使周期阶段的预算变更作用到中间态。（**`impl=offload` 无此步骤。**）
5. **最后一层**：按 **最终** 各层 `old_budget` 为每层构造 keep（old 段 + window），从 full KV 中 gather 出压缩后的 KV 写回各层 paged cache 前缀 slots，或按 `validation_mode` 仅导出 mask/元数据而不改 KV
6. 输出 `per_layer_kv_lens`、`per_layer_keep_indices` 等（供 PD 与 decode 使用；`mask` 时还可能带 `per_layer_important_indices`，见 §2.2）

关键实现位置（按职责拆分）：

- **共用算法库**：`code/vllm-ascend/vllm_ascend/attention/dynamic_kv.py`
  - `scores_and_indices_old_perhead_aggregated`：per-head scores + 聚合 indices
  - `update_and_reset_budget_per_kv_head`：跨层预算（`tk` 等与实现注释一致）
- **`impl=attn` 前向编排（含每 4 层周期）**：`code/vllm-ascend/vllm_ascend/attention/attention_v1.py`
- **`impl=offload` 步后一轮预算 + 写回/导出**：`code/vllm-ascend/vllm_ascend/worker/dynamic_kv_offload.py`；调度入口见 `model_runner_v1.py`

### 3.3 Decode 如何按层生效（关键点：kv_len 与内容一致）

decode 侧生效依赖两件事：

1. **内容**：在 **`validation_mode=none`**（及物理压缩路径）下，依赖 prefill 侧把各层压缩后的 KV 写回各层 paged cache；**`mask` / `zero`** 等验证路径下可能仍持全量物理 KV，通过 mask 或置零表达「有效注意力」，以代码与 §2.1 为准。
2. **长度**：decode attention kernel 在每层使用该层自己的 **`kv_len`**（经 metadata 注入后以 **`context_lens`** 等形式进入 kernel）。

#### 3.3.1 `per_layer_kv_lens` 与 `per_layer_keep_indices`（offload 常为「长度有、indices 空」）

- **`per_layer_kv_lens`**：每层逻辑上的 KV 有效 token 数；**凡是 PD decode 需要 per-layer 长度的，均应以此字段为真源**（含 **`impl=offload`** pack 之后）。
- **`per_layer_keep_indices`**：
  - **`impl=attn`** 或 **offload 未压缩回退**（如全长前缀 `_dynamic_kv_payload_full_prefix`）：可为 **非空** 列表（通常为原始 prompt 下标），可与 `kv_len` 做 `len(indices) == kv_len` 一类一致性检查。
  - **`impl=offload` + `validation_mode=none` + 已 pack 写回**：`dynamic_kv_offload.py` **故意** 每层 `append([])`，注释写明 decode **仅用 `kv_len`**；校验逻辑亦规定 **仅当 indices 非空** 时才要求 `kv_len == len(indices)`。**全空 `[]` 不是丢字段，而是与连续前缀布局一致的设计。**

**Metadata 注入（`vllm_ascend/worker/v2/attn_utils.py`）**：对每个 request、每层，若 **`per_layer_keep_indices[layer]` 为非空 list**，则该 request 该项长度取 **`len(indices)`** 并记入 `dynamic_kv_keep_indices_list`；**否则** 回退到 **`per_layer_kv_lens[layer]`**。**仅当本 batch 至少一个 request 在该层 indices 非空时**，才会挂上 **`dynamic_kv_keep_indices_list`**；若全部为空，则 **`dynamic_kv_keep_indices_list` 为 `None`**，仅 **`dynamic_kv_seq_lens_list`**（来自各层 `per_layer_kv_lens`）驱动长度。

**Kernel 路径（`attention_v1.py::forward_paged_attention`）**：若 **`dynamic_kv_keep_indices_list` 存在且非空**（实现上为 truthy 列表），则用 **`[len(x) for x in dyn_keep]`** 覆盖内部长度列表；否则使用已注入的 **`dynamic_kv_seq_lens_list`**（即主要来自 **`per_layer_kv_lens`**）。因此 offload 压缩后 **不会** 因「全空 indices」误把 `kv_len` 推成 0。

**Decode 组 batch（`model_runner_v1.py`）**：注释明确 **勿把 `per_layer_keep_indices` 再当作 paged decode 的 gather 坐标**：indices 在 **原始 prompt** 空间，而 offload 已将 KV **pack 到 slot `0..Li-1`**；按层长度用 **`Li`（来自 `per_layer_kv_lens`）+ decode 步增量** 与 **`context_lens`** 对齐。

落地方式（数据流）：

- `kv_transfer_params.dynamic_kv` 中 **`per_layer_kv_lens`**（必选语义）与 **`per_layer_keep_indices`**（offload 压缩后常全空）经 prefill → proxy → decode。
- 按 `layer_name` 注入 **`dynamic_kv_seq_lens_list`**；**仅当存在非空 keep 列表时** 才附加 **`dynamic_kv_keep_indices_list`**。
- NPU paged attention 以 **`context_lens`** 接收 per-layer `kv_len`（由上述 metadata 推导）。

关键实现位置：

- metadata 注入（v2 runner）：`code/vllm-ascend/vllm_ascend/worker/v2/attn_utils.py`
- kernel 使用：`code/vllm-ascend/vllm_ascend/attention/attention_v1.py::forward_paged_attention`
- decode 侧长度与注释：`code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py`（DynamicKV + `decode_extra`）

---

## 4. 关键日志说明（如何验收"真的生效"）

日志前缀以 **`[DynamicKV]`** 为主；Decode / PD 子阶段见下。

### 4.1 Prefill（预算/导出）

- **`[DynamicKV] exported results`**：本轮 prefill（`PrefillNoCache` 或 ChunkedPrefill 最后一块）已对 batch 导出 per-request 的分层结果

### 4.2 Mooncake（PD 传递是否带上分层信息）

- **`[DynamicKV][PD] request_finished per_layer_kv_lens stats`**（及同类汇总日志）
- **`unique` / `min` / `max` / `sum` 的读法**：在 **`validation_mode=none`** 且已做 **物理压缩写回** 时，`per_layer_kv_lens` 常反映各层压缩后长度，可用于观察「分层非均匀」与总量级是否接近 `num_layers × prompt_kv_len_budget` 的量级（允许实现细节导致的偏差）。在 **`validation_mode=mask`** 等路径下，实现可能仍向 PD 携带 **全长** `per_layer_kv_lens`，而真实「每层参与注意的有效长度」由 **`per_layer_important_indices` / mask** 等体现；此时 **不宜** 仅用 `sum(per_layer_kv_lens)` 等同于「已压缩掉的 token 数」。详见 §2.2 与实现代码。
- **`[DynamicKV] request_finished per_layer_kv_lens(fallback)`（WARNING）**
  - 出现说明分层结果未成功 attach，PD 退化为均匀 cap（需要排查）

### 4.2.1 物理块是否真的下降（decode kv_cache_usage）

验收点：

- decode `GPU KV cache usage` 应从"full prompt blocks"级别下降到"`max(per_layer_kv_len)` 对应 blocks"级别。
- 直观上，对于 20k prompt（160 blocks@128），若 `max(per_layer_kv_len)` 约 3.5k（约 27 blocks），decode `kv_cache_usage` 应接近原来的 \(27/160\) 倍（再乘以全局 block pool 分母的影响）。

### 4.3 Decode（按层 kv_len 是否被 kernel 使用）

- **`[DynamicKV][Decode] ... kv_lens(...)`**
  - 每层只打印一次，用于确认该层拿到了 per-layer kv_len（由 indices 推导或直接来自 per_layer_kv_lens）

---

## 5. 已知约束与注意事项

- **`impl=attn` 与 `impl=offload` 的时序不同**：前者在 **attention 前向最后一阶段**（整段 PrefillNoCache 或 chunked 最后一块）内嵌压缩与（可选）每 4 层周期预算；后者在 **prefill 步后** 做一轮全层预算与改写。二者共用 `dynamic_kv.py` 中的核心函数，但 **编排与是否「每 4 层」不一致**（见 §1.3）。
- **这是 PD 场景优先实现**：prefill 本地仍先完整写入 KV；**`impl=attn`** 再在 **同一轮 prefill 的最后一层** 所在的前向路径内做分层压缩写回（`PrefillNoCache` 整段或 ChunkedPrefill 最后一块）；**`impl=offload`** 则在 **该步 forward 完成之后** 在 worker 内改写。
- **Paged KV 的写回策略**：压缩后的 KV 写回到每层 cache 的"前缀 slots"，保证 decode 按 kv_len 读的是连续有效前缀
- **Prefix caching 与 DynamicKV 互斥**：启用 DynamicKV 时，vLLM-Ascend 会在初始化配置阶段 **强制关闭** `enable_prefix_caching`。原因是 DynamicKV 会在 prefill 上 **改写** paged KV 内容；而前缀缓存假设「相同 token 前缀对应可复用的稳定 KV」。若同时开启，可能导致错误复用或与 DynamicKV 导出的 per-layer 元数据不一致。若用户在 YAML/CLI 中打开了 prefix caching，实际运行仍会被置为关闭（日志中有 `forcing prefix caching off` 提示）。
- **并发安全**：per-request 结果按 `request_id` 存放并在 Mooncake attach 后 pop，同时有 TTL/容量清理
- **图捕获（graph capture）**：录制图期间不做分层压缩，只做普通 `reshape_and_cache`，避免动态形状破坏捕获
- **`impl=offload` 与 Q 捕获**：若 **`L > prompt_kv_len_budget`** 但拿不到 tail query，见 **§2.1.1**；可通过 **`VLLM_ASCEND_DYNKV_STRICT=1`** 在「无 Q」时强制失败以便暴露配置/钩子问题。

---

## 6. 与开源 DynamicKV 的差异

本实现针对 vLLM-Ascend PD 场景做了以下适配：

### 6.1 架构差异（必要适配）

| 方面　　　　　　　　 | 开源实现　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 本实现　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ----------------------| ----------------------------------------------------------------------------------------------| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **应用场景**　　　　 | HuggingFace Transformers 单机推理　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| vLLM-Ascend PD 分布式分离　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **KV Cache**　　　　 | `DynamicCache` 连续内存　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| Paged KV Cache（block_table 映射）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| **集成方式**　　　　 | 在模型 **forward 内** 替换/包裹注意力与 cache（Monkey-patch），推理过程中逐层读写压缩后的 KV | **默认 `impl=offload`**：forward 不内嵌压缩，prefill 后再 **后处理** 改写 Paged KV；可选 **`impl=attn`**：在 **Attention 后端 forward** 内嵌压缩流程，与开源「前向内」形态 **更接近**（但并非同一套代码仓库） |
| **跨层预算调用节奏** | 与具体开源挂载方式相关；常见为前向内随层推进　　　　　　　　　　　　　　　　　　　　　　　　 | **`impl=attn`**：`attention_v1.py` 内 **每 4 层** 一次周期预算（子集层）+ **最后一层**全层收尾；**`impl=offload`**：**步后单轮**全层 `update_and_reset_budget_per_kv_head`，**无**每 4 层周期　　　　　　　　 |

### 6.2 算法对齐（已完成）

| 方面　　　　　　 | 开源实现　　　　　　　　　　| 本实现　　　　　　　　　　　　　　　　　　　　　　　|
| ------------------| -----------------------------| -----------------------------------------------------|
| **scores 形状**　| `[bsz, num_heads, old_len]` | `[Hkv, old_len]`（per-head）　　　　　　　　　　　　|
| **跨层 tk**　　　| `base × heads × layers`　　 | `base × Hkv × layers`（一致）　　　　　　　　　　　 |
| **indices 聚合** | per-head gather　　　　　　 | per-head scores + token-level 聚合（兼容 Paged KV） |

### 6.3 本实现扩展功能

这些是开源实现没有的 PD 场景增值功能：

- **物理块压缩传输**：`per_layer_num_prompt_blocks` / `per_layer_remote_block_ids`
- **kv_transfer_params 跨进程传递**：prefill → proxy → decode
- **`impl=offload`（默认）**：Q 捕获 hook + prefill 后改写，**不**改 attention 算子主路径，便于与 PD、Mooncake 组合；开源参考实现 **没有** 与此一一对应的同名模式，仅可类比为「把压缩从 forward 挪到步后」的工程折中。

---

## 7. 参考（开源 DynamicKV）

对照 `code/DynamicKV/kv_compression/token_drop/` 下分层相关 **methods** 与 **mistral_model_impl** 源码（与本文算法同思路）。
