## DynamicKV（vLLM / vLLM-Ascend，PD 场景）说明

本文是一份面向工程落地的 **DynamicKV** 说明；算法与开源仓库 `code/DynamicKV/` 中跨层预算、周期重分配等思路对齐。当前适配目标模型：Mistral-7B-Instruct-v0.2。

---

## 1. 功能目的与总体策略（通俗版）

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

#### 1.2.1 Per-Head Scores + 聚合 Indices（与开源对齐）

为与开源 DynamicKV 的跨层密度语义对齐，本实现采用：

1. **Per-head scores**：每个 KV head 独立计算 token importance，得到 `[Hkv, old_len]` 的 scores
2. **跨层 top-k**：使用 `tk = base × Hkv × num_layers`（与开源一致），在 `[layer, head, token]` 三维空间做全局 top-k
3. **聚合 indices**：由于 Paged KV Cache 是 token-level 布局（同一 slot 的所有 head 来自同一 token），最终 indices 需聚合为 token-level

聚合策略：

```
combined_score = token_被选中的head数 × token_聚合importance
indices = combined_score.topk(budget_size)
```

这样既保持了开源的跨层预算分配密度，又兼容 vLLM 的 Paged KV Cache 语义。

### 1.3 分层的关键：跨层预算 + 周期重分配

分层策略不要求每层保留相同的 old token 数量，而是：

- 每层先得到一份候选的重要性分布（scores/indices）
- 每隔若干层（参考实现为 **每 4 层**）进行一次跨层预算重分配（`update_and_reset_budget_per_kv_head`）
- 最终在最后一层将"各层的压缩 KV"写回 cache，使得后续 decode 按层读取到的 KV 内容真实不同

---

## 2. 接口与配置参数说明（无新 API，新增配置）

DynamicKV 通过 vLLM 的 `additional_config["dynamic_kv"]` 下发（由 vLLM-Ascend 解析）。

配置入口：`code/vllm-ascend/vllm_ascend/ascend_config.py`

### 2.1 配置项

- **enabled**：是否启用 DynamicKV（默认 `false`）
- **model_types**：允许启用的 HF `model_type` 列表（默认 `["mistral"]`）
- **window_size**：窗口大小 window（默认 `16`）
- **prompt_kv_len_budget**：每层 KV 长度的预算目标（默认 `512`，通常约等于 \(old + window\_size\) 的目标值）。**不是硬上限**：个别层可能 \(kv\_len > prompt\_kv\_len\_budget\)，硬上限仍为 prompt_len
- **pooling**：对 token importance 做 1D 平滑（`none|avgpool|maxpool`，默认 `none`）
- **kernel_size**：pooling 的 kernel（默认 `1`）
- **radio_max / radio_min**：跨层重分配的上/下限相关系数（默认 `10.0/0.1`，与上游参考实现一致）

### 2.2 与 PD（Mooncake）的关系

PD 场景下，prefill 完成后会通过 `kv_transfer_params` 把 DynamicKV 的结果带给 decode：

- **per_layer_kv_lens**：长度为 num_layers 的列表，每层一个 kv_len
- **per_layer_keep_indices**：长度为 num_layers 的列表，每层一个 indices 列表（可 JSON 序列化）

为支持 **少传 + 少占物理块**，在 `kv_transfer_params.dynamic_kv` 下新增可选字段（DynamicKV 关闭时不会出现）：

- **per_layer_num_prompt_blocks**：长度为 num_layers 的列表，每层压缩 KV 需要的 blocks 数（\(ceil(kv\_len / block\_size)\)）
- **per_layer_remote_block_ids**：长度为 num_layers 的二维列表，每层要传输的远端 block id 序列（当前实现为远端 `remote_block_ids` 的前缀切片，依赖压缩 KV 写入连续前缀 slots）

此外，Mooncake 会把 **整体 `num_prompt_blocks` / `remote_block_ids`** 收缩到 `max(per_layer_num_prompt_blocks)`，用于驱动 decode 侧的 **物理 blocks 分配与传输规模**。

代理（proxy）会把 prefill 返回的 `kv_transfer_params` 原样传给 decode 请求。

---

## 3. 算法/流程（人话版）

本节按"真实推理路径"描述，不逐行翻译代码。

### 3.1 PrefillNoCache（非 chunked）

触发条件：Prefill 且此前没有 cache（`PrefillNoCache`），且 DynamicKV 开启、模型类型允许。

流程与 **ChunkedPrefill 的最后一块** 相同（见 3.2）：每层先 `reshape_and_cache`，在最后一层前累积各层 `q_last`/scores/indices，最后一层对全 prompt 做 gather、跨层预算与按层写回压缩 KV。

实现：`attention_v1.py` 中与 `ChunkedPrefill` 共用同一套分层分支（非 chunked 时不再要求 `dynamic_kv_is_last_chunk`）。

### 3.2 ChunkedPrefill（PD 常见）压缩流程：只在最后 chunk 执行

触发条件：chunked prefill 且最后一块（`dynamic_kv_is_last_chunk=True`）。

流程：

1. 先把本 chunk 的 KV 正常写入 paged cache（保证 cache 完整）
2. 从每层的 paged cache 中 gather 出"全 prompt 的 K/V"（按 request）
3. 对每层、每个 request：计算旧 token 的 per-head scores/indices（top-k 候选）
4. **每 4 层**触发一次跨层预算重分配（对每个 request 独立）：
   - 使用 `update_and_reset_budget_per_kv_head` 计算每层应该保留多少 old tokens
   - 立即对已算过层的 indices 做截断（预算沿途更新生效）
5. 最后一层：按最终预算为每层构造 keep_idx（old_budget + window），从 full KV 中 gather 出压缩后的 KV，并写回该层 paged cache 的前缀 slots
6. 输出 per-layer kv_lens 与 per-layer keep indices（供 PD 传递与 decode 生效）

关键实现位置：

- 评分/indices 与跨层预算：`code/vllm-ascend/vllm_ascend/attention/dynamic_kv.py`
  - `scores_and_indices_old_perhead_aggregated`：per-head scores + 聚合 indices
  - `update_and_reset_budget_per_kv_head`：跨层预算重分配（`tk = base × Hkv × layers`）
- last_chunk / PrefillNoCache 主流程：`code/vllm-ascend/vllm_ascend/attention/attention_v1.py`
- offload 模式实现：`code/vllm-ascend/vllm_ascend/worker/dynamic_kv_offload.py`

### 3.3 Decode 如何按层生效（关键点：kv_len 与内容一致）

decode 侧生效依赖两件事：

1. **内容**：prefill 最后一层把各层压缩 KV 写回到各层自己的 paged cache（每层 cache 内容真实不同）
2. **长度**：decode attention kernel 在每层使用该层自己的 `kv_len`

落地方式：

- `kv_transfer_params.dynamic_kv.per_layer_kv_lens` / `per_layer_keep_indices` 从 prefill → proxy → decode
- decode 的 attention metadata（按 layer_name）注入：
  - `dynamic_kv_seq_lens_list`
  - `dynamic_kv_keep_indices_list`（存在时优先用其长度推导 kv_len）
- NPU paged attention 以 `context_lens` 接收 per-layer kv_len

关键实现位置：

- metadata 注入（v2 runner）：`code/vllm-ascend/vllm_ascend/worker/v2/attn_utils.py`
- kernel 使用：`code/vllm-ascend/vllm_ascend/attention/attention_v1.py::forward_paged_attention`

---

## 4. 关键日志说明（如何验收"真的生效"）

日志前缀以 **`[DynamicKV]`** 为主；Decode / PD 子阶段见下。

### 4.1 Prefill（预算/导出）

- **`[DynamicKV] exported results`**：本轮 prefill（`PrefillNoCache` 或 ChunkedPrefill 最后一块）已对 batch 导出 per-request 的分层结果

### 4.2 Mooncake（PD 传递是否带上分层信息）

- **`[DynamicKV][PD] request_finished per_layer_kv_lens stats`**
- `unique/min/max/sum` 用于验收"分层预算非均匀"且总预算符合预期（例如 sum≈num_layers*prompt_kv_len_budget）
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

- **这是 PD 场景优先实现**：prefill 本地仍先完整写入 KV，再在 **同一轮 prefill 的最后一层**做分层压缩写回（`PrefillNoCache` 整段或 ChunkedPrefill 最后一块）
- **Paged KV 的写回策略**：压缩后的 KV 写回到每层 cache 的"前缀 slots"，保证 decode 按 kv_len 读的是连续有效前缀
- **并发安全**：per-request 结果按 `request_id` 存放并在 Mooncake attach 后 pop，同时有 TTL/容量清理
- **图捕获（graph capture）**：录制图期间不做分层压缩，只做普通 `reshape_and_cache`，避免动态形状破坏捕获

---

## 6. 与开源 DynamicKV 的差异

本实现针对 vLLM-Ascend PD 场景做了以下适配：

### 6.1 架构差异（必要适配）

| 方面 | 开源实现 | 本实现 |
|------|----------|--------|
| **应用场景** | HuggingFace Transformers 单机推理 | vLLM-Ascend PD 分布式分离 |
| **KV Cache** | `DynamicCache` 连续内存 | Paged KV Cache（block_table 映射） |
| **集成方式** | Monkey-patch forward | vLLM Attention Backend + offload 模式 |

### 6.2 算法对齐（已完成）

| 方面 | 开源实现 | 本实现 |
|------|----------|--------|
| **scores 形状** | `[bsz, num_heads, old_len]` | `[Hkv, old_len]`（per-head） |
| **跨层 tk** | `base × heads × layers` | `base × Hkv × layers`（一致） |
| **indices 聚合** | per-head gather | per-head scores + token-level 聚合（兼容 Paged KV） |

### 6.3 本实现扩展功能

这些是开源实现没有的 PD 场景增值功能：

- **物理块压缩传输**：`per_layer_num_prompt_blocks` / `per_layer_remote_block_ids`
- **kv_transfer_params 跨进程传递**：prefill → proxy → decode
- **offload 模式**：Hook Q-proj 延迟计算，不修改 attention 执行路径

---

## 7. 参考（开源 DynamicKV）

对照 `code/DynamicKV/kv_compression/token_drop/` 下分层相关 **methods** 与 **mistral_model_impl** 源码（与本文算法同思路）。
