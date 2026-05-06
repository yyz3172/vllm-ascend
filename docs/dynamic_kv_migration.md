## DynamicKV（vLLM / vLLM-Ascend，PD 场景）说明
## 1. 功能目的与总体策略
### 1.1 目的：压缩的是什么？为什么要压缩？
大模型推理时，注意力需要读写 **KV Cache（Key/Value）**。当 prompt 很长（例如 20K tokens），KV Cache 会占用大量显存，并且在 PD 场景下还会带来：

- **Prefill → Decode 的 KV 传输代价**（传多少 blocks、耗时多少）
- **Decode 侧每步 attention 读取、计算代价**

DynamicKV 的核心思想是：**不必保留所有历史 token 的 KV**。对较旧 token 做"保留重要的少量 token"，并始终保留最近一段窗口以维持局部一致性。

补充：在 PD 场景里，"压缩"有两层含义：
- **少算（compute-side）**：decode 每层用自己的 `kv_len`（更短的 `context_lens`）做 attention，减少算力/带宽。
- **少传 + 少占物理块（memory/transfer-side）**：prefill → decode 只传输压缩后需要的 KV blocks，并让 decode 侧只分配这些 blocks（`kv_cache_usage` 随之下降）。

### 1.2 压缩策略：怎么决定"重要 token"？
压缩是**按层独立**进行的：对第 \(L\) 层来说，会从该层 KV cache 取出完整的序列 \(K_L\)，再用「最后 `window_size` 个 query」去计算它们对**旧段 keys**的注意力权重，从而得到旧 token 的重要性分数。
基于分数，每层会先选出一份 **top-k 候选**；随后再通过跨层预算，确定该层最终保留的旧 token 数 **`old_budget_L`**

- **旧 token（old）**：在第 \(L\) 层按重要性从候选里保留 **`old_budget_L`** 个（与全层预算约束一致）
- **新 token（cur/window）**：最近 `window_size` 个 token， 在 **每一层** 都全保留

第 \(L\) 层逻辑 KV 长度（该层参与注意的有效 prompt token 数）为：`kv_len_L = old_budget_L + window_size`
#### 1.2.1 Per-Head打分 + 并集保留
先说明一个实现约束：vLLM 的 Paged KV Cache 是 **token-level** 的布局——每个 token 在 KV cache 里对应一个“槽位”（slot，可以理解为“这个 token 的 KV 记录位置”），并且同一个 token 的所有 KV heads 都共用这一条记录。因此，不能像开源参考那样做到“**每个 head 保留一套不同的 token 集合**”并在物理上分别压缩；一旦某个 token 被丢弃，就意味着该 token 的所有 heads 都一起丢弃。

为尽量贴近开源语义（不丢掉任何 head 认为重要的 token），这里采用 **per-head 打分 + 并集保留**：
1. **每个 head 各自“打分”**：把旧段里的每个 token 都看作一个候选，让每个 KV head 分别算一遍“它有多重要”。
2. **每个 head 先各自“挑一份”**：每个 head 从旧段里先挑出自己最认可的 top-k 个 token。
3. **把大家挑的合在一起**：把所有 head 选中的 token 取并集，得到最终要保留的 token 集合（保证任何一个 head 觉得重要的 token 都不会被丢掉）。

**与开源的对比**：
- 开源（按 head 压缩）：每个 head 各保留 `budget_size` 个（各 head 的集合可不同）。
- 本实现（token-level 并集）：每个 head 先各自选出一份 top-k，再把所有 head 的结果取并集，形成最终的 token 保留集合。直观上：不同 head 选中的 token 越重合，并集越小、压缩越强；不同 head 的选择越分散，并集越大、压缩越弱。

**保留位置（keep indices）的两种顺序**：
- **重要性顺序（中间态）**：在算法内部，会把“要保留的旧 token 位置”按重要性从高到低排好（可理解为一个 `keep_indices` 列表）。这样当预算从 `budget_size` 收敛到 `old_budget_L` 时，只需要取前 `old_budget_L` 个即可。
- **时间顺序（最终态）**：在真正写回 KV / 导出元数据前，会调整为按 token 时间先后递增的顺序，保证写回后的 KV 是连续前缀，便于 paged attention 只用 `kv_len` 就能正确生效。

### 1.3 分层的关键：跨层预算
分层的意思是：**不同层可以保留不同数量的旧 token**。但在“全局预算”约束下（由参数`prompt_kv_len_budget`决定，平均每层保留的Token数），需要决定每一层最终保留多少旧 token（`old_budget_L`）。

这里有两类共同点，以及两种实现路径（`impl`）的差异。

**共同点（不论 `impl` 取值）**

- 每层先得到一份候选的重要性分布
- 然后把所有层的分数放在一起做一次“跨层分配”：由**跨层预算分配逻辑**决定各层的 `old_budget_L`（会受 `radio_max` / `radio_min` / `prompt_kv_len_budget` 等配置影响），再结合 `window_size` 得到每层最终 `kv_len_L`，并据此写回 KV 或导出元数据。

**差异（取决于 `impl`）**

| 项目 | **`impl=attn`**（`attention_v1.py`） | **`impl=offload`（默认）**（`dynamic_kv_offload.py` + `model_runner_v1.py`） |
|------|--------------------------------------|--------------------------------------------------------------------------|
| 跨层预算什么时候算 | 在 **prefill 的 attention 前向里**边算边积累：除最后一层外，大约 **每 4 层**会用“目前已累积的层”先算一次临时预算，并可能立即截断已算层的候选；到 **最后一层**再用全层信息收尾。 | 在 **prefill 整段 forward 结束后**再统一处理：先通过 hook 拿到各层 `q_last`，再对 **所有层一次性**计算跨层预算并完成改写/导出。 |
| 是否存在“每 4 层一次”的周期 | **有**（仅该路径）。 | **无**（只有一次性全层预算）。 |
| 直觉类比 | 更像“把压缩逻辑嵌进模型前向，随层推进逐步调整”。 | 更像“prefill 先正常写满 KV，结束后再做一次后处理压缩”。 |

---

## 2. 接口与配置参数说明（无新 API，新增配置）
DynamicKV 通过 vLLM 的 `additional_config["dynamic_kv"]` 下发（由 vLLM-Ascend 解析）。
配置入口：`code/vllm-ascend/vllm_ascend/ascend_config.py`
### 2.1 配置项
下面按“你会怎么用”来介绍各参数（从必选到进阶）。

- **enabled**：是否启用 DynamicKV（默认 `false`）。

- **impl**：选择压缩逻辑的执行方式（默认 **`offload`**）。
  - **`offload`（默认，推荐先用）**：prefill 正常写满 KV，prefill 结束后再做压缩。
  - **`attn`**：在 prefill 的 attention 前向内部完成压缩/导出（只在 `PrefillNoCache` 或 ChunkedPrefill 的最后一块触发，见 §3）。更接近“前向内压缩”的形态，但对图捕获/执行路径更敏感。

- **model_types**：允许启用的模型 `model_type` 白名单（默认 `["mistral"]`）。如果你的模型不在列表里，DynamicKV 不会生效。

- **prompt_kv_len_budget**：你希望“**每层保留的 prompt KV 长度**”大致控制在多少（默认 `512`）。它是**目标/预算量级**，不是硬上限（上限仍是 prompt_len）。
  - 直觉：值越小，压缩越激进、越省显存/传输/算力；但质量风险更高。

- **window_size**：每层**无条件保留**的尾部窗口长度（默认 `16`）。同时，它也是“用来打分旧段 token”的 query 窗口长度。
  - 旧段长度：`old_len = max(prompt_len - window_size, 0)`
  - 每层逻辑 KV 长度：`kv_len_L = old_budget_L + window_size`
  - 直觉：`window_size` 越大，越稳但压缩空间越小；越小，越省但更依赖打分/预算分配。

- **radio_max**：旧段 top-k 的**候选池放大系数**（默认 `10.0`）。它主要影响“候选池够不够大”，而不是最终每层一定保留多少。
  - 直觉：`radio_max` 越大，候选池越大（更不容易漏掉重要 token），但计算/并集可能更大。

- **radio_min**：跨层预算分配时给每层 old budget 的**下限系数**（默认 `0.1`）。用于避免某些层被分到过少的旧 token。

- **pooling / kernel_size**：对旧段重要性分数做平滑（默认 `avgpool` + `kernel_size=7`）。
  - `avgpool`：让分数更平滑，减少“单点尖峰”。
  - `maxpool`：更偏向“邻域内只要有高响应就抬高”。
  - `none`：不做平滑。

- **validation_mode**：验证/对照模式（默认 `none`）。
  - `none`：正常压缩（可能改写/pack KV，并导出 `per_layer_kv_lens` 等）。
  - `mask`：不做物理压缩，decode 通过 attention mask 仅关注“重要 token”（便于验证选 token 是否合理）。
  - `zero`：不做物理压缩，但把不重要的 K/V 置零，让 decode 在常规 paged attention 下对比效果。

- **min_rewrite_delta**（仅 `impl=offload`时有效）：当 prompt 只“略微”超过 `prompt_kv_len_budget` 时，是否跳过物理 pack（默认 `128`）。
  - 直觉：设大一些更保守（更少触发 pack），设小一些更激进（更容易触发 pack）。
  - **offload 什么时候会真的做物理 pack（压缩写回 KV）**：仅当 `prompt_len > prompt_kv_len_budget` 且 `prompt_len - prompt_kv_len_budget >= min_rewrite_delta`，并且各层 `q_last` 捕获完整，同时 `validation_mode=none`。

### 2.2 与 PD（Mooncake）的关系

一句话：PD 场景下，prefill 会把“**每层该看多少 KV**”以及（可选）“**要传哪些 KV blocks**”打包到 `kv_transfer_params.dynamic_kv`，随请求传给 decode。

#### 2.2.1 decode 侧最关心的 3 个字段（看懂就够用）
- **`per_layer_kv_lens`（核心）**：长度为 `num_layers` 的列表。第 \(L\) 个值表示“decode 在第 \(L\) 层 attention 里，prompt 部分最多只看这么多 token 的 KV”。  
  - 这是 **PD decode 按层生效** 的主要依据（尤其 `impl=offload` 物理 pack 后，只靠它就能工作）。
- **`per_layer_keep_indices`（可选）**：每层在“原始 prompt 下标空间”里保留了哪些 token。  
  - **`impl=offload` + `validation_mode=none` + 已 pack** 时通常为 `[]`：因为 KV 已经被按时间顺序 pack 到 cache 的连续前缀 `0..kv_len-1`，这时再传原始下标反而容易被误用，所以只依赖 `per_layer_kv_lens`。  
  - `impl=attn` 或未 pack 回退（全长前缀）时，可能会是非空列表，用于对照/校验。
- **`per_layer_important_indices`（仅验证用，可选）**：只在 `validation_mode=mask` 时出现。每层给出“重要 token 的下标”，decode 用它来构造 attention mask 做对照验证。

#### 2.2.2 为了“少传 blocks / 少占物理块”，额外携带哪些字段

当开启“物理块收缩传输”时，会额外出现下面字段（DynamicKV 关闭时不会出现）：

- **`per_layer_num_prompt_blocks`**：每层压缩后的 prompt KV 需要多少 blocks（近似 \(ceil(kv\_len / block\_size)\)）。
- **`per_layer_remote_block_ids`**：每层要传的远端 block id 列表（通常是“同一份前缀 block 列表”的不同截断）。
- **`prefix_remote_block_ids`（offload 专用，强烈推荐有）**：一份“**按逻辑 prompt 顺序排列的前缀 block id 列表**”。Mooncake 会优先用它来 shrink 传输，避免误用 allocator 顺序导致拉错块、decode 乱码（见下一节）。

代理（proxy）会把 prefill 返回的 `kv_transfer_params` 原样传给 decode 请求。

---

## 3. 算法/流程（人话版）
**阅读约定（重要）**
- **`impl=attn`**：压缩逻辑在 prefill 的 attention 前向里完成（见 §3.1、§3.2）。
- **`impl=offload`（默认）**：压缩逻辑在 prefill 结束后一次性完成（见 §2.1 的 `impl` 说明），因此不展开在 §3.1–3.2。

**名词说明（通用，不是 DynamicKV 专有）**
- **PrefillNoCache / ChunkedPrefill**：是 vLLM（及 vLLM-Ascend）里对“prefill 计算阶段”的两种常见形态描述：前者是一次性 prefill（没有历史 cache 命中），后者是把长 prompt 拆成多块分段 prefill；DynamicKV 的 `impl=attn` 只在 **ChunkedPrefill 的最后一块** 或 **PrefillNoCache** 的末尾阶段触发压缩/导出，以保证拿到完整 prompt 的 KV 视图。
  - **如何通过参数控制是否启用 ChunkedPrefill**：这是 vLLM 调度侧的开关，而不是 DynamicKV 的参数。
    - 启用：`--enable-chunked-prefill`（`scheduler_config.enable_chunked_prefill=true`）
    - 关闭：`--no-enable-chunked-prefill`（`scheduler_config.enable_chunked_prefill=false`）

### 3.1 PrefillNoCache（一次性 prefill）— 仅 `impl=attn`适用

这一节只想说明一件事：**在一次性 prefill 的末尾，DynamicKV 会对整段 prompt 做一次“按层压缩/导出”。**

触发条件：`impl=attn`并且本次是一次性 prefill（无历史 cache 命中）

流程上与 **§3.2（ChunkedPrefill 最后一块）** 完全一致，只是这里没有“分块/最后一块”的概念。你可以直接按 §3.2 的步骤理解：在 prefill 末尾计算每层重要性 → 跨层分配预算 → 按层写回/导出。

### 3.2 ChunkedPrefill 最后一块 — 仅 `impl=attn`适用
触发条件：**`impl=attn`**，chunked prefill 且最后一块（以及 DynamicKV、模型类型等前置条件满足）。

这一节只想说明一件事：**chunked prefill 只有在“最后一块”才会触发压缩/导出**，前面的块只是在把 KV 写完整。

流程（只抓关键动作）：

1. 先把本 chunk 的 KV 正常写入 paged cache（保证 cache 完整）
2. （最后一块才做）每层计算旧段 token 重要性，得到候选 keep set（每个 head 先挑 top-k，再做并集）
3. （可选）大约每 4 层先做一次临时跨层分配，并把候选截断到“当前预算”大小
4. 到最后一层：得到最终每层 `old_budget_L`，对每层把 KV 写回到连续前缀 slots（或验证模式仅导出信息）
5. 导出给 PD/decode 使用的结果：`per_layer_kv_lens`（核心），以及（可选）`per_layer_keep_indices` / `per_layer_important_indices`

### 3.3 Decode 如何按层生效（关键点：kv_len 与内容一致）

这一节只想说明一件事：**decode 侧要同时满足“KV 内容对”+“每层长度对”，按层的压缩才会真正生效。**

decode 侧生效依赖两件事：

1. **内容**：在 **`validation_mode=none`**（及物理压缩路径）下，依赖 prefill 侧把各层压缩后的 KV 写回各层 paged cache；**`mask` / `zero`** 等验证路径下可能仍持全量物理 KV，通过 mask 或置零表达「有效注意力」。
2. **长度**：decode attention kernel 在每层使用该层自己的 **`kv_len`**（经 metadata 注入后以 **`context_lens`** 等形式进入 kernel）。

#### 3.3.1 `per_layer_kv_lens` 与 `per_layer_keep_indices`（offload 常为「长度有、indices 空」）

把它当作两条规则就够了：

- **`per_layer_kv_lens`（核心）**：每层 prompt KV 的有效长度。decode 侧“按层生效”主要就靠它。
- **`per_layer_keep_indices`（可选）**：只有在需要“告诉你具体保留了哪些 prompt 下标”时才有用。
  - `impl=attn` 或全长回退时可能是非空列表（便于对照/校验）。
  - **`impl=offload` + `validation_mode=none` + 已 pack** 时常见为 `[]`：这是正常设计，因为 KV 已被 pack 到连续前缀 `0..kv_len-1`，这时只用 `per_layer_kv_lens` 更安全。

你只需要记住：`per_layer_kv_lens` 会一路从 prefill 传到 decode，并最终影响每层 attention 的 `context_lens`。

### 3.4 PD + DynamicKV 关键机制：`transferred_tokens` 与 `decode_extra`

这一节只想说明一件事：**PD 场景下，decode 侧必须知道“prefill 实际传过来的 prompt 容量”，才能让新生成的 KV 被纳入注意力范围。**

- prefill 会把“本次实际传输的 prompt token 容量”记录成 `transferred_tokens`（本质是 `num_blocks * block_size`）。
- decode 会用它推导 `decode_extra`（已经 decode 生成了多少新 token），从而让每层 `context_lens` 随 decode 步数增长：
  - 直觉公式：`context_lens_L ≈ per_layer_kv_lens[L] + decode_extra`

只要这个机制成立，decode 每步新写入的 KV 就能被下一步 attention 正确读取。

---

## 4. 关键日志说明

日志前缀以 **`[DynamicKV]`** 为主；Decode / PD 子阶段见下。

### 4.1 `impl=offload`（Prefill 后处理）相关日志

- **是否已启用 offload 路径**：启动后若出现 **`[DynamicKV][offload] installed ...`**，表示已安装 Q 捕获 hook。

按 `validation_mode` 分：

- **`validation_mode=none`（正常压缩）**：
  - **`[DynamicKV][offload] rewrite ... kv_len min/max/mean ...`**：确认本次确实做了压缩写回，以及 `per_layer_kv_lens` 的量级。
- **`validation_mode=mask` / `zero`（验证模式）**：
  - **`[DynamicKV][offload] validation_mode=...: skip KV pack, full_prefix metadata ...`**：确认本轮不做物理 pack；其中 `mask` 会附带重要位置用于构造 attention mask，`zero` 会把不重要的 K/V 置零。

### 4.2 `impl=attn`（Prefill 前向内）相关日志

按 `validation_mode` 分：

- **`validation_mode=mask`**：
  - **`[DynamicKV][attn] validation_mode=mask: skip KV pack ...`**：确认本轮不做物理 pack、仅导出验证信息（重要位置/掩码）。
- **`validation_mode=none/zero`**：
  - 该路径下日志不一定稳定出现；若配置为 `attn` 但一直没有任何 `[DynamicKV]` 相关输出，优先检查是否处于 ChunkedPrefill 的非最后一块或是否命中图捕获。

### 4.3 Mooncake（PD 传递是否带上分层信息）

- **`[DynamicKV][PD] request_finished per_layer_kv_lens stats`**（及同类汇总日志）
- **`unique` / `min` / `max` / `sum` 的读法**：在 `validation_mode=none` 且已做物理压缩写回时，`per_layer_kv_lens` 常反映各层压缩后长度，可用于观察「分层非均匀」与总量级是否接近 `num_layers × prompt_kv_len_budget` 的量级（允许实现细节导致的偏差）。
- **uniform_fallback（WARNING）**：若看到 **`[DynamicKV] request_finished per_layer_kv_lens(uniform_fallback, not_worker_budget)`**，表示本次未拿到 worker 下发的分层结果，PD 只能用“均匀长度上界”兜底；此时 DynamicKV 的分层收益可能不完整。

---
## 5. 注意事项和待办事项
- **Prefix caching 与 DynamicKV 互斥**：启用 DynamicKV 时，vLLM-Ascend 会在初始化配置阶段 **强制关闭** `enable_prefix_caching`。原因是 DynamicKV 会在 prefill 上 **改写** paged KV 内容；而前缀缓存假设「相同 token 前缀对应可复用的稳定 KV」。若同时开启，可能导致错误复用或与 DynamicKV 导出的 per-layer 元数据不一致。若用户在 YAML/CLI 中打开了 prefix caching，实际运行仍会被置为关闭（日志中有 `forcing prefix caching off` 提示）。
- impl=attn模式下，推理效果不可行，还存在乱码，需要进一步优化

---

## 6. 与开源 DynamicKV 的差异

这一节只想说明一件事：**开源实现的核心思路我们沿用了，但因为 vLLM 的 KV cache 形态与 PD（prefill/decode 分离）架构不同，落地方式会有一些“不得不这样做”的差异。**

### 6.1 架构差异（必要适配）

| 方面　　　　　　　　 | 开源实现　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 本实现　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ----------------------| ----------------------------------------------------------------------------------------------| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **应用场景**　　　　 | HF/Transformers 单机推理　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| vLLM-Ascend 的 PD（prefill/decode 分离）推理　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　																 |
| **KV Cache 形态**　　 | 连续 KV（更像“一整条序列”的缓存）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | Paged KV Cache（按 block/slot 管理，需要通过 block_table 映射）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **集成方式**　　　　 | 通常在模型 forward 内直接改写 cache / 注意力逻辑　　　　　　　　　　　　　　　　　　　　　　　 | 两种方式：默认 `impl=offload`（prefill 结束后再压缩并写回）；可选 `impl=attn`（prefill 前向内压缩，更接近开源形态） |
| **跨层预算触发时机**　 | 取决于开源挂载方式，常见为前向内随层推进　　　　　　　　　　　　　　　　　　　　　　　　　　　 | `impl=attn`：前向内可选“每 4 层一次”的临时分配 + 最后一层收尾；`impl=offload`：prefill 结束后一次性全层分配 |

### 6.2 算法对齐

| 方面　　　　　　 | 开源实现　　　　　　　　　　| 本实现　　　　　　　　　　　　　　　　　　　　　　　|
| ------------------| -----------------------------| -----------------------------------------------------|
| **scores 形状**　| `[bsz, num_heads, old_len]` | `[Hkv, old_len]`（per-head）　　　　　　　　　　　　|
| **跨层 tk**　　　| `base × heads × layers`　　 | `base × Hkv × layers`（一致）　　　　　　　　　　　 |
| **indices 聚合** | per-head gather　　　　　　 | per-head scores + token-level 聚合（兼容 Paged KV） |

**补充说明（结构性差异，非 bug）**：
vLLM 的 Paged KV Cache 以 **token** 为最小管理粒度（一个 token 对应一个 slot，slot 内包含该 token 的所有 KV heads）。
因此无法像开源参考那样做到“每个 head 保留不同的 token 集合并物理压缩”；这里采用 **per-head 打分 + 并集保留** 来保证“只要某个 head 认为重要，这个 token 就不会被丢掉”。
带来的直观影响是：同样的 `budget_size` 下，最终保留的 token 数量落在 \([k, k \times H_{kv}]\) 之间，压缩率不可能与开源逐 head 的实现一一对应。

### 6.3 PD 场景下的工程化扩展

- **分层结果需要跨进程传递**：prefill 侧算出的 `per_layer_kv_lens` 等，需要通过 `kv_transfer_params` 传给 decode 侧才能生效。
- **物理传输/显存收缩（可选）**：在 offload + PD 场景下，可以进一步只传输/只占用“压缩后前缀”所需的 blocks，以降低 decode 侧 KV 占用。

---
