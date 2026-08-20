# TurboQuant Attention Paged 4-bit 高性能设计

## 1. 功能概述

TurboQuant Attention Paged 4-bit 算子是一个融合了 **4-bit 量化 KV cache 写入**、**4-bit 量化 KV cache 反量化** 与 **paged attention 计算** 的 AscendC 自定义算子系统。它包含两个子算子：

1. **Pack 算子**（`turboquant_pack_kv_for_cache_to_cache_4bit`）：将 fp16/bf16 的 K/V 向量量化为 4-bit packed 格式，并直接写入 paged KV cache
2. **Attention 算子**（`turboquant_attention_paged4bit`）：直接从 paged KV cache 中读取 packed 4-bit 行，在片上完成 codebook 查表与旋转矩阵乘，norm 延迟到 QK/PV 阶段应用，再执行 paged attention 计算，最终输出 fp16 attention 结果

### 1.1 4-bit 量化原理

TurboQuant 4-bit 量化采用与 8-bit 相同的 MSE 最优量化框架，但 codebook 大小从 256 缩减为 16：

```
量化流程:
  norms = vector_norm(x, dim=-1, keepdim=True)     # 逐行 L2 范数
  y = (x / (norms + eps)) @ R^T                     # 归一化 + 旋转
  d = (y.unsqueeze(-1) - codebook.view(1,1,-1)).abs()  # 与 codebook 距离
  idx = d.argmin(dim=-1)                            # 最近邻编码，idx ∈ [0, 15]

反量化流程:
  y_hat = codebook[idx]                             # 查表恢复
  z_hat = y_hat @ R                                 # 逆旋转
  x_hat = z_hat * norm                              # 标准反量化输出
```

Attention kernel 中不显式生成完整 `x_hat`：K 路径把 `norm` 乘到 QK score，V 路径把 `norm` 合并到 PV 权重，从而避免对 `M×128` decoded tile 做整行缩放。

4-bit 量化下每个维度仅需 4 bit（0.5 字节）存储索引，相比 8-bit 的 1 字节/维度，KV cache 读取带宽约减少 49%（130B/行 → 66B/行）。若以 fp16 的 128 维向量（256B）为基准，8-bit packed 行实际约为 1.97× 压缩，4-bit packed 行实际约为 3.88× 压缩。

### 1.1.1 首轮性能优先决策

本设计按测试探索算子处理，不要求兼容现有 TurboQuant cache 格式。首版先固定以下决策：

1. **直接使用新 cache 物理布局**：`[num_blocks, num_kv_heads, 8448]`，每个 `(block, head)` 内先放 128 行 64B index，再放 128 个 fp16 norm
2. **Pack 只做 to-cache 主路径**：不以 `[T, H, 66]` packed row 作为运行时接口，避免再通过通用 scatter 表达 norm 分离布局
3. **Pack 首版按行并行**：并行粒度为 `(token, kv_head)`，每个 worker 独立写 64B index 和 2B norm；norm 最终仍连续存储，但写入事务暂不合并
4. **首版固定 block_size=128**：让 `kvTileRows=32` 永远不跨 page block，换取最简单、最高效的连续 DataCopy
5. **反量化延迟 norm 缩放**：Decode 只输出 `codebook[idx] @ R`，K norm 乘到 score，V norm 合并到 PV 权重
6. **Attention 首版实现 Cube QK/PV**：目标是最优性能，QK/PV 都走 Cube/KFC；Vector QK/PV 只保留为调试或极小形状 fallback
7. **首版只支持 v1/v2 codebook lookup 反量化**：`MSE_IMPL=v3` 公式反量化先走 fallback，后续再决定是否进 kernel

### 1.2 4-bit Packed 数据格式

每行 packed 数据由两部分组成：**索引区** 和 **norm 区**。

**8-bit 格式（每行 130 字节）**：
```
|<--- 128 字节索引 --->|<- 2 字节 norm ->|
|  idx[0] ... idx[127] |   fp16 norm     |
  每个索引 1 字节         每行 1 个 norm
```

**4-bit 逻辑行格式（每个 token/head 共 66 字节）**：
```
|<--- 64 字节索引 --->|<- 2 字节 norm ->|
|  idx[0:1] ... idx[126:127] | fp16 norm |
  每字节存 2 个 4-bit 索引    每行 1 个 norm
```

- 索引区：128 个 4-bit 索引打包为 64 字节。约定 `byte[i]` 的低 4 位存 `idx[i]`，高 4 位存 `idx[i+64]`，便于后续用 Shift/And 批量拆出低半区和高半区
- norm 区：2 字节 fp16，与 8-bit 格式相同
- 每个逻辑行总大小：64 + 2 = **66 字节**

### 1.3 性能优先的物理 cache 布局

本 4-bit 方案是测试探索方案，不要求兼容现有 TurboQuant row-major cache。性能优先目标是：**索引行 64B 对齐连续读取，norm 在 block 内连续批量读取，Attention 不再处理 66B 非对齐行 stride**。

8-bit 格式中，每行的 norm 紧跟在该行的索引区后面，即 `[idx_row0 | norm_row0 | idx_row1 | norm_row1 | ...]`。这种交织布局会产生 130B 行 stride。4-bit 若继续使用 `[idx64 | norm2]` 的 66B 行 stride，索引区从第 1 行开始不再保持 32B/64B 对齐，不利于 MTE2 连续搬运。

**最终方案：按 `(block, kv_head)` 分组，block 内索引区与 norm 区分离**

```
交织布局（不采用）:
  [idx0(128B) | norm0(2B) | idx1(128B) | norm1(2B) | ... | idx_{BS-1}(128B) | norm_{BS-1}(2B)]

4-bit 物理布局（采用）:
  cache[block][kv_head]:
  [idx0(64B) | idx1(64B) | ... | idx_{BS-1}(64B) | norm0(2B) | norm1(2B) | ... | norm_{BS-1}(2B)]
  block_bytes = BS × 64 + BS × 2 = BS × 66 字节
```

Python 层建议的 cache 形状：

```python
# key_cache/value_cache 分别传入 kernel 后的视图
key_cache:   torch.uint8  # [num_blocks, num_kv_heads, block_bytes]
value_cache: torch.uint8  # [num_blocks, num_kv_heads, block_bytes]
block_bytes = block_size * 66
```

首版性能优先只支持 `block_size = 128`，因此每个 `(block, kv_head)` slab 固定为 `128 × 66 = 8448B`。若 vLLM 外层仍需要一个 K/V 合并 tensor，则建议 `get_kv_cache_shape()` 返回：

```python
(2, num_blocks, num_kv_heads, 8448)
```

自定义 pack/attention kernel 只按 flat byte buffer 解释 cache，不使用 `[block_size, num_kv_heads, 66]` 的 row-major stride 语义。

**地址公式**：

```cpp
constexpr uint32_t IDX_BYTES_PER_ROW = 64;
constexpr uint32_t NORM_BYTES_PER_ROW = 2;

blockBytes = blockSize * (IDX_BYTES_PER_ROW + NORM_BYTES_PER_ROW);
blockHeadBase = (blockId * numKvHeads + kvHead) * blockBytes;
idxBase = blockHeadBase;
normBase = blockHeadBase + blockSize * IDX_BYTES_PER_ROW;

idxOffset(row)  = idxBase  + row * IDX_BYTES_PER_ROW;   // 64B aligned when block base aligned
normOffset(row) = normBase + row * NORM_BYTES_PER_ROW;  // block 内连续 fp16 norm
```

**优势**：
1. **向量化 norm 读取**：norm 连续存储后，可以一次 DataCopy 读取整个 block 的所有 norm（BS × 2 字节），然后用 Cast 批量转换，避免逐行标量读取
2. **索引区对齐**：64 字节索引区是 32B 对齐的，有利于 DataCopy 效率；而 66 字节行不是 32B 对齐的
3. **Decode 阶段优化**：反量化时按 block 加载 norm，再按 tile 加载索引行；tile 不跨 page block，避免跨 block gather
4. **实现路径更直接**：4-bit 不再走 pack-to-temp + `_npu_reshape_and_cache`，而是新建 pack-to-cache 算子直接按上述物理地址写入

**可行性分析**：
- **Pack 算子**：由 slot_mapping 得到 `(blockId, row)`，对每个 token/head 写 64B 索引和 2B norm。按行并行不会改变 norm 的最终物理地址，所有 worker 都写入同一段 `normBase + row * 2` 连续区域；区别只是写入事务是多个 2B 小写，而不是一次合并写。prefill 连续写入时，norm 后续可按 block/head 合并批量写；decode 单 token 写入时，2B norm 写开销相对 attention 主路径较小
- **Attention 算子**：KV tile 被限制在单个 page block 内。处理一个 block 内 tile 时，索引区是 `rows × 64B` 连续区间，norm 区是 `rows × 2B` 连续区间，MTE2 事务数明显低于逐行 66B gather
- **Python 层**：4-bit cache 宽度不再表示每行 stride，而表示整个 `(block, head)` 的 byte slab。所有 cache 读写必须经过 4-bit 自定义算子

### 1.4 逐 token 写入对 page block 更新的冲击分析

Paged attention 在 decode 阶段逐 token 追加 KV cache。每个新 token 的 K/V 向量被写入当前序列所属 page block 的下一个空闲行。Norm 分离存储布局对这一逐 token 写入流程产生以下影响：

**写入流程对比**：

```
交织布局 (8-bit, 逐 token 写入):
  写 token i → 一次 DataCopy 130B 到 offset i × 130
  (索引与 norm 在同一连续行内，单次写入完成)

分离布局 (4-bit, 逐 token 写入):
  写 token i → 两次写入:
    ① 索引区: DataCopy 64B 到 idxOffset(row)
    ② norm 区: 写 fp16 到 normOffset(row)
  (索引与 norm 位于 block 内不同区域，需两次寻址)
```

**冲击 1：单 token 写入变为两次 GM 写入**

交织布局下，一个 token 的索引和 norm 在同一连续行内，单次 DataCopy 即可完成。分离布局下，索引区位于 block 前部（offset `row × 64`），norm 区位于 block 尾部（offset `BS × 64 + row × 2`），两者相距 `BS × 64 - row × 62` 字节。对于 BS=128 的 block，索引与 norm 相距最多 8192 字节，必须分两次写入。

**影响评估**：单次 2 字节 norm 写入是极小的 burst，GM 带宽利用率低。但 pack 写路径不是性能主瓶颈；decode attention 会反复读取历史 KV，收益远大于单步写入损失。prefill 阶段 slot 往往连续，可以在同一 `(block, head)` 内把多个 norm 合并成一次连续写。

**冲击 2：部分填充 block 的 norm 区空洞**

当 block 未填满时（如 BS=128 但仅写入 3 个 token），norm 区仅有前 3 个 norm 有效，其余为未初始化数据。Attention 算子在读取该 block 时，必须依据 `actual_seq_lengths_kv` 限制扫描范围，不能读取超出实际 KV 长度的 norm。

**影响评估**：这与交织布局下的处理方式一致——Attention 算子始终通过 `causalKvEnd` 限制 KV 扫描范围，不会读取未填充的行。分离布局不引入额外的越界风险。

**冲击 3：block 跨界写入的原子性**

当 token 写入跨越 block 边界时（当前 block 已满，新 token 分配到新 block），新旧两个 block 都需要更新。分离布局下，旧 block 的 norm 区已完整（所有 BS 个 norm 已写入），新 block 的 norm 区从第 0 个 norm 开始写入。

**影响评估**：block 分配由 vLLM scheduler 管理，Pack 算子只负责写入指定 slot。分离布局下，每个 block 的 norm 区独立，跨 block 写入不存在数据依赖。与交织布局行为一致。

**冲击 4：读写在途一致性（Read-During-Write）**

在并发场景下，Attention 算子可能正在读取某个 block 的 norm 区，而 Pack 算子正在向同一 block 追加新 token 的 norm。分离布局下，norm 区的批量读取（一次 DataCopy BS × 2B）与逐个 norm 写入可能产生竞争。

**影响评估**：vLLM 的调度模型保证同一序列的 KV cache 写入与 attention 计算不会同时发生（decode 步内先写 KV 再算 attention）。跨序列的 block 级隔离也保证不同序列的 block 不会冲突。因此读写在途一致性由上层调度保证，分离布局不引入新的竞争风险。

**冲击 5：block 释放与复用**

当序列结束、block 被释放并复用给新序列时，block 内的索引区和 norm 区都需要被新数据覆盖。分离布局下，索引区和 norm 区是同一 block 内的两个区域，block 释放/分配的粒度不变。

**影响评估**：block 复用时，新序列从第 0 行开始写入，覆盖旧数据。分离布局下索引区和 norm 区都会被新数据覆盖，无需额外清理操作。与交织布局行为一致。

**总结**：

| 冲击项 | 严重程度 | 说明 |
|---|---|---|
| 单 token 写入变两次 | 低 | 2B norm 写入开销极小，decode 总写入量小 |
| 部分填充 norm 空洞 | 无 | Attention 始终通过 causalKvEnd 限制扫描范围 |
| block 跨界原子性 | 无 | block 级隔离，无跨 block 数据依赖 |
| 读写在途一致性 | 无 | 上层调度保证读写不并发 |
| block 释放与复用 | 无 | block 粒度不变，新数据覆盖旧数据 |

**结论**：Norm 分离存储在 4-bit 格式下可行且推荐。逐 token 写入的唯一额外开销是 2 字节 norm 的独立写入，但在 decode 场景下该开销可忽略。Attention 侧的批量 norm 读取收益远大于 Pack 侧的小写入开销。

**当前约束**：
- 仅支持 `head_size = 128`、`bits_key = bits_value = 4`
- 仅支持 fp16 query
- GQA group size `G = num_heads / num_kv_heads` 首版按 Cube 路径支持到 tiling 可承载上限；若 `G > qTileHeads`，按 Q head tile 分块处理
- 首版仅支持 `block_size = 128`；这样 `kvTileRows = 32` 永远不跨 page block，norm 区 `128 × 2B = 256B` 天然 32B 对齐
- KV cache 物理格式：`[num_blocks, num_kv_heads, block_size * 66]` uint8（block/head 内 norm 分离存储）

---

## 2. 算子接口

### 2.1 Pack 算子 Python 接口

```python
def turboquant_pack_kv_for_cache_to_cache_4bit(
    *,
    key: torch.Tensor,             # [num_tokens, num_kv_heads, 128] fp16/bf16
    value: torch.Tensor,           # [num_tokens, num_kv_heads, 128] fp16/bf16
    key_cache: torch.Tensor,       # [num_blocks, num_kv_heads, block_size * 66] uint8
    value_cache: torch.Tensor,     # [num_blocks, num_kv_heads, block_size * 66] uint8
    slot_mapping: torch.Tensor,    # [num_tokens] int32, vLLM linear slot id
    block_size: int,               # paged cache block size，首版固定 128
    bits_key: int = 4,             # 必须为 4
    bits_value: int = 4,           # 必须为 4
) -> None
```

### 2.2 Pack 算子参数详细说明

| 参数 | 类型 | 形状 | 取值约束 | 说明 |
|---|---|---|---|---|
| `key` | fp16/bf16 | `[T, H_kv, 128]` | T ≥ 1, H_kv ≥ 1 | 本次写入的 K 向量。T = 新写入的 token 总数 |
| `value` | fp16/bf16 | `[T, H_kv, 128]` | 同 key | 本次写入的 V 向量 |
| `key_cache` | uint8 | `[B, H_kv, BS*66]` | contiguous | 4-bit K cache，按 `(block, kv_head)` slab 存储 |
| `value_cache` | uint8 | `[B, H_kv, BS*66]` | contiguous | 4-bit V cache，布局同 K |
| `slot_mapping` | int32 | `[T]` | 值 ∈ [0, B*BS) | vLLM linear slot id。`blockId = slot // block_size`，`row = slot % block_size` |
| `block_size` | int | 标量 | = 128 | page block 行数。首版固定 128，保证 tile 不跨 block |
| `bits_key` | int | 标量 | = 4 | K 量化位数 |
| `bits_value` | int | 标量 | = 4 | V 量化位数 |

首版不暴露 K/V 独立量化表，K 和 V 使用同一组 4-bit `codebook/R`。Python wrapper 负责从 TurboQuant quantizer 获取表，再调用内部 torch op。

> 性能优先决策：4-bit 不再定义 pack-to-temp rows 作为主路径。主路径必须直接写 cache，因为 norm 分离布局无法通过通用 row-major scatter 高效表达。若需要调试，可另写 reference pack-to-row helper，但不作为运行时路径。

### 2.3 Attention 算子 Python 接口

```python
def turboquant_attention_paged4bit(
    *,
    query: torch.Tensor,           # [num_tokens, num_heads, 128] fp16
    key_cache: torch.Tensor,       # [num_blocks, num_kv_heads, block_size * 66] uint8
    value_cache: torch.Tensor,     # [num_blocks, num_kv_heads, block_size * 66] uint8
    block_tables: torch.Tensor,    # [batch_size, max_blocks_per_seq] int32
    actual_seq_lengths_q: list[int],   # 长度 batch_size，前缀和，最后一个值 = num_tokens
    actual_seq_lengths_kv: list[int],  # 长度 batch_size，每个序列的实际 KV 长度
    head_size: int,                # 必须为 128
    num_heads: int,                # Q 头数，必须 > 0
    num_key_value_heads: int,      # KV 头数，必须 > 0 且整除 num_heads
    block_size: int,               # paged cache 的 block 大小，首版固定 128
    scale: float,                  # attention scale = 1/sqrt(head_size)
    bits_key: int = 4,             # 必须为 4
    bits_value: int = 4,           # 必须为 4
) -> torch.Tensor | None           # [num_tokens, num_heads, 128] fp16，不可用返回 None
```

### 2.4 Attention 算子参数详细说明

| 参数 | 类型 | 形状 | 取值约束 | 说明 |
|---|---|---|---|---|
| `query` | fp16 | `[T, H, 128]` | T ≥ 1, H = num_heads | 所有 token 的 Q 向量。T = batch 中所有序列的 token 总数（含 prefill 和 decode） |
| `key_cache` | uint8 | `[B, H_kv, BS*66]` | B ≥ 1, BS = block_size, H_kv = num_kv_heads | 4-bit packed K cache。B 是 KV cache 池中的总物理块数。每个 `(block, head)` slab 内部为 `[BS*64B idx][BS*2B norm]` |
| `value_cache` | uint8 | `[B, H_kv, BS*66]` | 同 key_cache | 4-bit packed V cache，形状与 key_cache 相同 |
| `block_tables` | int32 | `[batch, max_blocks]` | 值 ∈ [0, B)，-1 表示未使用 | Paged KV cache 的逻辑块→物理块映射表。batch 是当前请求中的序列数；max_blocks 是每序列最多可引用的物理块数。`block_tables[i][j]` 表示第 i 个序列的第 j 个逻辑块对应的物理块编号 |
| `actual_seq_lengths_q` | int list | `[batch]` | 严格递增，最后一个值 = T | Q token 的前缀和序列长度。`actual_seq_lengths_q[i]` 表示前 i+1 个序列的 Q token 累计数。例如 batch=3 且各序列分别有 1、1、128 个 token 时，该列表为 [1, 2, 130]。kernel 通过二分查找此列表确定每个 token 属于哪个序列 |
| `actual_seq_lengths_kv` | int list | `[batch]` | 值 ∈ [0, maxKvLen]，非负 | 每个序列当前实际拥有的 KV token 数，必须是本 step KV 写入后的 post-write 长度。decode 序列的 KV 长度随生成步递增，prefill 序列的 KV 长度等于其本次输入的 token 数 |
| `head_size` | int | 标量 | = 128 | 注意力头维度 |
| `num_heads` | int | 标量 | > 0，且 `num_heads % num_key_value_heads == 0` | Q 头数 |
| `num_key_value_heads` | int | 标量 | > 0 | KV 头数 |
| `block_size` | int | 标量 | = 128 | paged cache block 大小。首版固定为 128 |
| `scale` | float | 标量 | > 0 | attention 缩放因子，通常 = 1/√128 ≈ 0.0884 |
| `bits_key` | int | 标量 | = 4 | K 量化位数 |
| `bits_value` | int | 标量 | = 4 | V 量化位数 |

### 2.5 C++ / Kernel 接口

**Pack 算子**：
```cpp
extern "C" __global__ __aicore__ void turboquant_pack_kv_for_cache_to_cache_4bit(
    GM_ADDR key,              // [T, H_kv, 128] fp16/bf16
    GM_ADDR value,            // [T, H_kv, 128] fp16/bf16
    GM_ADDR slot_mapping,      // [T] int32
    GM_ADDR codebook,         // [16] fp16 — 4-bit codebook, K/V 共用
    GM_ADDR rotation_t,       // [128, 128] fp16 — R^T
    GM_ADDR key_cache,        // [B, H_kv, BS*66] uint8 — 输出
    GM_ADDR value_cache,      // [B, H_kv, BS*66] uint8 — 输出
    GM_ADDR workspace,        // Matmul/KFC system workspace, not tensor scratch
    GM_ADDR tiling            // tiling data
);
```

**Attention 算子**：
```cpp
extern "C" __global__ __aicore__ void turboquant_attention_paged4bit(
    GM_ADDR query,              // [T, H, 128] fp16
    GM_ADDR key_cache,          // [B, H_kv, BS*66] uint8
    GM_ADDR value_cache,        // [B, H_kv, BS*66] uint8
    GM_ADDR block_table,        // [batch, max_blocks] int32
    GM_ADDR actual_seq_len_q,   // [batch] int64
    GM_ADDR actual_seq_len_kv,  // [batch] int64
    GM_ADDR codebook,           // [16] fp16 — 4-bit codebook, K/V 共用
    GM_ADDR rotation,           // [128, 128] fp16 — 逆旋转矩阵 R
    GM_ADDR out,                // [T, H, 128] fp16 — 输出
    GM_ADDR workspace,          // user workspace (scheduling metadata + FlashDecode partial)
    GM_ADDR tiling              // tiling data
);
```

### 2.6 Python wrapper、内部 op 与 table 获取

Python 业务 wrapper 不直接暴露 `codebook/rotation`，保持与现有 attention 调用路径一致。内部 torch op 接收 wrapper 准备好的 table、shape 参数和 workspace size，再由 C++ binding 生成 tiling 并启动 raw AscendC kernel；raw kernel 只接收 GM 地址、system/user workspace 和 tiling。

```python
def turboquant_pack_kv_for_cache_to_cache_4bit(...):
    if bits_key != bits_value or bits_key != 4:
        return fallback
    if key.shape[-1] != 128 or value.shape[-1] != 128 or block_size != 128:
        return fallback
    if current_mse_impl == "v3":
        return fallback

    q = _get_quantizer(128, 4, key.device)
    codebook = q._codebook_fp16       # [16], fp16
    rotation_t = q._rotation_t_fp16   # [128,128], fp16, pack 使用 R^T
    torch.ops._C_ascend.turboquant_pack_kv_for_cache_to_cache_4bit(
        key, value, slot_mapping, codebook, rotation_t,
        key_cache.view(torch.uint8), value_cache.view(torch.uint8), block_size)
```

```python
def turboquant_attention_paged4bit(...):
    if bits_key != bits_value or bits_key != 4:
        return None
    if head_size != 128 or block_size != 128:
        return None
    if num_heads % num_key_value_heads != 0:
        return None
    if current_mse_impl == "v3":
        return None

    q = _get_quantizer(128, 4, query.device)
    codebook = q._codebook_fp16      # [16], fp16
    rotation = q._rotation_fp16      # [128,128], fp16, attention decode 使用 R
    return torch.ops._C_ascend.turboquant_attention_paged4bit(
        query, key_cache.view(torch.uint8), value_cache.view(torch.uint8),
        block_tables, actual_seq_lengths_q, actual_seq_lengths_kv,
        codebook, rotation,
        num_heads, num_key_value_heads, head_size, block_size,
        max(actual_seq_lengths_kv), scale)
```

首版只支持 K/V 使用相同 4-bit table。后续若要实验 K/V 独立 table，再新增接口参数；本探索版本不为兼容预留未使用参数。

---

## 3. 输入场景

### 3.1 DecodeOnly（纯解码）

- **特征**：每个 batch 序列只有 1 个 Q token
- **参数**：`T = batch_size`，`actual_seq_lengths_q = [1, 2, 3, ..., batch_size]`
- **KV 长度**：通常 1024 ~ 32768
- **BNQ = T × H_kv × qHeadTileCount**：通常较小，远小于 AIC 核数
- **splitMode**：当 `max_kv_len ≥ 2048` 且 `BNQ ≤ 0.5 × parallelCoreNum` 时走 SplitBNS（FlashDecode），否则走 SplitBN
- **QK/PV 模式**：首版默认 Cube QK/PV；`G < 16` 时 Q tile pad 到 16 行

**典型配置**：Qwen3-0.6B：`H=4, H_kv=4, G=1`；Qwen3-8B：`H=32, H_kv=8, G=4`

### 3.2 ChunkedPrefill（分块预填充，混合解码）

- **特征**：batch 中部分序列在做 prefill（多个 Q token），部分在做 decode（1 个 Q token）
- **参数**：`T = sum(batch 内 token 数)`，`actual_seq_lengths_q` 为前缀和
- **KV 长度**：prefill 序列的 KV 长度等于其 Q token 数，decode 序列的 KV 长度可能很长
- **splitMode**：通常走 SplitBN（BNQ 较大，不适合 FlashDecode）
- **causal mask**：每个 Q token 只能关注 KV 序列中位置不超过自身的 token。扫描上界必须与 kernel 侧一致：`qAbsPos = kvLen - qLenInSeq + qLocalIdx`，`causalKvEnd = min(kvLen, qAbsPos + 1)`，其中 `kvLen` 是 post-write 长度

**典型配置**：batch 中 1 个 prefill（128 tokens）+ 3 个 decode（各 1 token）

### 3.3 FlashDecode（KV 分段并行）

- **特征**：将一个 decode 请求的 KV 序列切分为多个 segment，分配到不同核并行计算
- **触发条件**：`splitMode = SplitBNS`，即 `max_kv_len ≥ 2048` 且 `BNQ ≤ 0.5 × parallelCoreNum`
- **参数**：`kvSplitPart` = segment 数，`kvSegmentLen` = 每 segment 的 KV 长度
- **输出**：每个 segment 独立计算得到一组 partial 结果：该 segment 内 attention score 的最大值（max，fp32）、softmax 指数和（sum，fp32）、以及归一化前的加权 V 累加值（accum，fp32[128]）。最后由 core 0 执行 CombineFlashDecode，将所有 segment 的 partial 结果按 online softmax 规则合并为最终完整的 attention 输出
- **workspace**：需要额外 HBM 存放 partial 结果，大小 = `num_tokens × num_heads × kvSplitPart × (128×4 + 2×4)` 字节。若后续为了省 HBM 把 partial accum 改为 fp16，必须重新评估长上下文精度

### 3.4 场景与分支矩阵

| 场景 | splitMode | QK/PV 模式 | tiling key | 说明 |
|---|---|---|---|---|
| 短 KV decode | SplitBN (0) | Cube (1) | 1 | 默认路径；Q head 维按 `qTileHeads` 分块 |
| 长 KV decode | SplitBNS (1) | Cube (1) | 3 | FlashDecode + Cube QK/PV |
| ChunkedPrefill | SplitBN (0) | Cube (1) | 1 | BNQ 大，不走 FlashDecode，仍使用 Cube |
| Debug/极小形状 | SplitBN/SplitBNS | Vector (0) | 0/2 | 仅用于 bringup/fallback，不作为性能目标 |

---

## 4. Tiling 设计

### 4.1 基本任务单元

**`(tokenIdx, kvHead, qHeadTile)` 三元组** — 为一个 Q token、一个 KV head 下的一组 Q heads 计算完整 attention。

**基本任务数**：`BNQ = num_tokens × num_kv_heads × qHeadTileCount`

其中：

```text
G = num_heads / num_kv_heads
qTileHeads = 16  // Cube M 维按 16 对齐；G < 16 时 pad 到 16
qHeadTileCount = ceil(G / qTileHeads)
```

首版推荐 `qTileHeads = 16`。这样即使 `G=1/4/8`，Cube QK/PV 仍以 16 行 Q tile 运行，padding 的输出不写回。该策略牺牲小 G 的少量无效计算，换取统一 Cube 数据流和更少标量热点。

### 4.2 两种调度模式的核心区别

| | SplitBN（标准模式） | SplitBNS（FlashDecode） |
|---|---|---|
| **KV 序列是否跨核切分** | **否** — 每个 `(tokenIdx, kvHead, qHeadTile)` 的完整 KV 序列在同一个核上串行处理 | **是** — 每个 `(tokenIdx, kvHead, qHeadTile)` 的 KV 序列被切分为多个 segment，分配到不同核并行处理 |
| **跨核合并** | 不需要 — 每个核独立产出最终结果 | 需要 — 各核产出 partial 结果，由 core 0 执行 CombineFlashDecode 合并 |
| **额外 workspace** | 不需要 | 需要额外 HBM 存放 partial 结果 |
| **适用场景** | BNQ 较大（多 token × 多 KV head × Q tile），每个任务的 KV 长度适中 | BNQ 较小（少 token × 少 KV head × Q tile），但单个任务的 KV 序列很长 |
| **负载均衡方式** | 将长短不同的任务搭配分配给各核 | 将 KV 序列切分为等长 segment，天然均衡 |
| **触发条件** | 默认模式 | `maxActualSeqLen ≥ 2048` 且 `BNQ ≤ 0.5 × parallelCoreNum` |

**直觉理解**：

- SplitBN 像"分活儿"：把不同的活儿（token×KV head×Q head tile）分给不同的人，每个人干完自己的活儿直接交差
- SplitBNS 像"拆活儿"：一个活儿太大（KV 太长），拆成几段分给不同人同时干，最后汇总

### 4.3 负载不均衡问题

不同序列的 KV 长度不同，导致不同 `(tokenIdx, kvHead, qHeadTile)` 任务的执行时间差异巨大。一个任务的执行时间正比于其 KV 扫描行数，而不同序列的 KV 长度可能相差数十倍。

**示例**：batch 中序列 A 的 KV 长度为 1024，序列 B 的 KV 长度为 32768。同一 KV head 下，序列 B 的任务耗时是序列 A 的 32 倍。若两者被分配到不同核，则处理序列 A 的核将空闲等待。

**影响程度分析**：

| 场景 | KV 长度差异 | 负载不均衡程度 | 说明 |
|---|---|---|---|
| 纯 Decode（同模型同时刻） | 小（各序列 KV 长度相近） | **低** | 同一调度步的 decode 序列 KV 长度通常相差不超过 2-3× |
| ChunkedPrefill（混合 decode + prefill） | 大（prefill 几十 vs decode 数千） | **高** | prefill 序列 KV 短，decode 序列 KV 长，差异可达 100× |
| 多用户不同进度 | 中（KV 长度 1k~30k） | **中** | 用户请求先后到达，KV 长度分散 |

**SplitBNS 天然均衡**：SplitBNS 将 KV 序列切分为等长 segment，各核处理的 segment 数量相同，负载天然均衡（仅最后一个 segment 可能不满，影响可忽略）。

**SplitBN 需要额外处理**：SplitBN 不切分 KV 序列，长短任务耗时差异大，需要通过加权分配来缓解。

### 4.4 SplitBN 模式（标准模式）

将 BNQ 个基本任务分配到 `usedCoreNum` 个核，每个核串行处理其分配的所有 `(tokenIdx, kvHead, qHeadTile)`，每个任务内按 KV tile 遍历完整 KV 序列，无需跨核合并。

#### 方案 A：按基本任务数均匀分配

```
BNQ tasks: [0, 1, 2, ..., BNQ-1]
         ↓ 均匀分配
Core 0:  [0, ..., blockSplitRange-1]
Core 1:  [blockSplitRange, ...]
...
Core usedCoreNum-1: [..., BN-1]
```

每个核分配到相同数量的基本任务（最多差 1）。**问题**：不感知 KV 长度，长短任务分配不均导致负载不均衡。

#### 方案 B：按 KV 工作量加权分配

Host 侧读取 `actual_seq_lengths_kv`，计算每个基本任务的 KV 工作量（以 KV 行数为度量），然后按总工作量均匀分配到各核：

```
BNQ tasks: [task_0, task_1, ..., task_{BNQ-1}]
每个 task_i 的 KV 行数: kvRows[i] = causalKvEnd[i]
总 KV 行数: totalKvRows = Σ kvRows[i]
目标每核 KV 行数: kvRowsPerCore = ceil(totalKvRows / usedCoreNum)

加权分配:
Core 0:  从 task_0 开始，累加 kvRows 直到达到 kvRowsPerCore
Core 1:  从 Core 0 结束处继续
...
Core usedCoreNum-1: 剩余所有任务
```

- `usedCoreNum = min(BNQ, parallelCoreNum)`，`parallelCoreNum = min(aicNum, 8)`
- 每个核分配到大致相同数量的 KV 行，而非相同数量的基本任务
- 同一 `(tokenIdx, kvHead, qHeadTile)` 的所有 KV tile 在同一核上串行完成，无需 CombineFlashDecode

Host 侧计算 `causalKvEnd[i]` 必须与 kernel 侧使用同一公式：`qStart/qEnd` 由 `actual_seq_lengths_q` 得到，`kvLen` 使用 post-write `actual_seq_lengths_kv[seqIdx]`，`qAbsPos = kvLen - qLenInSeq + qLocalIdx`，`causalKvEnd = min(kvLen, qAbsPos + 1)`。否则 ChunkedPrefill 下的工作量估计会偏离实际扫描行数。

> 当 batch 内各序列 KV 长度相近时，方案 A 和方案 B 等价；当差异大时，方案 B 显著改善负载均衡。

### 4.5 SplitBNS 模式（FlashDecode）

将每个 `(tokenIdx, kvHead, qHeadTile)` 基本任务的 KV 序列切分为 `kvSplitPart` 个 segment，分配到不同核并行处理。SplitBNS 首版采用 **连续 range 分配**，不再按 stride 跳跃分配，kernel 可由 `coreIdx` 直接算出本核的线性任务范围：

```
totalSplitTasks = BNQ × kvSplitPart
linearTaskId ∈ [0, totalSplitTasks)

// segment-major 线性化，先枚举 segIdx，再枚举 BNQ 基本任务
segIdx  = linearTaskId / BNQ
bnqTask = linearTaskId % BNQ
Task(linearTaskId) = (tokenIdx, kvHead, qHeadTile, segIdx)

usedCoreNum = min(parallelCoreNum, totalSplitTasks)
coreStart = floor(coreIdx × totalSplitTasks / usedCoreNum)
coreEnd   = floor((coreIdx + 1) × totalSplitTasks / usedCoreNum)

Core 0:  [coreStart(0), coreEnd(0))
Core 1:  [coreStart(1), coreEnd(1))
...
```

- `kvSplitPart = parallelCoreNum / BNQ`（向下取整，最小为 1）
- `kvSegmentLen = AlignUp(CeilDiv(maxActualSeqLen, kvSplitPart), blockSize)`
- 每个 segment 独立计算 partial attention 结果（max, sum, accum）
- 最后由 core 0 执行 CombineFlashDecode 合并所有 segment

这里的 `blockSize` 是 paged KV cache 的一个 page block 中包含的 KV 行数，不是字节数；首版固定为 128。`kvSegmentLen` 也以 KV 行为单位。按 `blockSize` 对齐后，每个 segment 的起点都落在 page block 边界，block_table 查找和 idx/norm 连续 DataCopy 更简单；实际扫描范围仍然用 `kvEnd = min((segIdx + 1) × kvSegmentLen, causalKvEnd)` 裁剪，不会读取超过真实 KV 长度的数据。

**SplitBNS 的直接收益**：

1. 当 `BNQ` 很小但 `kvLen` 很长时，SplitBN 只能使用约 `BNQ` 个 core；SplitBNS 可把每个长 KV 任务拆成 `kvSplitPart` 段，使 `usedCoreNum ≈ BNQ × kvSplitPart`，提高 AIC/AIV 占用率
2. 每个 segment 只扫描约 `kvSegmentLen` 行 KV，单个 attention 任务的尾延迟从完整 `kvLen` 降到约 `kvLen / kvSplitPart`，再加一次 partial combine
3. segment 起点按 page block 对齐后，一个 segment 处理完整 block 集合，减少跨 block 边界上的分支和小粒度 DataCopy

**SplitBNS 的负载均衡特点**：

SplitBNS 模式下，所有基本任务的 KV 序列被统一切分为等长的 segment，天然具有较好的负载均衡。但存在两个残余不均衡：

1. **最后一个 segment 可能不满**：若 `causalKvEnd` 不是 `kvSegmentLen` 的整数倍，最后一个 segment 较短
2. **不同序列的 causalKvEnd 不同**：短序列的最后一个 segment 可能远短于 `kvSegmentLen`

这些残余不均衡在实际中影响较小（仅最后一个 segment），可接受。

### 4.6 KV Tile 切分

在每个 `(tokenIdx, kvHead, qHeadTile, segIdx)` 任务内，KV 序列按 `kvTileRows = 32` 行为一个 tile 串行处理：

```
KV[0, kvTileRows)     → tile 0: Load + Decode + QK + Softmax
KV[kvTileRows, 2*kvTileRows) → tile 1: Load + Decode + QK + Softmax
...
KV[last_start, causalKvEnd)   → last tile (可能不满 32 行)
```

### 4.7 Tiling 数据结构

```cpp
BEGIN_TILING_DATA_DEF(TurboquantAttentionPaged4bitTilingData)
// ---- Cube Matmul Tiling ----
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, decodeRotateTiling);  // decode 旋转矩阵乘 tiling
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, qkTiling);            // QK Cube matmul tiling (新增)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, pvTiling);            // PV Cube matmul tiling (新增)

// ---- 模型参数 ----
TILING_DATA_FIELD_DEF(uint32_t, numTokens);       // Q token 总数
TILING_DATA_FIELD_DEF(uint32_t, batchSize);       // batch 大小
TILING_DATA_FIELD_DEF(uint32_t, numHeads);        // Q 头数 H
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);      // KV 头数 H_kv
TILING_DATA_FIELD_DEF(uint32_t, gqaGroupSize);    // GQA group = H / H_kv
TILING_DATA_FIELD_DEF(uint32_t, qTileHeads);      // Cube QK/PV 的 Q head tile，首版 16
TILING_DATA_FIELD_DEF(uint32_t, qHeadTileCount);  // ceil(G / qTileHeads)
TILING_DATA_FIELD_DEF(uint32_t, headSize);        // 头维度，固定 128
TILING_DATA_FIELD_DEF(uint32_t, blockSize);       // paged cache block 行数，首版固定 128，不是字节数
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerSeq); // 每序列最大 block 数
TILING_DATA_FIELD_DEF(uint32_t, totalCacheBlocks); // 总物理 block 数
TILING_DATA_FIELD_DEF(uint32_t, maxKvLen);        // = maxBlocksPerSeq × blockSize
TILING_DATA_FIELD_DEF(uint32_t, maxActualSeqLen); // 实际最大 KV 长度

// ---- Tiling 决策 ----
TILING_DATA_FIELD_DEF(uint32_t, kvTileRows);      // KV tile 行数，固定 32
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);     // 实际使用的逻辑核数
TILING_DATA_FIELD_DEF(uint32_t, splitMode);       // 0=SplitBN, 1=SplitBNS
TILING_DATA_FIELD_DEF(uint32_t, kvSplitPart);     // FlashDecode segment 数，SplitBN 时为 1
TILING_DATA_FIELD_DEF(uint32_t, qkPvMode);        // 首版默认 1=Cube；0=Vector 仅 fallback
TILING_DATA_FIELD_DEF(uint32_t, kvSegmentLen);    // 每 segment 的 KV 长度

// ---- SplitBN 加权分配参数 (新增) ----
// 每个 core 的任务范围由 coreTaskStart[coreIdx] 和 coreTaskEnd[coreIdx] 描述，
// 存储在 user workspace 中（因为大小与 usedCoreNum 成正比，不适合放入固定大小 tiling）。
// tiling 中保存 byte offset，kernel 不通过固定地址假设 workspace 布局：
TILING_DATA_FIELD_DEF(uint64_t, coreTaskStartOffset); // byte offset，uint32_t[usedCoreNum]
TILING_DATA_FIELD_DEF(uint64_t, coreTaskEndOffset);   // byte offset，uint32_t[usedCoreNum]
TILING_DATA_FIELD_DEF(uint32_t, totalKvRows);     // 所有基本任务的 KV 行总数
TILING_DATA_FIELD_DEF(uint32_t, kvRowsPerCore);   // 每 core 目标 KV 行数

// ---- FlashDecode workspace ----
TILING_DATA_FIELD_DEF(uint64_t, partialAccumOffset); // byte offset，fp32[numTokens,numHeads,kvSplitPart,128]
TILING_DATA_FIELD_DEF(uint64_t, partialMaxOffset);   // byte offset，fp32[numTokens,numHeads,kvSplitPart]
TILING_DATA_FIELD_DEF(uint64_t, partialSumOffset);   // byte offset，fp32[numTokens,numHeads,kvSplitPart]
TILING_DATA_FIELD_DEF(uint64_t, workspaceSize);      // user workspace 总字节数，32B 对齐
TILING_DATA_FIELD_DEF(uint32_t, partialAccumSize);   // partial accum 元素数，numTokens*numHeads*kvSplitPart*128
TILING_DATA_FIELD_DEF(uint32_t, partialMaxSize);     // partial max 元素数，numTokens*numHeads*kvSplitPart
TILING_DATA_FIELD_DEF(uint32_t, partialSumSize);     // partial sum 元素数，numTokens*numHeads*kvSplitPart

// ---- 其他 ----
TILING_DATA_FIELD_DEF(float, scaleValue);         // attention scale
TILING_DATA_FIELD_DEF(uint32_t, dataCores);       // 新增：实际 AIV 核数
END_TILING_DATA_DEF;
```

> **8-bit → 4-bit Tiling 变化**：8-bit 使用 `formerCoreNum/blockSplitRange/tailSplitRange` 三个字段描述按基本任务数的均匀分配；4-bit 改为 `totalKvRows/kvRowsPerCore` 描述按 KV 工作量的加权分配。Kernel 侧通过读取 workspace 中的 `coreTaskStart/coreTaskEnd` 数组确定每个核的任务范围。

**user workspace 布局**：

```text
base = workspace

SplitBN 加权分配:
  coreTaskStartOffset = Align32(0)
  coreTaskEndOffset   = Align32(coreTaskStartOffset + usedCoreNum * sizeof(uint32_t))
  scheduleEndOffset   = Align32(coreTaskEndOffset + usedCoreNum * sizeof(uint32_t))

SplitBNS FlashDecode partial:
  partialMaxOffset    = scheduleEndOffset
  partialSumOffset    = Align32(partialMaxOffset + partialMaxSize * sizeof(float))
  partialAccumOffset  = Align32(partialSumOffset + partialSumSize * sizeof(float))
  workspaceSize       = Align32(partialAccumOffset + partialAccumSize * sizeof(float))
```

`partial*Size` 全部是 fp32 元素数，只有 `partial*Offset/workspaceSize` 是字节数。SplitBN 不使用 partial 区；SplitBNS 可不读取 `coreTaskStart/coreTaskEnd`，因为连续 range 可由 `totalSplitTasks/usedCoreNum/coreIdx` 直接计算，但 offset 仍按统一 workspace 布局保留。该 workspace 只保存调度元数据和 FlashDecode partial 结果，不保存 Decode/QK/PV 中间 tensor。

### 4.8 Tiling 决策流程

```
输入: numTokens, numHeads, numKvHeads, blockSize, maxActualSeqLen, aicNum, actualSeqLenKv[]

1. 计算派生量:
   gqaGroup = numHeads / numKvHeads          // GQA group size
   qTileHeads = 16                            // Cube QK/PV Q tile
   qHeadTileCount = ceil(gqaGroup / qTileHeads)
   bnq = numTokens × numKvHeads × qHeadTileCount
   maxKvLen = maxBlocksPerSeq × blockSize     // 最大 KV 长度
   kvTileRows = 32 (固定)                    // KV tile 行数

2. 选择可用逻辑 core:
   parallelCoreNum = min(aicNum, 8)            // 首版 MIX core 上限

3. 选择 splitMode:
   if maxActualSeqLen ≥ 2048 AND bnq ≤ 0.5 × parallelCoreNum:
       splitMode = SplitBNS (1)              // FlashDecode
   else:
       splitMode = SplitBN  (0)              // 标准模式

4. 选择 qkPvMode:
   qkPvMode = Cube (1)                       // 性能优先默认路径
   // Vector (0) 仅用于 bringup/debug，或 Cube tiling 失败时 fallback

5. 分配核任务:
   if splitMode == SplitBNS:
       SplitBns(bnq, maxActualSeqLen, blockSize, parallelCoreNum)
       → usedCoreNum, kvSplitPart, kvSegmentLen
   else:
       SplitBnWeighted(bnq, actualSeqLenKv, parallelCoreNum)
       → usedCoreNum, totalKvRows, kvRowsPerCore
       → coreTaskStart[], coreTaskEnd[] 写入 workspace

6. 设置 blockDim:
   blockDim = usedCoreNum × 3  // 每个 MIX core = 1 AIC + 2 AIV
```

---

## 5. 量化算子设计（Pack）

### 5.1 量化流程

Pack 算子将 fp16/bf16 的 K/V 向量量化为 4-bit packed 格式。流程与现有 8-bit pack 算子（`turboquant_pack_kv_for_cache_v2`）基本一致，主要区别在 Encode 阶段：

```
输入: key[T, H_kv, 128] fp16, value[T, H_kv, 128] fp16

对每行 x (128 维 fp16):
  1. Normalize: norms[i] = ||x[i]||; x[i] /= (norms[i] + eps)
  2. Rotate:    y = x @ R^T    (Cube KFC matmul [1,128]×[128,128])
  3. Encode:    对 y 的每个维度，找 codebook 中最近邻的索引
  4. Pack:      将 128 个 4-bit 索引打包为 64 字节，附 2 字节 fp16 norm
```

### 5.2 4-bit Encode 设计

8-bit pack 算子中，codebook 大小为 256，使用 `WholeReduceMin` 在 256 个距离中找最小值。4-bit codebook 大小为 16，距离向量仅 16 元素，使用 `WholeReduceMin` 效率低（mask=64，需填充）。

**4-bit Encode 方案**：直接使用标量遍历 16 个 codebook 项，因为 16 次标量比较的开销远小于 WholeReduceMin 的 sync 开销。

```cpp
__aicore__ inline uint8_t Argmin4bitScalar(float yf, const LocalTensor<half>& cbLocal) {
    float best = AbsF32(yf - static_cast<float>(cbLocal.GetValue(0)));
    uint8_t bestIdx = 0;
    for (uint32_t k = 1; k < 16; ++k) {
        const float d = AbsF32(yf - static_cast<float>(cbLocal.GetValue(k)));
        if (d < best) {
            best = d;
            bestIdx = static_cast<uint8_t>(k);
        }
    }
    return bestIdx;
}
```

**4-bit Pack 索引打包**：

```cpp
// 将 128 个 4-bit 索引打包为 64 字节
// byte[i] 的低 4 位存 idx[i]，高 4 位存 idx[i+64]
for (uint32_t i = 0; i < 64; ++i) {
    uint8_t low = indices[i] & 0x0F;
    uint8_t high = indices[i + 64] & 0x0F;
    packedRow.SetValue(i, low | (high << 4));
}
```

### 5.3 Norm 分离存储写入

4-bit 格式采用 norm 分离存储，Pack 算子直接写 cache。每个 token 的 `slot_mapping[t]` 映射到 `(blockId, row)`，每个 KV head 独立写一个 `(block, kv_head)` slab。

```
交织布局写入 (8-bit):
  for i in [0, m):
    write packedRow[i] (128B idx + 2B norm) → GM at row_i * 130

分离布局写入 (4-bit):
  slot = slot_mapping[token]
  blockId = slot / 128
  row = slot % 128
  blockHeadBase = (blockId * numKvHeads + kvHead) * 8448
  idxOffset = blockHeadBase + row * 64
  normOffset = blockHeadBase + 128 * 64 + row * 2

  write packedIdx[token, kvHead] (64B) → cache + idxOffset
  write norm[token, kvHead] (2B)       → cache + normOffset
```

首版采用按行并行写入，保证 pack 实现简单并利于排查正确性：并行任务为 `(token, kvHead)`，每个任务独立完成 normalize、rotate、encode、pack，并写入对应 cache 地址。此时 norm 仍然是连续存储：第 `row` 行 norm 固定写到 `normBase + row * 2`。不连续的是 **写入事务**，因为不同 worker 分别发起 2B 写。prefill 连续写入时，slot 往往在同一 block 内单调递增，二阶段可按 `(blockId, kvHead)` 聚合 UB 中的 norm，再对连续 row 发起一次 `rows × 2B` 的合并写。

### 5.4 Pack 算子 UB 预算

| 缓冲区 | 大小 | 说明 |
|---|---|---|
| xBatchQue_ | 2 × 32×128×2B = 16,384 B | KFC A 输入队列 |
| aBatchQue_ | 2 × 32×128×2B = 16,384 B | KFC A 副本队列 |
| yBatchQue_ | 2 × 32×128×2B = 16,384 B | KFC C 输出队列 |
| packedRowQue_ | 2 × 32×66B = 4,224 B | packed 行输出队列 |
| normsBuf_ | 32 × 2B = 64 B | 逐行 norm |
| codebookBuf_ | 16 × 2B = 32 B | 4-bit codebook (远小于 8-bit 的 512B) |
| rotationTBuf_ | 128×128×2B = 32,768 B | R^T 矩阵 |
| normScalarBuf_ | 32 B | ReduceSum 结果 |
| reduceOutBuf_ | 128×2×4B = 1,024 B | 归一化临时 |
| rotateWorkBuf_ | 32,768 B | KFC workspace |
| 其他 | ~4 KB | bf16 临时等 |
| **合计** | **~122 KB** | 占 192 KB 的 64% |

4-bit codebook (32B) 远小于 8-bit (512B)，但整体 UB 用量主要由 KFC 队列和旋转矩阵决定，因此节省有限。

---

## 6. 反量化算子设计（Decode，Attention 内部）

### 6.1 4-bit 反量化流程

```
输入: packed[M, 64] uint8 (索引区), norms[M] fp16 (norm 区, 已批量加载)
输出: zHat[M, 128] fp16，未乘 norm

  1. Unpack: 将 64 字节拆包为 128 个 4-bit 索引
     for i in [0, 128):
       if i < 64:
         idx[i] = packed[i] & 0xF
       else:
         idx[i] = (packed[i - 64] >> 4) & 0xF
  2. LUT:    y_hat[k] = codebook[idx[k]]  (16 项 LUT 查表)
  3. Rotate: z_hat = y_hat @ R             (Cube KFC matmul)
  4. Norm:   延迟到 QK/PV 阶段应用，避免对 M×128 元素逐行乘 norm
```

**性能优先决策：延迟 norm 缩放**

TurboQuant 的反量化为 `x = (codebook[idx] @ R) * norm`。在 attention 中可以把 norm 缩放延迟到更小的张量上：

```text
K: score[g, m] = dot(q[g], zK[m]) * normK[m] * scale
V: out[g, :]  += prob[g, m] * normV[m] * zV[m, :]
```

这样 K 路径避免 `M×128` 次乘 norm，改为 `G×M` 次 score 缩放；V 路径把 norm 合并到 softmax probability，避免在 decoded V tile 上做一整行 `Muls`。首版 DecodeRows4bit 输出未乘 norm 的 `z_hat`，同时返回对应 tile 的 `normK/normV`。

### 6.2 4-bit Unpack 设计

4-bit 索引的拆包是 8-bit 反量化中不存在的额外步骤。有两种方案：

**方案 A：标量拆包 + 向量化查表**

```cpp
// 标量拆包: 64 字节 → 128 个 uint8 索引
uint8_t indices[128];
for (uint32_t i = 0; i < 64; ++i) {
    uint8_t byte_val = packed.GetValue(i);
    indices[i]      = byte_val & 0xF;
    indices[i + 64] = (byte_val >> 4) & 0xF;
}
// 向量化查表: 128 个索引 → 128 个 fp16 值
// 将 indices 写入 UB LocalTensor，然后用 Gather 查表
```

**方案 B：向量化拆包（待 probe）**

```cpp
// Step 1: Cast uint8 → fp16 (zero-extend)
Cast(idxHalf, packed, CAST_NONE, 64);  // 64 字节 → 64 个 fp16
PipeBarrier<PIPE_V>();

// Step 2: 提取低 4 位和高 4 位
// 低 4 位: idx_lo = idx & 0xF
auto idxLo = idxLoBuf_.Get<half>();   // [64]
auto idxHi = idxHiBuf_.Get<half>();   // [64]
Duplicate(maskBuf_, (half)0xF, 64);
BitwiseAnd(idxLo, idxHalf, maskBuf_, 64);  // 需要确认 AscendC 是否支持 half 的 BitwiseAnd

// 若 AscendC 不支持 half BitwiseAnd，则用算术方式:
// idx_lo = idx - 16 * floor(idx / 16)  (即 idx % 16)
// idx_hi = floor(idx / 16)
```

**首版方案**：由于 AscendC 对 half 的位操作支持有限，先采用 **标量拆包 + 向量化 Gather** 的混合方案。标量拆包 64 字节仅需 64 次 GetValue（远少于 8-bit 的 128 次），且拆包后的 Gather 查表是向量化操作，整体开销可接受。向量化拆包作为后续 probe 项，不作为首版默认路径。

```cpp
void DecodeRows4bit(
    const LocalTensor<uint8_t>& packedIdx,   // [M, 64] 索引区
    const LocalTensor<half>& normsLocal,      // [M] norm 区 (已批量加载；Decode 内不乘，只透传给 QK/PV)
    const LocalTensor<half>& codebook,        // [16] fp16
    GlobalTensor<half>& rotationGm,
    TqDecodeRotateMatmulOp& rotateMm,
    const LocalTensor<half>& yHat,            // [M, 128] 临时
    const LocalTensor<uint8_t>& rotateWork,
    const LocalTensor<half>& zHat,            // [M, 128] 输出，Matmul C 直接落片上，未乘 norm
    uint32_t M, uint32_t mPad)
{
    const uint32_t D = 128;
    const uint32_t IDX_BYTES = 64;  // 128 个 4-bit 索引打包为 64 字节

    // Step 1: 标量拆包 4-bit 索引 → 向量化 Gather
    auto idxS32 = idxS32Buf_.Get<int32_t>();  // [M*128]
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t b = 0; b < IDX_BYTES; ++b) {
            uint8_t byteVal = packedIdx.GetValue(i * IDX_BYTES + b);
            uint8_t low = byteVal & 0xF;
            uint8_t high = (byteVal >> 4) & 0xF;
            idxS32.SetValue(i * D + b,      low * sizeof(half));
            idxS32.SetValue(i * D + b + 64, high * sizeof(half));
        }
    }

    // Step 2: Gather 查表 y_hat = codebook[idx]
    Gather(yHat, codebook, idxS32.ReinterpretCast<uint32_t>(), 0, M * D);
    PipeBarrier<PIPE_V>();

    // Step 3: Cube 旋转 z_hat = y_hat @ R。norm 延迟到 QK/PV 阶段应用。
    // 性能路径要求 C 为 VECIN/VECOUT local tensor，不能写 GM 后再读回。
    if (mPad > M) {
        Duplicate(yHat[M * D], (half)0, (mPad - M) * D);
        PipeBarrier<PIPE_V>();
    }
    rotateMm.SetOrgShape(mPad, D, D);
    rotateMm.SetSingleShape(M, D, D);
    rotateMm.SetTensorA(yHat, false);
    rotateMm.SetTensorB(rotationGm, false);
    rotateMm.SetLocalWorkspace(rotateWork);
    rotateMm.IterateAll(zHat);
    rotateMm.End();
}
```

Decode rotate 的 Matmul 类型应对齐现有 pack v2 的片上 C 模式：

```cpp
using TqDecodeRotateAT = MatmulType<TPosition::VECOUT, CubeFormat::ND, half>;
using TqDecodeRotateBT = MatmulType<TPosition::GM,     CubeFormat::ND, half>;
using TqDecodeRotateCT = MatmulType<TPosition::VECIN,  CubeFormat::ND, half>;
```

现有 8-bit decode rotate 里 `C=GM fp32` 再 `DataCopy` 回 UB 的实现不作为 4-bit 方案参考。若某个 CANN 版本无法让 rotate C 直接落片上，应先用 probe 明确限制并调整 Matmul 配置；不能在 4-bit 性能路径里引入 GM 中间结果。

### 6.3 Norm 批量加载

Norm 分离存储后，Attention 算子可以在处理一个 block 内连续 KV 行时，一次性加载当前 tile 的 norm。K 和 V 的 norm 不同，必须分别加载到 `normKBuf_` 与 `normVBuf_`。

```cpp
void LoadTileNorms(__gm__ uint8_t* cacheBase,
                   LocalTensor<half>& normBuf,
                   uint32_t blockId,
                   uint32_t kvHead,
                   uint32_t rowStart,
                   uint32_t rows)
{
    constexpr uint32_t IDX_BYTES_PER_ROW = 64;
    constexpr uint32_t BLOCK_SIZE = 128;
    constexpr uint32_t BLOCK_BYTES = BLOCK_SIZE * 66;  // 8448
    const uint64_t blockHeadBase =
        (static_cast<uint64_t>(blockId) * numKvHeads_ + kvHead) * BLOCK_BYTES;
    const uint64_t normBase = blockHeadBase + BLOCK_SIZE * IDX_BYTES_PER_ROW;
    const uint64_t normGmOffset = normBase + rowStart * sizeof(half);

    if (rows * sizeof(half) >= 32 && (rows * sizeof(half)) % 32 == 0) {
        GlobalTensor<half> normGm;
        normGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(cacheBase + normGmOffset),
                               rows);
        DataCopy(normBuf, normGm, rows);
    } else {
        // Last tile may be smaller than 32B. First version reads a 32B
        // aligned range into a temporary half buffer, then copies only rows
        // into normBuf. This avoids a sub-32B direct DataCopy.
        const uint64_t alignedOffset = normGmOffset & ~static_cast<uint64_t>(31);
        const uint32_t elemShift = (normGmOffset - alignedOffset) / sizeof(half);
        GlobalTensor<half> normAlignedGm;
        normAlignedGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(cacheBase + alignedOffset),
                                      16);
        auto normTmp = normPadTmpBuf_.Get<half>();  // at least 16 half
        DataCopy(normTmp, normAlignedGm, 16);
        PipeBarrier<PIPE_V>();
        CopyLocalSlice(normBuf, normTmp[elemShift], rows);
    }
    TqSyncMte2ToV();
}
```

**收益**：相比 8-bit 逐行从 packed 行尾部标量读取 norm，4-bit 一次 DataCopy 加载整个 block 的 norm，减少 MTE2 事务数。满 tile `rows=32` 时 norm 为 64B，可以直接 `DataCopy`；last tile 小于 16 行时不足 32B，首版按 32B 对齐读取到临时 buffer 后只消费有效 `rows`。

---

## 7. 计算流程

### 7.1 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Kernel Entry (MIX: AIC + 2 AIV)                                │
│                                                                  │
│  AIC: 执行 Decode rotate / QK / PV Cube Matmul                   │
│                                                                  │
│  AIV0: 主控 + 当前 tile 后处理                                   │
│    1. Init: 加载 tiling, 设置 GM buffer, 分配 UB buffer         │
│    2. LoadCodebook: codebook (16 项) → UB                       │
│    3. 驱动 Cube 调用，执行 score 缩放、softmax、outAcc 更新      │
│                                                                  │
│  AIV1: 下一 tile 数据准备                                        │
│    1. 预取 block_table / packed idx / norm                      │
│    2. 执行 4-bit unpack 与 codebook gather                       │
│    3. 与 AIV0 通过 ping-pong buffer 和 flag 交接                 │
└─────────────────────────────────────────────────────────────────┘
```

首版按 `KERNEL_TYPE_MIX_AIC_1_2` 设计，目标是让 1 个 Cube 与 2 个 Vector 都参与工作：

| 单元 | 主要职责 | 关键约束 |
|---|---|---|
| AIC | Decode rotate、QK、PV 三类 Matmul | 不参与标量调度；通过 KFC/Matmul 对象消费片上 A/B 并产出片上 C |
| AIV0 | 任务调度、Q tile 加载、score 后处理、online softmax、outAcc 更新、最终写回 | 负责维护 `mState/sState/outAcc`，避免跨 AIV 共享复杂状态 |
| AIV1 | 下一 K/V tile 的 block_table 预取、idx/norm 加载、4-bit unpack、codebook gather | 尽量覆盖当前 tile 的 QK/softmax/PV 时间；只写 ping-pong buffer 中未被 AIV0 消费的一半 |

同步粒度按 KV tile，而不是按行。AIV1 准备好 `packedIdx/norm/yHat` 后置 ready flag；AIV0 等 ready 后触发 decode rotate，并在该 buffer 完成 QK/PV 消费后置 free flag。该方案必须先 probe AIV0/AIV1 的 local tensor ownership、queue 内存可见性和 KFC 输入位置是否允许跨 AIV 交接。若 probe 不通过，退化方案是 AIV0 独立完成 decode/QK/PV，AIV1 只做 block_table 与 packed idx/norm 预取；不把中间 tensor 写 GM。

### 7.2 单任务计算流程

对每个 `(tokenIdx, kvHead, qHeadTile)` 任务（SplitBNS 下还有 `segIdx`）：

```
ComputeAttention(tokenIdx, kvHead, qHeadTile, segIdx, numSegs, writePartial):

  1. 查找序列: seqIdx = FindSeqForToken(tokenIdx, actualSeqLenQ, batchSize)
  2. 计算本序列 Q 范围:
     qStart = (seqIdx == 0) ? 0 : actualSeqLenQ[seqIdx - 1]
     qEnd = actualSeqLenQ[seqIdx]
     qLenInSeq = qEnd - qStart
     qLocalIdx = tokenIdx - qStart
  3. 读取 KV 长度: kvLen = actualSeqLenKv[seqIdx]  // post-write length
  4. 计算 causal 上界:
     qAbsPos = kvLen - qLenInSeq + qLocalIdx
     causalKvEnd = min(kvLen, qAbsPos + 1)
  5. 确定扫描范围: [kvStart, kvEnd)
     - SplitBN: kvStart=0, kvEnd=causalKvEnd
     - SplitBNS: kvStart=segIdx×kvSegmentLen, kvEnd=min((segIdx+1)×kvSegmentLen, causalKvEnd)
  6. 加载 Q: LoadQTile(tokenIdx, kvHead, qHeadTile) → qTile[qTileHeads, 128] fp16
  7. 初始化 online softmax: m=-inf, s=0, out=0
  8. KV tile 循环:
     for pos = kvStart; pos < kvEnd; pos += kvTileRows:
       a. 预取 block_table → UB
       b. 根据 pos 计算 blockId 与 block 内 rowStart；tile 不跨 block
       c. 批量加载 K norm 区 → normKBuf
       d. 加载 K 索引区: LoadPackedIdxRows(keyCache) → packedIdx[M, 64]
       e. 反量化 K: DecodeRows4bit(packedIdx, codebook, rotation) → zKTile[M, 128] fp16
       f. Cube QK: qTile[qTileHeads,128] × zKTile[M,128]^T → score[qTileHeads,M]，再乘 normK 和 scale
       g. 批量加载 V norm 区 → normVBuf
       h. 加载 V 索引区: LoadPackedIdxRows(valueCache) → packedIdx[M, 64]
       i. 反量化 V: DecodeRows4bit(packedIdx, codebook, rotation) → zVTile[M, 128] fp16
       j. Online softmax: score → probNorm[qTileHeads,M] fp16
       k. Cube PV: probNorm[qTileHeads,M] × zVTile[M,128] → pvOut[qTileHeads,128]
  9. 输出:
     - SplitBN: WriteFinalOutput → out[tokenIdx, qHeadTile, :] fp16
     - SplitBNS: WritePartial → workspace (partial max/sum/accum)
```

`qAbsPos` 必须使用序列内绝对位置，不能直接使用本次 chunk 内的 `qLocalIdx`。这里的 `kvLen` 必须是当前 step KV 写入后的 post-write 长度。DecodeOnly 场景下 `qLenInSeq=1`，因此 `qAbsPos=kvLen-1`，可关注完整历史 KV；ChunkedPrefill 场景下第 `qLocalIdx` 个 Q 只能关注到自身对应的绝对 KV 位置。

### 7.3 计算流程阶段划分

本设计将计算流程划分为以下 6 个阶段，每个阶段有明确的输入/输出接口：

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1: LoadQ (MTE2 + V)                                      │
│    输入: query GM[tokenIdx, qHeadTile:qHeadTile+qTileHeads, :]   │
│    输出: qTile[qTileHeads, 128] fp16，pad 到 16 对齐             │
│    操作: DataCopy → 片上 local tensor                            │
├──────────────────────────────────────────────────────────────────┤
│  Stage 2: LoadKV (MTE2 + S)                                     │
│    输入: keyCache/valueCache GM + blockTable GM                  │
│    输出: packedIdx[M, 64] uint8 (UB) + normK/normV[M] fp16 (UB) │
│    操作: 预取 blockTable → 连续加载 idx 区和 norm 区             │
├──────────────────────────────────────────────────────────────────┤
│  Stage 3: Decode (V + Cube)                                     │
│    输入: packedIdx[M, 64], codebook[16], R[128,128]             │
│    输出: zHat[M, 128] fp16 (UB)，未乘 norm                       │
│    操作: Unpack4bit → Gather → Cube KFC matmul                  │
├──────────────────────────────────────────────────────────────────┤
│  Stage 4: Cube QK                                               │
│    输入: qTile[qTileHeads,128], zK[M,128], normK[M]             │
│    输出: score[qTileHeads,M] fp32，保留在片上                    │
│    操作: qTile × zK^T via Cube → V 后处理乘 normK 和 scale       │
├──────────────────────────────────────────────────────────────────┤
│  Stage 5: OnlineSoftmax (V)                                     │
│    输入: score[qTileHeads,M], mState/sState                     │
│    输出: probNorm[qTileHeads,M] fp16，更新 mState/sState        │
│    操作: ReduceMax + Exp + sum，probNorm 片上供 PV 使用         │
├──────────────────────────────────────────────────────────────────┤
│  Stage 6: Cube PV + WriteOutput                                 │
│    输入: probNorm[qTileHeads,M], zV[M,128]                      │
│    输出: outAcc[qTileHeads,128] fp32，最终 out fp16              │
│    操作: probNorm × zV via Cube，online 累加后归一化写回         │
└──────────────────────────────────────────────────────────────────┘
```

### 7.4 流水线设计

本设计采用 **K/V 双缓冲 + MTE2/V 重叠** 的流水线：

```
时间轴 →

Tile 0:
  [MTE2: Load K0 idx+normK] [V/C: Decode K0] [Cube: QK0] [V: Softmax0]
                              [MTE2: Load V0 idx+normV] [V/C: Decode V0] [Cube: PV0]
Tile 1:
                              [MTE2: Load K1 idx+normK] [V/C: Decode K1] [Cube: QK1] ...
Tile 2:
                                                      [MTE2: Load K2 idx+norm] [V: Decode K2] ...

重叠说明:
  - Tile N 的 Decode/QK/Softmax/PV 与 Tile N+1 的 Load K 尽量重叠
  - Decode/QK/PV 中间 tensor 禁止写入 GM scratch；Matmul A/B/C 必须使用可被 CANN 接受的片上位置
  - 若某个 QK/PV Matmul 配置不支持片上 A/B/C，优先调整 tensor position、tiling 或片上转置；不能把 GM scratch 当作性能路径
  - 需要双缓冲: packedIdxBuf_[0,1] 和 zHatBuf_[0,1]
  - normKBuf/normVBuf 跟随 tile 双缓冲，K/V 分开；首版每 tile 加载 rows 个 norm
  - 使用 Que 队列管理缓冲区生命周期
```

**双缓冲 UB 分配**：

```
packedIdxBuf_[0]:  tile 0 的 packed K/V 索引行 (M × 64 字节)
packedIdxBuf_[1]:  tile 1 的 packed K/V 索引行 (tile 0 计算时预取)
zHatBuf_[0]:       tile 0 的 decoded K/V（未乘 norm），片上供 QK/PV Cube 使用
zHatBuf_[1]:       tile 1 的 decoded K/V（未乘 norm），片上供 QK/PV Cube 使用
normKBuf_[2]:      K norm，双缓冲 (M × 2 字节)
normVBuf_[2]:      V norm，双缓冲 (M × 2 字节)
qTileBuf_:         [qTileHeads, 128] fp16，QK A
scoreBuf_:         [qTileHeads, M] fp32，QK C 和 softmax 输入
probBuf_:          [qTileHeads, M] fp16，PV A
pvOutBuf_:         [qTileHeads, 128] fp32，PV C，随后累加到 outAcc
```

### 7.5 CombineFlashDecode 子流程

```
仅 SplitBNS 模式，由 core 0 在所有核完成后执行:

  for each (tokenIdx, headIdx):
    1. 收集所有 segment 的 partial 结果:
       partialM[p], partialS[p], partialOut[p, :]  (p = 0..kvSplitPart-1)
    2. 求全局 max: globalM = max(partialM[:])
    3. 加权合并:
       for p in [0, kvSplitPart):
         w[p] = exp(partialM[p] - globalM) × partialS[p]
         globalS += w[p]
         globalOut[:] += w[p] × partialOut[p, :]
    4. 归一化: out = globalOut / globalS
```

首版选择 core 0 单核 combine 的原因是 SplitBNS 只在 `BNQ` 很小、KV 很长时触发：主耗时来自每个 segment 的 Decode/QK/PV 扫描，combine 只处理 `numTokens × numHeads × kvSplitPart × 128` 个 fp32 accum 和少量 max/sum 标量。与把完整 KV 从 `kvLen` 串行扫描拆成 `kvSplitPart` 段并行相比，单核 combine 的额外 HBM 读写和向量归一化开销通常更小。

收益边界：如果后续放宽 SplitBNS 触发条件，让 `numTokens × numHeads` 较大，core 0 combine 会变成瓶颈。届时应把 combine 改成按 `(tokenIdx, headIdx)` 维度连续 range 分配到多个 core，每个 core 独立合并一段输出 head，不改变 partial workspace 布局。

---

## 8. 8-bit 性能瓶颈

### 8.1 QK/PV 纯标量实现（最大瓶颈，~55% 耗时）

`attention_device.h` 中 `VectorQk` 和 `VectorPv` 使用三层嵌套标量循环：

```cpp
for (g = 0; g < G; ++g)
    for (m = 0; m < M; ++m)
        for (d = 0; d < 128; ++d)  // 128 次标量 GetValue + 乘加
            dot += qGroup.GetValue(g*128+d) * (float)kTile.GetValue(m*128+d);
```

`GetValue` 是标量操作，触发 V→S 同步。G×M×D = 8×32×128 = 32768 次标量操作，而向量单元一次可处理 128 个 fp32。**性能差距约 100-200×**。

### 8.2 Online Softmax 标量实现（~15% 耗时）

`OnlineSoftmaxUpdateTile` 中每次 `ExpScalar` 调用都是标量 Exp + V→S 同步，且 outAcc 累加也是 128 次标量乘加。

### 8.3 LoadQGroup 标量 fp16→fp32（~5% 耗时）

`LoadQGroup` 中用 `GetValue/SetValue` 逐元素转换，应使用 `Cast` 向量化。

### 8.4 WriteOutput 标量输出（~5% 耗时）

`WriteFinalOutput` 和 `WritePartial` 用 `SetValue` 逐元素写 GM。

### 8.5 LoadPackedTileRows 逐行标量寻址（~10% 耗时）

每行 1 次标量 `blockTableGm_.GetValue()` + 1 次 `DataCopy`，32 行 = 32 次标量 GM 读取。

### 8.6 CombineFlashDecode 标量合并（FlashDecode 下 ~10% 耗时）

所有 partial 结果合并都是标量循环。

### 8.7 K/V 串行加载

K 和 V 的 packed 行串行加载和解码，无法重叠 MTE2 和计算。

### 8.8 4-bit 反量化额外开销

4-bit 格式引入了索引拆包步骤（64 字节 → 128 个 4-bit 索引），这在 8-bit 格式中不存在。标量拆包 64 字节约 64 次 GetValue + 64 次位操作。相比 8-bit 的 128 次 GetValue（用于 Cast），4-bit 拆包开销更小。

---

## 9. 4-bit 设计方案

### 9.1 关键设计

#### 9.1.1 Cube QK/PV 主路径

首版按最优性能实现 Cube QK/PV，不再把 Cube 路径限制为 `G>=8`。每个任务处理一个 Q head tile：

```text
qTileHeads = 16
M = kvTileRows = 32

QK: [16, 128] fp16 × [32, 128]^T fp16 → [16, 32] fp32
PV: [16, 32] fp16 × [32, 128] fp16 → [16, 128] fp32
```

当实际 `G < 16` 时，Q tile 的无效行补 0，最终只写回有效 Q heads。这样 Qwen3-0.6B (`G=1`) 和 Qwen3-8B (`G=4`) 也走同一条 Cube 路径，牺牲少量 padding 计算来消除 QK/PV 标量热点。

关键实现约束：

1. Decode rotate 的输出 `zK/zV` 必须直接落到片上 local tensor，并作为 QK/PV Cube 输入
2. QK 的 K tile 需要转置语义，优先使用片上 `SetTensorB(zKLocal, true)`；若 Matmul 不支持该布局，则在片上显式转置到 `zKTLocal`
3. QK 输出 `score` 保留在片上，随后做 `score *= normK * scale` 和 online softmax
4. Softmax 输出 `probNorm = exp(score - mNew) * normV` 保留在片上，并作为 PV Cube 的 A 矩阵输入
5. PV 输出 fp32 tile 保留在片上，与 online softmax 的历史 `outAcc` 按 `alpha` 合并
6. 首版性能路径禁止为 `qTile/zK/zV/score/probNorm/pvOut` 分配 GM scratch；如果 CANN Matmul 配置无法满足片上链路，需要先做 probe 并重审方案

```cpp
// QK: qTileLocal[16,128] @ zKLocal[32,128]^T -> scoreLocal[16,32]
qkMm.SetOrgShape(16, 32, 128);
qkMm.SetSingleShape(qRowsPad, tileRows, 128);
qkMm.SetTensorA(qTileLocal, false);
qkMm.SetTensorB(zKLocal, true);
qkMm.IterateAll(scoreLocal);
qkMm.End();

// V 后处理: score *= normK * scale, online softmax -> probNormLocal[16,32]
PostProcessScoreAndSoftmax(scoreLocal, normKBuf, normVBuf, probNormLocal,
                           mState, sState, qRows, tileRows, scale);

// PV: probNormLocal[16,32] @ zVLocal[32,128] -> pvOutLocal[16,128]
pvMm.SetOrgShape(16, 128, 32);
pvMm.SetSingleShape(qRowsPad, 128, tileRows);
pvMm.SetTensorA(probNormLocal, false);
pvMm.SetTensorB(zVLocal, false);
pvMm.IterateAll(pvOutLocal);
pvMm.End();
```

#### 9.1.2 Online Softmax + Cube PV 合并

QK score 经过 `normK * scale` 后进入 online softmax。性能路径必须按行向量化处理 `[qTileHeads, M]`，不能在 `G×M` 上写 `GetValue/SetValue + expf` 标量循环。标量版本只允许作为 host/reference golden。

片上处理流程：

```text
输入:
  scoreLocal[qTileHeads, M] fp32
  normK[M] fp16
  normV[M] fp16
  mState[qTileHeads] fp32
  sState[qTileHeads] fp32
  outAcc[qTileHeads, 128] fp32

1. ScoreScale:
   对每个 q row 向量化执行:
     score[row, :] *= Cast(normK[:]) * scale

2. RowMax:
   对每个 q row 做 WholeReduceMax，得到 tileMax[row]
   mNew[row] = max(mState[row], tileMax[row])
   alpha[row] = exp(mState[row] - mNew[row])

3. ExpAndSum:
   expBuf[row, :] = exp(score[row, :] - mNew[row])
   tileSum[row] = WholeReduceSum(expBuf[row, :])
   sNew[row] = sState[row] * alpha[row] + tileSum[row]

4. BuildProbNorm:
   probNorm[row, :] = CastToFp16(expBuf[row, :] * Cast(normV[:]))
   probNorm 保留在片上，作为 PV Cube 的 A

5. CubePV:
   pvOut[row, :] = probNorm[row, :] @ zV[:, :]

6. Accumulate:
   outAcc[row, :] = outAcc[row, :] * alpha[row] + pvOut[row, :]
   mState[row] = mNew[row]
   sState[row] = sNew[row]
```

PV Cube 产出 `pvOut = Σ exp(score - mNew) * normV * zV`。最终写回时执行 `out = outAcc / sState`。`alpha/sState` 是 fp32 状态，`probNorm` 首版用 fp16 喂 PV Cube；如果精度验证不达标，再评估 fp32 A 或分段补偿，但不改变片上驻留原则。

Vector QK/PV 代码只保留为 bringup fallback，不作为性能目标。

#### 9.1.3 LoadQTile

```cpp
void LoadQTile(uint32_t tokenIdx, uint32_t kvHead, uint32_t qHeadTile,
               LocalTensor<half> qLocal)
{
    const uint32_t qBaseHead = kvHead * gqaGroup_;
    const uint32_t gStart = qHeadTile * qTileHeads_;
    const uint32_t validRows = Min(qTileHeads_, gqaGroup_ - gStart);
    for (uint32_t r = 0; r < qTileHeads_; ++r) {
        if (r < validRows) {
            const uint64_t off = (tokenIdx * numHeads_ + qBaseHead + gStart + r) * TQ_HEAD;
            DataCopy(qLocal[r * TQ_HEAD], queryGm_[off], TQ_HEAD);
        } else {
            Duplicate(qLocal[r * TQ_HEAD], static_cast<half>(0), TQ_HEAD);
        }
    }
    TqSyncMte2ToV();
}
```

#### 9.1.4 WriteOutput

```cpp
void WriteFinalOutput(uint32_t tokenIdx, uint32_t kvHead, uint32_t qHeadTile,
                      LocalTensor<float> sState, LocalTensor<float> outAcc)
{
    auto outFp16 = outFp16Buf_.Get<half>();
    const uint32_t qBaseHead = kvHead * gqaGroup_;
    const uint32_t gStart = qHeadTile * qTileHeads_;
    const uint32_t validRows = Min(qTileHeads_, gqaGroup_ - gStart);
    for (uint32_t r = 0; r < validRows; ++r) {
        float invS = 1.f / sState.GetValue(r);
        TqSyncSToV();
        Muls(outAcc[r * TQ_HEAD], outAcc[r * TQ_HEAD], invS, TQ_HEAD);
        PipeBarrier<PIPE_V>();
        Cast(outFp16, outAcc[r * TQ_HEAD], CAST_ROUND, TQ_HEAD);
        PipeBarrier<PIPE_V>();
        const uint64_t off = (tokenIdx * numHeads_ + qBaseHead + gStart + r) * TQ_HEAD;
        DataCopy(outGm_[off], outFp16, TQ_HEAD);
    }
    TqSyncVToMte3();
}
```

#### 9.1.5 批量 block_table 预取

```cpp
void PrefetchBlockTable(uint32_t seqIdx)
{
    auto btLocal = blockTableBuf_.Get<int32_t>();
    const uint64_t btOff = seqIdx * maxBlocksPerSeq_;
    DataCopy(btLocal, blockTableGm_[btOff], maxBlocksPerSeq_);
    TqSyncMte2ToS();
}
```

---

## 10. UB 内存预算

Ascend 910B3 每个 AIV sub-block 有 **192 KB UB**。

### 10.1 Attention 算子 UB 用量

| 缓冲区 | 大小 | 说明 | 与 8-bit 差异 |
|---|---|---|---|
| codebookBuf_ | 64 B | 2 × 16 × 2B | 8-bit: 1024B，**大幅减小** |
| packedIdxBuf_[2] | 4,096 B | 双缓冲 2×(32×64) | 8-bit: 8448B (2×32×130+2)，**减小** |
| normKBuf_[2] | 128 B | 双缓冲 2×32×2B | 新增，K norm 延迟乘到 score |
| normVBuf_[2] | 128 B | 双缓冲 2×32×2B | 新增，V norm 合并到 PV 权重 |
| normPadTmpBuf_ | 32 B | 16×2B，last tile norm 对齐读取临时 | 新增 |
| zHatBuf_[2] | 16,384 B | 双缓冲 2×32×128×2B，未乘 norm | 同 8-bit |
| rotateWorkBuf_ | 32,768 B | 128×128×2B | 同 8-bit |
| expBuf_ | 16 B | 4×4B | 同 8-bit |
| qTileBuf_ | 4,096 B | 16×128×2B | Cube QK/PV Q tile |
| probBuf_ | 1,024 B | 16×32×2B | softmax 后 probNorm，片上供 PV |
| scoreBuf_ | 2,048 B | 16×32×4B | QK score，片上做 norm/softmax |
| softmaxTmpBuf_ | 512 B | 128×4B | 新增 |
| reduceTmpBuf_ | 512 B | 128×4B | 新增 |
| normScalarBuf_ | 4 B | 1×4B | 新增 |
| mStateBuf_ | 64 B | 16×4B | qTileHeads=16 |
| sStateBuf_ | 64 B | 16×4B | qTileHeads=16 |
| alphaStateBuf_ | 64 B | 16×4B | PV 后 online 合并 |
| outAccBuf_ | 8,192 B | 16×128×4B | qTileHeads=16 |
| outFp16Buf_ | 256 B | 128×2B | 新增 |
| blockTableBuf_ | 4,096 B | maxBlocksPerSeq×4B (最大 1024) | 新增 |
| idxS32Buf_ | 16,384 B | 32×128×4B (4-bit unpack 临时) | 8-bit: 同 (Gather 索引) |
| yHatBuf_ | 8,192 B | 32×128×2B，codebook lookup 后、旋转前 | 同 8-bit |
| pvOutBuf_ | 8,192 B | 16×128×4B，PV Cube C | 新增 |
| combineOutBuf_ | 512 B | 128×4B (SplitBNS only) | 同 8-bit |
| **合计** | **~109 KB + TQue/对齐开销** | Cube QK/PV 主路径 | 低于 Vector 方案，无 GM 中间结果 |

4-bit 格式相比 8-bit 格式，逻辑 packed 行从 130B 缩减到 66B（索引区 64B + norm 2B），使 packedIdxBuf 双缓冲占用从 8448B 降至 4096B。Cube QK/PV 主路径不再需要 `kFp32/vFp32` 大缓冲，UB 压力低于 Vector 方案；同时要求 Decode/QK/PV 中间结果都在片上流转，避免额外 HBM 流量。

4-bit Cube 主路径合计约 109KB，加上 TQue 元数据、Matmul 本地 workspace 和对齐后仍应低于 192KB UB。实际实现需以 CANN 编译后的 UB 静态分析为准。

### 10.2 片上驻留与实现前 probe

首版禁止为 Decode/QK/PV 中间结果分配 GM scratch。GM 只允许承载以下数据：

1. 输入输出：`query`、`key_cache`、`value_cache`、`block_table`、最终 `out`
2. 常量表：`codebook`、`rotation`
3. SplitBNS 的 FlashDecode partial 结果
4. `coreTaskStart/coreTaskEnd` 等调度元数据
5. CANN Matmul/KFC 的 system workspace；该 workspace 不能显式保存 `qTile/zK/zV/score/probNorm/pvOut`

实现前必须先完成 Matmul probe，验收条件如下：

| 路径 | 目标 Matmul 位置 | 验收条件 |
|---|---|---|
| Decode rotate | A=VECOUT, B=GM, C=VECIN/VECOUT | `IterateAll(zHatLocal)` 后直接供 QK/PV 使用，无 `cubeC` GM 回读 |
| QK | A=片上 `qTile`, B=片上 `zK` 或 `zKT`, C=片上 `score` | 支持 `zK` 转置语义；不支持时在片上转置 |
| PV | A=片上 `probNorm`, B=片上 `zV`, C=片上 `pvOut` | `pvOut` 直接累加到片上 `outAcc` |

如果 QK 必须使用片上 `zKT`，需要额外 `32×128×2B = 8KB` UB；首选复用空闲的 `zHat` 双缓冲槽，无法复用时再降低 K/V 预取重叠，不允许退回 GM 转置。

若 probe 证明某个 Cube 配置暂不可行，bringup 可以临时走 Vector fallback 验证正确性，但不能把 GM scratch 版本定义为性能主路径。

---

## 11. 4-bit vs 8-bit 对比

| 维度 | 8-bit | 4-bit |
|---|---|---|
| **Codebook 大小** | 256 项 | 16 项 |
| **每行 packed 大小** | 130 字节 (128 索引 + 2 norm) | 66 字节 (64 索引 + 2 norm) |
| **实际存储压缩比** (相对 fp16 256B/行) | 256B → 130B，约 1.97× | 256B → 66B，约 3.88× |
| **KV cache 带宽** | 130B/行 | 66B/行 (减少 49%) |
| **量化精度** | MSE 最优 256 级 | MSE 最优 16 级 (精度损失更大) |
| **Encode 复杂度** | WholeReduceMin (256 项) | 标量遍历 (16 项) |
| **Decode 额外步骤** | 无 | 4-bit 索引拆包 |
| **Norm 读取** | 逐行标量读取（交织布局，每行尾部 2 字节） | 批量 DataCopy（分离布局，整个 block 一次加载） |
| **Norm 存储位置** | 紧跟每行索引区后（交织） | 每个 `(block, head)` slab 尾部连续存储（分离） |
| **索引区对齐** | 128B 对齐 | 64B 对齐（32B 对齐，利于 DataCopy） |
| **Codebook UB 占用** | 512B (256×2B) | 32B (16×2B)，大幅减小 |
| **packedIdx 双缓冲** | 8448B (2×32×130+2) | 4096B (2×32×64)，减小 51% |
| **Attention UB 总用量** | ~161 KB (Vector 8-bit 方案) | ~109 KB + 对齐开销；若启用片上 `zKT` 转置约 ~117 KB |
| **QK/PV** | 现有实现多为 Vector/标量热点 | 首版目标 Cube QK/PV |
| **适用场景** | 高精度量化场景 | 极致存储/带宽优化场景，可接受精度损失 |

### 11.1 精度与性能权衡

4-bit 量化相比 8-bit 量化，主要权衡如下：

**精度影响**：
- 4-bit codebook 仅 16 个聚类中心，量化误差大于 8-bit 的 256 级
- 对于 attention 计算中的 K/V cache，4-bit 量化在长序列场景下误差累积更明显
- 建议在精度敏感场景（如代码生成、数学推理）评估 4-bit 量化对模型输出的影响

**性能收益**：
- KV cache 带宽减少 49%（130B → 66B/行），在长序列 decode 场景下显著降低 MTE2 压力
- QK/PV 走 Cube，消除 `G×M×128` 标量乘加热点
- Norm 分离存储使批量加载成为可能，减少 MTE2 事务数

**推荐策略**：
- 短序列 / 高精度需求：使用 8-bit 量化
- 长序列 / 高吞吐需求：使用 4-bit 量化
- 外层调度根据 `bits_key/bits_value` 选择 8-bit 或 4-bit 算子；本算子内部只接受 `bits_key = bits_value = 4`
