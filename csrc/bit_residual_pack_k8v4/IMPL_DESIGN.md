# BitResidualPackK8v4 实现设计

## 1. 目标

本设计用于把当前 `BitResidualPackK8v4` 从旧的 4-bit TurboQuant
码本量化路径，改成 Key 和 Value 使用不同编码方式的新 KV cache pack
格式。

成功标准：

- Key 和 Value 继续共用 `normalize + rotate` 前处理流程。
- Key 使用 `1-bit sign + 7-bit residual`，每 2 行合并到一个
  `uint16[128]` group。
- Value 使用 4-bit uniform quantization，每 4 行合并到一个
  `uint16[128]` group。
- 移除 `codebook` 输入和所有旧码本量化 helper。
- cache GM 布局、UB 中间布局、group merge、host tiling 和 op
  接口保持一致。

本文只描述实现方案，不引入额外功能。

## 2. 固定约束

- `TQ_PACK_D = 128`。
- 输入 dtype 仅支持 `fp16` 和 `bf16`，Key、Value、`rotation_t`
  必须同 dtype。
- `block_size` 必须是 4 的倍数。Value 需要 4 行 group，Key 需要
  2 行 group，4 的倍数同时满足两者。
- cache 输出仍按 `uint8_t*` 暴露，但内部布局按 `uint16` word 解释。
- 每个 token/head 对应一行 128 维向量。
- 当前调度仍沿用 sequence contiguous segment 逻辑，slot 在同一请求内按
  `first_slot + token_offset` 推导。

## 3. 算法

### 3.1 Key 和 Value 共用前处理

对每一行 `x[128]`：

```text
norm = ||x||                         # fp32 accumulate
n = x / (norm + eps)
y = n @ rotation_t                   # [M,128] @ [128,128]
```

实现上继续复用现有流程：

- `NormalizeBatch(m)` 负责 L2 norm 和归一化。
- `RotateBatchMatmul(m, 0, TQ_PACK_D)` 负责 KFC matmul。
- Key 和 Value 分别读取自己的输入，因此分别执行上述流程。

### 3.2 Key 编码

Key 每行输出 128 个 8-bit code。每个维度：

```text
sign = y > -0.0f
sign_val = sign ? +1/sqrt(128) : -1/sqrt(128)
err = y - sign_val
base = min(err)
maxv = max(err)
step = (maxv - base) / 127
q7 = round((err - base) / step)
q7 = clip(q7, 0, 127)
code = (q7 << 1) | sign
```

边界处理：

- 当 `maxv == base` 或 range 非正时，`step` 取安全非零值，推荐
  `step = 1.0f`，并使 `q7 = 0`。
- `q7` 必须是 clamp，不应依赖 bit mask 处理溢出。浮点舍入可能在
  极端情况下得到 128。
- `sign` 的 LSB 约定为：当前实现使用
  `CompareScalar(..., -0.0f, CMPMODE::GT)`，比较结果为 true 时写 1，
  否则写 0。这样与当前 AscendC 编译环境中已有比较写法保持一致。

Key 每 2 行 merge：

```text
uint16[d] = (code_row1[d] << 8) | code_row0[d]
```

低 byte 是 group row 0，高 byte 是 group row 1。

### 3.3 Value 编码

Value 每行输出 128 个 4-bit index。每个维度：

```text
vmin = min(y)
vmax = max(y)
vstep = (vmax - vmin) / 15
idx4 = round((y - vmin) / vstep)
idx4 = clip(idx4, 0, 15)
```

边界处理：

- 当 `vmax == vmin` 或 range 非正时，`vstep` 取安全非零值，推荐
  `vstep = 1.0f`，并使 `idx4 = 0`。
- Value cache 不保存 norm，只保存 `vmin` 和 `vstep`。

Value 每 4 行 merge：

```text
uint16[d] =
    (idx4_row3[d] << 12) |
    (idx4_row2[d] << 8)  |
    (idx4_row1[d] << 4)  |
     idx4_row0[d]
```

bits 0..3 是 row 0，bits 4..7 是 row 1，bits 8..11 是 row 2，
bits 12..15 是 row 3。

## 4. Cache GM 布局

本方案采用 group stride 作为 GM 中的间距。group 有效字节数小于
stride 时，尾部 padding 不参与语义。

### 4.1 Key cache

每个 block/head：

```text
block_size / 2 个 Key group
```

每个 Key group：

```text
byte[0..255]     = uint16[128]，2 行 packed code
byte[256..259]   = uint16[2] norm，顺序为 row0, row1
byte[260..267]   = float[2] base，顺序为 row0, row1
byte[268..275]   = float[2] step，顺序为 row0, row1
byte[276..287]   = padding
```

大小：

```cpp
static constexpr uint32_t TQ_KEY_GROUP_ROWS = 2;
static constexpr uint32_t TQ_KEY_GROUP_INDEX_BYTES = TQ_PACK_D * sizeof(uint16_t); // 256
static constexpr uint32_t TQ_KEY_GROUP_NORM_BYTES =
    TQ_KEY_GROUP_ROWS * sizeof(uint16_t);                                          // 4
static constexpr uint32_t TQ_KEY_GROUP_BASE_BYTES =
    TQ_KEY_GROUP_ROWS * sizeof(float);                                             // 8
static constexpr uint32_t TQ_KEY_GROUP_STEP_BYTES =
    TQ_KEY_GROUP_ROWS * sizeof(float);                                             // 8
static constexpr uint32_t TQ_KEY_GROUP_BYTES =
    TQ_KEY_GROUP_INDEX_BYTES + TQ_KEY_GROUP_NORM_BYTES +
    TQ_KEY_GROUP_BASE_BYTES + TQ_KEY_GROUP_STEP_BYTES;                             // 276
static constexpr uint32_t TQ_KEY_GROUP_STRIDE =
    (TQ_KEY_GROUP_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;            // 288
```

GM offset：

```cpp
key_group_base =
    ((uint64_t)block_idx * num_heads + head_idx) *
        (block_size / TQ_KEY_GROUP_ROWS) * TQ_KEY_GROUP_STRIDE +
    group_in_block * TQ_KEY_GROUP_STRIDE;
```

### 4.2 Value cache

每个 block/head：

```text
block_size / 4 个 Value group
```

每个 Value group：

```text
byte[0..255]     = uint16[128]，4 行 packed idx4
byte[256..271]   = float[4] vmin，顺序为 row0..row3
byte[272..287]   = float[4] vstep，顺序为 row0..row3
```

大小：

```cpp
static constexpr uint32_t TQ_VAL_GROUP_ROWS = 4;
static constexpr uint32_t TQ_VAL_GROUP_INDEX_BYTES = TQ_PACK_D * sizeof(uint16_t); // 256
static constexpr uint32_t TQ_VAL_GROUP_VMIN_BYTES =
    TQ_VAL_GROUP_ROWS * sizeof(float);                                             // 16
static constexpr uint32_t TQ_VAL_GROUP_VSTEP_BYTES =
    TQ_VAL_GROUP_ROWS * sizeof(float);                                             // 16
static constexpr uint32_t TQ_VAL_GROUP_BYTES =
    TQ_VAL_GROUP_INDEX_BYTES + TQ_VAL_GROUP_VMIN_BYTES +
    TQ_VAL_GROUP_VSTEP_BYTES;                                                      // 288
static constexpr uint32_t TQ_VAL_GROUP_STRIDE =
    (TQ_VAL_GROUP_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;            // 288
```

GM offset：

```cpp
value_group_base =
    ((uint64_t)block_idx * num_heads + head_idx) *
        (block_size / TQ_VAL_GROUP_ROWS) * TQ_VAL_GROUP_STRIDE +
    group_in_block * TQ_VAL_GROUP_STRIDE;
```

### 4.3 cache 总大小

```text
key_cache_bytes =
    num_blocks * num_heads *
    (block_size / TQ_KEY_GROUP_ROWS) * TQ_KEY_GROUP_STRIDE

value_cache_bytes =
    num_blocks * num_heads *
    (block_size / TQ_VAL_GROUP_ROWS) * TQ_VAL_GROUP_STRIDE
```

例如 `block_size = 16`：

- Key: `num_blocks * num_heads * 8 * 288`
- Value: `num_blocks * num_heads * 4 * 288`

注意：attention/dequant 侧 cache reader 尚未实现。后续设计 attention
计算时，必须同步采用这里的 group stride、aux dtype 和 offset。

## 5. UB 中间 encoded row 布局

中间 encoded row 用于 `Encode*Batch` 到 group merge 之间。最终决定：
Key 和 Value 都使用 `uint16[128]` 存储每维 code/index。这是性能优先的
选择：AscendC 当前用于 merge/decode 的 `And` 和 `ShiftLeft` 走 16-bit
vector lane；Key 两行 8-bit code 正好组成一个 `uint16`，Value 四行
4-bit idx 也正好组成一个 `uint16`，不会浪费最终 packed 表示。

不采用 `uint8[128]` 作为 Key encoded row 的原因是：merge 前仍要 widen
到 `uint16[128]` 才能做 `ShiftLeft` 和 `Or`，会额外增加一次向量转换，
且相邻 byte 不能直接 reinterpret 成按维度排列的 `uint16`。

### 5.1 Key encoded row

```text
byte[0..255]   = uint16[128] code，取值 0..255
byte[256..257] = uint16 norm，fp16 bits
byte[258..259] = padding，保证后续 float 4B 对齐
byte[260..263] = float base
byte[264..267] = float step
byte[268..287] = padding
```

常量：

```cpp
static constexpr uint32_t TQ_KEY_ENCODED_ROW_BYTES =
    TQ_PACK_D * sizeof(uint16_t) + 2 * sizeof(uint16_t) + 2 * sizeof(float); // 268
static constexpr uint32_t TQ_KEY_ENCODED_NORM_WORD_OFFSET = TQ_PACK_D;      // 128
static constexpr uint32_t TQ_KEY_ENCODED_BASE_BYTE_OFFSET =
    TQ_PACK_D * sizeof(uint16_t) + 2 * sizeof(uint16_t);                    // 260
static constexpr uint32_t TQ_KEY_ENCODED_STEP_BYTE_OFFSET =
    TQ_KEY_ENCODED_BASE_BYTE_OFFSET + sizeof(float);                        // 264
static constexpr uint32_t TQ_KEY_ENCODED_ROW_STRIDE_BYTES =
    (TQ_KEY_ENCODED_ROW_BYTES + TQ_UB_ALIGN - 1) /
    TQ_UB_ALIGN * TQ_UB_ALIGN;                                             // 288
static constexpr uint32_t TQ_KEY_ENCODED_ROW_STRIDE_WORDS =
    TQ_KEY_ENCODED_ROW_STRIDE_BYTES / sizeof(uint16_t);                    // 144
```

### 5.2 Value encoded row

```text
byte[0..255]   = uint16[128] idx4，取值 0..15
byte[256..259] = float vmin
byte[260..263] = float vstep
byte[264..287] = padding
```

常量：

```cpp
static constexpr uint32_t TQ_VAL_ENCODED_ROW_BYTES =
    TQ_PACK_D * sizeof(uint16_t) + 2 * sizeof(float);              // 264
static constexpr uint32_t TQ_VAL_ENCODED_VMIN_BYTE_OFFSET =
    TQ_PACK_D * sizeof(uint16_t);                                  // 256
static constexpr uint32_t TQ_VAL_ENCODED_VSTEP_BYTE_OFFSET =
    TQ_VAL_ENCODED_VMIN_BYTE_OFFSET + sizeof(float);               // 260
static constexpr uint32_t TQ_VAL_ENCODED_ROW_STRIDE_BYTES =
    (TQ_VAL_ENCODED_ROW_BYTES + TQ_UB_ALIGN - 1) /
    TQ_UB_ALIGN * TQ_UB_ALIGN;                                     // 288
static constexpr uint32_t TQ_VAL_ENCODED_ROW_STRIDE_WORDS =
    TQ_VAL_ENCODED_ROW_STRIDE_BYTES / sizeof(uint16_t);            // 144
```

## 6. Kernel 结构

### 6.1 Compute dispatch

拆分 Key 和 Value 的 encode queue：

```cpp
TQue<VECOUT, TQ_QUEUE_DEPTH> keyEncodedQue_;
TQue<VECOUT, TQ_QUEUE_DEPTH> valEncodedQue_;
```

拆分 compute 入口：

```cpp
__aicore__ inline void ComputeKeyBatch(uint32_t m) {
    NormalizeBatch(m);
    RotateBatchMatmul(m, 0, TQ_PACK_D);
    EncodeKeyBatch(m);
}

__aicore__ inline void ComputeValueBatch(uint32_t m) {
    NormalizeBatch(m);
    RotateBatchMatmul(m, 0, TQ_PACK_D);
    EncodeValueBatch(m);
}
```

### 6.2 EncodeKeyBatch

每行流程：

1. `Cast` `y` 到 fp32。
2. `Compare` 得到 sign mask，并保留到最终 OR。
3. 根据 sign mask 选择 `+/-1/sqrt(128)` 得到 `sign_val`。
4. `err = y - sign_val`。
5. `ReduceMin` 和 `ReduceMax` 得到 `base` 和 `maxv`。
6. 计算安全 `step`。
7. `q7 = clip(round((err - base) / step), 0, 127)`。
8. `code = (q7 << 1) | sign`。
9. 将 `code` widening 到 `uint16[128]` 写入 Key encoded row。
10. 将 `norm` 以 fp16 bits 写入 aux，将 `base/step` 直接以 float 写入 aux。

需要的临时 buffer：

- `signMaskBuf_`: `uint8[128]`，保存 sign。
- `signValBuf_`: `float[128]`，保存 `+/-1/sqrt(128)`。
- `errBuf_`: `float[128]`，保存 residual，也可复用为量化输入。
- `q7Buf_`: `uint8[128]` 或 `int16/uint16[128]`，保存 clamp 后 q7。
- `codeBuf_`: `uint16[128]`，保存 widening 后 code。
- `reduceScalarBuf_`: 至少 3 个 float slot，保存 `base/maxv/step`。
- `reduceTmpBuf_`: `float[128]`，作为 reduce workspace。
- `stepScalarBuf_`: 1 个 float，用于保存 scalar step 并写入 float aux。

### 6.3 EncodeValueBatch

每行流程：

1. `Cast` `y` 到 fp32。
2. `ReduceMin` 和 `ReduceMax` 得到 `vmin` 和 `vmax`。
3. 计算安全 `vstep`。
4. `idx4 = clip(round((y - vmin) / vstep), 0, 15)`。
5. 将 `idx4` widening 到 `uint16[128]` 写入 Value encoded row。
6. 将 `vmin/vstep` 直接以 float 写入 aux。

Value 可以复用旧的 `argminIndexBuf_` 和 `argminIndexU16Buf_` 作为
`int32` cast 中间结果和 `uint16` widening 结果，但命名应在实现时尽量
调整成更贴近 uniform quantization 的含义。

### 6.4 Clip 实现

`turboquant_pack_kv_for_cache4bit` 的旧 4-bit 编码路径是：

```cpp
Cast(argminIndex, qFloat, CAST_RINT, TQ_PACK_D);
Cast(argminIndexU16, argminIndex, CAST_NONE, TQ_PACK_D);
Duplicate(argminMask, 0x000F, TQ_PACK_D);
And(argminIndexU16, argminIndexU16, argminMask, TQ_PACK_D);
```

这条路径能成立，是因为旧 `qFloat` 来源于 0..15 的 code 选择，本身已经
保证范围正确，`And 0x000F` 只是保留 nibble 位，不能作为通用 clamp。

新 Key/Value uniform quantization 中，`round((x - min) / step)` 可能因
浮点误差或 degenerate range 产生越界值。因此目标语义必须是 saturating
clip：

```text
Key:   q7   = min(max(q7,   0), 127)
Value: idx4 = min(max(idx4, 0), 15)
```

推荐实现顺序：

1. 在 fp32 上完成 `(x - base) / step` 或 `(x - vmin) / vstep`。
2. 使用 fp32 `Maxs` 把下界夹到 0。
3. 使用 fp32 `Mins` 把上界夹到 `127` 或 `15`。
4. 使用 `Cast(..., CAST_RINT)` 得到 int32。
5. 再 cast 到 `uint16/int16`，用于后续 `ShiftLeft`/`Or` merge。

示意：

```cpp
AscendC::Maxs(qFp32, qFp32, 0.0f, TQ_PACK_D);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Mins(qFp32, qFp32, static_cast<float>(upper), TQ_PACK_D);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Cast(qI32, qFp32, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Cast(qU16, qI32, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
```

当前实现采用上述 fp32 clamp 路径。这样不依赖 `int32 Maxs/Mins` 的
AscendC 支持情况，同时和数学上的 saturating clip 等价。不要用
`And 0x7F` 或 `And 0x000F` 代替 clip。

### 6.5 旧码本量化删除

需要删除：

- `TqFyScalar`
- `TqQuantThreshold`
- `TqCodeAsFloat`
- `FillQuantCodeVector`
- `FillQuantCodeVectors`
- `PrepareQuantCodeVectors`
- `FillQuantThresholdTable`
- `PrepareQuantThresholdTable`
- `ApplyQuantCode`
- `EncodeQuantCodesByCompare`
- `EncodeQuantCodesByReduceSum`
- `quantBufferMode_`
- `quantMaskBuf_`
- `quantCodeBuf_`
- `quantOnesBuf_`
- `codeIndexBuf_`

归一化使用的 fp32 工作区可以保留现有 `reduceOutBuf_`，也可以重命名为
`normWorkBuf_`。如果重命名会扩大 diff，则优先保留现有名字。

## 7. Group merge

### 7.1 Key merge

```cpp
ShiftLeft(shifted_code, encoded_code, group_row * 8, TQ_PACK_D);
Or(packed_u16, packed_u16, shifted_code, TQ_PACK_D);
```

partial update 时先清除目标 byte：

```cpp
clear_mask = ~(0xFFu << (group_row * 8));
```

aux 写入位置：

```text
norm: byte[256 + group_row * sizeof(uint16_t)]，uint16 fp16 bits
base: byte[260 + group_row * sizeof(float)]，float
step: byte[268 + group_row * sizeof(float)]，float
```

实现上可以把 `packedGroup` 同时 reinterpret 为 `uint16` 和 `float`：

```cpp
auto packedU16 = packedGroup.ReinterpretCast<uint16_t>();
auto packedBase = packedGroup[TQ_KEY_GROUP_BASE_OFFSET].ReinterpretCast<float>();
auto packedStep = packedGroup[TQ_KEY_GROUP_STEP_OFFSET].ReinterpretCast<float>();
```

其中 `code/norm` 用 `packedU16` 写，`base/step` 分别用按 byte offset
对齐后的 `float` view 写。不要把 `base/step` cast 成 fp16。

### 7.2 Value merge

```cpp
ShiftLeft(shifted_idx, encoded_idx, group_row * 4, TQ_PACK_D);
Or(packed_u16, packed_u16, shifted_idx, TQ_PACK_D);
```

partial update 时先清除目标 nibble：

```cpp
clear_mask = ~(0x0Fu << (group_row * 4));
```

aux 写入位置：

```text
vmin:  byte[256 + group_row * sizeof(float)]，float
vstep: byte[272 + group_row * sizeof(float)]，float
```

Value 不保存 norm，`vmin/vstep` 都直接保存 float。Value group 有效大小
正好是 288B，没有尾部 padding。

### 7.3 可复用逻辑

以下逻辑保留，但要按 Key/Value layout 参数化：

- `ClearPackedGroup`
- `InitFullGroupFromRow0`
- `FlushResolvedGroups`
- `CopyOutPhysicalGroupRowsFast`
- `CopyOutPhysicalGroupRowsKnown`
- `CopyOutPhysicalFullGroupRunKnown`
- async write pending 管理：`packedWritePending_` 和 `packedGroupSlot_`

建议用 layout trait 或模板参数表达差异：

```cpp
enum class PackKind { KEY, VALUE };
```

每种 kind 至少提供：

- `GROUP_ROWS`
- `GROUP_BYTES`
- `GROUP_STRIDE`
- `ENCODED_ROW_STRIDE_WORDS`
- `MakeCacheGroupBaseOffset`
- `MergeEncodedRowToGroup`
- encoded queue 选择
- cache GM 选择

## 8. 每核任务划分

任务划分参考 `csrc/turboquant_pack_kv_for_cache4bit` 的 key1 路径：

```text
Process()
  -> worker = GetBlockIdx()
  -> ProcessCachePairBySequenceContiguousSegments(worker)
```

当前算子仍采用 MIX 1C2V。一个 data-parallel group 包含 1 个 AIC 和
2 个 AIV，真正执行 pack 的是 AIV worker：

```text
active_workers = data_cores * TQ_AIV_SUB_BLOCKS
TQ_AIV_SUB_BLOCKS = 2
```

`worker >= active_workers` 时直接返回。AIC 侧只参与 KFC matmul，不执行
pack 主流程。

### 8.1 worker 线性 token 区间

每个 AIV worker 先按全局 token 维度拿一个连续区间：

```text
tokens_per_worker = ceil(token_count / active_workers)
raw_begin = worker * tokens_per_worker
raw_end = min(token_count, raw_begin + tokens_per_worker)
last_worker_range = raw_end == token_count || worker == active_workers - 1
```

这个区间只是初始范围，不直接作为最终写 cache 的边界。后续必须结合
request 范围和 cache group 边界修正。

### 8.2 request 交集和 slot 推导

每个 worker 遍历所有 request：

```text
seq_start = query_start_loc[req_idx]
seq_end = query_start_loc[req_idx + 1]
segment_start = max(raw_begin, seq_start)
segment_end = min(raw_end, seq_end)
```

空交集直接跳过。对非空交集，只读取该 request 首 token 的物理 slot：

```text
first_slot = slot_mapping[seq_start]
slot(token_idx) = first_slot + (token_idx - seq_start)
```

这是 key1 路径的重要前提：同一个 request 内的 slot 连续。实现时不在
每个 token 上重复读取 `slot_mapping`，以减少 GM 标量访问。

### 8.3 按 PackKind 独立修正 group 边界

Key 和 Value 的 group size 不同：

```text
Key:   GROUP_ROWS = 2
Value: GROUP_ROWS = 4
```

因此二者不能共享同一组 `groupBases/groupRows/preserveRows`。推荐按
`PackKind` 各自执行一遍 key1 风格的 sequence 划分：

```cpp
ProcessCacheBySequenceContiguousSegments<PackKind::KEY>(worker);
ProcessCacheBySequenceContiguousSegments<PackKind::VALUE>(worker);
```

每个 kind 使用自己的 `GROUP_ROWS` 修正边界：

```text
slot_at_begin = first_slot + (segment_start - seq_start)
back_rows = slot_at_begin % GROUP_ROWS
segment_start -= min(back_rows, segment_start - seq_start)

slot_at_end = first_slot + (segment_end - seq_start)
if (!last_worker_range && segment_end < seq_end) {
    segment_end -= slot_at_end % GROUP_ROWS
}
```

修正规则：

- 起点落在 group 中间时，当前 worker 向前回退到 group 起点，负责处理
  这个完整 group 的相关行。
- 终点落在 group 中间时，非最后 worker 回退到 group 起点，把该 group
  留给下一个 worker。
- 最后 worker 可以处理序列末尾 partial group。
- 修正后如果 `segment_start >= segment_end`，该 request 对当前 worker
  没有可处理任务。

这样保证同一个 PackKind 下，一个 cache group 不会被两个 worker 同时
读改写。

### 8.4 block 内 leading/full/tail 三段

修正后的 segment 按 cache block 切分。对每个 block range：

```text
slot = first_slot + (token_idx - seq_start)
block_idx = slot / block_size
block_offset = slot % block_size
rows_in_block = min(block_size - block_offset, segment_end - token_idx)
```

然后进入 key1 同款三段处理：

```text
ProcessSequenceBlockRange<PackKind>()
  1. leading partial group
  2. full group run
  3. tail partial group
```

leading partial group：

```text
leading_group_row = block_offset % GROUP_ROWS
if leading_group_row != 0:
    leading_rows = min(GROUP_ROWS - leading_group_row, remaining_rows)
    group_in_block = block_offset / GROUP_ROWS
    AppendPhysicalCacheGroupRows(..., first_group_row=leading_group_row,
                                 preserve_existing=true)
```

full group run：

```text
full_rows = floor(remaining_rows / GROUP_ROWS) * GROUP_ROWS
if full_rows > 0:
    FlushSequenceBatch()
    PackCachePhysicalFullGroupRun<PackKind>(..., full_rows, group_in_block)
```

tail partial group：

```text
if remaining_rows > 0:
    group_in_block = current_offset / GROUP_ROWS
    AppendPhysicalCacheGroupRows(..., first_group_row=0,
                                 preserve_existing=false)
```

`preserve_existing=true` 表示 group 中有当前 task 不覆盖的旧行，copyout 前
需要先从 GM 读旧 group，再清除目标 byte/nibble 并 merge 新行。
`preserve_existing=false` 表示可以从空 group 开始构造，不需要读旧 GM。

### 8.5 head 维度 batch 划分

`AppendPhysicalCacheGroupRows` 和 full-group task 都沿用 key1 的 head
分批策略，避免单批超过 `GetBatchCapacity()`：

```text
batch_capacity = vec_per_core == 0 ? TQ_MAX_BATCH_M : vec_per_core
max_heads_per_batch = batch_capacity / row_count
head_count = min(num_heads - head_start, max_heads_per_batch)
```

若 `max_heads_per_batch == 0`，则强制 `head_count = 1`，保证任务继续
推进。每个 batch 的逻辑行数：

```text
m = row_count * head_count
```

full-group run 中，`rows_this_batch` 必须按当前 kind 的 `GROUP_ROWS`
对齐：

```text
rows_this_batch =
    floor((batch_capacity / num_heads) / GROUP_ROWS) * GROUP_ROWS
```

如果结果小于一个 group，则退化为一次处理一个完整 group。

### 8.6 Key/Value 执行顺序

由于 Key 和 Value 的 group 边界不同，推荐顺序是：

```text
ProcessCacheBySequenceContiguousSegments<KEY>(worker)
WaitPendingPackedWrites()
ProcessCacheBySequenceContiguousSegments<VALUE>(worker)
WaitPendingPackedWrites()
```

每个 PackKind 内部独立维护：

- `vecIndices[TQ_MAX_BATCH_M]`
- `groupBases[TQ_MAX_BATCH_M]`
- `groupRows[TQ_MAX_BATCH_M]`
- `preserveRows[TQ_MAX_BATCH_M]`

Key 使用 `MakeKeyCacheGroupBaseOffset`，Value 使用
`MakeValueCacheGroupBaseOffset`。两者不要复用同一组 group base 计算结果。

## 9. Host tiling 和接口变更

### 9.1 移除 codebook 输入

`aclnnBitResidualPackK8v4GetWorkspaceSize` 移除：

```cpp
const aclTensor* codebook
```

OpDef 移除：

```cpp
this->Input("codebook")
```

kernel entry 移除：

```cpp
GM_ADDR codebook
```

新输入顺序：

```text
0 key
1 value
2 rotation_t
3 slot_mapping
4 query_start_loc
```

输出顺序保持：

```text
0 key_cache
1 value_cache
```

### 9.2 tiling 校验

tiling 侧需要同步调整 input desc index：

```cpp
keyDesc      = context->GetInputDesc(0)
valueDesc    = context->GetInputDesc(1)
rotationDesc = context->GetInputDesc(2)
```

dtype 校验改为：

```text
key/value/rotation_t 必须同为 fp16 或 bf16
```

`block_size % 4 == 0` 的校验保留。

workspace 暂不因 cache 布局变化调整，仍保留 KFC message queue 和 CANN
内部 workspace。

## 10. 常量清理

删除旧常量：

- `TQ_PACKED_INDEX_BYTES`
- `TQ_ROW_BYTES`
- `TQ_GROUP_ROWS`
- `TQ_GROUP_INDEX_BYTES`
- `TQ_GROUP_BYTES`
- `TQ_GROUP_STRIDE`
- `TQ_ENCODED_ROW_WORDS`
- `TQ_ENCODED_ROW_BYTES`
- `TQ_ENCODED_ROW_STRIDE_BYTES`
- `TQ_ENCODED_ROW_STRIDE_WORDS`
- `TQ_CODE_INDEX_BYTES`
- `TQ_ARGMIN_INDEX_BYTES`，如保留复用则重命名或重新定义为通用 cast buffer
- `TQ_ARGMIN_INDEX_U16_BYTES`，同上
- `TQ_COMPARE_MASK_BYTES`
- `TQ_REDUCE_SRC_REP_STRIDE`
- `TQ_QUANT_CODE_*`
- `TQ_QUANT_TABLE_ELEMS`
- `TQ_QUANT_ONES_BYTES`
- `TQ_REDUCE_SUM_MIN_BATCH_ROWS`
- `TQ_QUANT_BUFFER_*`
- `TQ_FY_LINEAR`
- `TQ_FY_CUBIC`

新增或保留的核心常量：

```cpp
static constexpr uint32_t TQ_PACK_D = 128;
static constexpr uint32_t TQ_UB_ALIGN = 32;
static constexpr uint32_t TQ_CUBE_M_ALIGN = 16;
static constexpr uint32_t TQ_MAX_BATCH_M = 64;
static constexpr uint32_t TQ_QUEUE_DEPTH = 2;
static constexpr uint32_t TQ_ROT_K = TQ_PACK_D;
static constexpr uint32_t TQ_ROT_N = TQ_PACK_D;
static constexpr uint32_t TQ_DTYPE_BYTES = sizeof(uint16_t);
static constexpr uint32_t TQ_NORM_STRIDE = TQ_UB_ALIGN / TQ_DTYPE_BYTES;
static constexpr float TQ_NORM_EPS_F = 1e-10f;
static constexpr float TQ_INV_SQRT_D_F = 0.08838834764831845f;
static constexpr float TQ_KEY_QUANT_LEVELS_F = 127.0f;
static constexpr float TQ_VAL_QUANT_LEVELS_F = 15.0f;
```

`TQ_INV_SQRT_D_F` 固定为 `1 / sqrt(128)`，避免在 AscendC 编译环境中
引入额外 constexpr math 兼容性问题。

Key/Value aux offset 建议显式定义，避免用 magic number：

```cpp
static constexpr uint32_t TQ_KEY_GROUP_NORM_OFFSET = TQ_KEY_GROUP_INDEX_BYTES;    // 256
static constexpr uint32_t TQ_KEY_GROUP_BASE_OFFSET =
    TQ_KEY_GROUP_NORM_OFFSET + TQ_KEY_GROUP_NORM_BYTES;                          // 260
static constexpr uint32_t TQ_KEY_GROUP_STEP_OFFSET =
    TQ_KEY_GROUP_BASE_OFFSET + TQ_KEY_GROUP_BASE_BYTES;                          // 268

static constexpr uint32_t TQ_VAL_GROUP_VMIN_OFFSET = TQ_VAL_GROUP_INDEX_BYTES;    // 256
static constexpr uint32_t TQ_VAL_GROUP_VSTEP_OFFSET =
    TQ_VAL_GROUP_VMIN_OFFSET + TQ_VAL_GROUP_VMIN_BYTES;                          // 272
```

## 11. 实施顺序

1. 常量和 layout：新增 Key/Value group、encoded row、cache size 常量。
   验证：`rg "TQ_GROUP|TQ_ROW_BYTES|TQ_ENCODED_ROW"` 只剩预期新符号。
2. 接口去 `codebook`：更新 aclnn header、OpDef、tiling input index、
   kernel entry 和 `Init` 签名。
   验证：`rg "codebook" csrc/bit_residual_pack_k8v4` 不再命中实现文件。
3. 删除旧量化 helper 和旧 buffer。
   验证：旧 `TqFy*`、`QuantCode`、`quantBufferMode_` 不再存在。
4. 实现 `EncodeKeyBatch`。
   验证：编译通过，encoded row aux 顺序为 `norm(fp16 bits)/base(float)/step(float)`。
5. 实现 `EncodeValueBatch`。
   验证：编译通过，encoded row aux 顺序为 `vmin(float)/vstep(float)`。
6. 拆分 Key/Value merge 和 copyout。
   验证：Key 使用 2-row byte merge，Value 使用 4-row nibble merge。
7. 参考 key1 路径按 `PackKind` 参数化每核任务划分。
   验证：Key 和 Value 分别按自己的 `GROUP_ROWS` 修正 worker segment
   边界，并分别计算 `group_in_block` 和 `group_row`。
8. 运行格式和可用构建检查。
   验证：至少运行 `bash format.sh ci`；如环境具备 CANN/AscendC 构建链，
   再运行对应 op 编译。

## 12. 确认事项和后续项

- `AscendC::ReduceMin<float>` 和 `AscendC::ReduceMax<float>` 已确认可用，
  Key residual 和 Value uniform quantization 直接使用这两个 API。
- `base/step/vmin/vstep` 不再转 fp16，encoded row 和 GM group 中都直接
  保存 float。这样避免精度损失，也省掉 scalar float 到 fp16 bits 的
  转换路径。
- `clip` 已在 6.4 节明确语义和推荐实现。当前实现采用 fp32 阶段
  `Maxs/Mins` 后再 `CAST_RINT`，避免依赖 `int32 Maxs/Mins` 的 AscendC
  支持情况。
- 本文选择 GM group stride 为 288 bytes。attention 计算尚未实现，后续
  设计 attention/dequant 时再同步定义 cache reader，不在本设计中展开。
- Key encoded row 使用 `uint16[128]` 是性能优先的选择：两个 Key code
  pack 后正好是 16bit，AscendC 的 `And` 和 `ShiftLeft` 也按 16bit
  vector lane 工作，一次操作即可拿到完整 Key 或 Value 的 packed/decode
  载体。后续不建议再回退到 `uint8[128]` 中间布局，除非有明确 profile
  证明 UB 占用比额外 widen 成本更关键。
