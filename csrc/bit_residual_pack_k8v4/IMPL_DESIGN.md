# BitResidualPackK8v4 实现设计

## 1. 目标

`bit_residual_pack_k8v4` 用于把 fp16/bf16 的 Key、Value 写入新的
KV cache 压缩格式：

- Key：`1-bit sign + 7-bit residual`，每 2 行合并为一个
  `uint16[128]` group。
- Value：4-bit uniform quantization，每 4 行合并为一个
  `uint16[128]` group。
- Key 和 Value 共用 `normalize + rotate` 前处理。
- 当前实现只保留 direct Mmad manual 路径，不再使用
  `REGIST_MATMUL_OBJ_STATIC + Matmul::IterateAll`。
- 代码、测试和工具统一使用 `bit_residual_pack_k8v4`。

本文描述当前实现，不包含 attention 读取侧设计。Attention 尚未实现，所有后续
cache reader 必须按本文的 cache group stride 和字段布局同步。

## 2. 固定约束

- `head_dim = 128`。
- 输入 dtype 支持 `fp16` 和 `bf16`，Key、Value、`rotation_t` 必须同 dtype。
- 输入 shape 为 `[num_tokens, num_heads, 128]`。
- `block_size` 必须是 4 的倍数。Key group 需要 2 行，Value group 需要 4 行。
- `num_heads >= 8` 且 `num_heads % 8 == 0`。
- 当前 manual 路径要求 head stride 为 128，token stride 为正数。
- 同一 request 内的 slot 按 `first_slot + token_offset` 连续推导。
- cache tensor 暴露为 `uint8`，内部按 `uint16` 和 `float` 字段解释。

## 3. 算法

### 3.1 前处理

每个 token/head 行向量 `x[128]` 先做 L2 normalize，再乘旋转矩阵：

```text
norm = ||x||_2
n = x / (norm + 1e-10)
y = n @ rotation_t
```

实现流程：

1. AIV 从 GM 读取 Key 或 Value 切片到 UB。
2. AIV 使用 fp32 中间量计算 norm，并把 normalized 行写入 A workspace。
3. AIC 从 A workspace 加载 A tile 到 L1，`rotation_t` 常驻 L1。
4. AIC 使用 direct `AscendC::Mmad` 计算 `32x128 @ 128x128`，再通过
   `AscendC::Fixpipe` 写回 C workspace。
5. AIV 从 C workspace 读取旋转后结果，分别执行 Key 或 Value 编码并写入 cache。

`rotation_t` 是 kernel 的第三个输入，布局为 `[128,128]`，当前实现直接计算
`normalized @ rotation_t`。

### 3.2 Key 编码

Key 每行生成 128 个 8-bit code：

```text
sign = y > -0.0
sign_val = sign ? +1/sqrt(128) : -1/sqrt(128)
err = y - sign_val
base = min(err)
maxv = max(err)
step = (maxv - base) / 127
q7 = round((err - base) / step)
q7 = clip(q7, 0, 127)
code = (q7 << 1) | sign
```

实现细节：

- `min/max` 使用 `AscendC::ReduceMin<float>` 和
  `AscendC::ReduceMax<float>`。
- 当 `maxv - base <= 0` 时，`step = 1.0f`，`inv_step = 0.0f`，因此
  `q7 = 0`。
- clip 使用 `AscendC::Maxs(..., 0.0f)` 和
  `AscendC::Mins(..., 127.0f)`，再 `Cast(..., CAST_RINT)` 到 int32。
- `base` 和 `step` 直接按 `float` 写入 cache，不再转 fp16 bits。
- norm 按输入 dtype 存为一个 16-bit word，便于后续 reader 按 dtype 解释。

Key 每 2 行合并：

```text
uint16[d] = (code_row1[d] << 8) | code_row0[d]
```

低 byte 是 group row 0，高 byte 是 group row 1。

### 3.3 Value 编码

Value 每行生成 128 个 4-bit index：

```text
vmin = min(y)
vmax = max(y)
vstep = (vmax - vmin) / 15
idx4 = round((y - vmin) / vstep)
idx4 = clip(idx4, 0, 15)
```

实现细节：

- `vmin/vmax` 同样使用 `AscendC::ReduceMin<float>` 和
  `AscendC::ReduceMax<float>`。
- 当 `vmax - vmin <= 0` 时，`vstep = 1.0f`，`inv_step = 0.0f`，
  `idx4 = 0`。
- clip 使用 `AscendC::Maxs(..., 0.0f)` 和
  `AscendC::Mins(..., 15.0f)`。
- `vmin` 和 `vstep` 直接按 `float` 写入 cache，不转 fp16。
- Value cache 不保存 norm。

Value 每 4 行合并：

```text
uint16[d] =
    (idx4_row3[d] << 12) |
    (idx4_row2[d] << 8)  |
    (idx4_row1[d] << 4)  |
     idx4_row0[d]
```

## 4. Cache GM 布局

本文选择 Key 和 Value 的 GM group stride 均为 288 bytes。所有 cache
reader 必须同步该 stride。

### 4.1 Key cache

Key cache shape：

```text
[num_blocks, num_heads, (block_size / 2) * 288]
```

每个 Key group：

```text
byte[0..255]     = uint16[128] packed code，2 行
byte[256..259]   = uint16[2] norm，row0, row1
byte[260..267]   = float[2] base，row0, row1
byte[268..275]   = float[2] step，row0, row1
byte[276..287]   = padding
```

offset：

```cpp
key_group_base =
    ((uint64_t)block_idx * num_heads + head_idx) *
        (block_size / 2) * 288 +
    group_in_block * 288;
```

### 4.2 Value cache

Value cache shape：

```text
[num_blocks, num_heads, (block_size / 4) * 288]
```

每个 Value group：

```text
byte[0..255]     = uint16[128] packed idx4，4 行
byte[256..271]   = float[4] vmin，row0..row3
byte[272..287]   = float[4] vstep，row0..row3
```

offset：

```cpp
value_group_base =
    ((uint64_t)block_idx * num_heads + head_idx) *
        (block_size / 4) * 288 +
    group_in_block * 288;
```

## 5. `uint16[128]` 的选择

Key encoded row 使用 `uint16[128]` 是性能优先选择。

原因：

- AscendC 的 `And`、`Or`、`ShiftLeft` 路径按 16-bit lane 操作更直接。
- 两个 Key 8-bit code 合并后正好占满一个 `uint16`，没有空间浪费。
- Value 4 行 4-bit index 合并后也正好占满一个 `uint16`。
- cache 中的解码值可以通过一次 16-bit load 加一次 mask/shift 取得，不需要
  `uint8 -> uint16` widening merge。

如果后续为了降低 UB 占用考虑 `uint8[128]` 中间格式，应单独设计 widening
merge，并用 profile 证明收益。当前实现不引入这条路径。

## 6. 每核任务划分

任务划分参考 `turboquant_pack_kv_for_cache4bit` 的 key1 manual core
划分方式。

### 6.1 MIX 1C2V

kernel 使用 `KERNEL_TYPE_MIX_AIC_1_2`：

- 每个 data group 包含 1 个 AIC 和 2 个 AIV。
- `dataCores = min(aic_num, aiv_num / 2)`。
- host 通过 `CalcTschBlockDim(dataCores * 2, dataCores, dataCores * 2)`
  生成 block dim。

AIV 侧：

```cpp
manualGroupId = TqManualRawBlockIdx();
aivSlice = GetSubBlockIdx() % 2;
```

AIC 侧：

```cpp
manualGroupId = GetBlockIdx();
```

### 6.2 Tile 形状

一个 manual tile 固定为：

```text
8 heads * 4 rows = 32 rows
```

两个 AIV 各处理半个 tile：

```text
AIV slice 0: 4 heads * 4 rows = 16 rows
AIV slice 1: 4 heads * 4 rows = 16 rows
```

AIC 对完整 32 行做一次 direct Mmad。

### 6.3 tile 归属

AIV 和 AIC 必须以完全相同顺序遍历：

1. request 递增。
2. request 内按 4-row value group 切分 token rows。
3. head tile 按 8 heads 递增。

全局 tile 编号 `tileOrdinal` 每看到一个 tile 加 1。当前 group 只处理：

```cpp
tileOrdinal % dataCores_ == manualGroupId
```

命中当前 group 后，使用本 group 内的 `manualTileOrdinal` 生成 stream id：

```cpp
keyStream = manualTileOrdinal * 2;
valueStream = keyStream + 1;
```

不能直接使用全局 `tileOrdinal` 的奇偶做 ping-pong。`dataCores_` 为偶数时，
同一 group 命中的全局 tile 奇偶可能固定，导致 ping-pong buffer 被错误复用。

### 6.4 非对齐 slot

request 的首 slot 可能不是 4-row 对齐：

```cpp
leadingGroupRow = firstSlot % 4;
```

第一组可能只覆盖 group 后半段，最后一组也可能不足 4 行。写 cache 时：

- full group：先清 UB group，再写完整 group。
- partial group：先从 GM 读旧 group 到 UB，再用 mask 清理目标 row bits，
  最后 merge 新 row 后写回 GM。

Key group 是 2 行，Value group 是 4 行。manual tile 以 4 行切分，但
`EncodeManualSliceToCache<true>` 会按 Key 的 2-row group 重新计算
`groupInBlock` 和 `firstGroupRow`，不会把 Value 的 4-row group 逻辑直接套到
Key。

## 7. AIC/AIV 同步

当前实现使用 `CrossCoreSetFlag/CrossCoreWaitFlag`，没有使用 `TQue` 做
AIC/AIV 跨核同步。

原因：

- `TQue` 属于当前 core 的 `TPipe` 本地队列，不是跨 core 共享队列。
- AIC 和 AIV 之间需要同步 GM workspace 的生产/消费状态，必须使用跨核同步原语。
- Direct `AscendC::Mmad` 路径已经显式启用，因此可以与 CrossCore flag 配套使用。

每个 ping-pong buffer 有 4 个 flag：

```text
A_FREE   = 0
A_READY  = 1
C_FREE   = 2
C_READY  = 3
```

两个 ping-pong buffer 的 flag base 相差 4：

```cpp
flagBase = (streamOrdinal & 1) * 4;
```

同步流程：

1. AIC set `A_FREE`，两个 AIV 开始写 A workspace。
2. 每个 AIV 完成自己的 16-row A slice 后 set `A_READY`。
3. AIC wait 两个 AIV 的 `A_READY`，加载完整 32-row A tile 到 L1。
4. AIC set `A_FREE`，允许 AIV 复用 A buffer。
5. AIV set `C_FREE`，表示 C workspace 可被 AIC 写。
6. AIC wait 两个 AIV 的 `C_FREE`，执行 Mmad + Fixpipe，写 C workspace。
7. AIC set `C_READY`。
8. AIV wait `C_READY`，读取自己的 C slice，编码并写 cache。
9. AIV set `C_FREE`，表示 C buffer 可再次复用。

## 8. Workspace

host 申请固定 16 MiB workspace。当前 manual bridge 从 offset 512 KiB
开始使用：

```cpp
TQ_MANUAL_WORKSPACE_BYTE_OFFSET = 512 * 1024
```

每个 data group 需要：

```text
2 ping-pong buffers * 64*128 elements * sizeof(T) for A
2 ping-pong buffers * 64*128 elements * sizeof(T) for C
```

代码中用 `TQ_MAX_BATCH_M = 64` 作为每个 buffer stride，虽然 direct Mmad
tile 实际只用 32 行。这样可以复用原有 batch buffer 尺寸，并保留对齐余量。

以 fp16/bf16 和 20 个 data group 估算：

```text
20 * 2 buffers * 2(A/C) * 64*128 * 2 bytes = 1.25 MiB
```

加上 512 KiB 预留，仍远小于 16 MiB。

## 9. TPipe 生命周期

AIV 侧 UB buffer 仍由 `TPipe/TQue/TBuf` 管理：

- `xBatchQue_`
- `aBatchQue_`
- `yBatchQue_`
- norm、reduce、pack merge 等 `TBuf`

AIC direct Mmad 使用 Catlass `Resource<AtlasA2>` 管理 L1/L0 资源，不走
`REGIST_MATMUL_OBJ_STATIC`。

kernel 入口创建 `AscendC::TPipe pipe`，初始化 op 后执行 `op.Process()`，最后
显式调用：

```cpp
pipe.Destroy();
```

如果后续引入 LocalTensor 静态存储，也必须继续保证 `TPipe` 生命周期被正确销毁。

## 10. Host tiling 和接口

Torch binding schema：

```text
bit_residual_pack_k8v4(
    Tensor key,
    Tensor value,
    Tensor slot_mapping,
    Tensor query_start_loc,
    Tensor rotation_t,
    Tensor! key_cache,
    Tensor! value_cache,
    int num_reqs,
    int block_size
) -> ()
```

binding 校验：

- cache shape 必须是 `[num_blocks, num_heads, packed_bytes]`。
- `key_cache.shape[-1] == (block_size / 2) * 288`。
- `value_cache.shape[-1] == (block_size / 4) * 288`。
- `query_start_loc.numel() >= num_reqs + 1`。

tiling data 里仍保留 `TCubeTiling` 字段用于生成代码的 ABI 兼容，但 kernel 不再
注册 Matmul 对象，也不再调用 `IterateAll`。

## 11. 后续工作

- Attention/cache reader 需要单独设计，并同步 288-byte group stride。
- 如果要优化 Key 中间 UB 占用，可以设计 `uint8[128] -> uint16[128]`
  widening merge，但必须用 profile 证明收益超过额外指令和同步成本。
- 如果要扩大支持范围，例如非 8-head 对齐或非连续 slot，需要重新设计 tile 遍历
  和 partial group 写回策略。
