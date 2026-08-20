# TurboQuant Pack KV For Cache V3 设计说明

`turboquant_pack_kv_for_cache_v3` 基于 v2 提取，输入、输出和属性保持一致，
但只实现 AIV-only 路径。v2 保留 1C2V KFC/Cube rotate，v3 用 vector 指令
实现 rotate matmul，便于后续对比 Cube 与 vector 计算的差异。

## 范围

- 支持 fp16/bf16 的 key/value，head size 固定为 128。
- 支持 8-bit TurboQuant pack，输出 `packed_key` 和 `packed_value`。
- 默认每批次最多处理 64 行，host tiling 会把更大的 `vec_per_core` 截断到 64。
  kernel 内部的 batch 队列使用 depth=1；当前实现没有跨 batch overlap，
  双缓冲只会增加 UB 占用，bf16 路径下容易触发 vector-core invalid-address。
- kernel task type 固定为 `KERNEL_TYPE_AIV_ONLY`，不注册 Matmul 对象，不申请 KFC
  system workspace。
- 前端 Python 调用不变；`turboquant_pack_kv_for_cache_to_cache` 在 Python 层按
  `VLLM_ASCEND_TURBOQUANT_PACK_OP` 路由：默认 `fused` 走单体
  `aclnnTurboquantPackKvForCacheToCache`；`v2`/`v3` 分别优先调用
  `aclnnTurboquantPackKvForCacheV2ToCache` /
  `aclnnTurboquantPackKvForCacheV3ToCache`，不可用时回退 pack + scatter。

## 数据流

1. `CopyInMergedBatch` 从 GM 依次搬入 key/value 行到同一个 VECIN 缓冲区。
2. `Compute` 完成归一化、尾块 padding、vector rotate matmul 和 encode。
3. `CopyOutPackedBatch` 从 VECOUT 搬出结果，按行写回 `packed_key` 或
   `packed_value`。

## Vector Rotate

`RotateBatchMatmulVector` 将 `rotation_t` 固定加载到 UB，直接消费归一化后的
`xBatch`。每个输出维度使用 fp32 累加后 cast 到 fp16，输出布局保持与 v2
Cube 路径一致，后续 `EncodeBatch` 直接消费本地 `yBatch`，不经过 GM scratch。
v3 不再保留 v2 为 KFC Matmul A 输入准备的 VECOUT 中转 batch。

## 同步与缓冲

- `CopyInMergedBatch`、`Compute`、`CopyOutPackedBatch` 之间通过 `TQue` 的
  `EnQue`/`DeQue` 同步。
- 输入、Y 和 packed row 队列使用单缓冲。当前数据流是
  `CopyInMergedBatch -> Compute -> CopyOutPackedBatch` 顺序执行，没有跨 batch
  pipeline，单缓冲能保留 64 行批处理并降低 UB 压力。
- codebook 和 `rotation_t` 在 `Init` 阶段加载到 UB，避免批次间重复从 GM 读取。
