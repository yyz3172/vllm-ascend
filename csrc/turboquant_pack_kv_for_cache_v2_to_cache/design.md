# TurboQuant Pack KV For Cache V2 设计说明

`turboquant_pack_kv_for_cache_v2` 基于 `turboquant_pack_kv_for_cache_fused`
提取，只保留 KFC 融合路径。

范围：

- 只支持当前 8-bit KV pack 路径，`head_size=128`，K/V 输入为 fp16 或 bf16。
- 复用已注册的 TurboQuant codebook 和 `R^T` 表。
- 只运行 1C2V MIX KFC 路径（`KERNEL_TYPE_MIX_AIC_1_2`）。
- 不暴露旧的 AIV-only `pack_mode=1` 调试路径。

实现约束：

- codebook 在 kernel 初始化阶段从 GM 复制到 UB，并提前转换出 fp32 codebook 表；后续 K/V
  encode 批次不再重复读取或转换 codebook。
- `R^T` 在 kernel 初始化阶段从 GM 复制到 UB，表数据保持常驻；Rotate Matmul
  的 B 输入仍沿用原版稳定的 GM 路径，避免 KFC Cube 读取 UB B 矩阵时触发本地
  地址/tiling 对齐问题。
- `NormalizeBatch` 只保留必要的 V/Scalar 同步，移除 Cast 后无 GM->UB 依赖的冗余
  MTE2->V 同步。
- 输入 dtype 由 `TqInputTraits` 模板在编译期分派，FP16/BF16 路径不引入运行时分支。
- `CopyInMergedBatch` 只负责搬入和队列入队；BF16 输入路径在 `PackMergedBatch`
  计算阶段从输入队列取出后，按行转换到 fp32 并完成 Normalize，再写回 fp16
  `xBatch`，跳过旧的整批 `bf16 -> fp32 -> fp16` 后再 `fp16 -> fp32`
  Normalize 往返转换。
- K/V 按线性行号合并搬入：先搬 key，key 尾部不足一个 `vecPerCore_` 批次时，
  同一批次继续搬 value 到同一个 `TPosition::VECIN` 缓冲区；归一化后再复制到
  `TPosition::VECOUT` 作为 KFC Matmul 的 A 输入，保持原版稳定的 Cube 输入位置。
- Rotate Matmul 的 C 输出直接写入 `TPosition::VECIN` 的 `yBatch`，后续
  `EncodeBatch` 直接消费本地结果；算子不再为 C 输出申请 per-core GM scratch，
  也不再执行显式的 GM 回拷和 `float -> half` 转换。
- `PackMergedBatch` 只保留搬入、计算、搬出三段；归一化、padding、Cube Rotate
  Matmul、encode 统一封装在 `Compute` 中。
- 每批次最多处理 32 行。即使前端传入更大的 `vecPerCore`，host tiling 和 kernel
  初始化都会限制到 32，以降低 UB 队列和 Matmul 本地输出缓冲占用。
- 搬入阶段不提前 padding；仅在 Rotate Matmul 前按 16 行对齐补零，encode 和搬出阶段
  只处理真实行数，补齐行不会写回。
- pack 后的结果先写入 `TPosition::VECOUT` 队列，队列内部行 stride 按 32B
  对齐，再按线性行号拆分搬出到 `packed_k` 和 `packed_v`；GM 侧只写真实
  `slot_w_k` / `slot_w_v` 字节。
- 搬入、计算中间结果、搬出队列均使用 `TQue` 的 `EnQue/DeQue` 做阶段同步，
  队列深度为 2，保留双缓冲空间。

Python wrapper 在该算子已构建时优先调用它；若运行环境中没有该算子，则回退到
已有的 registered-table pack 算子以保持兼容。
