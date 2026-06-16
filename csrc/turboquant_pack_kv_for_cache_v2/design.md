# TurboQuant Pack KV For Cache V2 设计说明

`turboquant_pack_kv_for_cache_v2` 基于 `turboquant_pack_kv_for_cache_fused`
提取，只保留 KFC 融合路径。

范围：

- 只支持当前 8-bit KV pack 路径，`head_size=128`，K/V 输入为 fp16 或 bf16。
- 复用已注册的 TurboQuant codebook 和 `R^T` 表。
- 只运行 1C2V MIX KFC 路径（`KERNEL_TYPE_MIX_AIC_1_2`）。
- 不暴露旧的 AIV-only `pack_mode=1` 调试路径。

Python wrapper 在该算子已构建时优先调用它；若运行环境中没有该算子，则回退到
已有的 registered-table pack 算子以保持兼容。
