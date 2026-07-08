# BitResidualPackK8v4 简要说明

`bit_residual_pack_k8v4` 将 token-major 的 Key/Value 张量写入新的
BitResidual KV cache 布局。

当前实现范围：

- `head_size = 128`。
- Key/Value/`rotation_t` 支持 `fp16` 和 `bf16`，三者 dtype 必须一致。
- 输入布局为 `[num_tokens, num_heads, 128]`，支持 stride 传入。
- `slot_mapping` 和 `query_start_loc` 为 `int32`。
- `block_size` 必须是 4 的倍数。
- Key cache 和 Value cache 均以 `uint8` 暴露，但内部按 `uint16[128]`
  group 解释。

cache 布局：

- Key: 每 2 行组成一个 group，每个 group stride 为 288 bytes。
- Value: 每 4 行组成一个 group，每个 group stride 为 288 bytes。
- Key group 保存 `uint16[128]` packed code、`uint16[2]` norm、
  `float[2]` base 和 `float[2]` step。
- Value group 保存 `uint16[128]` packed idx4、`float[4]` vmin 和
  `float[4]` vstep。

详细算法、每核任务划分、group merge 和接口变更见
`IMPL_DESIGN.md`。
