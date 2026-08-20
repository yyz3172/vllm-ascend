# `torch_npu.npu_fused_infer_attention_score` 调用链与 Workspace 机制分析

> 文件路径：`op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp`
> 相关头文件：`op-plugin/op_plugin/utils/op_api_common.h`、`update_op_api_common.h`
> 涉及版本：CANN v2.1 起（`op_api: [v2.1, newest]`）

---

## 1. 整体调用链

```
Python 侧
  torch_npu.npu_fused_infer_attention_score(query, key, value, ...)
       │
       ▼  PyTorch dispatch 机制
  ATen dispatcher → op_plugin 注册的 kernel
       │
       ▼  op_plugin/config/op_plugin_functions.yaml 注册
  "op_api: [v2.1, newest]"
  → 代码路由到
  FusedInferAttentionScoreKernelNpuOpApi.cpp
  npu_fused_infer_attention_score_symint(...)
       │
       ├─ [shape 推导] get_query_and_attention_out_layout()
       │    └─ LAYOUT_MAP 查表: BSH/BSND/TND/NTD/NSD → (qLayout, outLayout)
       │
       ├─ [output tensor] construct_fia_output_tensor()
       │    └─ infer_attention_out_shape() 按 layout 推导输出 shape
       │       例如 BSH → [B, S, H], TND → [T, N, D]
       │    └─ npu_preparation::apply_tensor_without_format()   ← 纯 host 内存分配
       │
       ├─ [softmax_lse] 申请 0-size 占位 tensor（softmax_lse_flag=False 时）
       │
       ├─ [CANN 版本路由]
       │    ├─ is_gte_cann_version_810rc1() ?
       │    │    YES → EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV3, ...)
       │    │    NO  → EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV2, ...)
       │
       ▼  EXEC_NPU_NO_FORMAT_CHECK_CMD 宏展开（op_api_common.h:1410）
  ┌─────────────────────────────────────────────────────────────────┐
  │  HOST 侧（同步）                                                 │
  │  1. GetOpApiFuncAddr("aclnnFIA...GetWorkspaceSize")             │
  │     → dlsym(libopapi.so, name)  首次查符号，此后 static 缓存    │
  │  2. GetOpApiFuncAddr("aclnnFusedInferAttentionScoreV2/V3")      │
  │  3. InitHugeMemThreadLocal()   初始化 host-side huge memory     │
  │  4. ConvertTypes(params)       Tensor→aclTensor*，仅读 metadata  │
  │  5. getWorkspaceSizeFunc(params, &workspace_size)               │
  │     ← 调用 libopapi.so，计算本次所需 workspace 字节数           │
  │     ← **HOST 同步调用**，是每次 dispatch 的 latency 主要来源    │
  │  6. unsafe_empty_workspace(workspace_size)                      │
  │     ← 在 NPU HBM 上动态分配 workspace                          │
  │                                                                 │
  │  DEVICE 侧（异步）                                               │
  │  7. opApiFunc(params, workspace_addr, workspace_size, stream)   │
  │     ← 调用 libopapi.so 的 aclnnFusedInferAttentionScoreV2/V3   │
  │     ← 把 kernel 投递到 NPU stream，立即返回                     │
  │  8. UnInitHugeMemThreadLocal() / ReleaseHugeMem()               │
  └─────────────────────────────────────────────────────────────────┘
       │
       ▼  libopapi.so（闭源 CANN runtime）
  aclnnFusedInferAttentionScoreV2/V3/V4
       │
       ├─ [tiling] 查缓存 / 计算 tiling 配置
       │    key = (B, S_q, S_kv, H, D, block_size, layout, sparse_mode, ...)
       │    cache hit  → 复用 tile，无 host 阻塞
       │    cache miss → 计算 tile，host wait，更新缓存（首次出现时）
       │
       ├─ [PA 寻址] 从 block_table 计算物理 KV 地址
       │    for seq s, k_block in [0, ceil(actual_seq_lengths_kv[s] / block_size)):
       │      phys = block_table[s, k_block]
       │      kv_ptr = kv_base + phys * block_stride
       │    ← 按 actual_seq_lengths_kv 切外层循环上界
       │    ← block_table 在上界之外的元素理论上不解引用
       │      但 kernel 可能按 max_blocks_per_seq 整行做地址预算（影响 HBM 流量）
       │
       ├─ [antiquant 反量化]  若 antiquant_scale/offset 不为 None
       │    kv_fp16 = (kv_int8 - antiquant_offset) * antiquant_scale
       │    ← per-channel 线性反量化，写死在 kernel 内
       │    ← turboquant（码本 + Haar 旋转）无法走此通道
       │
       ├─ [FlashAttention 在线 softmax + Q·K·V 矩阵乘]
       │    for each q_tile:
       │      for each k_block in valid range:
       │        kv_tile = gather(key/value, block_table[s, k_block])
       │        S = (Q_tile @ kv_tile.T) * scale    ← Cube 矩阵乘
       │        S = apply_mask(S, sparse_mode, atten_mask, pre_tokens, next_tokens)
       │        online_softmax_update(m, l, O, S, V_tile)  ← Vector 单元
       │      O /= l                                 ← 最终归一化
       │
       ├─ [quant 输出]  若 quant_scale2/quant_offset2 不为 None
       │    out_int8 = clamp(O * quant_scale2 + quant_offset2).to(int8)
       │
       └─ 写回 attention_out, softmax_lse
```

---

## 2. 三个 Python 入口及其对应的 C++ 宏

| Python 入口 | C++ 函数 | 宏 | 行为 |
|---|---|---|---|
| `_npu_fused_infer_attention_score_get_max_workspace` | `npu_fused_infer_attention_score_get_max_workspace_symint` | `EXEC_GET_MAX_WORKSPACE_CMD` | **只查大小**，分配 HBM tensor，**不执行** kernel |
| `npu_fused_infer_attention_score` | `npu_fused_infer_attention_score_symint` | `EXEC_NPU_NO_FORMAT_CHECK_CMD` | 内部查大小 + 分配 workspace + 执行 |
| `npu_fused_infer_attention_score.out` | `npu_fused_infer_attention_score_out` | `EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD` | **复用外部 workspace** + 执行，**无查大小** |

`.out` 变体（`EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD`）的精简流程：

```
普通路径:
  ① getWorkspaceSizeFunc(params, &workspace_size)   ← HOST 同步
  ② unsafe_empty_workspace(workspace_size)           ← HBM 分配
  ③ opApiFunc(workspace_addr, workspace_size, stream) ← 异步投流

.out 路径（ACL Graph 用）:
  ③ opApiFunc(workspace_addr, workspace_size, stream) ← 只有这一步，无同步
```

省掉 ① 和 ② 消除了每层 attention 一次 host round-trip（`GetWorkspaceSize` 调用）。

---

## 3. Workspace 机制详解

### 3.1 为什么需要 workspace

FIA kernel 在计算过程中需要 HBM 上的中间缓存：

- FlashAttention 在线归一化状态（`m_i`、`l_i`）
- 跨 tile 的部分累加 `O_i`
- GQA/MQA 展开的中间矩阵
- 双缓冲 prefetch 区（CANN 内部实现）

这些中间状态的大小随输入 shape 变化，不能静态分配在 kernel 寄存器/L2 里，必须动态申请 HBM 空间。

### 3.2 普通路径：每次动态申请

```
Python 调用:
  torch_npu.npu_fused_infer_attention_score(query, key, value, ...)
       │
       ▼ [HOST 同步]
  GetWorkspaceSize(params) → 返回 workspace_size（字节数）
       │
       ▼ [HOST 分配]
  unsafe_empty_workspace(workspace_size) → workspace_addr（NPU HBM 指针）
       │
       ▼ [DEVICE 异步]
  opApiFunc(workspace_addr, workspace_size, stream)
       │
       ▼ [HOST]
  workspace_tensor 生命周期绑定到本次调用，kernel 完成前 tensor 不被释放
```

**性能影响**：每次 forward 每层 attention 都有一次 `GetWorkspaceSize` 的 host 同步调用，这是 profiling 里 `aclrtSynchronizeEvent` 的来源之一。

### 3.3 ACL Graph 路径：workspace 预申请与复用

ACL Graph 捕获阶段不允许动态内存分配。`attention_v1.py` 的 `full_graph_fia` 采用三段式：

**阶段 A — 捕获前预查（只在首次或 shape 改变时执行）：**

```python
workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
    query=query, key=key, value=value,
    atten_mask=attn_metadata.attn_mask,
    block_table=block_table,
    input_layout="TND",
    block_size=block_size,
    actual_seq_lengths=actual_seq_lengths_q,
    actual_seq_lengths_kv=actual_seq_lengths_kv,
    num_heads=self.num_heads,
    ...
)
graph_params.workspaces[num_tokens] = workspace   # 缓存到 num_tokens → tensor 的字典
```

**阶段 B — ACL Graph 捕获期间，用 `.out` 变体复用外部 workspace：**

```python
torch_npu.npu_fused_infer_attention_score(
    query, key, value,
    ...,
    out=[attn_output, softmax_lse],
)
# .out 变体走 EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD
# 跳过 GetWorkspaceSize，直接调用 opApiFunc(workspace_addr, ...)
```

**阶段 C — 图执行时：**

```
每次 decode step 重复复用同一块 workspace tensor
→ 无动态分配，无 GetWorkspaceSize 同步
→ 只有 opApiFunc 的异步投流
```

实际代码（`attention_v1.py:590-628`）：

```python
workspace = graph_params.workspaces.get(num_tokens)
if workspace is None:
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(...)
    update_graph_params_workspaces(num_tokens, workspace)
# workspace 已缓存，直接进图捕获
```

### 3.4 Workspace 大小的决定因素

`GetWorkspaceSize` 返回的字节数取决于：

| 因素 | 影响 |
|---|---|
| `(batch, S_q, S_kv, N_heads, D)` | 主要 shape 参数，正比于中间状态大小 |
| `block_size` | 决定 tile 粒度 |
| `sparse_mode` | 不同 mask 路径需要不同 tile 策略 |
| `softmax_lse_flag=True` | 需要额外存储 log-sum-exp |
| CANN 版本（V2/V3/V4） | V3 支持 RoPE in-kernel，可能需更多中间状态 |

以 `num_tokens` 作为缓存 key，同一 batch shape 下只申请一次。

---

## 4. CANN 版本差异

| CANN 版本 | aclnn 入口 | 新增能力 |
|---|---|---|
| `< 8.1.RC1`（V2R1） | `aclnnFusedInferAttentionScoreV2` | PA + 基础量化 |
| `≥ 8.1.RC1`（V3R0） | `aclnnFusedInferAttentionScoreV3` | **RoPE in-kernel** + per-KV-head antiquant |
| V4（最新） | `aclnnFusedInferAttentionScoreV4` | 更多量化组合（V2 接口的扩展参数集） |

版本路由在 `FusedInferAttentionScoreKernelNpuOpApi.cpp` 中判断：

```cpp
if (op_plugin::utils::is_gte_cann_version_810rc1()) {
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV3, ...);
} else {
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV2, ...);
}
```

---

## 5. 关键参数的语义与在 kernel 内的使用

### 5.1 `actual_seq_lengths_kv` — 切外层循环

```cpp
// kernel 内部伪码
for seq_id in range(batch):
    max_kblocks = ceil(actual_seq_lengths_kv[seq_id] / block_size)
    for k_block in range(max_kblocks):           // ← 由此参数切边界
        phys = block_table[seq_id * max_blocks_per_seq + k_block]
        kv_ptr = kv_base + phys * block_stride
        // tile 计算
```

`block_table` 在 `max_kblocks` 之外的元素**理论上不被解引用**，但 kernel 可能按 `max_blocks_per_seq` 整行预算地址，导致 padding 槽位的取值仍影响 HBM 访问模式（cache line 聚合 vs 分散）。

### 5.2 `block_table` — PA 物理块寻址

- 类型：`int32 [batch, max_blocks_per_seq]`
- 元素语义：物理块索引，**必须 ≥ 0**（负数会被当成 `kv_base + 负偏移`）
- vLLM v1 约定：未使用槽位填 0（`null_block`），不用 -1

### 5.3 `antiquant_scale/offset` — per-channel 线性反量化

```
kv_fp16 = (kv_int8 - antiquant_offset) * antiquant_scale
```

写死在 aclnn kernel 内，**无法表达 turboquant 的码本 + Haar 旋转结构**。因此 vllm-ascend 的 turboquant 路径必须在 Python 层先解码到 fp16 workspace，再调用 FIA。

### 5.4 `sparse_mode` — mask 路径

| 值 | 含义 |
|---|---|
| `0` | 无 mask |
| `3` | 因果（causal，下三角）|
| `4` | SWA（滑动窗口，由 `pre_tokens`/`next_tokens` 控制）|
| `10` | 高精度无 mask |
| `14` | 高精度 + 因果（band）|

---

## 6. ConvertTypes — 参数格式转换

胶水层把 PyTorch 类型转成 aclnn 期待的格式，**只读 tensor 的 metadata，不读数据内容，无同步**：

| PyTorch 类型 | aclnn 类型 |
|---|---|
| `at::Tensor` | `aclTensor*`（NPU 内存描述符指针）|
| `c10::optional<at::Tensor>` | `aclTensor*`（None → `nullptr`）|
| `c10::SymInt[]` | `int64_t*` + size |
| `std::string`/`c10::string_view` | `const char*` |

---

## 7. 对 TurboQuant 工作的直接结论

| 问题 | 结论 |
|---|---|
| turboquant 能复用 FIA 的 antiquant 通道吗 | ❌ aclnn 内写死 per-channel 线性反量化，码本+旋转无法表达 |
| `bt_compact` 能含 -1 吗 | ❌ aclnn 把 int32 直接当块索引，-1 → 负地址越界 |
| padding 槽位取值影响性能吗 | ✅ 影响。kernel 可能按整行预算地址，padding 收敛到同一 slot（如 0）可利用 cache line 复用，避免 N 次独立 HBM 读 |
| decode_only_arange_fast_path 的正确 padding 值 | 0（指向 workspace 哨兵 slot，内容为 null_block 解码 ≈ 全 0）|
| turboquant 普通路径的额外同步来源 | `GetWorkspaceSize` 的 host 同步（每层 attention 一次）|
| 如何消除 GetWorkspaceSize 同步 | 切到 PIECEWISE 图模式，用预申请 workspace + `.out` 变体（Phase 0.5 guard 落地后）|
| 方案 B（片上融合）的难点 | 要在自定义 AscendC kernel 里复刻 aclnn 的 paged FlashAttention，KV 解码改成码本路径 |

---

## 8. 关键文件索引

| 文件 | 作用 |
|---|---|
| `op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp` | 胶水层主体：shape 推导、版本路由、宏调用 |
| `op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreV2KernelNpuOpApi.cpp` | V4 接口的更多量化参数变体 |
| `op-plugin/op_plugin/utils/op_api_common.h` | `EXEC_NPU_NO_FORMAT_CHECK_CMD*` 宏定义，含 GetWorkspaceSize + 分配 + 投流 |
| `op-plugin/op_plugin/utils/custom_functions/opapi/update_op_api_common.h` | `EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD` 宏，仅投流（用于 ACL Graph）|
| `op-plugin/op_plugin/config/op_plugin_functions.yaml` | op 注册声明，决定走 `op_api` 还是 `acl_op` 路径 |
| `vllm_ascend/attention/attention_v1.py` | `full_graph_fia`：workspace 预申请 + 复用的实现 |
| `vllm_ascend/ops/turboquant_kv_cache.py` | turboquant decode + FIA 调用入口 |
