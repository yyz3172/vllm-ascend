# TurboQuant 反量化与 FIA 融合：特性优化设计

> 适用范围：`vllm_ascend/ops/turboquant_kv_cache.py` + `csrc/turboquant_fused_infer_attention_score8bit/` + `attention_v1.py` 的 turboquant 分支
> 分支：`v0.18.0-branch`
> 前置阅读：《attention_v1_mechanism.md》

本文档面向后续开发，给出 TurboQuant 量化融入 Attention 的优化方案，包含三部分：
1. 纯 PyTorch 实现说明 + 不支持的状态 + 如何支持；
2. 反量化与 FIA 融合 —— **方案 A（解码算子化 + 复用 FIA）**；
3. 反量化与 FIA 融合 —— **方案 B（片上真正融合）**。
并给出开发计划与测试用例设计。

---

## 0. 背景与目标

TurboQuant 低比特 KV cache 的注意力当前走"**先解码到紧凑 buffer，再调 FIA**"的两段式（纯 PyTorch）。相对非量化路径，它额外引入：

- **block 压缩重映射** `unique/searchsorted/index_select`（host 侧，含数据相关 shape，见机制文档 §5.3）；
- **整份被引用 KV 的解码物化**（HBM 读写放大）。

优化目标：在保证数值正确的前提下，**逐步消除上述额外开销**，并支持更多状态。

### 范围约定：仅 8-bit，暂不考虑 4-bit

> **本文档只针对 8-bit TurboQuant。** 4-bit 路径（含已有的 `turboquant_decode_packed_blocks_compact` 自定义算子）**正确性同样未经验证**，因此 **不作为参考模板、不在本期开发范围内**。
>
> 优先级：**先把 8-bit 的全部能力（方案 A → 方案 B → 扩展状态）做对做稳**；待 8-bit 完整可用并通过验证后，再单独评估是否复用其经验推进 4-bit。下文所有自定义算子/测试默认 `bits_key == bits_value == 8`。

三种实现的关系：

```
纯 PyTorch(现状) ──A──> 解码算子化 + 复用 FIA ──B──> 片上真正融合(解码内联进 FlashAttention)
   多 kernel        单解码 kernel + FIA(已验证)      单 kernel，无 compaction、无解码物化
```

---

## 1. 纯 PyTorch 实现说明

### 1.1 数据流

入口：`forward_fused_infer_attention`（`attention_v1.py:827`）→ `turboquant_decode_kv_cache_compact`（`turboquant_kv_cache.py:710`）→ `npu_fused_infer_attention_score`。

```
① bt = block_tables.to(int32)
② valid = bt >= 0
③ any(valid) + Python 分支            # D2H 同步
④ used = bt[valid]                    # 布尔 gather，数据相关 shape
⑤ used_sorted = used.unique()         # 排序去重，数据相关 shape
⑥ key_packed = key_cache.index_select(0, used_sorted)   # gather U 个块 [U,BS,H,P]
   value_packed = ...
   ─ 解码：codebook[idx] → @R → ×norm  →  k,v: [U,BS,H,D] fp16
⑦ bt_compact = bt.clone()
⑧ bt_compact[valid] = searchsorted(used_sorted, used)   # 反向查表 + 散射
→ npu_fused_infer_attention_score(query, key.flatten, value.flatten, block_table=bt_compact, ...)
```

解码语义（`turboquant_kv_cache.py:347` / `:532`）：
```
y_hat = codebook[idx]      # 8-bit: 256 项码本
x_hat = y_hat @ R          # 反 Haar 旋转
x_hat = x_hat * norm       # fp16 norm
```

> 注：**8-bit 解码目前走纯 PyTorch 参考路径**（`turboquant_dequantize_from_packed_bytes`，`turboquant_kv_cache.py:532`），这是本期要替换/加速的对象。
> （仓库里另有 4-bit 的 `turboquant_decode_packed_blocks_compact`，但其正确性未验证，本文档不依赖、不参考。）

### 1.2 当前支持矩阵

| 状态 | 纯 PyTorch 解码+FIA | 8-bit 融合算子(已加) |
|---|---|---|
| PrefillNoCache | ✅（block_table=None，不进 compact）| ❌（命中门要求 block_table≠None）|
| PrefillCacheHit | ✅ | ❌ 未实现多 token 因果 |
| DecodeOnly | ✅ | ⚠️ 仅此状态正确，但门未收窄 |
| ChunkedPrefill | ✅ | ❌ 未实现多 token 因果 + decode/prefill 混合 |
| SpecDecoding | ✅ | ❌ 未实现多 query token 因果 |

纯 PyTorch 路径 **功能上覆盖全部状态**（因为最终复用通用 FIA），问题只在 **性能**（compaction + 解码物化 + 多 kernel launch）。

### 1.3 纯 PyTorch 算子不支持/待改进项 + 如何支持

| 问题 | 现状 | 如何支持 |
|---|---|---|
| 8-bit 无 compact 解码算子 | 走纯 PyTorch 逐元素解码，慢 | 新增 **独立的 8-bit** compact 解码自定义算子（码本 256、无 nibble 解包、行宽 130），不依赖 4-bit 实现，需独立数值验证 |
| `unique/searchsorted` 数据相关 shape | 每步执行，破坏图捕获 | DecodeOnly 走 `arange` 快路径替换 ⑤⑧（见机制文档 §6 Q2）|
| 解码 KV 物化到 HBM | `index_select`+`cat`+`contiguous` 多次读写 | 方案 A 解码 kernel 减少中间物化；方案 B 彻底不物化 |
| 多 kernel launch | ~8–10 次 | 方案 A 合并解码为 1 次；方案 B 合并解码+attention 为 1 次 |

**纯 PyTorch 层面可立即落地的低风险改进（Phase 0）**：
1. 新增 **8-bit** compact 解码自定义算子（独立实现，不复用未验证的 4-bit 算子）。
2. `turboquant_decode_kv_cache_compact` 按 `attn_state==DecodeOnly` 走 `arange`，其余保留 `unique`。
3. 给所有 8-bit turboquant 算子路径补 env 开关与 `_c_ascend_turboquant_op_available` 双重门控（与现有 `VLLM_ASCEND_TURBOQUANT_DECODE_OP` 风格一致）。

---

## 2. 方案 A —— 解码算子化 + 复用 FIA

### 2.1 思路

把"解码 packed KV → fp16"做成 **一个向量化的 AscendC 解码算子**（codebook 查表 + Cube 做 `y_hat@R` + ×norm），输出 fp16 K/V 到 workspace；随后调 **已验证、已优化的 `npu_fused_infer_attention_score`**。

```
[A] decode_kernel(packed K/V, codebook, R) → fp16 K/V(workspace) → FIA(已验证)
```

### 2.2 与纯 PyTorch 对比的优势

| 维度 | 纯 PyTorch | 方案 A |
|---|---|---|
| 解码 kernel launch | ~8–10 | **1** |
| 中间物化 | `codebook[idx]`、decode、cat、contiguous 多份 | y_hat 留 UB，仅写一次 decoded |
| 解码 HBM 流量 | ~5–7× 最小 | ~2× 最小（约 **3–4× 下降**）|
| 片上复用 | codebook/R 每次从 HBM 读 | codebook 常驻 UB、R 常驻 L1 |
| host 串行点 | unique/searchsorted 卡 stream | 折进 kernel 寻址 |
| 计算单元 | 通用 batched matmul | Cube，R 作 stationary |
| FIA 段 | 同 | **完全相同（复用）** |

> 注意：方案 A **仍保留** block 压缩重映射（除非配合 §1.3 的 `arange` 改进），且 decoded KV 仍落 workspace 再被 FIA 读回——这部分 HBM 往返是方案 B 才消除的增量。

### 2.3 性能预期

- 解码阶段 HBM 流量约降到 1/3~1/2，launch ~10→1。
- 端到端受 **解码:FIA 占比** 制约（Amdahl）：**长上下文、小 batch（典型 decode）收益最大**。
- 不能给确切倍数，需 profiling 实测。

### 2.4 状态支持

方案 A 因复用通用 FIA，可覆盖 **全部状态**（与纯 PyTorch 一致），只是把解码换成快算子。

### 2.5 复杂度/风险

- **无可信模板**：4-bit 算子未验证，不作参考；8-bit 解码算子需从头实现并独立验证。
- 验证相对简单：只需对"解码算子输出"与纯 PyTorch 8-bit 解码（`turboquant_dequantize_from_packed_bytes`）做数值对拍即可，golden 明确。

### 2.6 Phase 1 算子实现流程

> 本节是 Phase 1（方案 A）从零落地 8-bit 解码 AscendC 算子的实现规约，覆盖：算子接口、内存布局、kernel 内部数据流、tiling 设计、host/Python 改造、数值验证、与 Phase 2 的复用契约。
>
> 选型已固定：**仅支持 8-bit 标准 130 字节行宽（`P == head_size + 2 == 130`）；R 矩阵常驻 L1（stationary）**。

#### 2.6.1 算子定位与新增目录

> **方案 X**：把 block_table 寻址逻辑折进 decode 算子内部，host 侧只做轻量 `ceil`+`cumsum`，**消除原来的 `unique`/`searchsorted`/`index_select` 链路**与 packed KV 的中间物化。算子直接吃完整 packed cache + 紧凑物理块号列表，输出按 seq 顺序紧凑排布的 fp16 K/V workspace。

新算子目录：`csrc/turboquant_decode_paged_8bit/`（**独立**，**不复用** 4-bit 的 `turboquant_decode_packed_blocks_compact`）。

```
csrc/turboquant_decode_paged_8bit/
├── op_host/
│   ├── CMakeLists.txt
│   ├── turboquant_decode_paged_8bit_def.cpp        # OpDef 注册
│   ├── turboquant_decode_paged_8bit_tiling.h       # TilingData 字段
│   ├── turboquant_decode_paged_8bit_tiling.cpp     # Tiling 函数
│   └── aclnn_turboquant_decode_paged_8bit.h        # 对外 aclnn 头
├── op_kernel/
│   ├── decode_device.h                             # DecodeRows8bit device 函数（Phase 2 复用）
│   └── turboquant_decode_paged_8bit.cpp            # AscendC kernel
└── turboquant_decode_paged_8bit_torch_adpt.h       # torch 绑定
```

参考目录结构：`csrc/turboquant_fused_infer_attention_score8bit/`（方案 B 的占位）和 `csrc/turboquant_pack_kv_for_cache_fused/`（已有的 encode 算子，KFC 模板的源）。

#### 2.6.2 算子接口

算子直接吃**完整 packed KV cache + 紧凑物理块号列表**，内部完成"按物理块号寻址 → 解码"两件事。输出按 seq 顺序紧凑排布，下游 FIA 用 `bt_compact` 索引访问。

| 项 | 类型/Shape | 含义 |
|---|---|---|
| **Inputs** | | |
| `key_cache` | `uint8 [num_blocks_total, BS, H, P]` | **完整** packed K cache，**不预 gather** |
| `value_cache` | `uint8 [num_blocks_total, BS, H, P]` | **完整** packed V cache（与 K 同形状）|
| `gather_block_ids` | `int32 [total_blocks]` | 按 seq 顺序排好的紧凑物理块号列表，host 侧 `ceil+cumsum` 生成 |
| `codebook` | `fp16 [256]` | 8-bit 码本，K/V 共用 |
| `rotation` | `fp16 [128, 128]` | Haar 旋转矩阵 R，K/V 共用 |
| **Attrs** | | |
| `head_size` | `int64`（必为 128）| 校验 `P == head_size + 2 == 130` |
| `block_size` | `int64`（128）| BS，从 cache shape 也能推出来，作 attr 是为 tiling 阶段固化 |
| `out_dtype` | `int64`（0=fp16, 1=bf16）| 输出 dtype；本期仅支持 fp16=0 |
| **Outputs** | | |
| `key_out` | `fp16 [total_blocks, BS, H, 128]` | K 解码 fp16，按 seq 顺序紧凑排布 |
| `value_out` | `fp16 [total_blocks, BS, H, 128]` | V 解码 fp16，与 `key_out` 同 layout |

设计抉择：

- **K/V cache 形状必须相同**：vllm-ascend 当前 KV cache layout 保证 K/V 引用同一组 block、同一份 `block_table`，且 K/V 各槽 `P=130`。tiling 阶段对 `key_cache.shape == value_cache.shape` 做严格校验。
- **codebook + rotation 单份共享**：K/V 共用 8-bit Lloyd-Max 码本与 Haar 旋转（`bits_key == bits_value == 8`，`_get_quantizer(head_size, 8, ...)` 返回同一对象）。kernel 内 codebook 常驻 UB 512 字节、rotation 由 Cube 子系统自动 stationary L1 32 KB，K/V 复用同一份。
- **不返回 `bt_compact`**：`bt_compact` 由 host 侧在调用算子前用 `block_offsets[s] + j` 配 `where` 收敛 padding 一次性生成（见 §2.6.6），不增加算子输出耦合。
- **不接受原始 block_table**：算子只吃已展开的紧凑列表 `gather_block_ids`，避免 kernel 内做跨 core prefix-sum。host 侧 `ceil+cumsum` 是 batch 长度的小张量 op，成本 < 几 µs。
- **不带 batched rotation**：4-bit 路径里有 `rot_batched`（`[batch, D, D]`）。8-bit 算子内 **不批量化 rotation**——它在 L1 stationary，所有行（K 与 V）复用同一份。

紧凑布局示意（`batch=3, max_bps=4, BS=128, actual_seq_lengths_kv=[200, 50, 350]`）：

```
num_blocks_per_seq = ceil([200,50,350]/128) = [2, 1, 3]
block_offsets       = [0, 2, 3]            # cumsum 起点
total_blocks        = 6

gather_block_ids    = [bt[0,0], bt[0,1],   # seq0 的 2 块
                       bt[1,0],            # seq1 的 1 块
                       bt[2,0], bt[2,1], bt[2,2]]  # seq2 的 3 块

key_out 排布:
  workspace[0] = decode(key_cache[bt[0,0]])
  workspace[1] = decode(key_cache[bt[0,1]])
  workspace[2] = decode(key_cache[bt[1,0]])
  workspace[3] = decode(key_cache[bt[2,0]])
  workspace[4] = decode(key_cache[bt[2,1]])
  workspace[5] = decode(key_cache[bt[2,2]])
```

无 padding 块占位、无 `unique` 排序去重，**workspace 大小 = 真实引用块数 × BS × H × D**。

#### 2.6.3 Tiling 设计

work 单位 = "一个紧凑块解码"（共 `total_blocks * BS * H` 行 packed），按行一维切核：

```cpp
// turboquant_decode_paged_8bit_tiling.h
BEGIN_TILING_DATA_DEF(TurboquantDecodePaged8bitTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);  // Cube matmul 子 tiling
TILING_DATA_FIELD_DEF(uint32_t, blockDim);              // 实际使用的 core 数
TILING_DATA_FIELD_DEF(uint32_t, totalBlocks);           // total_blocks
TILING_DATA_FIELD_DEF(uint32_t, blockSize);             // BS=128
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);            // H
TILING_DATA_FIELD_DEF(uint32_t, headSize);              // 128
TILING_DATA_FIELD_DEF(uint32_t, packedBytes);           // 130
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);           // 每核处理多少 packed 行（K 段）
TILING_DATA_FIELD_DEF(uint32_t, outDtype);              // 0=fp16
END_TILING_DATA_DEF;
```

Tiling 函数行为（参考 `turboquant_pack_kv_for_cache_fused_tiling.cpp` 的 KFC + Cube 模板）：
1. `coreNum = ascendcPlatform.GetCoreNumAic()`（混合任务以 AIC 数为基准，1 AIC + 2 AIV）；
2. `totalRowsKv = total_blocks * BS * H`（K 一段 V 一段）；
3. `rowsPerCore = ceil(totalRowsKv / coreNum)`，`blockDim = ceil(totalRowsKv / rowsPerCore)`；
4. **不手动指定 corenum**——按 platform AIC 数自动取，剩余 work 自然落在最后一个 core；
5. 每核负责 `[start, end)` 行号，K 段先做、V 段紧接做，复用同一 Matmul 实例；
6. 调用 `MultiCoreMatmulTiling.GetTiling(cubeTiling)` 填 Cube 子 tiling，固定 `M=T_rows, N=128, K=128`，A 是 fp16 VECOUT，B 是 fp16 GM rotation，C 是 fp32 GM workspace。

每核 UB 占用预估（仅含一组处理流水，K/V 复用相同 buffer）：

| Buffer | 大小 | 说明 |
|---|---|---|
| codebook UB | 512 B | fp16 [256] 常驻 |
| packed double-buffer | 2 × T_rows × 130 ≈ 8 KB（T_rows=32）| MTE2 双缓冲 |
| y_hat UB | T_rows × 128 × 2 = 8 KB | Vector 计算输出（VECOUT）|
| norm UB | T_rows × 2 = 64 B | reinterpret 后 |
| Cube C readback UB | T_rows × 128 × 4 = 16 KB | fp32 → 转 fp16 |
| Matmul 局部 workspace | ~16 KB | KFC 模式下 Matmul 内部 |
| **合计** | **~50 KB** | 远小于 910B 单核 UB（192~256 KB） |

**`T_rows = 32`** 起步。Cube tiling 选 `baseM=32, baseN=128, baseK=128`，单次 Mmad 即覆盖一个 tile，无 K 方向 split。

#### 2.6.4 Kernel 内部数据流

> 本节先讲数据流，§2.6.5 详细说明 Cube/Vector 融合机制。

**寻址逻辑（与原算子最大差异）**：每个 tile 解码前，先按"紧凑块序号 i"从 `gather_block_ids` 读出物理块号 `phys`，再去 `key_cache` / `value_cache` 的对应位置取 packed bytes。这把原来 host 侧的 `index_select` 折进了 kernel 内部一次 int32 GM 读 + 地址计算。

```
对每个 core（AIC + 2 AIV 协同，KERNEL_TYPE_MIX_AIC_1_2）：
  ┌─ 一次性加载（核启动时）──────────────────────┐
  │ codebook[256] fp16 → UB（512 B 常驻）        │
  │ rotation 留在 GM；KFC Cube 在首次 Iterate 时   │
  │   把 R 拉到 L1 stationary（对 AIV 透明）      │
  └──────────────────────────────────────────────┘

  // 紧凑序号编址：[0, total_blocks * BS * H) 行，前一半 K，后一半 V
  // —— 实际实现按"块"切（每块 BS*H 行连续），少几次寻址
  for kv_round in {KEY, VALUE}:                # K 段做完做 V 段，资源全部复用
    src_cache_gm = (kv_round == KEY) ? key_cache_gm : value_cache_gm
    dst_out_gm   = (kv_round == KEY) ? key_out_gm   : value_out_gm

    for compact_blk_id in core 分到的紧凑块序号段:
      // 寻址（每个紧凑块查一次）
      phys = gather_block_ids_gm.GetValue(compact_blk_id)   // int32 GM 读
      src_block_addr = src_cache_gm + phys           * (BS * H * 130)
      dst_block_addr = dst_out_gm   + compact_blk_id * (BS * H * 128 * 2)

      // 块内 BS*H 行按 T_rows 切 tile
      for tile_start in range(0, BS*H, T_rows):
        M = min(T_rows, BS*H - tile_start)

        ┌─ 1. 加载 packed (UB 双缓冲) ───────────┐
        │ DataCopy(packedUB,                     │
        │          src_block_addr + tile_start*130, │
        │          M * 130)                       │
        └─────────────────────────────────────────┘

        ┌─ 2. 解析 idx 与 norm ──────────────────┐
        │ idxUB  = packedUB[:, :128]   (M, 128)   │
        │ normUB = packedUB[:, 128:130]           │
        │           .ReinterpretCast<half> (M, 1) │
        └─────────────────────────────────────────┘

        ┌─ 3. codebook 查表（向量化，AIV）───────┐
        │ y_hatUB[i, j] = codebook[idxUB[i, j]]   │
        │ Gather + 256 路 LUT，详见 §2.6.5         │
        │ 输出：fp16 (M, 128) on VECOUT           │
        └─────────────────────────────────────────┘

        ┌─ 4. y_hat × norm（broadcast，AIV）─────┐
        │ y_hatUB[i, j] *= normUB[i]              │
        │ AscendC::Mul / Muls（dim=1 broadcast）  │
        │ 仍在 VECOUT，准备喂给 Cube              │
        └─────────────────────────────────────────┘

        ┌─ 5. Cube: x_hat = y_hat @ R（AIC，KFC）┐
        │ A = y_hatUB on VECOUT (M, 128) fp16     │
        │ B = rotationGm        (128,128) fp16    │
        │ C = cubeCGm           (M, 128) fp32     │
        │ rotateMm.SetTensorA / SetTensorB / Iter │
        │ —— AIC 异步，AIV 并行准备下一 tile      │
        └─────────────────────────────────────────┘

        ┌─ 6. Cube 输出回 UB + Cast fp16（AIV）──┐
        │ DataCopy(yFp32UB, cubeCGm, M * 128)     │
        │ Cast(xHatUB, yFp32UB, fp32 → fp16)      │
        └─────────────────────────────────────────┘

        ┌─ 7. 写出（紧凑 workspace）─────────────┐
        │ DataCopy(dst_block_addr +               │
        │            tile_start * 128 * 2,        │
        │          xHatUB, M * 128)               │
        └─────────────────────────────────────────┘
```

关键点：

- **`gather_block_ids` 是 int32 GM 读取**：每个紧凑块查一次（不是每行），`total_blocks` 量级通常 < 1000，访存量微不足道。
- **K 段与 V 段紧凑块编址一致**：第 `i` 个紧凑块在 K 和 V 的物理块号是同一个 `gather_block_ids[i]`，因为 vllm-ascend KV cache 的 K/V 是按相同 block_table 索引存放。所以 V 段直接复用 K 段算出的 `phys`，不重读 `gather_block_ids`。
- **没有 padding 块进 kernel**：紧凑布局保证每个紧凑序号都对应真实引用块，**无 padding 计算浪费**。
- **dst 写出地址静态**：`compact_blk_id` 与 workspace 行号一一对应，`bt_compact[s, j] = block_offsets[s] + j` 在 host 侧生成（见 §2.6.6）。

#### 2.6.5 Cube/Vector 融合机制

**结论先放**：本算子使用 **AscendC KFC（Kernel Function Call）混合编程模型**，1 个 AIC + 2 个 AIV 协同，把 `y_hat = codebook[idx] * norm`（Vector）与 `x_hat = y_hat @ R`（Cube）融合成单核内的流水。这与 `csrc/turboquant_pack_kv_for_cache_fused/` 已验证的 pack 算子完全同构（同样 `(M, 128) × (128, 128)` GEMM），可直接套用其 `REGIST_MATMUL_OBJ` 模式。

##### 2.6.5.1 为什么需要融合，而不是 Cube/Vector 各自独立

如果不融合，分两个 kernel：

```
kernel_1 (AIV-only):
  load packed → idx + norm → codebook[idx] → ×norm → 写 y_hat 到 GM workspace

kernel_2 (AIC-only):
  load y_hat from GM workspace → @R → 写 x_hat 到 GM
```

代价：
- y_hat 在 GM 上往返一次，增加 `M * 128 * 2` 字节 HBM 流量（M 大时可观）；
- 两个 kernel launch + workspace 分配；
- AIC 和 AIV 之间靠 stream 同步，无法 overlap。

KFC 融合后：
- y_hat 留在 UB（VECOUT 位置），AIC 直接通过 KFC 通道拉取，**不落 GM**；
- AIC 可以在 AIV 准备下一 tile 时并行计算前一 tile 的 GEMM；
- 单 launch。

##### 2.6.5.2 KFC 模型本质

KFC = **AIC 把自己注册成 AIV 可调用的 "matmul 协处理器"**。代码层面：

```cpp
using TqRotateAT   = MatmulType<TPosition::VECOUT, CubeFormat::ND, half>;   // A 在 VECOUT UB
using TqRotateBT   = MatmulType<TPosition::GM,     CubeFormat::ND, half>;   // B 在 GM (rotation)
using TqRotateCT   = MatmulType<TPosition::GM,     CubeFormat::ND, float>;  // C 落 GM (workspace)
using TqRotateBiasT= MatmulType<TPosition::GM,     CubeFormat::ND, half>;
using TqRotateMatmulOp = AscendC::Matmul<TqRotateAT, TqRotateBT, TqRotateCT, TqRotateBiasT>;

// kernel 入口
KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);   // 声明 1 AIC + 2 AIV 协同
AscendC::SetSysWorkspace(workspace);
AscendC::TPipe pipe;
TqRotateMatmulOp rotateMm;
TCubeTiling cubeTiling = tilingData.cubeTiling;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), rotateMm, &cubeTiling);
//                                              ↑
//                          建立 AIV ↔ AIC 的请求/响应通道
```

`REGIST_MATMUL_OBJ` 在 AIC 侧启动事件循环（接收 AIV 的 matmul 调用请求），在 AIV 侧把 `rotateMm` 变成可调用句柄。这是与 `csrc/turboquant_pack_kv_for_cache_fused/op_kernel/...cpp:600-602` **完全相同的模板**。

##### 2.6.5.3 融合点 1：codebook 查表（Vector）→ y_hat 留在 VECOUT

```
AIV 侧（每个 tile）：
  ① DataCopy: packed (130 B/行) GM → UB (MTE2)
  ② idx + norm 切片：UB 内 view，无 op
  ③ Gather LUT：codebook[256] 已常驻 UB，对 idx 做查表
       y_hatUB[i, j] = codebook[idx[i, j]]
       —— 用 AscendC 的 Gather 指令实现：
          Gather(dst=y_hatUB[i*128 + ...],  src=codebook,
                 offset=idx[i*128 + ...]*sizeof(half),  count=128)
       —— 这是 256 项的 LUT，远小于 Vector 单元 LUT 容量
  ④ Mul broadcast：y_hatUB[i] *= normUB[i]
       AscendC::Mul / Muls，沿 dim=1 broadcast
  ⑤ 同步标记：y_hatUB 准备好供 Cube 读取
       PipeBarrier<PIPE_V>;  // 确保 ④ 完成
       SetFlag<HardEvent::V_M>;  // 通知 Cube 可读
```

**关键点**：y_hatUB 的 TPosition 是 `VECOUT`——这是 AscendC 里"可被 Cube 通过 KFC 读取的 UB 区域"的专用位置标识。Vector 写完之后**不需要 DataCopy 到 GM**，Cube 通过 `SetTensorA(y_hatUB)` 直接读。

##### 2.6.5.4 融合点 2：Cube 内部把 R 加载到 L1（stationary）

```
AIC 侧（接收到 AIV 的 matmul 请求后）：
  rotateMm.SetOrgShape(M, 128, 128);
  rotateMm.SetSingleShape(M, 128, 128);
  rotateMm.SetTensorA(y_hatUB, /*isTransposed=*/false);  // 从 VECOUT 读
  rotateMm.SetTensorB(rotationGm, false);                 // 从 GM 读

  while (rotateMm.Iterate()) {
      rotateMm.GetTensorC(cubeCGm);                       // 落 fp32 GM workspace
  }
  rotateMm.End();
```

**R 的 L1 stationary 行为**：
- 第一次 `SetTensorB(rotationGm)` + `Iterate` 时，Cube 子系统把 R 从 GM 拉到 L1（一次 32 KB MTE2 → L1）；
- 后续 tile（不论 K 段还是 V 段）只要 `cubeTiling.depthB1` 配置不变，R 就**留在 L1 不重 load**——这是 Cube 子系统内部的 L1 cache 行为，**对 AIV 透明**；
- KFC 模型下，AIC 串行处理多个 tile 请求时 R 命中 L1，是 N 次 tile 解码不会被 R 加载拖慢的关键。

如果 R 已经预先 `npu_format_cast` 成 NZ 格式，可以省一次 ND→NZ 转换，但本期保持 ND 与 pack 算子一致。

##### 2.6.5.5 融合点 3：Cube 输出 fp32 → AIV 转 fp16 → 写出

```
AIV 侧（在 Cube 完成 tile k 之后，准备 tile k+1 的同时）：
  WaitFlag<HardEvent::M_V>;  // 等 Cube 完成
  DataCopy(yFp32UB, cubeCGm, M * 128);   // GM→UB MTE2
  TqSyncMte2ToV();
  Cast(xHatUB, yFp32UB, RoundMode::CAST_NONE, M * 128);  // fp32 → fp16
  PipeBarrier<PIPE_V>;
  DataCopy(dst_out_gm[base*128], xHatUB, M * 128);       // UB→GM MTE3
```

为什么 Cube C 走 fp32 GM 而不是 fp16 VECIN：
- **fp32 累积更精**：fp16 GEMM 在 K=128 时累积误差可达 1e-3 量级，fp32 累积保守很多。pack 算子（pack_kv_for_cache_fused）也用同一选择（`csrc/turboquant_pack_kv_for_cache_fused/op_kernel/...cpp:141`）。
- **CubeFormat::ND 对 VECIN 限制**：910B 上 Cube 直写 VECIN（UB）只支持 NZ 格式，ND 必须经 GM 中转。继续保持 ND 是为了避免 NZ ↔ ND 来回换格式的开销。

代价是一次 GM→UB MTE2 的 16 KB 流量，但相对 N 次 tile 处理来说，对总 HBM 流量的影响 < 5%。

##### 2.6.5.6 K 段与 V 段的流水

```
时间轴 →

AIV1: [load Kp0][LUT Kp0][×norm Kp0]       [load Kp1][LUT Kp1][×norm Kp1]   ...
AIV2:                    [Cast K0][写 K0]                [Cast K1][写 K1]    ...
AIC :                    [GEMM K0]                        [GEMM K1]          ...

—— K 段全部 tile 处理完后：
     无需重新 REGIST_MATMUL_OBJ；rotateMm 实例继续用，R 已在 L1。

AIV1: [load Vp0][LUT Vp0][×norm Vp0]       [load Vp1][LUT Vp1][×norm Vp1]   ...
AIV2:                    [Cast V0][写 V0]                [Cast V1][写 V1]    ...
AIC :                    [GEMM V0]                        [GEMM V1]          ...
```

- **AIV1 / AIV2 分工**：参考 pack 算子的 `TqIsPrimaryAivSub()`，2 个 AIV 子核分担 ① load+LUT+×norm 和 ⑥ Cast+写出 两个阶段，让 Cube 阶段不阻塞 AIV。
- **K→V 切换零开销**：rotateMm、cubeTiling、L1 上的 R、UB 上的 codebook 全部复用，只切换 GM 基址 `src_packed_gm` 和 `dst_out_gm`，相当于一次指针更新。
- **stream level 的好处**：原本 K/V 两次独立调用（如果不融合）需要 2 次 KFC handshake + 2 次 R 上 L1，融合后**只做一次**。

##### 2.6.5.7 与 pack 算子的复用矩阵

| 子机制 | pack 算子 | decode 算子（本期） | 复用方式 |
|---|---|---|---|
| KFC 注册 (`REGIST_MATMUL_OBJ`) | ✅ | ✅ | 同模板 |
| MatmulType 五元组（`VECOUT/GM/GM/GM`）| ✅ | ✅ | 完全相同 |
| `KERNEL_TYPE_MIX_AIC_1_2` | ✅ | ✅ | 同 |
| Cube C → fp32 GM workspace → AIV Cast | ✅ | ✅ | 同 pattern |
| Vector NormalizeBatch | ✅ | 不需要（norm 已在 packed 末 2 字节）| 不复用 |
| Cube `(M, 128) × (128, 128)` rotate | ✅ | ✅ | **本算子可直接复用 pack 算子的 cubeTiling 参数** |
| Vector LUT 查表 | ❌（pack 算子做 quantize argmin）| ✅ | 新增 Gather |
| Vector ×norm broadcast | ❌ | ✅ | 新增 |

意思是：**本算子的 Cube 部分和 pack 算子完全同构，开发只需新增 LUT 查表 + ×norm 这两步 Vector 逻辑**，Cube 配置一行不改。这极大降低了实现风险。

##### 2.6.5.8 Cube vs 纯 Vector 路径并存

为方便调试与小规模回归，本算子额外提供一条 **AIV-only 参考路径**（`tiling_key=1`）：

```cpp
if (TILING_KEY_IS(0)) {
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    /* KFC + Cube 路径（性能版）*/
} else if (TILING_KEY_IS(1)) {
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
    /* AIV-only 标量 GEMM（参考版，与 PyTorch 对拍用）*/
}
```

参考路径用 `RotateBatchMatmulManual`（与 pack 算子的同名函数完全同形）做标量 `y @ R`。这条路径不进生产，仅用于：
- 数值对拍 golden（确认 AIV 部分的 LUT + ×norm 正确）；
- KFC handshake 出问题时的兜底（Cube driver bug 时可临时切回）；
- 单核 profiling 对照（去掉 Cube 看 pure Vector 流水的瓶颈）。

Python 侧通过 env 切换：
- `VLLM_ASCEND_TURBOQUANT_DECODE_OP_8BIT_MODE=0`（默认）→ Cube 融合
- `VLLM_ASCEND_TURBOQUANT_DECODE_OP_8BIT_MODE=1` → AIV-only 参考

#### 2.6.6 OpDef 注册

参考 `csrc/turboquant_pack_kv_for_cache_fused/op_host/turboquant_pack_kv_for_cache_fused_def.cpp`：

```cpp
this->Input("key_cache")        .DataType({ge::DT_UINT8})  .Format({ge::FORMAT_ND}).AutoContiguous();
this->Input("value_cache")      .DataType({ge::DT_UINT8})  .Format({ge::FORMAT_ND}).AutoContiguous();
this->Input("gather_block_ids") .DataType({ge::DT_INT32})  .Format({ge::FORMAT_ND}).AutoContiguous();
this->Input("codebook")         .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();
this->Input("rotation")         .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();

this->Output("key_out")         .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();
this->Output("value_out")       .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();

this->Attr("head_size").Int();
this->Attr("block_size").Int();
this->Attr("out_dtype").Int();

OpAICoreConfig aicoreConfig;
aicoreConfig.DynamicCompileStaticFlag(true)
    .DynamicShapeSupportFlag(true)
    .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
    .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
this->AICore().AddConfig("ascend910_93", aicoreConfig);
this->AICore().AddConfig("ascend910b", aicoreConfig);
```

#### 2.6.7 Python/torch 绑定与门控

新增 torch op：`torch.ops._C_ascend.turboquant_decode_paged_8bit`。

替换 `vllm_ascend/ops/turboquant_kv_cache.py:710` 的 `turboquant_decode_kv_cache_compact`，新逻辑只在 8-bit 路径启用，其它路径继续走旧版。

```python
def _try_8bit_decode_paged(key_cache, value_cache, block_table,
                           actual_seq_lengths_kv, head_size, dtype):
    """8-bit packed → fp16 via fused decode op.

    Returns (key_ws, value_ws, bt_compact) on hit, None on miss.
    """
    if not envs_ascend.VLLM_ASCEND_TURBOQUANT_DECODE_OP_8BIT:
        return None
    if not _c_ascend_turboquant_op_available("turboquant_decode_paged_8bit"):
        return None
    if dtype != torch.float16 or head_size != 128:
        return None
    if int(key_cache.shape[-1]) != head_size + 2:
        return None  # mixed/padded slot 不命中

    block_size = int(key_cache.shape[1])
    bs = block_size
    batch, max_bps = block_table.shape
    device = key_cache.device

    # ---- host 侧轻量预处理（消除 unique/searchsorted 路径）----
    # 1. 每个 seq 用几块
    actual_kv = actual_seq_lengths_kv if isinstance(actual_seq_lengths_kv, torch.Tensor) \
        else torch.tensor(actual_seq_lengths_kv, dtype=torch.int32, device=device)
    num_blocks_per_seq = (actual_kv + bs - 1) // bs                      # [batch] int32
    # 2. 紧凑起点
    block_offsets = num_blocks_per_seq.cumsum(0) - num_blocks_per_seq    # [batch]
    # 3. 紧凑总块数（一次 D2H 用作 workspace 行数）
    total_blocks = int(num_blocks_per_seq.sum().item())
    if total_blocks == 0:
        empty = torch.empty((0,) + tuple(key_cache.shape[1:-1]) + (head_size,),
                            dtype=dtype, device=device)
        bt_compact = torch.zeros_like(block_table)
        return empty, empty, bt_compact
    # 4. gather_block_ids[i] = block_table[seq_of(i), j_of(i)]
    arange_j = torch.arange(max_bps, device=device, dtype=torch.int32)
    j_mask = arange_j.unsqueeze(0) < num_blocks_per_seq.unsqueeze(1)     # [batch, max_bps]
    gather_block_ids = block_table[j_mask].to(torch.int32)               # [total_blocks]
    # 5. bt_compact: valid 位置 = block_offsets[s]+j；padding 位置压回 block_offsets[s]
    bt_compact = block_offsets.unsqueeze(1) + arange_j.unsqueeze(0)      # [batch, max_bps]
    bt_compact = torch.where(j_mask, bt_compact, block_offsets.unsqueeze(1))
    bt_compact = bt_compact.to(block_table.dtype)

    # ---- 算子调用 ----
    quantizer = _get_quantizer(head_size, 8, str(device), _current_mse_impl())
    if quantizer._codebook_fp16 is None or quantizer._codebook_fp16.device != device:
        quantizer._codebook_fp16 = quantizer.codebook.to(device=device, dtype=torch.float16)
    if quantizer._rotation_fp16 is None or quantizer._rotation_fp16.device != device:
        quantizer._rotation_fp16 = quantizer.rotation.to(device=device, dtype=torch.float16)

    key_ws, value_ws = torch.ops._C_ascend.turboquant_decode_paged_8bit(
        key_cache, value_cache, gather_block_ids,
        quantizer._codebook_fp16, quantizer._rotation_fp16,
        head_size, block_size, 0,        # out_dtype=fp16
    )
    return key_ws, value_ws, bt_compact
```

挂入主流程（替换原 `turboquant_decode_kv_cache_compact` 调用点）：

```python
fast = _try_8bit_decode_paged(
    self.key_cache, self.value_cache, block_table,
    actual_seq_lengths_kv, head_size=self.head_size, dtype=query.dtype,
)
if fast is not None:
    key, value, block_table = fast
    # 直接走 npu_fused_infer_attention_score
else:
    # 回落：旧版 turboquant_decode_kv_cache_compact + FIA 路径
    key, value, block_table = turboquant_decode_kv_cache_compact(...)
```

环境变量（在 `vllm_ascend/envs.py` 注册）：

```python
"VLLM_ASCEND_TURBOQUANT_DECODE_OP_8BIT":
    lambda: bool(int(os.getenv("VLLM_ASCEND_TURBOQUANT_DECODE_OP_8BIT", "0"))),
"VLLM_ASCEND_TURBOQUANT_DECODE_OP_8BIT_MODE":
    lambda: int(os.getenv("VLLM_ASCEND_TURBOQUANT_DECODE_OP_8BIT_MODE", "0")),  # 0=KFC, 1=AIV-only
```

**默认关**，与 0.2 章 `VLLM_ASCEND_TURBOQUANT_FUSED_FIA_8BIT` 一致的灰度策略。两道门：env + op 可用性，任一不满足回落旧路径。

> 注：与 4-bit `VLLM_ASCEND_TURBOQUANT_DECODE_OP` 共用一个 env 的诱惑要拒绝——4-bit 路径的正确性未验证，不能用一个开关同时控制两条不可信程度不同的路径。

#### 2.6.8 数值正确性验证

**Golden = `turboquant_dequantize_from_packed_bytes`**（`turboquant_kv_cache.py:532`，纯 PyTorch 8-bit 解码）+ host 侧 prefix-sum 等价实现。

UT 文件位置：`tests/ut/ops/test_turboquant_decode_paged_8bit_op.py`（新增），与 `tests/ut/ops/test_turboquant_kv_cache_perf.py` 风格对齐。

| 用例 | 构造 | 校验 |
|---|---|---|
| **D1 单 seq 单块** | `batch=1, num_blocks_per_seq=[1]` | `key_out[0]` vs PyTorch `dequantize(key_cache[bt[0,0]])`，`allclose(atol=1e-3, rtol=1e-3)` |
| **D2 多 seq 紧凑** | `batch=3, num_blocks=[2,1,3], max_bps=4` | 验证 `total_blocks=6`、`gather_block_ids` 顺序、`bt_compact = [[0,1,0,0],[2,2,2,2],[3,4,5,3]]` 与 §2.6.2 示例一致 |
| **D3 跨 tile 边界** | `total_blocks * BS * H` 跨多个 `T_rows` 边界（如 33 / 65 / 1024 行）| 解码内容与 PyTorch 逐块对拍 |
| **D4 null_block 共享** | `block_table` 含重复的物理块 0 | 紧凑 workspace 中对应位置都是 null_block 解码（≈0），验证不报错且数值正确 |
| **D5 padding 压回安全** | seq0 `actual_kv=200`（用 2 块），但 `max_bps=4`，padding 槽 `bt_compact[0,2:]=block_offsets[0]=0` | FIA 即便 prefetch 读 `bt_compact[0,2]`，落在 workspace[0]（seq0 的真实块 0），数值合法 |
| **D6 极值 norm** | norm = 65504(fp16 max) / 0 / nan / inf | nan/inf 由调用方 `nan_to_num` 兜底；算子本身不做 sanity，让数值如实穿透 |
| **D7 不命中条件** | `dtype=bf16` / `head_size=64` / `slot_w=132` | `_try_8bit_decode_paged` 返回 None，回落旧路径 |
| **D8 与 unique 路径对拍** | 同 batch/block_table，分别走方案 X 与原 unique 路径 | attention output `allclose(rtol=2e-2, atol=0.2)`（fp16 累积误差）|
| **D9 端到端** | `forward_fused_infer_attention` 全链路 vs PyTorch 路径 | 同 D8 标准 |
| **D10 性能（NPU only）** | 在 `tests/ut/ops/test_turboquant_kv_cache_perf.py` 内加 `decode_paged_8bit_ms` 列 | 期望比 unique 路径快 2~4×（不作为通过条件，仅记录） |

**误差容忍说明**：解码段是 Vector Mul + Cube fp32 累积 GEMM + Cast fp16，与 PyTorch 的 fp32 中间 + fp16 输出存在累积差异。`atol=1e-3, rtol=1e-3` 的 per-element 容忍是对单块解码的，对端到端 attention output 放宽到 D8/D9 的水平。

**bt_compact 验证要点**：D2 的具体值（`[[0,1,0,0],[2,2,2,2],[3,4,5,3]]`）必须严格相等，因为它直接决定 FIA 寻址；padding 位置错位会导致 prefetch 读到别的 seq 的 KV，数值未必立刻报错但属于隐患。

#### 2.6.9 与 Phase 2 的复用契约

为让 Phase 2 方案 B 能复用 Phase 1 的解码前端，把核心计算抽成 **device 函数**（不是 kernel），写在 `decode_device.h` 里：

```cpp
// csrc/turboquant_decode_paged_8bit/op_kernel/decode_device.h
namespace turboquant {

// 解码 M 行 packed 到 fp16（VECOUT），要求：
// - codebook 已在 UB（caller 传 LocalTensor）
// - rotation 已在 GM，且 rotateMm 已 REGIST_MATMUL_OBJ
// - packed 已在 UB（含 idx 段和 norm 段）
// - 输出 xHat 在 VECOUT UB（caller 提供）
__aicore__ inline void DecodeRows8bit(
    const AscendC::LocalTensor<uint8_t>& packed,    // [M, 130] UB
    const AscendC::LocalTensor<half>& codebook,     // [256] UB
    AscendC::GlobalTensor<half>& rotationGm,        // [128, 128] GM
    AscendC::GlobalTensor<float>& cubeCGm,          // workspace fp32 [M, 128]
    TqRotateMatmulOp& rotateMm,                     // 已 REGIST 的 KFC handle
    AscendC::LocalTensor<half>& xHat,               // [M, 128] VECOUT UB out
    uint32_t M);

}  // namespace turboquant
```

Phase 1 的 kernel 入口做：
- GM↔UB 搬运（packed/output）
- 紧凑块寻址（`gather_block_ids`）
- 调用 `DecodeRows8bit` 完成解码

Phase 2 方案 B 的 kernel：
- 在 attention 主循环里**对每个 KV tile 直接调** `DecodeRows8bit`
- **不再写 workspace**，xHat 留在 VECOUT 直接喂给 attention 的 `Q·K`/`P·V` Cube

这是设计文档 §3.5 "A→B 切换成本中等"的具体落点。**方案 X 已经把 block_table 寻址折进 kernel，Phase 2 方案 B 在此基础上的剩余工作只是"把 decode 结果留在片上不落 GM" + "FlashAttention 主循环"**——比原计划少一步（host compaction 已经消除）。

#### 2.6.10 落地顺序与里程碑

| 步骤 | 产出 | 验证 |
|---|---|---|
| **S1** OpDef + Tiling 骨架 + 空 kernel（按 `gather_block_ids` 搬运 packed→out 的 idx 字段，不解码）| 编译过 + aclnn.h 生成 + host 调用不崩 + 紧凑寻址正确 | python 侧把 packed idx 字节直接读出来对拍 |
| **S2** 加 codebook 查表 + ×norm（无 rotation）| `x_hat = codebook[idx] * norm` 紧凑写出 | 与 PyTorch `codebook[idx] * norm`（跳过 @R）数值对拍，UT D1/D2 通过 |
| **S3** 加 Cube `y_hat @ R`（KFC）| 完整解码 | UT D1~D7 通过 |
| **S4** 双缓冲 + tile 切分 + AIV1/AIV2 分工调优 | 性能版 | UT D10 性能记录 |
| **S5** 抽出 `DecodeRows8bit` device 函数到 `decode_device.h` | header 复用契约 | 与 Phase 2 框架联调 |
| **S6** Python 侧 `_try_8bit_decode_paged` + env 开关 + 主流程接入 | 端到端可用 | UT D8/D9 通过 |
| **S7**（可选）AIV-only 参考路径 `tiling_key=1` | 调试兜底 | 与 KFC 路径数值对拍 |

**S1 → S2 是关键节点**：S1 通过证明紧凑寻址 + GM 搬运链路正确（idx 字段一致），S2 引入 LUT 出问题时就只可能在 codebook 查表上，定位精确。S3 引入 Cube 时只剩 R 的加载与对齐一个变量。

#### 2.6.11 与设计文档其它章节的关系

- **§1.3** 列出"如何支持"中的"新增 8-bit compact 解码自定义算子" + "DecodeOnly arange 快路径"——本节 2.6 把两件事合并到一个算子内（block_table 寻址折进 kernel），同时解决两个问题。
- **§2.5** 提到的"无可信模板"——通过 2.6.8 的 D2 (紧凑布局) + D5 (padding 安全) + D8 (与 unique 路径对拍) + D9 (端到端) 四层验证补足。
- **§3.5** "A→B 切换成本中等"——方案 X 把 host compaction 已消除，Phase 2 方案 B 只剩 "decode 不落 GM" + "FlashAttention 主循环"，2.6.9 的 `DecodeRows8bit` 是承接点。
- **§4 Phase 1**（开发计划）——本节是 Phase 1 的"如何做"，§4 是"做什么"的清单。
- **arange / unique 路径的命运**：方案 X 接入后，原有 `turboquant_decode_kv_cache_compact` 的两条路径（unique 与 `decode_only_arange_fast_path`）都被替代，仅作为 env 关闭时的回落保留。`decode_only_arange_fast_path` 在 PIECEWISE 下的 HBM 放大问题被方案 X 自然消除。

---

## 3. 方案 B —— 片上真正融合

### 3.1 思路

单个 AscendC kernel 内完成"**解码内联 + FlashAttention**"，KV 不落 GM、不做 host compaction（kernel 内用 **原始 block_table** 寻址，与原生 paged attention 一致）。

```
[B] for tile in kv(按 block_size, via 原始 block_table):
        gather packed K/V tile → 片上解码(查表 + y_hat@R(Cube) + ×norm) → K_tile/V_tile(fp16)
        S = Q·K_tileᵀ * scale → 施 atten_mask/因果 → 在线 softmax → acc += P·V_tile
    O = acc / l
```

### 3.2 PyTorch → kernel 阶段映射

| PyTorch 参考 | 方案 B kernel 阶段 |
|---|---|
| `block_table` 寻址 | kernel 内 MTE gather（原始页表，**无 compaction**）|
| `codebook[idx] * norm` | tile 内查表 + 向量乘（codebook 常驻 UB）|
| `y_hat @ R` | Cube 矩阵乘（R 常驻 L1，stationary）|
| `Q·Kᵀ * scale` | Cube + scale |
| `softmax`（因果/mask）| 在线 softmax + **应用 atten_mask** |
| `P·V` | Cube，在线累加 |

### 3.3 相对方案 A 的增量收益

- **消除 block 压缩重映射**（④⑤⑥⑧ 全无），恢复对 ACL Graph 友好的静态 shape。
- **消除 decoded KV 的 HBM 落盘与回读**（方案 A 仍有）。

### 3.4 当前 kernel 现状与差距

现有 `csrc/turboquant_fused_infer_attention_score8bit/op_kernel/...cpp` 是 **correctness-first 标量基线**，差距：
- 纯标量、`GetValue/SetValue` 逐元素，无 Cube、无向量化、无双缓冲 → 性能不可用；
- 只对 **DecodeOnly** 正确（`lastPos=kvLen-1`，单 query token）；
- `(void)attenMask` 丢弃掩码；
- 无 env 开关，命中门未收窄。

方案 B 需在此基础上重写为 Cube + 向量化在线 softmax，并补齐多 query token 的 per-token 因果掩码。

### 3.5 A→B 切换成本

**中等，且前置去风险**。A 与 B 共享"解码前端"（packed 布局解析、codebook UB、R 的 Cube 乘、norm 解析、host/tiling/build 脚手架、对拍 golden）。A→B 丢弃的是"写 workspace + 单独调 FIA"，新增的是 kernel 内 FlashAttention 后端（标准但需正确数值 + 掩码）。前提：**方案 A 的解码必须写成可复用 device 函数**，而非与写回揉死。

---

## 4. 开发计划

### Phase 0 —— 纯 PyTorch 去风险（低风险，先做）
- [ ] 0.1 命中门收窄到 `attn_state == DecodeOnly`（`attention_v1.py:827`），把多 token 状态挡在融合算子外。
- [ ] 0.2 新增 env 开关（如 `VLLM_ASCEND_TURBOQUANT_FUSED_FIA_8BIT`，默认关）+ `_c_ascend_turboquant_op_available` 门控。
- [ ] 0.3 `turboquant_decode_kv_cache_compact` 增加 DecodeOnly `arange` 快路径（替换 `unique/searchsorted`）。
- [ ] 0.4 新增 8-bit compact 解码自定义算子（独立实现 + 数值验证；不复用未验证的 4-bit 算子）。
- [ ] 0.5 图模式 guard（见下「turboquant 与图模式兼容性」）。
- [ ] 0.6 SWA + turboquant：`_forward_fia_slidingwindow`（`attention_v1.py:767`）依赖永不被赋值的 `_decoded_key_cache` → 当前 **assert 崩溃**；要么实现该 cache，要么对 `turboquant + sliding_window` 显式报错不支持。
- 交付：纯 PyTorch 路径在 DecodeOnly 下更快且图友好，融合算子安全 opt-in，图模式/SWA 边界不再出错。

#### turboquant 与图模式兼容性（Phase 0 必修）

> 背景原理见 `fx_and_acl_graph.md`。结论:**turboquant 只能在 PIECEWISE 或 NONE 下正确运行,不能进 FULL / FULL_DECODE_ONLY。**

| 图模式 | 注意力 | turboquant 正确性 | 说明 |
|---|---|---|---|
| `NONE`（全 eager）| eager | ✅ | 最稳 |
| **`PIECEWISE`（默认）** | eager | ✅ | 注意力 eager → `decode_compact` 动态 shape 在 eager 下合法；piece 不碰 KV cache。**推荐** |
| `FULL_DECODE_ONLY` | 进图 | ❌ | decode 走 `full_graph_pa`（`attention_v1.py:663`）**不解码 packed cache** → 输出错；compact 动态 shape 也不可捕获 |
| `FULL` | 进图 | ❌ | 同上，`full_graph_fia`（`attention_v1.py:570`）不解码 |

两条根因（均见 `fx_and_acl_graph.md` §7）：
1. `turboquant_decode_kv_cache_compact` 的 `unique()` / `bt[valid]` 是**数据相关 shape**，触发 graph break / 不可捕获。
2. 捕获路径 `full_graph_fia` / `full_graph_pa` **没有 turboquant 解码分支**，会直接在 packed uint8 cache 上算注意力 → 结果错误。

**开发项（Phase 0.5）**：
- [ ] 在 `platform.py` 配置阶段加 guard：当 `kv_cache_dtype == "turboquant"` 且 `cudagraph_mode in {FULL, FULL_DECODE_ONLY}` 时，**强制降级为 `PIECEWISE` 并 `logger.warning`**（默认 PIECEWISE 本就安全，仅拦截用户显式开 full 的情况）。
- [ ] （可选加固）在 `full_graph_fia` / `full_graph_pa` 入口对 `kv_cache_dtype == "turboquant"` 直接 `raise`，作为第二道防线。
- [ ] UT：见测试用例 G1（turboquant 下 full 模式被拦截/降级，piecewise 数值正确）。

> 方案 B（片上寻址、静态 shape、kernel 内解码）的目标之一，就是让 turboquant 也能进 **FULL** 图——届时注意力可被捕获、host 开销彻底隐藏，本节 guard 可放开。

### Phase 1 —— 方案 A（解码算子化）
- [ ] 1.1 实现/复用 8-bit 解码 AscendC 算子：Cube 做 `y_hat@R`，codebook UB、R L1。
- [ ] 1.2 Python 侧改为 "解码算子 → FIA"，解码前端封装为可复用 device 函数。
- [ ] 1.3 数值对拍 + profiling（解码阶段、端到端）。
- 交付：DecodeOnly（及可选全状态）下解码加速，FIA 复用。

### Phase 2 —— 方案 B（片上融合，DecodeOnly）
- [ ] 2.1 重写 kernel：tile gather（原始 block_table）+ 片上解码 + Cube QKᵀ/PV + 向量化在线 softmax。
- [ ] 2.2 复用 Phase 1 解码前端 device 函数。
- [ ] 2.3 与方案 A / 纯 PyTorch 三方数值对拍 + profiling。
- 交付：DecodeOnly 无 compaction、无解码物化的真正融合。

### Phase 3 —— 方案 B 扩展状态
- [ ] 3.1 支持多 query token 的 per-token 因果掩码（PrefillCacheHit / SpecDecoding）。
- [ ] 3.2 支持 ChunkedPrefill 的 decode/prefill 混合（参考 `_forward_c8_chunked_prefill` 拆段思路，`attention_v1.py:1277`）。
- [ ] 3.3 放开命中门到对应状态。

### Phase 4 —— 4-bit（仅在 8-bit 全部能力 OK 后启动）
- [ ] 4.1 在 8-bit 方案 A/B 验证通过后，评估是否将经验推广到 4-bit（含先验证现有 `turboquant_decode_packed_blocks_compact` 的正确性）。
- 说明：**4-bit 不在本期范围**，此 Phase 仅为占位，待 8-bit 收尾后单独立项。

> 依赖关系：Phase 0 可独立先行；Phase 1 是 Phase 2 的解码前端来源；Phase 3 依赖 Phase 2；**Phase 4（4-bit）依赖 Phase 0~3 全部完成**。

---

## 5. 测试用例设计

### 5.1 数值正确性（golden = 纯 PyTorch 参考）

参考实现：`_turboquant_fused_infer_attention_score_8bit_impl`（`turboquant_kv_cache.py:811`）与 `turboquant_dequantize_from_packed_bytes`。扩展 `tests/ut/attention/test_attention_v1.py`。

| 用例 | 输入构造 | 校验 |
|---|---|---|
| T1 解码算子对拍（A）| 随机 packed K/V cache + block_table（无共享）| 解码输出 vs PyTorch 解码，`allclose(atol/rtol 按 fp16)` |
| T2 融合输出对拍（A/B）| 随机 Q + packed cache + block_table，DecodeOnly | attn_output vs fallback |
| T3 跨序列块共享 | block_table 含重复物理块（模拟前缀命中）| 共享块解码一致；`arange` 路径与 `unique` 路径结果一致 |
| T4 padding/-1 | block_table 含 -1 尾部填充 | 无越界，结果与 golden 一致 |
| T5 kvLen 非块整除 | kvLen=300（最后块 44 行）| 因果界正确，尾块只用有效行 |
| T6 GQA | num_heads≠num_kv_heads | kv_head 分组正确 |
| T7 mixed/4-bit 不命中融合 | bits≠8 或 bits_key≠bits_value | **不进 8-bit 融合算子**，回落纯 PyTorch，数值对 |

### 5.2 状态门控

| 用例 | 校验 |
|---|---|
| S1 命中门 | DecodeOnly 命中融合算子；PrefillCacheHit/ChunkedPrefill/SpecDecoding 回落（Phase0 后）|
| S2 env 开关 | 开关关闭时走纯 PyTorch；打开且算子存在时走融合 |
| S3 算子缺失回落 | `_C_ascend` 无该 op 时 `turboquant_fused_infer_attention_score_8bit` 正确回落 fallback |
| S4 dtype/head_size 边界 | 非 fp16 / head_size≠128 / sinks≠None 时不命中融合 |

### 5.3 compact 等价性

| 用例 | 校验 |
|---|---|
| C1 arange vs unique | DecodeOnly 下两路径 `k,v,bt_compact` 等价（共享块时 arange 允许重复解码但 attn 结果一致）|
| C2 空/全 -1 block_table | 返回空 decoded，无异常 |

### 5.4 图捕获 / 端到端

| 用例 | 校验 |
|---|---|
| G1 图模式 guard（Phase 0）| turboquant + `FULL/FULL_DECODE_ONLY` 被强制降级为 `PIECEWISE` 并告警；turboquant + `PIECEWISE`（默认）数值正确 |
| G2 图捕获（方案 B 目标）| 方案 B 下 DecodeOnly 注意力可被 full graph 捕获（静态 shape，无数据相关 shape）|
| E1 端到端精度 | 小模型 8-bit turboquant 生成结果与非量化基线在可接受误差内（如 perplexity / 关键 token 一致率）|
| E2 性能回归 | profiling 记录解码阶段耗时、端到端吞吐，长上下文/小 batch 重点观测 |

### 5.5 测试矩阵建议

- bits：**仅 8（A/B 主线）**；4-bit 不在本期范围（仅验证"不命中 8-bit 融合、正确回落"）
- 状态：DecodeOnly（主线）、其余（回落或 Phase3）
- shape：head_size=128；num_heads∈{8,32}，num_kv_heads∈{1,8,32}（含 MQA/GQA）；kvLen∈{1,127,128,300,2048}
- dtype：fp16（融合主线）、bf16（纯 PyTorch）

---

## 6. 参考代码索引

| 主题 | 位置 |
|---|---|
| FIA + turboquant 分支 + 8bit 命中门 | `attention_v1.py:803-899` |
| `turboquant_decode_kv_cache_compact` | `turboquant_kv_cache.py:710-808` |
| 8-bit 融合封装 + fallback | `turboquant_kv_cache.py:811-953` |
| 4-bit compact 解码算子（**未验证，不参考**，仅本期范围外）| `turboquant_kv_cache.py:763-808` |
| 量化/解码语义 | `turboquant_kv_cache.py:336-403, 532-607` |
| AscendC kernel（待重写）| `csrc/turboquant_fused_infer_attention_score8bit/op_kernel/` |
| tiling | `csrc/turboquant_fused_infer_attention_score8bit/op_host/` |
| C8 chunked prefill 拆段参考 | `attention_v1.py:1277-1375` |
| UT | `tests/ut/attention/test_attention_v1.py` |
