# BitResidualAttentionPagedK8v4 设计文档

> AIV 主导的 paged-attention：在 BitResidual K8/V4 压缩 cache 上做 Decode / Prefill。
> Prefill（GQA≤2）可经 KFC 做 Cube QK / 条件 Cube PV。
>
> Cache 布局真源见 `../bit_residual_pack_k8v4/design.md`。
> FIA 流水线对照见 `../bit_residual_fia_paged_k8v4/DESIGN.md`。

## 1. 定位

| 项 | 值 |
|----|-----|
| 目录 | `csrc/bit_residual_attention_paged_k8v4/` |
| OpDef | `BitResidualAttentionPagedK8v4` |
| aclnn | `aclnnBitResidualAttentionPagedK8v4` |
| Torch | `torch.ops._C_ascend.bit_residual_attention_paged_k8v4` |
| Wrapper | `vllm_ascend.ops.turboquant_kv_cache.bit_residual_attention_paged_k8v4` |

**默认 Serving 路径**（`cache_dtype=turboquant` + `turboquant_kv_bits=[8,4]`）：

| `AscendAttentionState` | Op | 开关 |
|------------------------|----|------|
| `DecodeOnly` | **本算子** | `VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA=1` 时先试 FIA |
| PrefillCacheHit / ChunkedPrefill | **本算子** | `VLLM_ASCEND_BIT_RESIDUAL_FIA=1` 时先试 FIA |
| PrefillNoCache | stock FIA（float K/V） | 不读 KV cache |

两开关默认均为 `0`（`vllm_ascend/envs.py`）。

```text
pack_k8v4 → uint8 PA cache
query → LoadQ → Cube rotate (Q @ R_key) → × scale
      → tile loop:
           Load/Prefetch K → DecodeK → QK (Vector | Cube)  ∥  Prefetch V
           Finalize V → DecodeV
           Softmax+PV (Vector)  |  SoftmaxOnly + Cube PV   ∥  预取下一 tile Key
      → Normalize → Cube rotate (out @ R_value) → WriteOut
```

## 2. IO

**Inputs**

| 名 | dtype | shape / 说明 |
|----|-------|----------------|
| `query` | fp16 / bf16 | TND `[T, H, D]`，`D=128` |
| `key_cache` / `value_cache` | uint8 | `[num_blocks, num_kv_heads, packed_bytes]` |
| `block_table` | int32 | `[batch, maxBlocksPerSeq]` |
| `actual_seq_len_q` / `actual_seq_len_kv` | int64 | ValueDepend；前缀和风格 |
| `rotation_key` / `rotation_value` | 同 query | `[D, D]`；Python 侧 key=`R^T`，value=`R` |

**Output**

- `attention_out`：与 query 同 shape / dtype；Torch binding 支持可选 `out=` 原地写

**Attrs**（顺序）：`num_heads`, `num_kv_heads`, `head_size`, `block_size`,
`max_actual_seq_len`, `scale_value`

**硬约束**

- `head_size == 128`；`block_size % 16 == 0`
- `num_heads % num_kv_heads == 0`；GQA 按 `TQ_BR_ATTN_GQA_CAP=8` 分 chunk
- query / `rotation_key` / `rotation_value` **同 dtype**
- key last-dim = `(block_size/16)*2112`；value = `(block_size/16)*1088`
- `max_actual_seq_len` ≥ observed `max(actual_seq_len_kv)`
- Workspace：system ≥ 16MB；user 上限 128MB

## 3. Cache 布局

与 pack 共用——**整 block 独立行平面**（非「16-row 交错 sub-block 串」）：

```text
Key head:   [block_size × 128 codes]
            [block_size × 2 base]
            [block_size × 2 step]
            headStride = block_size × (128 + 4)

Value head: [block_size × 64 packed nibbles]
            [block_size × 2 vmin]
            [block_size × 2 vstep]
            headStride = block_size × (64 + 4)
```

每 16 行字节量：`TQ_BR_KEY_BLOCK_STRIDE=2112`、`TQ_BR_VAL_BLOCK_STRIDE=1088`
（tiling / pack 真源；`decode_device.h` 注释中的 2176/1152 为历史笔误，常量值为 2112/1088）。

反量化（kernel）：

```text
K: q7=code&0x7F, sign=code>>7
   err = base + q7*step
   decoded = err * (±1)

V: signed idx4 ∈ [-8, 7]
   Load/Finalize 将 +8*vstep 折入 vmin'
   decoded = vmin' + idx4_s * vstep
```

Load 路径：同 page 连续行复用 `blockId`；codes 按 ≤16 行一次 `DataCopyPad`；
base/step（或 vmin/vstep）各一次平面 `DataCopyPad`，Cast 后 **GetValue→栈数组**
再喂 Decode（meta 留 UB 曾乱码，已回退）。

## 4. Tiling Key 与调度

| Key | 常量 | 模式 | 典型场景 |
|-----|------|------|----------|
| `0` | `TQ_BR_ATTN_KEY_SPLITBN_VECTOR` | SplitBN + Vector QK/PV | 短 KV Decode；未进 qTile 的 Prefill |
| `1` | `TQ_BR_ATTN_KEY_SPLITBNS_VECTOR` | SplitBNS + Vector（FlashDecode） | 长 KV、小 Q（`1≤T≤32` 且代价模型 `P>1`） |
| `2` | `TQ_BR_ATTN_KEY_SPLITBN_QTILE` | SplitBN + qTile + Cube QK；**条件 Cube PV** | Prefill / ChunkedPrefill（`GQA≤2` 且 `T>usedCoreNum`） |

硬件：`KERNEL_TYPE_MIX_AIC_1_2`（每 MIX：1 AIC + 2 AIV）。  
Host：`parallelCoreNum = min(aicNum, 20)`，`blockDim = parallelCoreNum * mixBlockDim`；
闲置 MIX 在 `mixCoreIdx >= usedCoreNum` 后 return。

选择顺序（`GetTiling`）：**先 FD → 否则 SplitBN → 再叠 qTile**。
qTile 成功则 key=2，且 `qkPvMode = TQ_BR_ATTN_QKPV_CUBE`。

> `TQ_BR_ATTN_CUBE_MIN_G` / `TQ_BR_ATTN_CUBE_MIN_TILE` 在 tiling.h 中**未被引用**；
> 真门控见 §5。

### 4.1 SplitBN（key 0）

- 任务：`(token, kvHead, gqaChunk)`，`taskCount = T * numKvHeads * ceil(gqa/GQA_CAP)`
- `usedCoreNum = min(taskCount, parallelCoreNum)`（`formerCoreNum` / range 字段 host 写入、**device 未用**）
- Device：`workerIdx = mixCore*2+subIdx`，`workerNum = usedCoreNum*2`，task **round-robin**
- AIC entry 在 SplitBN 上直接 return；Cube 由 AIV 经共享 KFC 驱动

### 4.2 FlashDecode（key 1）

`IsFlashDecodeK8v4`：

1. `1 ≤ numTokens ≤ TQ_BR_FLASH_DECODE_MAX_Q_TOKENS`（32）
2. `PickFlashDecodeKvSplitPart(taskCount, maxActualSeqLen, parallelCoreNum) > 1`
3. 编译期 `TQ_BR_FORCE_DISABLE_FLASH_DECODE=1` 可强制关

代价模型：

```text
cost(P) = ceil(taskCount / (coreNum/P)) * ceil(maxKvLen / P)
约束：maxKvLen ≥ 1024，segment ≥ 512，P ≤ min(coreNum, maxKvLen/512)
```

调用时传入的长度参数为 **`maxActualSeqLen`**（tiling 字段名 `maxActualSeqLen`）。

流程：双 AIV 写 partial（`accumOut` + LSE）→ `SyncAll` → **仅 primary AIV**
（`subIdx==0`）`CombineFlashDecode`（Rotate 共用每 MIX 一个 KFC）。

探针：`tests/e2e/singlecard/xrx_k8v4_fd_cost_model.py`。

### 4.3 Prefill qTile（key 2）

```text
splitMode == SplitBN
&& gqaGroup ≤ TQ_BR_ATTN_QTILE_GQA_CAP（2）
&& usedCoreNum > 0
&& numTokens > usedCoreNum
```

Host 为每个 MIX 写 `[qTileTokenStart, qTileTokenEnd)`：默认按 token 均分；
有 ValueDepend seq 时按 **causal KV 工作量前缀和** 再平衡。

Device（`ProcessSplitBn` qTile 分支）：

- 两 AIV **共享同一 token 区间**，按 `kvHead % 2 == subIdx` 分工（不对半切 token）
- 同 seq 内连续 token 聚成 `qRows ≤ TQ_BR_UB_QTILE_CAP(16)`；`qRows>1` →
  `ComputeAttentionQTile`，否则回落单 token `ComputeAttention`
- Cube QK/PV 的 GM staging 按 **AIV slot** 隔离：`aivSlot = mixCore*2+subIdx`

## 5. Cube QK / Cube PV 门控

### 5.1 Cube QK

Host：qTileMode → `qkPvMode=1` + 分配 qk workspace。

Device：

```text
preferCubeQk = (qkPvMode_ == 1) && (qRows > 1)
```

`CubeQkQTile` 额外要求：`compactRows=qRows*gqa ≤ TQ_BR_QK_CUBE_MAX_M(32)`，
`mRows ≤ TQ_BR_QK_CUBE_MAX_N(64)`，`qkGmReady_` / `matmulReady_`，
workspace stride ≥ `D×64` half。

实现：K fp32→half → 行主序写入 per-slot GM →
`SetTensorA(Q)` / `SetTensorB(K, true)`（Cube 侧转置）→ `IterateAll` →
score half→float。失败回落 `VectorQkFloatPreScaled(Gqa2)`。

**Decode 永不开 Cube QK**（小 M 税使 Q=16 约 +13%，已回退）。
手工 AIV 16×16 vtranspose 已由 `SetTensorB(true)` 替代（Prefill 约 −30%）。

### 5.2 Cube PV（仅 Prefill qTile）

```text
tileFullForAllQ =
  gqaCount==2 && qRows≥4 && mRows≥32
  && compactRows≤32 && mRows≤64 && preferCubeQk
  && ∀q: ¬(pos < causalEnds[q] < pos+mRows)   // 禁止部分因果 tile
```

通过后：`OnlineSoftmaxOnlyTileFloatPreScaledGqa2Scalar`（更新 m/s，留下 P）→
`CubePvQTile`（P@V，V 行主序 staging，无转置）。失败则 GQA=2 双 head `Axpy` 回落。

Cube PV 占用 `CodeFloat` 时 **延后**下一 tile 的 Key-code prefetch（先 SoftmaxOnly+PV，再 Issue）。

## 6. UB 账本（≤192 KiB）

静态偏移布局（`bit_residual_attention_paged_k8v4.cpp`），合计约 **171 KiB**：

| Buffer | 约 Bytes | 用途 |
|--------|----------|------|
| PackedRaw | 2112 | K/V 平面临时 staging |
| CodeI16 | 16384 | codes / Cube P half |
| CodeFloat | 32768 | decode scratch；下一 tile Key codes；Cube KT/V stage |
| QGroupFloat | 4096 | Decode Q |
| Score | 2048 | Decode scores（stride=`KV_TILE_CAP`） |
| KBase/KStep/Vmin/Vstep | 256×4 | meta Cast 中转 |
| MState/SState | 32×2 | UB 态（热路径多用栈 scalar） |
| OutAcc | 4096 | Decode accum |
| Decoded | 32768 | K/V float（时分复用） |
| FloatScratch | 1024 | reduce / ExpScalar |
| Mask / ValMask | 256×2 | 常量掩码 |
| RotateWork | 32768 | KFC workspace；DecodeKey `signBits` |
| QTileQ / Score / OutAcc | 16384+8192+16384 | Prefill qTile（Cap=16×GQA2） |
| VStage | ≤8192（实约 4608） | V 预取：`[codes][vmin][vstep]` 连续平面 |

常量：`TQ_BR_UB_KV_TILE_CAP=64`，`TQ_BR_UB_GQA_CAP=8`，
`TQ_BR_UB_QTILE_CAP=16`，`TQ_BR_UB_QTILE_GQA_CAP=2`。

## 7. 热路径流水

### 7.1 Decode / 非 qTile（`ComputeAttention`）

```text
LoadQ → Cast → Rotate(Q@R_key) → ×scale
for KV tile:
  LoadK | 上一轮 Key 预取 Finalize → Prefetch V Issue
  DecodeK → Vector QK (Gqa2)
  Prefetch V Finalize → DecodeV
  Issue 下一 tile Key → Softmax+PV Scalar (Gqa2) → Finalize 该 Key
Normalize → [FD WritePartial | Rotate(out@R_value) → WriteOut]
```

### 7.2 Prefill qTile（`ComputeAttentionQTile`）

```text
LoadQ tile → Rotate → ×scale；预取 causalEnds[]
for KV tile (至 maxCausalKvEnd):
  LoadK | 上一轮 Key 预取 Finalize → Prefetch V Issue
  DecodeK → Cube QK (prefer) | Vector QK
  Prefetch V Finalize → DecodeV
  if tileFullForAllQ:
      SoftmaxOnly → CubePv | Axpy fallback
      then Issue 下一 tile Key   // CodeFloat 已释放
  else:
      Issue 下一 tile Key → Softmax+PV Scalar → …
  Finalize 该 Key
WriteFinalOutputQTile（×1/s → Rotate → Cast → GM）
```

### 7.3 `attention_device.h` 热路径实际调用

| 函数 | 场景 |
|------|------|
| `VectorQkFloatPreScaled` / `…Gqa2` | Decode；qTile Cube QK 失败回落 |
| `OnlineSoftmaxUpdateTileFloatPreScaledScalar` / `…Gqa2Scalar` | Decode；qTile Vector PV |
| `OnlineSoftmaxOnlyTileFloatPreScaledGqa2Scalar` | Prefill Cube PV 前 |

其余 LocalTensor-m/s、带 vNorm、未接本 kernel 的变体视为遗留，勿当热路径文档。

## 8. 已落地优化（摘要）

| 做法 | 说明 | 锚点 |
|------|------|------|
| V 预取与 DecodeK/QK 重叠 | V 的 MTE2 与当前 tile DecodeK+QK 并行；`VStage` 分平面 | `PrefetchPackedValueTileRows*` |
| 同 page 批量 Load K/V | ≤16 行 codes 一次 `DataCopyPad`；meta 分平面拷贝 | `LoadPackedKeyTileRows` 等 |
| Softmax 期间预取下一 tile Key | codes 先落到 `CodeFloat`，Softmax 后再 Finalize | `PrefetchPackedKeyCodes*` |
| FlashDecode 双 AIV | partial 双写；Combine 仅 primary AIV | `Process` / `CombineFlashDecode` |
| Prefill 按 kvHead 拆双 AIV | 两 AIV 共享同一 token 区间，避免因果前缀双读 | `ProcessSplitBn` qTile |
| Cube QK 按 AIV slot 隔离 staging | 双 AIV 可并行发 Matmul | `CubeQkQTile` |
| Prefill 条件 Cube PV | SoftmaxOnly + `CubePvQTile`；门控见 §5.2 | `ComputeAttentionQTile` |
| GQA=2 复用 K/V 行 | 同一 KV 行服务两个 Q head | `attention_device.h` |
| qTile 按 causal 工作量切核 | host 写 `[tokenStart, tokenEnd)` | tiling.cpp |
| 原地写 output | binding `out=`；同 buffer 跳过 copy | `torch_binding` / `attention_v1` |

## 9. 刻意不做 / 已回退（勿重复试）

对照基线（Prefill 已开 Cube PV）：long_query 上 **Decode Q=16 ≈2.23 ms**，
**Prefill Q=241 ≈6.77 ms**。下列尝试已实测或论证为负向/不可行。

### 9.1 微优化回退表

| 尝试了什么 | 结果 | 结论 |
|------------|------|------|
| codes+meta 合成一次 MTE2 / dst 对齐错误 | AICORE 异常 | 保持 codes / meta **分平面** bulk |
| Softmax/PV 用 `Brcb`+`BinaryRepeat` 广播 | Prefill **+14%** / 乱码 | 禁止盲 Brcb |
| Softmax 两 head 合并 V↔S、Exp 批处理 | Prefill **+1.3%** | 无净收益 |
| Softmax+PV 把 beta 先搬到栈再 Axpy | Decode **+8.2%** | 禁止「只搬栈」 |
| Decode 也走 SoftmaxOnly + Cube PV（`compactRows=2`） | Decode **+2.5%** | 小 M 上 Cube PV 税更大 |
| Prefill 部分因果 tile 仍走 Cube PV | Prefill **−0.1%** | 保持 `tileFullForAllQ` 门控 |
| qTile 容量 16→24 | 持平 | 保持 Cap=16 |
| Prefill「KV 外环」轻量版（未改调度） | Prefill **+1.3%** | 关键路径未少扫 KV |
| Prefill「KV 外环」+ 状态 GM spill | Prefill **+0.8%** | GM 税抵消少扫 |
| `SetTensorB(true)` 省掉手工 Kᵀ | Prefill **约 −30%**（2026-07-21 复测） | **已落地**；旧「持平」结论作废 |
| Decode 开 Cube QK | Q=16 约 **+13%** | Decode 保持 Vector QK |
| 盲目抬 FlashDecode 的 `MAX_Q_TOKENS` | Q≥32 收益 ≤1% | 禁止盲目抬 cap |
| DecodeK/V 合并 per-row barrier / 整 tile Mul+Add | **+1.8~1.9%** | 行向解压已非瓶颈 |
| Decode：Softmax 前用 `RotateWork` 提前 Issue Key | Decode **+0.3%** | 无收益 |
| Cube PV 期间改用 `VStage` 做 Key 预取 | Prefill **−0.6%** | SoftmaxOnly 窗口太短 |
| meta 留在 UB、去掉 GetValue→栈 | 假加速 + smoke **乱码** | 禁止未验证 offset Cast |
| Vector QK GQA2 减 PipeBarrier | **0.0%** | Prefill 主路径已是 Cube QK |
| Prefill 再把 QK ReduceSum 搬上 Cube | — | 已有 `CubeQkQTile` |
| DecodeKey 改走 Cube / 用 AIC MTE1 卸 MTE2 | — | 否决 |
| Softmax 按 W=2 分块 + 合批 Cube PV（K≤128） | Prefill **+0.6%** | 持平即停；勿再抬 W=4 |
| 跨 page 批量读 `blockId` 进 UB 再拷栈 | Decode **+1.7%** / Prefill **+2.5%** | 批 MTE2+拷栈税更大 |
| Prefill：Cube QK 的 `IterateAll` 期间并行 DecodeV | 不可落地 | `IterateAll` 占用 Decoded/CodeI16，无多余 UB |
| 短 KV 动态抬高 `kvTileRows` | 否 | UB 已是 CAP=64；短序列已用尾 tile `mRows` |
| 强制少核 `usedCoreNum=10`（仍按 qTile 内环扫 KV） | Prefill **+43.4%**（6.77→9.70 ms） | Softmax 串行；**永久关闭**少核线 |
| GM/UB 改 32B meta 槽再 Brcb 解压 | 不单开 | 与上表「整 tile Mul+Add / Softmax Brcb」同构；FIA 的 32B 槽只为 Cast 对齐，仍 GetValue→Duplicate |

### 9.2 为何不做「KV 外环」（先扫一遍 KV，内层滚多个 qTile）

当前 Prefill 按 **causal 工作量** 把 Q 分到约 20 个 MIX；long-query（Q≈241、
seqKv≈2k）下每个核大约只有 **1 个 qTile**。墙钟由「晚 token、causal 最长」的核
决定，这些核上 **没有**「跨 qTile 重复扫同一段 KV」可省。

若强行少核、拉大每核 token 区间，才会出现多 qTile 重扫——但 Softmax/Cube QK
并行度从 ~20 降到更少，墙钟 Softmax 近似按核数放大。实测只减核、不改算法时
Prefill **+43%**，说明 Softmax 税先到账，KV 外环回不来。旧轻量版 / GM spill
亦为 **+0.8%~+1.3%**。本 KPI **不要**再试少核或 KV 外环。

### 9.3 为何不做「改 meta 布局降 scalar」

pack/attn 已是整 block SoA 平面 + 批量 Cast；profile 上 MTE2 只占墙钟一小部分，
scalar 主要来自 Decode 的 `Duplicate`+`Axpy` 与 Softmax 侧 GetValue。FIA 的
32B meta 槽是 half→float Cast 对齐用的，解码仍 GetValue + Duplicate。再做
「UB 驻留 meta + Brcb 整 tile 解压」等于重做 §9.1 已回退项；只改 GM 宽度增
HBM、不消主因。

### 9.4 Serving / host（非本算子墙钟）

`async_scheduling=True` 在本场景 e2e 约 **−7~−9%**（不是同事估的 +50~100%）。
`_torch_cuda_wrapper` 在 `NPUModelRunner` 成功 init 后须保持 `cuda.Event→npu`
别名，勿再打回 no-op Placeholder。更深 Preparing / IPC / lookahead 属
runtime/scheduler，勿在本算子内冒充。

**小结**：Cube PV 基线上，AIV 微优化与上述结构线均已停损；优化时请对照本表，
勿重复已否决路径。

## 10. Workspace

```text
[0, systemWs)                         // libapi，≥16MB
[systemWs, …)  FD partial（仅 SplitBNS）:
               accumOut: T × H × kvSplitPart × D   fp32
               LSE:      T × H × kvSplitPart × 2   fp32
[…]  qk staging（仅 qkPvMode==CUBE）:
     parallelCoreNum × 2 slots × (D × KV_TILE_CAP) × sizeof(uint16)
     // Cube QK 存 K^T；Cube PV 复用槽位存 V（行主序）
```

Device Init：`qkSlots = usedCoreNum * 2`（host 按 `parallelCoreNum*2` 分配，更宽）。

## 11. Serving 接入

```text
AscendAttentionBackendImpl
  └─ reshape_and_cache → bit_residual_pack_k8v4
  └─ forward / paged path
        ├─ (optional) bit_residual_fia_paged_k8v4   # env
        └─ bit_residual_attention_paged_k8v4(out=caller_buf)
              └─ None → RuntimeError（fail-fast，禁止静默落到 MSE/slab）
```

1. Wrapper 条件不满足返回 `None`；serving **直接报错**。
2. 调用方传 `out=`；仅当 `attn_output is not output` 时 copy。
3. bf16：query / rotation / pack meta 同 dtype，禁止边界 `→fp16` cast。

## 12. 验证与 Profile

| 工具 | 用途 |
|------|------|
| `xrx_bit_residual_k8v4_smoke.py` | 短序列 + optional profiler（`XRX_K8V4_PROFILE_DIR`） |
| `xrx_bit_residual_k8v4_long_query_profile.py` | 长 query / chunked prefill |
| `xrx_bit_residual_k8v4_bs1_long_profile.py` | bs=1 长 KV |
| `xrx_bit_residual_k8v4_golden.py` | 数值对照 |
| `summarize_k8v4_profiler.py` | host/device；按 Q 首维分 decode/prefill（`--decode-q-max` 默认 32） |
| `xrx_k8v4_fd_cost_model.py` | FD `PickFlashDecodeKvSplitPart` |

典型 long-query：decode `Q≈16`，chunked prefill `Q≈241`。

```bash
rm -rf build csrc/build
python3 setup.py build_ext
```

## 13. 目录结构

```text
csrc/bit_residual_attention_paged_k8v4/
  DESIGN.md
  op_host/
    bit_residual_attention_paged_k8v4_def.cpp
    bit_residual_attention_paged_k8v4_tiling.{h,cpp}
    aclnn_bit_residual_attention_paged_k8v4.h
    CMakeLists.txt
  op_kernel/
    bit_residual_attention_paged_k8v4.cpp   # entry + Kernel + UB
    attention_device.h                      # Vector QK / Softmax / SoftmaxOnly
    decode_device.h                         # 布局常量 + TqRotateMatmulOp
```

## 14. 与 FIA 的边界

| | 本算子 | FIA (`bit_residual_fia_paged_k8v4`) |
|--|--------|-------------------------------------|
| 流水 | AIV decode + Vector/Cube QK/PV | vendor Cube MM1/MM2 + BR dequant |
| Decode 默认 | **是** | env A/B |
| Prefill cache-hit 默认 | **是** | env A/B |
| 长 Q Prefill | qTile + 工作量均衡 + 条件 Cube PV | FIA 自身 tiling / FD |
| 失败（serving） | 返回 None → **抛错** | 返回 None → 再试本算子 |

优化本算子时**不要**改 FIA kernel / tiling；A/B 只通过 env 切换。
