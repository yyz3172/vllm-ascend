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
           Load/Prefetch K → DecodeK → QK (Vector | Cube)  ∥  Prefetch V (A1)
           Finalize V → DecodeV
           Softmax+PV (Vector)  |  SoftmaxOnly + Cube PV   ∥  M3 next-K（非 Cube PV）
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

实现：K fp32→half → 16×16 vtranspose 得 `K^T` → 写入 per-slot GM →
`SetTensorA(Q)` / `SetTensorB(KT, false)` → `IterateAll` → score half→float。
失败回落 `VectorQkFloatPreScaled(Gqa2)`。

**Decode 永不开 Cube QK**（小 M 的 K^T 税使 Q=16 约 +13%，已回退）。

### 5.2 Cube PV（仅 Prefill qTile）

```text
tileFullForAllQ =
  gqaCount==2 && qRows≥4 && mRows≥32
  && compactRows≤32 && mRows≤64 && preferCubeQk
  && ∀q: ¬(pos < causalEnds[q] < pos+mRows)   // 禁止部分因果 tile
```

通过后：`OnlineSoftmaxOnlyTileFloatPreScaledGqa2Scalar`（更新 m/s，留下 P）→
`CubePvQTile`（P@V，V 行主序 staging，无转置）。失败则 GQA=2 双 head `Axpy` 回落。

Cube PV 占用 `CodeFloat` 时 **延后 M3** Key-code prefetch（先 SoftmaxOnly+PV，再 Issue）。

## 6. UB 账本（≤192 KiB）

静态偏移布局（`bit_residual_attention_paged_k8v4.cpp`），合计约 **171 KiB**：

| Buffer | 约 Bytes | 用途 |
|--------|----------|------|
| PackedRaw | 2112 | K/V 平面临时 staging |
| CodeI16 | 16384 | codes / Cube P half |
| CodeFloat | 32768 | decode scratch；M3 codes；Cube KT/V stage |
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
| VStage | ≤8192（实约 4608） | A1：`[codes][vmin][vstep]` 连续平面 |

常量：`TQ_BR_UB_KV_TILE_CAP=64`，`TQ_BR_UB_GQA_CAP=8`，
`TQ_BR_UB_QTILE_CAP=16`，`TQ_BR_UB_QTILE_GQA_CAP=2`。

## 7. 热路径流水

### 7.1 Decode / 非 qTile（`ComputeAttention`）

```text
LoadQ → Cast → Rotate(Q@R_key) → ×scale
for KV tile:
  LoadK | M3-Finalize → PrefetchV Issue (A1)
  DecodeK → Vector QK (Gqa2)
  PrefetchV Finalize → DecodeV
  M3 Issue next-K → Softmax+PV Scalar (Gqa2) → M3 Finalize
Normalize → [FD WritePartial | Rotate(out@R_value) → WriteOut]
```

### 7.2 Prefill qTile（`ComputeAttentionQTile`）

```text
LoadQ tile → Rotate → ×scale；预取 causalEnds[]
for KV tile (至 maxCausalKvEnd):
  LoadK | M3-Finalize → PrefetchV Issue
  DecodeK → Cube QK (prefer) | Vector QK
  PrefetchV Finalize → DecodeV
  if tileFullForAllQ:
      SoftmaxOnly → CubePv | Axpy fallback
      then M3 Issue          // CodeFloat 已释放
  else:
      M3 Issue → Softmax+PV Scalar → …
  M3 Finalize
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

| 主题 | 做法 | 锚点 |
|------|------|------|
| A1 V prefetch | V MTE2 ∥ DecodeK+QK；`VStage` 分平面 | `PrefetchPackedValueTileRows*` |
| H1 Key/Value bulk | 同 page ≤16 行 codes 一次 + meta 分平面 | `LoadPackedKeyTileRows` 等 |
| M3 Softmax∥LoadK | Softmax 期间预取下一 tile Key codes→`CodeFloat` | `PrefetchPackedKeyCodes*` |
| B3 双 AIV FD | partial 双 AIV；Combine primary-only | `Process` / `CombineFlashDecode` |
| Prefill kvHead 拆双 AIV | 共享 token 区间，避免因果前缀双读 | `ProcessSplitBn` qTile |
| Cube QK per-AIV-slot | KT GM 隔离，双 AIV 可并行发 Matmul | `CubeQkQTile` |
| Prefill Cube PV | SoftmaxOnly + `CubePvQTile`；门控见 §5.2 | `ComputeAttentionQTile` |
| GQA=2 | K/V 行复用两 Q head | `attention_device.h` |
| qTile 负载均衡 | host 按 causal work 切区间 | tiling.cpp |
| 原地输出 | binding `out=`；同 buffer 跳过 copy | `torch_binding` / `attention_v1` |

## 9. 刻意不做 / 已回退（勿重复试）

对照基线：Prefill Cube PV 后 long_query（Decode Q=16 ≈2.23 ms，Prefill Q=241 ≈6.77 ms）。

| 尝试 | 结果 | 结论 |
|------|------|------|
| codes+meta 融合单次 MTE2 / 错对齐 dst | AICORE | 分平面 bulk（已落地） |
| Softmax/PV `Brcb`+`BinaryRepeat` | Prefill **+14%** / 乱码 | 禁止盲 Brcb |
| Softmax 两 head 合 V↔S / Exp 批处理 | Prefill **+1.3%** | 无净收益 |
| Softmax+PV：beta 先搬栈再 Axpy | Decode **+8.2%** | 禁止「只搬栈」 |
| Decode SoftmaxOnly + Cube PV（`compactRows=2`） | Decode **+2.5%** | 小 M Cube PV 税＞收益 |
| Prefill 部分因果仍走 Cube PV | Prefill **−0.1%** | 保持 `tileFullForAllQ` |
| qTile Cap 16→24 | 持平 | 保持 Cap=16 |
| Prefill KV-outer lite | Prefill **+1.3%** | 未少扫 KV |
| Prefill 真 KV-outer GM spill | Prefill **+0.8%** | GM 税抵消少扫；需重设计 |
| `SetTensorB(true)` 消手工 K^T | 持平 | 转置税不在此 |
| Decode 开 Cube QK | Q=16 约 **+13%** | Decode 保持 Vector QK |
| 盲目抬 `FLASH_DECODE_MAX_Q_TOKENS` | Q≥32 收益 ≤1% | 禁止盲目抬 cap |
| DecodeK/V 合并 per-row barrier / 整 tile Mul+Add | **+1.8~1.9%** | 解压行向已非瓶颈 |
| Decode M3 改 `RotateWork` 提前 Issue | Decode **+0.3%** | 无收益 |
| Cube PV 期 M3 改 `VStage` | Prefill **−0.6%** | SoftmaxOnly 窗短 |
| meta 留 UB（去 GetValue→栈） | 假加速 + smoke **乱码** | 禁止未验证 offset Cast |
| Vector QK GQA2 减 barrier | **0.0%** | Prefill 主路径已是 Cube QK |
| Prefill 再搬 QK ReduceSum 上 Cube | — | 已有 `CubeQkQTile` |
| DecodeKey 迁 Cube / AIC MTE1 卸 MTE2 | — | 否决 |

AIV 微优化在 Cube PV 基线上已榨干。结构性破局方向（未落地）：真 KV-outer、
多 tile 合批 Cube、FIA 旁路——见会话设计页 / Todos，**勿**再重复上表路径。

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
