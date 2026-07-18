# BitResidualAttentionPagedK8v4 设计文档

> Vector paged-attention 算子：在 BitResidual K8/V4 压缩 cache 上做 Decode / Prefill。
> Cache 布局真源见 `../bit_residual_pack_k8v4/design.md`。FIA 流水线对照见
> `../bit_residual_fia_paged_k8v4/DESIGN.md`。

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

FIA 与本算子均为可选 A/B；两开关默认均为 `0`。

```text
pack_k8v4 → uint8 PA cache
query → LoadQ → (optional Cube rotate Q@R)
      → tile loop:
           DecodeK (AIV) → QK (Vector / Cube)  ∥  Prefetch V (MTE2)
           DecodeV finalize → Softmax+PV (Vector)
      → WriteOut (optional Cube rotate @R_value)
```

## 2. IO

**Inputs**

- `query`：fp16 / bf16，TND `[T, H, D]`，`D=128`
- `key_cache` / `value_cache`：uint8 PA layout（见 §3）
- `block_table`：int32
- `actual_seq_len_q` / `actual_seq_len_kv`：int64（ValueDepend）
- `rotation_key` / `rotation_value`：`[D,D]`，与 query 同 dtype

**Output**

- `attention_out`：与 query 同 shape / dtype；binding 支持可选 `out=` 原地写

**Attrs**：`num_heads`, `num_kv_heads`, `head_size(=128)`, `block_size(%16==0)`,
`max_actual_seq_len`, `scale_value`

**约束**：`num_heads % num_kv_heads == 0`；GQA 组大小 UB 上限 `TQ_BR_ATTN_GQA_CAP=8`（按 chunk 处理）。

## 3. Cache 布局

与 pack 共用（每 16-row sub-block）：

```text
Key:   [16*128 codes][16*2 base][16*2 step]     → 2112 B
Value: [16*64 nibbles][16*2 vmin][16*2 vstep]   → 1088 B
```

反量化：

```text
K: q7=code&0x7F, sign=code>>7 → y = ±(base + q7*step)
V: idx4 ∈ [0,15]            → V = vmin + idx4*vstep
```

## 4. Tiling Key 与调度

| Key | 模式 | 典型场景 |
|-----|------|----------|
| `0` | SplitBN + Vector QK/PV | 短 KV Decode；非 qTile Prefill |
| `1` | SplitBNS + Vector QK/PV（FlashDecode） | 长 KV、小 Q（`numTokens ≤ 32`） |
| `2` | SplitBN + qTile + Cube QK / Vector PV | Prefill / ChunkedPrefill（`GQA≤2`） |

硬件：`KERNEL_TYPE_MIX_AIC_1_2`（每 MIX 组 1 AIC + 2 AIV）。

### 4.1 SplitBN（key 0 / 2）

- 任务粒度：`(token, kvHead, gqaChunk)`
- Host 按 task 均分到 `usedCoreNum` 个 MIX 组
- Device：**两个 AIV subcore 都参与**（`workerIdx = mixCoreIdx*2 + subIdx`）
- AIC：Decode / FD 路径下直接 return；qTile Cube QK 时 primary AIV 经 KFC 发 Matmul

### 4.2 SplitBNS / FlashDecode（key 1）

开启条件（`IsFlashDecodeK8v4`）：

1. `1 ≤ numTokens ≤ TQ_BR_FLASH_DECODE_MAX_Q_TOKENS`（32）
2. `PickFlashDecodeKvSplitPart(taskCount, maxKvLen, coreNum) > 1`

分段代价模型（host）：

```text
cost(P) = ceil(taskCount / (coreNum/P)) * ceil(maxKvLen / P)
约束：maxKvLen ≥ 1024，segment ≥ 512，P ≤ coreNum
```

流程：

1. 双 AIV 写 FD partial（`accumOut` / LSE）到 workspace
2. `SyncAll`
3. **仅 primary AIV** 做 `CombineFlashDecode`（Rotate 共用每组一个 KFC）

调试：编译期 `TQ_BR_FORCE_DISABLE_FLASH_DECODE=1` 可强制关 FD。  
Host 探针：`tests/e2e/singlecard/xrx_k8v4_fd_cost_model.py`。

### 4.3 Prefill qTile（key 2）

开启条件：

```text
splitMode == SplitBN
&& gqaGroup ≤ 2
&& numTokens > usedCoreNum
```

Host 为每个 MIX 组写 `[qTileTokenStart, qTileTokenEnd)`：

- 默认按 token 数均分
- 有 ValueDepend seq 时，按 **causal KV 工作量前缀和** 再平衡（避免尾部长 causal 偏重）

Device：每个 MIX 组的两个 AIV 再把该区间对半切，同组共享已 decode 的 K/V tile；
Cube QK 把物理 `K^T` 落在 per-core qk workspace。

## 5. 热路径优化（摘要）

相对早期 vector 基线，当前实现的主要收益点：

| 主题 | 做法 | 代码锚点 |
|------|------|----------|
| AIV 解包 | 向量化 K/V 解包；`Duplicate+Axpy`；栈上 scalar base/step | `DecodePacked*`, `decode_device.h` |
| 寻址缓存 | 同 page 连续行复用 `blockId`；qTile 预取 `causalEnds` | `GetBlockId` / qTile loop |
| A1 V prefetch | `PrefetchPackedValueTileRowsIssue` 把 V MTE2 叠在 DecodeK+QK 上；独立 `VStage` | kernel main / qTile loop |
| B3 双 AIV FD | SplitBNS partial 双 AIV；Combine 仍 primary-only | `Process()` |
| 双 AIV SplitBN | Decode / qTile 均映射 `workerNum = usedCoreNum*2` | `ProcessSplitBn` |
| Prefill 按 kvHead 拆双 AIV | qTile 路径两 AIV 按 `kvHead % 2` 分工，避免 token 对半切导致的 KV 前缀双读（Prefill Q=241 约 −27%） | `ProcessSplitBn` qTile |
| H1 Key code plane bulk | 同 page ≤16 行 codes 一次 MTE2；meta 仍逐行；Cast 后 `V_MTE2` 再复用 PackedRaw（Decode 约 −2%） | `LoadPackedKeyTileRows` |
| M2 更大 qTile | Prefill `qTile` 8→16，`CubeQk MAX_M` 16→32（覆盖 GQA=2）；Prefill 约 −1.6% | `TQ_BR_UB_QTILE_CAP` |
| Cube QK 按 AIV slot | KT GM 用 `mixCore*2+subIdx`，每 MIX 分配 2 槽；双 AIV Prefill 可并行 Cube（约 −29%） | `CubeQkQTile` / tiling ws |
| M3 Softmax∥LoadK | Softmax 期间 MTE2 预取下一 tile Key codes（暂存 CodeFloat）；Decode 约 −1% | `PrefetchPackedKeyCodes*` |
| qTile 负载均衡 | host 按 causal work 切 token 区间 | tiling.cpp |
| GQA=2 特化 | `VectorQkFloatPreScaledGqa2` / `OnlineSoftmaxUpdateTileFloatPreScaledGqa2Scalar`：K/V 行复用两 Q head | `attention_device.h` |
| 原地输出 | binding `out=`；eager 路径若返回同一 buffer 则跳过 self-copy | `torch_binding.cpp`, `attention_v1.py` |

**刻意不做 / 已回退**：bulk MTE2 metadata batch（曾触发 AICORE）；`SetTensorB(true)`
消 K^T 手工转置（相对 H2 持平）；Decode 路径开 Cube QK（小 M 的 K^T 税使
Q=16 约 +13%）；盲目抬高 `TQ_BR_FLASH_DECODE_MAX_Q_TOKENS`（需先更新 FD cost model）。

## 6. Workspace

```text
[0, systemWs)                              // libapi / system
[systemWs, …)  partial accum + LSE         // 仅 SplitBNS
[…, …)         qk K^T staging per core     // 仅 qTile Cube QK
```

- FD partial：`accumOutSize = T * H * kvSplitPart * D`（fp32）  
  `logSumExpSize = T * H * kvSplitPart * 2`（fp32）
- qTile：`qkWorkspaceStride = D * KV_TILE_CAP` half 元素 / core

## 7. Serving 接入（attention_v1）

```text
AscendAttentionBackendImpl
  └─ reshape_and_cache → bit_residual_pack_k8v4
  └─ forward / paged path
        ├─ (optional) bit_residual_fia_paged_k8v4   # env 打开时
        └─ bit_residual_attention_paged_k8v4(out=caller_buf)
              └─ None → RuntimeError（fail-fast，禁止静默落到 MSE/slab）
```

要点：

1. Wrapper 条件不满足时返回 `None`；serving **直接报错**，不再 silent fallback。
2. 调用方传入 `out=`，避免额外分配；`forward` 末尾仅在 `attn_output is not output` 时 copy。
3. bf16 serving：query / rotation / pack meta 同 dtype，禁止边界 `→fp16` cast。

## 8. 验证与 Profile

| 工具 | 用途 |
|------|------|
| `tests/e2e/singlecard/xrx_bit_residual_k8v4_smoke.py` | 短序列正确性 + optional profiler（`XRX_K8V4_PROFILE_DIR`） |
| `tests/e2e/singlecard/xrx_bit_residual_k8v4_long_query_profile.py` | 长 query / chunked prefill |
| `tests/e2e/singlecard/xrx_bit_residual_k8v4_bs1_long_profile.py` | bs=1 长 KV 微基准 |
| `tests/e2e/singlecard/xrx_bit_residual_k8v4_golden.py` | 数值对照 |
| `tests/e2e/singlecard/summarize_k8v4_profiler.py` | 汇总 host/device；按 `Input Shapes` 首维 Q 分 decode / prefill（`--decode-q-max`，默认 32） |
| `tests/e2e/singlecard/xrx_k8v4_fd_cost_model.py` | 打印 FD `PickFlashDecodeKvSplitPart` 代价 |

典型 long-query：decode `Q≈num_prompts`（如 16），chunked prefill `Q≈241`。

改 C++ 后需重建扩展：

```bash
rm -rf build csrc/build
python3 setup.py build_ext --inplace   # 或项目惯用 rebuild 脚本
```

## 9. 目录结构

```text
csrc/bit_residual_attention_paged_k8v4/
  DESIGN.md                          # 本文档
  op_host/
    bit_residual_attention_paged_k8v4_def.cpp
    bit_residual_attention_paged_k8v4_tiling.{h,cpp}
    aclnn_bit_residual_attention_paged_k8v4.h
  op_kernel/
    bit_residual_attention_paged_k8v4.cpp   # entry + Kernel
    attention_device.h                      # Vector QK / Softmax+PV / GQA=2
    decode_device.h                         # packed K/V decode helpers
```

## 10. 与 FIA 的边界

| | Attention（本算子） | FIA (`bit_residual_fia_paged_k8v4`) |
|--|---------------------|-------------------------------------|
| 流水 | AIV decode + Vector/Cube QK/PV | vendor FIA Cube MM1/MM2 + BR dequant |
| Decode 默认 | **是** | env A/B |
| Prefill cache-hit 默认 | **是** | env A/B |
| 长 Q Prefill | qTile + 工作量均衡 | FIA 自身 tiling / FD |
| 失败策略（serving） | 返回 None → **抛错** | 返回 None → 再试本算子 |

优化本算子时**不要**改 FIA kernel / tiling；A/B 只通过 env 切换。
