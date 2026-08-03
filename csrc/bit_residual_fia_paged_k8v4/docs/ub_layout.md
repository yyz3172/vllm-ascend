# BitResidual FIA AIV：UB 布局图示

> 基线：**P18**（`tmpBuff1` A1 双 half staging + 尾部 decode scratch；meta P9b+P18 ping-pong）。  
> **tile 分路径**：half → `BR_*_DECODE_TILE_MAX_HALF=16`；bf16/Scheme C → `BR_*_DECODE_TILE_MAX=8/13`（见 §2）。  
> 源码：`fia_block_vec_turboquant_p0.h`（`InitBuffers` / `DequantKvImpl`）、`br_dequant_device.h`。  
> 关联：[`dequant_pa_run_tile_hierarchy.md`](dequant_pa_run_tile_hierarchy.md)、[`dtype_flow_and_dequant_examples.md`](dtype_flow_and_dequant_examples.md)、[`dequant_half_affine_fp16.md`](dequant_half_affine_fp16.md)。

## 1. 总览：AIV 上 InitBuffers 分配了什么

`InitBuffers` 一次性申请的主要 UB（每 AIV 一份；两 AIV 子核各有独立 TPipe）：

```text
┌──────────────────────────────── AIV UB（示意，非严格地址序）────────────────────────────┐
│                                                                                        │
│  TQue（进出队，带自动同步）                                                              │
│  ┌ inputQue1 ×2 ×32KB ┐  MM1/MM2 结果、部分 sink 等 GM→UB                              │
│  ┌ inputQue2 ×2 ×16KB ┐  PSE / atten_mask 等                                          │
│  ┌ outputQue1 ×1 ×32KB┐  Softmax P、最终 Cast 输出、部分中间                            │
│  ┌ outputQue2 ×1 ×8KB  ┐  LSE / FD 辅助                                                 │
│                                                                                        │
│  TBuf tmpBuff1 = 32KB  ★ Dequant 期间被切成 staging∥scratch；Vec1/2 收回整块            │
│                                                                                        │
│  Softmax 常驻（`SOFTMAX_TMP_BUFFER_SIZE = 2KB`，`preLoadNum = PRELOAD_NUM = 2`）        │
│  ┌ max/exp/sum 各 2KB×2 = 4KB ┐  跨 S2 内环 ping-pong（`info.loop % 2` 轮换）          │
│  ┌ max/sum default 各 2KB ┐      仅首个 S2 内环初值（−inf / 0），不乘 preLoadNum        │
│                                                                                        │
│  Dequant 专用（与 tmpBuff1 并列）                                                       │
│  ┌ dequantInt8Buf_ = 1536B ┐  meta 双槽（P18 ping-pong，2×768B）——在用                 │
│  ┌ dequantFp32Buf_ = 1536B ┐  历史「1 行 ×3×fp32」dedicated；P18 Value 几乎不用        │
│  ┌ dequantFp16Buf_ = 256B  ┐  历史「1 行 ×half」dedicated；P18 完全未引用              │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

| Buffer | 大小 | 用途（一句话） |
|--------|------|----------------|
| `inputQue1` | 2×32KB | 读入 MM1/MM2 fp32 结果等到 UB |
| `inputQue2` | 2×16KB | PSE、attn_mask 等旁路输入 |
| `outputQue1` | 1×32KB | Softmax 概率、attention 输出 Cast、部分写回 |
| `outputQue2` | 1×8KB | LSE / FD 小块输出 |
| `tmpBuff1` | **32KB** | Softmax 临时 / Acc@Π 临时；**Dequant 时拆成 codes∥out staging + decode scratch** |
| `softmax{Max,Exp,Sum}Buff` | **各 4KB**（`2KB × preLoadNum`，`preLoadNum=2`） | Softmax 状态按 `info.loop % 2` 轮换 |
| `softmax{Max,Sum}DefaultBuff` | **各 2KB**（不乘 `preLoadNum`） | 仅第一个 S2 内环作 `inMax/inSum` 初值 |
| `dequantInt8Buf_` | **1536B** | 行 meta 双槽（见 §3） |
| `dequantFp32Buf_` | **1536B** | 遗留 1 行 fp32 槽（见 §4；P18 热路径几乎不写） |
| `dequantFp16Buf_` | **256B** | 遗留 1 行 half 槽（见 §4；P18 未引用） |


> Dequant 结束后注释写明：`Vec1/Vec2 reclaim the full 32KB`——即 Softmax 等再次独占 `tmpBuff1`，与 decode scratch **时分复用**，不是常驻叠加。

---

## 1.1 Softmax 缓冲取值

| 项 | 取值 | 来源 |
|--|--|--|
| `preLoadNum` | **2** | `fia_kernel_turboquant_p0.h`：`PRELOAD_NUM = 2`，赋给 `constInfo.preLoadNum` |
| 单槽大小 | **2KB** | `SOFTMAX_TMP_BUFFER_SIZE = BUFFER_SIZE_BYTE_2K` |
| max / exp / sum | **各 4KB** | `softmaxPingPongBytes = 2KB × preLoadNum` |
| max / sum default | **各 2KB** | `softmaxDefaultBytes = SOFTMAX_TMP_BUFFER_SIZE`，**不乘** `preLoadNum` |

用途：`info.loop % preLoadNum` 在 **0/1** 两槽间轮换（与 MM workspace ping-pong 一致）；`default` 只在 **第一个 S2 内环**当 `inMax/inSum` 初值（−inf / 0），之后用上一轮写进 `softmaxMax/SumUb` 的槽。

## 2. 焦点：`tmpBuff1` 在 Dequant 期间的布局

> **tile 常量已按路径拆分**（勿把 half 的 16 套到 bf16）：
>
> | 常量 | 值 | 用于 |
> |------|---:|------|
> | `BR_KEY_DECODE_TILE_MAX` | **8** | bf16 / Scheme C（fp32 affine）Key |
> | `BR_VALUE_DECODE_TILE_MAX` | **13** | bf16 / Scheme C Value |
> | `BR_KEY_DECODE_TILE_MAX_HALF` | **16** | `WS_T=half` 原生 half affine Key |
> | `BR_VALUE_DECODE_TILE_MAX_HALF` | **16** | 同上 Value |
>
> half 必须 tile=16：`metaHalf[j0]` 直喂 `Brcb` 要求 `j0%16==0`（32B）。  
> bf16 保持 8/13：fp32 scratch 按 `tile×128×(2×4+2)` 计，抬到 16 会挤爆 staging（L6 bf16 曾 +30%+）。

### 2.1 顶层切分（A1）

```text
tmpBuff1 = 32KB = 32768 B
────────────────────────────────────────────────────────────────
│◀────────── kStageBytes（前部 staging）──────────▶│◀─ scratch ─▶│
│          再均分两个 half（ping-pong）              │  尾部固定    │
────────────────────────────────────────────────────────────────

kScratchElems = TILE_MAX × 128
kScratchBytes = kFp32ScratchBytes + kFp16ScratchBytes
kStageBytes   = 32768 − kScratchBytes
kHalfBytes    = kStageBytes / 2
```

#### 路径总账（字节精确）

| 路径 | `TILE_MAX` | `kScratchElems` | fp32 槽数 | scratch 构成 | **kScratchBytes** | **kStageBytes** | **kHalfBytes** | `perRow` | **maxSub** |
|------|----------:|----------------:|----------:|--------------|------------------:|----------------:|---------------:|---------:|-----------:|
| **half Key** | 16 | 2048 | 0 | 2×half×2048 | **8192** | **24576** | **12288** | 384 | **31** |
| **half Value** | 16 | 2048 | 0 | 2×half×2048 | **8192** | **24576** | **12288** | 320 | **38** |
| **bf16 Key**（A/B） | 8 | 1024 | 2 | 2×fp32×1024 + half×1024 | **10240** | **22528** | **11264** | 384 | **29** |
| **bf16 Value** | 13 | 1664 | 2 | 2×fp32×1664 + half×1664 | **16640** | **16128** | **8064** | 320 | **24** |
| **Scheme C Key** | 8 | 1024 | 3 | 3×fp32×1024 + half×1024 | **14336** | **18432** | **9216** | 384 | **23** |
| **Scheme C Value** | 13 | 1664 | 2 | 同 bf16 Value | **16640** | **16128** | **8064** | 320 | **24** |

`maxSub = min(64, ⌊(kHalfBytes − 128) / perRow⌋)`；`perRow = codeRowBytes + outRowBytes`（Key `128+256=384`，Val `64+256=320`，`WS_T` 按 2B）。

相对「旧 half 路径」：`kStageBytes = 32768 − kScratchBytes`，**scratch↑ ⇒ staging↓**。

**字节公式（与 `DequantKvImpl` 一致，`sizeof(half)=2`）**

```text
kScratchElems = TILE_MAX × 128

旧 half（tile Key=8 / Val=13）：
  kScratchBytes = (kScratchElems×2 + kMetaAlignElems×2) × 2
                = halfSrc + halfBroadcast + meta0Align(16) + meta1Align(16)
  Key:  TILE=8  → elems=1024 → (1024×2 + 16×2)×2 = 2080×2 = 4160
  Val:  TILE=13 → elems=1664 → (1664×2 + 16×2)×2 = 3360×2 = 6720

现 half（tile=16，无 metaAlign）：
  kScratchBytes = kScratchElems×2 × 2     // 仅 halfSrc + halfBroadcast
                = 2048×2×2 = 8192
  （halfSrc 4096B + halfBroadcast 4096B；meta 直喂 packed SoA，不再占尾部 64B）
```

| 路径 | 旧 scratch | 现 scratch | 旧 staging / 单 half | 现 staging / 单 half |
|------|-----------:|-----------:|---------------------:|---------------------:|
| half Key | **4160** | **8192** | 28608 / 14304 | **24576 / 12288** |
| half Value | **6720** | **8192** | 26048 / 13024 | **24576 / 12288** |
| bf16/fp32 | 10240 / 16640 | **不变** | 22528 / 16128 | **不变** |

half Key：scratch **+4032B** → staging **−4032B**；`maxSub` Key 旧 ⌊(14304−128)/384⌋=37 → 现 **31**。

```text
half Key 主例（数字）：
┌──── half0 12288B ────┬──── half1 12288B ────┬── scratch 8192B ──┐
│ run 偶数 codes∥out    │ run 奇数 codes∥out    │ halfSrc+Broadcast │
└──────────────────────┴──────────────────────┴───────────────────┘
 0                   12288                  24576               32768

bf16 Key 主例（数字，与历史 A/B 账一致）：
┌──── half0 11264B ────┬──── half1 11264B ────┬─ scratch 10240B ─┐
│                      │                      │ A+B+halfScratch  │
└──────────────────────┴──────────────────────┴──────────────────┘
 0                   11264                  22528              32768
```

### 2.2 单个 staging half：codes ∥ out

每个 half 服务 **一个 PA run**（最多 `maxSub` 行）：

```text
batchUb = tmpBuff1[bufIdx * kHalfBytes ..]   // bufIdx = runId % 2

Key 一行: codes 128B + out 256B (WS_T=B16×128)  → perRow = 384
Val 一行: codes  64B + out 256B                 → perRow = 320

┌─────────────────── 一个 half（示意）───────────────────┐
│ codes[0..n) 连续 uint8     │ pad→32B │ out[0..n) WS_T │
│ MTE2 ← pack cache          │         │ VEC → MTE3 WS  │
└────────────────────────────┴─────────┴────────────────┘
         codesBytes = n×codeRowBytes     outOff = AlignUp32(codesBytes)
```

| 子区 | 谁写 / 谁读 | 用途 |
|------|-------------|------|
| **codes** | MTE2←GM pack；VEC 读 | K8 字节码 / V4 nibble 包 |
| **pad** | — | `out` 起始 32B 对齐（`BrAlignUp32`） |
| **out** | VEC 写；MTE3→GM workspace | 本 run 反量化结果（`WS_T`） |

**为何双 half（A1）**

- half0 的 MTE3 还在吐 `out` 时，half1 可先 MTE2 拉下一 run 的 `codes`
- 与 P18「VEC 末尾预取下一 run」叠加

### 2.3 尾部 decode scratch（按 tile，不是按 maxSub）

预留宽度 = 当前路径的 `TILE_MAX`（half→`*_HALF`，否则→`BR_*_DECODE_TILE_MAX`）。

#### half Key / Value（`kUseHalfAffine`，tile=16）

无 fp32 tile scratch；meta 不经 `BrCastPackedMetaToFp32`。

```text
offset = kStageBytes = 24576
kScratchElems = 2048

┌─ halfSrc       2048×2 = 4096B ─┐  u8/int4 → half（affine src）
├─ halfBroadcast 2048×2 = 4096B ─┤  Brcb 行广播工作区
└────────────────────────────────┘  合计 8192B
```

| 块 | 字节 | 角色 |
|----|-----:|------|
| **halfSrc** | 4096 | codes Cast 到 half；`BrApplyRowAffine(half)` 的 `src` |
| **halfBroadcast** | 4096 | `Brcb` 展开 `meta0/1` 行标量 |
| meta | 0（本尾） | 直接用 `dequantInt8Buf_` 里 packed half SoA；`j0%16==0` |

Value fold：`BrFoldValueMetaHalf` 把 `foldTmp` 叠在 meta 槽的 **fp32 区**（`BR_META_FP32_0_UB_OFF`）reinterpret 成 half，**不占** `tmpBuff1` 尾。

#### bf16 Key（A/B，tile=8，2×fp32 + half）

```text
offset = kStageBytes = 22528
kScratchElems = 1024

┌─ fp32UbA      1024×4 = 4096B ─┐  q (u8→fp32)
├─ fp32UbB      1024×4 = 4096B ─┤  BrApplyRowAffine 输出 y
├─ halfScratch  1024×2 = 2048B ─┤  u8→half；Brcb 广播区（reinterpret fp32）
└───────────────────────────────┘  合计 10240B
```

#### bf16 Value（tile=13，2×fp32 + half）

```text
offset = kStageBytes = 16128
kScratchElems = 1664

┌─ fp32UbA      1664×4 = 6656B ─┐  s=Cast(int4)；兼 vmin' fold 临时（前 n 标量）
├─ fp32UbB      1664×4 = 6656B ─┤  affine 输出 y
├─ halfScratch  1664×2 = 3328B ─┤  int4→half；Brcb
└───────────────────────────────┘  合计 16640B
```

#### Scheme C Key（tile=8，3×fp32 + half）

```text
offset = kStageBytes = 18432
┌─ fp32UbA 4096B ─┐  …
├─ fp32UbB 4096B ─┤
├─ fp32UbC 4096B ─┤  sign / q7 链第三块
├─ halfScratch 2048B ┤
└────────────────────┘  合计 14336B
```

Scheme C Value 与 bf16 Value 同宽（仍 2×fp32；`fp32UbC` 可指到 `dequantFp32Buf_` 占位）。

---

## 3. `dequantInt8Buf_`：meta 双槽（P9b + P18）

与 `tmpBuff1` **分开**；容量按 `BR_S2_SUB_MAX=64` 行设计，**不限制** decode tile，但给出 `maxSub` 的 meta 上限 64。

```text
dequantInt8Buf_ = 2 × 768B = 1536B

单槽 768B 字节账：
  BR_PACKED_META_BYTES     = 64 × 2 = 128   // meta0 packed B16
  BR_PACKED_META_BYTES     = 128             // meta1 @ +128
  BR_META_FP32_0_UB_OFF    = align32(256) = 256
  BR_META_FP32_ROW_BYTES   = 64 × 4 = 256   // meta0 fp32
  BR_META_FP32_1_UB_OFF    = 256 + 256 = 512
  BR_DEQUANT_UB_BYTES      = 512 + 256 = 768

┌──────────── slot0 (run 偶数) 768B ────────────┬──────── slot1 (run 奇数) 768B ────────┐
│ [0,128)   meta0 packed B16 ×64 行             │ 同布局                                │
│ [128,256) meta1 packed B16 ×64 行             │                                      │
│ [256,512) meta0 fp32 ×64                      │                                      │
│ [512,768) meta1 fp32 ×64                      │                                      │
└───────────────────────────────────────────────┴──────────────────────────────────────┘
```

| 子区 | 字节 | Key 语义 | Value 语义 |
|------|-----:|----------|------------|
| packed meta0 | 128 | `base`（B16） | `vmin` |
| packed meta1 | 128 | `step` | `vstep` |
| fp32 meta0 | 256 | bf16/Scheme C：`Cast` 后 base | bf16：vmin / vmin' |
| fp32 meta1 | 256 | bf16/Scheme C：step | bf16：vstep |

**用途（按路径）**

| 路径 | packed B16 | fp32 区 |
|------|------------|---------|
| **half** | MTE2 写入；**直接** `metaHalf[j0]` → Brcb | Value fold 临时（reinterpret half）；**不做** `BrCastPackedMetaToFp32` |
| **bf16 / Scheme C** | MTE2 写入 | `BrCastPackedMetaToFp32` → fp32 affine / Brcb |

P18：slot 与 staging `bufIdx` 对齐，下一 run 的 meta DMA 可与当前 VEC 重叠。  
**不是** codes/out 缓冲区；codes/out 只在 `tmpBuff1` 的 half 里。

---

## 4. 专用小缓冲：`dequantFp32Buf_` / `dequantFp16Buf_`（遗留 1 行槽）

这两块和 `dequantInt8Buf_`（meta，**在用**）并列申请，但角色完全不同：它们是 **tile scratch 搬进 `tmpBuff1` 尾之前** 留下的「单行 dedicated」UB，P18 热路径几乎不再依赖它们。

### 4.1 怎么申请的

```cpp
// InitBuffers（headDimAlign = 128，COMPUTE_T = float）
fp32Bytes = align32(128 * 3 * 4) = 1536;   // 「1 行 × 3 个 fp32 向量」
fp16Bytes = align32(128 * 2)     = 256;    // 「1 行 × 1 个 half 向量」
pipe->InitBuffer(dequantFp32Buf_, fp32Bytes);
pipe->InitBuffer(dequantFp16Buf_, fp16Bytes);
```

注释原话：*`1-row dedicated scratch only. Tile=8 Key decode scratch overlays tmpBuff1 tail.`*

含义：大块 A/B/C/half 已叠在 `tmpBuff1` 尾（§2.3），这两块只按 **1×headDim** 留口，不再按 `tile×headDim` 扩。

```text
历史单行布局（申请口径，不等于今天怎么用）：

dequantFp32Buf_ 1536B
┌─ 128×fp32 ─┬─ 128×fp32 ─┬─ 128×fp32 ─┐
│  槽0 512B  │  槽1 512B  │  槽2 512B  │   ← 曾对应「一行」的 A/B/C 或 outFp32
└────────────┴────────────┴────────────┘

dequantFp16Buf_ 256B
┌──────── 128×half = 256B ────────┐
│  曾对应一行 halfScratch / Q_T out │
└─────────────────────────────────┘
```

### 4.2 P18 实际谁在读写

| Buffer | Key 路径 | Value 路径 | 结论 |
|--------|----------|------------|------|
| **`dequantFp32Buf_`** | **不用**（Key 仅 A/B，均在 `tmpBuff1` 尾） | 代码仍可能取假 `fp32UbC`，但 **`BrDecodeValueTile` 只收 A/B** | 实质 **空转占位** |
| **`dequantFp16Buf_`** | **无任何 `.Get<>()`** | 同左 | **完全闲置**；`halfScratch` 一律 overlay 在 `tmpBuff1` 尾 |

Value 的 **vmin' fold**（`vmin' = vmin + 8·vstep`）发生在：

- **bf16 / fp32**：`Muls/Add` 在 meta **fp32** 区；临时可复用 `tmpBuff1` 尾 `fp32UbA` 前 `n` 个 float（`n≤maxSub≤64`），**不是** `dequantFp32Buf_`。
- **half**：`BrFoldValueMetaHalf`；临时叠在 meta 槽 fp32 区 reinterpret 的 half 缓冲。

### 4.3 和 `tmpBuff1` scratch 怎么分工（一张图）

```text
                    ┌─ 真正干活 ──────────────────────────────────┐
                    │ half:  tmpBuff1 尾 halfSrc+Broadcast (8KB) │
                    │ bf16:  tmpBuff1 尾 A/B/halfScratch         │
                    │ meta:  dequantInt8Buf_ 双槽 (2×768B)         │
                    └───────────────────────────────────────────────┘

                    ┌─ 仍 Init、几乎不读写 ─────────────────────────┐
                    │ dequantFp32Buf_ 1536B  遗留假 fp32UbC         │
                    │ dequantFp16Buf_  256B  无引用                  │
                    └───────────────────────────────────────────────┘
```

| 需要的空间 | 放哪里 | 宽度（元素） | 字节 |
|------------|--------|-------------:|-----:|
| half Key/Val affine | `tmpBuff1` 尾 | `16×128` ×2 half | **8192** |
| bf16 Key u8→fp32 + affine | `tmpBuff1` 尾 | `8×128` ×(2 fp32+half) | **10240** |
| bf16 Value int4 + affine | `tmpBuff1` 尾 | `13×128` ×(2 fp32+half) | **16640** |
| Value vmin' fold 短向量 | meta fp32 区或 `fp32UbA` 前缀 | `n≤64` 标量 | ≤256 |
| 行 meta packed + fp32 槽 | `dequantInt8Buf_` | 最多 64 行 ×2 槽 | **1536** |
| 单行历史 A/B/C、单行 half | `dequantFp32/16Buf_` | **1×128**（遗留） | 1536+256 |

### 4.4 为何还留着

1. **Init 口径未删**：注释仍按「dedicated 1-row」申请，避免和早期单行 decode / `architecture.md` 布局脱节。  
2. **可回收空间**：若确认无单行回退路径，可去掉这两块（约 **1.75KB** 常驻 UB），不影响 `maxSub`（`maxSub` 只吃 `tmpBuff1` staging）。

---


## 5. 数据流 ↔ 布局（一张总图）

```text
GM pack cache                    GM dequant workspace
 (codes + meta)                   (K_ws / V_ws, WS_T)
        │                                ▲
        │ MTE2                           │ MTE3
        ▼                                │
┌─ staging half (tmpBuff1) ─┐            │
│ codes ──► VEC decode ──► out ──────────┘
└────────────▲──────────────┘
             │
┌─ dequantInt8Buf_ slot ────┐
│ packed B16                │── half: 直喂 Brcb
│ optional Cast → fp32      │── bf16/Scheme C: fp32 affine
└───────────────────────────┘
             │
┌─ tmpBuff1 scratch ────────┐
│ half: halfSrc+Broadcast   │
│ bf16: A/B/halfScratch     │
└───────────────────────────┘
```

时间上（P18）：

```text
run r:   MTE2(half_r, meta_r) → VEC(scratch) → Set V_MTE3
         └─ 可预取 MTE2(half_{r+1}, meta_{r+1}) ──┘
run r:   Wait V_MTE3 → MTE3(out_r)
```

---

## 6. 与 Que / Softmax 的时分关系

| 阶段 | `tmpBuff1` 角色 | Dequant 专用缓冲 |
|------|-----------------|------------------|
| **DequantK/V** | staging + scratch（本节布局） | meta / 小 fp32·fp16 **在用** |
| **Vec1 Softmax** | Softmax 临时、mask 工作区等（整 32KB 收回） | meta 槽闲置 |
| **Vec2 / 输出** | Acc@Π、invalid-row、Cast 辅助等 | 闲置 |

因此看 profile 时：Dequant 热时盯 `tmpBuff1` 切分与 `maxSub`；Softmax 热时盯 Que + softmax Buff，不要把 32KB 算成「Dequant 常驻独占」。

---

## 7. 数字速查

### half（`WS_T=half`，tile=16）

| 项 | Key | Value |
|----|----:|------:|
| `tmpBuff1` | 32768 | 32768 |
| scratch | **8192**（halfSrc 4096 + Broadcast 4096） | **8192** |
| 单 half staging | **12288** | **12288** |
| `perRow` | 384 | 320 |
| `maxSub` | \(\lfloor(12288-128)/384\rfloor=\)**31** | \(\lfloor(12288-128)/320\rfloor=\)**38** |

### bf16（fp32 affine，tile 8/13）

| 项 | Key | Value |
|----|----:|------:|
| scratch | **10240**（A 4096 + B 4096 + half 2048） | **16640**（A 6656 + B 6656 + half 3328） |
| 单 half staging | **11264** | **8064** |
| `maxSub` | **29** | **24** |

### meta / 遗留

| 项 | 值 |
|----|-----|
| meta 单槽 / 双槽 | **768** / **1536** |
| meta 行上限 | 64（今日松于 staging） |
| `dequantFp32Buf_` + `dequantFp16Buf_` | 1536 + 256 = **1792**（遗留） |
| A1 双 half 硬顶（scratch→0，Key） | \(\lfloor(16384-128)/384\rfloor=42\) |

---

## 8. 相关常量位置

| 常量 / 逻辑 | 文件 |
|-------------|------|
| `BR_S2_SUB_MAX`、`BR_DEQUANT_UB_BYTES*`、`BR_*_DECODE_TILE_MAX`、`BR_*_DECODE_TILE_MAX_HALF`、`BR_KEY_UNIFORM_SCHEME` | `br_dequant_device.h` |
| `InitBuffers`、Que/Softmax/`dequant*` 申请 | `fia_block_vec_turboquant_p0.h` |
| `kUseHalfAffine` 选 tile、`kStageBytes` / `kHalfBytes` / `maxSub` / half 内 `codes∥out` | `DequantKvImpl` 同文件 |

---

*文档状态：默认 Scheme A；half 原生 affine（tile=16）与 bf16 fp32 affine（tile 8/13）分账。bf16 A/B 见 `/root/cyl/perflog2/PROFILE_20260803_ab_rebaseline.md`。*
