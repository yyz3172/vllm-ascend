# BitResidual FIA AIV：UB 布局图示

> 基线：**P18**（`tmpBuff1` A1 双 half staging + 尾部 decode scratch；meta P9b+P18 ping-pong）。  
> 源码：`fia_block_vec_turboquant_p0.h`（`InitBuffers` / `DequantKvImpl`）、`br_dequant_device.h`。  
> 关联：[`dequant_pa_run_tile_hierarchy.md`](dequant_pa_run_tile_hierarchy.md)、[`dtype_flow_and_dequant_examples.md`](dtype_flow_and_dequant_examples.md)。

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

### 2.1 顶层切分（A1）

```text
tmpBuff1 = 32KB = 32768 B
────────────────────────────────────────────────────────────────
│◀────────── kStageBytes（前部 staging）──────────▶│◀─ scratch ─▶│
│          再均分两个 half（ping-pong）              │  尾部固定    │
────────────────────────────────────────────────────────────────
```

| 符号 | Key（tile=8） | Value（tile=13） | 含义 |
|------|---------------|------------------|------|
| `kScratchElems` | 8×128=1024 | 13×128=1664 | 一个 decode tile 的元素数 |
| `kFp32ScratchCount` | **2** | **2** | Key 均匀量化后与 Value 同形（无 sign/C） |
| `kScratchBytes` | **10240** | **16640** | `elems×(4×nFp32+2)` |
| `kStageBytes` | 22528 | 16128 | `32768 − scratch` |
| `kHalfBytes` | **11264** | **8064** | 每个 staging half 的字节 |
| 有效 `maxSub` | **≈29** | **≈24** | `min(64, (kHalfBytes−128)/perRow)` |

```text
Key 主例（数字，均匀量化后）：
┌──────── half0 11264B ───────┬──────── half1 11264B ───────┬── scratch 10240B ──┐
│ run 偶数: codes ∥ out        │ run 奇数: codes ∥ out        │ A+B+halfScratch     │
└─────────────────────────────┴─────────────────────────────┴─────────────────────┘
 0                        11264                       22528                   32768
```

### 2.2 单个 staging half：codes ∥ out

每个 half 服务 **一个 PA run**（最多 `maxSub` 行）：

```text
batchUb = tmpBuff1[bufIdx * kHalfBytes ..]   // bufIdx = runId % 2

Key 一行: codes 128B + out 256B (WS_T=B16×128)  → perRow = 384
Val 一行: codes  64B + out 256B                 → perRow = 320

┌─────────────────── 一个 half（Key，示意 n=23）───────────────────┐
│ codes[0..n) 连续 uint8     │ pad→32B │ out[0..n) 连续 WS_T      │
│ MTE2 从 pack cache 写入     │         │ VEC decode 写入 → MTE3   │
│                             │         │ 写 dequant K/V workspace │
└─────────────────────────────┴─────────┴──────────────────────────┘
         codesBytes = n×128              outOff = AlignUp32(codesBytes)
```

| 子区 | 谁写 / 谁读 | 用途 |
|------|-------------|------|
| **codes** | MTE2←GM pack；VEC 读 | K8 字节码 / V4 nibble 包 |
| **pad** | — | `out` 起始 32B 对齐（`BrAlignUp32`） |
| **out** | VEC 写；MTE3→GM workspace | 本 run 反量化结果（`WS_T`） |

**为何双 half（A1）**

- half0 的 MTE3 还在吐 `out` 时，half1 可先 MTE2 拉下一 run 的 `codes`
- 与 P18「VEC 末尾预取下一 run」叠加，换流水重叠；Key 均匀量化把 scratch 3→2 后 `maxSub` 抬到 ~29

### 2.3 尾部 decode scratch（按 tile，不是按 maxSub）

预留宽度 = **`BR_*_DECODE_TILE_MAX`**（Key 8 / Val 13），与 `maxSub` 无关。

#### Key（2×fp32 + half；均匀量化）

```text
offset = kStageBytes
┌─ fp32UbA  1024×4 = 4096B ─┐  scratchA：q (u8→fp32)
├─ fp32UbB  4096B ──────────┤  scratchB：BrApplyRowAffine 输出 y
├─ halfScratch 1024×2=2048B ┤  u8→half；Brcb 广播区
└───────────────────────────┘  合计 10240B
```

| 块 | 生命周期内角色 |
|----|----------------|
| **A** | `Cast` 后的 \(q\) fp32（affine 的 `src`） |
| **B** | \(y=\mathrm{meta0}+q\cdot\mathrm{meta1}\) |
| **halfScratch** | ① `u8→half` 中转；② reinterpret 成 fp32 给 `Brcb` |

#### Value（2×fp32 + half；无 C）

```text
┌─ fp32UbA  1664×4 ─┐  s = Cast(int4) 的 fp32；兼 vmin' fold 临时
├─ fp32UbB  同宽 ───┤  BrApplyRowAffine 输出 y
├─ halfScratch ─────┤  int4→half；Brcb
└───────────────────┘
（无 tile 级 fp32UbC。代码里 `fp32UbC` 变量指向 dequantFp32Buf_，但 Value 分支从不传给 BrDecodeValueTile）
```

| 块 | 用途 |
|----|------|
| **A** | int4→fp32 的 \(s\)；PA run 级 `vmin'=vmin+8·vstep` 时复用前 `n` 个元素做 fold（**在 tmpBuff1 尾，不是 dequantFp32Buf_**） |
| **B** | \(y=\mathrm{vmin}'+s\cdot\mathrm{vstep}\) |
| **halfScratch** | nibble→half；affine 广播 |

---

## 3. `dequantInt8Buf_`：meta 双槽（P9b + P18）

与 `tmpBuff1` **分开**；容量按 `BR_S2_SUB_MAX=64` 行设计，**不限制** decode tile，但给出 `maxSub` 的 meta 上限 64。

```text
dequantInt8Buf_ = 2 × 768B = 1536B

┌──────────── slot0 (run 偶数) 768B ────────────┬──────── slot1 (run 奇数) 768B ────────┐
│ [0,128)   meta0 packed B16 ×64 行             │ 同布局                                │
│ [128,256) meta1 packed B16 ×64 行             │                                      │
│ [256,512) meta0 fp32 ×64                      │                                      │
│ [512,768) meta1 fp32 ×64                      │                                      │
└───────────────────────────────────────────────┴──────────────────────────────────────┘
```

| 子区 | 字节 | Key 语义 | Value 语义 |
|------|------|----------|------------|
| packed meta0 | 128 | `base`（B16） | `vmin` |
| packed meta1 | 128 | `step` | `vstep` |
| fp32 meta0 | 256 | `Cast` 后的 base（A）/ −127.5·s（B） | vmin（可被 fold 成 vmin'） |
| fp32 meta1 | 256 | step（A）/ s（B） | vstep |

**用途**

- MTE2：`BrCopyPackedMetaTile` 一次搬本 run 的 `n` 行 SoA meta  
- VEC：`BrCastPackedMetaToFp32` → 供 `BrApplyRowAffine` / Brcb  
- P18：slot 与 staging `bufIdx` 对齐，下一 run 的 meta DMA 可与当前 VEC 重叠  

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

```cpp
Muls(fp32UbA, meta1Fp32, 8.0f, n);   // n ≤ maxSub ≤ 64 个标量
Add(meta0Fp32, meta0Fp32, fp32UbA, n);
```

这里的 `fp32UbA` 是 **`tmpBuff1` 尾部**那块（宽 `13×128`），只用了前 `n` 个 float 当短向量临时；**不是** `dequantFp32Buf_`。

### 4.3 和 `tmpBuff1` scratch 怎么分工（一张图）

```text
                    ┌─ 真正干活（均匀 Key）────────────────────────┐
                    │ tmpBuff1 尾：tile 级 A / B / halfScratch     │
                    │ meta：dequantInt8Buf_ 双槽                     │
                    └───────────────────────────────────────────────┘

                    ┌─ 仍 Init、几乎不读写 ─────────────────────────┐
                    │ dequantFp32Buf_ 1536B  遗留假 fp32UbC         │
                    │ dequantFp16Buf_  256B  无引用                  │
                    └───────────────────────────────────────────────┘
```

| 需要的空间 | 放哪里 | 宽度 |
|------------|--------|------|
| Key u8→fp32 + affine（A/B + half） | `tmpBuff1` 尾 | `8×128` |
| Value int4 + affine（A/B + half） | `tmpBuff1` 尾 | `13×128` |
| Value vmin' fold 短向量 | **复用** `tmpBuff1` 的 `fp32UbA` 前 `n` 元 | `n≤64` |
| 行 meta packed + fp32 | `dequantInt8Buf_` | 最多 64 行 |
| 单行历史 A/B/C、单行 half | `dequantFp32/16Buf_` | **1×128**（遗留） |

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
             │ 读 meta fp32（行广播）
┌─ dequantInt8Buf_ slot ────┐
│ packed B16 → Cast → fp32  │
└───────────────────────────┘
             │
┌─ tmpBuff1 scratch ────────┐
│ Key: A/B/half  u8+affine      │
│ Val: A/B/half  int4+affine    │
└───────────────────────────────┘
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

## 7. 数字速查（Key 主例）

| 项 | 值 |
|----|-----|
| `tmpBuff1` | 32768 |
| Key scratch | 10240（2×fp32×1024 + half×1024） |
| Key 单 half | 11264 |
| Key `perRow` | 384 |
| Key `maxSub` | \(\lfloor(11264-128)/384\rfloor=29\) |
| meta 单槽 | 768；双槽 1536 |
| meta 行上限 | 64（今日松于 staging） |
| A1 双 half 硬顶（scratch→0） | \(\lfloor(16384-128)/384\rfloor=42\) |

---

## 8. 相关常量位置

| 常量 / 逻辑 | 文件 |
|-------------|------|
| `BR_S2_SUB_MAX`、`BR_DEQUANT_UB_BYTES*`、`BR_*_DECODE_TILE_MAX`、`BR_KEY_UNIFORM_SCHEME` | `br_dequant_device.h` |
| `InitBuffers`、Que/Softmax/`dequant*` 申请 | `fia_block_vec_turboquant_p0.h` |
| `kStageBytes` / `kHalfBytes` / `maxSub` / half 内 `codes∥out` | `DequantKvImpl` 同文件 |

---

*文档状态：Key 均匀量化方案 A（`nFp32=2`，`maxSub≈29`）。方案 B 同 UB 账；A/B 性能见 `tools/.../key_uniform_ab_report.md`。*
