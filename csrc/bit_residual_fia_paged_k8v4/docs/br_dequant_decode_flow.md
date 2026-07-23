# BitResidual K8/V4 AIV Decode 流程说明

> 源码：`op_kernel/vendored/arch32/br_dequant_device.h`  
> 覆盖：`BrDecodeKeyTile` / `BrDecodeValueTile` / `BrApplyRowAffine`，以及 `FP32_BLOCK_ELEMS` / `FP32_REPEAT_ELEMS` 的硬件依据。

## 目录

1. [总览](#1-总览)
2. [Key Decode](#2-key-decode)
3. [Value Decode](#3-value-decode)
4. [BrApplyRowAffine（按行仿射）](#4-brapplyrowaffine按行仿射)
5. [分块与广播图示](#5-分块与广播图示)
6. [为什么是 8 和 64](#6-为什么是-8-和-64)
7. [AscendC 256B 上限如何确认](#7-ascendc-256b-上限如何确认)

---

## 1. 总览

Key / Value 在 AIV 上按 **tile（多行）** 解码：先把 pack 码还原成 FP32，再 Cast 成输出 dtype（fp16/bf16），供后续 Cube attention 使用。`headDim = 128`。

| 路径 | 入口 | 公式（每行一套 meta） |
|------|------|----------------------|
| Key K8 | `BrDecodeKeyTile` | \(y = \mathrm{sign}_{\pm1}\cdot(\mathrm{base}+q7\cdot\mathrm{step})\) |
| Value V4 | `BrDecodeValueTile` | \(y=\mathrm{vmin}'+s\cdot\mathrm{vstep}\)，\(s=\mathrm{Cast}(\mathrm{int4})\in[-8,7]\)；\(\mathrm{vmin}'=\mathrm{vmin}+8\cdot\mathrm{vstep}\)（P17b-B：PA run 级折算） |

两边共用 `BrApplyRowAffine`：对 `[numRows, headDim]` 做 **按行广播** 的 `dst = src * scale + offset`。

Pack 布局、PA 寻址等见 [`architecture.md`](architecture.md)。

---

## 2. Key Decode

### 2.1 Pack 约定（LSB sign）

上游当前布局：

\[
\mathrm{code} = (q7 \ll 1) \mid \mathrm{sign},\quad q7\in[0,127],\ \mathrm{sign}\in\{0,1\}
\]

- sign 在 **bit0（LSB）**
- q7 在 **bits 1..7**，`>> 1` 取出

> 历史曾用 MSB sign：`code = q7 | (sign << 7)`。与 LSB 不兼容，rebase 时必须跟 pack 约定对齐。

### 2.2 流水

```
codesUb [numRows × headDim] uint8
        │
        ▼ Cast uint8→half→int16          (P13: 合并相邻 Cast)
        │
        ├─ And 0x01  → signStorage       (LSB sign)
        └─ >> 1      → q7 in codeU16
        │
        ├─ sign: Cast→Muls(-2)→Adds(1) → ±1  (scratchB)
        └─ q7:   Cast→FP32                 (scratchC)
                    │
                    ▼ BrApplyRowAffine
               err = base + q7 * step      (scratchA)
                    │
                    ▼ Mul(err, ±1)
               y_fp32
                    │
                    ▼ Cast → outUb (headDim 或按行写到 headDimAlign)
```

要点：

- `Duplicate(0x01)` → `And` 需要 barrier（RAW：mask）
- `And` → `ShiftRight`、`ShiftRight` → `Cast(sign)` 写不同 UB，P13b 可去掉中间 barrier
- `Cast(q7)` 依赖 `ShiftRight` 写完；由 Adds 链后的 barrier 一并保护

---

## 3. Value Decode

### 3.1 Pack

nibble / `int4` 打包：`nibbleUb` 为 `numRows × (headDim/2)` uint8。

\[
y = \mathrm{vmin} + \underbrace{(s+8)}_{0..15}\cdot\mathrm{vstep}
  = \underbrace{(\mathrm{vmin}+8\cdot\mathrm{vstep})}_{\mathrm{vmin}'} + s\cdot\mathrm{vstep}
\]

`Cast(int4b_t)` 得到有符号 \(s\in[-8,7]\)。P17b-B 在 `DequantKvImpl` 每个 PA run 上对 `n` 行做一次 `vmin' = vmin + 8*vstep`，tile 内不再 `Adds(+8)`。

### 3.2 流水

```
PA run meta (n rows):  vmin' = vmin + 8*vstep   (P17b-B, once per run)
nibbleUb (int4 packed)
        │
        ▼ Cast int4→half                 (s ∈ [-8,7])
        ▼ Cast half→FP32  (scratchA)
        │
        ▼ BrApplyRowAffine
   y = vmin' + s * vstep  (scratchB)
        │
        ▼ Cast → outUb
```

`BrApplyRowAffine` 前保留一道 `PipeBarrier`：后续会把 `halfScratch` reinterpret 成 `metaBroadcast`。

---

## 4. BrApplyRowAffine（按行仿射）

```cpp
// dst[r, c] = src[r, c] * scales[r] + offsets[r]
BrApplyRowAffine(dst, src, offsets, scales, broadcast, numRows, headDim);
```

| 参数 | Key | Value |
|------|-----|-------|
| `src` | q7（FP32） | \(s\)（有符号 int4→FP32） |
| `scales` | step | vstep |
| `offsets` | base | \(\mathrm{vmin}'\)（P17b-B 已折入 \(+8\cdot\mathrm{vstep}\)） |

实现骨架：

1. `Brcb(broadcast, scales, …)`：每行 1 个 scale → 1 个 FP32 block（8 个相同值）
2. 列方向按 **64** 切分，对每段 `Mul(..., numRows, BinaryRepeatParams)`
3. `Brcb(broadcast, offsets, …)` + 同样切分的 `Add`

```cpp
constexpr uint32_t FP32_BLOCK_ELEMS = 8U;   // datablock = 32B / 4
constexpr uint32_t FP32_REPEAT_ELEMS = 64U; // 一次 repeat = 256B / 4
rowStrideBlocks = headDim / 8;             // 128/8 = 16
columnLoops     = ceil(headDim / 64);      // 128 → 2
```

`BinaryRepeatParams`：

| 字段 | 值 | 含义 |
|------|-----|------|
| `src1BlkStride` | 0 | 同一行内反复用同一个 scale block |
| `src1RepStride` | 1 | 下一行换下一个 broadcast block |
| `dstRepStride` / `src0RepStride` | `rowStrideBlocks` | 行主序下跳到下一行 |

---

## 5. 分块与广播图示

以下用 `numRows=3`、`headDim=128` 举例（真实 tile 可达 Key≤8 / Value≤13 行，逻辑相同）。

### 5.1 目标布局

```
src / dst [numRows × headDim]，行主序：

行0:  s0_0 … s0_63 | s0_64 … s0_127
行1:  s1_0 … s1_63 | s1_64 … s1_127
行2:  s2_0 … s2_63 | s2_64 … s2_127
      <── 列块0 (64) ──> <── 列块1 (64) ──>

scales: [σ0, σ1, σ2]   ← 每行只有 1 个数
```

### 5.2 Brcb：标量 → 8×FP32 block

```
scales                 broadcast（Brcb 后）
  σ0   ──►  [σ0 σ0 σ0 σ0 σ0 σ0 σ0 σ0]   block 0
  σ1   ──►  [σ1 σ1 σ1 σ1 σ1 σ1 σ1 σ1]   block 1
  σ2   ──►  [σ2 σ2 σ2 σ2 σ2 σ2 σ2 σ2]   block 2
```

### 5.3 列方向两次 Mul（Add 同构）

```
                    scales ──Brcb──► broadcast
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
   ┌──────── 列循环 0 ─────────┐   ┌──────── 列循环 1 ─────────┐
   │ Mul 列[0:64)  × broadcast │   │ Mul 列[64:128) × broadcast│
   │   repeat 0: 行0           │   │   repeat 0: 行0           │
   │   repeat 1: 行1           │   │   repeat 1: 行1           │
   │   repeat 2: 行2           │   │   repeat 2: 行2           │
   └───────────────────────────┘   └───────────────────────────┘
                    │                             │
                    └────────► dst = src * σ ─────┘

再 Brcb(offsets) + 两段 Add → dst = dst + offset
```

### 5.4 一次 Mul 的“视野”（列块 0）

```
        列 0‥7  8‥15 … 56‥63     （8 个 block = 64 FP32）
行0:   [████][████]…[████]  × σ0-block
行1:   [████][████]…[████]  × σ1-block   ← repeat 推进
行2:   [████][████]…[████]  × σ2-block

src / dst 指针每次 repeat += 16 blocks（rowStrideBlocks）
broadcast 指针每次 repeat += 1 block
```

### 5.5 两层粒度对照

```
                ┌────── headDim=128 ──────┐
                │  64 elems  │  64 elems  │  ← FP32_REPEAT_ELEMS（列切分）
行内:  [b0][b1]…[b7] | [b8]…[b15]
         └── 每块 8×FP32 ──┘                 ← FP32_BLOCK_ELEMS（Brcb / stride）
行间:  隔 16 个 block 到下一行               ← rowStrideBlocks = 128/8
```

---

## 6. 为什么是 8 和 64

| 常量 | 值 | 能否改大一次算完 128？ | 绑定约束 |
|------|-----|------------------------|----------|
| `FP32_BLOCK_ELEMS` | 8 | **否** | datablock=32B；`Brcb` 与 `*BlkStride` 按 block 计 |
| `FP32_REPEAT_ELEMS` | 64 | **否**（单次列宽） | 一次 vector repeat = 256B → FP32 最多 64 元素 |

`headDimAlign`（常为 128，给 Cube 对齐）**不会**绕过单次 64 元素上限。Affine 按有效 `headDim` 计算；写出时再按 align 处理。

若要“整行一次”，需换实现（例如先把 σ 扩成整行再 Mul），仍可能被 256B/repeat 拆成两段，且更费 UB/指令——不是把 `FP32_BLOCK_ELEMS` 改成 128 能解决的。

---

## 7. AscendC 256B 上限如何确认

容器 CANN **8.5.1** 头文件（910B / dav_c220 使用）：

`/usr/local/Ascend/cann-8.5.1/aarch64-linux/asc/impl/basic_api/utils/kernel_utils_constants.h`

```c
const int32_t DEFAULT_BLK_NUM = 8;
const uint8_t DEFAULT_REPEAT_STRIDE = 8;      // 连续时 8 block = 1 repeat
const uint32_t ONE_BLOCK_SIZE = 32;           // datablock
const uint32_t VECTOR_REG_WIDTH = 256;

const int32_t ONE_REPEAT_BYTE_SIZE = 256;     // ★ 单次 repeat 字节数
const uint8_t B32_DATA_NUM_PER_BLOCK = 8;     // = FP32_BLOCK_ELEMS
const int32_t B32_DATA_NUM_PER_REPEAT = 64;   // = FP32_REPEAT_ELEMS
const int32_t B16_DATA_NUM_PER_REPEAT = 128;

const uint32_t FLOAT_REPEAT_SIZE =
    ONE_REPEAT_BYTE_SIZE / B32_BYTE_SIZE;     // 256/4 = 64
```

带 `BinaryRepeatParams` 的 Level-0 `Mul`/`Add`（`dav_c220/kernel_operator_vec_binary_impl.h`）路径为：

```text
SetMask → vmul/vadd(..., repeatTime, BlkStride, RepStride)
```

- **每个 repeat**：最多 `ONE_REPEAT_BYTE_SIZE`（256B）  
- **`repeatTime`**：最多 `MAX_REPEAT_TIMES`（255），用于行方向  
- 因此 FP32 下列方向单次安全上限为 **64**；`headDim=128` → `columnLoops=2`

**结论**：910B 上 256B/repeat、64 FP32/repeat、8 FP32/block 是 AscendC 明确定义的常量，与本文件中的 `FP32_*` 一一对应。
