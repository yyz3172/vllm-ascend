# BitResidual FIA Paged K8V4 架构文档

## 目录

1. [算子概述](#1-算子概述)
2. [K8/V4 数据布局详解](#2-k8v4-数据布局详解)
3. [BlockTable 寻址机制](#3-blocktable-寻址机制)
4. [反量化流程](#4-反量化流程)
5. [UB 内存预算分析](#5-ub-内存预算分析)
6. [性能瓶颈与优化思路](#6-性能瓶颈与优化思路)

---

## 1. 算子概述

BitResidual FIA Paged K8V4 是 Ascend NPU 上的融合推理注意力算子，针对 BitResidual K8/V4 量化方案。

### 1.1 核心架构

```
┌─────────────────────────────────────────────────┐
│  Query/Rotation (fp16/bf16)                     │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  QK MatMul (AIC)                                │
│  - 从 KV Cache 读取 K8 压缩数据                 │
│  - AIV 反量化: uint8 → float (base + q7*step)   │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  Softmax + Mask (AIV)                           │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  SV MatMul (AIC)                                │
│  - 从 KV Cache 读取 V4 压缩数据                 │
│  - AIV 反量化: int4 → float (vmin + idx4*vstep) │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  Output (fp16/bf16)                             │
└─────────────────────────────────────────────────┘
```

### 1.2 关键概念澄清

| 概念 | 值 | 含义 |
|------|-----|------|
| `kvCacheBlockSize` (PA block_size) | **128**（vLLM 默认） | PagedAttention 中一个物理 block 容纳的 **token 数** |
| `BR_BLOCK_ROWS` | **16** | BitResidual pack 布局中的 **tile 行数**（一个 pack tile 包含 16 行 token） |

**关系**：一个 PA block（128 tokens）= 128/16 = **8 个 BR pack tile**。

```
┌──────────────────── PA Block (blockSize=128) ────────────────────┐
│                                                                   │
│  ┌─── BR Tile 0 (16 rows) ──┐  ┌─── BR Tile 1 (16 rows) ──┐    │
│  │ row 0:  token[0]  K8/V4  │  │ row 0:  token[16] K8/V4  │    │
│  │ row 1:  token[1]  K8/V4  │  │ row 1:  token[17] K8/V4  │    │
│  │ ...                       │  │ ...                       │    │
│  │ row 15: token[15] K8/V4  │  │ row 15: token[31] K8/V4  │    │
│  └───────────────────────────┘  └───────────────────────────┘    │
│                                                                   │
│  ┌─── BR Tile 2 ──┐ ... ┌─── BR Tile 7 ──┐                      │
│  │ token[32..47]  │     │ token[112..127]│                      │
│  └────────────────┘     └────────────────┘                      │
└───────────────────────────────────────────────────────────────────┘
```

---

## 2. K8/V4 数据布局详解

### 2.1 整体内存组织

**BR K8V4 使用 `[physBlock][kvHead][rows]` 布局**，而非标准 PageAttention 的 `[physBlock][rows][kvHead]`。

**选择这种布局的原因**：
1. Pack 压缩需要同一 head 的 16 行连续数据
2. 反量化时需要批量读取同一 head 的多行
3. Meta 数据需要连续存放以提高局部性

**K8 和 V4 是独立的 buffer**：
```
key_cache (uint8 buffer):
┌──────────────────────────────────────────────────┐
│ [block0,head0,K8][block0,head1,K8]...            │
│ [block1,head0,K8][block1,head1,K8]...            │
└──────────────────────────────────────────────────┘

value_cache (uint8 buffer):
┌──────────────────────────────────────────────────┐
│ [block0,head0,V4][block0,head1,V4]...            │
│ [block1,head0,V4][block1,head1,V4]...            │
└──────────────────────────────────────────────────┘
```

### 2.2 K8 布局（每 head，每行）

K8 = 8-bit 量化 Key，head_dim=128。

**每行 token 的 K8 数据**（132 bytes）：

```
┌──────────── 128 bytes ────────────┬── 2 bytes ──┬── 2 bytes ──┐
│         codes[128]                │   base      │   step      │
│  每个 code 是 uint8:              │  (fp16/bf16)│  (fp16/bf16)│
│  ┌──────────────────────┐        │             │             │
│  │ bit7: sign (0=+,1=-) │        │             │             │
│  │ bit6-0: q7 [0,127]  │        │             │             │
│  └──────────────────────┘        │             │             │
└──────────────────────────────────┴─────────────┴─────────────┘
```

**一个 PA block（128 行）内单个 kvHead 的 K8 数据在 GM 中的布局**：

```
headBase + 0:         codes[0]  (128 bytes)
headBase + 128:       codes[1]  (128 bytes)
headBase + 256:       codes[2]  (128 bytes)
  ...
headBase + 127*128:   codes[127] (128 bytes)
                      ──────── 共 16384 bytes ────────

headBase + 16384:     base[0]   (2 bytes)
headBase + 16386:     base[1]   (2 bytes)
  ...
headBase + 16638:     base[127] (2 bytes)
                      ──────── 共 256 bytes ────────

headBase + 16640:     step[0]   (2 bytes)
headBase + 16642:     step[1]   (2 bytes)
  ...
headBase + 16894:     step[127] (2 bytes)
                      ──────── 共 256 bytes ────────

Total: 16384 + 256 + 256 = 16896 bytes = 128 * 132
```

**注意**：一行的完整数据（codes + base + step）在 GM 端分成了三段，彼此间隔很远：

```
row 72 的完整数据：
  codes: headBase + 9216    (128 bytes)
  base:  headBase + 16528   (2 bytes)    ← 距 codes 偏移 7312 bytes!
  step:  headBase + 16784   (2 bytes)    ← 距 base 偏移 256 bytes
```

**反量化公式**：
```
code = codes[i]                    // uint8
q7   = code & 0x7F                 // 取低 7 位 → [0, 127]
sign = code >> 7                    // 取最高位 → 0 或 1
sign_val = (sign == 0) ? +1 : -1   // 符号映射

y = sign_val * (base + q7 * step)  // 最终 float 值
```

### 2.3 V4 布局（每 head，每行）

V4 = 4-bit 量化 Value，head_dim=128。

**每行 token 的 V4 数据**（68 bytes）：

```
┌──── 64 bytes ────┬── 2 bytes ──┬── 2 bytes ──┐
│  nibbles[64]     │   vmin      │   vstep     │
│  每个 byte 含     │  (fp16/bf16)│  (fp16/bf16)│
│  2 个 int4:       │             │             │
│  ┌────────────┐   │             │             │
│  │ high: [0,15]│   │             │             │
│  │ low:  [0,15]│   │             │             │
│  └────────────┘   │             │             │
└───────────────────┴─────────────┴─────────────┘
```

64 bytes × 2 nibbles/byte = 128 个 int4 值，正好对应 head_dim=128。

**一个 PA block（128 行）内单个 kvHead 的 V4 数据在 GM 中的布局**：

```
偏移 0 ~ 8191:      nibbles[0..127]   ← 128行 × 64B = 8192B（连续）
偏移 8192 ~ 8447:   vmin[0..127]      ← 128行 × 2B  = 256B（连续）
偏移 8448 ~ 8703:   vstep[0..127]     ← 128行 × 2B  = 256B（连续）
                    ──────────────────
                    总计 = 8704B = 128 × 68
```

**反量化公式**：
```
byte = nibbles[i / 2]
idx4 = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F)
// 实际实现中用 int4b_t 硬件解包: int4 → +8 → [0, 15]

V = vmin + idx4 * vstep
```

### 2.4 K8 vs V4 对比

| 维度 | K8 (Key) | V4 (Value) |
|------|----------|------------|
| 每行 codes 大小 | 128 bytes (uint8) | 64 bytes (int4 packed) |
| 每行 meta 大小 | 4 bytes (base + step) | 4 bytes (vmin + vstep) |
| 量化位宽 | 8-bit (7-bit + sign) | 4-bit |
| 解包方式 | `code & 0x7F`, `code >> 7` | `int4b_t` 硬件解包, `+8` 转无符号 |
| 反量化公式 | `sign * (base + q7 * step)` | `vmin + idx4 * vstep` |
| Cast 链 | uint8→half→float (2步) | int4b→half (+8)→float (2步) |
| 每 PA block 大小 | 16896 bytes | 8704 bytes |
| 压缩比 (vs fp16) | 128×2 / 132 = **1.94x** | 128×2 / 68 = **3.76x** |

---

## 3. BlockTable 寻址机制

### 3.1 BlockTable 结构

```
blockTable: int32[batchSize][maxBlockNumPerBatch]

示例 (batchSize=2, maxBlockNumPerBatch=4, blockSize=128):
┌──────────────────────────────────────────┐
│ batch=0: [physBlock=5, 3, 7, -1]        │
│ batch=1: [physBlock=2, 8, -1, -1]       │
└──────────────────────────────────────────┘
```

- `blockTable[b][i]` = batch `b` 的第 `i` 个逻辑 block 对应的物理 block ID
- `-1` 表示无效（未分配）

### 3.2 Head 基址计算公式

**K8 Head 基址**（`BrKeyHeadBase`）：

```
headBase = (physBlock * numKvHeads + kvHead) * (blockSize * 132)
           ───────────────────────────────   ─────────────────────
                    ↑ 线性 head 索引                  ↑ 每个 head 占的字节数
```

**展开说明**：
- `physBlock * numKvHeads + kvHead`：将二维 `[physBlock][kvHead]` 展平成一维索引
- `blockSize * 132`：每个 head 在一个 PA block 内占的字节数（128 行 × 132 bytes/行 = 16896 bytes）

**V4 Head 基址**（`BrValHeadBase`）：

```
headBase = (physBlock * numKvHeads + kvHead) * (blockSize * 68)
           ───────────────────────────────   ───────────────────
                    ↑ 同样的线性 head 索引            ↑ 每个 head 占的字节数不同
```

- `blockSize * 68`：每个 head 在一个 PA block 内占的字节数（128 行 × 68 bytes/行 = 8704 bytes）

### 3.3 完整寻址示例

假设当前处理：
- `bIdx = 0`（batch 内第 0 个请求）
- `n2Idx = 3`（KV head 索引）
- `globalS2 = 200`（KV 序列中第 200 个 token）
- `kvCacheBlockSize = 128`
- `numKvHeads = 4`

**Step 1: 逻辑 token → block + posInBlock**
```
blockInBatch = globalS2 / blockSize = 200 / 128 = 1
posInBlock   = globalS2 % blockSize = 200 % 128 = 72
```

**Step 2: blockTable 查物理 block**
```
btIdx = bIdx * maxBlockNumPerBatch + blockInBatch
      = 0 * 4 + 1 = 1
physBlock = blockTable[1] = 3
```

**Step 3: 物理 block + head → KV cache 基址**
```
K8: headBase = (3 * 4 + 3) * (128 * 132) = 15 * 16896 = 253440
V4: headBase = (3 * 4 + 3) * (128 * 68)  = 15 * 8704  = 130560
```

**Step 4: 基址 + posInBlock → 具体行数据**

对于 K8：
```
codeOffset  = headBase + posInBlock * 128        // 第 72 行的 codes
            = 253440 + 72 * 128 = 262656

baseOffset  = headBase + blockSize * 128 + posInBlock * 2   // 第 72 行的 base
            = 253440 + 16384 + 72 * 2 = 269968

stepOffset  = headBase + blockSize * 130 + posInBlock * 2   // 第 72 行的 step
            = 253440 + 16640 + 72 * 2 = 270224
```

对于 V4：
```
codeOffset  = headBase + posInBlock * 64         // 第 72 行的 nibbles
            = 130560 + 72 * 64 = 135168

vminOffset  = headBase + blockSize * 64 + posInBlock * 2
            = 130560 + 8192 + 72 * 2 = 138896

vstepOffset = headBase + blockSize * 66 + posInBlock * 2
            = 130560 + 8448 + 72 * 2 = 139152
```

### 3.4 完整寻址图

```
globalS2 = 200
    │
    ▼
┌─────────────────────────────────┐
│ blockInBatch = 200 / 128 = 1   │
│ posInBlock   = 200 % 128 = 72  │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ blockTable[0][1] = 3           │  ← 物理 block ID
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ K8 headBase = (3 * 4 + 3) *    │
│               (128 * 132)      │  ← K8 head 基址
│             = 253440           │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ K codes:  253440 + 72*128      │  → GM[262656]
│ K base:   253440 + 128*128     │  → GM[269968]
│            + 72*2              │
│ K step:   253440 + 128*130     │  → GM[270224]
│            + 72*2              │
└─────────────────────────────────┘
```

---

## 4. 反量化流程

### 4.1 DequantKvImpl 完整流程

```
┌──────────────────────────────────────────────────────────────────┐
│                    DequantKvImpl(info, isKey=true)               │
│                                                                  │
│  s2Count = info.actualSingleProcessSInnerSize                    │
│  si = 0                                                          │
│                                                                  │
│  ┌─── while (si < s2Count) ──────────────────────────────────┐  │
│  │                                                            │  │
│  │  ① 计算 globalS2, blockInBatch, pos0                      │  │
│  │     globalS2 = s2Idx * s2BaseSize + si                     │  │
│  │     blockInBatch = globalS2 / 128                          │  │
│  │     pos0 = globalS2 % 128                                  │  │
│  │                                                            │  │
│  │  ② BlockTable 查物理 block                                 │  │
│  │     btIdx = bIdx * maxBlockNumPerBatch + blockInBatch      │  │
│  │     physBlock = blockTable[btIdx]                          │  │
│  │                                                            │  │
│  │  ③ 计算 head 基址                                          │  │
│  │     headBase = BrKeyHeadBase(physBlock, n2Idx, numKvH, bs) │  │
│  │                                                            │  │
│  │  ④ Run 扩展: 贪心合并同一物理 block 的连续 token            │  │
│  │     n = 1                                                  │  │
│  │     while (si+n < s2Count && n < maxSub):                  │  │
│  │       gNext = s2Idx * s2BaseSize + si + n                  │  │
│  │       if (gNext / 128 != blockInBatch) break               │  │
│  │       n++                                                  │  │
│  │     // n = 同一 block 内可批量处理的行数 (≤ maxSub=32)      │  │
│  │                                                            │  │
│  │  ⑤ 批量拷贝 codes + meta 到 UB (tmpBuff1)                 │  │
│  │     DataCopy(batchUb, srcGm[codeOffset], n * 128)          │  │
│  │     BrCopyMetaRun(batchUb[meta0], srcGm[baseOffset], n)    │  │
│  │     BrCopyMetaRun(batchUb[meta1], srcGm[stepOffset], n)    │  │
│  │                                                            │  │
│  │  ⑥ 逐行反量化                                              │  │
│  │     for j in [0, n):                                       │  │
│  │       base = BrReadMeta16(batchUb, meta0 + j*32)           │  │
│  │       step = BrReadMeta16(batchUb, meta1 + j*32)           │  │
│  │       BrDecodeKeyRow(codes[j], base, step, out[j])         │  │
│  │       // uint8→half→float, 提取 q7/sign, y=sign*(b+q7*s)  │  │
│  │                                                            │  │
│  │  ⑦ 拷贝反量化结果到 workspace GM                           │  │
│  │     DataCopy(dstWsGm[wsOffset + si*headDimAlign], out,     │  │
│  │              n * headDimAlign)                             │  │
│  │                                                            │  │
│  │  si += n                                                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 BrDecodeKeyRow 详细实现

```cpp
// 输入: codeUb (128 bytes uint8), base (float), step (float)
// 输出: outUb (128 elements half/bf16)

// Step 1: uint8 → half → float (两次 Cast)
Cast(halfScratch, codeUb, RoundMode::CAST_NONE, n);      // uint8 → half
Cast(scratchA, halfScratch, RoundMode::CAST_NONE, n);    // half → float

// Step 2: 提取 sign bit (floor(code / 128))
Muls(scratchB, scratchA, 1.0f / 128.0f, n);
Cast(signI16, scratchB, RoundMode::CAST_FLOOR, n);       // float → int16
Cast(scratchB, signI16, RoundMode::CAST_NONE, n);        // int16 → float

// Step 3: 提取 q7 (code - sign * 128)
Muls(outFp32, scratchB, 128.0f, n);
Sub(outFp32, scratchA, outFp32, n);

// Step 4: 计算 sign_val (1 - 2 * sign)
Muls(scratchB, scratchB, -2.0f, n);
Adds(scratchB, scratchB, 1.0f, n);

// Step 5: 计算 y = sign_val * (base + q7 * step)
Muls(outFp32, outFp32, step, n);
Adds(outFp32, outFp32, base, n);
Mul(outFp32, outFp32, scratchB, n);

// Step 6: float → half/bf16
Cast(outUb, outFp32, RoundMode::CAST_RINT, n);
```

**总共 3 次 Cast**：
1. uint8 → half
2. half → float
3. float → half/bf16（最终输出）

**AscendC VEC Cast 限制**：不支持 uint8 → float 直接转换，必须经过 half 中间步骤。

### 4.3 BrDecodeValueRow 详细实现

```cpp
// 输入: nibbleUb (64 bytes packed int4), vmin (float), vstep (float)
// 输出: outUb (128 elements half/bf16)

// Step 1: int4b_t → half (硬件解包)
Cast(halfScratch, nibbleUb.ReinterpretCast<int4b_t>(), RoundMode::CAST_NONE, headDim);

// Step 2: 有符号 int4 → 无符号 [0, 15]
Adds(halfScratch, halfScratch, static_cast<half>(8.0f), headDim);

// Step 3: half → float
Cast(idx4F32, halfScratch, RoundMode::CAST_NONE, headDim);

// Step 4: V = vmin + idx4 * vstep
Muls(outFp32, idx4F32, vstep, headDim);
Adds(outFp32, outFp32, vmin, headDim);

// Step 5: float → half/bf16
Cast(outUb, outFp32, RoundMode::CAST_RINT, headDim);
```

### 4.4 Run 扩展优化

这是 P2 阶段的关键优化——**同一物理 block 内的连续 token 批量处理**：

```
假设 s2Count = 50, blockSize = 128, maxSub = 32

globalS2:  120  121  122  ... 127 | 128  129  ... 177
block:       0    0    0   ...   0 |   1    1   ...   1
posInBlock: 120  121  122  ... 127|   0    1   ...  49

Run 1: n = 8  (pos 120~127, 同一 block 0, 剩余 8 行)
Run 2: n = 32 (pos 0~31, 同一 block 1, maxSub 限制)
Run 3: n = 10 (pos 32~41, 同一 block 1, s2Count 耗尽)
```

**收益**：
1. **减少 blockTable 查找次数**：从 50 次降到 3 次
2. **批量 DataCopy**：codes 连续拷贝，提高 GM→UB 带宽利用率
3. **减少 pipeline flush**：一次 setup 处理多行

**约束**：
- `maxSub ≤ 32`（`BR_S2_SUB_MAX`），受 tmpBuff1 的 32KB 容量限制
- 每行需要 `codeRowBytes + outRowBytes + 2 * metaSlot` = 128 + 256 + 64 = 448 bytes
- 32 行 × 448 = 14336 bytes < 32KB ✓

### 4.5 Meta 拷贝的特殊处理

GM 端 base 数组是连续的（2B stride），但 UB 端必须 32B 对齐：

```cpp
// VEC Cast of half/bf16 requires 32B-aligned UB addresses (packed 2B stride faults).
```

所以不得不逐行拷贝到 32B slot：

```
GM 端: [base0][base1][base2]...   ← 连续 2B stride
         ↓       ↓       ↓        ← 逐行 DataCopyPad
UB 端: [base0___32B___][base1___32B___][base2___32B___]
         ↑ 32B slot        ↑ 32B slot     ↑ 每个只用了 2B，浪费 30B
```

**性能影响**：对于 n=32 行，需要 32 次 2B 的 DataCopyPad，每次都有 MTE 的 setup 开销。

---

## 5. UB 内存预算分析

### 5.1 tmpBuff1 (32KB) 内部布局

```
tmpBuff1 (32KB = 32768 bytes) 内部布局:

┌─────────────────── codes ───────────────────┬──── meta0 ────┬──── meta1 ────┬──── out ────┐
│ n × codeRowBytes                            │ n × 32B slot  │ n × 32B slot  │ n × outRowBytes │
│ K8: n × 128B    V4: n × 64B                │ 每行 32B 对齐  │ 每行 32B 对齐  │ K8/V4: n × 256B │
└─────────────────────────────────────────────┴───────────────┴───────────────┴───────────────┘
```

| 变量 | K8 值 | V4 值 | 含义 |
|------|-------|-------|------|
| `codeRowBytes` | 128 | 64 | 每行压缩码的字节数 |
| `metaSlot` | 32 | 32 | 每行 meta 在 UB 中占的 slot 大小（32B 对齐） |
| `outRowBytes` | 256 | 256 | 反量化输出每行大小 = `headDimAlign × sizeof(WS_T)` |

**每行实际占用**（含对齐 padding）：

```
K8: 128 + 32 + 32 + 256 = 448 bytes/row
V4: 64  + 32 + 32 + 256 = 384 bytes/row
```

### 5.2 中间 Cast 数据存储

`BrDecodeKeyRow` 需要 4 个工作 buffer，**都在 tmpBuff1 之外独立分配**：

```
UB 空间分配全景 (~192KB):

┌──────────────────────────────────────────────────────────────┐
│ tmpBuff1 (32KB)                                              │
│ [codes | meta0 | meta1 | out]                                │
├──────────────────────────────────────────────────────────────┤
│ dequantFp32Buf_ (3 × headDimAlign × 4 = 1536B)              │
│ [fp32UbA | fp32UbB | outFp32]                                │
│ ← BrDecodeKeyRow 的 scratchA, scratchB, 临时 decoded          │
├──────────────────────────────────────────────────────────────┤
│ dequantFp16Buf_ (headDimAlign × 2 = 256B)                    │
│ [halfScratch]                                                │
│ ← uint8→half 的中间 cast 结果                                 │
├──────────────────────────────────────────────────────────────┤
│ 其他队列: softmax, queues, tmpBuff2, ...                      │
│ (~155KB)                                                     │
└──────────────────────────────────────────────────────────────┘
```

### 5.3 maxSub 上限计算

```
K8 最大行数:
  每行 448B, 32KB = 32768B
  32768 / 448 = 73.1 → 最多 73 行

V4 最大行数:
  每行 384B, 32KB = 32768B
  32768 / 384 = 85.3 → 最多 85 行
```

**当前 `BR_S2_SUB_MAX = 32` 是保守值，UB 空间实际支持到 K8=73 行 / V4=85 行。**

---

## 6. 性能瓶颈与优化思路

### 6.1 当前性能瓶颈

1. **反量化开销占比高**：AIV 反量化时间 > AIC matmul 时间
2. **内存带宽放大**：K8 约 4x，V4 约 7.5x
3. **UB 几乎用满**：~184KB / 192KB (95.8%)，优化空间有限
4. **Meta 拷贝效率低**：逐行 2B 的 DataCopyPad，每次都有 MTE setup 开销

### 6.2 短期优化（P3 阶段）

#### 优化 1：扩展到 64 行批量处理

**可行性**：
```
K8, 64 行: 64 × 448 = 28672B = 28KB < 32KB ✓
V4, 64 行: 64 × 384 = 24576B = 24KB < 32KB ✓
```

**改动**：
```cpp
// br_dequant_device.h
static constexpr uint32_t BR_S2_SUB_MAX = 64U;  // 从 32 改为 64
```

**收益**：
- blockTable 查找次数减半
- codes DMA 拷贝次数减半（更大的 burst，更好的带宽利用率）
- **整体反量化加速约 10-20%**

#### 优化 2：Meta 批量拷贝 + UB 内展开

**当前**：
```cpp
for (uint32_t j = 0U; j < numRows; ++j) {
    DataCopyPad(ubDst[j * 32], srcGm[metaGmOff + j * 2], metaParams, padParams);
}
```

**优化**：
```cpp
// 一次拷贝 n*2 bytes 到连续 UB
DataCopy(ub_contiguous, srcGm[metaGmOff], n * 2);

// 然后用 VEC 或 MTE 指令展开到 32B slot
// （需要 AscendC 支持 scatter 或类似指令）
```

**收益**：n 次 DMA → 1 次 DMA，对于 n=32，减少 31 次 DMA 开销。

#### 优化 3：预取 + 双缓冲

**当前**：
```cpp
// 串行：读取 → 反量化 → 计算
LoadL1(codes);
BrDecodeKeyRow(codes, base, step, dequant_buf);
MatMul(query, dequant_buf, result);
```

**优化**：
```cpp
// 双缓冲流水线
__local__ float dequant_buf[2][128];  // ping-pong buffer

for (int block = 0; block < num_blocks; block++) {
    // 异步加载下一个 block
    if (block + 1 < num_blocks) {
        LoadL1_Async(codes[block+1], staging_buf[(block+1) % 2]);
    }
    
    // 反量化当前 block
    BrDecodeKeyRow(staging_buf[block % 2], base, step, dequant_buf[block % 2]);
    
    // 计算当前 block
    MatMul(query, dequant_buf[block % 2], result);
    
    // 等待下一个 block 加载完成
    if (block + 1 < num_blocks) {
        WaitLoad();
    }
}
```

**收益**：隐藏内存加载延迟，约 20-30% 整体加速。

### 6.3 中期优化（1-2 个月）

#### 优化 4：队列合并

**当前**：
```cpp
// 多个独立队列
Queue<K_dequant> k_queue;
Queue<V_dequant> v_queue;
Queue<Softmax> softmax_queue;
```

**优化**：
```cpp
// 合并为统一队列
struct DequantTask {
    enum Type { K8, V4 };
    Type type;
    void* input;
    void* meta;
    void* output;
};

Queue<DequantTask> unified_queue;
```

**收益**：
- 减少队列切换开销 ~30%
- 提高指令发射效率 ~20%
- 目标：msprof ≤ TQ FIA +15%

#### 优化 5：合并 KV Cache 读取

**当前**：
```cpp
// K 和 V 分开读取
ReadKCache(k_codes, k_base, k_step);
ReadVCache(v_nibbles, v_vmin, v_vstep);
```

**优化**：
```cpp
// 合并读取，减少 HBM 事务开销
struct KVBlock {
    uint8_t k_codes[128];
    uint8_t k_meta[4];    // base + step
    uint8_t v_nibbles[64];
    uint8_t v_meta[4];    // vmin + vstep
};

// 一次性读取 200 bytes/block
ReadKVCache(kv_block);
```

**收益**：减少 HBM 事务数 ~40%，提升带宽利用率。

**约束**：需要修改 pack 布局，**影响上游 pack 算子**。

### 6.4 长期优化（3-6 个月）

#### 优化 6：细粒度流水线编排

**当前**：
```cpp
// 粗粒度：整个 block 为单位
for each block:
    AIV: DequantK();
    AIC: MatMulQK();
    AIV: Softmax();
    AIV: DequantV();
    AIC: MatMulSV();
```

**优化**：
```cpp
// 细粒度：按 row 流水线
for each row in block:
    AIV: DequantK_Row(row);      // 反量化第 row 行
    AIC: MatMulQK_Row(row);      // 计算第 row 的 QK
    AIV: Softmax_Row(row);       // softmax 第 row
    
    // 同时：
    AIV: DequantK_Row(row+1);    // 预取下一行
    AIC: MatMulQK_Row(row+1);
```

**收益**：AIC/AIV 利用率从 ~50% 提升到 ~85%。

**约束**：需要重构 kernel 代码，工作量大。

### 6.5 优化优先级总结

| 优化项 | 预期收益 | 实施难度 | 优先级 |
|--------|---------|---------|--------|
| 扩展到 64 行批量 | 10-20% 反量化加速 | 低 | **P0** |
| Meta 批量拷贝 | 30-50% meta 拷贝加速 | 中 | **P1** |
| 预取 + 双缓冲 | 20-30% 整体加速 | 中 | **P2** |
| 队列合并 | 20-30% 整体加速 | 中 | **P3** |
| 合并 KV Cache 读取 | 40% HBM 事务减少 | 高 | **P4** |
| 细粒度流水线编排 | 35% AIC/AIV 利用率提升 | 高 | **P5** |

---

## 附录

### A. 关键常量定义

```cpp
// br_pack_layout.h
static constexpr uint32_t BR_HEAD_SIZE = 128U;
static constexpr uint32_t BR_BLOCK_ROWS = 16U;
static constexpr uint32_t BR_KEY_CODE_BYTES = BR_HEAD_SIZE;           // 128
static constexpr uint32_t BR_VAL_CODE_BYTES = BR_HEAD_SIZE / 2U;      // 64
static constexpr uint32_t BR_META_BYTES = 4U;                         // base+step or vmin+vstep
static constexpr uint32_t BR_KEY_ROW_BYTES = BR_KEY_CODE_BYTES + BR_META_BYTES;  // 132
static constexpr uint32_t BR_VAL_ROW_BYTES = BR_VAL_CODE_BYTES + BR_META_BYTES;  // 68
static constexpr uint32_t BR_KEY_TILE_BYTES = BR_BLOCK_ROWS * BR_KEY_ROW_BYTES;  // 2112
static constexpr uint32_t BR_VAL_TILE_BYTES = BR_BLOCK_ROWS * BR_VAL_ROW_BYTES;  // 1088

// br_dequant_device.h
static constexpr uint32_t BR_S2_SUB_MAX = 32U;
static constexpr uint32_t BR_META_SLOT_BYTES = 32U;
```

### B. dtype 合约

Pack 元数据（base/step/vmin/vstep）使用 2-byte floats，匹配 pack 输入 dtype（half 或 bfloat16）。**FIA 必须使用相同的 dtype 解码**——将 bf16 元数据读为 half（或反之）会导致灾难性的反量化值。

### C. 相关文档

- [DESIGN.md](../DESIGN.md) - 算子设计文档
- [MANIFEST.txt](../MANIFEST.txt) - 文件清单
