# BitResidual FIA：dtype 流与 Dequant 位提取示例

> 基线：**P18**（dequant 中间计算 fp32；不含 P19 half）。  
> 源码：`br_dequant_device.h`、`fia_block_vec_turboquant_p0.h`、`fia_block_cube_turboquant_p0.h`、`vector_common.h`。  
> 关联：decode 指令级细节见 [`br_dequant_decode_flow.md`](br_dequant_decode_flow.md)。

## 约定

| 符号 | 含义 |
|------|------|
| `Q_T` / `OUT_T` / `WS_T` / `WS_KV_T` | 模型 dtype，三者绑定为同一类型 |
| B16 | fp16 或 bf16（由模型决定） |
| MetaT | pack 时写入的 2B 浮点，与 pack 输入 dtype 一致 |

**用例对照**

| | Qwen serving longquery | L6 harness |
|--|--|--|
| `Q_T` / `OUT_T` / meta | **bf16** | **fp16** |
| MM / softmax 累加 | fp32 | fp32 |

---

## 1. 端到端 dtype 流

```
输入 (GM)
  query           : Q_T   (B16)
  key/value_cache : uint8 (K8: 1B/元素; V4: nibble 打包)
  pack meta       : MetaT (B16, base/step 或 vmin/vstep)
  rotation Π      : Q_T   (B16)
  attn_mask       : bool  (1B/元素，仅作谓词)

       ┌──────────────── AIV: DEQUANT ────────────────┐
       │ Key:  u8 → half → int16 → 位提取 → fp32 affine │
       │ Val:  int4b → half → fp32 affine               │
       │ meta: MetaT → fp32                             │
       │ ▓▓ CAST #1: fp32 → WS_T(=Q_T, B16) ▓▓          │
       │ → dequant K/V workspace (GM, B16)              │
       └───────────────────────┬────────────────────────┘
                               │ K_ws, V_ws : B16
       ┌─────────────── CUBE: MM1 = Q·Kᵀ ───────────────┐
       │ L0A Q / L0B K : B16；Mmad 累加 L0C : fp32       │
       │ Fixpipe → mm1Res (GM) : fp32（NoQuant）         │
       └───────────────────────┬────────────────────────┘
                               │ scores : fp32
       ┌──────── AIV: scale + mask + SOFTMAX ───────────┐
       │ mmResUb: fp32 进 → fp32 出（主线不改宽度）       │
       │ ① Muls(scale) : fp32 × fp32                     │
       │ ② PSE(可选)   : GM B16 ─Cast→ fp32，再 Add      │
       │ ③ attn_mask   : bool 字节谓词 + SelectWithBytes │
       │                 Mask；被选中的是 fp32 分数或     │
       │                 -1e12(fp32)，bool 不参与算术     │
       │ ④ SoftmaxFlashV2 : COMPUTE_T = fp32             │
       │ ▓▓ CAST #2: P fp32 ─Cast(ROUND)→ WS_T(B16) ▓▓  │
       └───────────────────────┬────────────────────────┘
                               │ P : B16
       ┌─────────────── CUBE: MM2 = P·V ────────────────┐
       │ L0A P / L0B V : B16；累加 fp32；mm2Res GM fp32  │
       └───────────────────────┬────────────────────────┘
                               │ O(partial) : fp32
       ┌──────── AIV: Bmm2CastAndCopyOut ───────────────┐
       │ ▓▓ CAST #3: fp32 → OUT_T(=Q_T) ▓▓               │
       │   bf16 → CAST_RINT；fp16 → CAST_ROUND           │
       │ → attention_out (GM, B16)                       │
       └────────────────────────────────────────────────┘

旁路 Q@Π（Cube Fixpipe，非 vector Cast）:
  Q(B16)·Π(B16) → fp32 累加 → Fixpipe(F322F16/F322BF16) → Q_rot(B16, L1)

FlashDecode（长 KV 分片）:
  各 S2 分片 O 以 fp32 写 accumOut → combine 后同 CAST #3
```

### 1.1 三处 vector Cast（主路径）

| # | 位置 | 方向 | 用途 |
|---|------|------|------|
| CAST #1 | `BrDecode*Tile` 末尾 | fp32 → `WS_T` | 写 dequant KV workspace |
| CAST #2 | `DealBmm1ResBaseBlock` | fp32 → `WS_T` | softmax 概率 P 喂给 MM2 |
| CAST #3 | `Bmm2CastAndCopyOut` | fp32 → `OUT_T` | 最终 attention 输出 |

另有：**PSE**（可选）B16→fp32；**meta** MetaT→fp32；**Key 位提取前** u8→half→int16（见 §2）。

### 1.2 为何 `WS_T = OUT_T = Q_T`

Cube `Mmad` 不接受 `bf16 × half` 混算，故 dequant 写出的 K/V、softmax 写出的 P 必须与 query 同 dtype。累加与打分恒为 fp32（`T` / `COMPUTE_T` / `MM*_OUT_T = float`），与模型 dtype 无关。

---

## 2. Dequant：为什么 uint8 要先变成 int16

### 2.1 两条硬件硬约束（910B / dav_c220）

**约束 A — VEC 位运算没有 8-bit lane**

- `And` / `vand`：仅 `int16_t` / `uint16_t`
- `ShiftRight` / `vshr`：仅 `int16_t` / `uint16_t` / `int32_t` / `uint32_t`

`uint8_t` 不在列表中 → 编译期失败，不是“慢”。

**约束 B — `uint8 → int16` 无直连 Cast**

910B `CastIntrinsicsImpl` 相关窄类型路径（节选）：

| dst ← src | 支持 |
|-----------|------|
| half ← uint8 | ✅ |
| half ← int4b | ✅ |
| int16 ← half | ✅ |
| half ← int16 | ✅ |
| **int16 ← uint8** | ❌ |
| **float ← uint8** | ❌ |

因此 Key 必须走：`uint8 → half → int16`。`half` 精确表示 0..2048 整数，该往返对 0..255 **无损**。

文件头注释：

```text
AscendC on 910B does not support Cast<int16, uint8>; widen via half then int16.
```

### 2.2 为何不直接在 uint8 缓冲上做 SWAR

把两字节 `ReinterpretCast<uint16>` 做跨字节位运算看似省 widen，实际不划算：

1. `>>1` 会把高字节 bit0 挤进低字节 bit7，需额外 mask
2. affine 要求每元素独占 half/fp32 lane；没有“一 lane 拆两元素”的指令，最终仍要拆回
3. `Cast u8→half` 本身带宽满，占 Key tile 指令比例有限

---

## 3. Key（K8）位提取示例

编码：`code = (q7 << 1) | sign`（LSB = 符号）。  
解码：\(y = (1 - 2\cdot\mathrm{sign})\cdot(\mathrm{base} + q7\cdot\mathrm{step})\)。

源码路径（P18）：`BrDecodeKeyTile`。

### 3.1 数值例子

设某元素 `code = 0xB5`，行 meta `base = 0.01`，`step = 0.02`：

```
GM 一字节:  1 0 1 1 0 1 0 1  = 0xB5 = 181
            └──── q7 ────┘ ↑
                         sign

① Cast u8→half     181.0h          ← 约束 B：uint8 唯一出口
② Cast half→int16  0x00B5          ← 约束 A：位运算最窄 16-bit
                                     高 8 位为 0，不串扰邻元素

③ And(uint16, 0x0001) → sign = 1
   Cast i16→fp32 → 1.0f
   Muls(-2) Adds(+1) → signMul = -1.0f

④ ShiftRight(uint16, 1) → q7 = 90
   Cast i16→fp32 → 90.0f

⑤ affine: base + q7*step = 0.01 + 90*0.02 = 1.81f
⑥ Mul(signMul): 1.81 * (-1) = -1.81f
⑦ CAST #1: fp32 → WS_T (fp16 或 bf16)
```

要点：

- `ShiftRight` 用 **uint16 视图** → 逻辑右移（避免 int16 算术右移语义）
- sign 在 **LSB**：`And(&1)` 与 `ShiftRight(>>1)` 源数据可重叠读完再写，中间可少插 barrier（P13b）

### 3.2 指令序列（与代码对应）

```text
codesUb:uint8
  → Cast → halfScratch:half
  → Cast → codeI16:int16 (scratchA reinterpret)
  → Duplicate mask 0x01; And → signStorage
  → ShiftRight → q7 in codeU16
  → Cast sign → fp32; Muls/Adds → ±1
  → Cast q7 → fp32
  → BrApplyRowAffine(base, step) → |y|
  → Mul(sign) → y
  → Cast → OutT
```

---

## 4. Value（V4）示例：不需要手写位提取

Value 每元素 4 bit，硬件有原生 `int4b_t`；`Cast half ← int4b_t` 一次完成：**拆 nibble + 符号扩展 + 转浮点**。  
故 Value **没有** Key 那条 `And`/`ShiftRight` 路径。

公式（P17b-B）：调用方先折算 \(\mathrm{vmin}' = \mathrm{vmin} + 8\cdot\mathrm{vstep}\)，再  
\(y = \mathrm{vmin}' + s\cdot\mathrm{vstep}\)，\(s = \mathrm{Cast}(\mathrm{int4})\in[-8,7]\)。  
等价于无符号 \(u\in[0,15]\) 的 \(y = \mathrm{vmin} + u\cdot\mathrm{vstep}\)。

### 4.1 数值例子

设某字节 `0x3A`（两元素），`vmin = -1.0`，`vstep = 0.1`：

```
GM:  0 0 1 1  1 0 1 0  = 0x3A
     └高nib┘  └低nib┘
     elem1=3   elem0=0xA

ReinterpretCast<int4b_t>（two's complement）:
  elem0: 0xA → -6
  elem1: 0x3 → +3

Cast int4b→half→fp32:
  -6.0f , 3.0f

vmin' = -1.0 + 8*0.1 = -0.2
affine:
  elem0: -0.2 + (-6)*0.1 = -0.8
  elem1: -0.2 +  3 *0.1 =  0.1

校验（无符号 u = s+8）:
  u=2  → -1.0 + 2*0.1  = -0.8  ✓
  u=11 → -1.0 + 11*0.1 =  0.1  ✓
```

### 4.2 与 Key 对比

| | Key K8 | Value V4 |
|--|--|--|
| GM 存储 | 1B/元素 | 0.5B/元素（两 nibble/字节） |
| 是否手写位提取 | 是（And / ShiftRight） | 否（`int4b_t` Cast 内含） |
| widen 路径 | u8→half→int16→…→fp32 | int4b→half→fp32 |
| 为何不同 | “7bit 量级 + 1bit 符号”不是硬件 dtype | 4-bit 有原生 `int4b_t` |

---

## 5. Softmax 段 mask：纠正「fp32 与 int8 混算」误解

**打分张量 `mmResUb` 从 MM1 读入到 CAST #2 前始终为 fp32。**

`attn_mask`（bool / int8 兼容）只作 `SelectWithBytesMask` 的**字节谓词**：每个位置在「原 fp32 分数」与「fp32 标量 `-1e12`」之间二选一，bool **不进入** Mul/Add。

`-1e12` 以 `uint32_t` 位模式存在 `maskInfo.maskValue` 中，使用时 `*((T *)&maskValue)` reinterpret 回 fp32，无数值转换。

本段唯一可能的操作数 Cast：可选 **PSE**（GM B16 → fp32 再 Add）。

---

## 6. 小结

1. **带宽面**（Q / K_ws / V_ws / P / Π / out）= `Q_T`（B16）；**算分与累加** = fp32。  
2. **主路径三次 Cast**：dequant 出、softmax 概率出、最终输出。  
3. Key 必须 `u8→half→int16`：910B 无 8-bit 位运算、无 `int16←uint8` Cast。  
4. Value 用 `int4b_t` Cast，位提取被硬件吸收。  
5. mask 是谓词，不是 fp32 算术操作数。
