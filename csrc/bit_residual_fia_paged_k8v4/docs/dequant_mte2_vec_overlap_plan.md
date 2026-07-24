# Dequant MTE2⇄VEC 跨 run 重叠实施方案

> 算子: `BitResidualFiaPagedK8v4`  
> 分支: `v0.18.0-branch-dev`  
> 基线: P9b + P13-P15 + P13b + P17a + P17b-B(L6 kv=2000: decode ~807µs / prefill ~834µs)  
> 目标: 在 AIV-bound 瓶颈下,把下一个 run 的 MTE2(codes/meta 读 DMA)藏进当前 run 的 VEC decode 窗口,预期 **8-15%** 墙钟收益。  
> 关联文档: [dequant_opt_p0_p12_report.md](dequant_opt_p0_p12_report.md)、[opt_a1_a2_b1_report.md](opt_a1_a2_b1_report.md)

---

## 1. 背景与现状

### 1.1 已合入的优化(不可再动)

| 编号 | 内容 | 原因 |
|------|------|------|
| P0/P1 | meta 批量 Cast + bulk DMA | 消除 `GetValue`/`V_S` |
| P4/P5/P6 | BrApplyRowAffine / ShiftRight / V4 tile=13 | K8/V4 decode 瘦身 |
| P9b | 整 run 1 次 meta DMA+Cast(768B UB,64 行容量) | 消除 per-tile meta 重复 |
| **P13-P15** | K8 Cast/Muls+Adds 链合并 + **O(1) run-length** | barrier 已压到 RAW 必要项;scalar 循环已删 |
| **P13b** | LSB-sign 布局 `code = (q7<<1)|sign` | And→ShiftRight 不需 barrier |
| **P17a** | BrApplyRowAffine headDim=128 展开 | 删 column for/if 控制流 |
| **P17b-B** | Value `+8` 折进 vmin(每 run 一次) | 消除 per-tile Adds |

### 1.2 已否决的方向(勿再做)

| 方向 | 编号 | 结果 |
|------|------|------|
| 减 K8 PipeBarrier | P7a | +17~26µs 变慢(barrier 承担流水时序) |
| V4 16-entry LUT + Gather | P8 | +90% 变慢(随机访 UB >> 规则 Mul/Add) |
| 扩 maxSub / tile | P11 | UB 已满,OOB |
| UB bank pad | P10 | 冲突率降但墙钟不动 |
| A1 MTE3⇄MTE2 双缓冲 | — | 仅 -0.4%(重叠的是两个小窗口) |

### 1.3 瓶颈定位

算子 **AIV-bound**(`vec_ratio≈0.21`,`cube_ratio~1%`)。AIV 上有两类硬件单元:
- **MTE2**:GM → UB 的读 DMA(codes + meta)
- **VEC**:decode 指令链(Cast/ShiftRight/And/Mul/Add/Brcb)

当前 `DequantKvImpl` 一个 run 内部 **MTE2 与 VEC 严格串行**,无重叠:

```
当前单 run(AIV 视角):
┌─MTE2 codes─┤├─Wait─┤├─MTE2 meta─┤├─Wait─┤├═══ VEC decode ═══┤├─Wait─┤├─MTE3─┤
   VEC 空闲▲              VEC 空闲▲                VEC 满▲      MTE2 空闲▲
```

- MTE2 期间 VEC 空闲,VEC 期间 MTE2 空闲 → **单核内两类单元互不重叠**
- A1 只重叠了 MTE3(写,`mte3_ratio≈0.014-0.019`,极短)与下一 run 的 MTE2 → 收益小
- 真正的长窗口是 **VEC decode**(瓶颈本身),本方案把 **MTE2 读**藏进这个窗口

---

## 2. 方案总览

### 2.1 核心思想

在 `DequantKvImpl` 的 run 循环内,**当前 run 进入 VEC decode 时,提前发起下一个 run 的 codes/meta MTE2 DMA**,不 Wait;下一个 run 真正要用 codes 时才 `WaitFlag(MTE2_V)`。

```
run r:    ┌─MTE2 codes_r → MTE2 meta_r─┐
                                      │
                                      ▼
         ╔═══════════ VEC decode_r ═══════════╗  ← VEC 满,MTE2 空闲窗口
         ║   期间发起 run r+1 的 MTE2:        ║
         ║   ┌─MTE2 codes_{r+1} → meta_{r+1}─┐║  ← 藏进 VEC 窗口
         ╚═══╧════════════════════════════════╧╝
                                          ▼
run r+1:                              Wait codes_{r+1} → VEC decode_{r+1} → ...
```

AIV 的 MTE2 单元与 VEC 单元 **硬件并行**,这是纯单核内 DMA⇄计算重叠,不跨核、不新增同步原语。

### 2.2 为什么收益高于 A1

| 方案 | 重叠的两个窗口 | 重叠时长 | 实测/估计 |
|------|----------------|----------|-----------|
| A1 | MTE3(写) ⇄ MTE2(读) | 极短(`mte3_ratio≈0.014`) | -0.4% |
| **本方案** | MTE2(读) ⇄ VEC(decode) | VEC 是最长窗口 | **8-15%** |

AIV-bound 下,直接扩 VEC 利用率的边际收益最大。

---

## 3. 关键约束(实现前必须确认)

### 3.1 UB 生命周期约束

跨 run 重叠时,**run r 的 codes/meta 不能被 run r+1 的 DMA 覆盖**。

| Buffer | 当前 | 改造要求 |
|--------|------|----------|
| codes+out `batchUb` | A1 已 ping-pong(`bufIdx = runId % 2`,2× `kHalfBytes`) | ✅ 已满足,直接复用 |
| **meta `packedMetaUb`** | 单 768B 区(`dequantInt8Buf_`) | ❌ **需 ping-pong,否则跨 run 冲突** |
| meta fp32 `meta0Fp32/meta1Fp32` | 单区(`dequantInt8Buf_` 内偏移) | ❌ 同上,随 meta 区一起 ping-pong |
| fp32/half decode scratch | `tmpBuff1` 尾部,A1 已 ping-pong | ✅ |

**核心改动**:meta 区从单 768B 拆成 2×768B ping-pong。

### 3.2 当前 meta UB 布局(P9b)

```
dequantInt8Buf_ (单区,768B 对齐)
┌── meta0 half ×64 ──┬── meta1 half ×64 ──┬── meta0 fp32 ×64 ──┬── meta1 fp32 ×64 ──┐
│ 0              128  │ 128          256   │ 256          512   │ 512          768   │
└─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘
  ← P1 bulk DMA 落点                          ← P0 Cast 落点(Brcb 源,32B 对齐)
```

定义见 [br_dequant_device.h:43-57](../op_kernel/vendored/arch32/br_dequant_device.h#L43-L57):
- `BR_DEQUANT_UB_BYTES = 768`(单区总字节)
- `BR_META_FP32_0_UB_OFF = 256`、`BR_META_FP32_1_UB_OFF = 512`

### 3.3 UB 容量约束(需扩容 meta 区)

`dequantInt8Buf_` 当前分配([fia_block_vec_turboquant_p0.h:364](../op_kernel/vendored/arch32/fia_block_vec_turboquant_p0.h#L364)):

```cpp
uint32_t int8Bytes = ((br_dequant::BR_DEQUANT_UB_BYTES + 31U) / 32U) * 32U;  // = 768
```

ping-pong 后需 2×768 = 1536B。需确认 AIV UB 总账能否容纳 +768B:

- `dequantInt8Buf_` / `dequantFp32Buf_` / `dequantFp16Buf_` 是 **独立 TBuf**,不占 `tmpBuff1` 32KB
- P9b 注释明确 meta 专用区与 decode scratch(`tmpBuff1` 尾部)互不抢空间
- AIV UB ~192KB,当前三块 dequant TBuf 合计约 768 + 1.5KB + 0.5KB ≈ 2.7KB,余量充足
- **结论:+768B 安全**,但需实测确认(见 §6 验证)

### 3.4 事件 ID 约束

当前 `DequantKvImpl` 已用事件 ID([fia_block_vec_turboquant_p0.h:1353-1359](../op_kernel/vendored/arch32/fia_block_vec_turboquant_p0.h#L1353-L1359)):

```cpp
eventIdVWaitMte2      // MTE2_V  codes/meta→VEC 的 Wait
eventIdVWaitMte3      // V_MTE3  VEC→MTE3 的 Wait
eventIdMte3WaitV0     // MTE3_V  ping-pong half0 复用
eventIdMte3WaitV1     // MTE3_V  ping-pong half1 复用
eventIdMte2WaitS      // S_MTE2  blockTable GetValue 同步
eventIdMte2WaitVMeta  // V_MTE2  meta DMA 前的 Wait(等 codes decode?实际为流水隔离)
```

跨 run 重叠需 **额外 2 个事件 ID**(分别追踪 run r+1 的 codes/meta DMA 完成):
- `eventIdNextCodesMte2V`(MTE2_V):下一 run codes/meta DMA → VEC 的完成事件
- 可能还需 `eventIdNextMetaMte2V` 或复用上面的

**风险**:Ascend 910B 每类 HardEvent 事件 ID 数量有限。需在实现时 `FetchEventID` 后确认未超限;若超限,退化为"只重叠 codes、meta 仍串行"(收益约 5-8%)。

---

## 4. 详细实施步骤

### 步骤 1:meta 区 ping-pong 化(前置依赖)

**目标**:把 `dequantInt8Buf_` 从 768B 单区扩成 2×768B,run r/r+1 各用一半。

**改动 1 — `br_dequant_device.h`** 增加双区偏移常量:

```cpp
// br_dequant_device.h,在现有 BR_META_FP32_1_UB_OFF 之后新增

// P18: meta ping-pong for MTE2⇄VEC cross-run overlap.
// Two independent 768B meta regions; run r uses slot (runId % 2).
static constexpr uint32_t BR_META_SLOT_BYTES = BR_DEQUANT_UB_BYTES;  // 768
static constexpr uint32_t BR_META_SLOT0_OFF = 0U;
static constexpr uint32_t BR_META_SLOT1_OFF = BR_META_SLOT_BYTES;
static constexpr uint32_t BR_DEQUANT_UB_BYTES_PINGPONG =
    2U * BR_META_SLOT_BYTES;  // 1536
```

`BR_META_FP32_0_UB_OFF` / `BR_META_FP32_1_UB_OFF` 保持不变(它们是 slot 内偏移,slot 基址另算)。

**改动 2 — `fia_block_vec_turboquant_p0.h` InitBuffers** 扩容 `dequantInt8Buf_`:

```cpp
// fia_block_vec_turboquant_p0.h:364
// 旧: uint32_t int8Bytes = ((br_dequant::BR_DEQUANT_UB_BYTES + 31U) / 32U) * 32U;
// 新: P18 ping-pong meta region (2×768B)
uint32_t int8Bytes = ((br_dequant::BR_DEQUANT_UB_BYTES_PINGPONG + 31U) / 32U) * 32U;
```

**改动 3 — `DequantKvImpl` 内取 meta tensor** 按 runId 选 slot:

```cpp
// fia_block_vec_turboquant_p0.h:1344 附近
// 旧: LocalTensor<uint8_t> packedMetaUb = dequantInt8Buf_.Get<uint8_t>();
// 新:
uint32_t metaSlotOff = (runId % 2U) * br_dequant::BR_META_SLOT_BYTES;
// 注: runId 在循环内才递增,需在循环体顶部算 slotOff
LocalTensor<uint8_t> packedMetaUb =
    dequantInt8Buf_.GetWithOffset<uint8_t>(
        br_dequant::BR_DEQUANT_UB_BYTES, metaSlotOff);
LocalTensor<float> meta0Fp32 = packedMetaUb[
    br_dequant::BR_META_FP32_0_UB_OFF].template ReinterpretCast<float>();
LocalTensor<float> meta1Fp32 = packedMetaUb[
    br_dequant::BR_META_FP32_1_UB_OFF].template ReinterpretCast<float>();
```

> ⚠️ `runId` 当前在 while 循环体底部 `++runId`,slot 选择要在循环体顶部用当前 `runId`。同时下一个 run 的 DMA 要写到 `(runId+1) % 2` slot。

### 步骤 2:跨 run MTE2 预取(核心改动)

**目标**:在 run r 的 VEC decode 期间,发起 run r+1 的 codes+meta MTE2 DMA。

**当前 run 循环结构**(简化,[fia_block_vec_turboquant_p0.h:1378-1488](../op_kernel/vendored/arch32/fia_block_vec_turboquant_p0.h#L1378-L1488)):

```cpp
uint32_t si = siStart;
uint32_t runId = 0U;
while (si < siEnd) {
    // (A) blockTable GetValue + headBase
    // (B) 计算 n(run length,P15 已 O(1))
    // (C) 复用 ping-pong half(A1: runId>=2 才 Wait MTE3_V)
    // (D) MTE2 codes DMA → batchUb
    // (E) Wait(MTE2_V) codes
    // (F) MTE2 meta DMA → packedMetaUb
    // (G) Wait(MTE2_V) meta
    // (H) Cast meta → meta0Fp32/meta1Fp32
    // (I) VEC decode(BrDecodeKeyTile/ValueTile 循环)
    // (J) Wait(V_MTE3) → MTE3 写 WS
    si += n;
    ++runId;
}
```

**改后结构**(预取下一 run):

```cpp
uint32_t si = siStart;
uint32_t runId = 0U;
// 预取状态机:记录"下一 run 的 DMA 是否已发起"
bool nextDmaInFlight = false;
uint32_t nextSi = 0, nextN = 0, nextBufIdx = 0, nextMetaSlotOff = 0;
// 下一 run 的 codes/meta GM offset + blockTable 结果缓存
uint64_t nextCodeOff = 0, nextMeta0GmOff = 0, nextMeta1GmOff = 0;

while (si < siEnd) {
    uint32_t bufIdx = runId % 2U;
    uint32_t metaSlotOff = bufIdx * br_dequant::BR_META_SLOT_BYTES;

    // ── 阶段 1: 当前 run 的 codes/meta 落盘 ──
    // (首个 run 没有"上一轮预取",正常发 DMA;后续 run 复用上一轮已发的 DMA)
    if (!nextDmaInFlight) {
        // 首个 run:正常发起 codes/meta MTE2
        IssueCodesMetaDma(si, n, bufIdx, metaSlotOff, ...);
        // 内含 SetFlag(MTE2_V) eventIdVWaitMte2
    }
    // 后续 run:codes/meta DMA 在上一轮 VEC 期间已发,nextDmaInFlight=true
    // 此处只需 Wait(MTE2_V) 确认 DMA 完成
    WaitFlag<HardEvent::MTE2_V>(eventIdVWaitMte2);

    // ── 阶段 2: Cast meta + VEC decode(当前 run)──
    BrCastPackedMetaToFp32<Q_T>(packedMetaUb, n, meta0Fp32, meta1Fp32);
    for (j0 ...) BrDecodeKeyTile(...);  // 或 ValueTile

    // ── 阶段 3: 在 VEC decode 刚结束、MTE3 之前,预取下一 run ──
    // (关键:此处 VEC 已完成,MTE2 单元空闲,立刻发下一 run 的 DMA,
    //   让它在 MTE3(写)期间 + 下一个 run 的 VEC 前半段飞行)
    uint32_t nextSiCandidate = si + n;
    if (nextSiCandidate < siEnd) {
        uint32_t nextGlobalS2 = info.s2Idx * constInfo.s2BaseSize + nextSiCandidate;
        uint32_t nextBlockInBatch = nextGlobalS2 / bs;
        uint32_t nextPos0 = nextGlobalS2 % bs;
        uint32_t nextBtIdx = info.bIdx * constInfo.maxBlockNumPerBatch + nextBlockInBatch;
        // blockTable GetValue 是 scalar,需 S_MTE2 同步;提前做
        SetFlag<HardEvent::S_MTE2>(eventIdMte2WaitS);
        WaitFlag<HardEvent::S_MTE2>(eventIdMte2WaitS);
        int32_t nextPhysBlock = blockTableGm_.GetValue(nextBtIdx);
        uint64_t nextHeadBase = GetBrPackHeadBase(nextPhysBlock, info.n2Idx, isKey);

        uint32_t nextN = ComputeRunLength(nextSiCandidate, siEnd, bs, maxSub);
        uint32_t nextBufIdx = (runId + 1U) % 2U;
        uint32_t nextMetaSlotOff = nextBufIdx * br_dequant::BR_META_SLOT_BYTES;

        // 发起下一 run 的 codes + meta MTE2 DMA(不 Wait!)
        IssueCodesMetaDma(nextSiCandidate, nextN, nextBufIdx, nextMetaSlotOff,
                          nextHeadBase, /*setEvent=*/eventIdNextCodesMte2V);
        nextDmaInFlight = true;
    } else {
        nextDmaInFlight = false;
    }

    // ── 阶段 4: MTE3 写 WS(当前 run)──
    // A1 复用逻辑不变
    if (runId >= 2U) { WaitFlag<MTE3_V>(reuseEv); }
    SetFlag<V_MTE3>...; WaitFlag<V_MTE3>...;
    DataCopy(dstWsGm[...], outBatch, n * headDimAlign);
    SetFlag<MTE3_V>(curEv);

    si += n;
    ++runId;
}

// 收尾:若最后一个 run 之后还有 in-flight DMA(不会,因为预取只在 siEnd 前发),drain
```

**`IssueCodesMetaDma` 抽出的公共函数**(避免首 run 与预取重复代码):

```cpp
__aicore__ inline void IssueCodesMetaDma(
    GlobalTensor<uint8_t> srcGm, LocalTensor<uint8_t> batchUb,
    uint64_t headBase, uint32_t si, uint32_t n, uint32_t pos0,
    uint32_t codeRowBytes, uint32_t bs, bool isKey,
    LocalTensor<uint8_t> packedMetaUb, uint32_t metaSlotOff,
    uint64_t &outMeta0GmOff, uint64_t &outMeta1GmOff,
    event_t waitEvent /*MTE2_V*/)
{
    uint32_t codesBytes = n * codeRowBytes;
    uint64_t codeOff = isKey ? br_pack::BrKeyCodeOffset(headBase, pos0)
                             : br_pack::BrValCodeOffset(headBase, pos0);
    DataCopy(batchUb, srcGm[codeOff], codesBytes);
    outMeta0GmOff = isKey ? br_pack::BrKeyBaseOffset(headBase, bs, pos0)
                          : br_pack::BrValVminOffset(headBase, bs, pos0);
    outMeta1GmOff = isKey ? br_pack::BrKeyStepOffset(headBase, bs, pos0)
                          : br_pack::BrValVstepOffset(headBase, bs, pos0);
    br_dequant::BrCopyPackedMetaTile(srcGm,
        packedMetaUb[metaSlotOff], outMeta0GmOff, outMeta1GmOff, n);
    SetFlag<HardEvent::MTE2_V>(waitEvent);  // 发完即发完成事件,不 Wait
}
```

### 步骤 3:事件 ID 申请与流水隔离

在 `DequantKvImpl` 顶部新增事件:

```cpp
// P18: 下一 run codes/meta DMA 完成事件(MTE2 → V)
event_t eventIdNextCodesMte2V = static_cast<event_t>(
    GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
```

**Wait 位置调整**:
- 当前 run 的 `WaitFlag(MTE2_V, eventIdVWaitMte2)` → 改为等待"本 run 的 DMA 完成"
- 首个 run:DMA 同步发起 + Wait 用 `eventIdVWaitMte2`
- 后续 run:DMA 上一轮已发,Wait 用 `eventIdNextCodesMte2V`,**Wait 后立即 Cast + decode**

⚠️ **事件 ID 复用风险**:首个 run 用 `eventIdVWaitMte2`,后续 run 用 `eventIdNextCodesMte2V`。需保证一个 event 在 Set→Wait 配对期间不被重新 Set。建议:
- run r 的 DMA 用 `eventIdVWaitMte2`(r 偶)/ `eventIdNextCodesMte2V`(r 奇)轮转,或
- 统一用 2 个 event ping-pong:`eventIdCodesA` / `eventIdCodesB`,`runId % 2` 选

推荐后者(更清晰):

```cpp
event_t eventIdCodesWait[2] = {
    static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V)),
    static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V))
};
// run r 的 DMA 发起后 SetFlag(eventIdCodesWait[runId % 2])
// run r 进入 decode 前 WaitFlag(eventIdCodesWait[runId % 2])
```

### 步骤 4:blockTable GetValue 提前

当前每 run 顶部 `SetFlag/WaitFlag(S_MTE2)` 同步 scalar blockTable 读([fia_block_vec_turboquant_p0.h:1386-1389](../op_kernel/vendored/arch32/fia_block_vec_turboquant_p0.h#L1386-L1389))。这是 scalar→MTE2 同步,开销小但串行。

**优化**:把当前 run 的 blockTable 读**挪到上一 run 的 VEC decode 期间**(与下一 run 的 MTE2 预取合并)。即步骤 2 的"阶段 3"同时做:
- 下一 run 的 blockTable GetValue(scalar)
- 下一 run 的 headBase 计算
- 下一 run 的 codes/meta MTE2 DMA

这样当前 run 顶部不再有 S_MTE2 同步,直接 Wait codes DMA → Cast → decode。

> 若 blockTable 预取太复杂,可先跳过步骤 4,只做步骤 2 的 MTE2 预取(收益约 5-8%),验证后再加。

---

## 5. 数据流时序图

### 5.1 改造前(A1 + P9b)

```
时间 ──────────────────────────────────────────────────────────────────────▶

run0: ┌MTE2 c0┤├W┤├MTE2 m0┤├W┤├═══ VEC decode0 ═══┤├W┤├MTE3_0┤
                                              VEC满│       MTE2空
run1:                                              ┌MTE2 c1┤├W┤├MTE2 m1┤├W┤├═══ VEC decode1 ═══┤├W┤├MTE3_1┤
                                                                  ↑ run1 必须等 run0 的 MTE3_0 完(A1 复用 half)
```

MTE2 与 VEC **完全不重叠**;run 间只靠 A1 的 MTE3⇄MTE2 小窗口重叠。

### 5.2 改造后(P18: MTE2⇄VEC 跨 run 重叠)

```
时间 ──────────────────────────────────────────────────────────────────────▶

run0: ┌MTE2 c0 + m0┤├W┤├═══════════ VEC decode0 ═════════════┤├W┤├MTE3_0┤
                              │  VEC 满,MTE2 空闲窗口          │
                              ▼                                  │
run1(预取):              ┌MTE2 c1 + m1┤  ← 藏进 VEC decode0     │
                              │                                  │
run1:                    └→ Wait c1 → Cast m1 → ═══ VEC decode1 ═══┤├W┤├MTE3_1┤
                                                │  VEC 满         │
                                                ▼                 │
run2(预取):                                  ┌MTE2 c2 + m2┤      │
                                              └→ Wait → decode2 ...
```

**关键变化**:
1. **codes 和 meta 合并成一次 MTE2 burst 发起**(当前是 codes→Wait→meta→Wait 两次;P1 本就是 2 次 bulk DMA,但中间有 Wait)。预取时一次性发完,不 Wait。
2. **下一 run 的 MTE2 完全落在当前 run 的 VEC 窗口内**——这是本方案的核心收益点。
3. run 1 进入 decode 前,只需 `WaitFlag` 确认上一轮预取的 DMA 已完成,无需再发起。

### 5.3 重叠收益估算

设单 run 内:
- MTE2(codes+meta)耗时 = `T_dma`
- VEC decode 耗时 = `T_vec`
- MTE3 写耗时 = `T_mte3`(≈0.014 占比,可忽略)

**改造前** run 串行总时长 ≈ `T_dma + T_vec`(忽略 MTE3)
**改造后** run 重叠后有效时长 ≈ `max(T_dma, T_vec)` ≈ `T_vec`(因为 AIV-bound 时 `T_vec >> T_dma`)

**单 run 节省** ≈ `T_dma`,相对收益 ≈ `T_dma / (T_dma + T_vec)`

若 `T_vec` 占 80%、`T_dma` 占 15%(其余 5% scalar/同步):
- 收益 ≈ 15% / (15%+80%) × (重叠效率) ≈ **8-15%**(取决于 MTE2 能否完全藏进 VEC 窗口)

---

## 6. 验证计划

### 6.1 正确性验证(先于性能)

```bash
# 1. 重建算子
bash script/lcy/bit_residual_fia_paged_k8v4/rebuild_op.sh --soc=ascend910b

# 2. L2 单元测试(dequant 数值)
pytest -sv tests/ut/ops/test_bit_residual_fia_dequant.py

# 3. L5 E2E smoke(含 bf16 / FD / prefill 多路径)
python tests/e2e/singlecard/xrx_bit_residual_fia_paged_k8v4_smoke.py
```

**验收标准**:与 P9b 基线 `max_abs ≤ 2e-3`(DESIGN.md §7)。重点关注:
- bf16 路径(meta dtype 不能错位)
- FlashDecode 路径(split 边界)
- 短 run(`n < 8`)和长 run(`n ≈ 20+`)边界

### 6.2 性能验证

```bash
# L6 msprof,与 P9b 基线同口径(kv=2000,device=0)
bash tools/bit_residual_fia_paged_k8v4/prof_tnd_pa_bit_residual.sh \
    msprof --kv=2000 --device=0 --skip-build --warm-up=5 --launch-count=20

# 分析
python tools/bit_residual_fia_paged_k8v4/analyze_opprof.py
```

**验收指标**:

| 指标 | P9b 基线 | P18 目标 | 判定 |
|------|---------|----------|------|
| decode Task Duration | ~807µs | ≤740µs | -8% |
| prefill Task Duration | ~834µs | ≤770µs | -8% |
| `vec_ratio` | ~0.21 | 不显著上升 | VEC 仍是瓶颈,但被 MTE2 重叠填空 |
| `mte2_ratio` | — | 上升 | 预期:MTE2 占比上升(被藏进 VEC) |
| 数值 | PASS | PASS | 必须 |

### 6.3 回归边界用例

跨 run 重叠引入的新失败模式:

| 风险 | 用例 | 预期 |
|------|------|------|
| meta ping-pong slot 错位 | `n` 恰好让 run 数为奇 | 落到 slot1 正确 |
| 末 run 无预取 | `siEnd` 恰好整除 | `nextDmaInFlight=false` 正确 |
| 单 run 场景 | `s2Count` 很小,只 1 个 run | 退化为首 run 路径,不预取 |
| blockTable 预取越界 | `nextSiCandidate >= siEnd` | 跳过预取分支 |
| 事件 ID 超限 | `FetchEventID` 失败 | 退化方案见 §7 |

---

## 7. 退化与回退策略

### 7.1 分级退化

| 级别 | 触发条件 | 实现 | 保留收益 |
|------|----------|------|----------|
| L0(完整) | 事件 ID + UB 都够 | §4 全部步骤 | 8-15% |
| L1(仅 codes 预取) | meta ping-pong UB 不够 | 只预取 codes,meta 仍每 run DMA | 5-8% |
| L2(不预取,仅合并 MTE2) | 事件 ID 不够 | codes+meta 一次 burst 发起但同 run 内 Wait | 1-3% |
| L3(回退) | 数值 FAIL 或变慢 | git revert 到 P9b 基线 | 0 |

### 7.2 回退命令

```bash
# P18 全部改动集中在 3 文件,回退干净:
git checkout c40ff40a -- \
    csrc/bit_residual_fia_paged_k8v4/op_kernel/vendored/arch32/br_dequant_device.h \
    csrc/bit_residual_fia_paged_k8v4/op_kernel/vendored/arch32/fia_block_vec_turboquant_p0.h
# (host tiling 若改了 offset,一并 checkout)
```

---

## 8. 实施清单

- [ ] 步骤 1:meta 区 ping-pong(`br_dequant_device.h` + InitBuffers + DequantKvImpl 取 tensor)
- [ ] 步骤 1 验证:仅结构改动,数值仍 PASS(不引入预取)
- [ ] 步骤 2:抽 `IssueCodesMetaDma` 公共函数
- [ ] 步骤 2 验证:重构等价,数值 PASS
- [ ] 步骤 3:跨 run MTE2 预取 + 事件 ID ping-pong
- [ ] 步骤 3 验证:数值 PASS + L6 性能 A/B
- [ ] 步骤 4:blockTable GetValue 提前(可选,基于步骤 3 收益决定)
- [ ] 整体回归:bf16 / FD / prefill / 短 run 全用例
- [ ] 文档更新:本方案结果合入 [dequant_opt_p0_p12_report.md](dequant_opt_p0_p12_report.md) 作为 P18

---

## 9. 关键代码位置速查

| 改动点 | 文件 | 行号 |
|--------|------|------|
| meta UB 常量 | `br_dequant_device.h` | 43-57 |
| `dequantInt8Buf_` 分配 | `fia_block_vec_turboquant_p0.h` | 364-368 |
| `DequantKvImpl` 主循环 | `fia_block_vec_turboquant_p0.h` | 1294-1497 |
| blockTable GetValue | `fia_block_vec_turboquant_p0.h` | 1386-1389 |
| run-length 计算(P15) | `fia_block_vec_turboquant_p0.h` | 1391-1400 |
| codes/meta DMA | `fia_block_vec_turboquant_p0.h` | 1417-1449 |
| VEC decode 循环 | `fia_block_vec_turboquant_p0.h` | 1450-1477 |
| MTE3 写 WS | `fia_block_vec_turboquant_p0.h` | 1479-1484 |
| 事件 ID 申请 | `fia_block_vec_turboquant_p0.h` | 1353-1359 |
| `DequantK`/`DequantV` 入口 | `fia_block_vec_turboquant_p0.h` | 1241-1255 |
| pipeline 时序 | `fia_kernel_turboquant_p0.h` | 1044-1081 |

---

## 10. 经验教训(来自 P0-P17)

实施时务必遵循(均来自 [dequant_opt_p0_p12_report.md](dequant_opt_p0_p12_report.md) 实测):

1. **先消 DMA/同步,再抠 VEC 指令** — 但 P13 后 VEC 指令已压到 RAW 必要项,不可再砍(P7a 回退)
2. **对齐约束优先** — meta fp32 区必须 32B 对齐供 Brcb;ping-pong slot 基址必须 32B 对齐(`BR_DEQUANT_UB_BYTES=768` 已 32B 对齐 ✅)
3. **msprof 占比 ≠ 墙钟** — P10 反例;本方案以 L6 Task Duration 为准
4. **勿 int16 算术造符号** — 507015;本方案不碰符号提取,只动 DMA 时序
5. **A1 仅 -0.4% 的教训** — 重叠小窗口收益低;本方案重叠 VEC 长窗口,但需实测确认 MTE2 能否完全藏入

---

*文档生成: 2026-07-23*  
*基线 commit: `d50a1b22`(P17b-B)*  
*预期合入编号: P18*
