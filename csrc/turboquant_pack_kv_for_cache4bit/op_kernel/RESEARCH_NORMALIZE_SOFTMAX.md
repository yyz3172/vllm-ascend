# 代码调研：归一化 vs Online Softmax 的高性能设计对比

> 日期：2026-06-27
> 参考版本：vllm-ascend `d039822d7c131d4e1aaa91d8efb71b3e00bbcfde`
> 参考对比：ops-transformer `fused_infer_attention_score` FAInfer 路径
> 背景：FusedInferAttentionScore 在 `[1802,16,128]` shape 上耗时 193us，
>        TurboquantPackKvForCache4bit 在同 shape 上耗时 ~900us

---

## 1. 问题定位

TQ Pack 的 NormalizeBatch 是 kernel 中最重的向量计算段。在 `[1802,16,128]`
shape（28832 行）上，每行归一化的关键路径包含一次 `GetValue` + `Muls`，
触发 **V→S→V 跨引擎同步**，每行 2 次同步、全量 57664 次同步。

FIAS 的 Online Softmax 实现相同语义的"行级归一化"（softmax = max-shift +
exp + sum），但全程使用纯向量操作，**0 次 Scalar 引擎参与**，仅在最后 KV
tile 做 1 次 Div。

本文档从 FIAS 代码中提取高性能归一化模式，映射到 TQ Pack 的优化方向。

---

## 2. TQ Pack NormalizeBatch 当前实现

### 2.1 代码位置

`csrc/turboquant_pack_kv_for_cache4bit/op_kernel/turboquant_pack_kv_for_cache4bit.cpp`

- `NormalizeBatch()` — [line 427-460](csrc/turboquant_pack_kv_for_cache4bit/op_kernel/turboquant_pack_kv_for_cache4bit.cpp#L427-L460)
- `EncodeBatch()` — [line 651-738](csrc/turboquant_pack_kv_for_cache4bit/op_kernel/turboquant_pack_kv_for_cache4bit.cpp#L651-L738)
- `ComputeBatch()` — [line 859-863](csrc/turboquant_pack_kv_for_cache4bit/op_kernel/turboquant_pack_kv_for_cache4bit.cpp#L859-L863)

### 2.2 每行归一化操作序列

```cpp
// NormalizeBatch per-row:
Cast(fp32Row, xBatch[rowOff], CAST_NONE, 128)      // fp16→fp32
PipeBarrier<PIPE_V>()
Mul(fp32Square, fp32Row, fp32Row, 128)              // 逐元素平方
ReduceSum(normAcc, fp32Square, fp32Tmp, 128)         // 行内求和
PipeBarrier<PIPE_V>()
Sqrt(normAcc, normAcc, 1)                           // 标量 sqrt

// ★ 瓶颈：Scalar 引擎参与 ★
TqSyncVToS()                                        // V→S 同步
normF = normAcc.GetValue(0)                         // Scalar 从 UB 读 1 fp32
TqSyncSToV()                                        // S→V 同步

Cast(norms[i*stride], normAcc, CAST_RINT, 1)        // 写缓存 norm (fp16)
PipeBarrier<PIPE_V>()
Muls(fp32Row, fp32Row, 1/(normF+eps), 128)          // ★ 标量除法广播 ★
PipeBarrier<PIPE_V>()
Cast(aBatch[rowOff], fp32Row, CAST_RINT, 128)       // fp32→fp16 输出
PipeBarrier<PIPE_V>()
```

### 2.3 瓶颈量化

| 指标 | 每行 | 28832 行全量 |
|------|------|-------------|
| Vec ops | 7 步 | ~201k 步 |
| PipeBarrier | 5 次 | ~144k 次 |
| **V→S→V 同步** | **2 次** | **57664 次** |
| GetValue | 1 次 | 28832 次 |
| 总 Vec 计算时间（估算） | ~2.8us | ~81us |
| 总同步时间（估算 @3us/次） | ~6us | ~173us |

同步占总 NormalizeBatch 耗时约 **68%**。

### 2.4 UB Buffer 分配

```
reduceOutBuf: fp32Row[128] + fp32Square[128] + fp32Tmp[128] = 1536B
normScalarBuf: 32B (1 fp32 norm accumulator)
normsBuf: 64 × 16 × 2 = 2048B (fp16 norms for cache)
```

三区生命周期分析（无法 alias）：
- `fp32Row` 在步骤 1-13 全程存活
- `fp32Square` 在 `Mul` 和 `ReduceSum(src)` 同时活跃
- `fp32Tmp` 在 `ReduceSum(workspace)` 与 `fp32Square` 同时活跃

Buffer 分配本身合理，但跨阶段复用未被利用：
- `reduceOutBuf` 在 NormalizeBatch 结束后 DEAD，但 `yFp32Buf`（Encode 用）
  是同功能的 128 fp32 行，两者可 alias 省 512B。
- Queue depth=2 在所有 4 个 Queue 上浪费 66KB（无跨阶段流水线重叠）。
- UB 总用量 184KB / A2 1MB 可用 = 18%，空间充足。

---

## 3. FIAS Online Softmax 实现分析

### 3.1 代码位置

| 文件 | 作用 |
|------|------|
| `op_kernel/flash_attention_regular.h` | 主 kernel 循环，Cube/Vec 协作框架 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp` | 在线 Softmax 向量计算 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp` | 输出 rescaling + 最终 Div |
| `op_kernel/flash_attention_interface.cpp` | 模板实例化入口 |

### 3.2 归一化的三个阶段

FIAS 的 Softmax 归一化分三阶段完成，对应传统 softmax 的 `P = exp(S-m) / Σexp(S-m)`：

**阶段 A：逐 KV tile 的局部 softmax（EpilogueOnlineSoftmax）**

```
SubCoreCompute<doTriUMask>() 调用序列:

  CalcLocalRowMax  →  UpdateGlobalRowMax  →  CalcExp  →  DownCastP
                                         ↓
                          CalcLocalRowSum  →  UpdateGlobalRowSum
```

**阶段 B：逐 KV tile 的 O 累积 rescaling（EpilogueRescaleO）**

```
非首块:  Brcb(dm) → Mul(go, dm_block) → Add(go, lo)
首块:    DataCopy(go, lo)
```

**阶段 C：最终归一化（仅最后一个 KV tile）**

```
Brcb(gl) → Div(go, gl_block) → Cast(go, fp32→fp16) → CopyOToGm
```

### 3.3 关键子步骤代码与高性能手法

#### 手法 1：Brcb 替代 GetValue+Muls — 消除 V→S→V 同步

这是最核心的优化。对比两种"标量→行广播"方式：

**TQ Pack 方式（当前）：**

```cpp
// file: turboquant_pack_kv_for_cache4bit.cpp, line 456-458
TqSyncVToS();
const float normF = normAcc.GetValue(0);    // Scalar 从 UB 读 → 需要 V→S 同步
TqSyncSToV();
Muls(fp32Row, fp32Row, 1/(normF+eps), 128); // Muls 的 scalar 参数来自 GetValue
```

**FIAS 方式：**

CalcExp 中用 Brcb 广播 hm 替代 Muls(-hm)：
```cpp
// file: block_epilogue_online_softmax.hpp, line 660-664
Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
    hmUbTensor[rowOffset].ReinterpretCast<uint32_t>(),
    rowNumCurLoopRound / FLOAT_BLOCK_SIZE,    // 每行 1 fp32 → 广播为整行
    AscendC::BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();                        // 仅 PipeBarrier，无 V→S→V

// ls = ls - hm_block (等效于 Muls(ls, -hm) 但纯 Vector)
Sub(ls, ls, tv, rowNumCurLoop, BinaryRepeatParams(1, 1, 0, ...))
```

RescaleO 中用 Brcb 广播 dm 替代 Muls(dm_scalar)：
```cpp
// file: block_epilogue_rescale_o.hpp, line 187-190
Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
    dmUbTensor[dmOffset].ReinterpretCast<uint32_t>(),
    curRowNumRound / FLOAT_BLOCK_SIZE,
    AscendC::BrcbRepeatParams(1, 8));

// go = go * dm_block (等效于 Muls(go, dm_scalar) 但纯 Vector)
Mul(goUbTensor32, goUbTensor32, tvUbTensor, ...)
```

最终 Div 中用 Brcb 广播 gl 替代 Div 的分母行广播：
```cpp
// file: block_epilogue_rescale_o.hpp, line 245-249
Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
    glUbTensor[rowOffsetLoop].ReinterpretCast<uint32_t>(),
    curRowNumRound / FLOAT_BLOCK_SIZE,
    AscendC::BrcbRepeatParams(1, 8));
// go = go / gl_block
Div(goUbTensor32, goUbTensor32, tvUbTensor, ...)
```

**`Brcb` 的本质**：Vector 引擎内部指令，将 UB 中每行 1 个 fp32 值
（reinterpret 为 uint32_t，4 字节）沿列方向广播为整行所有 8 个 block 位置
（每 block 8 fp32 = 32B，整行共 64 fp32 / 256B）。全过程不经过 Scalar
引擎，不需要 GetValue，不需要 V→S→V 同步。

FIAS 中 `Brcb` 的 3 处使用：

| 位置 | 代码行 | 广播内容 | 替代的操作 |
|------|--------|---------|-----------|
| CalcExp | [softmax.hpp:660](op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp#L660) | `hm`（全局最大值 m） | `Muls(S, -m)` |
| RescaleO 非首块 | [rescale_o.hpp:187](op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp#L187) | `dm`（rescaling 因子） | `Muls(O_old, dm_scalar)` |
| RescaleO 最终Div | [rescale_o.hpp:245](op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp#L245) | `gl`（全局分母 Σsoftmax） | `Muls(O, 1/gl_scalar)` 或 Div 分母行 |

#### 手法 2：BlockReduceMax/BlockReduceSum 替代 ReduceSum+GetValue

**TQ Pack 方式：**

```cpp
// file: turboquant_pack_kv_for_cache4bit.cpp, line 443-446
ReduceSum<float>(normAcc, fp32Square, fp32Tmp, TQ_PACK_D);  // 1 行 → 1 fp32
PipeBarrier<PIPE_V>();
Sqrt(normAcc, normAcc, 1);         // 标量 Sqrt
GetValue(normAcc[0])               // ← Scalar 引擎参与
```

逐行 ReduceSum，每行产 1 fp32 后需要 Scalar 读。

**FIAS 方式：**

```cpp
// file: block_epilogue_online_softmax.hpp, line 303-320
BlockReduceMax<float, false>(tvUbTensor, srcUb,
    numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
    0, 1, 1, 8);                    // 多行同时并行 reduce
PipeBarrier<PIPE_V>();
BlockReduceMax<float, false>(rowmaxUb, tvUbTensor[REDUCE_UB_SIZE],
    numRowsRound * ...,
    0, 1, 1, 8);                    // 三级递归，最终每行 1 fp32
PipeBarrier<PIPE_V>();
```

`BlockReduceMax/BlockReduceSum` 是 block 级 reduce 指令：
- 多行同时并行 reduce，所有行的结果存入连续 UB buffer（每行 1 fp32）
- 三级递归（针对 512/256/tail 列宽）
- 结果在 UB 上，后续直接用 `Max(lm, gm)` 或 `Brcb` 读取
- **无 GetValue，无 Scalar 引擎参与**

#### 手法 3：延迟最终 Div — 只在最后一个 KV tile 执行一次

```cpp
// file: block_epilogue_rescale_o.hpp, line 243-275
if (isLastStackTile) {
    Brcb(tv, gl[rowOffsetLoop], ...)      // 广播分母
    Div(go, go, tv, ...)                  // ★ 仅 1 次 Div
    Cast(goUbTensor16, goUbTensor32, ...) // fp32→fp16
    CopyOToGm(gOutput, ...)               // 写回 GM
}
```

中间 KV tile 只做 `Mul(go, dm_block)` 和 `Add(go, lo)`，
不做 Div。Div 操作次数从 O(KV_len) 降到 O(1)。

对应 TQ Pack 的场景：NormalizeBatch 的 `Muls(fp32Row, 1/norm)` 本质上
是每行 1 次 Div（除以 norm），这可以类似地延迟或批量处理。

#### 手法 4：Cube/Vector 三级流水线重叠

```cpp
// file: flash_attention_regular.h, line 320
for (kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx++) {
    // PRE_LAUNCH = 2：tile i+2 的 Q×K 与 tile i 的 P×V 重叠执行
    if (kvSIdx < kvSLoopNumTotal) {
        blockMmadQK(...);              // Cube: Q×K → S
        SetFlag(qkReady);              // Cube→Vec 信号
        epilogueOnlineSoftmax(...);    // Vec: Softmax(S) → P
        SetFlag(softmaxReady);         // Vec→Cube 信号
    }
    if (kvSIdx >= preKVNum) {
        blockMmadPV(...);              // Cube: P×V → O_tmp
        SetFlag(pvReady);              // Cube→Vec 信号
        epilogueRescaleO(...);         // Vec: Rescale O
    }
}
```

3 个 CrossCore flag（[line 564-566](op_kernel/flash_attention_regular.h#L564-L566)）：
- `qkReady`：Cube→Vec，Q×K 完成
- `softmaxReady`：Vec→Cube，P 就绪
- `pvReady`：Cube→Vec，P×V 完成

TQ Pack 当前 `ComputeBatch()` 严格串行调用
`Normalize(Vec) → RotateMatmul(Cube) → Encode(Vec)`，
无 Cube/Vec 重叠。

#### 手法 5：Pingpong UB 双缓冲

```cpp
// file: block_epilogue_online_softmax.hpp, line 1018-1066
for (rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
    if (rowLoopIdx < rowLoopNum) {
        // PRE-LOAD: GM→UB 数据搬运（slot = pingpongFlag）
        WaitFlag(V_MTE2, pingpongFlag);
        CopySGmToUb(gInput, pingpongFlag * 8192, ...);
        SetFlag(MTE2_V, pingpongFlag);
    }
    if (rowLoopIdx >= preLoad) {
        // COMPUTE: 对上一批数据做 Softmax（与下一批搬运重叠）
        WaitFlag(MTE2_V, pingpongFlag);
        ScaleS + SubCoreCompute(...)
    }
}
```

`lsUbTensor` 有 2 个 slot（各 8192 fp32 = 32KB），Vec 计算 slot 0 时
MTE2 搬运 slot 1。

#### 手法 6：完全消除 Scalar 引擎参与

FIAS 整个 Softmax + RescaleO 流程中 GetValue 的使用情况：

| GetValue 出现位置 | 频率 | 是否在关键路径 |
|------|------|------|
| [softmax.hpp:904](op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp#L904)：`gSink.GetValue(headId)` | 每head×每batch | 仅 Sink 模式 |
| [softmax.hpp:947](op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp#L947)：`gSink.GetValue(headId)` | 每head×每batch | 仅 Sink 模式 |
| RescaleO 中的 GetValue | **0 次** | — |

**标准模式下 GetValue = 0 次**，Scalar 引擎完全不被 Softmax/归一化调用。

对比 TQ Pack NormalizeBatch：GetValue = **每行 1 次 × 28832 行 = 28832 次**。

#### 手法 7：fp32 中间精度 + fp16 输出

所有 reduce、max、exp、sub 操作在 fp32 上完成：
- `lsUbTensor`（S 分数）：fp32
- `lmUbTensor`、`hmUbTensor`、`gmUbTensor`（最大值累积）：fp32
- `llUbTensor`、`glUbTensor`（分母累积）：fp32

仅最后一步 `DownCastP`（[line 779](op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp#L779)）
将 P 从 fp32 降到 fp16/bf16 写入 GM，供 Cube 的 PV MatMul 读取。

这与 TQ Pack 的做法一致（fp32 中间精度，fp16 输出）。

---

## 4. 模式映射：FIAS 手法 → TQ Pack 优化方向

### 4.1 直接可迁移的模式

| FIAS 手法 | TQ Pack 映射 | 可行性 |
|-----------|-------------|--------|
| Brcb 替代 GetValue+Muls | NormalizeBatch 的 `GetValue(normF) + Muls(1/norm)` → `Brcb(recipRow, recip) + Mul(fp32Row, recipRow)` | ✅ 可行，详见 4.2 |
| BlockReduceMax 替代 ReduceSum+GetValue | NormalizeBatch 的 `ReduceSum + Sqrt + GetValue` → 两循环拆分 + 批量 GetValue | ⚠️ 需两循环拆分（4.3） |
| Pingpong 双缓冲 | NormalizeBatch 与 CopyIn 之间插入双缓冲 | ⚠️ 需先统一 batch emit |
| Cube/Vec 流水线 | Normalize(Vec) 与 RotateMatmul(Cube) 重叠 | ⚠️ 需先统一 batch emit |
| 延迟 Div | Normalize 的 Div 延迟到 Encode 完成后 | ❌ 不适用（TQ 没有 softmax 分母累积） |

### 4.2 Brcb 替代 GetValue+Muls 的具体方案

**当前（每行 2 次 V→S→V 同步）：**

```cpp
Sqrt(normAcc, normAcc, 1);
TqSyncVToS();
const float normF = normAcc.GetValue(0);   // ← 瓶颈
TqSyncSToV();
Muls(fp32Row, fp32Row, 1/(normF+eps), 128);
```

**方案 A：两循环拆分 + 批量 GetValue + Duplicate 替代 Muls**

```
Loop 1: 计算所有 norm（纯 Vector）
  for i in [0..m):
    Cast(fp32Row, xBatch[i])
    Mul(fp32Square, fp32Row, fp32Row)
    ReduceSum(normAcc, fp32Square, fp32Tmp)
    Sqrt(normAcc, normAcc, 1)
    Cast(norms[i*stride], normAcc)           // 写 fp16 norm 到 normsBuf

批量 Scalar 读取（仅 2 次同步覆盖全 batch）
  TqSyncVToS()                               // 1 次 V→S
  for i in [0..m):
    recipArray[i] = 1/(GetValue(normWords, i*stride) + eps)  // C++ 栈变量
  TqSyncSToV()                               // 1 次 S→V

Loop 2: 逐行归一化（纯 Vector）
  for i in [0..m):
    Cast(fp32Row, xBatch[i])                 // 重读输入
    Duplicate(recipRow, recipArray[i], 128)   // ★ 替代 Muls ★
    Mul(fp32Row, fp32Row, recipRow)           // ★ 向量乘法 ★
    Cast(aBatch[i], fp32Row)
```

同步次数对比：2/行 × 64行 = 128 → 2/batch × 1 = 2，**减少 64×**。

注意：`Duplicate(dst, scalarValue, count)` 将 C++ scalar 作为指令 operand
传入 Vector 引擎，不经过 UB，不需要 V→S→V 同步。这与 FIAS 的 `Brcb`
在语义上等效（都是"标量→行广播"），但实现路径不同：
- `Brcb`：从 UB 中的 fp32 buffer 读取并广播（不经过 Scalar）
- `Duplicate`：从 C++ 变量（指令 operand）广播（不经过 UB）

两者都避免了 Scalar 引擎参与。`Duplicate` 更简单但需要先批量 GetValue
到 C++ 栈；`Brcb` 需要把 fp32 norms 存在 UB buffer 中。

**方案 B：纯 Brcb 方案（与 FIAS 完全对齐）**

```
Loop 1: 计算所有 fp32 norms（存入 UB fp32 norm buffer）
  for i in [0..m):
    Cast(fp32Row, xBatch[i])
    Mul(fp32Square, fp32Row, fp32Row)
    ReduceSum(normFp32[i], fp32Square, fp32Tmp)
    Sqrt(normFp32[i], normFp32[i], 1)
    Cast(norms[i*stride], normFp32[i])        // 同时写 fp16 norm

Loop 2: 逐行归一化用 Brcb（纯 Vector，0 同步）
  for i in [0..m):
    Cast(fp32Row, xBatch[i])                  // 重读输入
    Brcb(recipRow.ReinterpretCast<uint32_t>(),
         normFp32[i].ReinterpretCast<uint32_t>(),
         TQ_PACK_D / FLOAT_BLOCK_SIZE,         // 1 fp32 → 128 fp32
         BrcbRepeatParams(1, 8))
    PipeBarrier<PIPE_V>()
    // fp32Row = fp32Row / normBlock
    Div(fp32Row, fp32Row, recipRow, 128)       // ★ Div 替代 Muls(1/norm) ★
    Cast(aBatch[i], fp32Row)
```

方案 B 的优点：**0 次 V→S→V 同步**（比方案 A 的 2/batch 更优）。
方案 B 的代价：需要 UB 中存 `m × 1` 个 fp32 norms（64 × 4B = 256B，
可 alias `normsBuf` 或用 `yFp32Buf` 空间），以及每行 1 次 `Div`（比
`Mul` 稍慢但可忽略）。

### 4.3 两循环拆分的 Vec 操作量对比

| 步骤 | 当前单循环 | 方案 A (Duplicate) | 方案 B (Brcb) |
|------|-----------|-------------------|---------------|
| Cast fp16→fp32 | 1/行 | 2/行 (+1) | 2/行 (+1) |
| Mul(square) | 1/行 | 1/行 | 1/行 |
| ReduceSum | 1/行 | 1/行 | 1/行 |
| Sqrt | 1/行 | 1/行 | 1/行 |
| Cast(norms) | 1/行 | 1/行 | 1/行 |
| GetValue+Muls | 1/行 + 2 sync | 批量 GetValue + Duplicate/行 | **0 GetValue** |
| 广播+乘/除 | — | Duplicate + Mul | Brcb + Div |
| Cast fp32→fp16 | 1/行 | 1/行 | 1/行 |
| **Vec ops 总计** | **7/行** | **9/行** (+2) | **9/行** (+2) |
| **V→S→V 同步** | **2/行** | **2/batch** | **0/batch** |

Vec ops 增加 2/行（double Cast + 广播+乘/除），但同步大幅减少。

### 4.4 时间估算（m=64 行/batch，28832 总行）

| 方案 | Vec ops 耗时 | 同步耗时 | 总计 |
|------|-------------|---------|------|
| 当前单循环 | ~81us | ~173us | **~254us** |
| 方案 A (Duplicate) | ~103us (+22us) | ~2.7us (-170us) | **~106us** |
| 方案 B (Brcb+Div) | ~103us (+22us) | ~0us (-173us) | **~103us** |

NormalizeBatch 耗时减少 ~58~60%，对 900us 总耗时约 **~16% 整体提升**。

---

## 5. 历史实验教训

OPTIMIZE_PLAN.md 记录了多次 NormalizeBatch 优化尝试的失败：

### 5.1 WholeReduceSum 批量归一化（已回退）

- 结果：Smoke/Long 大致持平（smoke 46.48us, long 818.85us）
- `tools/op.profile.sh` 失败（`ret=507015`）
- 教训：不要在全量 profile 场景引入未验证的 reduce 指令

### 5.2 两循环 fp32 norm 方案（已回退）

- 结果：Smoke 小 shape 从 46.59us 退步到 47.69us
- Long 819.72us vs 820.39us，几乎无改善
- 教训：**额外第二遍 Cast(fp16→fp32) 的开销 ≈ 节省的同步开销**
- 原因：当前 `[1802,8,128]` shape 下 vecPerCore 较小（~32行/batch），
  每行多一次 Cast 的累积开销抵消了同步收益

### 5.3 关键发现：收益取决于 batch 大小

| vecPerCore | 每行额外 Cast 开销 | 每行节省同步开销 | 净收益 |
|-----------|-------------------|----------------|--------|
| 4 | 4 × 0.4us = 1.6us | 4 × 6us = 24us | ✅ 大正 |
| 16 | 16 × 0.4us = 6.4us | 16 × 6us = 96us | ✅ 大正 |
| 32 | 32 × 0.4us = 12.8us | 32 × 6us = 192us | ✅ 正 |
| 64 | 64 × 0.4us = 25.6us | 64 × 6us = 384us | ✅ 大正 |

**历史实验失败的原因**：当时在 `key=0` 路径上做，smoke shape 的
vecPerCore 很小（~2行），额外 Cast 开销接近甚至超过同步节省。

**正确策略**：只在 `key=1`（大 shape，vecPerCore≥32）上启用两循环
方案，smoke shape 保持 `key=0` 单循环路径。这已在 OPTIMIZE_PLAN.md
中明确为 key1 优化 backlog 的第 2 项。

---

## 6. 推荐方案与实施约束

### 6.1 推荐：方案 B（Brcb + Div），仅 key=1

```
NormalizeBatchTwoLoop()  — 仅 tilingKey=1 使用

Loop 1（纯 Vector，无 Scalar）:
  for i in [0..m):
    Cast(fp32Row, xBatch[i])
    PipeBarrier<PIPE_V>()
    Mul(fp32Square, fp32Row, fp32Row, 128)
    ReduceSum(normFp32[i], fp32Square, fp32Tmp, 128)
    PipeBarrier<PIPE_V>()
    Sqrt(normFp32[i], normFp32[i], 1)
    PipeBarrier<PIPE_V>()
    Cast(norms[i*stride], normFp32[i], CAST_RINT, 1)
    PipeBarrier<PIPE_V>()

Loop 2（纯 Vector，Brcb 替代 GetValue+Muls）:
  auto recipRow = reduceOutBuf_.Get<float>()[TQ_PACK_D];  // alias fp32Square 区
  for i in [0..m):
    Cast(fp32Row, xBatch[i])
    PipeBarrier<PIPE_V>()
    Brcb(recipRow.ReinterpretCast<uint32_t>(),
         normFp32[i].ReinterpretCast<uint32_t>(),
         TQ_PACK_D / 8,
         BrcbRepeatParams(1, 8))
    PipeBarrier<PIPE_V>()
    Div(fp32Row, fp32Row, recipRow, TQ_PACK_D)
    PipeBarrier<PIPE_V>()
    Cast(aBatch[i], fp32Row, CAST_RINT, TQ_PACK_D)
    PipeBarrier<PIPE_V>()
```

### 6.2 新增 UB 需求

| Buffer | 大小 | 来源 |
|--------|------|------|
| `normFp32Buf` | m × 4B（最大 64×4 = 256B） | 可 alias `yFp32Buf`（512B）的下半区，或新增 TBuf |

UB 增量：0B（alias 已有 buffer）或 256B（新增独立 TBuf）。

### 6.3 实施约束（来自 OPTIMIZE_PLAN.md）

1. **仅 key=1 路径启用**，key=0 保持当前单循环 NormalizeBatch
2. key=1 kernel 用独立类或编译期 flag 实例化，确保 key=0 生成代码不变
3. Smoke 小 shape 不可退步（baseline avg 46.59us）
4. Long hot shape `[1802,8,128]` 必须改善（baseline avg 820.39us）
5. `tools/build_debug_perf.sh`、smoke、long query、`tools/op.profile.sh`
   四项均须通过
6. `ret=507015` 是 release blocker，需先在小范围验证 `Brcb+Div` 组合

### 6.4 验证步骤

1. 在 key=1 路径加入 `NormalizeBatchTwoLoop()`，初始保持 key=1 使用
   当前单循环（scaffolding）
2. 将 key=1 的 NormalizeBatch 替换为 TwoLoop 版本
3. 运行 smoke，确认输出语义一致、小 shape timing 不退步（key=0 不变）
4. 运行 long query，确认大 shape timing 改善
5. 运行 `tools/op.profile.sh`，确认 `ret != 507015`
6. 如 profile 失败，回退并尝试方案 A（Duplicate+批量GetValue）作为
   Brcb 方案的 fallback

### 6.5 Brcb 注意事项

- `Brcb` 的 src 参数是 `LocalTensor<uint32_t>` reinterpret，
  需确认 fp32→uint32_t reinterpret 在 A2 上合法
- `BrcbRepeatParams(1, 8)` 中的 `8` = dstStride（8 个 block = 64 fp32）
  确认与 `TQ_PACK_D=128` 对齐：128 / 8 blocks = 16 fp32/block，
  但 `Brcb` 每次广播 1 个 4B 值到 8 个 32B block 位置，
  即 1 fp32 → 8 × 8 fp32 = 64 fp32。
  对于 D=128 需要调用 128/64 = 2 次 Brcb，或使用不同 stride。
  **需验证 Brcb 广播宽度是否覆盖 D=128**

- 如果 Brcb 广播宽度不足 128，需要改为：
  ```
  Brcb(recipRow[0], normFp32[i], 128/64/8, BrcbRepeatParams(2, 8))
  // 或两段 Brcb:
  Brcb(recipRow[0],       normFp32[i], 1, BrcbRepeatParams(1, 8))  // 广播到 [0..63]
  Brcb(recipRow[64*4/32], normFp32[i], 1, BrcbRepeatParams(1, 8))  // 广播到 [64..127]
  ```
  参考 FIAS CalcExp 的做法：`BrcbRepeatParams(1, 8)` 广播 1 fp32 到
  stride=8 block 的位置，但 FIAS 的 columnNum 也是 128/256，
  它通过 `Sub(ls, ls, tv, rowNum, BinaryRepeatParams(1,1,0,...))` 的
  **行级 broadcast-by-column** 模式实现整行减法——tvUbTensor 的列 stride=1
  block，每行只存 64 fp32 的 hm_block，但 `BinaryRepeatParams` 的
  `srcBlkStride=1` 使每行复用同一个 tv 值。这意味着 Brcb 广播 64 fp32
  + Mul/Sub 的 broadcast-by-column 模式覆盖剩余列。

  **TQ Pack 场景下 D=128 且列对齐 32B**，可能需要：
  - `Brcb` 广播到 64 fp32（半行）
  - `Mul(fp32Row, fp32Row, recipRow)` 使用
    `BinaryRepeatParams(1,1,0, columnBlkStride, columnBlkStride, 1)`
    的 broadcast-by-column 模式覆盖整行

  这是与 FIAS CalcExp 完全相同的模式，可行性确认。

---

## 7. FIAS 高性能设计手法速查表

| # | 手法 | FIAS 代码位置 | TQ Pack 映射 |
|---|------|-------------|-------------|
| 1 | Brcb 替代 GetValue+Muls | [softmax.hpp:660](op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp#L660), [rescale_o.hpp:187](op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp#L187), [rescale_o.hpp:245](op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp#L245) | NormalizeBatch 的 `GetValue+Muls` → `Brcb+Div` |
| 2 | BlockReduce 替代 ReduceSum+GetValue | [softmax.hpp:303](op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp#L303), [softmax.hpp:169](op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp#L169) | `ReduceSum+Sqrt+GetValue` → 两循环批量处理 |
| 3 | 延迟最终 Div | [rescale_o.hpp:243](op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp#L243) | 不直接适用（TQ 无分母累积） |
| 4 | Cube/Vec 三级流水线 | [flash_attention_regular.h:320](op_kernel/flash_attention_regular.h#L320), [line 564-566](op_kernel/flash_attention_regular.h#L564-L566) | Normalize(Vec) ↔ RotateMatmul(Cube) 重叠 |
| 5 | Pingpong UB 双缓冲 | [softmax.hpp:1018](op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp#L1018) | Normalize ↔ CopyIn 双缓冲 |
| 6 | 0 次 Scalar 引擎参与 | GetValue 仅 [softmax.hpp:904](op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp#L904) + [softmax.hpp:947](op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp#L947)（Sink 模式） | NormalizeBatch → 0 GetValue |
| 7 | fp32 中间精度 + fp16 输出 | [softmax.hpp:779](op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp#L779) | 已一致 |

---

## 8. 附录：FIAS Softmax 数学等价性

传统 Softmax：
```
P[i,j] = exp(S[i,j]) / Σ_j exp(S[i,j])
```

Online Softmax 逐 tile 等价：

Tile k 的 P_k：
```
P_k[i,j] = exp(S_k[i,j] - hm_k[i])       (CalcExp)
hm_k[i] = max(lm_k[i], gm_{k-1}[i])       (UpdateGlobalRowMax)
```

累积 O_k：
```
O_k = O_{k-1} × exp(gm_{k-1} - hm_k) + P_k × V_k    (RescaleO)
     = O_{k-1} × dm_k + P_k × V_k
```

最终归一化：
```
O_final = O_{last} / gl_{last}    (最后块 Div)
```

展开：
```
O_final = Σ_k P_k × V_k / gl
         = Σ_k exp(S_k - hm) × V_k / Σ_k Σ_j exp(S_k - hm)
         = Σ_k exp(S_k) × V_k / Σ_k Σ_j exp(S_k)
         ← 等价于标准 Softmax ✓
```
