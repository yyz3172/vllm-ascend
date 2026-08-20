# FIAS Call Trace Review and TQ4bit Pack Implications

日期：2026-06-28

本文审阅 `RESEARCH_FIAS_CALL_TRACE.md` 对
`mytmp/fused_infer_attention_score` 中 FIAS SplitFuse 路径的分析，按
shape `[1802, 8, 128]` 重新对照代码校正调用次数和分支次数，并结合
`turboquant_pack_kv_for_cache4bit.cpp` 当前执行流程整理可借鉴点。

Scope 限制：

- FIAS 只展开 `mytmp/fused_infer_attention_score` 目录下函数。
- TQ4bit 只对照当前
  `csrc/turboquant_pack_kv_for_cache4bit/op_kernel/turboquant_pack_kv_for_cache4bit.cpp`
  代码路径。
- FIAS 当前没有可用 OPP source-line profile，因此 FIAS 部分只使用源码静态调用次数。
- TQ4bit 使用当前可用 OPP source-line `Instructions Executed` 作为热点佐证。
  OPP 总耗时仍只作辅助，不作为单点性能结论。

## 结论摘要

原报告的顶层 tiling 和 KV stack 数基本正确：

- `qSBlockTile = 128`。
- `qNBlockTile = 1`。
- `curQNBlockNum = 8`。
- `curQSBlockNum = 15`。
- task 总数 `120`。
- 每个 qSBlock 的 KV stack 数为
  `[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4]`，每 head 合计 `36`，
  全 head 合计 `288`。
- KV loop 带 `PRE_LAUNCH=2`，每 head loop 合计 `66`，
  全 head 合计 `528`。

需要纠正的核心问题在 lower-level 展开：

- `doTriUMask=true` 不是几乎全部，而是 `120 / 288` 次；
  no-mask overload 是 `168 / 288` 次。
- `EpilogueOnlineSoftmax` 不能全部按 causal overload、512 列、每 subblock 4 个
  row-loop 来统计。
- `SubCoreCompute<true>` 不是 `2304` 次，而是 `592` 次；
  `SubCoreCompute<false>` 是 `1200` 次；二者合计 `1792` 次。
- `CalcExp` 的 `Brcb(hm)` 不是 `2304` 次，而是 `1792` 次。
- `CalcExp` 内逐 64-element `Sub` 段数不是 `18432` 次，而是 `13008` 次。
- `isFirstStackTile` 的 `120` 是 operator/call 级别，不能直接和
  row-loop/subcore 级别的 `2304` 混算。按 subcore 级别，first-stack 是 `832`
  次，non-first 是 `960` 次。
- `EpilogueRescaleO` 的 `288` operator 调用、`576` subcore 调用、`Brcb(dm)=336`、
  `Brcb(gl)=240` 这些数在当前 shape 下成立。
- 原报告中的 “TQ Pack Brcb=0” 已不符合当前代码：key1 normalize 使用 `Brcb`，
  reduce-sum encode 也使用 `Brcb + Compare + Select + WholeReduceSum`。

FIAS 的流水值得借鉴，但不能照搬到 TQ4bit。FIAS 的流水是 QK Cube、Softmax Vec、
PV Cube、Rescale Vec 围绕 KV stack 的跨阶段生产消费流水；TQ4bit 的真实链路是
`CopyIn -> Normalize -> RotateMatmul -> Encode -> Pack/CopyOut`，其中 normalize
必须先于 rotate，encode 必须等待 rotate，CopyOut 必须等待 encode。二者依赖图不同，
强行仿照 FIAS 把中间结果落 GM 并用 `PRE_LAUNCH` 轮转，可能增加额外 GM 往返和同步，
不一定减少指令数。

当前 TQ4bit 的 OPP 指令数也支持这个判断：在最新可映射源码的 OPP 中，
`ComputeBatch` 的 source-line 指令数约 `1.75M`，其中 `EncodeBatch` 约 `0.97M`、
`NormalizeBatchBrcbScale` 约 `0.70M`，而 `RotateBatchMatmul` 只有约 `0.079M`。
这说明当前瓶颈主要在 Vec normalize/encode 和 pack emit，而不是 rotate Cube 本身。

## 代码入口校正

### FIAS 入口链

实际入口在 `fused_infer_attention_score.cpp`：

```text
fused_infer_attention_score()
  -> TILING_KEY_VAR >= FAI_FLAG_TILING
  -> KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
  -> SplitFuse::FAInfer<half, half, float, false, MASK_CAUSAL, TND>()
```

`flash_attention_interface.cpp` 内实例化：

```text
SplitFuse::FAInfer()
  -> FAInferKernel<BlockMmadQK, BlockMmadPV,
                  EpilogueOnlineSoftmax, EpilogueRescaleO,
                  EpilogueInitOut, false, MASK_CAUSAL, TND>
  -> FAInferKernel::operator()(params)
```

说明：源码层面 `FAInferKernel::operator()` 是 kernel 对象的执行体。原报告写
“全局调用 8 次”容易引入歧义。更准确的说法是：kernel 启动后，每个执行 block/core
都有一个执行实例，task 分配由 `coreIdx/coreNum` 控制。MIX AIC/VEC 下 Vec 侧会把
`coreIdx = GetBlockIdx() / GetSubBlockNum()`，用于让两个 subblock 处理同一个
逻辑 core 的不同 subblock。

### Tiling 计算

对 shape `[1802,8,128]`，假设 `qHeads=8`、`kvHeads=1`、`groupSize=8`、
`qSeqlen=kvSeqlen=1802`：

```text
GetQNBlockTile(qSeqlen=1802, groupSize=8)
  = max(1, min((128 / 1802) / 2 * 2, 8))
  = 1

GetQSBlockTile(kvSeqlen=1802)
  = Q_TILE_CEIL
  = 128

curQNBlockNum = ceil(groupSize / qNBlockTile) * kvHeads = 8
curQSBlockNum = ceil(1802 / 128) = 15
totalTaskNum = 8 * 15 = 120
```

每个 task 对应一个 `(qSBlockIdx, qNBlockIdx)`，覆盖一个 query head 和最多
128 个 token。

## 修正后的 FIAS 调用过程

### Task 和 KV Stack

`flash_attention_regular.h` 主循环：

```text
for taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += coreNum
```

每个 task 计算：

- `qSBlockIdx = taskIdxCurBatch / curQNBlockNum`
- `qNBlockIdx = taskIdxCurBatch % curQNBlockNum`
- `qSBlockSize = 128`，最后一个 qSBlock 为 `10`
- `qNBlockSize = 1`
- `rowNum = qSBlockSize * qNBlockSize`
- causal 场景下
  `noSkipKvS = min(kvSeqlen, (qSBlockIdx + 1) * 128)`
- `kvSLoopNumTotal = ceil(noSkipKvS / 512)`

逐 qSBlock 的 stack：

| qSBlock | qSBlockSize | noSkipKvS | kvSLoopNum | stack 模式 |
|---:|---:|---:|---:|---|
| 0 | 128 | 128 | 1 | `M128` |
| 1 | 128 | 256 | 1 | `M256` |
| 2 | 128 | 384 | 1 | `M384` |
| 3 | 128 | 512 | 1 | `M512` |
| 4 | 128 | 640 | 2 | `N512, M128` |
| 5 | 128 | 768 | 2 | `N512, M256` |
| 6 | 128 | 896 | 2 | `N512, M384` |
| 7 | 128 | 1024 | 2 | `N512, M512` |
| 8 | 128 | 1152 | 3 | `N512, N512, M128` |
| 9 | 128 | 1280 | 3 | `N512, N512, M256` |
| 10 | 128 | 1408 | 3 | `N512, N512, M384` |
| 11 | 128 | 1536 | 3 | `N512, N512, M512` |
| 12 | 128 | 1664 | 4 | `N512, N512, N512, M128` |
| 13 | 128 | 1792 | 4 | `N512, N512, N512, M256` |
| 14 | 10 | 1802 | 4 | `N512, N512, N512, M266` |

`M` 表示 `doTriUMask=true`，走 causal-mask overload；`N` 表示 `doTriUMask=false`，
走 no-mask overload。判断来自代码：

```cpp
uint32_t triUp = noSkipKvS - qSBlockSize;
uint32_t kvSStartIdx = kvSIdx * MAX_KV_STACK_LEN;
uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
bool doTriUMask = triUp < kvSEndIdx - 1;
```

由此得到：

- 每 head masked stack 数：`15`
- 每 head no-mask stack 数：`21`
- 全 head masked softmax operator：`15 * 8 = 120`
- 全 head no-mask softmax operator：`21 * 8 = 168`
- 全 head QK/PV stack：`(15 + 21) * 8 = 288`

### KV Loop 总次数

每 task 内：

```text
for kvSIdx = 0; kvSIdx < kvSLoopNumTotal + PRE_LAUNCH; ++kvSIdx
```

`PRE_LAUNCH = 2`，因此：

- 每 head KV loop 总数：`sum(kvSLoopNum + 2) = 66`
- 全 head KV loop 总数：`66 * 8 = 528`
- `kvSIdx < kvSLoopNumTotal` 分支触发 `288` 次，执行 QK + Softmax。
- `kvSIdx >= PRE_LAUNCH` 分支触发 `288` 次，执行 PV + RescaleO。

这部分原报告是正确的。

### QK + Softmax 分支

每个 stack 先在 Cube 侧执行 QK：

```text
blockMmadQK(...)
CrossCoreSetFlag(qkReady)
```

次数：

- `blockMmadQK.loadQGM`：`120` 次，每 task 一次。
- `blockMmadQK(...)`：`288` 次，每 KV stack 一次。
- `CrossCoreSetFlag(qkReady)`：`288` 次。

Vec 侧按 `doTriUMask` 分流：

- masked causal overload：`120` 次。
- no-mask overload：`168` 次。
- `CrossCoreSetFlag(softmaxReady)`：`288` 次。

原报告的问题是把 `288` 次全部算成 masked causal overload，并进一步把每次都按
512 列、每 subblock 4 个 row-loop 展开。这会高估 masked 路径和 mask 搬运。

### OnlineSoftmax Row Loop

`EpilogueOnlineSoftmax::operator()` 中，每个 subblock 的循环数量由
`rowActualThisSubBlock` 和 `columnNumRound` 决定：

```text
maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound
rowNumTile = min(RoundDown(maxRowNumPerLoop, 8), 64)
rowLoopNum = ceil(rowActualThisSubBlock / rowNumTile)
preLoad = 1
```

对于 `qSBlockSize=128`，两个 subblock 各 64 行。列宽为 512 时：

```text
rowNumTile = min(RoundDown(8192 / 512, 8), 64) = 16
rowLoopNum = ceil(64 / 16) = 4
```

但尾列 128/256/384 和最后 qSBlock 的 10 行并不是这个模式。

按代码公式对 shape `[1802,8,128]` 展开，得到 softmax subcore 级别：

| 项 | 次数 |
|---|---:|
| `EpilogueOnlineSoftmax::operator()` masked overload | 120 |
| `EpilogueOnlineSoftmax::operator()` no-mask overload | 168 |
| `SubCoreCompute<true>` | 592 |
| `SubCoreCompute<false>` | 1200 |
| `SubCoreCompute` 合计 | 1792 |
| masked overload 的 `rowLoopIdx` 循环迭代含 preload | 832 |
| no-mask overload 的 `rowLoopIdx` 循环迭代含 preload | 1536 |
| rowLoop 迭代合计含 preload | 2368 |

因此原报告中的：

```text
SubCoreCompute<true> = 2304
CopySGmToUb = 2304
UpCastMask = 2304
ApplyMask = 2304
```

不成立。更准确的理解：

- `CopySGmToUb` 发生在所有 masked/no-mask 的真实 row-loop load 分支中，按
  subcore compute loop 合计是 `1792` 次。
- `UpCastMask` 和 `ApplyMask` 只发生在 masked causal overload，按
  `SubCoreCompute<true>` 是 `592` 次。
- no-mask overload 不执行 `UpCastMask` 和 `ApplyMask`。

### OnlineSoftmax 内部热点

`SubCoreCompute<doTriUMask>` 内部顺序：

```text
CalcLocalRowMax
UpdateGlobalRowMax
CalcExp
DownCastP
CalcLocalRowSum
CopyPUbToGm
UpdateGlobalRowSum
```

对当前 shape：

| 项 | 修正后次数 | 说明 |
|---|---:|---|
| `CalcLocalRowMax` | 1792 | 每个 softmax subcore compute 一次 |
| `UpdateGlobalRowMax` | 1792 | 每个 softmax subcore compute 一次 |
| `CalcExp` | 1792 | 每个 softmax subcore compute 一次 |
| `DownCastP` | 1792 | 每个 softmax subcore compute 一次 |
| `CalcLocalRowSum` | 1792 | 每个 softmax subcore compute 一次 |
| `CopyPUbToGm` | 1792 | P 写 GM workspace |
| `UpdateGlobalRowSum` | 1792 | 每个 softmax subcore compute 一次 |
| `CalcExp` 内 `Brcb(hm)` | 1792 | 原报告写 2304，过高 |
| `CalcExp` 内 64-element `Sub` 段 | 13008 | 按 `ceil(columnNum/64)` 汇总 |

`isFirstStackTile` 的口径需要分层：

| 口径 | first | non-first |
|---|---:|---:|
| operator/call 级别 | 120 | 168 |
| subcore compute 级别 | 832 | 960 |

原报告把 “每 task 第一个 KV stack 是 first，共 120 次” 和
“row-loop/subcore 展开共 2304 次” 混在一张表中，导致 `2184` 这样的 else 次数不可靠。

按列宽分类的 softmax subcore compute：

| 列宽路径 | subcore compute 次数 |
|---|---:|
| 512 列 | 1392 |
| 256 列 | 128 |
| tail 路径 | 272 |

其中 128/384/266 等列宽走 `RowmaxTAILTILE` / `RowsumTAILTILE`，不能简单按
`SPECTILE512` 的 3 次 block reduce 来估算。

### PV + RescaleO 分支

每个 stack 在延迟 `PRE_LAUNCH=2` 后触发 PV：

```text
blockMmadPV(...)
CrossCoreSetFlag(pvReady)
CrossCoreWaitFlag(pvReady)
epilogueRescaleO(...)
```

次数：

| 项 | 次数 |
|---|---:|
| `blockMmadPV(...)` | 288 |
| `CrossCoreSetFlag(softmaxReady)` | 288 |
| `CrossCoreWaitFlag(softmaxReady)` | 在 `blockMmadPV` 内按 stack 等待 |
| `CrossCoreSetFlag(pvReady)` | 288 |
| `CrossCoreWaitFlag(pvReady)` | 288 |
| `EpilogueRescaleO::operator()` | 288 |
| `EpilogueRescaleO::SubCoreCompute` | 576 |

RescaleO 的原报告计数在本 shape 下基本成立：

| 分支 | subcore 次数 | 操作 |
|---|---:|---|
| `isFirstStackTile=true` | 240 | `go = lo` |
| `isFirstStackTile=false` | 336 | `Brcb(dm) + Mul(go, dm) + Add(go, lo)` |
| `isLastStackTile=true` | 240 | `Brcb(gl) + Div(go, gl) + Cast + CopyOToGm` |
| `isLastStackTile=false` | 336 | 中间 O update 相关路径 |

这里每个 subblock 的 row-loop 为 1，因为 `embedV=128`，`rowNumTile=64`，
128 行时每 subblock 64 行，最后 qSBlock 每 subblock 5 行，也仍然是 1 个 row-loop。

## 原报告逐项校正表

| 原报告结论 | 校正后结论 | 原因 |
|---|---|---|
| task 总数 120 | 正确 | `15 qSBlocks * 8 qHeads` |
| KV loop total 528 | 正确 | `sum(kvSLoopNum + 2) * 8 = 66 * 8` |
| QK/PV iterations 288 | 正确 | `sum(kvSLoopNum) * 8 = 36 * 8` |
| 几乎所有 KV 迭代走 causal mask path | 错误，masked 120，no-mask 168 | `doTriUMask = triUp < kvSEndIdx - 1`，只有每个 qSBlock 最后一个 stack masked |
| `EpilogueOnlineSoftmax::operator(causal)` 288 | 错误，应为 120；no-mask overload 168 | masked/no-mask overload 分开调用 |
| `SubCoreCompute<true>` 2304 | 错误，应为 592 | masked 且按真实列宽/尾行计算 |
| `SubCoreCompute<false>` 未展开 | 应为 1200 | no-mask stack 占多数 |
| `CalcExp Brcb` 2304 | 错误，应为 1792 | 每个 softmax subcore compute 一次 |
| `CalcExp Sub` 18432 | 错误，应为 13008 个 64-element 段 | 按各 stack `ceil(columnNum/64)` 统计 |
| `UpdateGlobalRowMax else=2184` | 口径错误 | first/non-first 不能用 operator 级 120 去减 row-loop 级 2304 |
| RescaleO operator 288 | 正确 | 每 PV stack 一次 |
| RescaleO subcore 576 | 正确 | 每 operator 两个 subblock，一个 row-loop |
| RescaleO `Brcb(dm)=336`、`Brcb(gl)=240` | 正确 | 按 subcore first/last stack 统计 |
| TQ Pack `Brcb=0` | 当前代码已不成立 | key1 normalize 和 reduce-sum encode 已使用 Brcb |

## 当前 TQ4bit 执行流程

当前入口：

```text
turboquant_pack_kv_for_cache4bit()
  -> TurboquantPackKVForCache4bitToCache::Process()
  -> ProcessCachePairBySequenceContiguousSegments(worker)
```

### Worker 范围

`ProcessCachePairBySequenceContiguousSegments` 按线性 token 均分：

```text
tokensPerWorker = ceil(tokenCount / activeWorkers)
rawBegin = worker * tokensPerWorker
rawEnd = min(tokenCount, rawBegin + tokensPerWorker)
```

然后对每个 request 与 `[rawBegin, rawEnd)` 求交集，并按 cache group 边界调整：

- begin 落在 group 中间时，回退到 group 起点，当前 worker 处理完整 group。
- end 落在 group 中间时，非最后 worker 回退到 group 起点，不处理该 group。
- 最后 worker 可以处理末尾 group。

每个 seq 只解析第一个 token 的物理 slot，后续 token 通过 `firstSlot + offset`
连续推导。

### Block 和 Group 路径

每段 token 进入：

```text
ProcessSequenceBlockRange()
  -> leading partial group: AppendPhysicalCacheGroupRows(...)
  -> full group run: PackCachePairPhysicalFullGroupRun(...)
  -> tail partial group: AppendPhysicalCacheGroupRows(...)
```

full-group 快路径：

```text
PackCachePairPhysicalFullGroupRun()
  while consumedRows < rowCount:
    rowsThisBatch = floor((batchCapacity / numHeads) / 4) * 4
    PackCachePhysicalFullGroupRunTask(key...)
    PackCachePhysicalFullGroupRunTask(value...)
```

其中 `batchCapacity = GetBatchCapacity()`，上限 `TQ_MAX_BATCH_M=64`，也可能受
tiling 中 `vecPerCore_` 限制。

### 单个 Task 流程

full-group 单 task：

```text
PackCachePhysicalFullGroupRunTask(xGm, packedGm, ...)
  -> CopyInPhysicalGroupRowsTask(...)
  -> ComputeBatch(m)
  -> CopyOutPhysicalFullGroupRunKnown(...)
```

`ComputeBatch(m)` 固定为：

```text
NormalizeBatch(m)
RotateBatchMatmul(m, 0, 128)
EncodeBatch(m)
```

当前 key1 normalize：

```text
Cast x row -> fp32
Mul square
ReduceSum
Sqrt
Cast norm
Adds eps
Div reciprocal
Brcb reciprocal to row scale
Mul fp32Row by scale
Cast aBatch
```

当前 reduce-sum encode：

```text
Brcb yFp32 into threshold-comparison layout
Compare against quant thresholds
Duplicate 1.0
Select mask -> 1/0
WholeReduceSum -> quant index
Cast fp32 -> int32 -> int16
And 0x0F
DataCopy index row
Copy norm word
```

CopyOut full-group：

```text
CopyOutPhysicalFullGroupRunKnown()
  for each cache group and head:
    InitFullGroupFromRow0()
    MergeEncodedRowToGroup(row1)
    MergeEncodedRowToGroup(row2)
    MergeEncodedRowToGroup(row3)
    copy_packed_ub_to_gm_async()
```

`MergeEncodedRowToGroup` 内仍有：

- `ShiftLeft` 把 4-bit code 放到 groupRow 对应 nibble。
- `Or` 合并到 packed group。
- `SetValue/GetValue` 写 norm word。

## TQ4bit OPP 指令数佐证

FIAS 没有 OPP，因此本节只统计 TQ4bit。使用的数据为：

```text
OPP: /root/x00827378/vllm-ascend/mytmp/OPPROF_20260628005409_PZNDWDIEMDWIZEHR
op:  TurboquantPackKvForCache4bit_cf5c5d6d4ad385ca5cf1f7b534e9a391_1_mix_aic
profile shape: q_lens=[512], kv_lens=[512], pack_tokens=512,
               heads=16, kv_heads=8, block_size=128
Task Duration(us): 299.540009
source-line total instructions: 26,414,734
selected turboquant_pack_kv_for_cache4bit.cpp instructions: 19,021,616
command:
  python3 tools/extract_opprof_source_lines.py \
    mytmp/OPPROF_20260628005409_PZNDWDIEMDWIZEHR \
    --source turboquant_pack_kv_for_cache4bit.cpp \
    --group-by function --top 40
```

注意：该 OPP 是最新一次可映射源码的 TQ4bit op profile，用于验证当前实现热点。
它不是 FIAS profile，也不用于推导 FIAS 的函数指令数；它也只作为当前 TQ4bit 代码
热点排序的佐证，不替代 `[1802,8,128]` 的 FIAS 源码调用次数推导。
`extract_opprof_source_lines.py` 按 MindStudio source-line 记录和源码函数范围聚合；
由于 AscendC 内联和调用点归因，上层 wrapper 行与被调用函数可能同时出现，表中函数
指令数用于排序和定位热点，不应把所有函数行简单相加当作互斥占比。

### 函数级热点

| 函数 | Instructions Executed | Selected% | 热点行 | 结论 |
|---|---:|---:|---|---|
| `Process` | 2,149,926 | 11.30% | `420:2149406` | 主要归因到调用 `ProcessCachePairBySequenceContiguousSegments` 的入口行，说明主执行体集中在该路径 |
| `ProcessCachePairBySequenceContiguousSegments` | 2,148,758 | 11.30% | `1740:2141662` | 主要归因到 block range 处理调用链，证实当前是 contiguous segment 路径 |
| `ProcessSequenceBlockRange` | 2,140,738 | 11.25% | `1637:2139058` | full-group run 是热路径，partial/fallback 不是本次 OPP 主体 |
| `PackCachePairPhysicalFullGroupRun` | 2,137,766 | 11.24% | `1439:1092632`, `1444:1038934` | key/value 两次 full-group task 指令接近，二者串行但热点均在 task 调用 |
| `PackCachePhysicalFullGroupRunTask` | 2,131,566 | 11.21% | `1401:1749542`, `1402:370104` | `ComputeBatch` 明显高于 full-group CopyOut，但 CopyOut 仍有可见成本 |
| `ComputeBatch` | 1,749,542 | 9.20% | `917:697904`, `918:78662`, `919:972976` | encode > normalize >> rotate，rotate Cube 不是当前指令主因 |
| `EncodeBatch` | 971,696 | 5.11% | `729:417792`, `727:98304`, `723:61984` | encode 是 ComputeBatch 内最大热点 |
| `NormalizeBatchBrcbScale` | 697,904 | 3.67% | `486:188416`, `508:98304`, `483:73728` | Brcb normalize 仍是第二大计算热点 |
| `EncodeQuantCodesByReduceSum` | 417,792 | 2.20% | `684:114688`, `663:65536`, `682:57344` | reduce-sum 查表路径已生效，但仍占 encode 的主要部分 |
| `CopyOutPhysicalFullGroupRunKnown` | 352,536 | 1.85% | `1248:208896`, `1246:47104`, `1252:43008` | full-group CopyOut 有优化空间 |
| `MergeEncodedRowToGroup` | 208,896 | 1.10% | `957:53248`, `968:43008`, `944:30720` | pack merge 中 shift/or/norm scalar 写是 CopyOut 的核心成本 |
| `RotateBatchMatmul` | 78,662 | 0.41% | `543:21622`, `536:13600`, `544:13280` | rotate matmul 启动有成本，但在当前 OPP 中远低于 normalize/encode |
| `CopyInPhysicalGroupRowsTask` | 10,800 | 0.06% | `892:4080`, `913:3680` | 连续 CopyIn 已不是主要热点 |

### 从 OPP 得到的直接判断

1. `ComputeBatch` 内部热点排序明确：

   ```text
   EncodeBatch              971,696
   NormalizeBatchBrcbScale  697,904
   RotateBatchMatmul         78,662
   ```

   因此后续优先级应继续放在 encode、normalize、CopyOut pack，而不是先大改 rotate。

2. full-group path 已经是当前主路径。`ProcessSequenceBlockRange` 的热点行集中在
   `PackCachePairPhysicalFullGroupRun` 调用，`FlushResolvedGroups` 在本次 OPP 中为
   `0`，说明 fallback/partial preserve 不是这次 profile 的主要成本。

3. key/value 两次 full-group task 的 call-site 指令数接近：

   ```text
   key task call line 1439:   1,092,632
   value task call line 1444: 1,038,934
   ```

   这支持“key/value 串行不是单独异常热点”的判断。后续即使考虑流水，也应围绕
   batch 间 copy/compute/copyout 重排，而不是简单把 key/value 交错当作主要方案。

4. CopyOut pack/emit 确实有价值，但不是唯一瓶颈：

   ```text
   CopyOutPhysicalFullGroupRunKnown 352,536
   MergeEncodedRowToGroup           208,896
   copy_packed_ub_to_gm_async        43,008
   InitFullGroupFromRow0             45,056
   ```

   这说明 CopyOut pack/emit 优化能降低指令数，但即使完全消除 merge，也不能覆盖
   normalize/encode 的全部差距。因此 CopyOut 优化之后还需要继续做 batched normalize
   或 encode 侧优化。

5. 当前 reduce-sum encode 已经将 `Brcb + Compare + Select + WholeReduceSum` 作为热点：

   ```text
   EncodeQuantCodesByReduceSum 417,792
   hot lines: 684, 663, 682
   ```

   这修正了旧报告中 “TQ Pack Brcb=0” 的结论。后续不应再把 “引入 Brcb” 作为笼统目标，
   而要看 Brcb 后的 `Compare/Select/WholeReduceSum` 是否还能减少，或者是否能在
   encode 输出布局上同时服务 CopyOut pack。

## 可借鉴点

### 1. 先建立 shape 级调用计数，再决定优化对象

FIAS 的教训是：顶层次数正确并不代表热点展开正确。原报告在 288 个 stack 的基础上
把全部 softmax 当 masked 512 列，导致 mask、row-loop、Brcb、Sub 等低层指令计数
明显偏离。

对 TQ4bit 后续每个实验也应先建立类似表：

- 每 worker 实际 token 范围。
- full-group、leading partial、tail partial 各自触发次数。
- `rowsThisBatch` 分布。
- key/value 各自 `CopyIn/Normalize/Rotate/Encode/CopyOut` 次数。
- full-group fast path 与 fallback path 次数。
- `GetValue/SetValue`、`Brcb`、`WholeReduceSum`、`ShiftLeft/Or` 次数。

这样可以避免只看 profile 总耗时时误判，比如 OPP 总耗时波动较大，但
`TurboquantPackKVForCache4bitToCache.Process()` 指令数和热点函数计数仍可辅助定位。
当前 OPP 已经给出一个基准：`EncodeBatch` 和 `NormalizeBatchBrcbScale` 是
`ComputeBatch` 内两大热点，CopyOut pack 是下一层热点。因此后续实验报告应同时记录：

- `ComputeBatch` 总指令数。
- `NormalizeBatch*` 指令数。
- `EncodeBatch` / `EncodeQuantCodesByReduceSum` 指令数。
- `CopyOutPhysicalFullGroupRunKnown` / `MergeEncodedRowToGroup` 指令数。
- `RotateBatchMatmul` 指令数，防止为小热点投入过大改造。

### 2. 借鉴 FIAS 的大 tile 思路，而不是照搬 tile 形状

FIAS 的主 task 是 `128 tokens * 1 head`，QK/PV 使用 `[128,512,128]`、
`[128,128,512]` 级别的大 Cube shape。大 tile 的收益是：

- task 数少，调度和固定开销被摊薄。
- Cube 每次启动做的有效计算量大。
- Vec epilogue 可以按行块批处理，scalar 行状态留在 UB。

TQ4bit 的 rotate 是 `[m,128] * [128,128]`，当前 `m` 最大 64，并且要受
cache group、head、UB live-set 约束。直接扩大到 128 行不一定可行，因为当前 UB 同时
保存：

- `xBatch` queue。
- `aBatch` queue。
- `yBatch` queue。
- `encodedBatch` queue。
- norms。
- reduce/quant/code/pack 临时 buffer。
- matmul local workspace。

可借鉴的方向是减少 live-set 后再扩大 batch，而不是简单把 `TQ_MAX_BATCH_M` 改大。
例如：

- key1 full-group 路径中，如果 encode 能直接生成 group-major packed buffer，
  可以缩小 `encodedBatch` 或减少 pack 临时区。
- normalize 如果能按更大行块批量处理 norm 标量，减少每行固定同步，才有扩大 m 的意义。
- 对 large shape 专门 tiling，保留 small shape 原路径，避免大 batch 优化拖慢小 shape。

### 3. FIAS 流水不能简单照抄到 TQ4bit

这是最关键的区别。

FIAS 的主循环围绕 KV stack：

```text
for kvSIdx in [0, kvSLoopNumTotal + PRE_LAUNCH):
  if kvSIdx < kvSLoopNumTotal:
    QK(kvSIdx) -> Softmax(kvSIdx)
  if kvSIdx >= PRE_LAUNCH:
    PV(kvSIdx - PRE_LAUNCH) -> RescaleO(kvSIdx - PRE_LAUNCH)
```

它能形成流水，是因为有四个阶段和两个独立 Cube 算子：

```text
QK Cube -> OnlineSoftmax Vec -> PV Cube -> RescaleO Vec
```

依赖关系是每个 stack 局部的：

- `Softmax(k)` 必须等 `QK(k)`。
- `PV(k)` 必须等 `Softmax(k)`。
- `RescaleO(k)` 必须等 `PV(k)`。
- 但是 `QK(k+2)` 和 `PV(k)` 使用不同输入/输出 workspace slot，可以在循环中错开。

FIAS 为此付出代价并获得收益：

- S/P/Otmp 写入 GM workspace。
- `qkReady`、`softmaxReady`、`pvReady` 做 Cube/Vec 跨核同步。
- `PRE_LAUNCH + 1` 个 workspace slot 作为 ring。
- `blockMmadQK` 和 `blockMmadPV` 是两个大 Cube MMAD，足以覆盖部分 GM workspace
  往返和同步成本。

TQ4bit 的依赖图不同：

```text
CopyIn(batch)
  -> Normalize(batch)
  -> RotateMatmul(batch)
  -> Encode(batch)
  -> Pack/CopyOut(batch)
```

这里没有 FIAS 那样的第二个大 Cube 消费第一个 Vec 输出后再回到 Vec 的结构：

- Normalize 输出是 Rotate 的输入，不能与同一 batch 的 Rotate 乱序。
- Rotate 输出是 Encode 的输入，Encode 不能提前。
- Pack/CopyOut 依赖 Encode 的 uint4 code 和 norm。
- key 和 value 是两套独立输入输出，但 key 完整处理完再处理 value，并不破坏
  batch 间 queue depth 的基本作用。把 key/value 交错并不等价于 FIAS 的
  QK/PV stack 流水，因为二者之间没有生产消费关系。

如果机械照搬 FIAS：

```text
Normalize(k) 写 GM
Rotate(k-1) 读 GM/写 GM
Encode(k-2) 读 GM
CopyOut(k-3)
```

风险很高：

- 原本 UB/queue 内传递的 `aBatch/yBatch/encodedBatch` 会变成 GM workspace 往返，
  增加 MTE2/MTE3 流量。
- normalize/encode 是短向量逐行热点，单阶段计算量不一定能覆盖额外搬运。
- rotate 只有一个 Cube stage，不像 FIAS 有 QK/PV 两个大 Cube stage 可错峰。
- 当前 packed cache 输出有 group/nibble/norm 合并约束，CopyOut 不是纯连续 O tile 写回。
- 需要更多 workspace slot 和 cross-stage flags，可能增加指令数，OPP 的
  `Process()` 指令数未必下降。

因此可借鉴的是 “把 batch loop 显式化、让已有 queue 和 async copy 有稳定跨 batch
重叠机会”，不是照搬 `PRE_LAUNCH=2` 的 GM workspace ring。

更适合 TQ4bit 的流水目标应是：

```text
for batch in batches:
  CopyIn(batch)              // xBatchQue depth=2
  Compute(batch)             // Normalize -> Rotate -> Encode
  CopyOut(batch)             // packed write async; 尽量延迟 Wait
```

并在 batch 边界保证：

- `copy_packed_ub_to_gm_async` 尽量晚 wait。
- 下一个 batch 的 CopyIn 可以在前一个 batch packed write pending 时推进。
- partial/preserve 路径因需要读旧 group，应显式 wait，避免覆盖 packed buffer。
- key/value 仍可串行，但要保证串行代码不提前 `WaitPendingPackedWrites()`。

### 4. 借鉴 FIAS 的分支专用化

FIAS 对 `MASK_CAUSAL` 下的 stack 又分成 masked/no-mask 两套 overload：

- no-mask 不做 mask GM copy、upcast、apply mask。
- masked 只在真正跨 causal 边界的最后 stack 执行。

TQ4bit 已经有类似思路：

- full-group run 使用 `PackCachePairPhysicalFullGroupRun`。
- full-group copyout 使用 `CopyOutPhysicalFullGroupRunKnown`。
- partial/preserve 走 `AppendPhysicalCacheGroupRows` 和 fallback。

后续可继续专用化：

- large shape key1 full-group path 保持独立，不影响 small shape。
- full group 且 `firstGroupRow=0`、`rowCount=4`、`headStart=0`、`headCount=numHeads`
  的热路径应尽量避免 preserve/fallback 分支。
- CopyIn 已有连续大块 copy，应统计 large shape 是否大部分走
  `rowCount * headCount * 128` 的单次 DataCopy。
- CopyOut 可探索 encode 阶段直接生成 group-major packed layout，减少
  `MergeEncodedRowToGroup` 中的 `ShiftLeft + Or + SetValue/GetValue`。

### 5. 借鉴 Brcb 广播标量，但要按当前数据布局选择位置

FIAS 的在线 softmax 和 RescaleO 大量使用 `Brcb`：

- `CalcExp` 把每行 `hm` 广播成 block。
- RescaleO 把 `dm`、`gl` 广播成 block，然后对 O 向量做 Mul/Div。

这证明对于“每行一个 scalar，应用到整行向量”的场景，`Brcb + vector op`
通常比 `GetValue + scalar Muls` 更适合。

TQ4bit 已经应用：

- key1 normalize 使用 `Brcb(norm reciprocal)` 后 `Mul`。
- reduce-sum encode 使用 `Brcb(yFp32)` 构造比较表，再 `WholeReduceSum`。

仍可继续挖掘：

- CopyOut norm 写入现在仍有 `encodedBatch.GetValue` 和 `packedU16.SetValue`。
  norm 只有 4 个 word/group，可以考虑在 group pack 时用更结构化的 DataCopy 或
  vector store 模式减少 scalar API，但需要验证指令数是否真的下降。
- 如果做 batched normalize，应优先减少 V/S 同步和 per-row scalar 读取，而不是
  回到逐行 `GetValue`。

### 6. 借鉴 FIAS 的 workspace ring 时必须限制范围

FIAS 的 GM workspace ring 是为 S/P/Otmp 这种大矩阵中间结果服务的。它的收益来自：

- 中间结果本身就是 QK/PV 两个大 Cube 的连接面。
- GM workspace slot 可让 Cube/Vec 用跨核 flag 解耦。
- 每个 stack 的粒度足够大。

TQ4bit 不能把所有中间张量都照搬到 GM workspace。更合理的限制是：

- 只对 packed CopyOut 做小 ring，因为它本来最终要写 GM。
- `aBatch/yBatch` 优先保留 queue/UB/L1 内传递。
- 如果要做 staged batch scheduler，也应该先在现有 `xBatchQue/aBatchQue/yBatchQue/encodedBatchQue`
  上重排 batch loop，而不是引入新的 GM 中间区。

### 7. 借鉴 FIAS 的最终连续写回，但要适配 4bit group layout

FIAS 的最终 `CopyOToGm` 是 O tile 写回，虽然要处理 TND stride，但每个 tile 的
数据语义是连续向量。

TQ4bit 的最终写回是 cache group：

```text
4 rows * 128 uint4 index + 4 norm words
```

当前 copyout 需要把 4 行 uint4 code 合并到一个 uint16 lane：

```text
row0 bits 0..3
row1 bits 4..7
row2 bits 8..11
row3 bits 12..15
```

这和 FIAS 的 O tile 不是同一种写回。可借鉴的是“让最终写回尽量连续”，具体到
TQ4bit 应落在：

- encode 阶段是否直接输出 packed group layout。
- full-group path 是否一次生成完整 group，避免 row-by-row merge。
- norm 是否与 packed index 一起构造好后一次 async copy。

## 后续建议

基于这次校正，建议下一步优化按以下优先级推进：

1. 为 TQ4bit large shape 建立同等级调用计数报告：统计 batch 数、rowsThisBatch、
   full/partial path、CopyIn 连续大块比例、CopyOut merge 次数和 scalar API 次数。
2. 先做 key1 full-group CopyOut pack/emit 专用化，目标是减少
   `MergeEncodedRowToGroup` 的 `ShiftLeft/Or/GetValue/SetValue`，并保持 small shape
   走旧路径。
3. 再做 batched vector normalize，目标是减少 per-row V/S 同步，保留 key1 的 Brcb
   思路，并严格对比 smoke/long query。
4. 如果要探索 batch 流水，先做现有 queue 和 async packed write 的 loop 重排，
   不引入 GM 中间 workspace；只有当 OPP 指令数和 smoke/long 都显示收益，再考虑更复杂
   的 ring buffer。

验证标准仍应保持：

- smoke 输出语义不能重复、不能异常。
- long query 必须不退化。
- OPP 不看总耗时单点波动，重点看
  `TurboquantPackKVForCache4bitToCache.Process()` 指令数和热点函数变化。
- small 和 large shape 要隔离评估；large 有收益但 small 退化时，应通过 tiling key
  或路径分流隔离。
