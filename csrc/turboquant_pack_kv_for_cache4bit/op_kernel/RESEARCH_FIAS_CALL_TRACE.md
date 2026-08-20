# FIAS 代码调用过程报告：shape [1802, 8, 128]

> 日期：2026-06-27
> 参考版本：ops-transformer `fused_infer_attention_score` FAInfer 路径
> Scope：仅展开 `fused_infer_attention_score` 目录下的代码，不展开
>        `incre_flash_attention` 和 `prompt_flash_attention` 的调用

---

## 1. 参数设定

| 参数 | 值 | 说明 |
|------|-----|------|
| qHeads (numHeads) | 8 | Query 头数 |
| kvHeads | 1 | KV 头数（GQA，groupSize=8） |
| embed (embeddingSize) | 128 | 头维度 |
| embedV (embeddingSizeV) | 128 | V 头维度 |
| qSeqlen | 1802 | Query token 数 |
| kvSeqlen | 1802 | KV token 数（自注意力= qSeqlen） |
| batch | 1 | 批次大小 |
| maskType | 1 (MASK_CAUSAL) | Causal mask |
| pagedCacheFlag | false | 无 paged cache |
| scaleValue | 1/√128 ≈ 0.0884 | Softmax 缩放因子 |
| dataType | FP16 | half |
| coreNum | 8 | A2 AICore 数量 |
| KERNEL_TASK_TYPE | MIX_AIC_1_2 | Cube+Vec 双核模式 |
| TILING_KEY | ≥ FAI_FLAG | SplitFuse FAInfer 路径 |

---

## 2. Tiling 计算

### 2.1 任务分解参数

```
qSBlockTile = Q_TILE_CEIL = 128                ← 固定值
qNBlockTile = min((128/1802)//2*2, 8) = 1     ← qSeqlen 太大，每 task 仅 1 个 qHead
groupSize = 8 / 1 = 8
qNBlockNumPerGroup = ceil(8/1) = 8
curQNBlockNum = 8 × 1 = 8                     ← 每 qSBlock 有 8 个 task
curQSBlockNum = ceil(1802/128) = 15            ← 15 个 qSBlock
totalTaskNum = 8 × 15 = 120
```

每个 task = `(qSBlockIdx, qNBlockIdx)`，覆盖 1 个 query head × 最多 128 tokens。

### 2.2 每个 qSBlock 的 KV stack 数

| qSBlockIdx | qSBlockSize | noSkipKvS | kvSLoopNum |
|-----------|-------------|-----------|------------|
| 0 | 128 | 128 | 1 |
| 1 | 128 | 256 | 1 |
| 2 | 128 | 384 | 1 |
| 3 | 128 | 512 | 1 |
| 4 | 128 | 640 | 2 |
| 5 | 128 | 768 | 2 |
| 6 | 128 | 896 | 2 |
| 7 | 128 | 1024 | 2 |
| 8 | 128 | 1152 | 3 |
| 9 | 128 | 1280 | 3 |
| 10 | 128 | 1408 | 3 |
| 11 | 128 | 1536 | 3 |
| 12 | 128 | 1664 | 4 |
| 13 | 128 | 1792 | 4 |
| 14 | 10 | 1802 | 4 |

KV loop iterations = kvSLoopNum + PRE_LAUNCH(=2)

### 2.3 Per-Core 任务分配

```
for taskIdx = coreIdx; taskIdx < 120; taskIdx += 8
```

每 core 15 个 task（120/8=15）。Core 0 处理 taskIdx 0,8,16,...,112。

---

## 3. 总入口调用链

### 3.1 入口函数

```
fused_infer_attention_score.cpp: fused_infer_attention_score()   [line 31]
  │
  │ TILING_KEY_VAR ≥ FAI_FLAG_TILING (≥5×10^18)
  │ → SplitFuse::FAInfer<half, half, float, false, MASK_CAUSAL, TND>()   [line 100]
  │
  │ else if TILING_KEY_VAR ≥ PFA_FLAG → prompt_flash_attention (不展开)
  │ else if TILING_KEY_VAR ≥ FIA_FLAG → fused_infer_attention (不展开)
  │ else → incre_flash_attention (不展开)
```

**调用次数：1 次**（kernel 入口）

### 3.2 FAInfer 模板实例化

```
flash_attention_interface.cpp: FAInfer()   [line 29-107]
  │
  │ 类型参数:
  │   InputDtypeQ = half
  │   InputDtypeKv = half
  │   IntermCalcPrec = float
  │   PagedCacheFlag = false
  │   maskCategory = MASK_CAUSAL
  │   inLayout = TND
  │   lseMode = NONE
  │   sinkMode = DISABLE
  │
  │ 实例化类型:
  │   FAInferKernel<BlockMmadQK, BlockMmadPV,
  │                 EpilogueOnlineSoftmax, EpilogueRescaleO, EpilogueInitOut,
  │                 false, MASK_CAUSAL, TND>
  │
  │ FAIKernelParams params{...};
  │ FAInferKernel flashAttnInfer;                        [line 105]
  │ flashAttnInfer(params);                              [line 106]
```

**调用次数：1 次**

---

## 4. FAInferKernel::operator() — 主循环

```
flash_attention_regular.h: FAInferKernel::operator()   [line 72-560]
```

### 4.1 初始化阶段（每 AICore 1 次）

**Cube 侧** (`__DAV_C220_CUBE__`)：

| 操作 | 代码行 | 说明 |
|------|--------|------|
| SetFlag M_MTE1 ×8 | [line 128-135] | 8 个 Cube→L1 数据搬运 event |
| SetFlag FIX_M ×2 | [line 136-137] | 2 个 FixPipe event |
| SetFlag MTE1_MTE2 ×8 | [line 138-145] | 8 个 L1→L2 event |
| 计算动态 L1 tile 参数 | [line 147-162] | nDynNum, kDynNum |
| blockMmadQK 初始化 | [line 159] | Cube QK MMAD 对象 |
| blockMmadPV 初始化 | [line 161] | Cube PV MMAD 对象 |

**Vec 侧** (`__DAV_C220_VEC__`)：

| 操作 | 代码行 | 说明 |
|------|--------|------|
| SetFlag MTE3_V ×6 | [line 164-169] | 6 个 Vec 搬入 event |
| SetFlag MTE3_MTE2 ×6 | [line 170-175] | 6 个搬出→搬入 event |
| SetFlag V_MTE2 ×4 | [line 177-180] | 4 个 Vec→L2 event |
| epilogueOnlineSoftmax 初始化 | [line 182] | Vec Softmax 对象 |
| epilogueRescaleO 初始化 | [line 183] | Vec RescaleO 对象 |
| epilogueInitOut 初始化 | [line 184] | Vec 初始化 O 对象 |

**调用次数：每 AICore 1 次**

### 4.2 任务循环 — 主执行体

```
for taskIdx = coreIdx; taskIdx < 120; taskIdx += 8   [line 228]
```

**每 AICore 执行 15 次**（15 个 task），8 个 core 共 120 次。

每次 task 循环体执行：

#### A. Batch 偏移追踪（1 次/task，仅在 batch > 1 时触发）

当前 batch=1，`curTotalTaskNum=120`，taskIdx 跳过 while 循环。

#### B. 计算当前 task 参数（1 次/task）

| 变量 | 计算公式 | 值范围 |
|------|---------|--------|
| qSBlockIdx | `taskIdxCurBatch / curQNBlockNum` | 0..14 |
| qNBlockIdx | `taskIdxCurBatch % curQNBlockNum` | 0..7 |
| kvNIdx | `qNBlockIdx / qNBlockNumPerGroup` | 0 (kvHeads=1) |
| qNStartIdx | `0 × 8 + qNBlockIdx × 1` | 0..7 |
| qSBlockSize | min(128, 1802 - qS×128) | 128 或 10 |
| qNBlockSize | 1 | 固定 |
| rowNum | qSBlockSize × qNBlockSize | 128 或 10 |
| noSkipKvS | min(1802, (qS+1)×128) | 128..1802 |
| kvSLoopNumTotal | ceil(noSkipKvS/512) | 1..4 |

#### C. EpilogueInitOut（仅当 kvSLoopNumTotal ≤ 0，此处不触发）

对于 shape [1802,8,128]，所有 task 的 kvSLoopNumTotal ≥ 1，
`EpilogueInitOut` **0 次调用**。

#### D. Cube 侧：loadQGM（1 次/task）

```
blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads)   [line 318]
```

**调用次数：120 次**（8 core × 15 task）

#### E. KV Loop — 核心计算循环

```
for kvSIdx = 0; kvSIdx < kvSLoopNumTotal + 2; kvSIdx++   [line 320]
```

循环总次数 = kvSLoopNumTotal + PRE_LAUNCH(=2)。
全局总计 528 次迭代（8 core × 各 task 的 KV 迭代之和）。

每 KV 迭代分两个条件分支：

##### 分支 1：`kvSIdx < kvSLoopNumTotal` — QK 计算 + Softmax

触发次数 = kvSLoopNumTotal（1..4 次/task）。全局共 36 × 8 = 288 次
（此为每 core 的 QK 迭代数 × 8 core；但精确计数按 qSBlockIdx 分组：
前 4 个 qSBlock × 1 KV × 8 head × 8 core = 256，
后 11 个 × 2~4 KV × 8 head × 8 core = ... 详见第 5 节）。

**Cube 侧执行：**

```
blockMmadQK(gQ, gK, gS, ...)   [line 336-363]
CrossCoreSetFlag(qkReady)       [line 364]
```

**Vec 侧执行（causal mask 路径）：**

`doTriUMask` 判断：当 `triUp < kvSEndIdx - 1` 时为 true。

| qSBlockIdx | triUp | doTriUMask | 路径 |
|-----------|-------|------------|------|
| 0 | 0 | true | causal mask path |
| 1 | 0 | true | causal mask path |
| ... | 0..varies | true | causal mask path |
| 所有 | ≤ kvSEnd | true | SubCoreCompute\<true> |

**几乎所有 KV 迭代走 causal mask 路径**（`doTriUMask=true`）。

```
epilogueOnlineSoftmax(gP, gS, gSink, gMask, ...)   [line 381-399]  ← causal mask overload
CrossCoreSetFlag(softmaxReady)                     [line 433]
```

##### 分支 2：`kvSIdx ≥ PRE_LAUNCH(=2)` — PV 计算 + RescaleO

触发次数 = kvSLoopNumTotal（延迟 2 tiles 后开始）。全局共 288 次。

**Cube 侧执行：**

```
blockMmadPV(gP, gV, gOTmp, ...)   [line 455-488]
CrossCoreSetFlag(pvReady)          [line 489]
```

**Vec 侧执行：**

```
CrossCoreWaitFlag(pvReady)          [line 497]
epilogueRescaleO(gO, gOTmp, gOUpdate, gLse, ...)   [line 499-513]
```

---

## 5. EpilogueOnlineSoftmax 详细调用过程

### 5.1 operator() causal mask overload

```
block_epilogue_online_softmax.hpp: operator()(gP, gS, gSink, gMask, ...)   [line 1069-1230]
```

**全局调用次数：288 次**（每个 QK+Softmax KV 迭代 1 次）

#### 内部参数计算（1 次/调用）

| 参数 | 计算 | 说明 |
|------|------|------|
| subBlockNum | 2 | `KERNEL_TYPE_MIX_AIC_1_2` |
| rowSplitSubBlock | qSBlockSize/2 = 64 | qNBlockSize=1 时 |
| rowActualThisSubBlock | 64 | subBlock 0 |
| tokenNumPerHeadThisSubBlock | min(qSBlockSize, 64) = 64 | |
| maxRowNumPerLoop | 8192/columnNumRound | |
| rowNumTile | min(RoundDown(16, 8), 64) = 16 | 512 cols 时 |
| rowLoopNum | ceil(64/16) = 4 | |
| preLoad | 1 | |

#### 行循环（rowLoopIdx: 0..rowLoopNum+preLoad-1 = 0..4）

**全局行循环迭代次数：288 × 2(subBlocks) × 5 = 2880 次**

每次迭代分两个条件分支：

##### 分支 A：`rowLoopIdx < rowLoopNum` — 数据加载

**全局次数：288 × 2 × 4 = 2304 次**

| 子函数 | 代码行 | 全局次数 | 说明 |
|--------|--------|---------|------|
| WaitFlag(V_MTE2) | [line 1161] | 2304 | pingpong 搬入同步 |
| CopySGmToUb | [line 1030-1031] | 2304 | GM→UB S 数据 |
| SetFlag(MTE2_V) | [line 1032] | 2304 | pingpong 搬入完成 |

**rowLoopIdx=0 时额外执行：**

| 子函数 | 代码行 | 全局次数 | 说明 |
|--------|--------|---------|------|
| WaitFlag(MTE3_MTE2, EVENT_ID0) | [line 1150] | 288 | mask 拷入同步（仅首次） |
| CopyMaskGmToUb | [line 1151-1155] | 288 | GM→UB mask 数据 |
| SetFlag(MTE2_V, EVENT_ID2) | [line 1156] | 288 | mask 搬入完成 |
| CrossCoreWaitFlag(qkReady) | [line 1157] | 288 | 等待 Cube QK 完成 |

##### 分支 B：`rowLoopIdx ≥ preLoad` — Softmax 计算

**全局次数：288 × 2 × 4 = 2304 次**

| 子函数 | 代码行 | 全局次数 | 说明 |
|--------|--------|---------|------|
| WaitFlag(MTE2_V, EVENT_ID2) | [line 1173] | 2304 | 等待 mask 搬入 |
| UpCastMask\<half, int8> | [line 1174] | 2304 | mask int8→half |
| UpCastMask\<float, half> | [line 1175] | 2304 | mask half→fp32 |
| WaitFlag(MTE2_V, pingpong) | [line 1177] | 2304 | 等待 S 搬入 |
| ScaleS | [line 1178] | 2304 | S × scaleValue |
| ApplyMask | [line 1179-1182] | 2304 | 加 -inf mask |
| **SubCoreCompute\<true>** | [line 1214] | 2304 | ★ 核心 softmax 计算 |
| CopyPUbToGm | SubCoreCompute 内 | 2304 | P 写回 GM |

### 5.2 SubCoreCompute\<true> 详细调用

```
block_epilogue_online_softmax.hpp: SubCoreCompute()   [line 812-870]
```

**全局调用次数：2304 次**

| 子函数 | 代码行 | 全局次数 | isFirstStackTile | else |
|--------|--------|---------|-----------------|------|
| CalcLocalRowMax | [line 835] | 2304 | — | — |
| UpdateGlobalRowMax | [line 836-844] | 2304 | 120 次 (DataCopy) | 2184 次 (Max+Sub+Exp) |
| CalcExp | [line 846] | 2304 | — | — |
| DownCastP | [line 851] | 2304 | — | — |
| CalcLocalRowSum | [line 854] | 2304 | — | — |
| UpdateGlobalRowSum | [line 868-869] | 2304 | 120 次 (DataCopy) | 2184 次 (Mul+Add) |

#### CalcLocalRowMax — 行最大值 reduce

```
block_epilogue_online_softmax.hpp: CalcLocalRowMax()   [line 562-591]
```

**全局调用次数：2304 次**

| 列宽 | 调用路径 | BlockReduceMax 次数 | PipeBarrier 次数 |
|------|---------|---------------------|------------------|
| 512 (KV=512) | RowmaxSPECTILE512 | 3 | 3 |
| 256 (KV=256) | RowmaxSPECTILE256 | 3 | 3 |
| <512 (tail) | RowmaxTAILTILE | 2+ | 2+ |

对于大多数 KV tile（512 列），每次 CalcLocalRowMax 调用 **3 次 BlockReduceMax**。

#### UpdateGlobalRowMax — 全局最大值更新

```
block_epilogue_online_softmax.hpp: UpdateGlobalRowMax()   [line 593-653]
```

**全局调用次数：2304 次**

| 分支 | 条件 | 次数 | 操作 |
|------|------|------|------|
| isFirstStackTile | 每个 task 的第 1 个 KV tile | 120 | hm=lm (DataCopy), gm=hm (DataCopy) |
| else | 所有后续 KV tile | 2184 | hm=Max(lm,gm), dm=Sub(gm,hm), dm=Exp(dm), gm=DataCopy(hm) |

#### CalcExp — exp(S - hm) 计算

```
block_epilogue_online_softmax.hpp: CalcExp()   [line 656-698]
```

**全局调用次数：2304 次**

每 CalcExp 调用内的原子操作：

| 操作 | 每调用次数 | 全局次数 | 说明 |
|------|-----------|---------|------|
| Brcb(hm → hm_block) | 1 | 2304 | ★ 标量→行广播 |
| Sub(ls, ls, hm_block) | 8 (512/64) | 18432 | 逐 64-element 段减法 |
| Exp(ls) | 1 | 2304 | 向量 exp |
| PipeBarrier | 2 | 4608 | |

#### CalcLocalRowSum — 行求和 reduce

```
block_epilogue_online_softmax.hpp: CalcLocalRowSum()   [line 700-730]
```

**全局调用次数：2304 次**

结构与 CalcLocalRowMax 相同：RowsumSPECTILE512 调用 3 次 BlockReduceSum。

#### UpdateGlobalRowSum — 全局求和更新

```
block_epilogue_online_softmax.hpp: UpdateGlobalRowSum()   [line 732-776]
```

**全局调用次数：2304 次**

| 分支 | 条件 | 次数 | 操作 |
|------|------|------|------|
| isFirstStackTile | 每个 task 第 1 个 KV tile | 120 | gl=ll (DataCopy) |
| else | 所有后续 KV tile | 2184 | gl=Mul(dm,gl), gl=Add(ll,gl) |

#### DownCastP — fp32→fp16 转换

```
block_epilogue_online_softmax.hpp: DownCastP()   [line 779-799]
```

**全局调用次数：2304 次**（1 次 Cast fp32→fp16/调用）

---

## 6. EpilogueRescaleO 详细调用过程

### 6.1 operator()

```
block_epilogue_rescale_o.hpp: operator()   [line 353-457]
```

**全局调用次数：288 次**

#### 内部参数计算（1 次/调用）

| 参数 | 计算 | 说明 |
|------|------|------|
| rowNumTile | min(8192/128, 128) = 64 | embedV=128 |
| inRowSplitSubBlock | qSBlockSize/2 = 64 | qNBlockSize=1 |
| rowLoop | ceil(64/64) = 1 | 仅 1 次 rowLoop |

#### SubCoreCompute 详细调用

```
block_epilogue_rescale_o.hpp: SubCoreCompute()   [line 155-350]
```

**全局调用次数：288 × 2(subBlocks) × 1(rowLoop) = 576 次**

| 分支 | 条件 | 次数 | 操作 |
|------|------|------|------|
| isFirstStackTile=true | 每个 task 第 1 个 PV iteration | 120 × 2 = 240 | DataCopy(go, lo) |
| isFirstStackTile=false | 所有后续 PV iteration | (576-240) = 336 | Brcb(dm) + Mul(go, dm) + Add(go, lo) |
| isLastStackTile=true | 每个 task 最后 1 个 PV iteration | 120 × 2 = 240 | Brcb(gl) + Div(go, gl) + Cast(fp32→fp16) + CopyOToGm |
| isLastStackTile=false + needRowLoop | 中间 PV iterations | 少量 | DataCopy(gUpdate, go) |

**isLastStackTile=true 时的原子操作明细（240 次）：**

| 操作 | 每调用次数 | 全局次数 | 说明 |
|------|-----------|---------|------|
| Brcb(gl → gl_block) | 1 | 240 | ★ 分母行广播 |
| Div(go, gl_block) | 2 (128/64) | 480 | 逐 64-element 段除法 |
| Cast(go, fp32→fp16) | 1 | 240 | 精度转换 |
| CopyOToGm | 1 | 240 | 写回 GM |

---

## 7. Cube 侧 MatMul 调用过程

### 7.1 blockMmadQK — Q × Kᵀ

**全局调用次数：288 次**（每 QK KV 迭代 1 次）

每次调用：
- loadQGM（1 次/task，预加载 Q 到 L1）
- IterateAll（KFC 单次原子 MMAD，每 AIV worker 独立）
- K 列数据从 GM/L2 → L1 → L0A/L0B
- S 结果 L0C → FIX → GM workspace

### 7.2 blockMmadPV — P × V

**全局调用次数：288 次**（每 PV KV 迭代 1 次）

每次调用：
- P 从 GM workspace 读入 L1
- V 从 GM 读入 L1（通过 blockTable 或连续地址）
- O_tmp L0C → FIX → GM workspace

---

## 8. CrossCore 同步统计

| Flag | 设置次数 | 等待次数 | 说明 |
|------|---------|---------|------|
| qkReady | 288 | 288 | Cube→Vec：Q×K 完成 |
| softmaxReady | 288 | 288 | Vec→Cube：P 就绪 |
| pvReady | 288 | 288 | Cube→Vec：P×V 完成 |

---

## 9. 全局调用次数汇总

### 9.1 按函数层级汇总

| 函数 | 全局次数 | 说明 |
|------|---------|------|
| **入口层** | | |
| `fused_infer_attention_score()` | 1 | Kernel 入口 |
| `SplitFuse::FAInfer()` | 1 | 模板实例化 |
| `FAInferKernel::operator()` | 8 | 每 AICore 1 次 |
| **主循环层** | | |
| Task loop iterations | 120 | 8 core × 15 task |
| KV loop iterations (total) | 528 | 含 PRE_LAUNCH |
| QK+Softmax iterations | 288 | kvSIdx < kvSLoopNumTotal |
| PV+RescaleO iterations | 288 | kvSIdx ≥ PRE_LAUNCH |
| **Cube 侧** | | |
| `blockMmadQK.loadQGM` | 120 | 每 task 1 次 |
| `blockMmadQK` (MMAD) | 288 | 每 QK KV iter 1 次 |
| `blockMmadPV` (MMAD) | 288 | 每 PV KV iter 1 次 |
| **Vec 侧 — OnlineSoftmax** | | |
| `EpilogueOnlineSoftmax::operator()` (causal) | 288 | 每 Softmax KV iter |
| 行循环迭代 (load+compute) | 2880 | 288 × 2 × 5 |
| `CopySGmToUb` | 2304 | 每 load rowLoop |
| `CopyMaskGmToUb` | 576 (288×2) | 仅 rowLoopIdx=0 + 后续 |
| `UpCastMask` (int8→half) | 2304 | 每 compute rowLoop |
| `UpCastMask` (half→fp32) | 2304 | 每 compute rowLoop |
| `ScaleS` | 2304 | 每 compute rowLoop |
| `ApplyMask` | 2304 | 每 compute rowLoop (causal) |
| `SubCoreCompute\<true>` | 2304 | 每 compute rowLoop |
| `CalcLocalRowMax` | 2304 | SubCoreCompute 内 |
| `UpdateGlobalRowMax` | 2304 | SubCoreCompute 内 |
| `CalcExp` | 2304 | SubCoreCompute 内 |
| `DownCastP` | 2304 | SubCoreCompute 内 |
| `CalcLocalRowSum` | 2304 | SubCoreCompute 内 |
| `UpdateGlobalRowSum` | 2304 | SubCoreCompute 内 |
| `CopyPUbToGm` | 2304 | SubCoreCompute 内 |
| **Vec 侧 — RescaleO** | | |
| `EpilogueRescaleO::operator()` | 288 | 每 RescaleO KV iter |
| `SubCoreCompute` | 576 | 288 × 2 subBlocks |
| `Brcb(dm)` | 336 | 非 first stack tile |
| `Mul(go, dm)` | 336 | 非 first stack tile |
| `Add(go, lo)` | 336 | 非 first stack tile |
| `DataCopy(go=lo)` | 240 | first stack tile |
| `Brcb(gl)` | 240 | last stack tile |
| `Div(go, gl)` | 480 | last stack tile (2段/次) |
| `Cast(fp32→fp16)` | 240 | last stack tile |
| `CopyOToGm` | 240 | last stack tile |

### 9.2 关键原子操作汇总

| 原子操作 | 全局次数 | 说明 |
|----------|---------|------|
| `Brcb` (行广播) | 2304 (hm) + 336 (dm) + 240 (gl) = **2880** | ★ 消除 GetValue+Muls |
| `BlockReduceMax` | ~6912 (2304×3 per SPECTILE512) | 行最大值 reduce |
| `BlockReduceSum` | ~6912 | 行求和 reduce |
| `Sub(ls, hm_block)` | 18432 (2304×8 per 512cols) | S - hm 减法 |
| `Exp(ls)` | 2304 | 向量 exp |
| `Mul(go, dm)` | 336 | O rescaling 乘法 |
| `Add(go, lo)` | 336 | O 累积加法 |
| `Div(go, gl)` | 480 | ★ 最终归一化除法（仅 last tile） |
| `Cast fp32→fp16` | 2304 (P) + 240 (O) = **2544** | 精度转换 |
| `GetValue` | **0** (标准模式) | ★ 无 Scalar 参与 |

---

## 10. 调用树（完整展开，仅 FIAS 目录下的函数）

```
fused_infer_attention_score()                                        [1 call]
│
├── SplitFuse::FAInfer<half,half,float,false,MASK_CAUSAL,TND>()    [1 call]
│   │
│   └── FAInferKernel::operator()                                   [8 calls, per AICore]
│       │
│       ├── [Cube] SetFlag ×18, blockMmadQK init, blockMmadPV init  [1×/core init]
│       ├── [Vec]  SetFlag ×16, epilogueOnlineSoftmax init,
│       │            epilogueRescaleO init, epilogueInitOut init      [1×/core init]
│       │
│       ├── [Task Loop] ×15/core, ×120 total
│       │   │
│       │   ├── [Cube] blockMmadQK.loadQGM()                        [120 total]
│       │   │
│       │   └── [KV Loop] ×1~6/task, ×528 total
│       │       │
│       │       ├── [QK+Softmax branch] ×288 total
│       │       │   │
│       │       │   ├── [Cube] blockMmadQK()                         [288 total]
│       │       │   ├── [Cube] CrossCoreSetFlag(qkReady)             [288 total]
│       │       │   │
│       │       │   ├── [Vec] EpilogueOnlineSoftmax::operator()      [288 total]
│       │       │   │   (causal mask overload)
│       │       │   │   │
│       │       │   │   ├── [Row Loop ×5/subBlock, ×2 subBlocks]
│       │       │   │   │   × 2880 iterations total
│       │       │   │   │   │
│       │       │   │   │   ├── [Load ×4] CopySGmToUb               [2304 total]
│       │       │   │   │   ├── [Load ×4] CopyMaskGmToUb            [~576 total]
│       │       │   │   │   │
│       │       │   │   │   ├── [Compute ×4] UpCastMask<int8→half>  [2304 total]
│       │       │   │   │   ├── [Compute ×4] UpCastMask<half→fp32>  [2304 total]
│       │       │   │   │   ├── [Compute ×4] ScaleS                 [2304 total]
│       │       │   │   │   ├── [Compute ×4] ApplyMask              [2304 total]
│       │       │   │   │   │
│       │       │   │   │   ├── [Compute ×4] SubCoreCompute<true>   [2304 total]
│       │       │   │   │   │   │
│       │       │   │   │   │   ├── CalcLocalRowMax                  [2304 total]
│       │       │   │   │   │   │   └── RowmaxSPECTILE512/256/TAIL  [2304 total]
│       │       │   │   │   │   │       └── BlockReduceMax ×3       [~6912 total]
│       │       │   │   │   │   │
│       │       │   │   │   │   ├── UpdateGlobalRowMax               [2304 total]
│       │       │   │   │   │   │   ├── [first=120] DataCopy(hm=lm, gm=hm)
│       │       │   │   │   │   │   ├── [else=2184] Max(hm=Max(lm,gm))
│       │       │   │   │   │   │   ├── [else=2184] Sub(dm=gm-hm)
│       │       │   │   │   │   │   ├── [else=2184] Exp(dm)
│       │       │   │   │   │   │   └── DataCopy(gm=hm)
│       │       │   │   │   │   │
│       │       │   │   │   │   ├── CalcExp                           [2304 total]
│       │       │   │   │   │   │   ├── Brcb(hm→hm_block)           [2304 total]
│       │       │   │   │   │   │   ├── Sub×8 (ls=ls-hm_block)      [18432 total]
│       │       │   │   │   │   │   └── Exp(ls)                     [2304 total]
│       │       │   │   │   │   │
│       │       │   │   │   │   ├── DownCastP                        [2304 total]
│       │       │   │   │   │   │   └── Cast(fp32→fp16)             [2304 total]
│       │       │   │   │   │   │
│       │       │   │   │   │   ├── CopyPUbToGm                      [2304 total]
│       │       │   │   │   │   │   └── DataCopy(P→GM)              [2304 total]
│       │       │   │   │   │   │
│       │       │   │   │   │   ├── CalcLocalRowSum                  [2304 total]
│       │       │   │   │   │   │   └── RowsumSPECTILE512/256/TAIL  [2304 total]
│       │       │   │   │   │   │       └── BlockReduceSum ×3       [~6912 total]
│       │       │   │   │   │   │
│       │       │   │   │   │   └── UpdateGlobalRowSum               [2304 total]
│       │       │   │   │   │       ├── [first=120] DataCopy(gl=ll)
│       │       │   │   │   │       └── [else=2184] Mul(dm,gl)+Add(ll,gl)
│       │       │   │   │
│       │       │   │   └── CrossCoreSetFlag(softmaxReady)           [288 total]
│       │       │
│       │       ├── [PV+RescaleO branch] ×288 total
│       │       │   │
│       │       │   ├── [Cube] blockMmadPV()                         [288 total]
│       │       │   ├── [Cube] CrossCoreSetFlag(pvReady)             [288 total]
│       │       │   │
│       │       │   ├── [Vec] CrossCoreWaitFlag(pvReady)             [288 total]
│       │       │   ├── [Vec] EpilogueRescaleO::operator()           [288 total]
│       │       │   │   │
│       │       │   │   └── SubCoreCompute ×2/subBlock              [576 total]
│       │       │   │       │
│       │       │   │       ├── [isFirst=true, ×240]
│       │       │   │       │   └── DataCopy(go=lo)                  [240 total]
│       │       │   │       │
│       │       │   │       ├── [isFirst=false, ×336]
│       │       │   │       │   ├── Brcb(dm→dm_block)                [336 total]
│       │       │   │       │   ├── Mul(go=go*dm_block)              [336 total]
│       │       │   │       │   └── Add(go=lo+go)                    [336 total]
│       │       │   │       │
│       │       │   │       ├── [isLast=true, ×240]  ← 仅最后一个 KV tile
│       │       │   │       │   ├── Brcb(gl→gl_block)                [240 total]
│       │       │   │       │   ├── Div(go=go/gl_block) ×2段        [480 total]
│       │       │   │       │   ├── Cast(go fp32→fp16)               [240 total]
│       │       │   │       │   └── CopyOToGm                        [240 total]
│       │       │   │       │
│       │       │   │       └── [isLast=false+needRowLoop, 少量]
│       │       │   │           └── DataCopy(gUpdate=go)              [少量]
│
│       ├── [Cube Wait+Clear] ×18 flags                              [1×/core cleanup]
│       ├── [Vec  Wait+Clear] ×16 flags                              [1×/core cleanup]
│       └── PipeBarrier<PIPE_ALL>                                    [1×/core]
```

---

## 11. 分支执行频率汇总

| 分支条件 | 全局执行次数 | 占比 | 说明 |
|----------|------------|------|------|
| `TILING_KEY ≥ FAI_FLAG` | 1 | 100% | 本 shape 走 FAInfer |
| `MASK_CAUSAL` | 288 | 100% | causal mask 模式 |
| `doTriUMask=true` | ~288 | ~100% | 几乎所有 KV tile 都需要 causal mask |
| `isFirstStackTile=true` (Softmax) | 120 | 5.2% | 每个 task 第 1 个 KV tile |
| `isFirstStackTile=false` (Softmax) | 2184 | 94.8% | 所有后续 KV tile |
| `isFirstStackTile=true` (RescaleO) | 240 | 41.7% | |
| `isFirstStackTile=false` (RescaleO) | 336 | 58.3% | |
| `isLastStackTile=true` (RescaleO) | 240 | 41.7% | 每个 task 最后 1 个 PV iteration |
| `RowmaxSPECTILE512` | ~较多 | 主导 | 512 列 KV tile |
| `RowmaxSPECTILE256` | ~较少 | 边缘 | 256 列 KV tile |
| `RowmaxTAILTILE` | 少量 | 少 | 最后 tile 尾部 |
| `PAGED_CACHE_FLAG=false` | 全部 | 100% | 无 paged cache |
| `SinkMode::DISABLE` | 全部 | 100% | 无 learnable sink |
| `LseMode::NONE` | 全部 | 100% | 无 LSE 输出 |
| `GetValue` | **0** | **0%** | ★ 标准 Scalar 不参与 |

---

## 12. 与 TQ Pack 对比要点

| 维度 | FIAS (193us) | TQ Pack (900us) |
|------|-------------|----------------|
| **总 task 数** | 120 (8 core) | ~451 batches (2 AIV subBlocks) |
| **Cube MatMul 总次数** | 288 (QK) + 288 (PV) = 576 | ~451 (rotate) |
| **Cube MatMul 形状** | QK: [128,512,128], PV: [128,128,512] | [4~64, 128, 128] |
| **Vec 操作总步数** | ~42000 (softmax+rescale) | ~201000 (normalize+encode+pack) |
| **GetValue 总次数** | **0** | **28832** |
| **Brcb 总次数** | **2880** | **0** |
| **Cube/Vec 重叠** | 3 级流水线 (PRE_LAUNCH=2) | 无重叠 |
