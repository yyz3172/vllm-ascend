# BitResidual / TurboQuant FIA 流水设计与 P1a 改动说明

> 算子：`BitResidualFiaPagedK8v4`（TQ P0 风格内核 + BR K8V4 dequant）  
> 主文件：  
> - Kernel 调度：`fia_kernel_turboquant_p0.h`  
> - Vec / Dequant：`fia_block_vec_turboquant_p0.h`  
> - Cube：`fia_block_cube_turboquant_p0.h`  
> - Decode 指令：`br_dequant_device.h`  
> 基线 shape：L6 decode，kv=2000，B=16，fp16，`s2Base=1024`  
> 相关 commit：P29 `40e76a39`（s2Base=1024）、P1a `deec5954`、P2b `1001fa1a`

---

## 1. 总览：三层流水

本算子的“流水”分三层，由外到内：

| 层级 | 名称 | 并行单元 | 作用 |
|------|------|----------|------|
| **L0** | Task / CrossCore 流水 | AIC ↔ AIV（跨核） | 3 拍 task 重叠 + `PRELOAD_NUM=2` 奇偶槽；见 §2 |
| **L1** | PA-run 流水（Dequant 内） | AIV 内 **MTE2 ∥ VEC ∥ MTE3** | 连续 PA 块的 codes/meta 加载、decode、写 WS |
| **L2** | Tile 指令流 | AIV VEC 内部 | 单次 run 内按 `hoistTile=16` 做 Cast / Brcb / Mul / Add |

**P1a 只改 L1**：把“下一 run 的 MTE2 prefetch”从 decode **之后**挪到 decode **之前**，让 MTE2 与整段 VEC 重叠。  
不改 L0 CrossCore 协议，不改 L2 公式（P2b 才动 L2 barrier）。

```
┌─────────────────────────────────────────────────────────────────────────┐
│ L0  Task 流水（ExecuteTaskTq；task 缓存=3，数据槽 PRELOAD=2）              │
│   墙钟 N 同时推进三个不同 age 的 task：                                    │
│     age0 = task N:     DequantK + MM1                                    │
│     age1 = task N-1:   ComputeVec1(=DequantV→Softmax) + MM2              │
│     age2 = task N-2:   ComputeVec2                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ L1  DequantKvImpl PA-run 流水（本文重点 + P1a）                           │
│   run r: Wait codes_r → Prefetch codes_{r+1} → Decode_r → Store WS_r    │
├─────────────────────────────────────────────────────────────────────────┤
│ L2  BrDecode*TileHalf（P2b 减 Cast 后 barrier）                          │
│   Cast → Brcb → Mul/Add                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. L0：Kernel Task 流水（跨核）

### 2.1 调度入口

`ExecuteTaskTq` 每个墙钟 `loop=N` 同时推进三拍（`FIA_PRELOAD_TASK_CACHE_SIZE=3`）：

```text
extraInfo0 = N % 3           → task N:     DequantK + MM1          （age 0，最新）
extraInfo2 = (N+2) % 3
           = (N-1) % 3       → task N-1:   ComputeVec1 + MM2       （age 1）
extraInfo1 = (N+1) % 3
           = (N-2) % 3       → task N-2:   ComputeVec2             （age 2，最老）
```

对应代码（`fia_kernel_turboquant_p0.h`）：

```cpp
// Pipeline: DequantK → MM1 (extraInfo0 = task N)
vectorService.DequantK(extraInfo0);   // AIV
matmulService.ComputeMm1(extraInfo0); // AIC

// Pipeline: Vec1 → MM2 (extraInfo2 = task N-1)
vectorService.ComputeVec1(extraInfo2);  // 内含一次 DequantV + Softmax
matmulService.ComputeMm2(extraInfo2);

// Pipeline: Vec2 (extraInfo1 = task N-2)
vectorService.ComputeVec2(extraInfo1);
matmulService.ComputeOutputPi(extraInfo1);
```

单个 task `T` 的墙钟寿命（3 拍）：

```text
墙钟 N=T:     DequantK(T) + MM1(T)
墙钟 N=T+1:   ComputeVec1(T) + MM2(T)    ← 仅此处做一次 DequantV
墙钟 N=T+2:   ComputeVec2(T)，然后 isValid=false
```

### 2.2 Dequant V 只执行一次（勿拆成两个 L0 阶段）

`ComputeVec1` 内部顺序：

```cpp
ComputeVec1(info) {
    DequantV(info);                          // ① 解 V → V-WS（与 MM1 无数据依赖）
    CrossCoreWaitFlag(syncC1V1);             // ② 等 MM1 分数就绪
    ProcessVec1SingleBuf(info);              // ③ Softmax → P
    CrossCoreSetFlag(syncV1C2);              // ④ 通知 MM2：P 就绪
}
```

| 易混说法 | 准确含义 |
|----------|----------|
| 「Vec1 + DequantV → MM2」 | 调度槽位名：这一拍对 **同一 task** 跑 `ComputeVec1` 与 Cube `MM2` |
| 「Softmax + Dequant V」两行 | **错误拆分**；二者同属 `ComputeVec1`，DequantV **在 Softmax 之前**只跑一次 |
| 单独的「Dequant V」L0 阶段 | **不存在** |

为何 DequantV 放在 `Wait(syncC1V1)` **之前**：DequantV 不依赖 MM1 分数，只依赖 paged V；先发可与「等 MM1」重叠。Softmax 才依赖 MM1。

单个 task 的真实依赖链：

```text
DequantK(T) ──▶ MM1(T) ──▶ Softmax/Vec1(T) ──▶ MM2(T) ──▶ Vec2(T)
     │                         ▲                    ▲
     │                         │                    │
     └──── DequantV(T) ────────┴── 可在 Softmax 前完成 ──┘
           （挂在 ComputeVec1 开头，整条链只出现一次）
```

### 2.3 AIV ↔ AIC 同步（CrossCore）

| Flag | 谁 Set | 谁 Wait | 含义 |
|------|--------|---------|------|
| `TQ_VEC_DEQ_K0_READY + (loop%2)` | AIV `DequantK` 末尾 | AIC `ComputeMm1` 开头 | K-WS 写完，可 `CopyKToL1` |
| `syncC1V1` | AIC `ComputeMm1` 末尾 | AIV Softmax 前 | MM1 分数就绪 |
| `TQ_VEC_DEQ_V0_READY + (loop%2)` | AIV `DequantV` 末尾 | AIC `ComputeMm2` 开头 | V-WS 写完 |
| `syncV1C2` | AIV Softmax 后 | AIC `ComputeMm2`（在 V-ready 之后） | P（softmax）就绪 |
| `syncC2V2` | AIC `ComputeMm2` 末尾 | AIV `ComputeVec2` 开头 | MM2 结果就绪 |

`+ (loop%2)`：K/V dequant ready 各两套 flag，对应 2 个 WS 奇偶槽，避免相邻 task 抢同一就绪信号。

Cube 侧顺序：

```text
ComputeMm1:
  Wait(DEQ_K_READY[loop%2]) → CopyK / Q@K → Set(syncC1V1)

ComputeMm2:
  Wait(DEQ_V_READY[loop%2]) → Wait(syncV1C2) → CopyV + P@V → Set(syncC2V2)
```

MM2 要 **两样都齐**：V-WS 与 Softmax 的 P。

同拍 `ExecuteTaskTq(N)` 示意：

```text
AIV: DequantK(task N)      |  AIC: MM1(task N)         ← DEQ_K 握手
AIV: ComputeVec1(task N-1) |  AIC: MM2(task N-1)       ← DequantV+Softmax ∥ MM2
AIV: ComputeVec2(task N-2) |  AIC: OutputPi(可选)
```

> Dequant(n) 与 Cube(n) 仍按 CrossCore **握手串行**；文档中的「Dequant∥Cube 四级 WS 流水」（P1b）尚未做。  
> 此处 `loop%2` 是 **跨 task 的 GM ping-pong**，不是 L1 Dequant 内部 PA-run 的 UB ping-pong。

### 2.4 为何 3 级流水，物理 buffer 却只有 2 个？

这是两件不同的事：

| 概念 | 常量 | 含义 |
|------|------|------|
| **流水深度 / task 槽** | `FIA_PRELOAD_TASK_CACHE_SIZE = 3` | 软件同时挂着 3 个 RunInfo（age 0/1/2） |
| **数据 double buffer** | `PRELOAD_NUM = 2` | GM 结果/WS 物理槽：`loop % 2` |

```text
┌─ 为什么要 3？ ─────────────────────────────────────┐
│ 一条 attention 链要拆成 3 个墙钟拍才能重叠：        │
│   拍0: K+MM1                                       │
│   拍1: V+Softmax+MM2                               │
│   拍2: Vec2                                        │
│ → RunInfo 缓存要 3 个槽记住 age0/1/2                 │
└────────────────────────────────────────────────────┘

┌─ 为什么 buffer 只要 2？ ───────────────────────────┐
│ 每种「数据槽」的占用时间（读写跨度）≤ 2 拍           │
│ 下一个复用同奇偶槽的是 task T+2（隔一个 task）      │
│ 它开始写时，task T 对该 buffer 的读已经结束         │
│ → 物理上 loop%2 两份即可                            │
└────────────────────────────────────────────────────┘
```

#### 2.4.1 「读写跨度 ≤ 2 拍」定义

对某一种中间结果（如只看 `mm1Res`）：

```text
跨度 = （该 task 最后一次读这个 buffer 的墙钟拍）
      − （该 task 第一次写这个 buffer 的墙钟拍）
      + 1
```

若跨度 ≤ 2，则：

1. **相邻 task** T 与 T+1 → 槽 `T%2` 与 `(T+1)%2` **不同**，永不抢同一块内存。  
2. **同槽 task** 是 T 与 T+2（隔一代）。  
3. T+2 最早在 **拍 T+2** 写回该槽。  
4. 只要 task T 对该 buffer 的「最后读」不晚于这个时刻（见下表），就不会撞车。

#### 2.4.2 分 buffer 核对

**`mm1Res[T%2]`** —— 跨度 = 2：

```text
拍 T:     task T 的 MM1     写 mm1Res[T%2]
拍 T+1:   task T 的 Softmax 读 mm1Res[T%2]   ← 最后一次读
拍 T+2:   task T 做 Vec2，不再碰 mm1Res
          task T+2 的 MM1   写 mm1Res[T%2]   ← 安全：旧读者已走
```

**`mm2Res[T%2]`** —— 跨度 = 2：

```text
拍 T+1:   task T 的 MM2  写 mm2Res[T%2]
拍 T+2:   task T 的 Vec2 读 mm2Res[T%2]   ← 最后一次读
拍 T+3:   task T+2 的 MM2 写 mm2Res[T%2]
```

拍 T+2 墙上同时有 task T（Vec2 读 mm2）与 task T+2（K+MM1 写 K-WS/mm1）——**不同数组**，不打架。

**`dequantKeyWs[T%2]` / `dequantValueWs[T%2]`** —— 跨度 = 1（更松）：同拍内 Dequant 写、Cube 读完；隔一代再写。

#### 2.4.3 反证：若跨度是 3

若某 buffer 需要拍 T 写、拍 T+1 读、拍 T+2 **还要读**，则拍 T+2 时旧 task 仍读 `槽 T%2`，新 task T+2 又要写 → 必须第三份物理 buffer（`%3`）。  
当前实现里 **没有任何一种 GM 中间结果跨到 3 拍**，所以 `%2` 够。

#### 2.4.4 槽0 时间表（偶数 task）

```text
墙钟:     T=0        T=1        T=2        T=3        T=4
          │          │          │          │          │
mm1[0]:   W0──R0     ·          W2──R2     ·          W4...
          └─task0─┘            └─task2─┘
          (跨2拍后空闲)         (task0已不碰mm1)

mm2[0]:   ·          W0──····R0 ·          W2──····R2
                     └─task0──┘            └─task2──┘
                     拍1写,拍2读完; task2 拍3才写 → OK

K-WS[0]:  W0/R0      ·          W2/R2      ·          W4...
          (单拍内结束)
```

#### 2.4.5 「3 个 task 同时活着」≠「同槽三人抢」

拍 T+2 示例：

| task | 在干 | 碰槽0 的什么 |
|------|------|----------------|
| T+2 | K+MM1 | 写 K-WS[0]、写 mm1[0] |
| T+1 | V+Soft+MM2 | 全在 **槽1** |
| T | Vec2 | 读 **mm2[0]**（不是 mm1/K-WS） |

三个 task 活着，但不会同时占用 **同一种 buffer 的同一槽**。

### 2.5 Dual-AIV 切分

每个 AIV subcore（`GetBlockIdx()%2`）处理当前 S2 段的一半行：

```cpp
split = s2Count / 2;
siStart = (subCoreId==0) ? 0 : split;
siEnd   = (subCoreId==0) ? split : s2Count;
```

两边都跑完整 `DequantKvImpl`，写不相交的 WS 行；`FIA_SYNC_MODE2` 要求两边都 `SetFlag` 后 Cube 才 Wait。

### 2.6 FlashDecode 与 s2Base

- Host `kS2BaseSize`：默认 **1024**（`BR_S2_BASICSIZE_IS_1024=1`）
- Cube `S2_BASICSIZE_IS_1024` 必须与 host 一致
- kv=2000 时外层 FD 段数：`ceil(2000/512)=4` → `ceil(2000/1024)=2`（P29，约 −13%）

---

## 3. L1：DequantKvImpl PA-run 流水（当前设计）

### 3.1 一次 Dequant 在干什么

对当前 task 的 `[siStart, siEnd)`：

1. 按 **PA block** 切成多次 **run**（每次最多 `maxSub=32` 行，且不跨 block）
2. 每次 run：GM 读 codes+meta → UB decode → MTE3 写回 WS
3. Dual-AIV 各自跑自己的 `while (si < siEnd)`

Run 长度（P15，O(1)，无标量 while 探测）：

```cpp
n = min(bs - pos0, siEnd - si, maxSub);  // maxSub 半路径钉 32
```

### 3.2 UB / Event 资源（理解 P1a 的前提）

```
tmpBuff1 (32KB)
┌──────────────── staging ping-pong ────────────────┬── scratch（共享）──┐
│ half0: codes+out (run 偶)  │ half1: codes+out (奇) │ halfSrc + Brcb    │
└────────────────────────────┴──────────────────────┴────────────────────┘
         ▲ bufIdx=runId%2              ▲ nextBufIdx=(runId+1)%2
         │ 当前 decode 读写             │ P1a prefetch 写入

dequantInt8Buf_（meta ping-pong，P18 起 2×slot）
┌── slot0 meta0/1 ──┬── slot1 meta0/1 ──┐
│ run 偶            │ run 奇            │
└───────────────────┴───────────────────┘
```

| Event | 用途 |
|-------|------|
| `eventIdCodesWait[2]` | MTE2→V：本 run / next run codes+meta 就绪 |
| `eventIdVWaitMte3` | V→MTE3：decode 完成才能写 WS |
| `eventIdMte3WaitV0/1` | MTE3→V：半区 store 完成才能复用该 half |
| `eventIdMte2WaitV` | V→MTE2：NaN 修复，DMA 前排空 VEC |
| `eventIdMte2WaitS` | S→MTE2：`blockTable.GetValue` 后 |

**关键约束**：跨 run prefetch 时，next 的 codes/meta **必须**落在另一半 ping-pong，否则会盖掉当前 decode 正在读的 UB。

### 3.3 单次 run 的硬件管道

AIV 上三类单元可硬件并行（同核、不同 pipe）：

| Pipe | 典型操作 |
|------|----------|
| **MTE2** | GM → UB：`DataCopy` codes、`BrCopyPackedMetaTile` |
| **VEC** | Cast / Brcb / Mul / Add（`BrDecode*TileHalf`） |
| **MTE3** | UB → GM：写 dequant WS |

若三者串行，VEC 干活时 MTE2 空闲 → AIV-bound 下浪费最大窗口。

---

## 4. P1a 改动详解

### 4.1 一句话

**把“下一 run 的 codes/meta MTE2”从「当前 decode 结束之后」挪到「当前 decode 开始之前」**，使 MTE2 与整段 VEC decode 重叠。

Commit：`deec5954`  
文件：`fia_block_vec_turboquant_p0.h::DequantKvImpl`  
实测：L6 kv=2000 fp16 **330 → 294 µs（−11%）**

### 4.2 改前：P18 流水（短窗口重叠）

P18 的 prefetch 插在 `SetFlag(V_MTE3)` 与 `WaitFlag(V_MTE3)` 之间：

```
时间 ──────────────────────────────────────────────────────────────▶

本 run:
  ┌─ Wait MTE2(本) ─┐┌════════ VEC decode ════════┐┌ Set V_MTE3 ┐┌ prefetch ┐┌ Wait V_MTE3 ┐┌ MTE3 ┐
                      │                             │             │  next     │              │ store│
                      │                             │             │  MTE2     │              │      │
                      └─────────────────────────────┘             └──短重叠───┘              └──────┘
                                                                    ▲
                                                                    │ 只藏在「VEC 已发完、等 store」的缝里
```

问题：真正最长的是 **VEC decode**；prefetch 发生时 decode 已经结束，只能与极短的 V→MTE3 排水重叠，MTE2 利用率低。

### 4.3 改后：P1a 流水（长窗口重叠）

Prefetch 整块前移到：**本 run MTE2 已 Wait 完成、VEC decode 尚未开始**：

```
时间 ──────────────────────────────────────────────────────────────▶

本 run:
  ┌─ Wait MTE2(本) ─┐┌─ Prefetch next MTE2 ─┐┌════════ VEC decode ════════┐┌ Set/Wait V_MTE3 ┐┌ MTE3 ┐
                      │  (只 Set，不 Wait)    ││     ▲                      ││                 ││store │
                      │  写 nextBuf / slot    ││     │ MTE2 与 VEC 并行      ││                 ││      │
                      └──────────────────────┘└─────┴──────────────────────┘└─────────────────┘└──────┘
                                                 ◀──────── 长重叠窗口 ────────▶

下一 run 开头:
  WaitFlag(MTE2_V, nextBuf)  ← 使用上轮 prefetch 的结果，通常已完成或只等尾巴
  Prefetch run+2 …
  Decode …
```

### 4.4 代码层：循环体顺序（当前实现）

```text
while (si < siEnd):
  ① 准备本 run 数据
       if !nextDmaInFlight:
           算 n / headBase；必要时 Wait MTE3_V(本 half)
           V_MTE2 drain → DataCopy 本 run codes+meta → Set+Wait MTE2_V
       else:
           n = prefN
           WaitFlag(MTE2_V, bufIdx)   // 等上轮 P1a prefetch
           nextDmaInFlight = false

  ②【P1a】若 si+n < siEnd：对 nextBuf 发起 codes+meta MTE2
       Wait MTE3_V(next half) if 复用
       V_MTE2 drain
       DataCopy → SetFlag(MTE2_V, nextBuf)   // 故意不 Wait
       prefN = nextN; nextDmaInFlight = true

  ③ VEC decode 本 run（BrDecode*TileHalf 循环）
       ← 此时 next 的 MTE2 可在硬件上并行

  ④ SetFlag(V_MTE3); WaitFlag(V_MTE3)
       DataCopy → WS; SetFlag(MTE3_V, bufIdx)
       si += n; runId++
```

对应源码锚点：

- Prefetch 块：`fia_block_vec_turboquant_p0.h` 约 1524–1598 行（decode **之前**）
- Decode：约 1600 行起
- Store：约 1678 行起（原先夹在 Set/Wait V_MTE3 中间的 prefetch 已删除）

### 4.5 与 P18 的差异表

| 项 | P18 | P1a |
|----|-----|-----|
| Prefetch 插入点 | `SetFlag(V_MTE3)` 之后 | `Wait` 本 run MTE2 之后、decode 之前 |
| 重叠对象 | V_MTE3 drain（短） | 整段 VEC decode（长） |
| Prefetch 内容 | 相同：nextN、headBase、codes+meta DMA | 相同 |
| Ping-pong / event | 相同 | 相同 |
| CrossCore / WS slot | 不变 | 不变 |
| 预期 / 实测 | 已有小收益 | **−11%**（330→294 µs） |

### 4.6 正确性要点（为何能提前发 DMA）

1. **UB 不冲突**：next 写 `nextBufIdx` 的 staging + meta slot；当前 decode 读 `bufIdx`。  
2. **scratch 共享但只服务当前 VEC**：`halfSrc` / `halfBroadcast` 在 decode 中使用；prefetch 的 MTE2 不写这两块。  
3. **复用 next half 前仍 Wait `MTE3_V`**：保证上一轮对该 half 的 WS store 结束。  
4. **保留 `V_MTE2` drain**：DMA 前排空 VEC，避免与 NaN 修复路径冲突。  
5. **下一 run 入口 `WaitFlag(MTE2_V)`**：decode 不会读到未完成的 codes/meta。

### 4.7 首 run / 末 run 行为

| 情况 | 行为 |
|------|------|
| **首 run**（`nextDmaInFlight=false`） | 本 run 同步 DMA+Wait；再 prefetch run1；再 decode run0 |
| **中间 run** | 开头 Wait 上轮 prefetch；再 prefetch 下一轮；再 decode |
| **末 run**（`si+n >= siEnd`） | 不发 prefetch；只 decode + store |

因此从 run0 起，除最后一次外，每次 decode 期间都有“下一 run 的 MTE2”在飞。

---

## 5. L2：Tile 指令流（及 P2b）

### 5.1 Half 路径（L6 fp16）

```text
BrDecodeKeyTileHalf / BrDecodeValueTileHalf:
  Cast(codes/nibble → halfSrc)
  BrApplyRowAffine:
      Brcb(scales → broadcast); PipeBarrier; Mul
      Brcb(offsets → broadcast); PipeBarrier; Add; PipeBarrier
```

每 run `n=32`，`hoistTile=16` → **2 个 tile**。

### 5.2 P2b（顺带记录）

删除 Cast 后、进入 `BrApplyRowAffine` 前的那道 `PipeBarrier`：

- Cast 写 `halfSrc`，Brcb 写 `broadcast`，无 RAW
- Mul 前 affine 内部仍有 barrier，会排空 Cast+Brcb

实测（在 P1a 之上）：**294 → ~286 µs（−3%）**，commit `1001fa1a`。

---

## 6. 端到端数据流（一张图）

```
Paged KV Cache (GM, int8 pack)
        │
        │ MTE2 (L1，P1a 可与 VEC 重叠)
        ▼
   UB: codes + meta (ping-pong)
        │
        │ VEC decode (L2)
        ▼
   UB: outBatch (fp16)
        │
        │ MTE3
        ▼
   Dequant WS (GM, loop%2)
        │
        │ CrossCore + Cube MTE2
        ▼
   L1 K/V → MM1 (Q@K) → Softmax → MM2 (P@V) → Out
```

---

## 7. 性能数字（便于对照）

| 配置 | Task Duration | 相对 |
|------|---------------|------|
| s2Base=512（旧） | ~381 µs | — |
| s2Base=1024（P29） | ~330 µs | −13% vs 512 |
| + P1a | **~294 µs** | **−11% vs 330** |
| + P2b | **~286 µs** | −3% vs 294 |

瓶颈仍为 **AIV**（AIV/AIC≈1.02）；P1a 提高的是 AIV 内 MTE2/VEC 重叠，不是 Cube 占比。

---

## 8. 相关文件与阅读顺序

1. 本文：流水分层 + P1a 位置  
2. `fia_kernel_turboquant_p0.h`：`ExecuteTaskTq`（L0）  
3. `fia_block_vec_turboquant_p0.h`：`DequantKvImpl`（L1，P1a）  
4. `br_dequant_device.h`：`BrDecode*TileHalf`（L2，P2b）  
5. `docs/dequant_mte2_vec_overlap_plan.md`：早期 MTE2∥VEC 方案草案（P1a 落地形态）  
6. `docs/performance_optimization_guide.md`：WS∥Cube（P1b）等未做项  

---

## 9. 小结

| 问题 | 答案 |
|------|------|
| TQ 流水有几层？ | L0 CrossCore task、L1 PA-run、L2 tile 指令 |
| Dequant V 跑几次？ | **一次**，在 `ComputeVec1` 开头（Softmax 之前） |
| 为何 3 拍流水只要 2 buffer？ | task 调度深度=3；同一种 GM buffer 读写跨度≤2 拍，奇偶槽隔代复用 |
| P1a 改哪一层？ | **只改 L1** |
| P1a 改什么？ | Prefetch **时机**：decode 后 → decode 前 |
| P1a 不改什么？ | DMA 内容、ping-pong、CrossCore、tiling、decode 公式 |
| 为何变快？ | MTE2 叠在最长的 VEC 窗口上，而不是叠在短 V_MTE3 drain 上 |
