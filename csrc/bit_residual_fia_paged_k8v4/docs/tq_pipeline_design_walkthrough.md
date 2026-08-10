# 从零设计 TQ 流水：原则 + 全程举例

> **视角**：把 BitResidual / TurboQuant FIA（TQ P0）当成一个**尚未实现的新特性**，按设计原则从头走一遍。  
> **目的**：把「依赖定拍、寿命定槽、两类同步」落到可照抄的步骤与表格上。  
> **实现落点**（对照用，设计时先别管文件名）：  
> - L0 调度：`fia_kernel_turboquant_p0.h::ExecuteTaskTq`  
> - Vec / Dequant：`fia_block_vec_turboquant_p0.h`  
> - Cube：`fia_block_cube_turboquant_p0.h`  
> 细节与优化史见：`tq_pipeline_and_p1a.md`

---

## 0. 场景与目标（设计 Day 0）

**要做什么**：Paged Attention 的一段 S2（例如 `s2Base=1024`）上，对量化 KV 做：

1. 解量化 K → 写 workspace  
2. Cube：`Q @ K` → 分数  
3. Softmax → P  
4. 解量化 V → 写 workspace（可与 Softmax 有重叠空间）  
5. Cube：`P @ V` → 累加结果  
6. Vec 归一化 / 收尾（Vec2）

**硬件约束（910B 类）**：

| 资源 | 角色 |
|------|------|
| **AIV**（Vector） | Dequant、Softmax、Vec2；内含 MTE2 / VEC / MTE3 三 pipe |
| **AIC**（Cube） | MM1、MM2 |
| CrossCore flag | AIV ↔ AIC 握手；不能当随意共享变量乱用 |

**性能目标**：稳态墙钟尽量接近 `max(AIV 关键路径, AIC 关键路径)`，而不是两者相加。  
若一步一步串完：`T_serial = T_deqK + T_mm1 + T_deqV + T_sm + T_mm2 + T_vec2` → 太慢。  
要靠 **多 task 重叠** + **同核 pipe 重叠** 把串行和压下去。

---

## 1. 原则回顾（后面每步都会引用）

| # | 原则 | 一句话 |
|---|------|--------|
| P1 | **依赖定拍** | 先画真实 DAG，再决定一拍里塞什么、跨几拍 |
| P2 | **寿命定槽** | 物理 buffer 份数 ≥ 写→最后读的跨拍数；≠ task 缓存个数 |
| P3 | **两类同步** | 「选哪槽就绪」用 `flag+slot`；「纯因果完成」用固定 flag（需证明 Wait≺Set） |
| P4 | **无依赖往前甩** | 同拍内：能提前的 DMA/计算放在跨核 Wait 之前 |
| P5 | **profile 闭环** | 先保证正确，再用墙钟是否接近 `max(资源)` 指导下一刀 |

下面按 **P1 → P2 → P3 → P4 → P5** 把 TQ 设计出来。

---

## 2. Step A：画依赖图（P1）

### 2.1 列出原子步（不要先起阶段名）

| 代号 | 内容 | 跑在 |
|------|------|------|
| DK | Dequant K → `K-WS` | AIV |
| MM1 | `Q @ K` → `mm1Res`（分数） | AIC |
| DV | Dequant V → `V-WS` | AIV |
| SM | Softmax(`mm1Res`) → `P` | AIV |
| MM2 | `P @ V` → `mm2Res` | AIC |
| V2 | Vec2 / 归一化读 `mm2Res` | AIV |

### 2.2 只画真实数据边

```text
DK ──▶ MM1 ──▶ SM ──▶ MM2 ──▶ V2
              ▲         ▲
DV ───────────┴─────────┘
```

说明：

- **MM1 只依赖 DK**（要读 K-WS），不依赖 DV。  
- **SM 只依赖 MM1**（要读分数），不依赖 DV。  
- **MM2 依赖 SM 和 DV**（要 P 和 V-WS）。  
- **V2 依赖 MM2**。

### 2.3 找关键路径 vs 可旁路边

```text
关键路径（必须串）：  DK → MM1 → SM → MM2 → V2
旁路（可提前）：    DV 可与「等 MM1 / 做 Softmax」重叠，只要在 MM2 前写完 V-WS
```

**设计决策 1（TQ 已采用）**：不要把 DV 拆成单独 L0 阶段；把它挂在「做 Softmax 的那一拍」的**开头**，且放在 `Wait(MM1就绪)` **之前**：

```text
拍内顺序（同一 task 的「Vec1 拍」）:
  DV → Wait(MM1) → SM → Set(P就绪)
```

这样 DV 的时间可以盖住「等 Cube」的气泡。若错误写成 `Wait(MM1) → DV → SM`，DV 就被塞进关键路径，白白拉长墙钟。

### 2.4 错误示范（假依赖）

```text
❌ 把「阶段名」当成依赖：
   DequantK → DequantV → MM1 → Softmax → MM2
   （强迫 DV 在 MM1 前 → AIC 空等）

❌ 把 Softmax 和 DequantV 当成两个 L0 墙钟拍：
   拍1 只 DV，拍2 只 Softmax → 增加 prologue，稳态无收益
```

---

## 3. Step B：定墙钟拍表（P1：依赖定拍）

目标：一个 task 的寿命拆成若干**墙钟拍**，使得 **同一墙钟上 AIV 与 AIC 尽量都有活干**，且干的是**不同 age 的 task**。

### 3.1 先按资源切「拍」

把关键路径按「必须跨核交班」切开：

| 拍内组合 | AIV | AIC | 为何捆在一起 |
|----------|-----|-----|----------------|
| 拍 α | DK | MM1 | MM1 紧跟 DK；同 task 同拍握手 |
| 拍 β | DV + SM | MM2 | MM2 要 P 与 V；DV+SM 同拍完成 |
| 拍 γ | V2 | （可选 OutputPi） | 收尾 |

→ **一个 task 活 3 个墙钟拍** → 调度上要同时记住 age0 / age1 / age2  
→ `FIA_PRELOAD_TASK_CACHE_SIZE = 3`（这是 **RunInfo 缓存**，还不是物理 WS 份数）。

### 3.2 画出稳态墙钟表（TQ 的 `ExecuteTaskTq`）

墙钟 `N` 上同时推进三个 task：

| 墙钟 N | age | task | AIV | AIC |
|--------|-----|------|-----|-----|
| N | 0 | **N** | DequantK(N) | MM1(N) |
| N | 1 | **N−1** | ComputeVec1(N−1)=DV→Wait→SM | MM2(N−1) |
| N | 2 | **N−2** | ComputeVec2(N−2) | OutputPi(可选) |

代码对应关系：

```cpp
// extraInfo0 = loop % 3     → task N
// extraInfo2 = (loop+2)%3   → task N-1   （因为 (N+2)%3 == (N-1)%3）
// extraInfo1 = (loop+1)%3   → task N-2
```

单个 task **T** 的一生：

```text
墙钟 T:     AIV: DK(T)           | AIC: MM1(T)
墙钟 T+1:   AIV: DV+SM(T)        | AIC: MM2(T)
墙钟 T+2:   AIV: V2(T)           | AIC: …
然后 isValid=false
```

### 3.3 用具体数字走 4 个墙钟（task 0,1,2,3）

```text
墙钟 0:
  AIV: DK(0)     AIC: MM1(0)
  （age1/age2 尚无 valid task）

墙钟 1:
  AIV: DK(1)     AIC: MM1(1)
  AIV: DV+SM(0)  AIC: MM2(0)      ← task0 进入拍 β

墙钟 2:  ← 进入稳态
  AIV: DK(2)     AIC: MM1(2)
  AIV: DV+SM(1)  AIC: MM2(1)
  AIV: V2(0)     AIC: …           ← task0 收尾

墙钟 3:
  AIV: DK(3)     AIC: MM1(3)
  AIV: DV+SM(2)  AIC: MM2(2)
  AIV: V2(1)     …
```

稳态下每个墙钟：AIV 做「新 K + 旧 Softmax/V + 更旧 Vec2」，AIC 做「新 MM1 + 旧 MM2」。  
串行六段变成约 **3 拍重叠**，墙钟 ≈ `max(AIV三段之和的均摊, AIC两段之和的均摊)`（再加同步缝）。

### 3.4 拍数选择的检查题

问自己：

1. 若压成 **2 拍**：能否在一拍里塞下 `DK+MM1` 和 `DV+SM+MM2`？  
   → AIV 上 `DK` 与 `DV+SM` 同 task 无法真正并行（单 AIV 线程顺序执行），同拍塞太多只会拉长单拍，**重叠度反而变差**。  
2. 若扩成 **4 拍**：多一拍 prologue/epilogue，稳态仍被最长资源限制，通常 **无收益**。  

TQ 结论：**3 拍是关键路径跨核交班次数决定的最小合理深度**。

---

## 4. Step C：定 double buffer（P2：寿命定槽）

### 4.1 先列「会跨拍存活」的中间结果

| Buffer | 谁写 | 谁读 | 写在哪拍 | 最后读在哪拍 |
|--------|------|------|----------|--------------|
| `K-WS[s]` | AIV DK | AIC MM1 | 拍 α（墙钟 T） | 同拍 α（墙钟 T） |
| `mm1Res[s]` | AIC MM1 | AIV SM | 拍 α（墙钟 T） | 拍 β（墙钟 T+1） |
| `V-WS[s]` | AIV DV | AIC MM2 | 拍 β（墙钟 T+1） | 同拍 β（墙钟 T+1） |
| `mm2Res[s]` | AIC MM2 | AIV V2 | 拍 β（墙钟 T+1） | 拍 γ（墙钟 T+2） |

跨度公式：

```text
跨度 = 最后读墙钟 − 写完墙钟 + 1
```

| Buffer | 跨度 | 至少几份物理槽？ |
|--------|------|------------------|
| K-WS | 1 | 2 也够（TQ 用 2） |
| mm1Res | 2 | **≥ 2** |
| V-WS | 1 | 2 |
| mm2Res | 2 | **≥ 2** |

**没有任何一种需要跨度 3** → 物理上 `PRELOAD_NUM = 2`，下标 `loop % 2`。

### 4.2 关键对照：3 个 task 缓存 ≠ 3 份 WS

```text
FIA_PRELOAD_TASK_CACHE_SIZE = 3   // 同时挂着 3 个 RunInfo（调度）
PRELOAD_NUM                 = 2   // 每种 WS / mmRes 两份（数据）
```

为什么 3 个活 task 不会三人抢同一槽？看墙钟 `T+2`（设 T 为偶数，槽 0 = 偶数 task）：

| 活着的 task | 在干 | 碰槽 0 的什么 |
|-------------|------|----------------|
| T+2（偶） | DK + MM1 | 写 **K-WS[0]**、写 **mm1[0]** |
| T+1（奇） | DV + SM + MM2 | 全在 **槽 1** |
| T（偶） | Vec2 | 读 **mm2[0]**（不是 mm1 / K-WS） |

同槽上：**写 mm1[0] 的是 T+2，读 mm2[0] 的是 T**——数组不同，不冲突。  
同数组同槽的复用间隔是 **隔一个 task（T 与 T+2）**，刚好等于双缓冲。

### 4.3 用时间线证明 mm1Res 双缓冲安全

```text
墙钟:     0         1         2         3         4
mm1[0]:   W0──R0              W2──R2              W4…
          └task0┘             └task2┘
mm1[1]:             W1──R1              W3──R3
                    └task1┘             └task3┘
```

- task0：墙钟 0 写、墙钟 1 Softmax 读完 → 跨度 2。  
- task2：墙钟 2 才再写槽 0 → 此时 task0 已不碰 mm1。  

**反证**：若 Softmax 拖到墙钟 2 还要读 mm1[0]，则与 task2 的写冲突 → 必须 `%3` 三缓冲。TQ 没有这种寿命 → `%2` 正确。

### 4.4 设计口诀（落到实现）

```text
slot = info.loop % PRELOAD_NUM;   // 数据身份跟 loop，不跟「阶段名」
写 K-WS[slot] / mm1Res[slot] / …
```

**不要**写 `slot = stageId` 或 `slot = age`——age 是调度概念，槽是数据寿命概念。

---

## 5. Step D：设计同步（P3：两类 flag）

### 5.1 把 DAG 的每条跨核边变成一条同步

| 边 | 生产者 | 消费者 | 同步类型 | TQ 实现 |
|----|--------|--------|----------|---------|
| DK → MM1 | AIV Set | AIC Wait | **槽就绪** | `TQ_VEC_DEQ_K0_READY + (loop%2)` |
| MM1 → SM | AIC Set | AIV Wait | **阶段握手** | `syncC1V1`（固定 id=7） |
| DV → MM2 | AIV Set | AIC Wait | **槽就绪** | `TQ_VEC_DEQ_V0_READY + (loop%2)` |
| SM → MM2 | AIV Set | AIC Wait | **阶段握手** | `syncV1C2`（固定 id=8） |
| MM2 → V2 | AIC Set | AIV Wait | **阶段握手** | `syncC2V2`（固定 id=9） |

### 5.2 为何 K/V ready 必须 `+ loop%2`？

含义是：「**第 s 号 WS 槽**已经写完」。

```cpp
// DequantK 末尾
CrossCoreSetFlag(..., TQ_VEC_DEQ_K0_READY_VEC + (info.loop % 2));

// ComputeMm1 开头
CrossCoreWaitFlag(TQ_VEC_DEQ_K0_READY + (info.loop % 2));
// 然后 CopyKToL1 读的就是 K-WS[info.loop % 2]
```

若只用一个 ready flag、不带槽：

- 语义上「就绪」与「读哪槽」脱节；  
- 相邻 task 的 Set/Wait 更容易在双 AIV（MODE2）下纠缠。  

TQ 选择：**槽就绪 flag 与 `loop%2` 一一对应**。

### 5.3 为何 `syncC1V1` 不需要 `+ loop`？（详细证明）

**语义**：只表示「某个 task 的 MM1 分数写完了」，消费者 Softmax 用 `info.loop%2` **自己知道读哪槽 mm1Res**，flag 不负责选槽。

**危险模式**：同一 flag id 上，新 task 的 Set 盖住旧 task 尚未 Wait 的 token。

证明 TQ **不会**落入危险模式——看墙钟 `T+1`：

```text
AIV 顺序:
  ① DequantK(T+1)          // 较长
  ② Wait(syncC1V1)         // Softmax(T) 消费「MM1(T) 在墙钟 T 的 Set」
  ③ Softmax(T) → Set(syncV1C2)
  …

AIC 顺序:
  ① Wait(DEQ_K[ (T+1)%2 ]) // 被 ① 挡住，直到 DequantK(T+1) 结束
  ② MM1(T+1) …
  ③ Set(syncC1V1)          // 给 Softmax(T+1) 用，要到墙钟 T+2 才 Wait
```

关键 happens-before：

```text
AIV 在 DequantK(T+1) 结束后立刻 Wait(旧 syncC1V1)
AIC 必须等同一 DequantK 结束后才能开 MM1(T+1)，做完才 Set(新 syncC1V1)

⇒ Wait(旧)  ≺  Set(新)
⇒ 单 flag 安全，不必 syncC1V1 + loop%2
```

**对照表**：

| Flag | 要不要绑 loop？ | 原因 |
|------|-----------------|------|
| `DEQ_K/V_READY` | **要** | 声明「哪一槽数据好了」 |
| `syncC1V1/V1C2/C2V2` | **不要** | 纯因果；流水顺序保证 Wait≺Set |

### 5.4 新特性设计时的判定流程

```text
这条跨核边是否表达「某槽数据可读」？
  ├─ 是 → flag_base + slot（slot = loop%N 或 run%N）
  └─ 否 → 固定 flag
           └─ 能否证明任意相邻两对 Set/Wait：Wait(旧)≺Set(新)？
                ├─ 能 → 单 flag
                └─ 不能 → ping-pong 两套握手 flag，或加深缓冲
```

### 5.5 同拍内 Cube 侧顺序（举例 MM2）

MM2 有两条入边（DV 与 SM），Wait 顺序要匹配生产顺序：

```text
ComputeVec1(T) 内:
  DV → Set(DEQ_V_READY[s])
  Wait(syncC1V1)
  SM → Set(syncV1C2)

ComputeMm2(T) 内:
  Wait(DEQ_V_READY[s])   // 可先等到 V（DV 在 Wait MM1 前就 Set 了）
  Wait(syncV1C2)         // 再等 P
  算 P@V → Set(syncC2V2)
```

设计时：**多入边的消费者，按「谁更早可能好」排序 Wait**，减少空等。

---

## 6. Step E：同拍 / 同核再挖重叠（P4）

L0 解决「多 task × AIC∥AIV」。L1 解决「**同一 Dequant 里** MTE2∥VEC∥MTE3」。

### 6.1 问题：Dequant 一步很长

`DequantKvImpl` 按 PA block 切成多次 **run**，每次：GM 读 codes/meta → VEC decode → 写 WS。  
若串行：`MTE2 → VEC → MTE3 → MTE2 → …`，VEC 忙时 MTE2 闲。

### 6.2 同核也要 double buffer

| UB 资源 | 槽数 | 下标 |
|---------|------|------|
| codes/out staging | 2 half | `runId % 2` |
| meta | 2 slot | `runId % 2` |

约束：**下一 run 的 prefetch 必须写另一半**，否则盖住当前 decode 正在读的 UB。

### 6.3 P1a：把 prefetch 甩到 Wait/重算之前

原则 P4 的 TQ 实例：

```text
每个 run:
  Wait(本 run 的 MTE2)     // 本数据齐
  Prefetch(下一 run MTE2)  // 只 Set，不 Wait ← 提前发
  VEC decode(本 run)       // 与上一行的 MTE2 硬件并行
  MTE3 store(本 run)
```

这就是「无依赖往前甩」：下一 run 的 DMA **不依赖**本 run 的 decode 结果，不应放在 decode 之后的缝里。

### 6.4 和 L0 的类比（帮助记忆）

| 层级 | 「loop」 | double buffer | 提前发的是什么 |
|------|----------|---------------|----------------|
| L0 | task `loop` | WS / mmRes `%2` | 下一 age 的 DequantV（在 Wait MM1 前） |
| L1 | PA `runId` | UB half `%2` | 下一 run 的 codes/meta MTE2（在 VEC 前） |

同一套原则，两套尺度。

---

## 7. Step F：prologue / epilogue 与正确性（常被忽略）

3 拍流水前两拍、后两拍有空槽：

```text
墙钟 0: 只有 age0
墙钟 1: age0 + age1
墙钟 2…: 满流水
最后:   不再 CreateTask，但仍 ExecuteTaskTq 直到三个 age 都清空
```

实现要点：

- 每拍用 `if (extraInfoX.isValid)` 包住；空槽不 Set/Wait 脏 flag。  
- age2 结束时 `extraInfo1.isValid = false`。  
- Dual-AIV（MODE2）：两半都要 Set 槽就绪，Cube 才 Wait——设计 flag 时按「两边都完成」建模。

---

## 8. Step G：用 profile 验收（P5）

设计完成后，用一张「期望 vs 实测」表验收：

| 检查项 | 期望 | TQ 例 |
|--------|------|-------|
| 正确性 | 与非流水 / golden 一致 | e2e + 单测 |
| 墙钟形态 | ≈ `max(AIV, AIC)` 而非相加 | L6 decode profile |
| 气泡位置 | 跨核缝 vs 同核 MTE2 空窗 | 决定下一刀是 L0 还是 L1 |
| 改动归因 | 一刀一改、可回退 | P1a −11%、P2b −3% |

**不要**在未画 DAG / 未证寿命前同时改拍数、槽数、flag 语义——出问题无法归因。

---

## 9. 全程一张总图（TQ 终态）

```text
═══════════════════════════════════════════════════════════════════
 L0  task 流水（3 拍寿命，2 槽数据）
═══════════════════════════════════════════════════════════════════

  task T 依赖:
    DK ─▶ MM1 ─▶ SM ─▶ MM2 ─▶ V2
           ▲            ▲
           └── DV ──────┘  （挂在拍 β 开头，Wait MM1 之前）

  墙钟 N:
    AIV: DK(N)  |  DV+SM(N-1)  |  V2(N-2)
    AIC: MM1(N) |  MM2(N-1)    |  …

  数据槽:  K-WS/V-WS/mm1/mm2  →  loop%2
  同步:
    DEQ_K/V_READY + loop%2     （槽就绪）
    syncC1V1 / V1C2 / C2V2     （阶段握手，单 flag）

═══════════════════════════════════════════════════════════════════
 L1  PA-run 流水（同核 MTE2∥VEC）
═══════════════════════════════════════════════════════════════════

  run r: Wait codes_r → Prefetch codes_{r+1} → Decode_r → Store_r
  UB: half / meta  →  runId%2
```

---

## 10. 新特性设计检查清单（可打印）

复制下面清单，把「TQ」换成你的特性名逐项填：

- [ ] **A1** 原子步表（名称 / 资源 / 输入输出 buffer）  
- [ ] **A2** DAG（只含真实数据边）；标出关键路径与可旁路边  
- [ ] **A3** 可旁路边是否挂在 Wait 之前？（TQ：DV 在 `Wait(syncC1V1)` 前）  
- [ ] **B1** 墙钟拍表（每拍 AIV/AIC 各跑哪个 age 的哪一步）  
- [ ] **B2** task 寿命拍数；RunInfo 缓存大小  
- [ ] **B3** 用 task0..3 手推 4 个墙钟，确认稳态重叠  
- [ ] **C1** 每个跨拍 buffer 的写拍 / 最后读拍 / 跨度  
- [ ] **C2** `PRELOAD_NUM = max(跨度, 2)`（或按跨度上取）；下标公式  
- [ ] **C3** 证明「同槽复用的下一 task」开始写时，旧读者已离开  
- [ ] **D1** 每条跨核边 → flag；标注槽就绪 or 阶段握手  
- [ ] **D2** 阶段握手：写出 Wait(旧)≺Set(新) 的证明，或改为 ping-pong  
- [ ] **D3** 多入边消费者的 Wait 顺序  
- [ ] **E1** 最长同核步是否还有 MTE2∥VEC 空间；UB 是否 double buffer  
- [ ] **F1** prologue/epilogue / `isValid` / MODE2 双边 Set  
- [ ] **G1** profile：墙钟是否接近 `max(AIV,AIC)`；气泡在哪一层  

---

## 11. 和 `tq_pipeline_and_p1a.md` 的分工

| 文档 | 角色 |
|------|------|
| **本文** | 从零设计教程：原则 → TQ 逐步举例 → 检查清单 |
| `tq_pipeline_and_p1a.md` | 现状说明：L0/L1/L2 细节、P1a/P2b 改动与性能数字 |

读法建议：先跟本文走完设计过程，再回 `tq_pipeline_and_p1a.md` 对代码与优化史。
