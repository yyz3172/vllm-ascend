# Dequant Three-Lane Timeline

> 相关实现：`csrc/bit_residual_fia_paged_k8v4/op_kernel/vendored/arch32/fia_block_vec_turboquant_p0.h`
> 中的 `DequantKvImpl`
>
> 相关背景：`br_dequant_decode_flow.md` 第 8 节（P18：跨 run MTE2<->VEC 重叠）

---

## 1. 目的

本文把 dequant steady-state 流水拆成 3 条泳道：

- `MTE2`：codes/meta 读入 UB
- `VEC`：`BrCastPackedMetaToFp32` + `BrDecode*Tile`
- `MTE3`：`outBatch` 写回 GM

重点说明为什么：

- `Spike1`（`maxSub ~= 29`）overlap 较好
- `Spike2`（`maxSub ~= 58`）虽然 DMA 次数减少，但总时延反而变差
- `Spike32`（`maxSub = 32`）是一个折中点

---

## 2. 代码里的真实时序

一个 PA run 在 `DequantKvImpl` 里可以抽象成下面 4 段：

1. `MTE2(in)`：当前 run 的 codes/meta 搬入 UB
2. `VEC`：meta cast + key/value decode
3. `MTE2(next)`：预取下一 run 的 codes/meta
4. `MTE3(out)`：当前 run 的解码结果写回 GM

对应代码骨架：

```cpp
// 1) 当前 run 的 MTE2(in)
DataCopy(batchUb, srcGm[codeOff], codesBytes);
BrCopyPackedMetaTile(...);
SetFlag(MTE2_V);
WaitFlag(MTE2_V);

// 2) 当前 run 的 VEC
BrCastPackedMetaToFp32(...);
for (...) {
    BrDecodeKeyTile(...);  // 或 BrDecodeValueTile(...)
}

// 3) 发起下一 run 的 MTE2(next)
SetFlag(V_MTE3);
if (nextSiCandidate < siEnd) {
    WaitFlag(MTE3_V);      // 仅在复用该半区时需要
    DataCopy(nextBatchUb, srcGm[nextCodeOff], nextCodesBytes);
    BrCopyPackedMetaTile(...);
    SetFlag(MTE2_V);
}

// 4) 当前 run 的 MTE3(out)
WaitFlag(V_MTE3);
DataCopy(dstWsGm[dstWsElemIdx], outBatch, n * headDimAlign);
SetFlag(MTE3_V);
```

注意这里最关键的一点：

`MTE2(next)` 不是在当前 run 的 VEC 一开始就发，而是要等当前 run 的整段 VEC 结束之后，才有机会发起。

所以：

- `maxSub` 变大时，单个 run 的 VEC 段会变长
- 下一 run 的 prefetch 最早发起时刻也会被整体后移

---

## 3. 三泳道定义

下面的图都采用同一约定：

- 横轴：时间从左到右
- 纵向 3 条泳道：`MTE2` / `VEC` / `MTE3`
- `r0`、`r1`、`r2` 表示 run0/run1/run2
- 图中长度是相对长度，不代表精确微秒

符号说明：

- `[M2 in r0]`：run0 的输入 DMA
- `[VEC r0]`：run0 的 dequant 计算
- `[M2 pf r1]`：预取 run1
- `[M3 out r0]`：run0 的输出 DMA
- `[wait reuse]`：为了复用 `tmpBuff` 半区，必须等待上一次 `MTE3` 释放

---

## 4. Spike1：`maxSub ~= 29`

`Spike1` 的特点是单次 run 较短，节拍更碎，交错更频繁。

```text
Time  --------------------------------------------------------------->

MTE2 | [M2 in r0]         [M2 pf r1]         [M2 pf r2]
VEC  |         [VEC r0]           [VEC r1]           [VEC r2]
MTE3 |                   [M3 out r0]        [M3 out r1]        [M3 out r2]
```

更细一点看 steady-state：

```text
run0: [M2 in 29] [VEC 29] [M2 pf run1] [M3 out 29]
run1:                      [VEC 29]     [M2 pf run2] [M3 out 29]
run2:                                               [VEC 29] ...
```

这里 overlap 好的原因是：

1. 单段 `VEC` 较短，不会形成很长的纯计算独占段
2. `M3 out` 也相对短，`tmpBuff` 半区释放快
3. 所以 `M2 pf run+1` 通常能比较早插进来，不会被 `reuse wait` 卡很久

结果就是 DMA 和 VEC 交错得更密，pipe busy sum / aiv_time 更高。

---

## 5. Spike2：`maxSub ~= 58`

`Spike2` 的特点是单个 run 更长，DMA 次数减少，但交错频率明显下降。

```text
Time  --------------------------------------------------------------------------->

MTE2 | [M2 in r0]                           [M2 pf r1]                           [M2 pf r2]
VEC  |         [========== VEC r0 =========]         [========== VEC r1 =========]
MTE3 |                                              [====== M3 out r0 ======]    [====== M3 out r1 ======]
```

如果把“半区复用等待”也画出来，会更接近真实行为：

```text
Time  ------------------------------------------------------------------------------->

MTE2 | [M2 in r0]                            [wait reuse][M2 pf r1]
VEC  |         [=========== VEC r0 ===========]         [=========== VEC r1 ===========]
MTE3 |                                               [====== M3 out r0 ======]
```

这里退化的关键不是“58 行本身一定慢”，而是：

1. `VEC r0` 变长了  
   `M2 pf r1` 的发起点在代码里固定是在 `VEC r0` 完成之后，所以 prefetch 最早开始时间整体后移。

2. `M3 out r0` 也变长了  
   下一次要写入的 `tmpBuff` 半区可能还没释放，于是在发 `M2 pf r1` 之前会遇到 `wait reuse`。

3. 于是 `M2 pf r1` 真正开始得更晚  
   虽然 run 数减少了，MTE2/MTE3 总次数下降了，但跨 run 的重叠窗口被压缩了。

可以把它理解成从：

```text
短 VEC + 短 DMA + 短 VEC + 短 DMA + ...
```

变成了：

```text
长 VEC + 较晚开始的 DMA + 长 VEC + ...
```

因此 profiler 上会出现：

- `aiv_mte2_time`、`aiv_mte3_time` 下降
- 但 `aiv_time` 和 `Task Duration` 上升

也就是“省掉的 DMA 时间，被更串行的 VEC bubble 吃掉了”。

---

## 6. Spike32：`maxSub = 32`

`Spike32` 是一个折中点：

- 比 `29` 稍大，run 数略少
- 又没有像 `58` 那样把单段 VEC 拉得太长
- 且 `128 / 32 = 4`，没有尾块

时序上更接近 `Spike1`，只是单段稍长一点：

```text
Time  ------------------------------------------------------------------>

MTE2 | [M2 in r0]            [M2 pf r1]            [M2 pf r2]
VEC  |         [VEC r0 32]            [VEC r1 32]            [VEC r2 32]
MTE3 |                      [M3 out r0]           [M3 out r1]           [M3 out r2]
```

因此它通常会表现为：

- overlap 明显好于 `58`
- 又可能比 `29` 少一点 run 开销

这也是为什么 `Spike32` 的 decode 结果接近 `Spike1`，而显著好于 `Spike2-58`。

---

## 7. 一句话总结

`Spike2` 退化的根因不是“每 run 行数变大”本身，而是：

- 当前实现里，`MTE2(next)` 必须等当前 run 的整段 `VEC` 做完后才能发起
- 同时还可能被 `tmpBuff` 半区复用的 `MTE3_V` 释放卡住

所以 `maxSub` 从 `29` 拉到 `58` 后，流水从“细粒度交错”变成了更偏“长 VEC 段 + 后置 DMA 段”的形态，overlap 自然下降。

---

## 8. 对应结论

- 如果目标是 decode 最优点，`32` 比 `58` 更像合理候选
- 如果目标是整体路径（含 prefill）收益，单纯依靠“拉大 `maxSub`”空间有限
- 真正决定收益的不是 run 数最少，而是 `MTE2 / VEC / MTE3` 是否还能保持足够细粒度的交错
