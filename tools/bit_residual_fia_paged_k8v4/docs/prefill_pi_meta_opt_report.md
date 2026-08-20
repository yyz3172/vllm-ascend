# BitResidual FIA Prefill 优化报告（kv=2000, Q=240）

**Workload**: `BR_FIA_WORKLOAD=prefill`, B=16, Q/seq=15 → T=240, KV=2000  
**Device**: Ascend 910B, CANN 8.5.1  
**Build**: 正式包（无 `ccec_g`），同脚本 `prof_tnd_pa_bit_residual.sh msprof`  
**日期**: 2026-07-20

> 说明：此前带 `--source`/`ccec_g` 的 13390 µs 与无 debug 基线同量级；本报告统一用无 Source 构建做 A/B。

---

## 总表

| 配置 | Task Duration | vs 基线 | aiv_mte2_ratio | aiv_vec_ratio | aiv_mte3_ratio | OPPROF |
|------|---------------|---------|----------------|---------------|----------------|--------|
| **基线**（ApplyPi 开，旧 Π 加载） | **13363.8 µs** | — | 0.564 | 0.242 | 0.025 | `.../OPPROF_20260720125625_AMIGOQLCRVYIRZYZ` |
| **Opt0** 关 ApplyPi（`TQ_PI_BISECT_*=0`） | **3292.8 µs** | **−75.4%** | 0.364 | 0.318 | 0.134 | `.../OPPROF_20260720125337_UIORFYRPDWIDXOIK` |
| **Opt1** Π 行级预取 | **6595.9 µs** | **−50.6%** | 0.461 | 0.410 | 0.054 | `.../OPPROF_20260720125757_NHMIRDIFOTIVPEDM` |
| **Opt2** Opt1 + Meta `DataCopyPad` 批量 | **6836.2 µs** | −48.8% | 0.478 | 0.396 | 0.076 | `.../OPPROF_20260720125942_LHECMIOLUWHUVTYU` |

落地结论：**合入 Opt1（Π 行预取）**；**不合入 Opt2**（相对 Opt1 回退约 +240 µs）。

---

## Opt0 — 量 Π 占比（关 ApplyPi / identity 短路）

### 做法
将 `fia_turboquant_pi_bisect.h` 中 `TQ_PI_BISECT_CUBE/VEC` 置 `0`，使：
- Cube `ApplyPiToQL1`、Vec `ApplyPiTransposeToRows` 直接 return（语义等同 Π=I）
- 不改 dequant / Softmax / MM

### 结果
| | 基线 | Opt0 | Δ |
|--|------|------|---|
| Duration | 13364 µs | 3293 µs | **−10071 µs（−75%）** |
| aiv_mte2_ratio | 0.564 | 0.364 | −0.20 |
| Pipe bound | AIV (1.20×) | AIV≈AIC (1.00×) | 更均衡 |

### 解读
- Prefill 墙钟 **约 3/4 在 ApplyPi**（含 Q 侧 Cube Π 与 O 侧 Vec Πᵀ）。
- L6 harness 使用 **identity Π**，但仍走完整 ApplyPi → 测量的是「实现开销」上限；serving 若 R≠I 仍需 ApplyPi，但优化空间已被 Opt0 标定。
- 产品侧：若 rotation 确为 I，host **不传** antiquant/rotation → `piApplyEnabled_=false`，可直接吃到接近 Opt0 的收益。

### 状态
测量用；**默认 bisect 保持 1**（不提交为 0）。

---

## Opt1 — 改 Π 加载：行级预取（合入）

### 做法
`ApplyPiTransposeToRows`（`fia_block_vec_turboquant_p0.h`）：

| | 旧 | 新 |
|--|----|----|
| GM 读 | 每个 `(m, dBlock, k)` 读 **32** 元 | 每个 `(m, k)` 读整行 **128** 元 |
| 事务数 | `m × (D/32) × D` = m×512 | `m × D` = m×128（**÷4**） |
| 内层 | tile 累加后再写 out | 行预取后按 dBlock `Muls/Add` 直接累加到 `outRow` |

未改 Cube `ApplyPiToQL1`（Q 侧已是整矩阵进 L1）。

### 结果
| | 基线 | Opt1 | Δ |
|--|------|------|---|
| Duration | 13364 µs | **6596 µs** | **−6768 µs（−50.6%）** |
| aiv_mte2_ratio | 0.564 | 0.461 | −0.10 |
| aiv_vec_ratio | 0.242 | 0.410 | 计算占比上升（MTE2 被砍后相对突出） |

相对 Opt0（完全关 Π）仍高约 3.3 ms → 剩余主要是 **Cube ApplyPiToQL1 + Vec 乘加/标量 sync**，不是小 DMA。

### 解读
- 与 source 分析一致：热点在 `DataCopy(pi…)` / `memory_copy.h`；减事务数直接砍墙钟约一半。
- 下一步若继续抠 Π：Cube 做 O@Πᵀ、或减少 `GetValue(accMk)` 的 V↔S；收益预期小于本次。

### 状态
**已合入工作树**（保留）。

---

## Opt2 — Meta 批量读（试做，不合入）

### 做法
`BrCopyMetaRun`：`blockCount=n`、`blockLen=2`、`rightPadding=30`，期望一次 GM→UB 写成 n 个 32B slot，替代 n 次 2B `DataCopyPad`。

### 结果
| | Opt1 | Opt1+Opt2 | Δ vs Opt1 |
|--|------|-----------|-----------|
| Duration | 6596 µs | **6836 µs** | **+240 µs（−3.6% 变差）** |
| aiv_mte2_ratio | 0.461 | 0.478 | 略升 |

### 解读
- Prefill 在 Opt1 后，dequant meta 已不是主瓶颈（Opt0 显示去 Π 后剩余 ~3.3 ms 才是 dequant+attn）。
- 该 `DataCopyPad` 形态可能未真正合并 DMA，或 padding/stride 行为与预期不符，带来额外开销。
- **回退**到逐行 `DataCopyPad`；后续若再做 meta，建议：先 bulk `n×2` 到连续 UB，再 UB 内展开到 32B slot，并用 **skip-Π** 或 decode L6 单独量 MTE2。

### 状态
**已回退**；报告保留作负面结果。

---

## Opt3 — Vec2 / decode 公式变薄

本轮 **未改**。理由：
- Prefill 主矛盾是 ApplyPi（Opt0/Opt1 已证明）。
- Opt1 后 AIV 仍 bound，但 vec_ratio↑、mte2↓，下一刀应优先 **Cube/Vec Π 算法** 或 **skip-Π 路径**，再考虑 `BrDecodeKeyTile` 指令变薄。

---

## 建议后续

1. Serving：identity R 时跳过传 rotation（接近 Opt0）。  
2. 非 I 的 R：在 Opt1 之上评估 Cube `O @ Πᵀ`。  
3. Meta P1：换实现后在 **skip-Π** 或 decode 上单独 msprof，避免被 Π 噪声淹没。  
4. Decode 回归：`--workload=decode --kv=2000` → **3706 µs**（此前 Source 版 decode ~4719 µs；无 debug 下亦健康，ApplyPi 行预取对 decode last-S2 同样受益）。
