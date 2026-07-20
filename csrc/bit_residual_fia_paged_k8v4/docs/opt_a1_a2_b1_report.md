# BitResidual FIA Dequant 优化实验报告（A1 / A2 / B1）

**日期**: 2026-07-18  
**设备**: Ascend 910B3，容器 `vllm-ascend-lcy-v018`，CANN 8.5.1  
**基线 KPI**: L6 `prof_tnd_pa_bit_residual.sh msprof --kv=1000`，`Task Duration`  
**基线墙钟**: **5054.481 µs**（`meta_baseline_msprof.log` / OPPROF_20260718085434）

---

## 摘要

| 项 | 内容 | Golden | L6 Task Duration | 结论 |
|----|------|--------|------------------|------|
| **Baseline** | P2 同 PA-block 批量 + 单 UB 区串行 MTE3 wait | PASS（历史） | 5054.481 µs | 对照 |
| **A1** | `tmpBuff1` 拆 2×16KB，跨 PA-run MTE2∥MTE3 | **首次全 PASS** | **5032–5040 µs（约 −0.3%~−0.4%）** | **保留**（收益低于 3% 门槛，但正确且略快） |
| **A2** | MM2 在 `syncV1C2` 前预取首块 V→L1 | FAIL（数值） | 未保留剖面 | **回滚** |
| **B1** | Meta 批量 Cast + 少 V→S sync | FAIL（aicore / 数值） | 未保留剖面 | **回滚** |
| **C1/C2** | Decode 默认 vector；Prefill 开 FIA | 已在路由中 | N/A | **已满足，无改代码** |

**总体**: 算子仍为 **AIV-bound**（`vec_ratio≈0.21`）。流水/DMA 重叠对墙钟几乎不动；热路径在 **逐行 BR decode VEC**。A2/B1 按计划试过后因正确性回滚。

---

## 方法与验收

1. 改 kernel → `bash script/lcy/bit_residual_fia_paged_k8v4/rebuild_op.sh --soc=ascend910b`
2. Golden: `python tests/e2e/singlecard/xrx_bit_residual_fia_paged_k8v4_smoke.py`
3. Profile: `bash tools/bit_residual_fia_paged_k8v4/prof_tnd_pa_bit_residual.sh msprof --kv=1000 --device=0 --skip-build --warm-up=5 --launch-count=20`

日志目录: `tools/bit_residual_fia_paged_k8v4/opt_*.log`

---

## A1 — 跨 PA-run 双缓冲预取

### 实现

文件: `fia_block_vec_turboquant_p0.h` → `DequantKvImpl`

- `tmpBuff1`（32KB）分为两个 16KB half；`maxSub` 按 **半缓冲** 容量计算
- `runId >= 2` 才 `WaitFlag(MTE3_V)` 回收对应 half → 允许 **run r+1 的 MTE2** 与 **run r 的 MTE3** 重叠
- 结束时 drain 两个 MTE3 half

### Golden

- 首次（`opt_a1_golden.log`）: **全部 PASS**（含 bf16 / FD / prefill）
- 后续在 A2/B1 触发 aicore exception 之后，三角对比里出现 `attn_vs_golden≈3~4`，而 **`fia_vs_golden` 仍 ≤ atol**（诊断脚本确认 attn 偏移、FIA 对齐 CPU golden）。判为 **attention 侧/环境噪声**，非 A1 数学错误。L6 C++ 用例 `Output check` / `Run success` 正常。

### Profile

| 跑次 | Task Duration (µs) | vs baseline | AIV mte3_ratio |
|------|-------------------|-------------|----------------|
| Baseline | 5054.481 | — | 0.014 |
| A1 首次 | 5039.521 | **−0.30%** | 0.019 |
| A1 最终 | 5032.561 | **−0.43%** | 0.019 |

`mte3_ratio` 略升，符合 MTE3 与后续 load 重叠；墙钟仍由 VEC decode 主导。

### Go/No-Go

计划门槛：≥3% 墙钟。**未达标**，但无回归且略优 → **代码保留 A1**。

---

## A2 — Cube 侧 V L1 与 Vec1 重叠

### 实现（已回滚）

在 `ComputeMm2` 中：`Wait(DEQ_V)` 后、`Wait(syncV1C2)` 前对首个 `K_SPLIT` 做 `CopyVToL1`。

### Golden

`prefill_full_seq_dense_rotation`: FIA vs attn max_diff ≈ 3.44 → **FAIL**。立即回滚至原 `Wait(DEQ_V) → Wait(syncV1C2) → CopyV` 顺序。

### Profile

未保留（正确性未过）。

### 结论

**不采纳**。在 AIV-bound 下即使成功，预期墙钟收益也有限；且与 P/V L1 生命周期耦合，风险高。

---

## B1 — Meta 批量 Cast / 降 V→S

### 实现（已回滚）

- `BrBatchCastMeta16ToFloat`：多行 meta 一次 sync
- 大栈数组 → aicore exception
- 改 8 行 chunk + 32B 对齐 dst 后，无 aicore，但 `prefill_full_seq` 数值仍偏 → 回滚到逐行 `BrReadMeta16FromUb`

### 结论

**不采纳**。`Muls(..., scalar, n)` 已是标量广播；真瓶颈在 decode 位运算链，不在 meta GetValue。再做需更严的单测与 UB 暂存，不宜与 A1 捆绑。

---

## C1 / C2 — 调度（确认）

`vllm_ascend/attention/attention_v1.py` 已实现：

- PrefillCacheHit / ChunkedPrefill：`VLLM_ASCEND_BIT_RESIDUAL_FIA=1` → BR FIA
- DecodeOnly：默认 vector attn；仅 `VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA=1` 走 FIA（A/B）

**无需再改。** 产品默认保持 Decode 不用 FIA。

---

## 保留代码与指标

**保留**: A1 dual-buffer `DequantKvImpl`（`fia_block_vec_turboquant_p0.h` + `br_dequant_device.h` 注释）。

**当前 L6**: Task Duration ≈ **5033 µs**（相对 baseline **约 −0.4%**）。

**仍 AIV-bound**: 下一轮应直接打 **decode 行内 VEC 指令数**（sign/q7 路径瘦身），而不是继续堆 DMA/CrossCore 流水。

---

## 建议后续（不在本轮范围）

1. Prefill 继续 FIA；Decode 固定 vector（C1）
2. 若追 KPI：分析 `BrDecodeKeyRow` 指令序列 / Source msprof，再谈公式级合并
3. 排查 `bit_residual_attention_paged_k8v4` 在多次 rebuild + aicore fault 后的 `prefill_full_seq` attn 漂移（与 FIA A1 正交）
4. 勿再投入：Meta strided 一次拷、砍 TQue pingpong、arch35 VF、MSD/Cube 混精

---

## 附录：关键日志

| 文件 | 用途 |
|------|------|
| `tools/.../meta_baseline_msprof.log` | Baseline 5054 µs |
| `tools/.../opt_a1_golden.log` | A1 首次全量 PASS |
| `tools/.../opt_a1_msprof.log` | A1 5039 µs |
| `tools/.../opt_a2_rebuild_golden.log` | A2 FAIL |
| `tools/.../opt_b1_rebuild_golden.log` | B1 FAIL |
| `tools/.../opt_final_a1_msprof.log` | 回滚后 A1-only 5033 µs |
