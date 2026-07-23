# P17a — BrApplyRowAffine headDim=128 unroll

## Setup

- Workload: decode L6 `msprof --source`, `kv=2000`, `device=0`, `OP_DEBUG=ccec_g`
- Baseline (pre-P17a): `OPPROF_20260723110415_HMHQCNQTURHVHSFT`
- P17a: `OPPROF_20260723112436_HGDRLMAGPWNHJHAZ`
- Correctness: `tests/ut/ops/test_bit_residual_fia_dequant.py` **9 passed**; FIA Prefill/FD smoke **all passed**

## Change

Specialize `BrApplyRowAffine` when `headDim == BR_HEAD_SIZE` (128):

- Emit two fixed 64-wide `Mul` and two `Add` (no `columnLoops` for/if)
- Keep the generic loop for non-128 `headDim` (untaken on this op; source shows **0** instr on that path)

File: `csrc/bit_residual_fia_paged_k8v4/op_kernel/vendored/arch32/br_dequant_device.h`

## Primary KPI (wall)

| Metric | Baseline | P17a | Δ |
|--------|----------|------|---|
| **Task Duration (µs)** | **767.735** | **723.774** | **-43.961 (−5.73%)** |
| AIV avg time (µs) | 762.180 | 718.499 | −43.681 |
| Pipe bound | AIV | AIV | — |

Single-shot L6 with `ccec_g`. Treat **~44 µs** as directional until a second sample or formal (non-debug) build confirms.

## Pipe / Arithmetic

| Metric | Baseline | P17a | Δ |
|--------|----------|------|---|
| Vec FOPS (avg) | 12,465,117 | 12,041,590 | −423,526 (−3.4%) |
| Vec active ratio | 0.5318 | 0.5238 | −0.008 |
| FP32 ratio | 0.2542 | 0.2654 | +0.011 |
| Misc ratio | 0.1458 | 0.1207 | **−0.025** |

Misc ratio drop is consistent with removing column-loop control / scalar-ish busywork. FOPS should be nearly unchanged for a pure unroll; the −3.4% is small and may be run noise or codegen difference under `ccec_g`.

## Function-level (source Instructions → Est(us))

Baseline numbers from extract **before** the edit (line map matched source). P17a from extract after rebuild.

| Function | Baseline Profile% | Baseline Est(us) | P17a Profile% | P17a Est(us) | Δ Est(us) |
|----------|-------------------|------------------|---------------|--------------|-----------|
| `BrApplyRowAffine` | 4.68% | 35.91 | **1.99%** | **14.41** | **−21.50** |
| `BrDecodeKeyTile` | 3.87% | 29.75 | 3.12% | 22.59 | −7.16 |
| `BrDecodeValueTile` | 3.08% | 23.66 | 2.20% | 15.90 | −7.76 |
| **Trio sum** | **11.63%** | **89.32** | **7.31%** | **52.90** | **−36.42** |

### Affine hot lines (P17a)

| Line | Instr | Op |
|------|------:|----|
| 149 | 992,640 | `Mul` col0 |
| 152 | 595,584 | `Mul` col64 |
| 153 | 563,584 | `Brcb(offsets)` |
| 156 / 160 | 198,528 | `Add` col0 / col64 |
| 144 / 151 / 158 | ~66,176 | `PipeBarrier` |

Generic (non-128) loop body: **0** instructions executed.

## Reading

1. **Wall improved ~44 µs (−5.7%)** on this decode L6 shot; AIV time moves in lockstep.
2. **Affine source Est dropped ~21 µs**, in line with killing the column for/if OTHER path that previously dominated Affine attribution.
3. Key/Value Est also fell (~7–8 µs each): partly less inlined Affine CALL cost, not only their own ALU.
4. **Do not over-read absolute source instr** across the edit: function line ranges shifted; compare Est(us) / Profile% and wall KPI.
5. **Confirm** with a second `--source` run or a formal (no `ccec_g`) msprof before declaring the win locked.

## Next

- Optional: 2nd L6 sample / formal build confirm
- P17b: Value `vmin += 8*vstep` on meta, drop N-element `Adds(+8)`
