# P17b-B — PA run-level vmin' fold

## Setup

- Scheme A (stashed as `stash@{1}`: "P17b scheme A: per-tile vmin fold"): fold inside each `BrDecodeValueTile`
- **Scheme B (this run)**: fold once per PA run in `DequantKvImpl` Value branch; `BrDecodeValueTile` only does signed `Cast(int4)→Cast(fp32)→Affine`
- Baseline: P17a `OPPROF_20260723112436_HGDRLMAGPWNHJHAZ`
- Scheme A: `OPPROF_20260723120130_QQWACDQXQTIEWPPF`
- Scheme B: `OPPROF_20260723121404_OCAYHEQOOWYHVVZN`
- Workload: decode L6 `msprof --source`, `kv=2000`, `device=0`, `ccec_g`

## Correctness (B)

| Check | Result |
|-------|--------|
| UT dequant | **9 passed** |
| FIA Prefill/FD smoke | **all passed** |

## Wall / FOPS A/B

| Metric | P17a | P17b-A (tile) | **P17b-B (run)** |
|--------|------|---------------|------------------|
| **Task Duration (µs)** | 723.774 | 735.795 | **713.074** |
| Δ vs P17a | — | **+12.0 (+1.7%)** | **−10.7 (−1.5%)** |
| Vec FOPS (avg) | 12.04M | 11.30M | **11.21M** |
| AIV avg (µs) | 718.5 | 731.0 | **708.0** |

## Reading

1. **B beats A on wall** (~23 µs better than A, ~11 µs better than P17a) on this single-shot L6.
2. Folding once per PA run (`n≤64`) avoids repeating `Muls/Add/Barrier` on every decode tile (`hoistTile=8` → multiple tiles/run).
3. FOPS drop vs P17a is real (no N-element `Adds(+8)`); B does not pay per-tile meta fold overhead.
4. Still `ccec_g` single shot — confirm with a 2nd sample / formal build if locking the win.
5. **Recommendation**: keep **scheme B**; drop / leave A in stash.

## Code (B)

- `fia_block_vec_turboquant_p0.h` Value branch after meta Cast:
  `Muls(fp32UbA, meta1, 8); Add(meta0, meta0, fp32UbA); PipeBarrier;` then tile decode
- `br_dequant_device.h` `BrDecodeValueTile`: no `Adds(+8)`; expects pre-folded `vmin'`

## Stash

```
stash@{0}: P17b scheme A report (untracked)
stash@{1}: P17b scheme A: per-tile vmin fold
```

Restore A with `git stash pop stash@{1}` if needed (after reverting B).
