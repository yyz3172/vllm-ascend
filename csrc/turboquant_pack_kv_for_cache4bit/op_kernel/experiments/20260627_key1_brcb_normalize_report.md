# Experiment Report: key1 Brcb NormalizeBatch

Date: 2026-06-27

## Conclusion

Rejected and reverted.

The V2 implementation gave a small long-query hot-shape improvement, but failed
the strict `tools/op.profile.sh` gate:

- Route-only refreshed `op.profile.sh`: runner 557.89 us, task duration
  287.50 us.
- key1 Brcb V2 `op.profile.sh`: runner 581.54 us, task duration 293.62 us.

Because profiler task duration and runner latency both regressed, the code was
reverted. The design document and this report are kept as the experiment
record.

## Design Reference

- Design: `20260627_key1_brcb_normalize_design.md`
- Scope: `tilingKey = 1` only
- Goal: remove per-row `TqSyncVToS() + GetValue() + TqSyncSToV()` by computing
  `1 / (norm + eps)` in vector UB and broadcasting it with `Brcb`.

## Baseline

Kept route-only commit: `a63ec1ba perf(turboquant): add large pack tiling route`

- Smoke log: `mytmp/tq4bit_smoke_route_only_20260627104414.log`
- Smoke profile: `mytmp/perflog_smoke_route_only_20260627104414`
- Smoke small pack shape
  `"2,8,128;2,8,128;16;128,128;2;3"`:
  count 56, avg 46.59 us, p50 46.44 us, p90 47.86 us, max 48.72 us.
- Long log: `mytmp/tq4bit_long_route_only_20260627104641.log`
- Long profile: `mytmp/perflog_long_route_only_20260627104641`
- Long hot shape
  `"1802,8,128;1802,8,128;16;128,128;1802;2"`:
  count 28, avg 820.39 us, p50 820.72 us, p90 822.22 us, max 823.88 us.
- Refreshed route-only `op.profile.sh`:
  - OPPROF: `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`
  - runner `pack4bit_to_cache_avg_us=557.89`
  - OpBasicInfo task duration 287.50 us.

## Validation Commands

- Build: `tools/build_debug_perf.sh`
- Smoke: profile-enabled smoke run with `XRX_TQ4BIT_PROFILE=1`
- Long query: same long-query workload used by the route-only baseline
- Profile: `tools/op.profile.sh` after the debug perf build
- Source analysis:
  `python tools/extract_opprof_source_lines.py <OPPROF> --source turboquant_pack_kv_for_cache4bit.cpp --group-by function --top 60 --hot-lines 10`

## V1 Result

The first buildable implementation used two `Mul` calls, one for each 64-fp32
half of the row after `Brcb`.

- Build: pass.
- Smoke output: matched the route-only semantic baseline.
- Smoke first run small pack: avg 46.48 us, p90 48.62 us, max 51.52 us.
- Smoke rerun small pack: avg 45.62 us, p90 47.22 us, max 49.50 us.
- Long hot shape: avg 817.68 us, p50 817.72 us, p90 819.62 us, max 821.10 us.
- `op.profile.sh`:
  - OPPROF: `mytmp/OPPROF_20260627142724_SIWETSGQOPZWJSAB`
  - runner `pack4bit_to_cache_avg_us=545.92`
  - task duration 301.44 us.

V1 was not accepted because task duration regressed heavily despite acceptable
smoke and long-query results. Source profile showed increased vector pressure
inside the new normalize path.

## V2 Result

V2 replaced the two half-row `Mul` calls with one repeat-strided `Mul` using
`src1RepStride = 0`, so the 64-element `Brcb` scale block is reused for both
halves of the 128-element row.

- Build: pass.
- Smoke log: `mytmp/tq4bit_smoke_key1_brcb_normalize_v2_20260627143929.log`
- Smoke profile: `mytmp/perflog_smoke_key1_brcb_normalize_v2_20260627143929`
- Smoke small pack: avg 46.5589 us, p50 46.34 us, p90 48.84 us, max 49.84 us.
- Smoke rerun log:
  `mytmp/tq4bit_smoke_key1_brcb_normalize_v2_rerun_20260627144107.log`
- Smoke rerun profile:
  `mytmp/perflog_smoke_key1_brcb_normalize_v2_rerun_20260627144107`
- Smoke rerun small pack: avg 46.3486 us, p50 46.06 us, p90 47.88 us,
  max 49.90 us.
- Long log: `mytmp/tq4bit_long_key1_brcb_normalize_v2_20260627144246.log`
- Long profile: `mytmp/perflog_long_key1_brcb_normalize_v2_20260627144246`
- Long hot shape:
  count 28, avg 817.18 us, p50 817.34 us, p90 819.08 us, max 819.26 us.
- `op.profile.sh` log:
  `mytmp/op_profile_key1_brcb_normalize_v2_20260627144612.log`
- `op.profile.sh` OPPROF:
  `mytmp/OPPROF_20260627144612_ZEMBAMQVWJVHZFIO`
- `op.profile.sh` runner `pack4bit_to_cache_avg_us=581.54`
- `op.profile.sh` task duration 293.62 us.

V2 passed build, smoke semantics, smoke timing parity, and long-query hot-shape
latency. It failed the profile gate.

## Source Profile Comparison

Route-only refreshed profile:

- OPPROF: `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`
- Total profile source instructions: 25,948,546.
- `NormalizeBatchScalarScale`: 695,216 instructions, estimated 7.703 us.
- `TqSyncVToS`: 49,152 instructions.
- `TqSyncSToV`: 49,432 instructions.
- Task duration: 287.50 us.

key1 Brcb V2 profile:

- OPPROF: `mytmp/OPPROF_20260627144612_ZEMBAMQVWJVHZFIO`
- Total profile source instructions: 26,434,494.
- `NormalizeBatchBrcbScale`: 697,904 instructions, estimated 7.752 us.
- `NormalizeBatch` wrapper dispatch line: 697,904 instructions.
- Task duration: 293.62 us.

The Brcb version removed the scalar `GetValue` path for key1 normalize, but
the replacement sequence `Adds + Duplicate + Div + Brcb + repeat Mul` did not
reduce normalize instruction count. The total profiled source instruction count
also increased by about 486k. The long-query standalone timing is slightly
better, but the default profile workload shows worse end-to-end task duration.

## Failure Analysis

The original scalar path is not dominated only by the V/S synchronization. The
row still pays for fp32 cast, square, reduce, sqrt, cache norm cast, scale, and
output cast. Replacing `GetValue + Muls` with vector reciprocal and `Brcb`
adds vector instructions and extra barriers. On the profile workload, that
extra vector work costs more than the removed scalar round trip.

The V2 cleanup addressed the obvious V1 issue, two row-scale `Mul` calls. It
was the right minimal improvement to try before rejecting the idea. After V2,
the core problem remained: single-loop Brcb normalize changes the instruction
mix but does not remove enough work from the hot path.

## Reflection

This experiment has been pushed far enough for the single-loop Brcb approach:

- V1 tested functional viability.
- V2 removed the most obvious redundant vector `Mul`.
- Smoke was repeated to guard against noise.
- Long-query and `op.profile.sh` disagreed, and the stricter profile gate
  showed a real regression.

The next normalize attempt should not be another small variation of the
single-loop Brcb reciprocal. A better candidate must either batch scalar
synchronization across many rows, reuse already cast fp32 rows without adding a
full second pass, or reduce another dominant cost first so normalize becomes a
cleaner target. Per the current plan, CopyOut/emit and source-profile guided
work should be attempted before returning to normalize.

## Final State

Code reverted to the original `NormalizeBatch` implementation. No runtime
optimization from this experiment is kept.
