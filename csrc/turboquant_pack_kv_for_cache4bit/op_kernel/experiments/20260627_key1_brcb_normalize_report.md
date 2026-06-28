# Experiment Report: key1 Brcb NormalizeBatch

Date: 2026-06-27

## Conclusion

Accepted and restored on 2026-06-28.

The V2 implementation passed the validation that matters for this experiment:

- Smoke output matched the route-only semantic baseline.
- Smoke small-shape timing stayed at parity or improved:
  - V2 first run avg 46.5589 us.
  - V2 rerun avg 46.3486 us.
  - Route-only baseline avg 46.59 us.
- Long-query hot shape improved from avg 820.39 us to avg 817.18 us.
- OPP source attribution shows
  `TurboquantPackKVForCache4bitToCache.Process()` instructions decreased from
  2,173,794 to 2,149,926.

The earlier rejection used `tools/op.profile.sh` runner and task duration as a
hard gate. That is not a valid acceptance rule for this kernel. The profile
total time is noisy and should be used only as a diagnostic. For OPP, inspect
source-level attribution, especially
`TurboquantPackKVForCache4bitToCache.Process()`, rather than rejecting a change
on overall profiler runtime.

## Restoration Validation

After restoring the V2 implementation on 2026-06-28:

- Build passed:
  `mytmp/build_key1_brcb_restore_20260628003911.log`.
- Smoke output matched the route-only semantic baseline in all fresh runs:
  - `mytmp/tq4bit_smoke_key1_brcb_restore_20260628004323.log`
  - `mytmp/tq4bit_smoke_key1_brcb_restore_rerun_20260628004718.log`
  - `mytmp/tq4bit_smoke_key1_brcb_restore_rerun2_20260628005220.log`
- Fresh smoke small-shape timing varied:
  - first run avg 46.7914 us.
  - rerun avg 48.7211 us.
  - second rerun avg 47.7175 us.
  This shape has tokenCount=2 and routes through `tilingKey = 0`, so it does
  not execute `NormalizeBatchBrcbScale`. Treat this as guard-shape variability
  to watch, not as evidence against the key1 Brcb path.
- Fresh long-query hot shape:
  `mytmp/tq4bit_long_key1_brcb_restore_20260628004541.log`,
  profile `mytmp/perflog_long_key1_brcb_restore_20260628004541`.
  Shape `"1802,8,128;1802,8,128;16;128,128;1802;2"`: count 28,
  avg 818.0621 us, p50 818.2600 us, p90 819.3800 us, max 821.1200 us.
  This remains better than the route-only baseline avg 820.39 us.
- Fresh `op.profile.sh`:
  - log `mytmp/op_profile_key1_brcb_restore_20260628005409.log`
  - OPPROF `mytmp/OPPROF_20260628005409_PZNDWDIEMDWIZEHR`
  - runner `pack4bit_to_cache_avg_us=531.85`
  - task duration 299.54 us
  - `TurboquantPackKVForCache4bitToCache.Process()`: 2,149,926 instructions.

The fresh OPP `Process()` instruction count matches the earlier accepted V2
profile and remains lower than route-only. The OPP runner/task duration is
recorded only as diagnostic data.

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
latency.

## Source Profile Comparison

Route-only refreshed profile:

- OPPROF: `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`
- Total profile source instructions: 25,948,546.
- `TurboquantPackKVForCache4bitToCache.Process()`: 2,173,794 instructions,
  estimated 24.085 us.
- `NormalizeBatchScalarScale`: 695,216 instructions, estimated 7.703 us.
- `TqSyncVToS`: 49,152 instructions.
- `TqSyncSToV`: 49,432 instructions.
- Task duration: 287.50 us.

key1 Brcb V2 profile:

- OPPROF: `mytmp/OPPROF_20260627144612_ZEMBAMQVWJVHZFIO`
- Total profile source instructions: 26,434,494.
- `TurboquantPackKVForCache4bitToCache.Process()`: 2,149,926 instructions,
  estimated 23.880 us.
- `NormalizeBatchBrcbScale`: 697,904 instructions, estimated 7.752 us.
- `NormalizeBatch` wrapper dispatch line: 697,904 instructions.
- Task duration: 293.62 us.

The Brcb version removed the scalar `GetValue` path for key1 normalize, but
the replacement sequence `Adds + Duplicate + Div + Brcb + repeat Mul` did not
reduce normalize function instruction count. At the `Process()` attribution
level requested for OPP review, the V2 profile is lower by 23,868 instructions.
The `op.profile.sh` runner and task duration regressed in this run, but those
overall timings are not stable enough to override smoke and long-query results.

## Profile Interpretation

The original scalar path is not dominated only by the V/S synchronization. The
row still pays for fp32 cast, square, reduce, sqrt, cache norm cast, scale, and
output cast. Replacing `GetValue + Muls` with vector reciprocal and `Brcb`
adds vector instructions and extra barriers. That explains why the normalize
function's own instruction count stays flat.

The V2 cleanup addressed the obvious V1 issue, two row-scale `Mul` calls. It
was the right minimal improvement before deciding. After V2, smoke and long
query both improved or held, and `Process()` source attribution decreased. This
is enough to keep the key1-only Brcb implementation.

## Reflection

This experiment has been pushed far enough for the single-loop Brcb approach:

- V1 tested functional viability.
- V2 removed the most obvious redundant vector `Mul`.
- Smoke was repeated to guard against noise.
- Long-query and `op.profile.sh` total time disagreed, so the acceptance rule
  was corrected to avoid rejecting based on noisy profiler totals.
- OPP remains useful for source attribution; the relevant `Process()` count
  decreased.

The single-loop Brcb reciprocal is accepted as the current key1 normalize path.
Future normalize work should target a materially larger change, such as
batching scalar synchronization across many rows or reducing another dominant
cost first, and should be judged by smoke/long results plus `Process()` source
attribution.

## Final State

Runtime code keeps the key1-only Brcb V2 implementation. `tilingKey = 0` stays
on the original scalar-scale normalize path.
