# Experiment Report: key1 Encode Direct I16 Postprocess

Date: 2026-06-27

## Conclusion

Rejected and reverted.

The implementation built successfully and smoke text matched the route-only
baseline exactly, but the required smoke small-shape performance gate failed:

- Route-only smoke small shape avg: 46.59 us.
- Direct i16 smoke avg: 47.22 us.
- Direct i16 smoke rerun avg: 47.79 us.

Because smoke regressed twice, the code was reverted and long query /
`tools/op.profile.sh` were not run. The design and this report are kept as the
experiment record.

## Design Reference

- Design: `20260627_key1_encode_direct_i16_design.md`
- Scope: reduce-sum branch of `EncodeBatch`.
- Intended change: replace
  `Cast(float -> int32) -> Cast(int32 -> int16) -> And(0x000F)` with direct
  `Cast(float -> int16)` before writing `encodedBatch`.

## Baseline

Kept route-only state:

- Smoke log: `mytmp/tq4bit_smoke_route_only_20260627104414.log`
- Smoke profile: `mytmp/perflog_smoke_route_only_20260627104414`
- Smoke small pack shape
  `"2,8,128;2,8,128;16;128,128;2;3"`:
  count 56, avg 46.59 us, p50 46.44 us, p90 47.86 us, max 48.72 us.
- Long hot shape
  `"1802,8,128;1802,8,128;16;128,128;1802;2"`:
  count 28, avg 820.39 us, p50 820.72 us, p90 822.22 us, max 823.88 us.
- Refreshed `op.profile.sh` for source comparison:
  `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`,
  runner 557.89 us, task duration 287.50 us.

## Implementation Summary

The experiment changed only the `m > TQ_REDUCE_SUM_MIN_BATCH_ROWS` branch:

```cpp
Cast(argminIndexU16, qFloat, CAST_RINT, 128);
DataCopy(encodedBatch, argminIndexU16, 128);
```

The small compare branch was left unchanged. Threshold table generation,
`Brcb`, `Compare`, `Select`, and `WholeReduceSum` were unchanged.

## Validation Results

Build:

- Command: `tools/build_debug_perf.sh`
- Log: `mytmp/build_key1_encode_direct_i16_20260627154151.log`
- Result: pass.

Smoke run 1:

- Log: `mytmp/tq4bit_smoke_key1_encode_direct_i16_20260627154718.log`
- Profile: `mytmp/perflog_smoke_key1_encode_direct_i16_20260627154718`
- Output: exactly matched route-only smoke text, including the known baseline
  `of of` repetition.
- Small pack shape:
  count 56, avg 47.22 us, p50 46.62 us, p90 48.56 us, max 50.42 us.

Smoke rerun:

- Log: `mytmp/tq4bit_smoke_key1_encode_direct_i16_rerun_20260627154956.log`
- Profile:
  `mytmp/perflog_smoke_key1_encode_direct_i16_rerun_20260627154956`
- Output: exactly matched route-only smoke text.
- Small pack shape:
  count 56, avg 47.79 us, p50 47.74 us, p90 49.06 us, max 51.12 us.

Long query:

- Not run because the smoke performance gate failed.

`tools/op.profile.sh`:

- Not run because the smoke performance gate failed.

## Failure Analysis

The direct cast itself is semantically viable: the smoke output was identical
to the fixed-seed baseline. The build also confirms that AscendC accepts this
direct `float -> int16` cast.

The performance failure is likely not from executing the changed reduce-sum
branch in smoke. The smoke shape has input shape
`"2,8,128;2,8,128;16;128,128;2;3"`, so the batch size is at the compare-branch
threshold and should not take `m > 16`. The regression therefore points to
generated-code/layout effects from changing `EncodeBatch`, not to the dynamic
work in the smoke hot path.

This also explains why the experiment is not a valid kept optimization: even a
large-shape-only source edit can perturb the key0 small-shape binary when the
implementation is not compile-time isolated.

## Reflection

This first version answered one useful question: direct `float -> int16`
postprocess is buildable and preserves smoke text. It did not answer whether
the idea helps the large key1 route, because it failed before the long-query
gate.

The experiment was not the best final form of the idea because it did not
isolate the new postprocess into a separate key1 instantiation. A better V2
should:

- instantiate the pack class with a compile-time flag such as
  `ENABLE_DIRECT_I16_POSTPROCESS`;
- keep `tilingKey = 0` on the original generated code path;
- enable direct i16 only for `TILING_KEY_IS(1)`;
- move `argminMask` initialization out of the direct reduce-sum branch, because
  that mask is no longer used there.

This V2 is worth one follow-up experiment before abandoning the direct-i16
postprocess idea. If V2 still regresses smoke or fails long/profile, the next
EncodeBatch target should be the actual reduce body rather than another
postprocess shuffle.

## Final State

The runtime code was reverted to the route-only implementation. No runtime
optimization from this experiment is kept.
