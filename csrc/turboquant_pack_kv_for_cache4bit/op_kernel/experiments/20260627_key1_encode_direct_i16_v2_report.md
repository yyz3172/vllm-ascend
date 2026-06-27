# Experiment Report: key1-Isolated Encode Direct I16 V2

Date: 2026-06-27

## Conclusion

Rejected and reverted.

The key1-isolated implementation built successfully and preserved smoke output
exactly, but it still failed the required small-shape smoke performance gate:

- Route-only smoke small shape avg: 46.59 us.
- V2 smoke small shape avg: 46.77 us.
- V2 p90/max also regressed: p90 48.36 us, max 49.28 us, versus baseline
  p90 47.86 us, max 48.72 us.

Because smoke regressed, the long-query and `tools/op.profile.sh` gates were
not run. The runtime code was reverted. This report and the design document
are kept as the experiment record.

## Design Reference

- Design: `20260627_key1_encode_direct_i16_v2_design.md`
- Target: reduce-sum branch postprocess after `EncodeQuantCodesByReduceSum`.
- Intended isolation:
  - keep `tilingKey = 0` on the original
    `float -> int32 -> int16 -> And(0x000F)` chain;
  - use direct `float -> int16` only for `tilingKey = 1`;
  - avoid the unused `argminMask` initialization in the key1 direct branch.

## Baseline

Kept route-only state:

- Baseline commit: `a63ec1ba perf(turboquant): add large pack tiling route`
- Smoke log: `mytmp/tq4bit_smoke_route_only_20260627104414.log`
- Smoke profile: `mytmp/perflog_smoke_route_only_20260627104414`
- Smoke small pack shape
  `"2,8,128;2,8,128;16;128,128;2;3"`:
  count 56, avg 46.59 us, p50 46.44 us, p90 47.86 us, max 48.72 us.
- Long hot shape
  `"1802,8,128;1802,8,128;16;128,128;1802;2"`:
  count 28, avg 820.39 us, p50 820.72 us, p90 822.22 us, max 823.88 us.
- Refreshed route-only `tools/op.profile.sh`:
  `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`,
  runner 557.89 us, task duration 287.50 us.

## Implementation Summary

The experiment added a compile-time boolean template parameter to
`TurboquantPackKVForCache4bitToCache`:

```cpp
template <typename T, typename RotateMmT, bool ENABLE_DIRECT_I16_POSTPROCESS>
```

The kernel entry instantiated the class with:

- `ENABLE_DIRECT_I16_POSTPROCESS = true` for `TILING_KEY_IS(1)`;
- `ENABLE_DIRECT_I16_POSTPROCESS = false` for all other routes.

Only the reduce-sum branch used direct postprocess:

```cpp
Cast(argminIndexU16, qFloat, CAST_RINT, TQ_PACK_D);
DataCopy(encodedBatch, argminIndexU16, TQ_PACK_D);
```

The compare branch retained the original conversion and mask chain.

## Validation Results

Build:

- Command: `tools/build_debug_perf.sh`
- Log: `mytmp/build_key1_encode_direct_i16_v2_20260627155855.log`
- Result: pass.

Smoke:

- Log: `mytmp/tq4bit_smoke_key1_encode_direct_i16_v2_20260627160513.log`
- Profile: `mytmp/perflog_smoke_key1_encode_direct_i16_v2_20260627160513`
- Output: exactly matched route-only smoke text, including the known baseline
  `of of` repetition.
- Small pack shape
  `"2,8,128;2,8,128;16;128,128;2;3"`:
  count 56, avg 46.77 us, p50 46.58 us, p90 48.36 us, max 49.28 us.

Long query:

- Not run because the smoke performance gate failed.

`tools/op.profile.sh`:

- Not run because the smoke performance gate failed.

## Failure Analysis

V2 confirms that direct `float -> int16` postprocess is semantically safe for
the smoke workload: output was byte-for-byte equivalent to the route-only
baseline.

The performance result is still negative. The smoke shape is the small
`tilingKey = 0` path, so it does not execute the key1 direct reduce-sum branch.
The regression therefore points to generated-code effects rather than dynamic
work in the changed branch. The compile-time flag removed the direct branch
from key0, but the kernel entry and generated binary still changed because two
class instantiations are now present in the same kernel source. That is enough
to perturb instruction layout and scheduling for the guarded small shape.

The direct-i16 optimization is also small in scope: it removes only one
intermediate cast and one mask operation after the dominant
`Brcb + Compare + Select + WholeReduceSum` body. Even if key1 long improved,
the risk of small-shape layout regression makes this postprocess-only idea a
poor candidate for the default operator.

## Reflection

This was the best reasonable version of the direct-i16 postprocess idea before
moving on: it isolated key1 at compile time and avoided dead mask setup in the
direct path. The remaining regression means the idea is not worth further
postprocess-only retries.

The next EncodeBatch experiment should target the actual reduce body rather
than conversion cleanup. Specifically, inspect whether the threshold
`Brcb + Compare + Select + WholeReduceSum` sequence can reduce repeated table
setup, mask generation, or per-row vector instructions without changing the
quantization result.

## Final State

The runtime code was reverted to the route-only implementation. No runtime
optimization from this experiment is kept.
