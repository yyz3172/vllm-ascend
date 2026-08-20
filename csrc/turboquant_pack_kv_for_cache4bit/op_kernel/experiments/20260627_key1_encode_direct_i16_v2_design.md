# Experiment Design: key1-Isolated Encode Direct I16 V2

Date: 2026-06-27

## Hypothesis

V1 direct i16 proved two things:

- `Cast(int16, float, CAST_RINT, 128)` builds successfully.
- Smoke text is unchanged, so the direct cast is semantically viable for the
  tested workload.

V1 failed because smoke small-shape latency regressed twice, even though the
smoke shape should not execute the changed reduce-sum branch. That points to
generated-code layout pollution: `tilingKey = 0` and `tilingKey = 1` currently
instantiate the same class body, so a key1-intended source edit can still
change the key0 generated binary.

V2 hypothesis: instantiate a compile-time key1-only postprocess path and keep
key0 on the original conversion chain. This should preserve smoke timing while
allowing long-query key1 to test the direct i16 benefit.

## Scope

- Add a compile-time flag to the pack kernel class, for example:

```cpp
template <typename T, typename RotateMmT, bool ENABLE_DIRECT_I16_POSTPROCESS>
class TurboquantPackKVForCache4bitToCache { ... };
```

- Instantiate:
  - `TILING_KEY_IS(0)`: `ENABLE_DIRECT_I16_POSTPROCESS = false`
  - `TILING_KEY_IS(1)`: `ENABLE_DIRECT_I16_POSTPROCESS = true`
- Only the reduce-sum branch may use direct `float -> int16` cast when the flag
  is true.
- The compare branch remains unchanged.
- No NormalizeBatch, RotateBatchMatmul, CopyOut, or host tiling changes.

## Implementation Sketch

Add a helper for reduce-sum postprocess:

```cpp
EncodeReduceSumRow(...) {
  EncodeQuantCodesByReduceSum(qFloat, ...);
  if constexpr (ENABLE_DIRECT_I16_POSTPROCESS) {
    Cast(argminIndexU16, qFloat, CAST_RINT, 128);
    DataCopy(encodedBatch, argminIndexU16, 128);
  } else {
    Cast(argminIndex, qFloat, CAST_RINT, 128);
    Cast(argminIndexU16, argminIndex, CAST_NONE, 128);
    And(argminIndexU16, argminIndexU16, argminMask, 128);
    DataCopy(encodedBatch, argminIndexU16, 128);
  }
}
```

Move `Duplicate(argminMask, 0x000F)` so it is only executed when required:

- Always execute it before the compare branch.
- Execute it before reduce-sum only when
  `!ENABLE_DIRECT_I16_POSTPROCESS`.

This avoids leaving dead mask initialization in the key1 direct path.

In the kernel entry, split dispatch into separate instantiations:

```cpp
if (TILING_KEY_IS(0)) {
  RunPack<false>(...);
} else if (TILING_KEY_IS(1)) {
  RunPack<true>(...);
}
```

Use a helper template if that keeps entry code readable.

## Semantic Requirements

- key0 generated behavior must remain equivalent to the route-only baseline.
- key1 direct postprocess must produce encoded indices in `[0, 15]`.
- Smoke output must exactly match the route-only fixed-seed text.
- No new repeated-token semantics are acceptable.

## Validation Gates

1. Build with `tools/build_debug_perf.sh`.
2. Run smoke twice if the first smoke is close to the baseline.
   - Output must match route-only text.
   - Small pack shape
     `"2,8,128;2,8,128;16;128,128;2;3"` must not regress from avg 46.59 us.
3. If smoke passes, run fixed long query.
   - Hot shape
     `"1802,8,128;1802,8,128;16;128,128;1802;2"` must improve from avg
     820.39 us.
4. If long improves, run `tools/op.profile.sh`.
   - Must improve versus refreshed route-only runner 557.89 us and task
     duration 287.50 us.
5. Keep the code only if all gates pass. Otherwise revert code and write a
   report.

## Reflection Before Coding

This V2 specifically addresses the weakness of V1. If it still regresses
smoke, the direct-i16 postprocess idea is not worth more time. If smoke passes
but long does not improve, the saved cast/and instructions are too small
relative to the actual reduce body, and the next EncodeBatch experiment should
target `Brcb + Compare + Select + WholeReduceSum` itself.
