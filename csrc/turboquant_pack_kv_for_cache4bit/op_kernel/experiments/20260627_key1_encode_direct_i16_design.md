# Experiment Design: key1 Encode Direct I16 Postprocess

Date: 2026-06-27

## Hypothesis

The current large-shape Encode path already uses the lookup/reduce idea from
the earlier `f3a96b3d` discussion in a different form:

- Build a 128x16 threshold table.
- Broadcast one fp32 row with `Brcb`.
- Compare against all thresholds.
- Convert the mask to 1/0 and use `WholeReduceSum` to count thresholds passed.

The refreshed route-only source profile shows Encode is still a major cost:

- OPPROF: `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`
- `EncodeBatch`: 993,792 instructions, estimated 11.011 us.
- `EncodeQuantCodesByReduceSum`: 450,560 instructions, estimated 4.992 us.
- Hot lines after reduce include the postprocess chain:
  `Cast(qFloat -> int32)`, `Cast(int32 -> int16)`, `And(0x000F)`, and
  `DataCopy(encodedBatch, int16)`.

For the reduce-sum path, `qFloat` is exactly the count of thresholds passed
and should be in `[0, 15]`. Therefore the `int32` intermediate and `And`
mask may be redundant. If AscendC supports `Cast(int16, float, CAST_RINT)`,
we can write:

```cpp
Cast(argminIndexU16, qFloat, CAST_RINT, 128);
DataCopy(encodedBatch, argminIndexU16, 128);
```

This removes one 128-element cast, one 128-element `And`, and their barriers
per encoded row on the large reduce-sum path.

## Scope

- Only change the `m > TQ_REDUCE_SUM_MIN_BATCH_ROWS` branch used by the large
  route.
- Keep the small compare branch unchanged for V1 because small-shape smoke is
  guarded and previous small-shape experiments were noisy.
- Keep threshold construction, `Brcb`, `Compare`, `Select`, and
  `WholeReduceSum` unchanged.
- Do not change CopyOut, NormalizeBatch, RotateBatchMatmul, or batch emission.

This is intentionally a narrow postprocess experiment, not a reimplementation
of codebook argmin lookup.

## Semantic Requirements

- Encoded row values must remain identical 4-bit indices in `[0, 15]`.
- Smoke output must exactly match the route-only baseline text, including the
  known baseline `of of` repetition and no new duplication.
- The direct cast must not change bf16/fp16 behavior near thresholds. The
  reduce result is an integer-valued fp32 sum, so `CAST_RINT` should preserve
  it if the compiler supports the conversion.
- If direct `float -> int16` cast is unsupported or emits invalid code, the
  experiment is rejected at the build gate without broadening the change.

## Implementation Sketch

In the reduce-sum branch of `EncodeBatch`:

1. Keep:
   - `Cast(yFp32, yBatch, CAST_NONE)`
   - `EncodeQuantCodesByReduceSum(qFloat, ...)`
2. Replace:

```cpp
Cast(argminIndex, qFloat, CAST_RINT, 128);
Cast(argminIndexU16, argminIndex, CAST_NONE, 128);
And(argminIndexU16, argminIndexU16, argminMask, 128);
DataCopy(encodedBatch, argminIndexU16, 128);
```

with:

```cpp
Cast(argminIndexU16, qFloat, CAST_RINT, 128);
DataCopy(encodedBatch, argminIndexU16, 128);
```

3. Keep the compare branch unchanged in V1.
4. If the build passes and smoke semantics match, compare long-query and
   `op.profile.sh` against route-only.

## Validation Gates

1. Build with `tools/build_debug_perf.sh`.
2. Run smoke with profile enabled.
   - Output must match the route-only baseline.
   - Small pack shape
     `"2,8,128;2,8,128;16;128,128;2;3"` must not regress from avg 46.59 us.
3. If smoke passes, run the fixed long-query workload.
   - Hot shape
     `"1802,8,128;1802,8,128;16;128,128;1802;2"` must improve from avg
     820.39 us.
4. If long improves, run `tools/op.profile.sh`.
   - Must improve versus refreshed route-only runner 557.89 us and task
     duration 287.50 us.
5. Keep the code only if all gates pass. Otherwise revert the code, keep this
   design, and write a report with the failure reason.

## Reflection Before Coding

This is probably the smallest credible EncodeBatch experiment because it
removes instructions after the already accepted reduce-sum lookup strategy
without changing threshold semantics. It may still be too small to move the
long hot shape, and direct `float -> int16` cast may compile into a slower or
unsupported sequence. If it fails, the next Encode experiment should target the
actual reduce body, not another postprocess shuffle.
