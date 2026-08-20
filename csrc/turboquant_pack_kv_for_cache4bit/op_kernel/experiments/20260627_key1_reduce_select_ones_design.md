# Experiment Design: key1 Reduce Select Persistent Ones

Date: 2026-06-27

## Hypothesis

`EncodeQuantCodesByReduceSum` currently does this for every encoded row:

```cpp
Brcb(quantOnes, yFp32, ...);
Compare(quantMask, quantOnes, quantThresholds, GT, 2048);
Duplicate(quantOnes, 1.0f, 2048);
Select(quantOnes, quantMask, quantOnes, 0.0f, VSEL_TENSOR_SCALAR_MODE, 2048);
WholeReduceSum(qFloat, quantOnes, ...);
```

The row-expanded `quantOnes` buffer first holds repeated `y` values, then gets
overwritten with 1.0 before the select. The route-only source profile
attributes 122,880 instructions to the duplicate line in the reduce body.

Hypothesis: keep a persistent 1.0 table in `quantCodeBuf_` after the threshold
table, then select from that table into `quantOnes`. This removes the per-row
`Duplicate(quantOnes, 1.0f, 2048)` while preserving the exact compare/select/
reduce semantics.

## Scope

- Runtime target: reduce-sum EncodeBatch branch.
- No threshold formula changes.
- No direct-i16 postprocess changes.
- No NormalizeBatch, RotateBatchMatmul, CopyOut, host tiling, or KFC changes.
- First implementation may be shared code because the smoke path
  (`m <= 16`) does not execute the reduce-sum branch; however, smoke timing is
  still a strict gate because code layout can move.

## Implementation Sketch

Increase `quantCodeBuf_` capacity by one reduce table:

```cpp
TQ_QUANT_WORK_TABLES = 2;
TQ_QUANT_CODE_BYTES = TQ_QUANT_WORK_TABLES * TQ_QUANT_TABLE_ELEMS * sizeof(float);
```

Use the first table for thresholds and the second table for persistent ones:

```cpp
auto quantThresholds = quantCodeBuffer;
auto quantSelectOnes = quantCodeBuffer[TQ_QUANT_TABLE_ELEMS];
```

In `PrepareQuantThresholdTable`:

1. Fill the threshold table exactly as today.
2. Duplicate the persistent ones table once:

```cpp
Duplicate(quantSelectOnes, 1.0f, TQ_QUANT_TABLE_ELEMS);
```

Change `EncodeQuantCodesByReduceSum` to receive both threshold and one tables:

```cpp
Compare(quantMask, quantOnes, quantThresholds, GT, TQ_QUANT_TABLE_ELEMS);
Select(quantOnes, quantMask, quantSelectOnes, 0.0f,
       VSEL_TENSOR_SCALAR_MODE, TQ_QUANT_TABLE_ELEMS);
WholeReduceSum(qFloat, quantOnes, ...);
```

No per-row duplicate remains.

## Semantic Requirements

- For every element, `quantOnes` after select must still be `1.0f` when
  `y > threshold`, else `0.0f`.
- `qFloat` from `WholeReduceSum` must match the route-only result.
- Smoke output must exactly match the route-only baseline. No new repeated
  token behavior is acceptable.

## Risks

- The persistent ones table increases UB allocation by 8 KB.
- Filling the ones table once may affect setup and small-shape code layout.
- If `Select` with a separate source table generates more expensive code than
  duplicate-in-place, long/profile may regress.
- `quantCodeBuf_` is also used by the compare branch for code vectors. Because
  the buffer mode tracks either codes or thresholds, this experiment must keep
  the two modes exclusive and not assume both are live at once.

## Validation Gates

1. Build with `tools/build_debug_perf.sh`.
2. Run smoke with profile enabled.
   - Output must match the route-only baseline exactly.
   - Small pack shape
     `"2,8,128;2,8,128;16;128,128;2;3"` must not regress from avg 46.59 us.
3. If smoke passes, run fixed long query.
   - Hot shape
     `"1802,8,128;1802,8,128;16;128,128;1802;2"` must improve from avg
     820.39 us.
4. If long improves, run `tools/op.profile.sh`.
   - Must improve versus refreshed route-only runner 557.89 us and task
     duration 287.50 us.
5. Keep runtime code only if all gates pass. Otherwise revert the code and
   write a failure report.

## Reflection Before Coding

This is a narrow experiment and may not be enough for a key breakthrough, but
it removes a measured hot operation inside the actual reduce body. If it fails,
the next reduce-body idea must change the table/reduction shape more
substantially; another postprocess or Select-source shuffle should not be
retried.
