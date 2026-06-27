# Experiment Design: key1 Brcb NormalizeBatch

Date: 2026-06-27

## Hypothesis

`NormalizeBatch` currently performs one scalar-engine round trip per row:

```cpp
TqSyncVToS();
const float normF = normAcc.GetValue(0);
TqSyncSToV();
Muls(fp32Row, fp32Row, 1.0f / (normF + eps), 128);
```

On the large contiguous route (`tilingKey = 1`), this happens for many rows
and may add avoidable instruction and synchronization cost. FIA softmax and
local RMSNorm implementations use `Brcb` plus repeat-strided vector binary ops
to broadcast row scalars from UB without `GetValue`. Applying the same pattern
to normalize may reduce large-shape latency while preserving output semantics.

## Target Scope

- Only enable for `TILING_KEY_IS(1)`.
- Keep `tilingKey = 0` on the existing scalar-scale implementation.
- Keep `tilingKey = 2` untouched.
- Target long hot shape:
  `"1802,8,128;1802,8,128;16;128,128;1802;2"`.

## Semantic Requirements

- `normsBuf_` must store the same cache norm semantics as before:
  `Cast(normAcc, CAST_RINT)` immediately after `Sqrt`, before adding eps or
  taking reciprocal.
- The normalized row must remain equivalent to `x / (norm + 1e-10)`.
- Smoke output must not introduce new repetition or semantic drift.
- Small smoke shape should still route through `key0`, but must be measured
  because generated code layout changes have previously affected timing.

## Implementation Sketch

Add a wrapper:

```cpp
NormalizeBatch(m) {
  if constexpr (TILING_KEY_IS(1)) {
    NormalizeBatchBrcbScale(m);
  } else {
    NormalizeBatchScalarScale(m);
  }
}
```

`NormalizeBatchBrcbScale` keeps the original single-loop structure:

1. Cast row to fp32.
2. Square row and `ReduceSum`.
3. `Sqrt(normAcc)`.
4. Cast/store cache norm to `normsBuf_`.
5. Vector-side reciprocal:
   - `Adds(normAcc, normAcc, eps, 1)`
   - `Duplicate(one, 1.0f, 1)`
   - `Div(normAcc, one, normAcc, 1)`
6. `Brcb(scaleBlock, normAcc, 1, BrcbRepeatParams(1, 8))`.
7. Use vector `Mul` to apply the scale to two 64-fp32 halves of the row.
8. Cast normalized row to `aBatch`.

The first implementation uses existing `reduceOutBuf_`:

- `fp32Row`: row data
- `scaleBlock`: aliases the square buffer after `ReduceSum`
- `fp32Tmp`: reduce workspace, then one-scalar constant buffer for reciprocal

## Validation Gates

1. Build with `tools/build_debug_perf.sh`.
2. Run smoke with `XRX_TQ4BIT_PROFILE=1`.
   - Output must match route-only semantics.
   - Small shape avg must not regress from 46.59 us.
3. If smoke passes, run long query.
   - Hot shape avg must improve from 820.39 us.
4. If long improves, run `tools/op.profile.sh`.
   - Must pass without `ret=507015`.
5. Keep code only if all gates pass. Otherwise revert code and keep this
   design plus the experiment report.

## Initial Risk Assessment

- Vector `Div` may be slower than scalar reciprocal plus `Muls`; the benefit
  depends on whether removing V/S sync outweighs extra vector work.
- `Brcb` broadcasts one 32B fp32 block. The first implementation applies that
  block to two 64-element halves, matching the row scalar semantics while
  avoiding an additional `Brcb`.
- Tiny numerical differences in reciprocal may change quantization near code
  thresholds; smoke output and long query are correctness gates.

## V2 Refinement

The first buildable implementation used two separate `Mul` calls:

```cpp
Mul(row[0:64], scaleBlock[0:64]);
Mul(row[64:128], scaleBlock[0:64]);
```

Source profile showed the Brcb version increased vector instructions and
`op.profile.sh` task duration. Before rejecting the whole idea, try a narrower
V2 change: replace the two half-row `Mul` calls with one repeat-strided `Mul`
using `src1RepStride = 0`, so the 64-element Brcb scale block is reused for the
second half of the row.

This does not change semantics and is the most direct cleanup of V1's obvious
extra vector issue. If V2 still fails `op.profile.sh` or long-query gates, the
Brcb single-loop approach should be rejected and reverted.
