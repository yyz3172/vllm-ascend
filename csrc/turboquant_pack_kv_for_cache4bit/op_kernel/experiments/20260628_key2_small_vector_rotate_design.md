# Key2 Small-Vector Rotate Design

Date: 2026-06-28

## Goal

Add `tilingKey = 2` for the tiny contiguous pack shape so it can avoid Cube
rotate matmul registration and KFC setup.

This is a performance experiment on top of the existing correctness baseline.
It must not change the default `key0` route or the large contiguous `key1`
route.

## Assumptions

- The target smoke shape is the tiny contiguous pack case:
  `[tokenCount=2, numHeads=8, headDim=128]`, i.e. `nVec <= 16`.
- Inputs are contiguous only when key/value head stride is 128 and token stride
  is `numHeads * 128`.
- `key2` should be AIV-only. It must return from the kernel entry before
  constructing or registering `TqRotateMatmulOp`.
- `key2` is not intended for long-query hot shapes; those must keep routing to
  `key1`.

## Rejected Prior Design

The previous key2 attempt used fp32 vector rotate but was still scalar-loop
dominated. It built and preserved smoke text, but smoke timing was not
repeatable and the rerun regressed. Do not retry that per-output scalar-loop
shape.

## Planned Implementation

Host routing:

- Add `TQ_PACK_TILING_KEY_SMALL_VECTOR = 2`.
- Select it only for contiguous input with:
  - `tokenCount <= 2`
  - `numHeads <= 8`
  - `nVec <= 16`
- Keep large contiguous routing on key1.

Kernel entry:

- Branch on key2 before `REGIST_MATMUL_OBJ_STATIC`.
- Use `KERNEL_TYPE_AIV_ONLY` for key2.
- Instantiate the pack class with a dummy rotate type and a null rotate pointer.
- Return before the key0/key1 path declares `TqRotateMatmulOp`.

Rotate implementation:

- Keep existing CopyIn, normalize, encode, and CopyOut stages.
- In `ComputeBatch()`, dispatch key2 to a vector rotate path.
- Cast normalized rows to fp32 once.
- Load each contiguous rotation row from GM once per batch.
- Accumulate `y[row, :] += x[row, k] * rotation_t[k, :]` with vector
  `Muls + Add`.
- Cast the fp32 accumulated rows back to `T` for the existing `EncodeBatch()`.

Temporary buffer lifetime:

- `rotateWorkBuf_` remains rotate-owned and holds the fp32 normalized rows plus
  rotation row scratch.
- `quantOnesBuf_` is used as the fp32 y accumulator only for key2. Key2 is
  limited to `m <= 16`, and `EncodeBatch()` takes the compare path for
  `m <= TQ_REDUCE_SUM_MIN_BATCH_ROWS`, so the reduce-sum quant ones table is
  not live in this route.

## Validation

1. Build with `tools/build_debug_perf.sh`.
2. Run the TurboQuant 4-bit smoke test with profiling enabled.
3. Verify smoke text has no new repetition or obvious corruption.
4. Compare small pack shape `[2,8,128]` against the fixed baseline
   avg `45.96us`.
5. If smoke passes and the small shape does not regress, run the long-query
   profile and confirm the hot shape still routes away from key2.

## Keep Or Revert Criteria

Keep only if:

- build passes;
- smoke output remains semantically acceptable;
- small shape timing is repeatably no worse than the fixed baseline.

Revert runtime code if the smoke gate fails. Keep this design/report record.
