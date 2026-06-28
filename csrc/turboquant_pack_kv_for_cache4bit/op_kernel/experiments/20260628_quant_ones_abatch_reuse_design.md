# Experiment Design: Quant Ones Reuse Released A Batch Slot

Date: 2026-06-28

## Hypothesis

Commit `532a507c` correctly stopped using `rotateWorkBuf_` as the
reduce-sum quantization temporary table, but the new permanent 8 KB
`quantOnesBuf_` appears to perturb UB layout enough to regress the long hot
shape `[1802,8,128]` by about 20.87 us.

The reduce-sum table only needs to be live during `EncodeBatch()`. At that
point `RotateBatchMatmul()` has already freed the `aBatchQue_` tensor used as
matmul input. Reusing one released `aBatchQue_` slot as a temporary float
table should remove the permanent 8 KB VECCALC buffer while keeping the
workspace isolated from `rotateWorkBuf_`.

## Scope

- Target path: `EncodeQuantCodesByReduceSum()` in the current key0/key1 shared
  kernel.
- Keep the quantization math and row output layout unchanged.
- Keep `rotateWorkBuf_` owned only by rotate matmul.
- Do not implement key2.
- Do not change CopyIn, Normalize, Rotate, CopyOut, tiling, or matmul setup.

## Lifetime Proof

`ComputeBatch()` runs stages in this order:

1. `NormalizeBatch(m)` dequeues `xBatchQue_`, allocates/enqueues `aBatchQue_`.
2. `RotateBatchMatmul(m, ...)` dequeues `aBatchQue_`, runs matmul, enqueues
   `yBatchQue_`, then frees the `aBatchQue_` tensor.
3. `EncodeBatch(m)` dequeues `yBatchQue_` and writes `encodedBatchQue_`.

The proposed temporary is allocated from `aBatchQue_` only inside
`EncodeBatch()` after step 2 has completed. It is freed before `EncodeBatch()`
returns. CopyOut starts only after `ComputeBatch()` returns and does not use
`aBatchQue_`.

Async CopyOut can leave `packedRowBuf_` referenced by MTE3 between tasks, but
that does not alias `aBatchQue_`. The temporary also does not alias
`rotateWorkBuf_`, so future rotate/matmul workspace changes cannot corrupt the
reduce-sum table.

## Implementation Sketch

- Remove `quantOnesBuf_` and its `pipe_->InitBuffer()` call.
- Change `EncodeQuantCodesByReduceSum()` to receive a
  `LocalTensor<float>& quantOnes`.
- In the reduce-sum branch of `EncodeBatch()`:
  - allocate one `aBatchQue_` tensor after preparing the threshold table;
  - reinterpret it as `float`;
  - pass it to each row's reduce-sum encode;
  - free it before leaving the branch.

The `aBatchQue_` slot is 16 KB at `TQ_MAX_BATCH_M * TQ_PACK_D * sizeof(T)`,
which is larger than `TQ_QUANT_ONES_BYTES` (8 KB).

## Validation Gates

1. Build with `bash tools/build_debug_perf.sh`.
2. Run smoke with profile enabled.
   - Output must remain semantically acceptable.
   - Smoke shape `[2,8,128]` must not regress versus current fixed baseline
     avg 45.96 us.
3. If smoke passes, run fixed long query.
   - Hot shape `[1802,8,128]` should recover toward the pre-fix baseline
     avg 818.06 us.
4. If long improves, run `tools/op.profile.sh` and record source attribution.

## Keep Or Revert Criteria

Keep the runtime change only if it builds, preserves smoke semantics, does not
regress the smoke guarded shape, and improves the long hot shape. Revert if
the queue-slot lifetime assumption fails in build/runtime validation or if the
long hot shape does not recover meaningfully.
