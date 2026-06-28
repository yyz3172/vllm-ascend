# TurboQuant 4-bit Pack KV Current Work Summary

## Current Understanding

The current committed change is a correctness bug fix, not a performance
experiment.

Commit `532a507c` fixes an unsafe workspace alias in
`EncodeQuantCodesByReduceSum()`. The old code used `rotateWorkBuf_` as the
temporary table for the reduce-sum quantization path. That buffer belongs to
the rotate matmul workspace, so reusing it from encode relies on fragile
lifetime assumptions. If the pipeline, matmul implementation, or scheduling is
changed later, the alias can become a real data hazard.

The fix adds a dedicated `quantOnesBuf_`:

- `TQ_QUANT_ONES_BYTES = TQ_QUANT_TABLE_ELEMS * sizeof(float)`
- `pipe_->InitBuffer(quantOnesBuf_, TQ_QUANT_ONES_BYTES)`
- `EncodeQuantCodesByReduceSum()` now uses `quantOnesBuf_`

This fix must be kept even though the long-query hot shape regressed.

## Current Baseline

Current validation after `532a507c`:

- Smoke log:
  `mytmp/tq4bit_smoke_current_baseline_20260628040907.log`
- Long query log:
  `mytmp/tq4bit_long_current_baseline_20260628041159.log`
- OPP:
  `mytmp/OPPROF_20260628041452_HGTZMMMAAPODVUYD`
- OPP function summary:
  `mytmp/opprof_current_baseline_function_top80_20260628041452.txt`

Smoke pack result:

- Shape `[2,8,128]`
- count `56`
- avg `45.96us`
- p50 `45.54us`
- p90 `47.62us`
- max `51.12us`

Long-query pack result:

- Hot shape `[1802,8,128]`
- count `28`
- avg `838.93us`
- p50 `838.32us`
- p90 `841.78us`
- max `843.68us`

Previous comparable baseline:

- `mytmp/tq4bit_long_key1_brcb_restore_20260628004541.log`
- Hot shape `[1802,8,128]` avg `818.06us`

The current long-query regression is therefore real:

- `838.93us - 818.06us = 20.87us`
- about `2.55%`

OPP source attribution did not show a meaningful instruction-count increase in
the main compute path:

- `ComputeBatch`: `1,749,542` in both runs
- `EncodeBatch`: `971,696` in both runs
- `NormalizeBatchBrcbScale`: `697,904` in both runs
- `EncodeQuantCodesByReduceSum`: `417,792` in both runs
- `RotateBatchMatmul`: `78,662` in both runs

The likely cause of the long-query regression is therefore not an obvious new
instruction path. It is more likely caused by the additional 8 KB VECCALC UB
buffer changing UB layout, resource pressure, or scheduling behavior. The OPP
task duration also moved from `299.54us` to `303.68us`, which supports that the
change is small but measurable.

## Why Key2 Was Not Added Yet

`tilingKey = 2` is planned as the small-vector no-matmul path. It is still not
implemented.

The reason is scope separation. The current work was triggered by a correctness
risk around `rotateWorkBuf_`. Mixing that bug fix with a new key2 performance
path would make the result hard to reason about:

- a semantic fix must be kept even if it regresses performance;
- a performance experiment must be kept only if it improves the target shape
  without breaking smoke semantics;
- combining both would make it unclear whether a regression comes from the bug
  fix or from the new path.

The correct sequence is:

1. Keep the workspace isolation fix as a standalone commit.
2. Rebuild performance on top of that fixed baseline.
3. Implement `key2` as a separate experiment and separate commit.

## Next Plan

### 1. Recover The Long-Query Regression

The first target is to keep the correctness fix while recovering the
`[1802,8,128]` regression.

Candidate directions:

- Avoid adding a permanent 8 KB UB buffer if a provably dead existing buffer can
  hold the reduce-sum temporary table.
- Already rejected: reusing a released `aBatchQue_` slot as the 8 KB
  reduce-sum temporary. It built and preserved smoke text, but the shared
  variant regressed smoke `[2,8,128]` from avg `45.96us` to `46.79us`, and the
  key1-only `if constexpr` variant regressed it to `47.58us`. Report:
  `experiments/20260628_quant_ones_abatch_reuse_report.md`.
- Do not return to `rotateWorkBuf_` unless the lifetime can be proven safe for
  every path and documented in code. The current understanding is that
  `rotateWorkBuf_` should remain owned by rotate matmul.
- Look for buffers that are dead during `EncodeQuantCodesByReduceSum()` and
  have sufficient capacity without overlapping async CopyOut or queued tensor
  lifetime.
- If no safe full-size buffer exists without key0 codegen perturbation, test a
  smaller/chunked reduce temporary design or find stronger key1 codegen
  isolation before another UB-layout experiment.

Acceptance gate:

- Smoke output must remain semantically acceptable.
- Smoke `[2,8,128]` must not regress.
- Long hot shape `[1802,8,128]` must recover toward the pre-fix baseline
  `818.06us`, or the remaining regression must have a clear correctness reason.
- OPP source attribution must be recorded, but OPP total time is only
  diagnostic.

### 2. Implement Key2 Separately

`key2` should be implemented only after the bug-fix baseline is stable.

Goal:

- Keep `key0` as the default stable path.
- Keep `key1` as the large-contiguous path.
- Add `key2` as a small-shape vector path that avoids unconditional Cube rotate
  matmul setup.

Important structure:

- Dispatch by tiling key before `REGIST_MATMUL_OBJ_STATIC`.
- `key2` must not instantiate or register `TqRotateMatmulOp`.
- Start with a minimal vector rotate path for small shapes.
- Keep CopyIn, normalize, encode, and CopyOut semantics identical first.
- Optimize only after smoke semantics and byte-level behavior are understood.

Validation:

- Run smoke first. Smoke output must not show new repetition, duplicate
  degeneration, or obvious language corruption.
- Compare smoke shape `[2,8,128]` against the fixed baseline.
- Do not use long-query improvement to justify a small-shape path, and do not
  use small-shape improvement to justify a large-shape path.

### 3. Continue Key1 Large-Shape Work

Key1 remains the large-contiguous route. Existing work indicates:

- Brcb normalize V2 is accepted and should remain part of the key1 baseline.
- Batch count is still a major issue: the hot shape currently produces many
  small full-group batches.
- Large-batch UB reduction and batch-loop redesign remain the credible next
  direction, but only after codegen isolation is controlled.

Do not retry rejected designs:

- direct packed encode/copyout with equivalent merge work;
- shared-entry full-class template instantiation that perturbs key0 codegen;
- released `aBatchQue_` slot reuse for the reduce-sum temporary table inside
  the current shared generated object;
- normalize variants that change cache norm semantics or require unsupported
  scalar bf16 conversion.

## Future Rules And Cautions

### Correctness Fixes And Performance Experiments Must Be Separate

Correctness fixes are allowed to regress performance if the old code is unsafe,
but the regression must be measured and recorded. Performance experiments must
not be hidden inside correctness commits.

Each experiment should have:

- a design document before implementation;
- a report after testing;
- clear keep/revert criteria;
- smoke, long query, and OPP records.

### Compare By Shape, Not Overall Average

Do not use overall profiler average as the main comparison. Always compare the
same kernel name and same input shape.

Important shapes:

- Smoke small shape: `[2,8,128]`
- Long hot shape: `[1802,8,128]`
- Secondary long shapes: `[151,8,128]`, `[16,8,128]`

### OPP Is Diagnostic, Not A Total-Time Gate

Use OPP primarily for source attribution:

- function instruction count;
- hot source lines;
- whether a change affects the intended function.

Do not reject or accept a runtime change solely from OPP total task duration.
The torch profiler shape-level kernel timing is the primary performance gate.

### Workspace Reuse Requires A Lifetime Proof

Before reusing any UB buffer across stages, document:

- who owns the buffer;
- when the producer is complete;
- whether async CopyOut still references it;
- whether queue tensors can still be live;
- whether a future pipeline split would invalidate the assumption.

Do not reuse `rotateWorkBuf_` outside rotate/matmul without a new written
lifetime proof.

### Build And Profile Discipline

The debug-line build path currently has a practical caveat: a local untracked
`csrc/fused_infer_attention_score/` tree can make `ASCEND_OP_NAME=ALL`
configuration fail because it expects external CMake helpers. Avoid confusing
that with pack4bit build failures.

For comparable baselines:

- record `git rev-parse HEAD`;
- record `git status --short`;
- use the same model and long-query environment variables;
- keep `XRX_TQ4BIT_LONG_PROMPT_TOKENS=2000`;
- keep `XRX_TQ4BIT_LONG_OUTPUT_TOKENS=8`;
- keep `XRX_TQ4BIT_LONG_MAX_MODEL_LEN=2040`;
- keep `XRX_TQ4BIT_LONG_NUM_PROMPTS=16`;
- keep `XRX_TQ4BIT_GPU_MEMORY_UTILIZATION=0.05`;
- preserve and report the exact profile directory.

### Commit Discipline

Commit only the intended source and documentation files. Do not accidentally
include:

- `mytmp/`
- profiler output
- temporary OPP folders
- unrelated untracked research directories
- generated build artifacts

Commit messages for performance-sensitive changes must include the measured
smoke, long-query, and OPP source-attribution results.
