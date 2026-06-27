# TurboQuant 4-bit Pack KV Optimization Plan

## Shape Tiling Strategy

FIA separates shape and feature choices in host tiling before entering the
kernel. TurboQuant 4-bit should follow the same direction, but keep the key
space small:

- `tilingKey = 0`: default path. Keep the current implementation unchanged for
  smoke, small shapes, non-contiguous input strides, and all unproven cases.
- `tilingKey = 1`: large contiguous shape path. Use it only when the shape is
  clearly the long-query hot path and input vectors are physically contiguous.

The first development step is only to establish the separate tilingKey path.
The large key initially reuses the default implementation so output semantics
and build integration can be verified before adding performance changes.

Large-contiguous selection is intentionally conservative:

```cpp
tokenCount = nVec / numHeads;
contiguous = keyStrideHead == 128 &&
             valueStrideHead == 128 &&
             keyStrideToken == numHeads * 128 &&
             valueStrideToken == numHeads * 128;
large = contiguous &&
        tokenCount >= 128 &&
        nVec >= dataCores * 64 &&
        vecPerCore >= 32;
```

This should route the long hot shape such as `[1802, 8, 128]` to `key=1`,
while smoke and short queries remain on `key=0`.

## Large-Key Candidate Optimizations

Once routing is verified, apply optimizations only under `key=1`:

1. Batched vector normalize.
   Previous experiments improved large shape but regressed smoke. Keeping it
   behind `key=1` avoids polluting the small-shape binary path.
2. Large contiguous CopyIn specialization.
   For a contiguous batch, use one `DataCopy` for the whole key/value batch.
   Per sequence, resolve only the first physical slot; subsequent slots are
   continuous.
3. Batch emit unification.
   Collect ranges first, then emit batches through a single flush point. The
   current flush calls are spread across `AppendPhysicalCacheGroupRows()`,
   `ProcessSequenceBlockRange()`, and the final sequence loop.
4. Double buffering after emit unification.
   Split batch execution into `CopyInBatch`, `ComputeBatch`, and `CopyOutBatch`
   so next-batch CopyIn can be issued before current-batch compute/copyout
   completes.

## Guardrails

- Smoke output must not show duplicates or semantic drift.
- Smoke latency must not regress versus the current key-0 baseline.
- Long hot shape `[1802, 8, 128]` must improve before any optimization is kept.
- Run `tools/build_debug_perf.sh`, smoke, long query, and `tools/op.profile.sh`
  for changes that alter the large-key compute path.

## 2026-06-27 Development Notes

Implemented and kept:

- Host tiling now routes only conservative large contiguous shapes to
  `tilingKey = 1`.
- Kernel dispatch builds both `tilingKey = 0` and `tilingKey = 1`.
- `tilingKey = 1` currently reuses the stable default compute path. This is
  intentional scaffolding for large-shape-only optimization.

Build notes:

- `KERNEL_TASK_TYPE` must be selected before `GET_TILING_DATA` for multi-key
  dynamic compile.
- `TurboquantPackKvForCache4bit_1` needs its own same-layout tiling data class;
  registering the default tiling data class for both keys generated class
  redefinition errors.
- `TILING_KEY_IS` should use numeric literals in this CANN setup.

Validation on the kept route-only state:

- `tools/build_debug_perf.sh`: pass.
- Smoke:
  - log: `mytmp/tq4bit_smoke_route_only_20260627104414.log`
  - profile: `mytmp/perflog_smoke_route_only_20260627104414`
  - output matches the previous route-only smoke text. It still contains the
    known baseline `of of` repetition, so it is not evidence of a new
    optimization regression.
  - `TurboquantPackKvForCache4bit` shape
    `"2,8,128;2,8,128;16;128,128;2;3"`:
    count 56, avg 46.59 us, p50 46.44 us, p90 47.86 us, max 48.72 us.
- Long query:
  - log: `mytmp/tq4bit_long_route_only_20260627104641.log`
  - profile: `mytmp/perflog_long_route_only_20260627104641`
  - fixed workload: 16 prompts, 2000 input tokens, 8 output tokens,
    max model len 2040, gpu memory utilization 0.05.
  - hot shape `"1802,8,128;1802,8,128;16;128,128;1802;2"`:
    count 28, avg 820.39 us, p50 820.72 us, p90 822.22 us, max 823.88 us.
- `tools/op.profile.sh` after rollback:
  - OPPROF: `mytmp/OPPROF_20260627103532_KOSJXSSXUAWLJSTX`
  - debug source check: `insight_source_status=ok`
  - default prefill pack runner: `pack4bit_to_cache_avg_us=503.47`.

Rejected experiment:

- Tried key1-only batched vector normalize with `WholeReduceSum`.
- Smoke and long query were roughly neutral:
  - smoke pack avg 46.48 us.
  - long hot shape avg 818.85 us.
- `tools/op.profile.sh` on default key1 pack-only failed with
  `aclrtSynchronizeStream repeat failed, ret=507015`.
- The experiment was fully reverted. Do not revive this path without first
  building a smaller standalone normalize kernel or adding a runner-level
  correctness guard for the exact large key1 shape.

Latest source-level profile hints:

- Top repo lines for route-only key1 are dominated by:
  - `REGIST_MATMUL_OBJ_STATIC` / KFC setup in the kernel entry.
  - `CopyOutPhysicalGroupRowsKnown` slot iteration and async packed-group
    writes.
  - `MergeEncodedRowToGroup` shift/or pack operations.
  - `EncodeBatch` per-row quant encode.
- Existing large contiguous path already uses one contiguous `DataCopy` for
  physical full-group input batches. The next implementation attempt should
  focus on reducing CopyOut pack work or reducing KFC/matmul setup frequency,
  not on another normalize-only rewrite.
