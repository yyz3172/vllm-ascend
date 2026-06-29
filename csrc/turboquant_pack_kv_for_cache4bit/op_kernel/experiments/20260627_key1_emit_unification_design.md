# Experiment Design: key1 Emit Unification Before Double Buffering

Date: 2026-06-27

## Hypothesis

The current full-group hot path enters `ComputeBatch` once per key batch and
once per value batch:

```cpp
PackCachePairPhysicalFullGroupRun
  while consumedRows < rowCount:
    PackCachePhysicalFullGroupRunTask(key, rowsThisBatch)
      CopyInPhysicalGroupRowsTask
      ComputeBatch
      CopyOutPhysicalFullGroupRunKnown
    PackCachePhysicalFullGroupRunTask(value, rowsThisBatch)
      CopyInPhysicalGroupRowsTask
      ComputeBatch
      CopyOutPhysicalFullGroupRunKnown
```

For the long hot shape with 8 heads and `TQ_MAX_BATCH_M = 64`, each full-group
batch covers at most 8 token rows. KFC attribution shows that the rotate/KFC
orchestration cost is visible, while actual Cube compute is small. Reducing or
regularizing batch emission is therefore a better next target than another
single-line encode/normalize micro-optimization.

Hypothesis: move full-group key1 emission to one explicit batch planner that
first computes the current-core physical range, then iterates batches from one
place. This should make batch counts measurable and prepare the code for a
later `CopyIn -> Compute -> CopyOut` pipeline without changing key0 or partial
group semantics.

## Scope

Target only `tilingKey = 1` large contiguous full-group runs.

Do not change:

- `tilingKey = 0` default/smoke route.
- partial group read-modify-write behavior.
- group boundary ownership rules:
  - if a core `begin` falls inside a cache group, process the complete group.
  - if a non-last core `end` falls inside a cache group, roll back to that
    group start and do not process the group.
  - the final core may process the final partial group.
- quantization, normalization, rotate math, or packed cache layout.

## Proposed First Runtime Experiment

Implement a key1-only planner for physical full-group runs:

1. Add a compile-time key1 specialization flag to the kernel class if needed,
   keeping key0 generated code unchanged.
2. In the key1 path, keep current range splitting and boundary semantics, but
   route full groups through a single function such as
   `ProcessKey1FullGroupRunBatches`.
3. The new function should compute batch metadata once:

```cpp
rowsPerBatch = floor(batchCapacity / numHeads_) / TQ_GROUP_ROWS * TQ_GROUP_ROWS;
rowsPerBatch = max(rowsPerBatch, TQ_GROUP_ROWS);
batchCount = ceil(fullRows / rowsPerBatch);
```

4. Iterate batches in one place and call key/value packing from there. The
   first version may still execute key then value serially per batch; it is
   accepted only if it preserves or improves performance. The purpose is to
   remove scattered flush decisions and prepare for stage splitting.
5. Add source-visible helper structure or local counters only if they compile
   away or are used for reportable reasoning. Do not add GM debug output.

The first implementation must not try true double buffering. Double buffering
requires separate `CopyInBatch`, `ComputeBatch`, and `CopyOutBatch` APIs and is
the follow-up experiment after emission is centralized.

## Why Not Just Increase `TQ_MAX_BATCH_M`

Current approximate UB usage:

| Max batch M | Approx UB bytes | Approx KiB |
| ---: | ---: | ---: |
| 64 | 184,352 | 180.0 |
| 96 | 261,152 | 255.0 |
| 128 | 337,952 | 330.0 |

The major buffers scale with `M`: `xBatchQue_`, `aBatchQue_`, `yBatchQue_`,
`encodedBatchQue_`, `normsBuf_`, and `rotateWorkBuf_`. A constant-only batch
increase is therefore not a safe experiment. Larger effective batches require
buffer lifetime changes, queue-depth changes, or a pipeline rewrite.

## Expected Effect

If this experiment is successful, it should:

- keep smoke small-shape generated behavior stable by isolating the change to
  key1;
- keep long hot shape output semantics unchanged;
- make `PackCachePairPhysicalFullGroupRun` and `ProcessSequenceBlockRange`
  easier to reason about;
- reduce repeated branch/flush overhead, or at least prove that the current
  per-batch structure is already optimal.

The likely direct speedup is modest unless the compiler can simplify the new
centralized loop. The main value is to create the correct structure for the
next double-buffer experiment.

## Validation Gates

1. Build with `tools/build_debug_perf.sh`.
2. Run smoke with profile enabled.
   - Output must match route-only baseline exactly.
   - Guarded small shape
     `"2,8,128;2,8,128;16;128,128;2;3"` must not regress from avg 46.59 us.
3. If smoke passes, run fixed long query.
   - Hot shape
     `"1802,8,128;1802,8,128;16;128,128;1802;2"` must improve from avg
     820.39 us to be kept as a runtime change.
4. If long improves, run `tools/op.profile.sh`.
   - Must not regress route-only refreshed profile:
     `pack4bit_to_cache_avg_us=557.89`, task duration 287.50 us.
5. If any gate fails, revert runtime code and keep the design/report.

## Report Requirements

The report must include:

- build/smoke/long/profile commands and log paths;
- smoke output comparison;
- small and long pack shape timing;
- estimated batch count for the hot shape before and after;
- whether KFC/matmul source attribution changed;
- a reflection on whether the experiment reached the best version of emit
  unification or whether a follow-up stage-split design is still justified.

## Reflection Before Coding

This is intentionally conservative. The codebase already has multiple flush
sites (`AppendPhysicalCacheGroupRows`, `ProcessSequenceBlockRange`, final
sequence flush, and the full-group direct task). Trying to add double
buffering before centralizing emission would make correctness and performance
regressions hard to attribute. A successful first step is a clearer key1
batch-loop shape with no small-shape cost.
