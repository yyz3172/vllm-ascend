# Experiment Design: key1 Batch Count Attribution

Date: 2026-06-27

## Hypothesis

Several code experiments failed because they changed generated code without
reducing the fundamental amount of work. Before attempting another runtime
change, quantify the batch structure of the long hot shape:

```text
"1802,8,128;1802,8,128;16;128,128;1802;2"
```

The current full-group path has `TQ_MAX_BATCH_M = 64`. With 8 heads, each
full-group batch can cover at most 8 token rows, and each batch is processed
once for key and once for value. If each AIV worker owns about 45 token rows,
then the hot path likely performs around 6 key batches and 6 value batches per
worker. That repeated structure may explain why KFC/matmul orchestration is
visible even though actual Cube compute is small.

## Scope

- No runtime code changes.
- Use route-only baseline numbers and source code formulas.
- If needed, parse existing profiler CSVs and OPPROF source data.
- Produce a report with concrete batch-count estimates and the next runtime
  design recommendation.

## Inputs

Route-only baselines:

- Smoke profile:
  `mytmp/perflog_smoke_route_only_20260627104414`
- Long profile:
  `mytmp/perflog_long_route_only_20260627104641`
- Source OPPROF:
  `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`

Code constants and hot-shape values:

- `TQ_MAX_BATCH_M = 64`
- `TQ_GROUP_ROWS = 4`
- `TQ_AIV_SUB_BLOCKS = 2`
- hot shape: `tokenCount = 1802`, `numHeads = 8`
- route-only OPPROF: `dataCores = 20`, active workers = `40`

## Method

Compute:

```cpp
tokensPerWorker = ceil(tokenCount / activeWorkers)
rawBegin = worker * tokensPerWorker
rawEnd = min(rawBegin + tokensPerWorker, tokenCount)
begin adjustment: if begin falls inside group, back up to group start
end adjustment: if non-last end falls inside group, roll back to group start
rowsPerBatch = floor((batchCapacity / numHeads) / groupRows) * groupRows
```

Then estimate, per worker:

```text
fullGroupRows
batches = ceil(fullGroupRows / rowsPerBatch)
computeBatchCalls = batches * 2  # key + value
rotateCalls = computeBatchCalls
```

Aggregate over all active workers. Also estimate the theoretical effect of
larger effective batches if UB pressure were reduced:

- current rowsPerBatch = 8 token rows for 8 heads.
- hypothetical rowsPerBatch = 16, 24, 32 token rows.

## Commands

Use a small local Python calculation saved in the report. Optionally verify
the active-worker count and hot shape count from:

```bash
python - <<'PY'
import csv, glob, statistics
path = glob.glob(
  'mytmp/perflog_long_route_only_20260627104641/rank0_*_ascend_pt/'
  'ASCEND_PROFILER_OUTPUT/kernel_details.csv')[0]
shape = '"1802,8,128;1802,8,128;16;128,128;1802;2"'
with open(path, newline='') as f:
    rows = list(csv.DictReader(f))
vals = [float(r['Duration(us)']) for r in rows
        if r.get('Name') == 'TurboquantPackKvForCache4bit'
        and r.get('Input Shapes') == shape]
print(len(vals), statistics.mean(vals))
PY
```

## Decision Criteria

The report should choose one next path:

1. If batch count is high and larger batches would materially reduce
   `ComputeBatch`/rotate calls, design a true key1 structural change that
   reduces buffer pressure first. Examples: queue-depth specialization,
   streaming packed output, or stage-split buffers.
2. If batch count is already low, stop pursuing emit/double-buffer structure
   and return to algorithmic encode/normalize changes.
3. If the expected improvement is blocked by key0 codegen pollution, design a
   truly separate key1 kernel entry/build specialization before any more
   runtime changes.

## Reflection Before Running

This experiment is deliberately cheap. The last runtime attempt showed that a
small shared-entry structural edit can cost 8.6% on smoke even when key0 should
not take the changed branch. A numeric batch-count attribution is required
before we spend another build/smoke cycle on a structural patch.
