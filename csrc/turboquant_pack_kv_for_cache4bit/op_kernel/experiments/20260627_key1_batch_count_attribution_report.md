# Experiment Report: key1 Batch Count Attribution

Date: 2026-06-27

## Summary

Accepted as a planning-only experiment. No runtime code was changed.

The long hot shape is split into many small full-group batches:

- active workers: 40
- current token rows per full-group batch: 8
- full-group key batches across workers: 235
- key/value `ComputeBatch` calls: 470
- key/value rotate calls: 470

This validates that batch count is a major structural issue. However, the
previous emit-unification runtime attempt showed that shared-entry template
specialization can badly regress smoke. The next runtime breakthrough should
not be another equivalent dispatch refactor. It should reduce UB pressure so
`rowsPerBatch` can increase, or create a truly separate key1 kernel path that
does not perturb key0 codegen.

## Inputs

Baseline hot shape:

```text
"1802,8,128;1802,8,128;16;128,128;1802;2"
```

Constants:

- `tokenCount = 1802`
- `numHeads = 8`
- `dataCores = 20`
- `TQ_AIV_SUB_BLOCKS = 2`
- active workers = 40
- `TQ_MAX_BATCH_M = 64`
- `TQ_GROUP_ROWS = 4`
- current `rowsPerBatch = floor((64 / 8) / 4) * 4 = 8`

Raw calculation output:

- `20260627_key1_batch_count_attribution_raw.txt`

Baseline long profile:

- `mytmp/perflog_long_route_only_20260627104641`
- hot shape count 28, avg 820.39 us, p50 820.72 us, p90 822.22 us,
  max 823.88 us.

## Worker Distribution

Each non-final worker owns either 44 or 48 adjusted token rows after group
boundary handling. The final worker owns 10 adjusted rows, with 8 full-group
rows and 2 tail rows.

Batch histogram for key only:

| Key batches per worker | Worker count |
| ---: | ---: |
| 1 | 1 |
| 6 | 39 |

Aggregate:

| Metric | Value |
| --- | ---: |
| adjusted rows sum | 1802 |
| full-group rows sum | 1800 |
| key-only full-group batches | 235 |
| key/value `ComputeBatch` calls | 470 |
| key/value rotate calls | 470 |

## Theoretical Larger-Batch Effect

If UB pressure were reduced enough to increase effective token rows per batch:

| Token rows per batch | Key batches | Key/value compute calls | Calls saved |
| ---: | ---: | ---: | ---: |
| 8 | 235 | 470 | 0 |
| 12 | 157 | 314 | 156 |
| 16 | 118 | 236 | 234 |
| 24 | 79 | 158 | 312 |
| 32 | 79 | 158 | 312 |
| 48 | 40 | 80 | 390 |

The first meaningful target is 16 token rows per batch. It would cut
`ComputeBatch` and rotate calls roughly in half.

## Analysis

The current `TQ_MAX_BATCH_M = 64` is not 64 token rows for the full-group hot
path. Because rows are vector rows and there are 8 heads, the hot path packs
only 8 token rows per batch. That creates 235 key batches and 235 value
batches for one long pack op.

This explains why KFC/matmul orchestration remains visible even though actual
Cube compute time is small. It also explains why equivalent helper extraction
did not help: it did not reduce the 470 `ComputeBatch`/rotate calls.

The theoretical win from larger batches is meaningful, but a constant-only
increase is blocked by UB usage:

- M=64 is about 180 KiB.
- M=96 is about 255 KiB.
- M=128 is about 330 KiB.

Therefore the next runtime optimization must first reduce live UB allocation
or queue depth on the key1 full-group path.

## Reflection

This attribution is a better guide than the last runtime experiment. It shows
where a real structural win would come from: fewer batches, not just a cleaner
loop shape. It also sets a concrete target: make 16 token rows per batch
possible for the `[1802, 8, 128]` family.

Possible next designs:

- key1 full-group queue-depth specialization:
  reduce one or more `TQue` depths from 2 to 1 only after proving the existing
  depth-2 overlap is not needed inside the full-group path, then try a larger
  `TQ_MAX_BATCH_M` for key1 only;
- key1 streaming encode/copyout:
  eliminate or shrink `encodedBatchQue_` for full groups and write packed
  groups directly, freeing enough UB for a larger `x/a/y` batch;
- truly separate key1 kernel entry/build specialization:
  avoid the key0 smoke regression caused by shared-entry template
  instantiation before attempting either of the above.

The strongest candidate is streaming encode/copyout plus larger effective
batch, because it attacks both the batch count and a measured CopyOut/merge
cost. However, previous direct packed encode failed when it did not reduce
batch count. A follow-up design must explicitly pair streaming output with a
larger key1 batch size target.

## Decision

Do not implement another shared-entry emit-unification patch. Add the next
runtime design around key1 UB reduction for larger batches, preferably
streaming full-group encode/copyout with a 16-token-row batch target.
