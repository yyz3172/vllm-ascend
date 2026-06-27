# Experiment Report: key1 Direct Packed Encode/CopyOut

Date: 2026-06-27

## Conclusion

Rejected and reverted.

The implementation built successfully and smoke output matched the route-only
baseline exactly. Smoke small-shape pack latency improved or stayed at parity,
but the required long-query hot shape did not improve:

- Route-only long hot shape avg: 820.39 us.
- Direct packed encode/copyout long hot shape avg: 820.65 us.

Because the target large shape regressed slightly, the code was reverted. The
design and this report are kept as the experiment record.

## Design Reference

- Design: `20260627_key1_direct_packed_encode_design.md`
- Scope: `tilingKey = 1`, physical full-group run only.
- Intended change: pack full cache groups directly during Encode, then make
  CopyOut only emit already-packed groups.

## Baseline

Kept route-only state:

- Smoke log: `mytmp/tq4bit_smoke_route_only_20260627104414.log`
- Smoke profile: `mytmp/perflog_smoke_route_only_20260627104414`
- Smoke small pack shape
  `"2,8,128;2,8,128;16;128,128;2;3"`:
  count 56, avg 46.59 us, p50 46.44 us, p90 47.86 us, max 48.72 us.
- Long log: `mytmp/tq4bit_long_route_only_20260627104641.log`
- Long profile: `mytmp/perflog_long_route_only_20260627104641`
- Long hot shape
  `"1802,8,128;1802,8,128;16;128,128;1802;2"`:
  count 28, avg 820.39 us, p50 820.72 us, p90 822.22 us, max 823.88 us.
- Refreshed `op.profile.sh` for source comparison:
  `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`,
  runner 557.89 us, task duration 287.50 us.

## Implementation Summary

The experiment added a key1-only full-group path:

- `ComputeBatchToPackedFullGroups`
- `EncodeBatchToPackedFullGroups`
- `MergeQuantIndexToDirectPackedGroup`
- `CopyOutPackedFullGroupRunKnown`

The new path kept the same quantization operations and the same
`ShiftLeft + Or` nibble merge semantics, but avoided the intermediate
`encodedBatch` queue for physical full groups. It packed rows into
`packedRowBuf_` during Encode and then emitted those groups directly to GM.

Partial groups, indexed fallback, and `tilingKey = 0` stayed on the original
path.

## Validation Results

Build:

- Command: `tools/build_debug_perf.sh`
- Log: `mytmp/build_key1_direct_packed_encode_20260627150947.log`
- Result: pass.

Smoke run 1:

- Log: `mytmp/tq4bit_smoke_key1_direct_packed_encode_20260627151436.log`
- Profile: `mytmp/perflog_smoke_key1_direct_packed_encode_20260627151436`
- Output: exactly matched route-only smoke text, including the known baseline
  `of of` repetition.
- Small pack shape:
  count 56, avg 45.71 us, p50 45.68 us, p90 47.10 us, max 48.60 us.

Smoke rerun:

- Log: `mytmp/tq4bit_smoke_key1_direct_packed_encode_rerun_20260627151721.log`
- Profile:
  `mytmp/perflog_smoke_key1_direct_packed_encode_rerun_20260627151721`
- Output: exactly matched route-only smoke text.
- Small pack shape:
  count 56, avg 46.36 us, p50 46.16 us, p90 47.92 us, max 48.66 us.

Long query:

- Log: `mytmp/tq4bit_long_key1_direct_packed_encode_20260627151949.log`
- Profile: `mytmp/perflog_long_key1_direct_packed_encode_20260627151949`
- Workload:
  - prompts: 16
  - prompt tokens: 2000
  - output tokens: 8
  - max model len: 2040
  - gpu memory utilization: 0.05
- Hot shape
  `"1802,8,128;1802,8,128;16;128,128;1802;2"`:
  count 28, avg 820.65 us, p50 820.10 us, p90 822.46 us, max 823.96 us.

`tools/op.profile.sh` was not run because the long-query gate failed.

## Failure Analysis

The design removed the `encodedBatch` intermediate from the full-group key1
path, but did not remove the dominant work:

- Quant encode still runs the same `EncodeQuantCodesByReduceSum` path.
- Full-group nibble packing still performs the same per-row shift/or merge for
  rows 1..3.
- Norms still use scalar `GetValue/SetValue` stores.

The main mechanical saving was eliminating the per-row `DataCopy` into
`encodedBatch` plus later reads from `encodedBatch`. On smoke this reduced the
small pack shape, likely because the shorter batch has more sensitivity to
queue/intermediate overhead. On the long hot shape, the saved intermediate
movement was too small compared with normalize, rotate matmul, quant encode,
and the still-present merge work.

There is also a scheduling risk: the old path separates Encode and CopyOut by
`encodedBatchQue_`, while the direct path writes `packedRowBuf_` during Encode
and emits from the same buffer later. Although the implementation waited before
reusing packed slots, the changed buffer lifetime and queue structure did not
translate into better large-shape pipeline behavior.

## Reflection

This was the best first version of the direct-pack idea because it minimized
semantic risk:

- It kept the existing quantization implementation.
- It kept the same nibble pack arithmetic.
- It changed only the physical full-group key1 path.
- It verified exact smoke text equality before long-query measurement.

The experiment also clarified an important point: simply moving CopyOut packing
earlier is not enough. A future CopyOut attempt needs to reduce the actual
`ShiftLeft + Or` merge cost or reduce the number of GM writes, not just remove
the `encodedBatch` queue. Candidate follow-ups should be treated as separate
experiments:

- Find a vector pack primitive or layout trick that packs four uint4 rows with
  fewer vector ops than three `ShiftLeft + Or` merges.
- Change Encode to produce already shifted row fragments only if it removes
  operations rather than moving them.
- Revisit source profile after any Encode optimization, because Encode remains
  larger than CopyOut on the large profile.

## Final State

The code was reverted to the route-only implementation. No runtime
optimization from this experiment is kept.
