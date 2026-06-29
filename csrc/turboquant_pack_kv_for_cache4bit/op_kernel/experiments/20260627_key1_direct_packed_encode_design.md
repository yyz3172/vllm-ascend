# Experiment Design: key1 Direct Packed Encode/CopyOut

Date: 2026-06-27

## Hypothesis

The route-only source profile shows that the large full-group path still spends
measurable work in CopyOut packing:

- `CopyOutPhysicalFullGroupRunKnown`: 348,840 instructions.
- `MergeEncodedRowToGroup`: 206,848 instructions on the hot merge line.
- `InitFullGroupFromRow0`: 45,056 instructions.
- `EncodeBatch` also writes every encoded row to `encodedBatch` before CopyOut
  reads it back and merges rows into the final packed group layout.

For the large contiguous physical full-group path, row/group/head order is
known. We can remove the `encodedBatch` intermediate for that path by packing
directly into `packedRowBuf_` while encoding each row, then making CopyOut only
emit already-packed cache groups.

Expected benefit:

- Remove `encodedBatchQue_` allocation/enqueue/dequeue/free for full-group
  key1 batches.
- Remove the per-row `DataCopy(encodedBatch, argminIndexU16, 128)` in
  `EncodeBatch` for this path.
- Avoid re-reading encoded rows from `encodedBatch` during CopyOut.
- Keep the same `ShiftLeft + Or` semantics initially, so correctness risk is
  lower than changing the nibble pack algorithm.

## Scope

- Enable only for `TILING_KEY_IS(1)`.
- Enable only inside `PackCachePhysicalFullGroupRunTask`, where:
  - `rowCount` is a multiple of `TQ_GROUP_ROWS`.
  - `firstGroupRow == 0` by construction of the caller.
  - cache groups are full and `preserveExisting == false`.
- Keep `tilingKey = 0` on the current `ComputeBatch + CopyOutPhysicalFullGroupRunKnown`
  path.
- Keep partial-group and indexed fallback paths unchanged.

## Source Profile Basis

Route-only refreshed OPPROF:

- OPPROF: `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`
- Runner: `pack4bit_to_cache_avg_us=557.89`
- OpBasicInfo task duration: 287.50 us.
- `REGIST_MATMUL_OBJ_STATIC`: about 673k instructions. This is fixed startup
  cost, but smaller than the cumulative Process/CopyIn/Encode/CopyOut path.
- `EncodeQuantCodesByReduceSum`: 848,112 instructions; this remains dominant.
- Full-group CopyOut is still a meaningful local target because previous
  non-isolated CopyOut experiments showed large-shape sensitivity.

## Implementation Sketch

Add a key1-only compute/copyout path:

```cpp
PackCachePhysicalFullGroupRunTask(...) {
  CopyInPhysicalGroupRowsTask(..., m);
  if constexpr (TILING_KEY_IS(1)) {
    ComputeBatchToPackedFullGroups(m, rowCount, headCount);
    CopyOutPackedFullGroupRunKnown(packedGm, rowCount, blockIdx,
                                   firstGroupInBlock, headStart, headCount);
  } else {
    ComputeBatch(m);
    CopyOutPhysicalFullGroupRunKnown(...);
  }
}
```

`ComputeBatchToPackedFullGroups`:

1. `NormalizeBatch(m)`.
2. `RotateBatchMatmul(m, 0, TQ_PACK_D)`.
3. `EncodeBatchToPackedFullGroups(rowCount, headCount)`.

`EncodeBatchToPackedFullGroups`:

- Dequeue `yBatch`.
- Use the same quantization implementation as `EncodeBatch`.
- For each encoded row `i`:
  - `logicalRow = i / headCount`
  - `headOff = i - logicalRow * headCount`
  - `groupOff = logicalRow / TQ_GROUP_ROWS`
  - `groupRow = logicalRow - groupOff * TQ_GROUP_ROWS`
  - `groupSlot = groupOff * headCount + headOff`
- For `groupRow == 0`, copy the unshifted 4-bit index words into the packed
  group index area and store norm0.
- For `groupRow > 0`, shift `argminIndexU16` by `groupRow * 4` and `Or` into
  the packed group index area, then store the corresponding norm.
- Do not allocate or enqueue `encodedBatch`.

`CopyOutPackedFullGroupRunKnown`:

- Iterate `groupOff/headOff` in the same order as the packed slots.
- Use existing `copy_packed_ub_to_gm_async`.
- Preserve existing `packedWritePending_` / `packedGroupSlot_` waiting logic,
  or use a local slot count if direct-packed slots do not need the rotating
  writer buffer.

## Semantic Requirements

- Final physical cache group layout must remain:
  - index words: row0 bits 0..3, row1 bits 4..7, row2 bits 8..11,
    row3 bits 12..15.
  - norm words: row0..row3 appended after the 256B index group.
- Norm bits must match the existing `EncodeBatch` path, using the first word
  of each `normsBuf_[i * TQ_NORM_STRIDE]`.
- Partial group read-modify-write behavior must remain on the old path.
- Smoke text must match the route-only semantic baseline. The known baseline
  `of of` repetition is not a new failure only if unchanged.

## Risks

- This path still uses `ShiftLeft + Or`; if CopyOut is not the actual limiting
  factor, the win may be too small.
- Writing packed groups during Encode may increase live UB pressure and reduce
  scheduling freedom.
- Norm storage currently uses scalar `GetValue/SetValue`; changing it in the
  first version would add correctness risk, so V1 keeps scalar norm stores.
- Previous CopyOut variants showed that small-shape timing can move even when
  the code is intended for key1. This experiment must rerun smoke and reject
  any small-shape regression.

## Validation Gates

1. Build with `tools/build_debug_perf.sh`.
2. Run smoke with profile enabled.
   - Output must match route-only semantics.
   - Small pack shape
     `"2,8,128;2,8,128;16;128,128;2;3"` must not regress from avg 46.59 us.
3. If smoke passes, run the long-query workload.
   - Hot shape
     `"1802,8,128;1802,8,128;16;128,128;1802;2"` must improve from avg
     820.39 us.
4. If long improves, run `tools/op.profile.sh`.
   - Must improve versus refreshed route-only runner 557.89 us and task
     duration 287.50 us.
5. Keep the code only if all gates pass. Otherwise revert code and write a
   report explaining the failure.
