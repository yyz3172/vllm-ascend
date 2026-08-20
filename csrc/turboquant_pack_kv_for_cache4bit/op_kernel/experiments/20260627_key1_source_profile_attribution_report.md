# Experiment Report: key1 Source Profile Attribution

Date: 2026-06-27

## Conclusion

Accepted as a planning-only experiment. No runtime code was changed.

The latest route-only source profile shows that postprocess-only EncodeBatch
changes are too small to justify more retries. The next code experiment should
target the reduce-body sequence itself, with a single variable that removes a
real repeated vector operation.

## Profile Inputs

- OPPROF: `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`
- Runner result from `tools/op.profile.sh`: `pack4bit_to_cache_avg_us=557.89`
- OpBasicInfo task duration: 287.50 us
- Source extraction command:

```bash
python tools/extract_opprof_source_lines.py \
  mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN \
  --source turboquant_pack_kv_for_cache4bit.cpp \
  --group-by function --top 80 --hot-lines 15
```

## Key Findings

Top local/source-attributed functions:

- Kernel entry `turboquant_pack_kv_for_cache4bit`: 2,887,022 instructions.
  Hot lines include `op.Process()` and `REGIST_MATMUL_OBJ_STATIC`.
- `Process`: 2,173,794 instructions, mostly the inlined call to
  `ProcessCachePairBySequenceContiguousSegments`.
- `ProcessCachePairBySequenceContiguousSegments`: 2,172,188 instructions,
  mostly inlined full-group processing.
- `ProcessSequenceBlockRange`: 2,164,728 instructions, mostly the full-group
  branch.
- `PackCachePairPhysicalFullGroupRun`: 2,157,398 instructions. The hot lines
  are the key and value `PackCachePhysicalFullGroupRunTask` calls.
- `PackCachePhysicalFullGroupRunTask`: 2,148,034 instructions. The hot lines
  are `ComputeBatch` and `CopyOutPhysicalFullGroupRunKnown`.
- `CopyInPhysicalGroupRowsTask`: 1,787,190 instructions.
- `EncodeBatch`: 993,992 instructions.
- `NormalizeBatch`: 703,776 instructions.
- `EncodeQuantCodesByReduceSum`: 450,560 instructions.
- `CopyOutPhysicalFullGroupRunKnown`: 330,920 instructions.
- `MergeEncodedRowToGroup`: 204,800 instructions.

Important line-level detail inside `EncodeQuantCodesByReduceSum`:

- line 609/610: local `rotateWorkBuf_` and first `Brcb`.
- line 615: second `Brcb`.
- line 621/628/630: compare, duplicate ones, select.
- line 638: `WholeReduceSum`.

## Interpretation

The largest top-level counts include inlined callees, so they cannot be read
as independent function costs. The stable local action items are:

- `EncodeQuantCodesByReduceSum` is a real local target, but its postprocess is
  not the main cost.
- `NormalizeBatch` is sizeable, but the Brcb reciprocal rewrite already failed
  the profile gate.
- CopyOut is smaller than Encode/Normalize and prior movement-only attempts
  failed long or smoke gates.
- KFC/matmul setup remains visible in the entry and CANN KFC source lines, but
  previous small-vector no-matmul work regressed repeatability. A KFC
  mitigation needs a structural design, not a scalar-loop fallback.

## Decision

The next code experiment will be a reduce-body single-variable attempt:
remove the per-row `Duplicate(quantOnes, 1.0f, TQ_QUANT_TABLE_ELEMS)` from
`EncodeQuantCodesByReduceSum` by supplying a persistent 1.0 source vector for
`Select`.

This is worth one build/smoke experiment because the source profile attributes
122,880 instructions to the duplicate line in the reduce body, and this change
does not alter thresholds, comparison mode, reduction dimensions, or quantized
output semantics.

## Reflection

This attribution pass prevents continuing the failed direct-i16 postprocess
line. The profile data also shows why previous CopyOut and normalize rewrites
were fragile: each touched a secondary cost without removing the large reduce
body or KFC structure. The next experiment is still modest, but it removes an
actual hot reduce-body operation rather than only changing conversion cleanup.
