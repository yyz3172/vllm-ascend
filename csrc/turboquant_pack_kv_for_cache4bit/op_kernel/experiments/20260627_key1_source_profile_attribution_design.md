# Experiment Design: key1 Source Profile Attribution

Date: 2026-06-27

## Hypothesis

Several recent code experiments failed because they targeted small local
operations while the long hot shape is dominated by a mix of KFC/matmul setup,
batch traversal, normalize, encode, and CopyOut. Before the next code change,
we need a source-profile attribution pass that separates:

- wrapper/inlined call-site counts;
- actual local functions that can be changed in this operator;
- CANN KFC/matmul runtime lines that need a structural mitigation rather than
  a local vector-op rewrite.

## Scope

- No runtime code changes.
- Use the kept route-only debug-line profile:
  `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`.
- Extract source-line function attribution with:

```bash
python tools/extract_opprof_source_lines.py \
  mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN \
  --source turboquant_pack_kv_for_cache4bit.cpp \
  --group-by function --top 80 --hot-lines 15
```

- Cross-check against `OpBasicInfo.csv` and `PipeUtilization.csv`.

## Validation Criteria

This experiment is accepted if it produces a clear next implementation target
with enough evidence to avoid another postprocess-only micro-optimization.

## Expected Decision Points

- If `EncodeBatch` remains large and the reduce body is a meaningful local
  fraction, prioritize a reduce-body experiment that removes a repeated vector
  operation.
- If KFC/matmul setup dominates and local encode is too small, prioritize a
  structural matmul-call reduction experiment.
- If CopyOut is no longer a major fraction, do not retry CopyOut movement
  without a new primitive-level packing idea.
