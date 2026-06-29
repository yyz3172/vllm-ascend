# Experiment Design: KFC/Rotate Matmul Setup Attribution

Date: 2026-06-27

## Hypothesis

The route-only profile shows visible source-line instructions at kernel entry
and around the Cube rotate matmul path:

- `REGIST_MATMUL_OBJ_STATIC`
- `op.Process()`
- `RotateBatchMatmul`
- `SetOrgShape`, `SetSingleShape`, `SetTensorA`, `SetTensorB`,
  `SetLocalWorkspace`, `IterateAll`, and `End`

The fixed KFC/matmul setup cost may explain why small shapes are sensitive to
code layout and why previous no-matmul attempts could sometimes improve the
first smoke run. However, the large hot shape may be dominated by per-row
normalize/encode/copyout work instead of setup. This experiment quantifies
which case is true before designing another runtime change.

## Scope

- No runtime code changes.
- Use the refreshed route-only OPPROF:
  `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`.
- Extract source-line instruction counts for:
  - kernel entry lines around matmul registration and `op.Process()`.
  - `RotateBatchMatmul`.
  - top function-level groups for context.
  - relevant CANN/KFC source entries if present in the same OPPROF data.
- Do not run long/smoke again because this is an attribution-only experiment
  against an already accepted route-only profile.

## Measurement Commands

Function-level context:

```bash
python tools/extract_opprof_source_lines.py \
  mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN \
  --source turboquant_pack_kv_for_cache4bit.cpp \
  --group-by function --top 80 --hot-lines 20
```

Kernel entry setup lines:

```bash
python tools/extract_opprof_source_lines.py \
  mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN \
  --source turboquant_pack_kv_for_cache4bit.cpp \
  --line-start 1775 --line-end 1818
```

Rotate matmul body lines:

```bash
python tools/extract_opprof_source_lines.py \
  mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN \
  --source turboquant_pack_kv_for_cache4bit.cpp \
  --line-start 462 --line-end 492
```

All-source scan for KFC/Cube/Matmul names:

```bash
python tools/extract_opprof_source_lines.py \
  mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN \
  --top 200 --hot-lines 5 | rg -i 'kfc|matmul|cube|iterate|kernel'
```

## Decision Criteria

This experiment can produce one of three conclusions:

1. KFC setup is a large fixed cost and separable from per-row work.
   - Next step: design a key2/small-shape route that avoids matmul
     registration, but only with a vector implementation that is repeatable
     and not scalar-loop dominated.
2. KFC setup is visible but not dominant for the long hot shape.
   - Next step: move to emit unification/double-buffer planning, because the
     larger costs are in repeated per-batch pack work.
3. `RotateBatchMatmul` issue/end lines dominate but setup lines do not.
   - Next step: design a key1-specific way to reduce the number of rotate
     calls or increase rows per rotate batch, not a registration-only change.

## Semantic Risks

None. This experiment only reads profile data.

## Reflection Before Running

Earlier key2 vector rotate skipped `REGIST_MATMUL_OBJ_STATIC` and was not
stable enough to keep. Therefore a profile result that merely shows KFC setup
is visible is not sufficient. The report must identify a concrete mechanism
that can improve repeatably; otherwise the plan should move to batch emit
unification rather than revisiting no-matmul.
