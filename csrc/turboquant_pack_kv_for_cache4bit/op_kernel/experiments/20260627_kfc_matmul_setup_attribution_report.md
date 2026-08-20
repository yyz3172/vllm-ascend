# Experiment Report: KFC/Rotate Matmul Setup Attribution

Date: 2026-06-27

## Summary

Accepted as a planning-only experiment. No runtime code was changed.

KFC/matmul setup is real and measurable, but a registration-only optimization
is not the right next implementation. The route-only OPPROF shows:

- `REGIST_MATMUL_OBJ_STATIC`: 673,388 instructions, about 2.60% of all
  source-attributed instructions, estimated 7.46 us of the 287.50 us task.
- `RotateBatchMatmul`: 78,662 instructions, about 0.30%, estimated 0.87 us.
- all matched CANN KFC/matmul/cube/fixpipe functions together: 3,093,324
  instructions, about 11.92%, estimated 34.27 us.

The hot path is therefore not just the explicit rotate body. The larger issue
is repeated mixed AIC/AIV orchestration plus per-batch fixed cost. The next
code experiment should not try another scalar no-matmul rotate or a
registration-only patch. It should first redesign batch emission so key1 can
reduce batch count and eventually form a cleaner `CopyIn -> Compute ->
CopyOut` loop.

## Inputs

Route-only OPPROF:

- `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`
- `tools/op.profile.sh` runner result: `pack4bit_to_cache_avg_us=557.89`
- `OpBasicInfo.csv` task duration: 287.50 us

Generated attribution files:

- `profile_20260627170000_kfc_matmul_setup/function_top80.txt`
- `profile_20260627170000_kfc_matmul_setup/kernel_entry_1775_1818.txt`
- `profile_20260627170000_kfc_matmul_setup/rotate_matmul_462_492.txt`
- `profile_20260627170000_kfc_matmul_setup/all_source_top200.txt`
- `profile_20260627170000_kfc_matmul_setup/all_source_kfc_matmul_filter.txt`
- `profile_20260627170000_kfc_matmul_setup/all_function_top300.txt`
- `profile_20260627170000_kfc_matmul_setup/all_function_kfc_matmul_filter.txt`
- `profile_20260627170000_kfc_matmul_setup/all_function_top500.json`

## Commands

```bash
python tools/extract_opprof_source_lines.py \
  mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN \
  --source turboquant_pack_kv_for_cache4bit.cpp \
  --group-by function --top 80 --hot-lines 20
```

```bash
python tools/extract_opprof_source_lines.py \
  mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN \
  --source turboquant_pack_kv_for_cache4bit.cpp \
  --line-start 1775 --line-end 1818
```

```bash
python tools/extract_opprof_source_lines.py \
  mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN \
  --source turboquant_pack_kv_for_cache4bit.cpp \
  --line-start 462 --line-end 492
```

```bash
python tools/extract_opprof_source_lines.py \
  mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN \
  --group-by function --top 300 --hot-lines 8
```

## Key Measurements

Kernel entry source lines:

| Line | Meaning | Instructions |
| ---: | --- | ---: |
| 1790 | `REGIST_MATMUL_OBJ_STATIC` | 673,388 |
| 1808 | `op.Init(...)` | 12,760 |
| 1817 | `op.Process()` including inlined work | 2,174,074 |
| 1818 | kernel exit/destructors | 17,220 |

Rotate body source lines:

| Line | Meaning | Instructions |
| ---: | --- | ---: |
| 482 | `SetOrgShape` | 13,760 |
| 483 | `SetSingleShape` | 2,080 |
| 484 | `SetTensorA` | 8,800 |
| 485 | `SetTensorB` | 2,080 |
| 489 | `IterateAll` | 21,622 |
| 490 | `End` | 13,280 |

Function-level context from the same profile:

| Function | Instructions | Est. us |
| --- | ---: | ---: |
| `EncodeBatch` | 982,512 | 10.89 |
| `NormalizeBatch` | 698,816 | 7.74 |
| `EncodeQuantCodesByReduceSum` | 364,768 | 4.04 |
| `CopyOutPhysicalFullGroupRunKnown` | 348,840 | 3.87 |
| `RotateBatchMatmul` | 78,662 | 0.87 |

Matched CANN KFC/matmul/cube/fixpipe functions:

| Group | Instructions | Profile % | Est. us |
| --- | ---: | ---: | ---: |
| all matched functions | 3,093,324 | 11.92 | 34.27 |
| `kernel_kfc.h::Run` | 568,084 | 2.19 | 6.29 |
| `matmul_server.h::MATMUL_POLICY_DEFAULT_OF` | 250,128 | 0.96 | 2.77 |
| `kernel_kfc.h::RunAux` | 229,220 | 0.88 | 2.54 |
| `matmul_server_impl_c220.h::Iterate` | 200,876 | 0.77 | 2.23 |
| `kfc_comm_server.h::RcvMessage` | 191,768 | 0.74 | 2.12 |
| `kfc_comm.h::RcvMessageImpl` | 191,248 | 0.74 | 2.12 |

Pipe utilization supports the same picture:

- AIC cube time per active cube block is only about 0.85-1.23 us.
- AIC scalar time is around 200-270 us on most cube blocks.
- AIV vector time is around 158 us or 210 us depending on sub-block work.

This means the current mixed kernel spends little time in actual Cube compute
for this rotate; orchestration, scalar, and vector work dominate.

## Batch Size Check

The current maximum local batch is `TQ_MAX_BATCH_M = 64`. On the long hot
shape with 8 heads, one full-group batch covers at most 8 token rows for key
and then 8 token rows for value.

Rough UB allocation from the current buffers:

| Max batch M | Approx UB bytes | Approx KiB |
| ---: | ---: | ---: |
| 64 | 184,352 | 180.0 |
| 96 | 261,152 | 255.0 |
| 128 | 337,952 | 330.0 |

Therefore simply increasing `TQ_MAX_BATCH_M` is not a safe first code
experiment. The queues and work buffers scale with M, and 96/128 rows risk
exceeding practical UB capacity. Larger batches require structural buffer
changes, not a constant-only patch.

## Analysis

`REGIST_MATMUL_OBJ_STATIC` is a hot line, but it is paid once per kernel
invocation. The long hot path still spends much more in repeated per-batch
work. The local `RotateBatchMatmul` body is small compared with
`NormalizeBatch`, `EncodeBatch`, and CopyOut. The CANN/KFC total is sizeable
only after including framework/server-side functions and message handling.

This explains why previous no-matmul experiments were unstable: skipping
registration can help small shapes, but replacing Cube rotate with scalar
vector loops introduces enough repeated work and layout risk to lose the
benefit. It also explains why postprocess-only encode and Select-source
experiments are weak: they do not reduce batch count, KFC messages, or the
main vector normalization/quantization bodies.

The better direction is to reduce how often the full sequence path enters
`ComputeBatch` and its rotate issue/end sequence, or to restructure the batch
loop so CopyIn/Compute/CopyOut stages are explicit and can be overlapped.

## Reflection

This experiment answers the immediate question: KFC/matmul overhead is large
enough to matter, but not isolated enough for a small registration-only fix.
The current implementation also cannot safely get larger batches by changing
`TQ_MAX_BATCH_M`, because UB usage grows from about 180 KiB at M=64 to about
255 KiB at M=96.

The next experiment should be a design-first batch emit unification for key1.
It must first quantify batch counts and preserve full-group semantics. Only
after that should it change runtime code. If the design cannot reduce batch
count or buffer pressure, we should return to algorithmic encode/normalize
changes rather than keep adjusting KFC setup.

## Decision

No runtime changes were made. Proceed to key1 emit unification and batch-loop
redesign planning.
