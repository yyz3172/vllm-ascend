# Experiment Report: key1 Codegen Isolation Prerequisite

Date: 2026-06-27

## Summary

Accepted as a planning-only experiment. No runtime code was changed.

The restored route-only build confirms that key0 and key1 are not compiled as
separate objects. Each dtype object contains both tiling-key symbols. The
generated Python wrapper also does not pass an explicit `-DTILING_KEY=<key>`
compile option; it calls `compile_op(...)` once for the dtype/op and CANN
generates both key symbols from the same source.

This means key1 changes can still perturb key0 codegen unless the source
structure is extremely constrained. The previous shared-entry template probe
was too large and regressed smoke. The next safe runtime probe, if any, must
be a minimal key1-only no-op/source-shape probe using `if constexpr
(TILING_KEY_IS(1))`, not a second full class instantiation.

## Inputs

Restored route-only build:

- `mytmp/build_restore_route_only_after_emit_failure_20260627170822.log`

Generated files inspected:

- `mytmp/build_libcust_opapi_nodebug_debug_line/binary/ascend910b/src/turboquant_pack_kv_for_cache4bit/TurboquantPackKvForCache4bit.py`
- `mytmp/build_libcust_opapi_nodebug_debug_line/binary/ascend910b/src/turboquant_pack_kv_for_cache4bit/turboquant_pack_kv_for_cache4bit.cpp`
- `mytmp/build_libcust_opapi_nodebug_debug_line/binary/ascend910b/bin/turboquant_pack_kv_for_cache4bit/*.o`
- `mytmp/build_libcust_opapi_nodebug_debug_line/binary/ascend910b/bin/turboquant_pack_kv_for_cache4bit/*.json`

## Findings

Generated objects:

```text
TurboquantPackKvForCache4bit_c294064959ba0f8c3a82d6096cf4ca4d.o  # bf16
TurboquantPackKvForCache4bit_cf5c5d6d4ad385ca5cf1f7b534e9a391.o  # fp16
```

Each object contains both key0 and key1 symbols:

```text
..._0_mix_aic
..._0_mix_aiv
..._1_mix_aic
..._1_mix_aiv
```

Route-only symbol sizes are identical between key0 and key1 for AIV:

| Object | Symbol | Size |
| --- | --- | ---: |
| bf16 | `_0_mix_aiv` | `0x3cee4` |
| bf16 | `_1_mix_aiv` | `0x3cee4` |
| fp16 | `_0_mix_aiv` | `0x3cee4` |
| fp16 | `_1_mix_aiv` | `0x3cee4` |

The generated JSON lists both tiling keys inside each dtype object:

```json
"kernelList": [
  {"tilingKey": 0, "kernelName": "..._0"},
  {"tilingKey": 1, "kernelName": "..._1"}
]
```

The generated wrapper builds from one source and does not add a per-key macro:

```python
origin_func_name = "turboquant_pack_kv_for_cache4bit"
compile_op(src, origin_func_name, op_info, options, code_channel, '{}',
           {'valueDepend': {}})
```

## Interpretation

The current dynamic op machinery gives separate key symbols but not separate
dtype objects or separate per-key compile options. That is enough for normal
multi-key dispatch, but it is not strong isolation for large source changes.

The route-only key0/key1 symbol sizes being identical is expected because the
source body is identical. The previous emit-unification V1 added a second full
class instantiation and a shared-entry runtime branch. That likely changed
object layout and generated key0 code enough to move the smoke path from
46.59 us to 50.59 us.

## Decision

Do not proceed directly to the larger-batch UB rewrite.

The next experiment should be a minimal key1 no-op isolation probe:

- design first;
- add a tiny `if constexpr (TILING_KEY_IS(1))` helper that should compile only
  into key1;
- do not instantiate a second full pack class;
- build and run smoke only;
- keep no runtime code unless smoke timing remains at parity.

If even this minimal probe regresses smoke, key1 performance work must wait
for a separate op/object build strategy.

## Reflection

This explains why several key1-intended changes failed at the smoke gate. The
problem is not only algorithmic; it is also build/codegen coupling. The next
probe is intentionally small so we can distinguish "all key1 source changes
pollute key0" from "the previous template split was too large".
