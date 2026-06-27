# Experiment Report: key1 UB Reduction For Larger Full-Group Batches

Date: 2026-06-27

## Summary

Stopped before runtime implementation.

The design requires key1 codegen isolation before attempting a large UB/batch
rewrite. The build artifacts show that the current dynamic-op generation does
not create a separate object per tiling key. It creates one object per dtype,
and each object contains both key0 and key1 kernel symbols. This matches the
observed smoke regression from the previous shared-entry template attempt.

No runtime code was changed for this experiment.

## Precheck Performed

Inspected the latest debug build artifacts under:

```text
mytmp/build_libcust_opapi_nodebug_debug_line/binary/ascend910b
```

Relevant files:

```text
bin/turboquant_pack_kv_for_cache4bit/
  TurboquantPackKvForCache4bit_c294064959ba0f8c3a82d6096cf4ca4d.o
  TurboquantPackKvForCache4bit_cf5c5d6d4ad385ca5cf1f7b534e9a391.o

gen/
  TurboquantPackKvForCache4bit-turboquant_pack_kv_for_cache4bit-0.sh
  TurboquantPackKvForCache4bit-turboquant_pack_kv_for_cache4bit-1.sh
```

The two `.o` files correspond to dtype variants:

- `c294...`: bfloat16 inputs.
- `cf5c...`: float16 inputs.

Each dtype object contains both tiling-key kernels:

```text
..._0_mix_aic
..._0_mix_aiv
..._1_mix_aic
..._1_mix_aiv
```

The generated JSON also lists both tiling keys inside the same object:

```json
"kernelList": [
  {"tilingKey": 0, "kernelName": "..._0"},
  {"tilingKey": 1, "kernelName": "..._1"}
]
```

## Interpretation

The current build does provide separate key0/key1 symbols, but not separate
object-level codegen or packaging. Adding a large key1-only class body can
still move key0 code layout and object-level behavior. This is not theoretical:
the previous emit-unification V1 patch only added a key1 template branch, yet
the smoke key0 shape regressed from 46.59 us to 50.59 us.

The larger-batch UB reduction design would be much more invasive than that
failed V1. It would likely add more key1-only code, larger buffers, and new
full-group encode/copyout logic. Without stronger isolation, it is very likely
to fail at the smoke gate before proving anything about long-shape batch
reduction.

## Decision

Do not implement the larger-batch runtime patch in the current shared
object/codegen structure.

The next prerequisite is a codegen isolation design:

- either split key1 into a genuinely separate op/kernel object;
- or prove a narrow use of `TILING_KEY_IS(1)` that does not perturb key0 with a
  small smoke-only probe before adding any UB/batch changes.

## Reflection

The batch-count attribution still points to the right performance target:
reduce key/value `ComputeBatch` calls from 470 toward 236 by doubling token rows
per batch. The issue is not the target; it is the implementation route.

The current experiment prevented a high-risk patch that would probably have
been rejected on smoke. That is useful progress, but the performance work is
blocked on a build/codegen strategy rather than on kernel math.
