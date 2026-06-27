# Experiment Design: key1 Codegen Isolation Prerequisite

Date: 2026-06-27

## Hypothesis

Repeated experiments show that key1-intended source changes can regress the
key0 smoke shape even when smoke should not execute the changed branch. The
current generated artifacts contain key0 and key1 symbols in the same dtype
object, so object-level layout and generated code may still be coupled.

Before any larger key1 runtime change, determine what isolation mechanisms are
available:

- whether `TILING_KEY_IS(1)` is a compile-time constant inside generated key1
  symbols;
- whether unused branches are eliminated per key symbol;
- whether separate source files, wrapper functions, or build registration can
  produce independent key1 objects;
- whether there is a small source pattern that keeps key0 smoke timing stable.

## Scope

- No runtime behavior changes.
- Inspect generated source, generated scripts, JSON metadata, and object
  symbols from the restored route-only debug build.
- Do not run smoke/long unless a runtime probe is explicitly designed after
  this attribution.

## Inputs

Restored route-only build:

- `mytmp/build_restore_route_only_after_emit_failure_20260627170822.log`
- generated source:
  `mytmp/build_libcust_opapi_nodebug_debug_line/binary/ascend910b/src/turboquant_pack_kv_for_cache4bit/turboquant_pack_kv_for_cache4bit.cpp`
- generated objects:
  `mytmp/build_libcust_opapi_nodebug_debug_line/binary/ascend910b/bin/turboquant_pack_kv_for_cache4bit/*.o`

## Method

1. Inspect generated scripts and JSON:
   - confirm object granularity by dtype vs tiling key;
   - confirm `kernelList` entries for tiling key 0 and 1.
2. Inspect generated source around `TILING_KEY_IS` expansion.
3. Inspect object symbols and function sizes for `_0_mix_*` and `_1_mix_*`.
4. Compare whether a key1-only source edit can be isolated by:
   - `if constexpr (TILING_KEY_IS(1))`;
   - normal `if (TILING_KEY_IS(1))`;
   - helper templates instantiated only inside key1 branch;
   - separate top-level functions selected by tiling key.

## Expected Outcomes

Possible conclusions:

1. Per-key compile-time elimination is strong enough, but the previous patch
   used a source structure that still bloated key0. Then the next runtime
   probe should be a smaller `if constexpr (TILING_KEY_IS(1))` helper with no
   second full-class instantiation.
2. Per-key elimination is not strong enough inside one object. Then the next
   task is build-system isolation: separate op name/object for key1 or a
   dedicated large-shape op.
3. The generated structure is too opaque to prove isolation statically. Then
   the next step is a minimal no-op key1 source probe with smoke profiling
   only; if it regresses, stop runtime key1 work until build isolation exists.

## Reflection Before Running

The larger-batch target is valid, but key0 smoke regressions are now the main
blocker. This experiment should prevent another expensive build/smoke cycle
that is rejected before reaching long-shape performance.
