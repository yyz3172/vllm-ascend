# TurboQuant 4-bit Pack KV Experiment Protocol

Date: 2026-06-27

This directory records single-variable optimization experiments for
`turboquant_pack_kv_for_cache4bit.cpp`. Every experiment must leave a design
document before coding and a report after measurement, regardless of whether
the code is kept or reverted.

## Current Todo Order

1. `key1` CopyOut pack/emit specialization.
   - Goal: reduce full-group pack/emit cost on the large contiguous path.
   - Status: direct packed encode/copyout rejected and reverted. Report:
     `20260627_key1_direct_packed_encode_report.md`.
2. `key1` NormalizeBatch Brcb/vector reciprocal.
   - Goal: remove per-row `TqSyncVToS() + GetValue() + TqSyncSToV()` from
     large-shape normalize without changing cache norm semantics.
   - Status: rejected and reverted. Report:
     `20260627_key1_brcb_normalize_report.md`.
3. `key1` large contiguous CopyIn review.
   - Goal: prove whether full-batch contiguous CopyIn is already optimal.
   - Status: source review shows physical full-group path already does one
     contiguous `DataCopy` when all heads are contiguous.
4. `key1` EncodeBatch quantization postprocess.
   - Goal: reduce instructions after `EncodeQuantCodesByReduceSum` by
     removing avoidable conversion/mask steps before writing encoded rows.
   - Status: V1 direct i16 rejected and reverted. Design:
     `20260627_key1_encode_direct_i16_design.md`. Report:
     `20260627_key1_encode_direct_i16_report.md`. V2 key1-isolated direct
     i16 also rejected and reverted. Design:
     `20260627_key1_encode_direct_i16_v2_design.md`. Report:
     `20260627_key1_encode_direct_i16_v2_report.md`. Stop
     postprocess-only direct-i16 experiments.
5. Source-profile attribution.
   - Goal: decide the next target from refreshed route-only OPPROF instead
     of continuing postprocess micro-optimizations.
   - Status: completed. Design:
     `20260627_key1_source_profile_attribution_design.md`. Report:
     `20260627_key1_source_profile_attribution_report.md`.
6. `key1` reduce-body persistent ones.
   - Goal: remove per-row `Duplicate(quantOnes, 1.0f, 2048)` in
     `EncodeQuantCodesByReduceSum` by selecting from a persistent 1.0 table.
   - Status: rejected and reverted. It preserved smoke output but regressed
     the guarded smoke shape from avg 46.59 us to 47.72 us. Design:
     `20260627_key1_reduce_select_ones_design.md`. Report:
     `20260627_key1_reduce_select_ones_report.md`.
7. KFC/matmul setup overhead analysis.
   - Goal: quantify and reduce `REGIST_MATMUL_OBJ_STATIC` and KFC setup cost
     without affecting `key0`.
   - Status: completed as a planning-only attribution experiment. Design:
     `20260627_kfc_matmul_setup_attribution_design.md`. Report:
     `20260627_kfc_matmul_setup_attribution_report.md`.
8. Emit unification and double-buffer redesign.
   - Goal: collect all current-core ranges first, then drive a single
     `CopyIn -> Compute -> CopyOut` batch loop.
   - Status: first shared-entry key1 template implementation rejected and
     reverted. It preserved smoke output but regressed the guarded smoke shape
     from avg 46.59 us to 50.59 us. Design:
     `20260627_key1_emit_unification_design.md`. Report:
     `20260627_key1_emit_unification_report.md`. Do not retry the same
     shared-entry template-flag structure.
9. `key2` small-shape no-matmul redesign.
   - Goal: avoid small-shape Cube startup overhead with a better vector path.
   - Status: previous scalar-loop vector rotate rejected; do not retry without
     a materially different implementation.
10. `key1` batch-count attribution.
    - Goal: quantify current long-shape batch count and theoretical value of
      reducing `ComputeBatch`/rotate calls before attempting another runtime
      structure change.
    - Status: completed as a planning-only experiment. Current long hot shape
      uses 235 key batches and 470 key/value `ComputeBatch`/rotate calls.
      Design: `20260627_key1_batch_count_attribution_design.md`. Report:
      `20260627_key1_batch_count_attribution_report.md`.
11. `key1` UB reduction for larger full-group batches.
    - Goal: make 16 token rows per full-group batch possible by reducing live
      UB allocation, then validate whether fewer batches improves long shape.
    - Status: stopped before runtime implementation. Current generated
      artifacts package key0/key1 symbols in the same dtype object, so a large
      key1-only UB rewrite is too likely to perturb key0 smoke timing. Design:
      `20260627_key1_larger_batch_ub_reduction_design.md`. Report:
      `20260627_key1_larger_batch_ub_reduction_report.md`.
12. `key1` codegen isolation prerequisite.
    - Goal: find a way to isolate key1 code changes from key0 smoke codegen
      before any larger-batch runtime patch.
    - Status: completed as a planning-only experiment. Current CANN dynamic
      op generation creates one object per dtype containing both key0/key1
      symbols. Design: `20260627_key1_codegen_isolation_design.md`. Report:
      `20260627_key1_codegen_isolation_report.md`.
13. `key1` minimal no-op isolation probe.
    - Goal: verify whether a tiny `if constexpr (TILING_KEY_IS(1))` source
      pattern can avoid key0 smoke regression before larger key1 runtime work.
    - Status: completed and runtime probe reverted. It preserved smoke output
      and did not regress the guarded smoke shape. Design:
      `20260627_key1_noop_isolation_probe_design.md`. Report:
      `20260627_key1_noop_isolation_probe_report.md`.
14. `key1` incremental larger-batch prerequisite.
    - Goal: use only small `if constexpr (TILING_KEY_IS(1))` helpers to reduce
      UB/batch cost incrementally, avoiding second full-class instantiation.
    - Status: next design target.

## Required Workflow

For each experiment:

1. Write a design document.
   - State hypothesis, target path, expected affected shape, semantic risks,
     implementation sketch, and validation gates.
2. Implement only the current experiment.
   - Avoid stacking unvalidated changes.
   - Keep `key0` stable unless the experiment explicitly targets `key0`.
3. Run strict validation.
   - Build with `tools/build_debug_perf.sh`.
   - Run smoke and compare output semantics and small-shape pack timing against
     the kept route-only baseline.
   - Run long query only if smoke passes.
   - Run `tools/op.profile.sh` only if long query improves or the experiment
     specifically targets profile-level evidence.
4. Decide keep or revert.
   - Keep code only if it improves the target baseline and does not regress
     guarded shapes.
   - Revert code if correctness changes, smoke regresses, long does not
     improve meaningfully, build fails, or `op.profile.sh` hits `ret=507015`.
5. Write an experiment report.
   - Include commands/log paths, measured numbers, keep/revert conclusion,
     root-cause analysis, and reflection on whether the implementation was the
     best version of the idea.
   - If there is a credible improvement to the idea, add the next attempt to
     `OPTIMIZE_PLAN.md` before running it.

## Baselines

Kept route-only baseline:

- Commit: `a63ec1ba perf(turboquant): add large pack tiling route`
- Smoke small shape
  `"2,8,128;2,8,128;16;128,128;2;3"`: avg 46.59 us, p50 46.44 us,
  p90 47.86 us, max 48.72 us.
- Long hot shape
  `"1802,8,128;1802,8,128;16;128,128;1802;2"`: avg 820.39 us,
  p50 820.72 us, p90 822.22 us, max 823.88 us.
- `tools/op.profile.sh`: `mytmp/OPPROF_20260627103532_KOSJXSSXUAWLJSTX`,
  `pack4bit_to_cache_avg_us=503.47`.
- Refreshed route-only `tools/op.profile.sh` for source comparison:
  `mytmp/OPPROF_20260627134444_LLPYGPYEZKODALFN`,
  `pack4bit_to_cache_avg_us=557.89`, task duration 287.50 us.
