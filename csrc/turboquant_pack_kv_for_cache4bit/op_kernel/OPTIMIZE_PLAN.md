# TurboQuant 4-bit Pack KV Optimization Plan

## Shape Tiling Strategy

FIA separates shape and feature choices in host tiling before entering the
kernel. TurboQuant 4-bit should follow the same direction, but keep the key
space small:

- `tilingKey = 0`: default path. Keep the current implementation unchanged for
  smoke, small shapes, non-contiguous input strides, and all unproven cases.
- `tilingKey = 1`: large contiguous shape path. Use it only when the shape is
  clearly the long-query hot path and input vectors are physically contiguous.
- `tilingKey = 2`: small-shape vector rotate experiment path. Select it only
  with a conservative host-side predicate; it must not replace `tilingKey = 0`
  as the default route.

The first development step is only to establish the separate tilingKey path.
The large key initially reuses the default implementation so output semantics
and build integration can be verified before adding performance changes.

Large-contiguous selection is intentionally conservative:

```cpp
tokenCount = nVec / numHeads;
contiguous = keyStrideHead == 128 &&
             valueStrideHead == 128 &&
             keyStrideToken == numHeads * 128 &&
             valueStrideToken == numHeads * 128;
large = contiguous &&
        tokenCount >= 128 &&
        nVec >= dataCores * 64 &&
        vecPerCore >= 32;
```

This should route the long hot shape such as `[1802, 8, 128]` to `key=1`,
while non-experimental short queries remain on `key=0`.

Small-vector selection is also intentionally conservative:

```cpp
smallVector = contiguous &&
              tokenCount <= smallTokenThreshold &&
              nVec <= smallVecThreshold;
```

The first implementation should target only the decode/smoke shape family,
then tune thresholds by benchmark. Unknown, mixed, or marginal shapes must stay
on `tilingKey = 0`.

## Large-Key Candidate Optimizations

Once routing is verified, apply optimizations only under `key=1`:

1. Batched vector normalize.
   Previous experiments improved large shape but regressed smoke. Keeping it
   behind `key=1` avoids polluting the small-shape binary path.
2. Large contiguous CopyIn specialization.
   For a contiguous batch, use one `DataCopy` for the whole key/value batch.
   Per sequence, resolve only the first physical slot; subsequent slots are
   continuous.
3. Batch emit unification.
   Collect ranges first, then emit batches through a single flush point. The
   current flush calls are spread across `AppendPhysicalCacheGroupRows()`,
   `ProcessSequenceBlockRange()`, and the final sequence loop.
4. Double buffering after emit unification.
   Split batch execution into `CopyInBatch`, `ComputeBatch`, and `CopyOutBatch`
   so next-batch CopyIn can be issued before current-batch compute/copyout
   completes.

## Key1 Optimization Backlog

`tilingKey = 1` is the large-shape-only route. Keep these items isolated from
`key0` and `key2` until each item passes correctness and target-performance
validation. Use OPP source attribution as a diagnostic, not as a total-time
gate:

1. CopyOut pack/emit specialization.
   This is the next key1 implementation target. Optimize full physical groups
   first so the hot long-query path writes contiguous packed cache groups with
   fewer per-row merge/emit operations. Preserve partial-group
   read-modify-write behavior.
   This must be isolated as a key1 compile-time specialization, not a shared
   helper change:
   - Keep `key0` on the current stable `CopyOutPhysicalFullGroupRunKnown`
     implementation.
   - Instantiate a separate key1 kernel/class variant with a compile-time
     flag such as `ENABLE_LARGE_COPYOUT=true`.
   - Route only `TILING_KEY_IS(1)` to the large CopyOut implementation.
   - Keep `TILING_KEY_IS(0)` instantiated without the large CopyOut code so
     small-shape generated code and instruction layout stay stable.
   - Accept the optimization only if smoke small-shape timing is at parity
     with the route-only baseline while the long hot shape still improves.
   Status 2026-06-27: direct packed encode/copyout was tried and rejected.
   It matched smoke semantics and improved/held smoke timing, but the long hot
   shape regressed slightly from avg 820.39 us to 820.65 us. Moving the merge
   from CopyOut into Encode did not reduce the dominant `ShiftLeft + Or` merge
   or quant encode work. Do not retry this exact direct-pack structure.
2. Batched vector normalize.
   Retry only after CopyOut pack/emit is stable. The previous WholeReduceSum
   experiment hit `ret=507015` in `tools/op.profile.sh`, so this item needs a
   smaller isolated validation step before it can be kept.
   New candidate from `RESEARCH_NORMALIZE_SOFTMAX.md`: key1-only single-loop
   vector reciprocal plus `Brcb` broadcast. Keep the existing per-row
   `Cast -> Mul -> ReduceSum -> Sqrt` sequence, cast/store
   `norms[i * TQ_NORM_STRIDE]` before applying eps, then compute
   `1 / (norm + eps)` in UB with vector ops and broadcast it with `Brcb`.
   Apply normalization with vector `Mul` using a repeated row scale instead of
   `TqSyncVToS() + GetValue() + TqSyncSToV() + Muls(...)`.
   This preserves cache norm semantics while removing the per-row scalar
   engine round trip, and should be tried before larger two-pass rewrites.
   Status 2026-06-28: V1 and V2 were tried; V2 is accepted and restored.
   V2 matched smoke semantics, held/improved the guarded smoke shape, improved
   the long hot shape from avg 820.39 us to avg 817.18 us, and reduced OPP
   `Process()` source instructions from 2,173,794 to 2,149,926. The
   `tools/op.profile.sh` runner/task duration regressed in that run, but those
   overall profiler timings are diagnostic only and are not a hard rejection
   gate. Do not continue small single-loop Brcb variants without a new
   source-profile reason.
   The next concrete attempt is a 32-row batched scalar-sync version of
   `NormalizeBatch` under `key=1` only:
   - For each chunk of up to 32 rows, cast rows into a fp32 temporary block,
     square and reduce each row, then run `Sqrt` on the chunked norm vector.
   - Do one `V->S` sync and read up to 32 norm scalars, then one `S->V` sync.
   - Reuse the saved fp32 rows to apply `1 / (norm + eps)` and cast into
     `aBatch`.
   - Use `rotateWorkBuf_` as the temporary fp32 row block when possible
     (`32 * 128 * sizeof(float)`), because it is not needed until
     `RotateBatchMatmul`.
   - Keep `key0` on the current per-row normalize path until the key1
     experiment passes long and smoke. Use OPP only for source attribution.
   Alternative lower-UB attempt:
   - Use a two-pass `NormalizeBatch`: first compute all row norms, then read
     all reciprocals with one `V->S` / `S->V` pair, then recast rows and apply
     normalization.
   - Do not read reciprocals from `normsBuf_`, because `normsBuf_` stores the
     cache norm after casting to `T`; using it as the normalization denominator
     would change the original fp32 normalization semantics.
   - Store the first-pass fp32 norms in a fp32 UB buffer such as `yFp32Buf_`
     before casting the cache norm into `normsBuf_`. Read scalar reciprocals
     from that fp32 norm buffer.
   - Avoid device-side `memcpy` and scalar `bfloat16_t -> float` conversion in
     `__aicore__` code; previous bf16 scalar casts have failed compilation.
   - In the second pass, prefer `Muls(fp32Row, fp32Row, recip, TQ_PACK_D)` over
     `Duplicate(recipRow) + Mul` unless profiling shows the duplicated-vector
     form is faster.
3. Large contiguous CopyIn review.
   Confirm the current large path already performs one contiguous `DataCopy`
   for full batches. Only add code here if profile data shows remaining
   per-row copy overhead on `key1`.
4. KFC/matmul setup overhead analysis.
   Compare source-level profile data before and after CopyOut changes. If
   `REGIST_MATMUL_OBJ_STATIC` and KFC setup remain dominant, design a separate
   key1-specific mitigation rather than changing `key0`.
   Status 2026-06-27: reduce-body persistent-ones was tried and rejected
   before this item. It preserved smoke output but regressed the guarded smoke
   shape from avg 46.59 us to 47.72 us, and it did not change the expanded
   compare/select/reduce structure. The next experiment is source/profile
   attribution of KFC and rotate matmul setup, not another local Select-source
   shuffle.
   KFC/matmul setup attribution completed:
   `experiments/20260627_kfc_matmul_setup_attribution_report.md`.
   `REGIST_MATMUL_OBJ_STATIC` is about 673,388 instructions and matched
   CANN KFC/matmul/cube/fixpipe functions total about 3,093,324 instructions
   in the refreshed route-only profile. This is meaningful, but not separable
   enough for a registration-only patch. The next step is batch emit
   unification and batch-loop redesign so key1 can reduce repeated
   `ComputeBatch`/rotate issue overhead.
5. Emit unification and double-buffer revisit.
   After CopyOut wins are proven, unify batch emission around one flush point
   and then revisit the standard `CopyIn -> Compute -> CopyOut` loop structure.
   Next design:
   `experiments/20260627_key1_emit_unification_design.md`. First implement a
   key1-only full-group batch planner without true double buffering. Keep key0
   untouched, prove smoke parity, then require long hot-shape improvement
   before keeping runtime code. Only after this passes should `CopyInBatch`,
   `ComputeBatch`, and `CopyOutBatch` stage splitting be attempted.
   Status 2026-06-27: first shared-entry template implementation was rejected
   and reverted. It preserved smoke output but regressed the guarded smoke
   shape from avg 46.59 us to 50.59 us. This shows that adding a second full
   class instantiation in the same kernel body perturbs key0 codegen too much.
   Do not retry this exact shared-entry template-flag structure. A future
   attempt needs a truly separate key1 kernel entry/build specialization or a
   runtime change large enough to reduce batch count/stage overhead and justify
   the risk.
   Batch-count attribution completed:
   `experiments/20260627_key1_batch_count_attribution_report.md`.
   For the long hot shape, current `rowsPerBatch` is only 8 token rows because
   `TQ_MAX_BATCH_M=64` is divided across 8 heads. This produces 235 key
   batches and 470 key/value `ComputeBatch`/rotate calls across 40 active
   workers. A 16-token-row effective batch would reduce key/value compute calls
   to 236. The next credible runtime design is key1 UB reduction for larger
   full-group batches, not another equivalent emit helper.
   Next design:
   `experiments/20260627_key1_larger_batch_ub_reduction_design.md`.
   The target is 128 vector rows for key1 full groups, i.e. 16 token rows at
   8 heads, by reducing live UB first. Do not start with a constant-only batch
   increase or another shared-entry template split.
   Status 2026-06-27: stopped before runtime implementation. The generated
   artifacts package both key0 and key1 symbols in the same dtype object, and
   the previous shared-entry template probe already regressed smoke from
   46.59 us to 50.59 us. A larger UB rewrite is blocked until key1 codegen can
   be isolated, or a much smaller probe proves that a specific key1-only source
   pattern does not perturb key0.
   Codegen isolation attribution completed:
   `experiments/20260627_key1_codegen_isolation_report.md`. There is no
   explicit per-key compile macro in the generated wrapper; CANN emits both
   tiling-key symbols into one dtype object. The next safe runtime step is only
   a minimal no-op key1 isolation probe using `if constexpr (TILING_KEY_IS(1))`
   without instantiating a second full class.
   Minimal no-op probe completed:
   `experiments/20260627_key1_noop_isolation_probe_report.md`. A tiny
   `if constexpr (TILING_KEY_IS(1))` helper preserved smoke output and improved
   smoke timing within noise (avg 45.57 us vs 46.59 us baseline). This coding
   style is viable; the rejected pattern is second full-class instantiation.

Key1 acceptance requirements:

- Long hot shape latency must improve versus the kept route-only baseline:
  `"1802,8,128;1802,8,128;16;128,128;1802;2"` avg 820.39 us.
- Smoke output must not introduce new duplicate/repeated semantics.
- Smoke small-shape timing must not regress; small shapes should stay on
  `key0` unless an accepted `key2` path explicitly replaces them.
- `tools/build_debug_perf.sh`, smoke, long query, and `tools/op.profile.sh`
  must all pass before committing key1 code changes.

## Small-Vector No-Matmul Plan

The small-shape regression risk is dominated by fixed overhead. The current
kernel registers the Cube rotate matmul unconditionally:

```cpp
REGIST_MATMUL_OBJ_STATIC(&pipe, GetSysWorkSpacePtr(), rotateMm,
                         (TCubeTiling*)nullptr);
```

For tiny `M`, this startup cost can be larger than the rotate work itself.
`tilingKey = 2` should therefore enter a separate vector-only implementation
and return before creating or registering `TqRotateMatmulOp`.

Planned structure:

- Keep `tilingKey = 0` as the stable default Cube path.
- Keep `tilingKey = 1` as the large contiguous Cube path.
- Add `tilingKey = 2` for the small-vector path.
- Dispatch by tiling key before matmul registration. Only keys `0` and `1`
  should instantiate/register the Cube rotate object.
- Reuse existing CopyIn, normalize, encode, and CopyOut semantics where
  practical. Limit duplication to the rotate implementation first; refactor
  common code only after correctness and performance are proven.
- Implement rotate as vector code for small `M`:
  `normalized_row[128] * rotation_t[128,128] -> rotated_row[128]`.
  Prefer fp32 accumulation if available; if fp16/bf16 accumulation is used,
  smoke output and quantized cache bytes must be treated as correctness gates.

Validation gates for `tilingKey = 2`:

- Build with `tools/build_debug_perf.sh`.
- Run smoke and verify generated text has no new repetition/duplicate
  semantics compared with the stable baseline.
- Record the small pack shape timing, especially
  `"2,8,128;2,8,128;16;128,128;2;3"`.
- Run long query to prove large-shape routing and latency are unchanged; long
  should route to `tilingKey = 1`, not `tilingKey = 2`.
- Run `tools/op.profile.sh` after rebuilding with `tools/build_debug_perf.sh`.
  If the default profile workload does not hit `tilingKey = 2`, add a small
  pack-only runner or profile command before accepting the change.

## Guardrails

- Smoke output must not show duplicates or semantic drift.
- Smoke latency must not regress versus the current key-0 baseline.
- Long hot shape `[1802, 8, 128]` must improve before any optimization is kept.
- Run `tools/build_debug_perf.sh`, smoke, long query, and `tools/op.profile.sh`
  for changes that alter the large-key compute path.
- Do not keep a small-vector experiment unless both correctness and the small
  pack latency improve versus the stable `tilingKey = 0` baseline.

## Research Reference

- [RESEARCH_NORMALIZE_SOFTMAX.md](RESEARCH_NORMALIZE_SOFTMAX.md):
  FIAS Online Softmax vs TQ Pack NormalizeBatch 高性能设计对比。
  涵盖 Brcb 替代 GetValue+Muls、BlockReduce 替代 ReduceSum+GetValue、
  Cube/Vec 流水线等 7 项手法，以及推荐方案 B（Brcb+Div 两循环拆分）
  仅用于 key=1 的实施约束和验证步骤。

## 2026-06-27 Development Notes

Implemented and kept:

- Host tiling now routes only conservative large contiguous shapes to
  `tilingKey = 1`.
- Kernel dispatch builds both `tilingKey = 0` and `tilingKey = 1`.
- `tilingKey = 1` currently reuses the stable default compute path. This is
  intentional scaffolding for large-shape-only optimization.
- `tilingKey = 2` is reserved for the future small-vector no-matmul path;
  `tilingKey = 0` remains the default stable path.

Build notes:

- `KERNEL_TASK_TYPE` must be selected before `GET_TILING_DATA` for multi-key
  dynamic compile.
- `TurboquantPackKvForCache4bit_1` needs its own same-layout tiling data class;
  registering the default tiling data class for both keys generated class
  redefinition errors.
- `TILING_KEY_IS` should use numeric literals in this CANN setup.

Validation on the kept route-only state:

- `tools/build_debug_perf.sh`: pass.
- Smoke:
  - log: `mytmp/tq4bit_smoke_route_only_20260627104414.log`
  - profile: `mytmp/perflog_smoke_route_only_20260627104414`
  - output matches the previous route-only smoke text. It still contains the
    known baseline `of of` repetition, so it is not evidence of a new
    optimization regression.
  - `TurboquantPackKvForCache4bit` shape
    `"2,8,128;2,8,128;16;128,128;2;3"`:
    count 56, avg 46.59 us, p50 46.44 us, p90 47.86 us, max 48.72 us.
- Long query:
  - log: `mytmp/tq4bit_long_route_only_20260627104641.log`
  - profile: `mytmp/perflog_long_route_only_20260627104641`
  - fixed workload: 16 prompts, 2000 input tokens, 8 output tokens,
    max model len 2040, gpu memory utilization 0.05.
  - hot shape `"1802,8,128;1802,8,128;16;128,128;1802;2"`:
    count 28, avg 820.39 us, p50 820.72 us, p90 822.22 us, max 823.88 us.
- `tools/op.profile.sh` after rollback:
  - OPPROF: `mytmp/OPPROF_20260627103532_KOSJXSSXUAWLJSTX`
  - debug source check: `insight_source_status=ok`
  - default prefill pack runner: `pack4bit_to_cache_avg_us=503.47`.

Experiment history:

- Tried key1-only batched vector normalize with `WholeReduceSum`.
- Smoke and long query were roughly neutral:
  - smoke pack avg 46.48 us.
  - long hot shape avg 818.85 us.
- `tools/op.profile.sh` on default key1 pack-only failed with
  `aclrtSynchronizeStream repeat failed, ret=507015`.
- The experiment was fully reverted. Do not revive this path without first
  building a smaller standalone normalize kernel or adding a runner-level
  correctness guard for the exact large key1 shape.
- Tried full-group CopyOut run emit with one multi-block `DataCopyPad` per
  head on the large full-group path.
  - Build passed with `tools/build_debug_perf.sh`.
  - Flex large pack-only improved to about 794 us.
  - Long hot shape improved from 820.39 us to 795.18 us.
  - Smoke small shape regressed twice:
    - first run avg 47.94 us.
    - rerun avg 49.68 us.
    - route-only baseline avg 46.59 us.
  - The experiment was reverted because small-shape regression is not
    acceptable. A future CopyOut attempt must be key1-isolated in both code
    path and generated binary behavior, or prove smoke timing parity before
    running long/profile.
- Retried CopyOut run emit as a key1-only branch inside
  `CopyOutPhysicalFullGroupRunKnown`.
  - Build passed with `tools/build_debug_perf.sh`.
  - Smoke output stayed semantically consistent with the route-only baseline.
  - Smoke small shape:
    - first profiled run avg 47.20 us.
    - rerun avg 46.69 us.
    - route-only baseline avg 46.59 us.
  - Long hot shape avg 818.43 us, only a marginal change from the route-only
    baseline avg 820.39 us and far from the earlier non-isolated 795.18 us
    result.
  - The experiment was reverted because it did not provide a meaningful large
    shape win and still carried small-shape timing risk.
- Tried a key1-only two-pass `NormalizeBatch` that stores fp32 norms in
  `yFp32Buf_`, performs one batched `V->S` scalar read, then recasts rows for
  normalization.
  - Build passed with `tools/build_debug_perf.sh`.
  - Smoke output stayed semantically consistent with the route-only baseline.
  - Smoke small shape regressed to avg 47.69 us versus route-only avg
    46.59 us.
  - Long hot shape avg 819.72 us versus route-only avg 820.39 us, so the
  large-shape benefit was negligible.
  - The experiment was reverted. The extra second-pass row cast costs about as
    much as the saved scalar synchronizations for the current hot shape.
- Tried key1-only single-loop Brcb normalize.
  - Design: `experiments/20260627_key1_brcb_normalize_design.md`
  - Report: `experiments/20260627_key1_brcb_normalize_report.md`
  - V1 used two half-row `Mul` calls after `Brcb`; V2 replaced them with one
    repeat-strided `Mul`.
  - V2 smoke output matched the route-only semantic baseline and small-shape
    timing stayed at parity:
    - first run avg 46.5589 us.
    - rerun avg 46.3486 us.
    - route-only baseline avg 46.59 us.
  - V2 long hot shape improved slightly to avg 817.18 us versus route-only
    avg 820.39 us.
  - `tools/op.profile.sh` overall timing regressed:
    - route-only refreshed runner 557.89 us, task duration 287.50 us.
    - V2 runner 581.54 us, task duration 293.62 us.
  - OPP source attribution for
    `TurboquantPackKVForCache4bitToCache.Process()` improved from 2,173,794
    to 2,149,926 instructions.
  - The experiment is accepted and restored. `Div + Brcb + repeat Mul`
    removed scalar `GetValue`; normalize-local instructions stayed flat, but
    smoke/long validation and `Process()` instruction attribution support
    keeping the change.
- Tried key1 direct packed encode/copyout for physical full-group runs.
  - Design: `experiments/20260627_key1_direct_packed_encode_design.md`
  - Report: `experiments/20260627_key1_direct_packed_encode_report.md`
  - Build passed with `tools/build_debug_perf.sh`.
  - Smoke output matched the route-only text exactly.
  - Smoke small shape:
    - first run avg 45.71 us.
    - rerun avg 46.36 us.
    - route-only baseline avg 46.59 us.
  - Long hot shape regressed slightly to avg 820.65 us versus route-only avg
    820.39 us.
  - The experiment was reverted without `op.profile.sh` because the long-query
    gate failed. The result shows that removing `encodedBatch` as an
    intermediate is insufficient; a future CopyOut attempt must reduce the
    actual nibble merge operations or GM write count.
- Tried direct `float -> int16` postprocess in the reduce-sum EncodeBatch
  branch.
  - Design: `experiments/20260627_key1_encode_direct_i16_design.md`
  - Report: `experiments/20260627_key1_encode_direct_i16_report.md`
  - Build passed with `tools/build_debug_perf.sh`.
  - Smoke output matched the route-only text exactly.
  - Smoke small shape regressed twice:
    - first run avg 47.22 us.
    - rerun avg 47.79 us.
    - route-only baseline avg 46.59 us.
  - The experiment was reverted without running long/profile because the smoke
    performance gate failed. The direct cast is buildable and semantically
    viable, but the first version was not key1-isolated and likely perturbed
    the key0 small-shape generated code.
- Tried the first `tilingKey = 2` small-vector no-matmul implementation.
  - Host routing was conservative: contiguous input, `tokenCount <= 2`,
    `numHeads <= 8`, and `nVec <= 16`.
  - Kernel skipped `REGIST_MATMUL_OBJ_STATIC` for key2 and used vector
    rotate with fp32 accumulation.
  - Build passed with `tools/build_debug_perf.sh`.
  - Smoke output matched the route-only baseline text exactly.
  - Smoke small shape `"2,8,128;2,8,128;16;128,128;2;3"`:
    - first run avg 45.50 us.
    - rerun avg 48.24 us.
    - route-only baseline avg 46.59 us.
  - The experiment was reverted because the speedup was not stable and the
    second run regressed. Do not retry this scalar-loop vector rotate design
    without a different implementation that avoids per-row scalar reads and
    proves repeatable smoke parity.
- Retried key1 full-group CopyOut run emit with one multi-block `DataCopyPad`
  per head.
  - Build had already passed with `tools/build_debug_perf.sh`.
  - Smoke output matched the route-only baseline text, including the known
    baseline `of of` repetition.
  - Smoke small shape `"2,8,128;2,8,128;16;128,128;2;3"` regressed to avg
    47.10 us, p50 47.20 us, p90 48.88 us, max 51.36 us versus the route-only
    baseline avg 46.59 us.
  - The experiment was reverted without running long/profile because
    small-shape regression is unacceptable.

Latest source-level profile hints:

- Top repo lines for route-only key1 are dominated by:
  - `REGIST_MATMUL_OBJ_STATIC` / KFC setup in the kernel entry.
  - `CopyOutPhysicalGroupRowsKnown` slot iteration and async packed-group
    writes.
  - `MergeEncodedRowToGroup` shift/or pack operations.
  - `EncodeBatch` per-row quant encode.
- Existing large contiguous path already uses one contiguous `DataCopy` for
  physical full-group input batches. The next implementation attempt should
  focus on reducing CopyOut pack work or reducing KFC/matmul setup frequency,
  not on another normalize-only rewrite.

Next implementation order:

1. Direct i16 postprocess is closed. V1 and key1-isolated V2 both preserved
   smoke text but failed smoke performance. Do not retry postprocess-only
   conversion/mask cleanup.
2. Implement the reduce-body persistent-ones experiment:
   `experiments/20260627_key1_reduce_select_ones_design.md`. The refreshed
   source profile shows `EncodeBatch` at about 993,992 instructions and
   `EncodeQuantCodesByReduceSum` at about 450,560 instructions; this
   experiment removes a measured hot reduce-body duplicate instead of only
   changing postprocess conversion.
3. If the reduce-body experiment fails, re-check route-only source profile
   around `REGIST_MATMUL_OBJ_STATIC` and KFC setup. The fixed setup cost is
   visible but a no-matmul path must avoid replacing Cube work with scalar
   loops.
4. After an Encode or KFC win is proven, retry batched vector normalize only on
   `key=1`,
   starting with the single-loop vector reciprocal plus `Brcb` scheme from
   `RESEARCH_NORMALIZE_SOFTMAX.md`. If that fails to compile or regresses,
   fall back to the 32-row chunked scalar-sync scheme. Treat the previous
   `ret=507015` failure as a release blocker until isolated.
5. Revisit batch emit unification and true CopyIn/Compute/CopyOut loop
   structure after the key1 CopyOut path shows a measurable win.
