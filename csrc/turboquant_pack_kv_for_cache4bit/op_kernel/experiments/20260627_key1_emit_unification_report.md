# Experiment Report: key1 Emit Unification Before Double Buffering

Date: 2026-06-27

## Summary

Rejected and reverted.

The first emit-unification implementation preserved smoke output semantics,
but it severely regressed the guarded small shape from 46.59 us to 50.59 us
average. The change was intentionally not taken to long query or
`tools/op.profile.sh`.

## Implementation Tested

The tested runtime patch:

- added a compile-time template flag
  `ENABLE_KEY1_EMIT_UNIFICATION` to
  `TurboquantPackKVForCache4bitToCache`;
- instantiated the class with `false` for `TILING_KEY_IS(0)` and `true` for
  `TILING_KEY_IS(1)`;
- added a key1-only `PackCachePairPhysicalFullGroupRunUnified` helper;
- routed full-group runs through that helper only when the key1 flag was true;
- left Normalize, Rotate, Encode, CopyOut, partial-group behavior, and cache
  layout unchanged.

The helper was semantically equivalent to the original full-group loop. It did
not increase batch size or implement true double buffering.

## Commands And Logs

Initial build attempt:

- command: `bash tools/build_debug_perf.sh`
- log: `mytmp/build_key1_emit_unification_20260627164942.log`
- result: failed during CMake configure because the untracked
  `csrc/fused_infer_attention_score/` directory was picked up by
  `ASCEND_OP_NAME=ALL` and its copied CMake files require
  `add_op_to_compiled_list`.

Build after temporarily moving the untracked FIA directory out of `csrc/`:

```bash
bash tools/build_debug_perf.sh
```

- log: `mytmp/build_key1_emit_unification_20260627165049.log`
- result: pass.
- the FIA directory was restored after the build and was not modified.

Smoke semantic-only run with wrong profile env:

- log: `mytmp/tq4bit_smoke_key1_emit_unification_20260627165557.log`
- result: output matched, but no profiler directory was created because the
  smoke script expects `XRX_TQ4BIT_PROFILE=1` and `TQ_SMOKE_PROFILE_DIR`.

Strict smoke run:

```bash
XRX_TQ4BIT_PROFILE=1 \
TQ_SMOKE_MODEL_PATH=/root/x00827378/model/Qwen3-0.6B \
TQ_SMOKE_PROFILE_DIR=/root/x00827378/vllm-ascend/mytmp/perflog_smoke_key1_emit_unification_20260627165917 \
python tests/e2e/singlecard/xrx_turboquant4bit_smoke.py
```

- log: `mytmp/tq4bit_smoke_key1_emit_unification_20260627165917.log`
- profile:
  `mytmp/perflog_smoke_key1_emit_unification_20260627165917/rank0_2624818_20260627170000646_ascend_pt`

## Correctness Result

Smoke output matched the route-only baseline exactly:

```text
" Shin. I'm a 25-year-old girl who is interested in the topic of 'Environmental Science - the sciences of of the environment. I need to find the question: I need help!"
"\n\nTurboQuant 是一种专注于GPU加速的高性能计算应用平台 (HPC-P) 核的软件加速框架，它基于 k88000 的0 和 NVIDIA 的 n"
```

The known baseline `"of of"` duplicate remained unchanged. The candidate did
not introduce new repetition or semantic drift.

## Performance Result

Guarded smoke shape:

```text
"2,8,128;2,8,128;16;128,128;2;3"
```

| Run | Count | Avg us | P50 us | P90 us | Max us |
| --- | ---: | ---: | ---: | ---: | ---: |
| route-only baseline | 56 | 46.59 | 46.44 | 47.86 | 48.72 |
| emit unification V1 | 56 | 50.59 | 50.00 | 52.80 | 55.26 |

The average regressed by 4.00 us, about 8.6%. P50, P90, and max all worsened.
This fails the smoke latency gate.

## Analysis

This result is more important than the implementation itself. The runtime
change was almost behaviorally equivalent for the large full-group path, and
the smoke workload should stay on `tilingKey = 0`. Yet merely adding a template
flag plus a second instantiation and branching in the shared kernel entry
caused a large small-shape regression.

The likely causes are generated code layout, register/ICache pressure, or
entry-level codegen changes from instantiating the full class twice in the same
kernel body. This matches earlier observations where key1-intended changes
could move key0 timing even when the runtime branch was not taken by smoke.

The experiment did not reach the performance question it was meant to answer:
it did not change batch count, KFC message count, or pipeline structure. It
only proved that this style of key1 isolation is not acceptable.

## Reflection

This is not the best possible emit-unification idea, but it is the best
minimal shared-entry implementation of that idea. The failure shows the next
attempt must avoid perturbing key0 generated code more aggressively.

Credible follow-up options:

- use truly separate kernel entry/symbol or build-time specialization for
  `tilingKey = 1`, if the CANN dynamic compile path supports it without
  placing both full class bodies in the same generated function;
- first perform an offline batch-count and source-profile analysis without
  runtime changes;
- if runtime code is attempted again, make it a real structural change that
  reduces batch count or stage overhead, not another equivalent helper
  extraction.

Do not retry this exact template-flag/shared-entry implementation.

## Decision

Runtime code was reverted. The design and this report are kept in git history
as a failed experiment record.
