# Experiment Report: key1 Minimal No-op Isolation Probe

Date: 2026-06-27

## Summary

Accepted as a diagnostic experiment. Runtime probe code was reverted after
measurement.

A tiny `if constexpr (TILING_KEY_IS(1))` no-op helper built successfully and
did not regress the guarded smoke shape. This narrows the codegen problem:
small key1 compile-time branches can be safe, while the rejected shared-entry
experiment failed because it instantiated a second full pack class and changed
the shared generated object too much.

## Implementation Tested

The probe added:

```cpp
__aicore__ inline void Key1NoopIsolationProbe() const {
    if constexpr (TILING_KEY_IS(1)) {
    }
}
```

and called it once in `Process()` after the AIC early return. The helper had
no side effects and no UB allocation.

## Commands And Logs

Build:

```bash
bash tools/build_debug_perf.sh
```

The untracked `csrc/fused_infer_attention_score/` directory was temporarily
moved out of `csrc/` during the build because `ASCEND_OP_NAME=ALL` otherwise
picks it up and fails configure. It was restored immediately after build.

- build log: `mytmp/build_key1_noop_probe_20260627171726.log`
- result: pass.

Smoke:

```bash
XRX_TQ4BIT_PROFILE=1 \
TQ_SMOKE_MODEL_PATH=/root/x00827378/model/Qwen3-0.6B \
TQ_SMOKE_PROFILE_DIR=/root/x00827378/vllm-ascend/mytmp/perflog_smoke_key1_noop_probe_20260627172208 \
python tests/e2e/singlecard/xrx_turboquant4bit_smoke.py
```

- smoke log: `mytmp/tq4bit_smoke_key1_noop_probe_20260627172208.log`
- profile:
  `mytmp/perflog_smoke_key1_noop_probe_20260627172208/rank0_2632187_20260627172254115_ascend_pt`

## Correctness Result

Smoke output matched the route-only baseline exactly:

```text
" Shin. I'm a 25-year-old girl who is interested in the topic of 'Environmental Science - the sciences of of the environment. I need to find the question: I need help!"
"\n\nTurboQuant 是一种专注于GPU加速的高性能计算应用平台 (HPC-P) 核的软件加速框架，它基于 k88000 的0 和 NVIDIA 的 n"
```

No new repeated-token semantics were introduced.

## Performance Result

Guarded smoke shape:

```text
"2,8,128;2,8,128;16;128,128;2;3"
```

| Run | Count | Avg us | P50 us | P90 us | Max us |
| --- | ---: | ---: | ---: | ---: | ---: |
| route-only baseline | 56 | 46.59 | 46.44 | 47.86 | 48.72 |
| no-op probe | 56 | 45.57 | 45.32 | 47.02 | 48.88 |

The probe did not regress smoke timing. Since it is not a real optimization,
long query and `tools/op.profile.sh` were not run.

## Analysis

This result shows that key1 compile-time source guards are not automatically
unsafe. The bad pattern was the previous shared-entry template split that
instantiated two complete pack classes and made the generated object much
larger. A tiny `if constexpr (TILING_KEY_IS(1))` branch is acceptable as a
building block for future key1 work.

The next key1 runtime experiment should therefore:

- keep one pack class;
- use local `if constexpr (TILING_KEY_IS(1))` blocks or small helpers;
- avoid second full-class instantiations;
- still rerun smoke before long, because generated object coupling remains.

## Reflection

The probe is intentionally minimal, so it does not prove that a larger key1
UB/batch rewrite will be safe. It only establishes a viable coding style for
the next attempt. Any larger change must grow from this pattern incrementally
and keep the smoke gate first.

## Decision

Probe runtime code was reverted. The report is kept as the evidence that small
`if constexpr (TILING_KEY_IS(1))` guards are viable.
