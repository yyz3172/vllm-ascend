# Experiment Design: key1 Minimal No-op Isolation Probe

Date: 2026-06-27

## Hypothesis

The previous key1 shared-entry template probe regressed smoke badly, but it
also instantiated the full pack class twice. We need to know whether any tiny
key1-only source change perturbs key0, or only large shared-entry
instantiations do.

Hypothesis: a minimal `if constexpr (TILING_KEY_IS(1))` helper inside an
existing function can compile without changing key0 smoke timing. If even this
regresses smoke, larger key1 work must wait for true build isolation.

## Scope

- Runtime target: source-shape probe only.
- No semantic behavior change.
- No new UB buffers.
- No second class instantiation.
- No Normalize/Rotate/Encode/CopyOut changes.
- Run only build and smoke; do not run long/profile because no performance
  improvement is expected.

## Implementation Sketch

Add a tiny helper:

```cpp
__aicore__ inline void Key1NoopIsolationProbe() const {
    if constexpr (TILING_KEY_IS(1)) {
        // no side effects
    }
}
```

Call it once near the start of `Process()`:

```cpp
Key1NoopIsolationProbe();
```

If `if constexpr (TILING_KEY_IS(1))` is supported as a compile-time condition,
key0 should receive no meaningful code. If CANN/macro handling rejects it or
smoke timing moves, the report should say so and revert.

## Validation Gates

1. Build with `tools/build_debug_perf.sh`.
2. Run smoke with correct profile variables:

```bash
XRX_TQ4BIT_PROFILE=1 \
TQ_SMOKE_MODEL_PATH=/root/x00827378/model/Qwen3-0.6B \
TQ_SMOKE_PROFILE_DIR=<dir> \
python tests/e2e/singlecard/xrx_turboquant4bit_smoke.py
```

3. Smoke output must match the route-only baseline exactly.
4. Guarded small shape must not regress from avg 46.59 us.
5. Revert runtime code regardless of result unless the probe is needed for a
   later accepted implementation. The report is the deliverable.

## Reflection Before Coding

This probe is intentionally not an optimization. It is a build/codegen
diagnostic. Its result determines whether future key1 experiments can use
small `if constexpr (TILING_KEY_IS(1))` branches safely.
