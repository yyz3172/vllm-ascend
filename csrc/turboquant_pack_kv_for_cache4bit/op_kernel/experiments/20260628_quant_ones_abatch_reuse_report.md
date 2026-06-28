# Experiment Report: Quant Ones Reuse Released A Batch Slot

Date: 2026-06-28

## Summary

Rejected and reverted.

The queue-slot reuse idea built successfully and preserved smoke text output,
but it failed the guarded smoke latency gate. A key1-only variant also failed
the same smoke gate, which means this source-level change perturbs the shared
generated object enough to hurt the key0 small shape even when the runtime
branch is guarded with `if constexpr (TILING_KEY_IS(1))`.

The long-query gate was not run because smoke performance failed.

## Implementation Tested

Two variants were tested:

1. Shared runtime variant:
   - removed permanent `quantOnesBuf_`;
   - allocated one released `aBatchQue_` slot inside the reduce-sum branch of
     `EncodeBatch()`;
   - passed the reinterpreted float table into `EncodeQuantCodesByReduceSum()`.

2. Key1-only variant:
   - kept `quantOnesBuf_` for non-key1 paths;
   - used the released `aBatchQue_` slot only under
     `if constexpr (TILING_KEY_IS(1))`;
   - restored key0 runtime semantics as much as possible without a separate
     generated object.

Both variants were reverted from `turboquant_pack_kv_for_cache4bit.cpp`.

## Commands And Logs

Shared variant build:

```bash
bash tools/build_debug_perf.sh
```

- `mytmp/build_quant_ones_abatch_reuse_20260628050217.log`

Shared variant smoke:

```bash
XRX_TQ4BIT_PROFILE=1 \
TQ_SMOKE_MODEL_PATH=/root/x00827378/model/Qwen3-0.6B \
TQ_SMOKE_PROFILE_DIR=/root/x00827378/vllm-ascend/mytmp/perflog_smoke_quant_ones_abatch_reuse_20260628050753 \
python tests/e2e/singlecard/xrx_turboquant4bit_smoke.py
```

- `mytmp/tq4bit_smoke_quant_ones_abatch_reuse_20260628050753.log`
- `mytmp/perflog_smoke_quant_ones_abatch_reuse_20260628050753`

Key1-only variant build:

```bash
bash tools/build_debug_perf.sh
```

- `mytmp/build_quant_ones_abatch_reuse_key1only_20260628051337.log`

Key1-only smoke:

```bash
XRX_TQ4BIT_PROFILE=1 \
TQ_SMOKE_MODEL_PATH=/root/x00827378/model/Qwen3-0.6B \
TQ_SMOKE_PROFILE_DIR=/root/x00827378/vllm-ascend/mytmp/perflog_smoke_quant_ones_abatch_reuse_key1only_20260628052454 \
python tests/e2e/singlecard/xrx_turboquant4bit_smoke.py
```

- failed first attempt due to unrelated NPU memory pressure:
  `mytmp/tq4bit_smoke_quant_ones_abatch_reuse_key1only_20260628051744.log`
- successful rerun:
  `mytmp/tq4bit_smoke_quant_ones_abatch_reuse_key1only_20260628052454.log`
- profile:
  `mytmp/perflog_smoke_quant_ones_abatch_reuse_key1only_20260628052454`

## Correctness Result

Smoke output matched the current fixed baseline exactly for both variants:

```text
" Shin. I'm a 25-year-old girl who is interested in the topic of 'Environmental Science - the sciences of of the environment. I need to find the question: I need help!"
"\n\nTurboQuant 是一种专注于GPU加速的高性能计算应用平台 (HPC-P) 核的软件加速框架，它基于 k88000 的0 和 NVIDIA 的 n"
```

The known baseline duplicate `"of of"` remained unchanged. No new language
corruption was observed.

## Performance Result

Guarded smoke shape:

```text
"2,8,128;2,8,128;16;128,128;2;3"
```

| Run | Count | Avg us | P50 us | P90 us | Max us |
| --- | ---: | ---: | ---: | ---: | ---: |
| current fixed baseline | 56 | 45.96 | 45.58 | 47.62 | 51.12 |
| shared queue reuse | 56 | 46.79 | 46.62 | 48.64 | 49.90 |
| key1-only queue reuse | 56 | 47.58 | 47.49 | 49.16 | 52.08 |

Both variants fail the smoke latency gate. The key1-only variant was worse
than the shared variant on this run.

## Analysis

The lifetime hypothesis was valid enough to compile:

- `RotateBatchMatmul()` frees `aBatchQue_` before `EncodeBatch()`;
- `EncodeBatch()` can allocate a released queue slot and reinterpret it as a
  float temporary table;
- the temporary does not alias `rotateWorkBuf_` and therefore preserves the
  correctness isolation goal from commit `532a507c`.

The performance hypothesis failed. Removing the permanent 8 KB buffer in the
shared variant changed generated layout/code enough to regress the smoke guard
shape. The key1-only variant showed that simply guarding the queue-slot path
with `if constexpr (TILING_KEY_IS(1))` is not enough to isolate key0 in the
current generated object. This is consistent with prior codegen-isolation
findings: key0/key1 are still packaged together for a dtype object, and source
changes can perturb the guarded smoke path even when runtime behavior is not
intended to change.

The first key1-only smoke attempt failed before kernel execution because an
unrelated orphan `VLLMEngineCore` from `xrx_bit_residual_smoke.py` held NPU
memory. Its parent `timeout 600` process exited, leaving PID `2697666` with
PPID 1 and about 3313 MB of NPU memory. The orphan was terminated before the
successful rerun.

## Decision

Runtime code was reverted. Do not keep or retry this queue-slot reuse pattern
inside the current shared generated object.

Future attempts to recover the 8 KB UB regression need stronger codegen
isolation than a local `if constexpr`, or a different approach that does not
change the guarded smoke path's generated object.
