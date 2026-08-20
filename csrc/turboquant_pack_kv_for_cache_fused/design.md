# TurboQuant Pack KV Optimization Design

## Background

The fixed e2e benchmark is `source xrx_infoenvs && python tests/e2e/singlecard/xrx_test_sampler.py`.
Per `benchmark_check.txt`, the accepted comparison uses the profiler directory printed by
`Start parsing profiling data in sync mode at:` and compares:

- Python wrapper event for TurboQuant pack.
- Kernel event for `TurboquantPackKvForCacheFused`.

Fixed baseline:

- Profile dir: `/root/x00827378/perflog2/rank0_2807414_20260612074600068_ascend_pt`
- Wrapper average: `247.86964285714285 us`
- Kernel average: `52.58533333333333 us`

The current bottleneck is not only the AscendC pack kernel. In the vLLM hot path,
TurboQuant cache write was:

1. `turboquant_pack_kv_for_cache(key, value)` -> packed K/V tensors.
2. `torch_npu._npu_reshape_and_cache(packed_k, packed_v, key_cache, value_cache, slot_mapping)`.

For single-token decode this extra ATB/cache operation is visible in the profiler and adds
host/wrapper latency even when its device kernel is small.

## Experiment 7B: Pack Directly To Paged Cache

### Plan

Add a new custom op `turboquant_pack_kv_for_cache_to_cache` that reuses the current
`turboquant_pack_kv_for_cache_fused` algorithm but writes encoded rows directly into paged
KV cache by using `slot_mapping`.

Scope:

- Support only the active 8-bit path, `head_size=128`, fp16/bf16 K/V.
- Reuse registered TurboQuant codebook/rotation tables.
- Keep fallback to old `pack + torch_npu._npu_reshape_and_cache` when the fused path is not available.
- Use `int32` slot mapping in the fused path.
- Cache the Python op availability check so the hot path does not call `hasattr(torch.ops...)` every token.

Rollback rule:

- Revert the experiment if correctness fails, e2e fails due to this op, or the accepted e2e
  profiler comparison shows wrapper/kernel regression against the fixed baseline.

### Implementation

Files added/changed:

- Added `csrc/turboquant_pack_kv_for_cache_to_cache/`.
- Added op to `csrc/build_aclnn.sh`.
- Added PyTorch binding and Meta binding in `csrc/torch_binding.cpp` and `csrc/torch_binding_meta.cpp`.
- Added Python wrapper `turboquant_pack_kv_for_cache_to_cache` in
  `vllm_ascend/ops/turboquant_kv_cache.py`.
- Replaced the TurboQuant branch in `vllm_ascend/attention/attention_v1.py` so it calls the
  fused pack-to-cache wrapper directly.
- Added focused bytewise equivalence UT in `tests/ut/ops/test_turboquant_kv_cache_perf.py`.

Kernel write layout:

- `vecIdx = tokenIdx * numHeads + headIdx`
- `slot = slot_mapping[tokenIdx]`
- output offset = `((slot * numHeads + headIdx) * slot_w)`
- `slot < 0` is skipped for padding.

### Validation

Build:

- `bash build_ext_clean_preprocess.sh`
- Verified `libcust_opapi.so` exports:
  - `aclnnTurboquantPackKvForCacheToCache`
  - `aclnnTurboquantPackKvForCacheToCacheGetWorkspaceSize`
- Verified Python extension exposes:
  - `torch.ops._C_ascend.turboquant_pack_kv_for_cache_to_cache`

Correctness:

- `source xrx_infoenvs && pytest -q tests/ut/ops/test_turboquant_kv_cache_perf.py::test_turboquant_pack_kv_for_cache_to_cache_matches_old_path`
- Result: passed.

Micro benchmark, Qwen3-0.6B-like shape (`H=8`, `D=128`, bf16, 8-bit, int32 slot mapping):

| tokens | old pack+reshape avg(us) | new pack-to-cache avg(us) | delta |
| --- | ---: | ---: | ---: |
| 1 | 173.532 | 154.279 | -11.09% |
| 2 | 158.830 | 154.409 | -2.78% |
| 4 | 187.512 | 191.417 | +2.08% |
| 8 | 256.406 | 257.906 | +0.59% |
| 16 | 392.751 | 391.119 | -0.42% |
| 32 | 667.427 | 670.692 | +0.49% |
| 64 | 1214.268 | 1198.736 | -1.28% |
| 128 | 2306.127 | 2275.633 | -1.32% |

Micro profiler, T=1 before op-availability caching:

- old record_function avg: `484.657 us`
- new record_function avg: `373.611 us`
- old pack kernel avg: `52.861 us`
- old reshape ATB avg: `57.925 us`, `ReshapeAndCacheNdKernel` avg `2.156 us`
- new fused kernel avg: `52.539 us`

Micro profiler, T=1 after op-availability caching:

- old record_function avg: `465.796 us`
- new record_function avg: `347.235 us`
- old pack kernel avg: `53.305 us`
- new fused kernel avg: `52.836 us`
- `_turboquant_pack_to_cache_op_ready` avg: `0.333 us`

Interpretation:

- The new kernel itself is not slower than the old pack kernel for T=1.
- The main win is removing `_npu_reshape_and_cache` wrapper/ATB overhead.
- The Python availability-cache change removes the previous per-call dynamic op lookup cost.
- Multi-token micro results are mixed and mostly within a small band; e2e trace is still required for final keep/revert decision.

### Current e2e Status

Command run:

```bash
source xrx_infoenvs && python tests/e2e/singlecard/xrx_test_sampler.py
```

Current result:

- Failed during `EngineCore` initialization before TurboQuant pack/cache execution.
- Reproduced twice after the successful build and focused UT.
- Error: KV cache needs `0.44 GiB`, available KV cache memory is `0.42 GiB`; estimated max model length is `7680` while script requests `8192`.
- `npu-smi info` shows no running NPU process, but HBM usage remains near the capacity boundary.
- No new e2e profiler directory was produced, so the accepted `benchmark_check.txt` comparison cannot be completed from this run.

Retry result:

- The same command later completed successfully with `gpu_memory_utilization=0.04`.
- Output included `shutdown complete` and the generated text was semantically normal.
- Profile dir: `/root/x00827378/perflog2/rank0_29533_20260615015725776_ascend_pt`
- Current wrapper event:
  `vllm_ascend/ops/turboquant_kv_cache.py(801): turboquant_pack_kv_for_cache_to_cache`
- Current kernel event: `TurboquantPackKvForCacheToCache`

Accepted profiler comparison:

| metric | baseline avg(us) | current avg(us) | delta |
| --- | ---: | ---: | ---: |
| wrapper | 247.8696 | 120.3006 | -51.47% |
| kernel | 52.5853 | 53.6567 | +2.04% |

Interpretation:

- The wrapper-level target is strongly improved by removing `_npu_reshape_and_cache`.
- The fused kernel is slightly slower than the old pack-only kernel by about `1.07 us`.
- Under the strict rollback rule "both wrapper and kernel averages must not regress", this
  experiment is not a clean pass yet. The likely source is the extra `slot_mapping`
  load/bounds check and direct cache GM write inside the fused kernel.

Next step:

- Decide whether to accept the wrapper win despite the small kernel regression, or optimize the
  direct-cache kernel further to bring `TurboquantPackKvForCacheToCache` back below the fixed
  `52.5853 us` baseline.
