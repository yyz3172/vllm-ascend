# TurboQuant 4bit normalize-and-encode perf record

Date: 2026-06-26

Base commit before change: `4cbb8ec60ff9ce38189ca8332b9f2a11f9b0c5f3`

Change under test:
- Followed `csrc/turboquant_pack_kv_for_cache4bit/op_kernel/OPTIMIZE_PLAN.md`.
- Removed `aBatchQue_`.
- Changed `xBatchQue_` from `VECIN` to `VECOUT`, so the copied input batch can be used directly as Matmul TensorA.
- Replaced pre-matmul unitization with:
  - `ComputeNormsOnly`: compute and store norms, keep `xBatch` unchanged.
  - `RotateBatchMatmul`: compute `z = x @ R^T` from the original input batch.
  - `NormalizeAndEncode`: normalize `z` with per-row `inv_norm`, then apply the 4-bit threshold encode.
- The first implementation stored `inv_norm` in `normsBuf_` and requeued `xBatch`; long-query hot shape was slower at about `1031.41us`.
- Final implementation keeps `inv_norm` in a scalar stack array and passes the dequeued `xBatch` directly through norm compute and Matmul.

Build:
- Clean build: yes
- Command: `rm -rf build csrc/build && COMPILE_CUSTOM_KERNELS=1 MAX_JOBS=64 python setup.py build_ext --inplace`
- Build log: `mytmp/tq4bit_build_norm_encode_scalar_20260626140641.log`

Correctness:
- Fused guard log: `mytmp/tq4bit_fused_guard_norm_encode_scalar_20260626141400.log`
- Random data:
  - fp16 key `TQ-Fused` cosine_similarity=0.995406
  - fp16 value `TQ-Fused` cosine_similarity=0.995285
  - bf16 key `TQ-Fused` cosine_similarity=0.995413
  - bf16 value `TQ-Fused` cosine_similarity=0.995279
- Real smoke dump:
  - fp16 key `TQ-Fused` cosine_similarity=0.995363
  - fp16 value `TQ-Fused` cosine_similarity=0.995360
  - bf16 key `TQ-Fused` cosine_similarity=0.995360
  - bf16 value `TQ-Fused` cosine_similarity=0.995357
- Smoke output log: `mytmp/tq4bit_smoke_norm_encode_scalar_20260626141228.log`
- Smoke output had no severe prior gibberish pattern, but still showed mild repetition and a malformed Chinese phrase.

Device note:
- Before the final long-query run, `npu-smi info` showed no running NPU processes, but HBM usage was about `60589/65536 MB`.
- The run used `XRX_TQ4BIT_GPU_MEMORY_UTILIZATION=0.05`.

Smoke workload:
- Model: `/root/x00827378/model/Qwen3-0.6B`
- Profile dir: `mytmp/perflog_smoke_norm_encode_scalar_20260626141228`
- Hot shape pack result:
  - `TurboquantPackKvForCache4bit` shape `"2,8,128;2,8,128;16;128,128;2;3"`
  - count=56 avg=46.4707us p50=46.44us p90=48.18us max=48.94us
- Attention:
  - `TurboquantAttentionPaged4bit` count=56 avg=25.1361us p50=24.6us p90=27.38us max=28.32us

Long query workload:
- Script: `tests/e2e/singlecard/xrx_turboquant4bit_long_query_profile.py`
- Profile dir: `mytmp/perflog_long_norm_encode_scalar_20260626141459`
- Log: `mytmp/tq4bit_long_norm_encode_scalar_20260626141459.log`
- Environment:
  - `XRX_TQ4BIT_LONG_NUM_PROMPTS=16`
  - `XRX_TQ4BIT_LONG_PROMPT_TOKENS=2000`
  - `XRX_TQ4BIT_LONG_OUTPUT_TOKENS=8`
  - `XRX_TQ4BIT_LONG_MAX_MODEL_LEN=2040`
  - `XRX_TQ4BIT_GPU_MEMORY_UTILIZATION=0.05`
  - `XRX_TQ4BIT_ENABLE_PROFILE=1`
  - `VLLM_VERSION=0.18.0`
- Script result:
  - exit code 0
  - elapsed_s=9.428350300062448
  - output_chars=[8, 8, 36, 16, 40, 50, 20, 27, 37, 36, 8, 40, 37, 8, 37, 28]

Long query kernel results:
- `TurboquantPackKvForCache4bit` overall:
  - count=140 avg=261.1723us p50=52.04us p90=1006.58us p99=1009.84us max=1010.88us
- `TurboquantPackKvForCache4bit` hot shape `"1802,8,128;1802,8,128;16;128,128;1802;2"`:
  - count=28 avg=1006.4814us p50=1006.58us p90=1009.26us max=1010.88us
- `TurboquantPackKvForCache4bit` shape `"151,8,128;151,8,128;16;128,128;151;17"`:
  - count=28 avg=150.6036us p50=150.04us p90=151.86us max=154.8us
- `TurboquantPackKvForCache4bit` shape `"16,8,128;16,8,128;16;128,128;16;17"`:
  - count=84 avg=49.5921us p50=49.4us p90=52.46us max=58.68us
- `TurboquantAttentionPaged4bit`:
  - count=112 avg=8503.0857us p50=4196.78us p90=21426.04us p99=21452.64us max=21466.86us
- `FusedInferAttentionScore` hot shape:
  - count=28 avg=192.4671us p50=192.06us p90=195.24us max=196.86us

Comparison with previous commit `4cbb8ec6`:
- Smoke small shape improved slightly: `46.6486us` -> `46.4707us`.
- Long-query hot shape regressed: `993.5071us` -> `1006.4814us` (`+1.31%`).

Conclusion:
- The planned UB reduction is implemented and correctness guards remain healthy.
- This change is not a long-prefill performance win for the `[1802,8,128]` hot shape. The likely cost is reading input rows from `VECOUT` during norm computation, which offsets the removed `aBatchQue_` copy/cast work.
