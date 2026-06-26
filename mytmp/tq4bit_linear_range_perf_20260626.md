# TurboQuant 4bit linear range split perf record

Date: 2026-06-26

Base commit before change: `8444038e3ceee28e3485c25bcb8bb8c302360a11`

Change under test:
- `TurboquantPackKvForCache4bit` cache packing work is split by linear token row range per core.
- Each core owns a continuous `[begin, end)` row interval.
- If `begin` falls inside a 4-row cache group, it backs up to process that full group.
- If non-last-core `end` falls inside a 4-row cache group, it backs down to skip that group.
- Last core keeps the tail group.
- Per-sequence slot lookup is done once at the first token; later token slots are treated as contiguous.

Build:
- Clean build: yes
- Command: `rm -rf build csrc/build && COMPILE_CUSTOM_KERNELS=1 MAX_JOBS=64 python setup.py build_ext --inplace`
- Build log: `mytmp/tq4bit_build_linear_sync_20260626132315.log`

Correctness:
- Custom cosine probe after MTE3 synchronization fix:
  - H=1 mean cosine about 0.9952
  - H=2 mean cosine about 0.9957
  - H=4 mean cosine about 0.99575
  - H=8 mean cosine about 0.99534
- Smoke output log: `mytmp/tq4bit_smoke_linear_sync_20260626133055.log`
- Smoke output had no severe prior degradation pattern; one English sample still had mild phrase repetition.

Device note:
- Before the long-query run, `npu-smi info` showed no running NPU processes, but HBM usage was about `60588/65536 MB`.
- The run used `XRX_TQ4BIT_GPU_MEMORY_UTILIZATION=0.05` as documented in `tips/tq4bit_perf_test_notes.md`.

Smoke workload:
- Model: `/root/x00827378/model/Qwen3-0.6B`
- Profile dir: `mytmp/perflog_smoke_linear_sync_20260626133055`
- Hot shape pack result:
  - `TurboquantPackKvForCache4bit` shape `"2,8,128;2,8,128;16;128,128;2;3"`
  - count=56 avg=46.6486us p50=46.48us p90=48.06us max=49.54us
- Attention:
  - `TurboquantAttentionPaged4bit` count=56 avg=25.0632us p50=25.06us p90=26.28us max=26.94us

Long query workload:
- Script: `tests/e2e/singlecard/xrx_turboquant4bit_long_query_profile.py`
- Profile dir: `mytmp/perflog_long_linear_range_20260626133721`
- Log: `mytmp/tq4bit_long_linear_range_20260626133721.log`
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
  - elapsed_s=9.509889050154015
  - output_chars=[8, 8, 32, 32, 8, 8, 37, 42, 36, 33, 42, 32, 22, 32, 8, 37]

Long query kernel results:
- `TurboquantPackKvForCache4bit` overall:
  - count=140 avg=258.032us p50=50.86us p90=992.56us p99=998.1us max=999.64us
- `TurboquantPackKvForCache4bit` hot shape `"1802,8,128;1802,8,128;16;128,128;1802;2"`:
  - count=28 avg=993.5071us p50=992.56us p90=998.06us max=999.64us
- `TurboquantPackKvForCache4bit` shape `"151,8,128;151,8,128;16;128,128;151;17"`:
  - count=28 avg=149.9864us p50=149.16us p90=154.64us max=155.4us
- `TurboquantPackKvForCache4bit` shape `"16,8,128;16,8,128;16;128,128;16;17"`:
  - count=84 avg=48.8888us p50=48.74us p90=51.64us max=55.04us
- `TurboquantAttentionPaged4bit`:
  - count=112 avg=8501.7036us p50=4194.66us p90=21423.24us p99=21448.84us max=21470.1us
- `FusedInferAttentionScore` shape `"1802,16,128;1802,8,128;1802,8,128;;2048,2048;1;1;;;;;;;;;;;;;;;;;;;;;;;;"`:
  - count=28 avg=192.0771us p50=191.48us p90=195.4us max=196.72us

Conclusion:
- The linear per-core row range split is functional and passes smoke/long-query execution.
- Long query hot shape `[1802,8,128]` is slower than the 2026-06-26 note baseline `ca909c76` at about `893us`, so the next optimization should target reducing the per-batch pack pipeline cost.
