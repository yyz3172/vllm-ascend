# TurboQuant 4bit linear range m64 perf record

Date: 2026-06-26

Base commit before working-tree change: `967d1741`

Change under test:
- Restore the numerically stable normalize-before-rotate path after the failed normalize-fold experiment.
- Split cache packing by linear token row range per AIV worker.
- Each worker owns a continuous `[begin, end)` token interval.
- If `begin` falls inside a 4-row cache group, it backs up and processes the full group.
- If non-last-worker `end` falls inside a 4-row cache group, it backs down and skips that group.
- The last worker keeps the tail group.
- Resolve only the first physical slot per sequence; later sequence tokens are treated as physically contiguous.
- For contiguous input rows, copy a whole batch with one `DataCopy`.
- Batch two full 4-row cache groups at a time on the hot path (`m=64`) and write packed full groups directly.

Build:
- Clean build: yes
- Command: `rm -rf build csrc/build && COMPILE_CUSTOM_KERNELS=1 MAX_JOBS=64 python setup.py build_ext --inplace`
- Build log: `mytmp/tq4bit_build_20260626185228.log`

Correctness:
- Fused guard: `mytmp/tq4bit_fused_guard_20260626185228.log`
- Result: 4 passed, 0 failed, 0 skipped
- Smoke log: `mytmp/tq4bit_smoke_20260626185228.log`
- Smoke output did not show the prior long repeated-token degeneration.

Smoke workload:
- Model: `/root/x00827378/model/Qwen3-0.6B`
- Profile dir: `mytmp/perflog_smoke_20260626185228`
- `TurboquantPackKvForCache4bit` shape `"2,8,128;2,8,128;16;128,128;2;3"`:
  - count=56 avg=47.3296us p50=46.50us p90=50.62us max=52.70us

Long query workload:
- Script: `tests/e2e/singlecard/xrx_turboquant4bit_long_query_profile.py`
- Profile dir: `mytmp/perflog_long_20260626185228`
- Log: `mytmp/tq4bit_long_20260626185228.log`
- Environment:
  - `XRX_TQ4BIT_LONG_NUM_PROMPTS=16`
  - `XRX_TQ4BIT_LONG_PROMPT_TOKENS=2000`
  - `XRX_TQ4BIT_LONG_OUTPUT_TOKENS=8`
  - `XRX_TQ4BIT_LONG_MAX_MODEL_LEN=2040`
  - `XRX_TQ4BIT_GPU_MEMORY_UTILIZATION=0.05`
  - `XRX_TQ4BIT_ENABLE_PROFILE=1`
  - `VLLM_VERSION=0.18.0`

Long query kernel results:
- `TurboquantPackKvForCache4bit` overall:
  - count=140 avg=232.1583us p50=50.96us p90=872.48us p99=876.14us max=878.02us
- `TurboquantPackKvForCache4bit` hot shape `"1802,8,128;1802,8,128;16;128,128;1802;2"`:
  - count=28 avg=872.7486us p50=872.48us p90=875.72us max=878.02us
- `TurboquantPackKvForCache4bit` shape `"151,8,128;151,8,128;16;128,128;151;17"`:
  - count=28 avg=139.8457us p50=139.44us p90=140.78us max=143.66us
- `TurboquantPackKvForCache4bit` shape `"16,8,128;16,8,128;16;128,128;16;17"`:
  - count=84 avg=49.3990us p50=49.06us p90=51.32us max=56.38us

Conclusion:
- Hot shape `[1802,8,128]` improved from the notes baseline around `893us` to `872.75us`.
- This also improves over the earlier valid direct-full-group result `906.50us`.
