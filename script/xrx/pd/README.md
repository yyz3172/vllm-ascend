# XRX PD one-click perf

This directory contains a lightweight 1P1D PD startup and benchmark flow for
vLLM Ascend. The default model is:

```bash
/root/x00827378/model/Qwen3-0.6B
```

## One-click run

```bash
bash script/xrx/pd/run_pd_perf.sh
```

The command starts prefill, decode, and proxy, runs `vllm bench serve`, then
stops the services.

TurboQuant mode:

```bash
bash script/xrx/pd/run_pd_perf.sh turboquant
```

TurboQuant 4bit mode:

```bash
bash script/xrx/pd/run_pd_perf.sh turboquant4bit
```

Keep services alive after benchmark:

```bash
KEEP_SERVER=1 bash script/xrx/pd/run_pd_perf.sh
```

## Manual control

```bash
bash script/xrx/pd/start_pd.sh start
bash script/xrx/pd/start_bench.sh
bash script/xrx/pd/start_pd.sh stop
```

## Common overrides

```bash
MODEL_PATH=/root/x00827378/model/Qwen3-0.6B \
ASCEND_RT_VISIBLE_DEVICES_PREFILL=0 \
ASCEND_RT_VISIBLE_DEVICES_DECODE=1 \
BENCH_INPUT_LEN=10 \
BENCH_OUTPUT_LEN=200 \
BENCH_NUM_PROMPTS=1000 \
BENCH_MAX_CONCURRENCY=64 \
bash script/xrx/pd/run_pd_perf.sh
```

By default, the scripts source `${REPO_ROOT}/xrx_infoenvs` if it exists. Override
or disable this behavior with:

```bash
SOURCE_ENV=/path/to/envs bash script/xrx/pd/run_pd_perf.sh
SOURCE_ENV=none bash script/xrx/pd/run_pd_perf.sh
```

Logs and PID files are stored under:

```bash
script/xrx/pd/run/
```

Unlike `script/PD/1P1_1D1/run_prefill.sh` and `run_decode.sh`, these scripts do
not pass `--profiler-config`.
