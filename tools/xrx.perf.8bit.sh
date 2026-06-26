BENCH_NUM_PROMPTS=16 \
BENCH_MAX_CONCURRENCY=16 \
BENCH_INPUT_LEN=2000 \
BENCH_OUTPUT_LEN=200 \
bash script/xrx/pd/run_pd_perf.sh "${1:-${MODE:-turboquant}}"
