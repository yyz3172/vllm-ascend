ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)
cd ${ROOT_DIR}
BENCH_NUM_PROMPTS=256 \
BENCH_MAX_CONCURRENCY=16 \
BENCH_INPUT_LEN=10 \
BENCH_OUTPUT_LEN=200 \
bash script/xrx/pd/run_pd_perf.sh "${1:-${MODE:-turboquant4bit}}"
