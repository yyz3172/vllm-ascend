#!/usr/bin/env bash
set -euo pipefail

# LCY one-click TurboQuant PD test wrapper.
#
# Usage:
#   bash script/lcy/test_tq.sh [base|turboquant|turboquant4bit] [perf|accu]
#   bash script/lcy/test_tq.sh turboquant4bit accu
#   PD_TASK=accu bash script/lcy/test_tq.sh turboquant
#   KEEP_SERVER=1 bash script/lcy/test_tq.sh turboquant4bit perf
#
# Task:
#   perf (default) -> script/xrx/pd/run_pd_perf.sh
#   accu           -> script/xrx/pd/run_pd_accu.sh
#
# Common overrides:
#   SOURCE_ENV=/path/to/envs
#   MODEL_PATH=/path/to/model
#   BENCH_INPUT_LEN=10 BENCH_OUTPUT_LEN=200 BENCH_NUM_PROMPTS=256

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PD_SCRIPT_DIR="${REPO_ROOT}/script/xrx/pd"

log() {
    printf '%s [lcy-test-tq] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

MODE="${1:-base}"
TASK="${PD_TASK:-${2:-perf}}"

case "${MODE}" in
  base|turboquant|turboquant4bit) ;;
  *)
    log "ERROR: unsupported MODE=${MODE}; expected base, turboquant, or turboquant4bit"
    exit 1
    ;;
esac

case "${TASK}" in
  perf|accu) ;;
  *)
    log "ERROR: unsupported TASK=${TASK}; expected perf or accu"
    exit 1
    ;;
esac

# ---- Runtime env (consumed by script/xrx/pd/common.sh) ----
export SOURCE_ENV="${SOURCE_ENV:-${REPO_ROOT}/infoenvs}"

# ---- Model & NPU devices ----
export MODEL_PATH="${MODEL_PATH:-/root/l00856060/model/Qwen3-0.6B}"
export ASCEND_RT_VISIBLE_DEVICES_PREFILL="${ASCEND_RT_VISIBLE_DEVICES_PREFILL:-0}"
export ASCEND_RT_VISIBLE_DEVICES_DECODE="${ASCEND_RT_VISIBLE_DEVICES_DECODE:-1}"

# ---- Network ----
export NIC_NAME="${NIC_NAME:-enp189s0f0}"
export LOCAL_IP="${LOCAL_IP:-192.168.9.160}"

# ---- PD service ----
export PREFILL_GPU_MEMORY_UTILIZATION="${PREFILL_GPU_MEMORY_UTILIZATION:-0.05}"
export DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-0.05}"
export STOP_BEFORE_START="${STOP_BEFORE_START:-1}"
export KEEP_SERVER="${KEEP_SERVER:-0}"

# ---- Benchmark (aligned with script/xrx/pd/start_bench.sh) ----
export BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-10}"
export BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-200}"
export BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-256}"
export BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-16}"
export BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"

cd "${REPO_ROOT}"

case "${TASK}" in
  perf)
    RUN_SCRIPT="run_pd_perf.sh"
    log "开始 PD 性能测试，mode=${MODE}, task=${TASK}"
    ;;
  accu)
    RUN_SCRIPT="run_pd_accu.sh"
    log "开始 PD 精度评测，mode=${MODE}, task=${TASK}"
    ;;
esac

log "SOURCE_ENV=${SOURCE_ENV}"
log "MODEL_PATH=${MODEL_PATH}"
log "PREFILL_DEVICE=${ASCEND_RT_VISIBLE_DEVICES_PREFILL}, DECODE_DEVICE=${ASCEND_RT_VISIBLE_DEVICES_DECODE}"

if [[ "${TASK}" == "perf" ]]; then
    log "bench: input_len=${BENCH_INPUT_LEN}, output_len=${BENCH_OUTPUT_LEN}, num_prompts=${BENCH_NUM_PROMPTS}, max_concurrency=${BENCH_MAX_CONCURRENCY}, request_rate=${BENCH_REQUEST_RATE}"
else
    log "eval: datasets=${EVAL_DATASETS:-gsm8k}, limit=${EVAL_LIMIT:-10}"
fi

exec bash "${PD_SCRIPT_DIR}/${RUN_SCRIPT}" "${MODE}"
