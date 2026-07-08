#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

init_pd_env
ensure_model_exists

BENCH_LOG="${BENCH_LOG:-${LOG_DIR}/bench_$(date '+%Y%m%d_%H%M%S').log}"
mkdir -p "$(dirname "${BENCH_LOG}")"

wait_for_url "proxy" "http://127.0.0.1:${PROXY_PORT}/healthcheck" "${PROXY_READY_TIMEOUT}"

log "开始性能测试，日志: ${BENCH_LOG}"
log "bench: input_len=${BENCH_INPUT_LEN}, output_len=${BENCH_OUTPUT_LEN}, num_prompts=${BENCH_NUM_PROMPTS}, max_concurrency=${BENCH_MAX_CONCURRENCY}, request_rate=${BENCH_REQUEST_RATE}"

vllm bench serve \
    --backend vllm \
    --model "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PROXY_PORT}" \
    --endpoint /v1/completions \
    --dataset-name random \
    --input-len "${BENCH_INPUT_LEN}" \
    --output-len "${BENCH_OUTPUT_LEN}" \
    --num-prompts "${BENCH_NUM_PROMPTS}" \
    --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --request-rate "${BENCH_REQUEST_RATE}" \
    --ignore-eos \
    2>&1 | tee "${BENCH_LOG}"
