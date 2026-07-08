#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

init_pd_env
ensure_model_exists

EVAL_LOG="${EVAL_LOG:-${LOG_DIR}/eval_$(date '+%Y%m%d_%H%M%S').log}"
EVAL_API_URL="${EVAL_API_URL:-http://127.0.0.1:${PROXY_PORT}/v1/chat/completions}"
EVAL_API_KEY="${EVAL_API_KEY:-EMPTY}"
EVAL_DATASETS="${EVAL_DATASETS:-gsm8k}"
EVAL_DATASET_ARGS="${EVAL_DATASET_ARGS:-{\"gsm8k\":{\"local_path\":\"/root/l00856060/dataset/gsm8k\"}}}"
EVAL_LIMIT="${EVAL_LIMIT:-10}"

mkdir -p "$(dirname "${EVAL_LOG}")"

wait_for_url "proxy" "http://127.0.0.1:${PROXY_PORT}/healthcheck" "${PROXY_READY_TIMEOUT}"

log "开始精度评测，日志: ${EVAL_LOG}"
log "eval: model=${MODEL_PATH}, api_url=${EVAL_API_URL}, datasets=${EVAL_DATASETS}, dataset_args=${EVAL_DATASET_ARGS}, limit=${EVAL_LIMIT}"

evalscope eval \
    --model "${MODEL_PATH}" \
    --api-url "${EVAL_API_URL}" \
    --api-key "${EVAL_API_KEY}" \
    --datasets "${EVAL_DATASETS}" \
    --dataset-args "${EVAL_DATASET_ARGS}" \
    --limit "${EVAL_LIMIT}" \
    2>&1 | tee "${EVAL_LOG}"
