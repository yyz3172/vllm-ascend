#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

MODE_ARG="${1:-${MODE:-base}}"
KEEP_SERVER="${KEEP_SERVER:-0}"
STOP_BEFORE_START="${STOP_BEFORE_START:-1}"

cleanup() {
    local status=$?
    if [[ "${KEEP_SERVER}" == "1" ]]; then
        log "KEEP_SERVER=1，保留 PD 服务。手动停止: bash ${SCRIPT_DIR}/start_pd.sh stop"
        exit "${status}"
    fi
    log "清理 PD 服务..."
    bash "${SCRIPT_DIR}/start_pd.sh" stop || true
    exit "${status}"
}

trap cleanup EXIT

init_pd_env
ensure_model_exists

log "一键 PD 精度评测开始，mode=${MODE_ARG}"
log "模型: ${MODEL_PATH}"

if [[ "${STOP_BEFORE_START}" == "1" ]]; then
    bash "${SCRIPT_DIR}/start_pd.sh" stop || true
fi

bash "${SCRIPT_DIR}/start_pd.sh" start "${MODE_ARG}"
bash "${SCRIPT_DIR}/start_eval.sh"

log "一键 PD 精度评测完成"
