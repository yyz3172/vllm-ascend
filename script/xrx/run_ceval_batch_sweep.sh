#!/usr/bin/env bash
# ceval computer_network 并发压测：不同 --eval-batch-size；
# 每个 batch_size 只启动一次 vllm serve，客户端连续压测 ROUNDS 轮后再停服。
#
# 用法:
#   bash script/xrx/run_ceval_batch_sweep.sh
#   BATCH_SIZES="1 8 16 32" ROUNDS=3 bash script/xrx/run_ceval_batch_sweep.sh
#   bash script/xrx/run_ceval_batch_sweep.sh --batch-sizes 16 --rounds 3
#
# 环境变量（均可覆盖）:
#   VISIBLE_DEVICES             NPU 卡号，默认 7（在 source infoenvs 之后强制生效，
#                               避免被 infoenvs 里的 ASCEND_RT_VISIBLE_DEVICES=0 覆盖）
#   SERVE_MODEL_PATH            服务加载权重，默认 /root/cyl/model/Qwen3-4B
#   EVAL_MODEL_PATH             evalscope --model，默认与 SERVE_MODEL_PATH 相同
#   PORT                        默认 9555
#   GPU_MEMORY_UTILIZATION      默认 0.3
#   MAX_MODEL_LEN               默认 8500
#   BATCH_SIZES                 默认 "16"（空格分隔多个并发）
#   ROUNDS                      每个 batch_size 下客户端压测轮数，默认 3（同一 serve）
#   WORK_ROOT                   evalscope 结果根目录，默认 ./output/ceval_batch_sweep
#   LOG_ROOT                    日志根目录，默认 ${WORK_ROOT}/logs
#   READY_TIMEOUT               服务就绪等待秒数，默认 600
#   SOURCE_ENV                  可选，默认仓库 infoenvs（存在则 source）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------- defaults（infoenvs 会覆盖 ASCEND_*，故设备/FIA 在 source 后再强制套一次）----------
# Use VISIBLE_DEVICES (default 7). Do NOT inherit ASCEND_RT_VISIBLE_DEVICES from
# the parent shell — it is usually 0 from a prior `source infoenvs`.
VISIBLE_DEVICES="${VISIBLE_DEVICES:-7}"
_FIA="${VLLM_ASCEND_BIT_RESIDUAL_FIA:-1}"
_DECODE_FIA="${VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA:-1}"

SERVE_MODEL_PATH="${SERVE_MODEL_PATH:-${MODEL_PATH:-/root/cyl/model/Qwen3-4B}}"
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-${SERVE_MODEL_PATH}}"
PORT="${PORT:-9555}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8500}"
BATCH_SIZES="${BATCH_SIZES:-16}"
ROUNDS="${ROUNDS:-3}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
CEVAL_LOCAL_PATH="${CEVAL_LOCAL_PATH:-/root/cyl/dataset}"
CEVAL_SUBSET="${CEVAL_SUBSET:-computer_network}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${PORT}/health}"
API_URL="${API_URL:-http://127.0.0.1:${PORT}/v1/chat/completions}"

STAMP="$(date '+%Y%m%d_%H%M%S')"
WORK_ROOT="${WORK_ROOT:-${PWD}/output/ceval_batch_sweep_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/vllm_serve.pid}"

# ---------- CLI ----------
usage() {
    sed -n '2,20p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --device|--devices)
            VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            HEALTH_URL="http://127.0.0.1:${PORT}/health"
            API_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
            shift 2
            ;;
        --work-root)
            WORK_ROOT="$2"
            LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs}"
            shift 2
            ;;
        *)
            echo "未知参数: $1" >&2
            usage
            ;;
    esac
done

mkdir -p "${LOG_ROOT}" "${WORK_ROOT}"

log() {
    printf '%s [ceval-sweep] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_ROOT}/sweep.log"
}

source_runtime_env() {
    local env_file="${SOURCE_ENV:-${REPO_ROOT}/infoenvs}"
    if [[ -n "${env_file}" && "${env_file}" != "none" && -f "${env_file}" ]]; then
        # shellcheck disable=SC1090
        source "${env_file}"
        log "sourced env: ${env_file}"
    fi
}

process_alive() {
    local pid="${1:-}"
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

stop_serve() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
        rm -f "${PID_FILE}"
        if process_alive "${pid}"; then
            log "停止 vllm serve pid=${pid}"
            # kill process group if started with setsid
            kill -TERM -- "-${pid}" >/dev/null 2>&1 || kill -TERM "${pid}" >/dev/null 2>&1 || true
            for _ in $(seq 1 40); do
                if ! process_alive "${pid}"; then
                    break
                fi
                sleep 0.5
            done
            if process_alive "${pid}"; then
                log "强制 SIGKILL pid=${pid}"
                kill -KILL -- "-${pid}" >/dev/null 2>&1 || kill -KILL "${pid}" >/dev/null 2>&1 || true
            fi
        fi
    fi
    # 兜底：按端口清理残留
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    elif command -v lsof >/dev/null 2>&1; then
        local pids
        pids="$(lsof -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
        if [[ -n "${pids}" ]]; then
            # shellcheck disable=SC2086
            kill -TERM ${pids} >/dev/null 2>&1 || true
            sleep 1
            # shellcheck disable=SC2086
            kill -KILL ${pids} >/dev/null 2>&1 || true
        fi
    fi
}

wait_ready() {
    local timeout="${1}"
    local url="${2}"
    local t0
    t0="$(date +%s)"
    log "等待服务就绪: ${url} (timeout=${timeout}s)"
    while true; do
        if curl -sf "${url}" >/dev/null 2>&1; then
            log "服务已就绪"
            return 0
        fi
        if ! process_alive "$(cat "${PID_FILE}" 2>/dev/null || true)"; then
            log "ERROR: serve 进程已退出，见日志 ${CURRENT_SERVE_LOG:-${LOG_ROOT}}"
            return 1
        fi
        if (( "$(date +%s)" - t0 >= timeout )); then
            log "ERROR: 等待就绪超时 ${timeout}s"
            return 1
        fi
        sleep 2
    done
}

start_serve() {
    local tag="$1"
    local serve_log="${LOG_ROOT}/serve_${tag}.log"
    CURRENT_SERVE_LOG="${serve_log}"

    stop_serve

    if [[ ! -e "${SERVE_MODEL_PATH}" ]]; then
        log "ERROR: SERVE_MODEL_PATH 不存在: ${SERVE_MODEL_PATH}"
        exit 1
    fi

    log "启动 vllm serve -> ${serve_log}"
    log "  model=${SERVE_MODEL_PATH} port=${PORT} util=${GPU_MEMORY_UTILIZATION} max_len=${MAX_MODEL_LEN}"
    log "  devices=${ASCEND_RT_VISIBLE_DEVICES} FIA=${VLLM_ASCEND_BIT_RESIDUAL_FIA}/${VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA}"

    (
        export ASCEND_RT_VISIBLE_DEVICES
        export VLLM_ASCEND_BIT_RESIDUAL_FIA
        export VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA
        # 避免父 shell 的 MODEL_PATH 干扰
        export MODEL_PATH="${SERVE_MODEL_PATH}"
        exec setsid vllm serve "${SERVE_MODEL_PATH}" \
            --kv_cache_dtype=turboquant \
            --additional-config='{"turboquant_kv_bits": [8, 4]}' \
            --port "${PORT}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --no-enable-prefix-caching \
            --kv-cache-metrics
    ) >"${serve_log}" 2>&1 &
    local pid=$!
    echo "${pid}" > "${PID_FILE}"
    sleep 2
    if ! process_alive "${pid}"; then
        log "ERROR: serve 启动后立即退出，tail:"
        tail -n 80 "${serve_log}" || true
        exit 1
    fi
    log "serve pid=${pid}"
    wait_ready "${READY_TIMEOUT}" "${HEALTH_URL}"
}

run_eval() {
    local batch_size="$1"
    local round="$2"
    local tag="bs${batch_size}_r${round}"
    local eval_log="${LOG_ROOT}/eval_${tag}.log"
    local work_dir="${WORK_ROOT}/${tag}"

    mkdir -p "${work_dir}"

    if [[ ! -e "${EVAL_MODEL_PATH}" ]]; then
        log "WARN: EVAL_MODEL_PATH 不存在: ${EVAL_MODEL_PATH}，改用 SERVE_MODEL_PATH"
        EVAL_MODEL_PATH="${SERVE_MODEL_PATH}"
    fi

    local dataset_args
    dataset_args="$(printf '{"ceval": {"local_path": "%s", "subset_list": ["%s"]}}' \
        "${CEVAL_LOCAL_PATH}" "${CEVAL_SUBSET}")"
    local gen_config
    gen_config="$(printf '{"temperature": %s, "max_tokens": %s}' \
        "${TEMPERATURE}" "${MAX_TOKENS}")"

    log "开始 eval: batch_size=${batch_size} round=${round}/${ROUNDS}"
    log "  api=${API_URL} work_dir=${work_dir} eval_log=${eval_log}"

    set +e
    evalscope eval \
        --model "${EVAL_MODEL_PATH}" \
        --api-url "${API_URL}" \
        --api-key EMPTY \
        --datasets ceval \
        --dataset-args "${dataset_args}" \
        --generation-config "${gen_config}" \
        --eval-batch-size "${batch_size}" \
        --debug \
        --work-dir "${work_dir}" \
        --no-timestamp \
        >"${eval_log}" 2>&1
    local rc=$?
    set -e

    if [[ ${rc} -ne 0 ]]; then
        log "ERROR: eval 失败 rc=${rc}，见 ${eval_log}"
        tail -n 40 "${eval_log}" || true
        return "${rc}"
    fi
    log "eval 完成: ${tag}"
    return 0
}

cleanup() {
    log "清理：停止 serve"
    stop_serve
}
trap cleanup EXIT

# ---------- main ----------
source_runtime_env
# infoenvs 会 export ASCEND_RT_VISIBLE_DEVICES=0；这里强制套回 harness 配置
export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export VLLM_ASCEND_BIT_RESIDUAL_FIA="${_FIA}"
export VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA="${_DECODE_FIA}"

log "WORK_ROOT=${WORK_ROOT}"
log "LOG_ROOT=${LOG_ROOT}"
log "VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} BATCH_SIZES=${BATCH_SIZES} ROUNDS=${ROUNDS}"

fail_count=0
for bs in ${BATCH_SIZES}; do
    log "======== start serve for batch_size=${bs} (then ${ROUNDS} client rounds) ========"
    start_serve "bs${bs}"
    for ((r = 1; r <= ROUNDS; r++)); do
        tag="bs${bs}_r${r}"
        log "-------- client ${tag} --------"
        if ! run_eval "${bs}" "${r}"; then
            fail_count=$((fail_count + 1))
            log "本轮失败，继续下一轮（服务保持运行）"
        fi
    done
    stop_serve
    sleep 3
done

log "全部结束 fail_count=${fail_count} logs=${LOG_ROOT}"
if [[ ${fail_count} -gt 0 ]]; then
    exit 1
fi
exit 0
