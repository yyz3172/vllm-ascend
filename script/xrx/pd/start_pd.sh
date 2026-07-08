#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

start_component() {
    local name="$1"
    local script="$2"
    local pid_file
    local launch_log
    pid_file="$(pid_file_for "${name}")"
    launch_log="${LOG_DIR}/${name}.launch.log"

    if [[ -f "${pid_file}" ]]; then
        local old_pid
        old_pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if process_alive "${old_pid}"; then
            log "ERROR: ${name} 已在运行 pid=${old_pid}; 请先执行 bash ${SCRIPT_DIR}/start_pd.sh stop"
            exit 1
        fi
        rm -f "${pid_file}"
    fi

    log "启动 ${name}: ${script}"
    if [[ "${script}" == *.py ]]; then
        setsid python3 "${script}" >> "${launch_log}" 2>&1 &
    else
        setsid bash "${script}" >> "${launch_log}" 2>&1 &
    fi
    local pid=$!
    echo "${pid}" > "${pid_file}"
    sleep 2
    if ! process_alive "${pid}"; then
        log "ERROR: ${name} 启动后立即退出，查看日志: ${launch_log}"
        tail -n 80 "${launch_log}" || true
        exit 1
    fi
    log "${name} pid=${pid}, launch_log=${launch_log}"
}

stop_component() {
    local name="$1"
    local pid_file
    pid_file="$(pid_file_for "${name}")"
    if [[ ! -f "${pid_file}" ]]; then
        return
    fi

    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    rm -f "${pid_file}"
    if ! process_alive "${pid}"; then
        log "${name} pid=${pid:-N/A} 已不存在"
        return
    fi

    log "停止 ${name} pid=${pid}"
    kill -TERM -- "-${pid}" >/dev/null 2>&1 || kill -TERM "${pid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
        if ! process_alive "${pid}"; then
            log "${name} 已停止"
            return
        fi
        sleep 0.5
    done
    log "${name} 未正常退出，发送 SIGKILL"
    kill -KILL -- "-${pid}" >/dev/null 2>&1 || kill -KILL "${pid}" >/dev/null 2>&1 || true
}

status_component() {
    local name="$1"
    local pid_file
    pid_file="$(pid_file_for "${name}")"
    if [[ ! -f "${pid_file}" ]]; then
        log "${name}: stopped"
        return
    fi
    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if process_alive "${pid}"; then
        log "${name}: running pid=${pid}"
    else
        log "${name}: stale pid=${pid}"
    fi
}

set_mode() {
    local mode="${1:-${MODE:-base}}"
    case "${mode}" in
        base|BASE)
            export TYPE="BASE"
            ;;
        turboquant|TURBOQUANT|tq)
            export TYPE="TURBOQUANT"
            export TURBOQUANT_KV_BITS="${TURBOQUANT_KV_BITS:-[8, 8]}"
            setup_turboquant_runtime_env
            ;;
        turboquant4bit|TURBOQUANT4BIT|tq4|4bit)
            export TYPE="TURBOQUANT"
            export TURBOQUANT_KV_BITS="${TURBOQUANT_KV_BITS:-[4, 4]}"
            setup_turboquant_runtime_env
            ;;
        *)
            log "ERROR: unknown mode=${mode}; expected base, turboquant, or turboquant4bit"
            exit 1
            ;;
    esac
}

start_stack() {
    ensure_model_exists
    log "MODEL_PATH=${MODEL_PATH}"
    log "TYPE=${TYPE}"
    log "LOCAL_IP=${LOCAL_IP}, NIC_NAME=${NIC_NAME}"
    log "PREFILL_DEVICE=${ASCEND_RT_VISIBLE_DEVICES_PREFILL}, DECODE_DEVICE=${ASCEND_RT_VISIBLE_DEVICES_DECODE}"
    log "PROXY_PORT=${PROXY_PORT}, PREFILL_PORT=${PREFILL_PORT}, DECODE_PORT=${DECODE_PORT}"
    log "LOG_DIR=${LOG_DIR}"

    start_component "prefill" "${TOPO_DIR}/run_prefill.sh"
    wait_for_url "prefill" "http://127.0.0.1:${PREFILL_PORT}/health" "${PD_READY_TIMEOUT}"

    start_component "decode" "${TOPO_DIR}/run_decode.sh"
    wait_for_url "decode" "http://127.0.0.1:${DECODE_PORT}/health" "${PD_READY_TIMEOUT}"

    start_component "proxy" "${TOPO_DIR}/pd_proxy.py"
    wait_for_url "proxy" "http://127.0.0.1:${PROXY_PORT}/healthcheck" "${PROXY_READY_TIMEOUT}"

    log "PD 服务已启动"
}

stop_stack() {
    stop_component "proxy"
    stop_component "decode"
    stop_component "prefill"
}

main() {
    local action="${1:-start}"
    local mode="${2:-${MODE:-base}}"
    init_pd_env

    case "${action}" in
        start)
            set_mode "${mode}"
            start_stack
            ;;
        stop)
            stop_stack
            ;;
        restart)
            stop_stack
            set_mode "${mode}"
            start_stack
            ;;
        status)
            status_component "prefill"
            status_component "decode"
            status_component "proxy"
            ;;
        *)
            cat <<EOF
Usage:
  bash ${SCRIPT_DIR}/start_pd.sh start [base|turboquant|turboquant4bit]
  bash ${SCRIPT_DIR}/start_pd.sh stop
  bash ${SCRIPT_DIR}/start_pd.sh restart [base|turboquant|turboquant4bit]
  bash ${SCRIPT_DIR}/start_pd.sh status
EOF
            exit 1
            ;;
    esac
}

main "$@"
