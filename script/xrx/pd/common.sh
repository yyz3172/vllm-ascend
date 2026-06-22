#!/usr/bin/env bash

set -euo pipefail

XRX_PD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${XRX_PD_DIR}/../../.." && pwd)"
TOPO_DIR="${XRX_PD_DIR}/1P1_1D1"

detect_nic_name() {
    if command -v ip >/dev/null 2>&1; then
        ip route show default 2>/dev/null \
            | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}' \
            || true
    fi
}

detect_local_ip() {
    local route_ip
    route_ip=""
    if command -v ip >/dev/null 2>&1; then
        route_ip="$(ip route get 1.1.1.1 2>/dev/null \
            | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}' \
            || true)"
    fi
    if [[ -n "${route_ip}" ]]; then
        echo "${route_ip}"
        return
    fi
    hostname -I 2>/dev/null | awk '{print $1}' || true
}

source_runtime_env() {
    local env_file="${SOURCE_ENV:-${REPO_ROOT}/xrx_infoenvs}"
    if [[ -n "${env_file}" && "${env_file}" != "none" && -f "${env_file}" ]]; then
        # shellcheck disable=SC1090
        source "${env_file}"
    fi
}

init_pd_env() {
    source_runtime_env

    export MODEL_PATH="${MODEL_PATH:-/root/x00827378/model/Qwen3-0.6B}"
    export PROXY_PORT="${PROXY_PORT:-9878}"
    export PREFILL_PORT="${PREFILL_PORT:-9000}"
    export DECODE_PORT="${DECODE_PORT:-9010}"
    export PREFILL_KV_PORT="${PREFILL_KV_PORT:-20001}"
    export DECODE_KV_PORT="${DECODE_KV_PORT:-20002}"

    export NIC_NAME="${NIC_NAME:-$(detect_nic_name)}"
    export LOCAL_IP="${LOCAL_IP:-$(detect_local_ip)}"
    export NIC_NAME="${NIC_NAME:-eth0}"
    export LOCAL_IP="${LOCAL_IP:-127.0.0.1}"

    export ASCEND_RT_VISIBLE_DEVICES_PREFILL="${ASCEND_RT_VISIBLE_DEVICES_PREFILL:-${ASCEND_RT_VISIBLE_DEVICES:-0}}"
    export ASCEND_RT_VISIBLE_DEVICES_DECODE="${ASCEND_RT_VISIBLE_DEVICES_DECODE:-${ASCEND_RT_VISIBLE_DEVICES:-0}}"

    export RUN_DIR="${RUN_DIR:-${XRX_PD_DIR}/run}"
    export LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
    export PID_DIR="${PID_DIR:-${RUN_DIR}/pids}"
    export TYPE="${TYPE:-BASE}"

    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    export LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-1024}"
    export PREFILL_GPU_MEMORY_UTILIZATION="${PREFILL_GPU_MEMORY_UTILIZATION:-0.15}"
    export DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-0.15}"

    export PD_READY_TIMEOUT="${PD_READY_TIMEOUT:-240}"
    export PROXY_READY_TIMEOUT="${PROXY_READY_TIMEOUT:-120}"

    export BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-10}"
    export BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-200}"
    export BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-1000}"
    export BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-64}"
    export BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"

    mkdir -p "${LOG_DIR}" "${PID_DIR}"
}

log() {
    printf '%s [xrx-pd] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ensure_model_exists() {
    if [[ ! -e "${MODEL_PATH}" ]]; then
        log "ERROR: MODEL_PATH 不存在: ${MODEL_PATH}"
        exit 1
    fi
}

pid_file_for() {
    case "$1" in
        prefill) echo "${PID_DIR}/prefill.pid" ;;
        decode) echo "${PID_DIR}/decode.pid" ;;
        proxy) echo "${PID_DIR}/proxy.pid" ;;
        *) echo "unknown component: $1" >&2; return 1 ;;
    esac
}

process_alive() {
    local pid="$1"
    [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
}

http_ready() {
    local url="$1"
    if command -v curl >/dev/null 2>&1; then
        curl -fsS --max-time 5 "${url}" >/dev/null 2>&1
        return
    fi
    python3 - "${url}" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as resp:
        raise SystemExit(0 if 200 <= resp.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_url() {
    local name="$1"
    local url="$2"
    local timeout_s="$3"
    local start_s
    start_s="$(date +%s)"
    log "等待 ${name} 就绪: ${url} (timeout=${timeout_s}s)"
    while true; do
        if http_ready "${url}"; then
            log "${name} 已就绪"
            return 0
        fi
        if (( $(date +%s) - start_s >= timeout_s )); then
            log "ERROR: ${name} 在 ${timeout_s}s 内未就绪"
            return 1
        fi
        sleep 2
    done
}

setup_common_runtime_env() {
    local libstdcpp=""
    for candidate in \
        /usr/lib/aarch64-linux-gnu/libstdc++.so.6 \
        /usr/lib64/libstdc++.so.6 \
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6; do
        if [[ -f "${candidate}" ]]; then
            libstdcpp="${candidate}"
            break
        fi
    done

    if [[ -n "${libstdcpp}" ]]; then
        export LD_PRELOAD="${libstdcpp}${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi
    if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
        export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi

    export HCCL_IF_IP="${LOCAL_IP}"
    export GLOO_SOCKET_IFNAME="${NIC_NAME}"
    export TP_SOCKET_IFNAME="${NIC_NAME}"
    export HCCL_SOCKET_IFNAME="${NIC_NAME}"
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
    export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
    export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"
    export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1
}

setup_turboquant_runtime_env() {
    case "${TURBOQUANT_KV_BITS:-[8, 8]}" in
        "[4, 4]"|"[4,4]")
            export VLLM_ASCEND_TURBOQUANT_ENCODE_OP=1
            export VLLM_ASCEND_TURBOQUANT_DECODE_OP=1
            export VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE="${VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE:-1}"
            export VLLM_ASCEND_TURBOQUANT_MSE_IMPL="${VLLM_ASCEND_TURBOQUANT_MSE_IMPL:-v1}"
            ;;
        *)
            export VLLM_ASCEND_TURBOQUANT_ENCODE_OP="${VLLM_ASCEND_TURBOQUANT_ENCODE_OP:-1}"
            export VLLM_ASCEND_TURBOQUANT_DECODE_OP="${VLLM_ASCEND_TURBOQUANT_DECODE_OP:-0}"
            export VLLM_ASCEND_TURBOQUANT_PACK_OP="${VLLM_ASCEND_TURBOQUANT_PACK_OP:-v3}"
            export VLLM_ASCEND_TURBOQUANT_ATTENTION_OP_8BIT="${VLLM_ASCEND_TURBOQUANT_ATTENTION_OP_8BIT:-0}"
            export VLLM_ASCEND_TURBOQUANT_FUSED_FIA_8BIT="${VLLM_ASCEND_TURBOQUANT_FUSED_FIA_8BIT:-0}"
            ;;
    esac
}
