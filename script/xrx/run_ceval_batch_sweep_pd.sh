#!/usr/bin/env bash
# C-Eval batch sweep over 1P1D PD stack (proxy → evalscope / ceval_full_eval).
#
# 用法（对齐 run_ceval_batch_sweep.sh 环境变量）:
#   PD_DEVICES="6,7" PROXY_PORT=9555 BATCH_SIZES="16" GPU_MEMORY_UTILIZATION=0.2 \
#     SERVE_MODEL_PATH=/root/yyz/models/Qwen3-4B \
#     CEVAL_LOCAL_PATH=/root/yyz/datasets/ceval \
#     bash script/xrx/run_ceval_batch_sweep_pd.sh --rounds 2
#
# 环境变量:
#   PD_DEVICES          prefill,decode NPU，默认 "6,7"
#   PROXY_PORT          默认 9555
#   BATCH_SIZES         默认 16（仅 evalscope 并发；fallback 脚本串行）
#   SERVE_MODEL_PATH    默认 /root/yyz/models/Qwen3-4B
#   CEVAL_LOCAL_PATH    默认 /root/yyz/datasets/ceval
#   CEVAL_SUBSET        默认 computer_network；空串跑 52 子集
#   GPU_MEMORY_UTILIZATION / MAX_MODEL_LEN / BLOCK_SIZE
#   SERVE_EXTRA         追加 vllm serve 参数（如 -cc.cudagraph_mode=PIECEWISE）
#   FIA_MODE            off|prefill|decode|all，默认 all
#   ROUNDS              默认 2
#   SOURCE_ENV          默认 ${REPO_ROOT}/infoenvs
#   EVAL_BACKEND        evalscope|lite，默认 evalscope；无 evalscope 时自动 lite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CEVAL_PIN="${CEVAL_PIN:-/root/yyz/perflog/k8v4_serve_bench/gs1_bind/ceval_pin}"
PD_STACK_PY="${CEVAL_PIN}/ceval_pd_stack.py"
LITE_EVAL_PY="${CEVAL_PIN}/ceval_full_eval.py"

SERVE_MODEL_PATH="${SERVE_MODEL_PATH:-/root/yyz/models/Qwen3-4B}"
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-${SERVE_MODEL_PATH}}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-turboquant}"
TURBOQUANT_KV_BITS="${TURBOQUANT_KV_BITS:-8,4}"
PD_DEVICES="${PD_DEVICES:-6,7}"
PROXY_PORT="${PROXY_PORT:-${PORT:-9555}}"
PREFILL_PORT="${PREFILL_PORT:-9001}"
DECODE_PORT="${DECODE_PORT:-9011}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-20011}"
DECODE_KV_PORT="${DECODE_KV_PORT:-20021}"
PD_NIC="${PD_NIC:-}"
PD_LOCAL_IP="${PD_LOCAL_IP:-}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8500}"
BATCH_SIZES="${BATCH_SIZES:-16}"
ROUNDS="${ROUNDS:-2}"
CEVAL_LOCAL_PATH="${CEVAL_LOCAL_PATH:-/root/yyz/datasets/ceval}"
CEVAL_SUBSET="${CEVAL_SUBSET-computer_network}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
FIA_MODE="${FIA_MODE:-all}"
SERVE_EXTRA="${SERVE_EXTRA:-}"
SOURCE_ENV="${SOURCE_ENV:-${REPO_ROOT}/infoenvs}"
EVAL_BACKEND="${EVAL_BACKEND:-evalscope}"
ASCEND_REPO="${ASCEND_REPO:-/vllm-workspace/vllm-ascend}"
export ASCEND_REPO VLLM_VERSION="${VLLM_VERSION:-0.23.0}"

STAMP="$(date '+%Y%m%d_%H%M%S')"
WORK_ROOT="${WORK_ROOT:-${PWD}/output/ceval_pd_sweep_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds) ROUNDS="$2"; shift 2 ;;
        --work-root) WORK_ROOT="$2"; LOG_ROOT="${WORK_ROOT}/logs"; shift 2 ;;
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${LOG_ROOT}" "${WORK_ROOT}"
SWEEP_LOG="${LOG_ROOT}/sweep.log"

log() {
    printf '%s [ceval-pd-sweep] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${SWEEP_LOG}"
}

source_runtime_env() {
    local env_file="${SOURCE_ENV}"
    if [[ -n "${env_file}" && "${env_file}" != "none" && -f "${env_file}" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "${env_file}"
        set -u
        log "sourced env: ${env_file}"
    fi
    export PYTHONPATH="/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:${PYTHONPATH:-}"
    if [[ -f /usr/local/Ascend/cann-9.1.0/set_env.sh ]]; then
        set +u
        # shellcheck disable=SC1091
        source /usr/local/Ascend/cann-9.1.0/set_env.sh
        set -u
    fi
    local custom_env="/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
    if [[ -f "${custom_env}" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "${custom_env}" || true
        set -u
    fi
}

detect_pd_network() {
    if [[ -n "${PD_NIC}" && -n "${PD_LOCAL_IP}" ]]; then
        return 0
    fi
    if command -v ip >/dev/null 2>&1; then
        while read -r nic _; do
            [[ "${nic}" == lo ]] && continue
            local ip
            ip="$(ip -4 addr show dev "${nic}" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)"
            if [[ -n "${ip}" && "${ip}" != 127.* ]]; then
                PD_NIC="${PD_NIC:-${nic}}"
                PD_LOCAL_IP="${PD_LOCAL_IP:-${ip}}"
                break
            fi
        done < <(ip -br link | awk '{print $1}')
    fi
    PD_NIC="${PD_NIC:-enp189s0f0}"
    PD_LOCAL_IP="${PD_LOCAL_IP:-192.168.9.160}"
}

build_serve_extra() {
    local base="--block-size ${BLOCK_SIZE} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} --max-model-len ${MAX_MODEL_LEN} --trust-remote-code"
    if [[ -n "${SERVE_EXTRA}" ]]; then
        SERVE_EXTRA_FINAL="${base} ${SERVE_EXTRA}"
    else
        SERVE_EXTRA_FINAL="${base}"
    fi
}

has_evalscope() {
    command -v evalscope >/dev/null 2>&1 && evalscope --help >/dev/null 2>&1
}

stop_pd_stack() {
    local worker_dir="$1"
    python3 "${PD_STACK_PY}" stop --log-root "${worker_dir}" 2>/dev/null || true
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${PROXY_PORT}/tcp" "${PREFILL_PORT}/tcp" "${DECODE_PORT}/tcp" >/dev/null 2>&1 || true
    fi
    sleep 2
}

start_pd_stack() {
    local worker_dir="$1"
    mkdir -p "${worker_dir}"
    stop_pd_stack "${worker_dir}"
    log "启动 PD 1P1D -> ${worker_dir}"
    log "  model=${SERVE_MODEL_PATH} P,D=${PD_DEVICES} proxy=${PROXY_PORT}"
    log "  util=${GPU_MEMORY_UTILIZATION} max_len=${MAX_MODEL_LEN} nic=${PD_NIC} ip=${PD_LOCAL_IP}"
    log "  serve_extra=${SERVE_EXTRA_FINAL}"

    python3 "${PD_STACK_PY}" start \
        --log-root "${worker_dir}" \
        --model "${SERVE_MODEL_PATH}" \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        --kv-bits "${TURBOQUANT_KV_BITS}" \
        --devices "${PD_DEVICES}" \
        --proxy-port "${PROXY_PORT}" \
        --prefill-port "${PREFILL_PORT}" \
        --decode-port "${DECODE_PORT}" \
        --prefill-kv-port "${PREFILL_KV_PORT}" \
        --decode-kv-port "${DECODE_KV_PORT}" \
        --pd-nic "${PD_NIC}" \
        --pd-local-ip "${PD_LOCAL_IP}" \
        --fia "${FIA_MODE}" \
        --serve-extra "${SERVE_EXTRA_FINAL}" \
        --ready-timeout "${READY_TIMEOUT}" \
        2>&1 | tee -a "${worker_dir}/pd_start.log"
}

run_eval_evalscope() {
    local batch_size="$1"
    local round="$2"
    local worker_dir="$3"
    local tag="bs${batch_size}_r${round}"
    local eval_log="${worker_dir}/eval_${tag}.log"
    local work_dir="${WORK_ROOT}/bs${batch_size}/${tag}"
    local api_url="http://127.0.0.1:${PROXY_PORT}/v1/chat/completions"

    mkdir -p "${work_dir}"
    local dataset_args
    if [[ -n "${CEVAL_SUBSET}" ]]; then
        dataset_args="$(printf '{"ceval": {"local_path": "%s", "subset_list": ["%s"]}}' \
            "${CEVAL_LOCAL_PATH}" "${CEVAL_SUBSET}")"
    else
        dataset_args="$(printf '{"ceval": {"local_path": "%s"}}' "${CEVAL_LOCAL_PATH}")"
    fi
    local gen_config
    gen_config="$(printf '{"temperature": %s, "max_tokens": %s}' "${TEMPERATURE}" "${MAX_TOKENS}")"

    log "evalscope bs=${batch_size} round=${round}/${ROUNDS}"
    set +e
    evalscope eval \
        --model "${EVAL_MODEL_PATH}" \
        --api-url "${api_url}" \
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
        log "ERROR evalscope rc=${rc} log=${eval_log}"
        tail -n 30 "${eval_log}" || true
        return "${rc}"
    fi
    local report
    report="$(find "${work_dir}" -path '*/reports/*/ceval.json' 2>/dev/null | head -1)"
    if [[ -n "${report}" ]]; then
        python3 -c "import json;d=json.load(open('${report}'));print('mean_acc', d.get('metrics',[{}])[0].get('macro_score'))"
    fi
    return 0
}

run_eval_lite() {
    local batch_size="$1"
    local round="$2"
    local worker_dir="$3"
    local tag="bs${batch_size}_r${round}"
    local eval_log="${worker_dir}/eval_${tag}.log"
    local work_dir="${WORK_ROOT}/bs${batch_size}/${tag}"
    local api_url="http://127.0.0.1:${PROXY_PORT}/v1/chat/completions"
    local subsets_arg=""

    mkdir -p "${work_dir}"
    if [[ -n "${CEVAL_SUBSET}" ]]; then
        subsets_arg="--subsets ${CEVAL_SUBSET}"
    fi

    log "ceval_full_eval round=${round}/${ROUNDS} (serial, bs=${batch_size} ignored)"
    set +e
    python3 -u "${LITE_EVAL_PY}" \
        --api-url "${api_url}" \
        --model "${EVAL_MODEL_PATH}" \
        --ceval-root "${CEVAL_LOCAL_PATH}" \
        --out-dir "${work_dir}/eval" \
        --max-tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --timeout 600 \
        ${subsets_arg} \
        >"${eval_log}" 2>&1
    local rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
        log "ERROR lite eval rc=${rc} log=${eval_log}"
        tail -n 30 "${eval_log}" || true
        return "${rc}"
    fi
    if [[ -f "${work_dir}/eval/ceval_report.json" ]]; then
        python3 -c "import json;d=json.load(open('${work_dir}/eval/ceval_report.json'));print('macro_mean_acc', d.get('macro_mean_acc'))"
    fi
    return 0
}

run_eval() {
    local batch_size="$1"
    local round="$2"
    local worker_dir="$3"
    if [[ "${EVAL_BACKEND}" == "evalscope" ]] && has_evalscope; then
        run_eval_evalscope "${batch_size}" "${round}" "${worker_dir}"
    else
        run_eval_lite "${batch_size}" "${round}" "${worker_dir}"
    fi
}

run_worker() {
    local batch_size="$1"
    local worker_dir="${LOG_ROOT}/bs${batch_size}_pd_${PD_DEVICES//,/_}_proxy${PROXY_PORT}"
    local fail=0
    mkdir -p "${worker_dir}"

    log "======== worker bs=${batch_size} PD=${PD_DEVICES} proxy=${PROXY_PORT} ========"
    if ! start_pd_stack "${worker_dir}"; then
        echo 1 > "${worker_dir}/fail_count"
        return 1
    fi

    for ((r = 1; r <= ROUNDS; r++)); do
        if ! run_eval "${batch_size}" "${r}" "${worker_dir}"; then
            fail=$((fail + 1))
        fi
    done

    stop_pd_stack "${worker_dir}"
    echo "${fail}" > "${worker_dir}/fail_count"
    log "======== worker done bs=${batch_size} fail=${fail} ========"
    return 0
}

cleanup_all() {
    log "清理 PD stack"
    for d in "${LOG_ROOT}"/bs*_pd_*; do
        [[ -d "${d}" ]] || continue
        stop_pd_stack "${d}" || true
    done
}
trap cleanup_all EXIT

source_runtime_env
export VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
export PYTHONPATH="/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:${PYTHONPATH:-}"
detect_pd_network
build_serve_extra

if [[ "${EVAL_BACKEND}" == "evalscope" ]] && ! has_evalscope; then
    log "WARN: evalscope 不可用，改用 ceval_full_eval (EVAL_BACKEND=lite)"
    EVAL_BACKEND="lite"
fi

if [[ ! -f "${PD_STACK_PY}" ]]; then
    log "ERROR: missing ${PD_STACK_PY}"
    exit 1
fi
if [[ "${EVAL_BACKEND}" == "lite" && ! -f "${LITE_EVAL_PY}" ]]; then
    log "ERROR: missing ${LITE_EVAL_PY}"
    exit 1
fi

log "WORK_ROOT=${WORK_ROOT}"
log "CEVAL=${CEVAL_LOCAL_PATH} subset=${CEVAL_SUBSET:-<all>} backend=${EVAL_BACKEND}"

fail_count=0
# shellcheck disable=SC2206
for bs in ${BATCH_SIZES}; do
    run_worker "${bs}" || true
    fc="${LOG_ROOT}/bs${bs}_pd_${PD_DEVICES//,/_}_proxy${PROXY_PORT}/fail_count"
    if [[ -f "${fc}" ]]; then
        fail_count=$((fail_count + $(cat "${fc}")))
    else
        fail_count=$((fail_count + 1))
    fi
done

log "全部结束 fail_count=${fail_count} WORK_ROOT=${WORK_ROOT}"
exit $(( fail_count > 0 ? 1 : 0 ))
