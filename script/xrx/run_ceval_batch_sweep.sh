#!/usr/bin/env bash
# ceval computer_network 并发压测：不同 --eval-batch-size；
# 每个 batch_size 启一次 vllm serve，客户端连跑 ROUNDS 轮后再停服。
# 多 batch_size 时可在多卡并行：BATCH_SIZES / VISIBLE_DEVICES / PORTS 一一对应。
#
# 用法:
#   bash script/xrx/run_ceval_batch_sweep.sh
#   BATCH_SIZES="16" ROUNDS=3 bash script/xrx/run_ceval_batch_sweep.sh
#   # 三卡并行三个 batch_size：
#   BATCH_SIZES="1 8 16" VISIBLE_DEVICES="5 6 7" PORTS="9555 9556 9557" \
#     ROUNDS=3 bash script/xrx/run_ceval_batch_sweep.sh
#
# 环境变量（均可覆盖）:
#   VISIBLE_DEVICES             NPU 卡号列表，默认 "7"；与 BATCH_SIZES/PORTS 等长一一对应
#   PORTS                       端口列表，默认取 PORT（9555）；与 BATCH_SIZES 等长
#   PORT                        单端口时的默认值（兼容旧用法），默认 9555
#   BLOCK_SIZE                  vllm --block-size，默认 128（所有 serve 共用）
#   VLLM_ASCEND_BIT_RESIDUAL_FIA            PrefillCacheHit/ChunkedPrefill FIA，默认 1
#   VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA     DecodeOnly FIA，默认 1
#   VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA    PrefillNoCache paged FIA，默认 0
#   SERVE_MODEL_PATH            服务加载权重，默认 /root/cyl/model/Qwen3-4B
#   EVAL_MODEL_PATH             evalscope --model，默认与 SERVE_MODEL_PATH 相同
#   GPU_MEMORY_UTILIZATION      默认 0.3
#   MAX_MODEL_LEN               默认 8500
#   BATCH_SIZES                 默认 "16"（空格分隔；多项时与卡/端口一一对应并行）
#   ROUNDS                      每个 batch_size 下客户端压测轮数，默认 3（同一 serve）
#   CEVAL_LOCAL_PATH            ceval 本地数据集根目录，默认 /root/cyl/dataset
#   CEVAL_SUBSET                ceval subset；默认 computer_network。
#                               设为空（CEVAL_SUBSET=）则不传 subset_list，跑全量子集
#   WORK_ROOT                   evalscope 结果根目录，默认 ./output/ceval_batch_sweep_<ts>
#   LOG_ROOT                    日志根目录，默认 ${WORK_ROOT}/logs
#   READY_TIMEOUT               服务就绪等待秒数，默认 600
#   SOURCE_ENV                  可选，默认仓库 infoenvs（存在则 source）
#   PARALLEL                    多 batch_size 是否并行，默认 1（1=并行，0=串行）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------- defaults（infoenvs 会覆盖 ASCEND_*，故设备/FIA 在 source 后再强制套一次）----------
VISIBLE_DEVICES="${VISIBLE_DEVICES:-7}"
_FIA="${VLLM_ASCEND_BIT_RESIDUAL_FIA:-1}"
_DECODE_FIA="${VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA:-1}"
_NOCACHE_FIA="${VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA:-0}"

SERVE_MODEL_PATH="${SERVE_MODEL_PATH:-${MODEL_PATH:-/root/cyl/model/Qwen3-4B}}"
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-${SERVE_MODEL_PATH}}"
PORT="${PORT:-9555}"
PORTS="${PORTS:-${PORT}}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8500}"
BATCH_SIZES="${BATCH_SIZES:-16}"
ROUNDS="${ROUNDS:-3}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
CEVAL_LOCAL_PATH="${CEVAL_LOCAL_PATH:-/root/cyl/dataset}"
CEVAL_SUBSET="${CEVAL_SUBSET:-computer_network}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
PARALLEL="${PARALLEL:-1}"

STAMP="$(date '+%Y%m%d_%H%M%S')"
WORK_ROOT="${WORK_ROOT:-${PWD}/output/ceval_batch_sweep_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs}"

# ---------- CLI ----------
usage() {
    sed -n '2,35p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --device|--devices)
            VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --ports)
            PORTS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            PORTS="$2"
            shift 2
            ;;
        --block-size)
            BLOCK_SIZE="$2"
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
        --work-root)
            WORK_ROOT="$2"
            LOG_ROOT="${WORK_ROOT}/logs"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1" >&2
            usage
            ;;
    esac
done

mkdir -p "${LOG_ROOT}" "${WORK_ROOT}"

SWEEP_LOG="${LOG_ROOT}/sweep.log"

log() {
    printf '%s [ceval-sweep] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${SWEEP_LOG}"
}

wlog() {
    local msg
    msg="$(printf '%s [ceval-sweep%s] %s' "$(date '+%Y-%m-%d %H:%M:%S')" "${WORKER_TAG:+ ${WORKER_TAG}}" "$*")"
    echo "${msg}" | tee -a "${SWEEP_LOG}" ${WORKER_LOG:+-a "${WORKER_LOG}"}
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

stop_serve_on() {
    local pid_file="$1"
    local port="$2"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        rm -f "${pid_file}"
        if process_alive "${pid}"; then
            wlog "停止 vllm serve pid=${pid} port=${port}"
            kill -TERM -- "-${pid}" >/dev/null 2>&1 || kill -TERM "${pid}" >/dev/null 2>&1 || true
            for _ in $(seq 1 40); do
                if ! process_alive "${pid}"; then
                    break
                fi
                sleep 0.5
            done
            if process_alive "${pid}"; then
                wlog "强制 SIGKILL pid=${pid}"
                kill -KILL -- "-${pid}" >/dev/null 2>&1 || kill -KILL "${pid}" >/dev/null 2>&1 || true
            fi
        fi
    fi
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    elif command -v lsof >/dev/null 2>&1; then
        local pids
        pids="$(lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
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
    local timeout="$1"
    local url="$2"
    local pid_file="$3"
    local t0
    t0="$(date +%s)"
    wlog "等待服务就绪: ${url} (timeout=${timeout}s)"
    while true; do
        if curl -sf "${url}" >/dev/null 2>&1; then
            wlog "服务已就绪"
            return 0
        fi
        if ! process_alive "$(cat "${pid_file}" 2>/dev/null || true)"; then
            wlog "ERROR: serve 进程已退出，见日志 ${CURRENT_SERVE_LOG:-${LOG_ROOT}}"
            return 1
        fi
        if (( "$(date +%s)" - t0 >= timeout )); then
            wlog "ERROR: 等待就绪超时 ${timeout}s"
            return 1
        fi
        sleep 2
    done
}

start_serve() {
    local tag="$1"
    local device="$2"
    local port="$3"
    local worker_dir="$4"
    local pid_file="${worker_dir}/vllm_serve.pid"
    local serve_log="${worker_dir}/serve_${tag}.log"
    CURRENT_SERVE_LOG="${serve_log}"

    stop_serve_on "${pid_file}" "${port}"

    if [[ ! -e "${SERVE_MODEL_PATH}" ]]; then
        wlog "ERROR: SERVE_MODEL_PATH 不存在: ${SERVE_MODEL_PATH}"
        return 1
    fi

    wlog "启动 vllm serve -> ${serve_log}"
    wlog "  model=${SERVE_MODEL_PATH} port=${port} block_size=${BLOCK_SIZE} device=${device}"
    wlog "  util=${GPU_MEMORY_UTILIZATION} max_len=${MAX_MODEL_LEN}"
    wlog "  FIA=${VLLM_ASCEND_BIT_RESIDUAL_FIA}/${VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA}/${VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA}"

    (
        export ASCEND_RT_VISIBLE_DEVICES="${device}"
        export VLLM_ASCEND_BIT_RESIDUAL_FIA
        export VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA
        export VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA
        export MODEL_PATH="${SERVE_MODEL_PATH}"
        exec setsid vllm serve "${SERVE_MODEL_PATH}" \
            --kv_cache_dtype=turboquant \
            --additional-config='{"turboquant_kv_bits": [8, 4]}' \
            --port "${port}" \
            --block-size "${BLOCK_SIZE}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --no-enable-prefix-caching \
            --kv-cache-metrics
    ) >"${serve_log}" 2>&1 &
    local pid=$!
    echo "${pid}" > "${pid_file}"
    sleep 2
    if ! process_alive "${pid}"; then
        wlog "ERROR: serve 启动后立即退出，tail:"
        tail -n 80 "${serve_log}" || true
        return 1
    fi
    wlog "serve pid=${pid}"
    wait_ready "${READY_TIMEOUT}" "http://127.0.0.1:${port}/health" "${pid_file}"
}

run_eval() {
    local batch_size="$1"
    local round="$2"
    local port="$3"
    local worker_dir="$4"
    local tag="bs${batch_size}_r${round}"
    local eval_log="${worker_dir}/eval_${tag}.log"
    local work_dir="${WORK_ROOT}/bs${batch_size}/${tag}"
    local api_url="http://127.0.0.1:${port}/v1/chat/completions"

    mkdir -p "${work_dir}"

    local eval_model="${EVAL_MODEL_PATH}"
    if [[ ! -e "${eval_model}" ]]; then
        wlog "WARN: EVAL_MODEL_PATH 不存在: ${eval_model}，改用 SERVE_MODEL_PATH"
        eval_model="${SERVE_MODEL_PATH}"
    fi

    local dataset_args
    if [[ -n "${CEVAL_SUBSET}" ]]; then
        dataset_args="$(printf '{"ceval": {"local_path": "%s", "subset_list": ["%s"]}}' \
            "${CEVAL_LOCAL_PATH}" "${CEVAL_SUBSET}")"
    else
        # 不配 subset_list：evalscope 跑 ceval 全量子集
        dataset_args="$(printf '{"ceval": {"local_path": "%s"}}' \
            "${CEVAL_LOCAL_PATH}")"
    fi
    local gen_config
    gen_config="$(printf '{"temperature": %s, "max_tokens": %s}' \
        "${TEMPERATURE}" "${MAX_TOKENS}")"

    wlog "开始 eval: batch_size=${batch_size} round=${round}/${ROUNDS}"
    wlog "  api=${api_url} work_dir=${work_dir} eval_log=${eval_log}"
    wlog "  ceval subset=${CEVAL_SUBSET:-<all>}"

    set +e
    evalscope eval \
        --model "${eval_model}" \
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
        wlog "ERROR: eval 失败 rc=${rc}，见 ${eval_log}"
        tail -n 40 "${eval_log}" || true
        return "${rc}"
    fi
    wlog "eval 完成: ${tag}"
    return 0
}

# One batch_size on one device/port: start serve once, run ROUNDS client evals.
run_worker() {
    local batch_size="$1"
    local device="$2"
    local port="$3"
    local worker_dir="${LOG_ROOT}/bs${batch_size}_dev${device}_port${port}"
    local pid_file="${worker_dir}/vllm_serve.pid"
    local fail=0

    mkdir -p "${worker_dir}"
    WORKER_TAG="bs${batch_size}/dev${device}/port${port}"
    WORKER_LOG="${worker_dir}/worker.log"

    wlog "======== worker start batch_size=${batch_size} device=${device} port=${port} ========"
    if ! start_serve "bs${batch_size}" "${device}" "${port}" "${worker_dir}"; then
        wlog "ERROR: worker serve 启动失败"
        echo 1 > "${worker_dir}/fail_count"
        return 1
    fi

    for ((r = 1; r <= ROUNDS; r++)); do
        wlog "-------- client bs${batch_size} r${r} --------"
        if ! run_eval "${batch_size}" "${r}" "${port}" "${worker_dir}"; then
            fail=$((fail + 1))
            wlog "本轮失败，继续下一轮（服务保持运行）"
        fi
    done

    stop_serve_on "${pid_file}" "${port}"
    echo "${fail}" > "${worker_dir}/fail_count"
    wlog "======== worker done batch_size=${batch_size} fail=${fail} ========"
    return 0
}

cleanup_all() {
    log "清理：停止所有 serve"
    local pf port
    for pf in "${LOG_ROOT}"/*/vllm_serve.pid; do
        [[ -f "${pf}" ]] || continue
        port="$(basename "$(dirname "${pf}")" | sed -n 's/.*_port\([0-9][0-9]*\)$/\1/p')"
        if [[ -z "${port}" ]]; then
            port="${PORT}"
        fi
        WORKER_TAG="" WORKER_LOG="" stop_serve_on "${pf}" "${port}" || true
    done
}
trap cleanup_all EXIT

# ---------- main ----------
source_runtime_env
export VLLM_ASCEND_BIT_RESIDUAL_FIA="${_FIA}"
export VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA="${_DECODE_FIA}"
export VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA="${_NOCACHE_FIA}"

# shellcheck disable=SC2206
BS_ARR=(${BATCH_SIZES})
# shellcheck disable=SC2206
DEV_ARR=(${VISIBLE_DEVICES})
# shellcheck disable=SC2206
PORT_ARR=(${PORTS})

n_bs=${#BS_ARR[@]}
n_dev=${#DEV_ARR[@]}
n_port=${#PORT_ARR[@]}

if [[ ${n_bs} -gt 1 && ${n_dev} -eq 1 ]]; then
    log "ERROR: 多 BATCH_SIZES (${n_bs}) 需要等长 VISIBLE_DEVICES（当前 ${n_dev}），以便卡一一对应"
    exit 1
fi
if [[ ${n_bs} -gt 1 && ${n_port} -eq 1 ]]; then
    log "ERROR: 多 BATCH_SIZES (${n_bs}) 需要等长 PORTS（当前 ${n_port}），以便端口一一对应"
    exit 1
fi
if [[ ${n_bs} -ne ${n_dev} || ${n_bs} -ne ${n_port} ]]; then
    log "ERROR: BATCH_SIZES/VISIBLE_DEVICES/PORTS 长度须一致: ${n_bs}/${n_dev}/${n_port}"
    log "  BATCH_SIZES=${BATCH_SIZES}"
    log "  VISIBLE_DEVICES=${VISIBLE_DEVICES}"
    log "  PORTS=${PORTS}"
    exit 1
fi

if [[ "${PARALLEL}" == "1" && ${n_bs} -gt 1 ]]; then
    for ((i = 0; i < n_bs; i++)); do
        for ((j = i + 1; j < n_bs; j++)); do
            if [[ "${DEV_ARR[i]}" == "${DEV_ARR[j]}" ]]; then
                log "ERROR: 并行时 VISIBLE_DEVICES 不能重复: ${DEV_ARR[i]}"
                exit 1
            fi
            if [[ "${PORT_ARR[i]}" == "${PORT_ARR[j]}" ]]; then
                log "ERROR: 并行时 PORTS 不能重复: ${PORT_ARR[i]}"
                exit 1
            fi
        done
    done
fi

log "WORK_ROOT=${WORK_ROOT}"
log "LOG_ROOT=${LOG_ROOT}"
log "BLOCK_SIZE=${BLOCK_SIZE} FIA=${_FIA}/${_DECODE_FIA}/${_NOCACHE_FIA} ROUNDS=${ROUNDS} PARALLEL=${PARALLEL}"
log "CEVAL_LOCAL_PATH=${CEVAL_LOCAL_PATH} CEVAL_SUBSET=${CEVAL_SUBSET:-<all>}"
for ((i = 0; i < n_bs; i++)); do
    log "  slot[${i}]: batch_size=${BS_ARR[i]} device=${DEV_ARR[i]} port=${PORT_ARR[i]}"
done

worker_pids=()
if [[ "${PARALLEL}" == "1" && ${n_bs} -gt 1 ]]; then
    for ((i = 0; i < n_bs; i++)); do
        run_worker "${BS_ARR[i]}" "${DEV_ARR[i]}" "${PORT_ARR[i]}" &
        wpid=$!
        worker_pids+=("${wpid}")
        log "spawned worker pid=${wpid} bs=${BS_ARR[i]} dev=${DEV_ARR[i]} port=${PORT_ARR[i]}"
    done
    for pid in "${worker_pids[@]}"; do
        wait "${pid}" || true
    done
else
    for ((i = 0; i < n_bs; i++)); do
        run_worker "${BS_ARR[i]}" "${DEV_ARR[i]}" "${PORT_ARR[i]}" || true
    done
fi

fail_count=0
for ((i = 0; i < n_bs; i++)); do
    fc_file="${LOG_ROOT}/bs${BS_ARR[i]}_dev${DEV_ARR[i]}_port${PORT_ARR[i]}/fail_count"
    if [[ -f "${fc_file}" ]]; then
        fail_count=$((fail_count + $(cat "${fc_file}")))
    else
        fail_count=$((fail_count + 1))
        log "WARN: missing fail_count for bs${BS_ARR[i]}"
    fi
done

log "全部结束 fail_count=${fail_count} logs=${LOG_ROOT}"
if [[ ${fail_count} -gt 0 ]]; then
    exit 1
fi
exit 0
