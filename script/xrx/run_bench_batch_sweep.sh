#!/usr/bin/env bash
# vllm bench serve 并发压测：不同 --max-concurrency；
# 每个 concurrency 启一次 vllm serve，客户端连跑 ROUNDS 轮后再停服。
# 服务启动与 run_ceval_batch_sweep.sh 一致；压测逻辑来自 script/xrx/pd/start_bench.sh。
# 多 concurrency 时可在多卡并行：MAX_CONCURRENCIES / VISIBLE_DEVICES / PORTS 一一对应。
#
# 用法:
#   bash script/xrx/run_bench_batch_sweep.sh
#   MAX_CONCURRENCIES="64" ROUNDS=1 bash script/xrx/run_bench_batch_sweep.sh
#   # 三卡并行三个 concurrency：
#   MAX_CONCURRENCIES="16 32 64" VISIBLE_DEVICES="5 6 7" PORTS="9555 9556 9557" \
#     ROUNDS=1 bash script/xrx/run_bench_batch_sweep.sh
#
# 环境变量（均可覆盖）:
#   VISIBLE_DEVICES             NPU 卡号列表，默认 "7"；与 MAX_CONCURRENCIES/PORTS 等长一一对应
#   PORTS                       端口列表，默认取 PORT（9555）；与 MAX_CONCURRENCIES 等长
#   PORT                        单端口时的默认值（兼容旧用法），默认 9555
#   BLOCK_SIZE                  vllm --block-size，默认 128（所有 serve 共用）
#   VLLM_ASCEND_BIT_RESIDUAL_FIA            PrefillCacheHit/ChunkedPrefill FIA，默认 1
#   VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA     DecodeOnly FIA，默认 1
#   VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA    PrefillNoCache paged FIA，默认 0
#   SERVE_MODEL_PATH            服务加载权重，默认 /root/cyl/model/Qwen3-4B
#   BENCH_MODEL_PATH            bench --model，默认与 SERVE_MODEL_PATH 相同
#   GPU_MEMORY_UTILIZATION      默认 0.3
#   MAX_MODEL_LEN               默认 8500
#   MAX_NUM_BATCHED_TOKENS      传给 vllm serve --max-num-batched-tokens；空=框架默认
#                               （常见默认 2048）。定位短/长 input TPOT 时建议 32768
#   ENFORCE_EAGER               设为 1 时给 vllm serve 加 --enforce-eager（关 ACL graph）
#   ENABLE_TQ                   1=turboquant k8v4（默认）；0=非 TQ（不传 kv_cache_dtype，
#                               等价框架默认 auto / fp KV）
#   TQ_KV_BITS                  ENABLE_TQ=1 时 additional-config turboquant_kv_bits，
#                               默认 "[8, 4]"
#   MAX_CONCURRENCIES           默认 "64"（空格分隔；多项时与卡/端口一一对应并行）
#   ROUNDS                      每个 concurrency 下客户端压测轮数，默认 1（同一 serve）
#   BENCH_INPUT_LEN             random dataset input_len；可为单值或空格分隔列表，默认 10
#                               列表长度须为 1（所有轮共用）或等于 ROUNDS（第 r 轮用第 r 项）
#   BENCH_OUTPUT_LEN            random dataset output_len；同上，默认 200
#   BENCH_NUM_PROMPTS           请求数，默认 1000
#   BENCH_REQUEST_RATE          请求速率，默认 inf
#   BENCH_ENDPOINT              默认 /v1/completions
#   BENCH_NUM_WARMUPS           传给 bench --num-warmups；profile 且非 timed 时默认 16
#                               （在 start_profile 前预热 ACL graph / 真实 batch）
#   BENCH_SAVE_DETAILED         1=--save-result --save-detailed；profile 且非 timed 时默认 1
#   AB_COMPARE                  1=串行跑 NOTQ+TQ 对齐 A/B，结束后 summarize + TPOT 分解表
#   AB_PROFILE_MODE             AB_COMPARE 子模式（可用 --ab-profile-mode）:
#                               mixed   = 默认，行为与旧 AB 相同（多 req 波次，易混 phase）
#                               decode  = 一波纯 decode 长窗（归因 TPOT：host vs device）
#                               prefill = 首波 prefill/mixed 短窗（归因 TTFT）
#                               别名: decode_only / prefill_only
#   AB_ONE_WAVE                 decode/prefill 模式下默认 1：BENCH_NUM_PROMPTS=首个
#                               max_concurrency（一波填满即停，避免换人再 prefill）
#   AB_SKIP_ANALYSIS            1=AB_COMPARE 时跳过 post analysis
#   AB_NO_PROFILE               1=AB_COMPARE 时仍跑 NOTQ/TQ，但不挂 profiler
#                               （用于无开销 e2e ITL 取证；默认 0=开 profile）
#   WORK_ROOT                   结果根目录，默认 ./output/bench_batch_sweep_<ts>
#                               AB_COMPARE=1 时默认 ./output/bench_ab_<mode>_<ts>
#   LOG_ROOT                    日志根目录，默认 ${WORK_ROOT}/logs
#   READY_TIMEOUT               服务就绪等待秒数，默认 600
#   SOURCE_ENV                  可选，默认仓库 infoenvs（存在则 source）
#   PARALLEL                    多 concurrency 是否并行，默认 1（1=并行，0=串行）
#
# Profiler（压测段采集 torch/Ascend trace）:
#   ENABLE_PROFILE              设为 1 开启：serve 挂 --profiler-config.*，
#                               压测期间 POST /start_profile 与 /stop_profile
#   PROFILE_DIR                 trace 根目录；默认 ${LOG_ROOT}/.../profiler（绝对路径）
#   PROFILE_IGNORE_FRONTEND     1=只采 EngineCore/NPU（ignore_frontend），默认 0
#   PROFILE_WITH_STACK          1=with_stack，默认 1
#   PROFILE_ACTIVE_ITERATIONS   传给 --profiler-config.active_iterations；空=框架默认(5)
#   PROFILE_DELAY_ITERATIONS    传给 --profiler-config.delay_iterations；start 后跳过 N engine step
#   PROFILE_DELAY_SEC           可选：bench 开始后延迟 N 秒再 start（不设则走
#                               `vllm bench serve --profile`：warmup 后 start、结束 stop）
#   PROFILE_DURATION_SEC        与 DELAY 联用：start 后再采 M 秒 stop；不设则 bench
#                               结束后再 stop
#
# 每轮不同 IO 长度示例:
#   ROUNDS=3 BENCH_INPUT_LEN="10 100 2000" BENCH_OUTPUT_LEN="200 200 500" \
#     bash script/xrx/run_bench_batch_sweep.sh
#   # 抬高 prefill budget，减弱 chunked-prefill 错峰对 TPOT 的影响:
#   MAX_NUM_BATCHED_TOKENS=32768 ROUNDS=2 BENCH_INPUT_LEN="200 2000" \
#     BENCH_OUTPUT_LEN="200 200" MAX_CONCURRENCIES=16 bash script/xrx/run_bench_batch_sweep.sh
#   # 压测全程 profile（warmup 后 start，跑完 stop）:
#   ENABLE_PROFILE=1 MAX_CONCURRENCIES=16 ROUNDS=1 bash script/xrx/run_bench_batch_sweep.sh
#   # NOTQ/TQ 对齐 A/B + summarize + TPOT 分解表（mixed，旧行为）:
#   AB_COMPARE=1 MAX_CONCURRENCIES=16 ROUNDS=1 \
#     BENCH_INPUT_LEN=2000 BENCH_OUTPUT_LEN=200 BENCH_NUM_PROMPTS=256 \
#     bash script/xrx/run_bench_batch_sweep.sh
#   # A/B decode-only 长窗（跳过首波 prefill，采 ~120 decode step）:
#   AB_COMPARE=1 --ab-profile-mode decode MAX_CONCURRENCIES=16 ROUNDS=1 \
#     BENCH_INPUT_LEN=2000 BENCH_OUTPUT_LEN=200 \
#     bash script/xrx/run_bench_batch_sweep.sh
#   # A/B prefill 短窗（只采首波 ~20 prefill/mixed step）:
#   AB_COMPARE=1 --ab-profile-mode prefill MAX_CONCURRENCIES=16 ROUNDS=1 \
#     BENCH_INPUT_LEN=2000 BENCH_OUTPUT_LEN=200 \
#     bash script/xrx/run_bench_batch_sweep.sh
#   # 压测中途采 30s（开跑 60s 后 start，再 30s stop）:
#   ENABLE_PROFILE=1 PROFILE_DELAY_SEC=60 PROFILE_DURATION_SEC=30 \
#     bash script/xrx/run_bench_batch_sweep.sh
#   # 非 TQ（fp KV baseline）:
#   ENABLE_TQ=0 bash script/xrx/run_bench_batch_sweep.sh
#   # 或: bash script/xrx/run_bench_batch_sweep.sh --no-tq

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------- defaults（infoenvs 会覆盖 ASCEND_*，故设备/FIA 在 source 后再强制套一次）----------
VISIBLE_DEVICES="${VISIBLE_DEVICES:-7}"
_FIA="${VLLM_ASCEND_BIT_RESIDUAL_FIA:-1}"
_DECODE_FIA="${VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA:-1}"
_NOCACHE_FIA="${VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA:-0}"

SERVE_MODEL_PATH="${SERVE_MODEL_PATH:-${MODEL_PATH:-/root/cyl/model/Qwen3-4B}}"
BENCH_MODEL_PATH="${BENCH_MODEL_PATH:-${SERVE_MODEL_PATH}}"
PORT="${PORT:-9555}"
PORTS="${PORTS:-${PORT}}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8500}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_TQ="${ENABLE_TQ:-1}"
TQ_KV_BITS="${TQ_KV_BITS:-[8, 4]}"
MAX_CONCURRENCIES="${MAX_CONCURRENCIES:-64}"
ROUNDS="${ROUNDS:-1}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-10}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-200}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-1000}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_ENDPOINT="${BENCH_ENDPOINT:-/v1/completions}"
BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS:-}"
BENCH_SAVE_DETAILED="${BENCH_SAVE_DETAILED:-}"
AB_COMPARE="${AB_COMPARE:-0}"
AB_PROFILE_MODE="${AB_PROFILE_MODE:-mixed}"
AB_ONE_WAVE="${AB_ONE_WAVE:-}"
AB_SKIP_ANALYSIS="${AB_SKIP_ANALYSIS:-0}"
AB_NO_PROFILE="${AB_NO_PROFILE:-0}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
PARALLEL="${PARALLEL:-1}"
ENABLE_PROFILE="${ENABLE_PROFILE:-0}"
PROFILE_DIR="${PROFILE_DIR:-}"
PROFILE_IGNORE_FRONTEND="${PROFILE_IGNORE_FRONTEND:-0}"
PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-1}"
PROFILE_ACTIVE_ITERATIONS="${PROFILE_ACTIVE_ITERATIONS:-}"
PROFILE_DELAY_ITERATIONS="${PROFILE_DELAY_ITERATIONS:-}"
PROFILE_DELAY_SEC="${PROFILE_DELAY_SEC:-}"
PROFILE_DURATION_SEC="${PROFILE_DURATION_SEC:-}"

STAMP="$(date '+%Y%m%d_%H%M%S')"
_WORK_ROOT_FROM_ENV=0
if [[ -n "${WORK_ROOT:-}" ]]; then
    _WORK_ROOT_FROM_ENV=1
fi
WORK_ROOT="${WORK_ROOT:-${PWD}/output/bench_batch_sweep_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs}"

# ---------- CLI ----------
usage() {
    sed -n '2,95p' "$0" | sed 's/^# \?//'
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
        --max-num-batched-tokens)
            MAX_NUM_BATCHED_TOKENS="$2"
            shift 2
            ;;
        --max-concurrencies|--concurrencies)
            MAX_CONCURRENCIES="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --work-root)
            WORK_ROOT="$2"
            LOG_ROOT="${WORK_ROOT}/logs"
            _WORK_ROOT_FROM_ENV=1
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --profile)
            ENABLE_PROFILE=1
            shift
            ;;
        --profile-dir)
            PROFILE_DIR="$2"
            shift 2
            ;;
        --profile-delay)
            PROFILE_DELAY_SEC="$2"
            shift 2
            ;;
        --profile-duration)
            PROFILE_DURATION_SEC="$2"
            shift 2
            ;;
        --profiler-ignore-frontend)
            PROFILE_IGNORE_FRONTEND=1
            shift
            ;;
        --profiler-active-iterations)
            PROFILE_ACTIVE_ITERATIONS="$2"
            shift 2
            ;;
        --no-tq|--disable-tq)
            ENABLE_TQ=0
            shift
            ;;
        --tq|--enable-tq)
            ENABLE_TQ=1
            shift
            ;;
        --tq-kv-bits)
            TQ_KV_BITS="$2"
            shift 2
            ;;
        --num-warmups)
            BENCH_NUM_WARMUPS="$2"
            shift 2
            ;;
        --save-detailed)
            BENCH_SAVE_DETAILED=1
            shift
            ;;
        --ab-compare)
            AB_COMPARE=1
            shift
            ;;
        --ab-profile-mode)
            AB_PROFILE_MODE="$2"
            shift 2
            ;;
        --profile-delay-iterations)
            PROFILE_DELAY_ITERATIONS="$2"
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
    printf '%s [bench-sweep] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${SWEEP_LOG}"
}

wlog() {
    local msg
    msg="$(printf '%s [bench-sweep%s] %s' "$(date '+%Y-%m-%d %H:%M:%S')" "${WORKER_TAG:+ ${WORKER_TAG}}" "$*")"
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
    local profiler_dir="${5:-}"
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
    wlog "  max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-<default>} enforce_eager=${ENFORCE_EAGER}"
    wlog "  ENABLE_TQ=${ENABLE_TQ} TQ_KV_BITS=${TQ_KV_BITS}"
    wlog "  FIA=${VLLM_ASCEND_BIT_RESIDUAL_FIA}/${VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA}/${VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA}"

    local batched_args=()
    if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
        batched_args+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
    fi
    local eager_args=()
    if [[ "${ENFORCE_EAGER}" == "1" ]]; then
        eager_args+=(--enforce-eager)
    fi
    local profiler_args=()
    if [[ -n "${profiler_dir}" ]]; then
        mkdir -p "${profiler_dir}"
        profiler_dir="$(cd "${profiler_dir}" && pwd)"
        profiler_args+=(
            --profiler-config.profiler=torch
            --profiler-config.torch_profiler_dir="${profiler_dir}"
        )
        if [[ "${PROFILE_WITH_STACK}" == "1" ]]; then
            profiler_args+=(--profiler-config.torch_profiler_with_stack=true)
        fi
        if [[ "${PROFILE_IGNORE_FRONTEND}" == "1" ]]; then
            profiler_args+=(--profiler-config.ignore_frontend=true)
        fi
        if [[ -n "${PROFILE_ACTIVE_ITERATIONS}" ]]; then
            if ! [[ "${PROFILE_ACTIVE_ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
                wlog "ERROR: PROFILE_ACTIVE_ITERATIONS 须为正整数，当前=${PROFILE_ACTIVE_ITERATIONS}"
                return 1
            fi
            profiler_args+=(--profiler-config.active_iterations="${PROFILE_ACTIVE_ITERATIONS}")
        fi
        if [[ -n "${PROFILE_DELAY_ITERATIONS}" ]]; then
            if ! [[ "${PROFILE_DELAY_ITERATIONS}" =~ ^[0-9]+$ ]]; then
                wlog "ERROR: PROFILE_DELAY_ITERATIONS 须为非负整数，当前=${PROFILE_DELAY_ITERATIONS}"
                return 1
            fi
            profiler_args+=(--profiler-config.delay_iterations="${PROFILE_DELAY_ITERATIONS}")
        fi
        wlog "  profiler_dir=${profiler_dir} ignore_frontend=${PROFILE_IGNORE_FRONTEND} with_stack=${PROFILE_WITH_STACK} active_iterations=${PROFILE_ACTIVE_ITERATIONS:-<default>} delay_iterations=${PROFILE_DELAY_ITERATIONS:-0}"
    fi
    # TQ: --kv_cache_dtype=turboquant + bits；非 TQ: 不传 dtype（框架默认 auto/fp KV）。
    local tq_args=()
    if [[ "${ENABLE_TQ}" == "1" ]]; then
        tq_args+=(
            --kv_cache_dtype=turboquant
            --additional-config="{\"turboquant_kv_bits\": ${TQ_KV_BITS}}"
        )
    fi

    (
        export ASCEND_RT_VISIBLE_DEVICES="${device}"
        export VLLM_ASCEND_BIT_RESIDUAL_FIA
        export VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA
        export VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA
        # Optional NaN/topk probe (set by caller): inherit into EngineCore.
        export VLLM_ASCEND_TOPK_DEBUG="${VLLM_ASCEND_TOPK_DEBUG:-0}"
        export VLLM_ASCEND_TOPK_DEBUG_PATH="${VLLM_ASCEND_TOPK_DEBUG_PATH:-}"
        export MODEL_PATH="${SERVE_MODEL_PATH}"
        exec setsid vllm serve "${SERVE_MODEL_PATH}" \
            "${tq_args[@]}" \
            --port "${port}" \
            --block-size "${BLOCK_SIZE}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            "${batched_args[@]}" \
            "${eager_args[@]}" \
            "${profiler_args[@]}" \
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

post_profile() {
    # POST /start_profile or /stop_profile; returns 0 on HTTP success.
    local port="$1"
    local action="$2" # start_profile | stop_profile
    local url="http://127.0.0.1:${port}/${action}"
    local tmp_out tmp_err http_code
    tmp_out="$(mktemp)"
    tmp_err="$(mktemp)"
    http_code="$(curl -sS -o "${tmp_out}" -w '%{http_code}' \
        -X POST "${url}" 2>"${tmp_err}" || true)"
    if [[ "${http_code}" == "200" ]]; then
        wlog "POST /${action} ok (port=${port})"
        rm -f "${tmp_out}" "${tmp_err}"
        return 0
    fi
    wlog "WARN: POST /${action} failed http=${http_code:-na} port=${port}"
    if [[ -s "${tmp_err}" ]]; then
        wlog "  curl err: $(head -c 200 "${tmp_err}")"
    fi
    if [[ -s "${tmp_out}" ]]; then
        wlog "  body: $(head -c 200 "${tmp_out}")"
    fi
    rm -f "${tmp_out}" "${tmp_err}"
    return 1
}

# Background: sleep DELAY, start_profile, optional DURATION then stop_profile.
# Writes pid to $1; caller must wait/kill and ensure final stop if needed.
start_timed_profile_watcher() {
    local pid_out="$1"
    local port="$2"
    local delay_sec="$3"
    local duration_sec="${4:-}"
    (
        if [[ -n "${delay_sec}" && "${delay_sec}" != "0" ]]; then
            sleep "${delay_sec}"
        fi
        post_profile "${port}" "start_profile" || true
        if [[ -n "${duration_sec}" ]]; then
            sleep "${duration_sec}"
            post_profile "${port}" "stop_profile" || true
        fi
    ) &
    echo $! > "${pid_out}"
}

# Expand scalar-or-list IO lens to exactly ROUNDS entries (1-indexed via arr[r-1]).
# shellcheck disable=SC2206
_INPUT_LEN_ARR=(${BENCH_INPUT_LEN})
# shellcheck disable=SC2206
_OUTPUT_LEN_ARR=(${BENCH_OUTPUT_LEN})

expand_io_lens_for_rounds() {
    local n_in=${#_INPUT_LEN_ARR[@]}
    local n_out=${#_OUTPUT_LEN_ARR[@]}
    local r i

    if [[ ${n_in} -eq 1 ]]; then
        local single_in="${_INPUT_LEN_ARR[0]}"
        _INPUT_LEN_ARR=()
        for ((i = 0; i < ROUNDS; i++)); do
            _INPUT_LEN_ARR+=("${single_in}")
        done
    elif [[ ${n_in} -ne "${ROUNDS}" ]]; then
        log "ERROR: BENCH_INPUT_LEN 项数须为 1 或等于 ROUNDS=${ROUNDS}，当前 ${n_in}: ${BENCH_INPUT_LEN}"
        exit 1
    fi

    if [[ ${n_out} -eq 1 ]]; then
        local single_out="${_OUTPUT_LEN_ARR[0]}"
        _OUTPUT_LEN_ARR=()
        for ((i = 0; i < ROUNDS; i++)); do
            _OUTPUT_LEN_ARR+=("${single_out}")
        done
    elif [[ ${n_out} -ne "${ROUNDS}" ]]; then
        log "ERROR: BENCH_OUTPUT_LEN 项数须为 1 或等于 ROUNDS=${ROUNDS}，当前 ${n_out}: ${BENCH_OUTPUT_LEN}"
        exit 1
    fi

    for ((r = 0; r < ROUNDS; r++)); do
        if ! [[ "${_INPUT_LEN_ARR[r]}" =~ ^[0-9]+$ && "${_OUTPUT_LEN_ARR[r]}" =~ ^[0-9]+$ ]]; then
            log "ERROR: 第 $((r + 1)) 轮 IO 长度须为正整数: input=${_INPUT_LEN_ARR[r]} output=${_OUTPUT_LEN_ARR[r]}"
            exit 1
        fi
    done
}

round_input_len() {
    local round="$1"
    echo "${_INPUT_LEN_ARR[$((round - 1))]}"
}

round_output_len() {
    local round="$1"
    echo "${_OUTPUT_LEN_ARR[$((round - 1))]}"
}

# Migrated from script/xrx/pd/start_bench.sh — targets this worker's serve port.
run_bench() {
    local max_concurrency="$1"
    local round="$2"
    local port="$3"
    local worker_dir="$4"
    local input_len output_len
    input_len="$(round_input_len "${round}")"
    output_len="$(round_output_len "${round}")"
    local tag="mc${max_concurrency}_r${round}_i${input_len}_o${output_len}"
    local bench_log="${worker_dir}/bench_${tag}.log"
    local work_dir="${WORK_ROOT}/mc${max_concurrency}/${tag}"

    mkdir -p "${work_dir}"

    local bench_model="${BENCH_MODEL_PATH}"
    if [[ ! -e "${bench_model}" ]]; then
        wlog "WARN: BENCH_MODEL_PATH 不存在: ${bench_model}，改用 SERVE_MODEL_PATH"
        bench_model="${SERVE_MODEL_PATH}"
    fi

    local profile_args=()
    local timed_profile=0
    local watcher_pid_file=""
    local watcher_pid=""
    if [[ "${ENABLE_PROFILE}" == "1" ]]; then
        if [[ -n "${PROFILE_DELAY_SEC}" ]]; then
            timed_profile=1
            watcher_pid_file="${worker_dir}/profile_watcher_${tag}.pid"
            wlog "timed profile: delay=${PROFILE_DELAY_SEC}s duration=${PROFILE_DURATION_SEC:-until-bench-end}"
            start_timed_profile_watcher \
                "${watcher_pid_file}" "${port}" \
                "${PROFILE_DELAY_SEC}" "${PROFILE_DURATION_SEC:-}"
            watcher_pid="$(cat "${watcher_pid_file}" 2>/dev/null || true)"
        else
            # vllm bench serve: POST /start_profile after warmup, /stop_profile after run.
            profile_args+=(--profile)
            wlog "bench --profile: warmup 后 start_profile，跑完 stop_profile"
        fi
    fi

    wlog "开始 bench: max_concurrency=${max_concurrency} round=${round}/${ROUNDS}"
    wlog "  host=127.0.0.1 port=${port} endpoint=${BENCH_ENDPOINT}"
    wlog "  input_len=${input_len} output_len=${output_len}"
    wlog "  num_prompts=${BENCH_NUM_PROMPTS} request_rate=${BENCH_REQUEST_RATE}"
    wlog "  num_warmups=${BENCH_NUM_WARMUPS} save_detailed=${BENCH_SAVE_DETAILED}"
    wlog "  work_dir=${work_dir} bench_log=${bench_log}"

    local bench_extra_args=()
    if [[ "${BENCH_NUM_WARMUPS}" =~ ^[0-9]+$ && "${BENCH_NUM_WARMUPS}" -gt 0 ]]; then
        bench_extra_args+=(--num-warmups "${BENCH_NUM_WARMUPS}")
    fi
    if [[ "${BENCH_SAVE_DETAILED}" == "1" ]]; then
        bench_extra_args+=(--save-result --save-detailed --result-dir "${work_dir}")
    fi

    set +e
    vllm bench serve \
        --backend vllm \
        --model "${bench_model}" \
        --host 127.0.0.1 \
        --port "${port}" \
        --endpoint "${BENCH_ENDPOINT}" \
        --dataset-name random \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --num-prompts "${BENCH_NUM_PROMPTS}" \
        --max-concurrency "${max_concurrency}" \
        --request-rate "${BENCH_REQUEST_RATE}" \
        --ignore-eos \
        "${profile_args[@]}" \
        "${bench_extra_args[@]}" \
        >"${bench_log}" 2>&1
    local rc=$?
    set -e

    if [[ ${timed_profile} -eq 1 ]]; then
        if [[ -n "${watcher_pid}" ]] && process_alive "${watcher_pid}"; then
            kill "${watcher_pid}" >/dev/null 2>&1 || true
            wait "${watcher_pid}" 2>/dev/null || true
        fi
        # Ensure stop even if duration window not reached or DELAY-only mode.
        post_profile "${port}" "stop_profile" || true
        rm -f "${watcher_pid_file}"
    fi

    if [[ ${rc} -ne 0 ]]; then
        wlog "ERROR: bench 失败 rc=${rc}，见 ${bench_log}"
        tail -n 40 "${bench_log}" || true
        return "${rc}"
    fi
    # Keep a copy under WORK_ROOT for easy aggregation.
    cp -f "${bench_log}" "${work_dir}/bench.log" 2>/dev/null || true
    wlog "bench 完成: ${tag}"
    return 0
}

apply_profile_bench_defaults() {
    # Aligned profile: bench --profile + warmups; skip fixed wall-clock delay.
    if [[ "${ENABLE_PROFILE}" != "1" || -n "${PROFILE_DELAY_SEC}" ]]; then
        return 0
    fi
    if [[ -z "${BENCH_NUM_WARMUPS}" ]]; then
        BENCH_NUM_WARMUPS=16
    fi
    if [[ -z "${BENCH_SAVE_DETAILED}" ]]; then
        BENCH_SAVE_DETAILED=1
    fi
}

normalize_ab_profile_mode() {
    # mixed | decode | prefill （decode_only / prefill_only 为别名）
    local mode="${1,,}"
    case "${mode}" in
        mixed|default|"")
            echo "mixed"
            ;;
        decode|decode_only|decode-only)
            echo "decode"
            ;;
        prefill|prefill_only|prefill-only)
            echo "prefill"
            ;;
        *)
            echo ""
            ;;
    esac
}

apply_ab_profile_mode() {
    # Tune bench + profiler windows so A/B can attribute host vs device by phase.
    # Does not override knobs the user already set (non-empty PROFILE_* / AB_ONE_WAVE).
    local mode first_mc
    mode="$(normalize_ab_profile_mode "${AB_PROFILE_MODE}")"
    if [[ -z "${mode}" ]]; then
        log "ERROR: AB_PROFILE_MODE 无效: ${AB_PROFILE_MODE}（须为 mixed|decode|prefill）"
        exit 1
    fi
    AB_PROFILE_MODE="${mode}"

    # shellcheck disable=SC2206
    first_mc="$(echo "${MAX_CONCURRENCIES}" | awk '{print $1}')"

    if [[ "${_WORK_ROOT_FROM_ENV}" != "1" ]]; then
        WORK_ROOT="${PWD}/output/bench_ab_${AB_PROFILE_MODE}_${STAMP}"
        LOG_ROOT="${WORK_ROOT}/logs"
        mkdir -p "${LOG_ROOT}" "${WORK_ROOT}"
        SWEEP_LOG="${LOG_ROOT}/sweep.log"
    fi

    case "${AB_PROFILE_MODE}" in
        mixed)
            # Keep caller bench shape; profiler active left to framework/user.
            if [[ -z "${AB_ONE_WAVE}" ]]; then
                AB_ONE_WAVE=0
            fi
            ;;
        decode)
            # One wave, skip chunked-prefill steps, capture long decode window.
            if [[ -z "${AB_ONE_WAVE}" ]]; then
                AB_ONE_WAVE=1
            fi
            if [[ -z "${PROFILE_DELAY_ITERATIONS}" ]]; then
                # ~16 chunked-prefill/mixed steps for input_len=2000 / budget=2048
                PROFILE_DELAY_ITERATIONS=20
            fi
            if [[ -z "${PROFILE_ACTIVE_ITERATIONS}" ]]; then
                PROFILE_ACTIVE_ITERATIONS=120
            fi
            ;;
        prefill)
            # One wave, profile from step0, only first prefill/mixed window.
            if [[ -z "${AB_ONE_WAVE}" ]]; then
                AB_ONE_WAVE=1
            fi
            if [[ -z "${PROFILE_DELAY_ITERATIONS}" ]]; then
                PROFILE_DELAY_ITERATIONS=0
            fi
            if [[ -z "${PROFILE_ACTIVE_ITERATIONS}" ]]; then
                PROFILE_ACTIVE_ITERATIONS=20
            fi
            ;;
    esac

    if [[ "${AB_ONE_WAVE}" == "1" ]]; then
        BENCH_NUM_PROMPTS="${first_mc}"
        log "AB_PROFILE_MODE=${AB_PROFILE_MODE}: AB_ONE_WAVE=1 → BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS}"
    fi

    log "AB_PROFILE_MODE=${AB_PROFILE_MODE} delay_iterations=${PROFILE_DELAY_ITERATIONS:-0} active_iterations=${PROFILE_ACTIVE_ITERATIONS:-<default>} one_wave=${AB_ONE_WAVE}"
    case "${AB_PROFILE_MODE}" in
        decode)
            log "  目标: 纯 decode 长窗 → 归因 TPOT（看 Stage/Computing/Free/Preparing + attn/pack Device）"
            ;;
        prefill)
            log "  目标: 首波 prefill/mixed → 归因 TTFT（同上，按 Q≥512 step 过滤）"
            ;;
        mixed)
            log "  目标: 旧行为混合波次（均值易混 phase；仅作回归对照）"
            ;;
    esac
}

find_latest_rank0_prof() {
    local parent="$1"
    local hit
    hit="$(find "${parent}" -maxdepth 3 -type d -name 'rank0_*_ascend_pt' 2>/dev/null | sort | tail -n 1)"
    if [[ -z "${hit}" ]]; then
        hit="$(find "${parent}" -maxdepth 5 -type d -name 'rank0_*_ascend_pt' 2>/dev/null | sort | tail -n 1)"
    fi
    echo "${hit}"
}

run_ab_post_analysis() {
    local ab_root="$1"
    local analysis_dir="${ab_root}/analysis"
    mkdir -p "${analysis_dir}"

    local notq_prof tq_prof
    notq_prof="$(find_latest_rank0_prof "${ab_root}/notq_prof")"
    tq_prof="$(find_latest_rank0_prof "${ab_root}/tq_prof")"
    if [[ -z "${notq_prof}" || -z "${tq_prof}" ]]; then
        log "WARN: AB analysis skip — profiler dump missing (notq=${notq_prof:-none} tq=${tq_prof:-none})"
        return 1
    fi

    log "AB analysis: notq_prof=${notq_prof}"
    log "AB analysis: tq_prof=${tq_prof}"

    local summarize_py="${REPO_ROOT}/tests/e2e/singlecard/summarize_k8v4_profiler.py"
    local decompose_py="${REPO_ROOT}/tools/bench_ab_tpot_decompose.py"

    python3 "${summarize_py}" --compare --labels=notq,tq \
        "${notq_prof}" "${tq_prof}" \
        >"${analysis_dir}/summarize_compare.txt" 2>&1 || true

    python3 "${summarize_py}" --json --labels=notq,tq \
        "${notq_prof}" "${tq_prof}" \
        >"${analysis_dir}/summarize_compare.json" 2>&1 || true

    python3 "${decompose_py}" --ab-root "${ab_root}" \
        -o "${analysis_dir}/tpot_decompose.txt" 2>&1 | tee "${analysis_dir}/tpot_decompose_run.log"

    log "AB analysis written: ${analysis_dir}/"
    log "  summarize_compare.txt  tpot_decompose.txt  tpot_decompose.json"
}

run_ab_compare_suite() {
    local ab_root="${WORK_ROOT}"
    local ab_fail=0
    local variant enable_tq_val saved_enable_tq

    saved_enable_tq="${ENABLE_TQ}"
    mkdir -p "${ab_root}"
    printf '%s\n' "${AB_PROFILE_MODE}" >"${ab_root}/ab_profile_mode.txt"
    cat >"${ab_root}/ab_profile_readme.txt" <<EOF
AB_PROFILE_MODE=${AB_PROFILE_MODE}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS}
BENCH_INPUT_LEN=${BENCH_INPUT_LEN}
BENCH_OUTPUT_LEN=${BENCH_OUTPUT_LEN}
PROFILE_DELAY_ITERATIONS=${PROFILE_DELAY_ITERATIONS:-0}
PROFILE_ACTIVE_ITERATIONS=${PROFILE_ACTIVE_ITERATIONS:-<default>}
AB_ONE_WAVE=${AB_ONE_WAVE}

How to read (host vs device):
  step_trace: Stage / Computing(=device) / Free / Preparing(~host)
  operator_details: Device Total Duration vs Host Total Duration for attn/pack
  decode mode: filter Q=16 (or decode steps after delay window)
  prefill mode: filter Q>=512 / Q in {1448,2000,2048}
EOF

    log "AB_COMPARE=1 mode=${AB_PROFILE_MODE} root=${ab_root} (NOTQ then TQ, bench --profile + warmups)"

    for variant in notq tq; do
        if [[ "${variant}" == "notq" ]]; then
            ENABLE_TQ=0
        else
            ENABLE_TQ=1
        fi
        WORK_ROOT="${ab_root}/${variant}"
        LOG_ROOT="${WORK_ROOT}/logs"
        PROFILE_DIR="${ab_root}/${variant}_prof"
        mkdir -p "${LOG_ROOT}" "${WORK_ROOT}" "${PROFILE_DIR}"
        SWEEP_LOG="${LOG_ROOT}/sweep.log"
        log "-------- AB variant=${variant} mode=${AB_PROFILE_MODE} ENABLE_TQ=${ENABLE_TQ} WORK_ROOT=${WORK_ROOT} --------"
        for ((i = 0; i < n_mc; i++)); do
            if ! run_worker "${MC_ARR[i]}" "${DEV_ARR[i]}" "${PORT_ARR[i]}"; then
                ab_fail=$((ab_fail + 1))
            fi
        done
    done

    ENABLE_TQ="${saved_enable_tq}"
    WORK_ROOT="${ab_root}"
    LOG_ROOT="${ab_root}/logs"
    mkdir -p "${LOG_ROOT}"

    if [[ "${AB_SKIP_ANALYSIS}" != "1" ]]; then
        run_ab_post_analysis "${ab_root}" || ab_fail=$((ab_fail + 1))
    fi

    log "AB_COMPARE done mode=${AB_PROFILE_MODE} fail=${ab_fail} root=${ab_root}"
    return "${ab_fail}"
}

# One max_concurrency on one device/port: start serve once, run ROUNDS benches.
run_worker() {
    local max_concurrency="$1"
    local device="$2"
    local port="$3"
    local worker_dir="${LOG_ROOT}/mc${max_concurrency}_dev${device}_port${port}"
    local pid_file="${worker_dir}/vllm_serve.pid"
    local fail=0
    local profiler_dir=""

    mkdir -p "${worker_dir}"
    WORKER_TAG="mc${max_concurrency}/dev${device}/port${port}"
    WORKER_LOG="${worker_dir}/worker.log"

    if [[ "${ENABLE_PROFILE}" == "1" ]]; then
        if [[ -n "${PROFILE_DIR}" ]]; then
            profiler_dir="${PROFILE_DIR}/mc${max_concurrency}_dev${device}_port${port}"
        else
            profiler_dir="${worker_dir}/profiler"
        fi
        mkdir -p "${profiler_dir}"
        profiler_dir="$(cd "${profiler_dir}" && pwd)"
        wlog "ENABLE_PROFILE=1 profiler_dir=${profiler_dir}"
    fi

    wlog "======== worker start max_concurrency=${max_concurrency} device=${device} port=${port} ========"
    if ! start_serve "mc${max_concurrency}" "${device}" "${port}" "${worker_dir}" "${profiler_dir}"; then
        wlog "ERROR: worker serve 启动失败"
        echo 1 > "${worker_dir}/fail_count"
        return 1
    fi

    for ((r = 1; r <= ROUNDS; r++)); do
        wlog "-------- client mc${max_concurrency} r${r} --------"
        if ! run_bench "${max_concurrency}" "${r}" "${port}" "${worker_dir}"; then
            fail=$((fail + 1))
            wlog "本轮失败，继续下一轮（服务保持运行）"
        fi
    done

    stop_serve_on "${pid_file}" "${port}"
    echo "${fail}" > "${worker_dir}/fail_count"
    if [[ -n "${profiler_dir}" ]]; then
        wlog "profiler traces: ${profiler_dir}"
    fi
    wlog "======== worker done max_concurrency=${max_concurrency} fail=${fail} ========"
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
MC_ARR=(${MAX_CONCURRENCIES})
# shellcheck disable=SC2206
DEV_ARR=(${VISIBLE_DEVICES})
# shellcheck disable=SC2206
PORT_ARR=(${PORTS})

n_mc=${#MC_ARR[@]}
n_dev=${#DEV_ARR[@]}
n_port=${#PORT_ARR[@]}

if [[ ${n_mc} -gt 1 && ${n_dev} -eq 1 ]]; then
    log "ERROR: 多 MAX_CONCURRENCIES (${n_mc}) 需要等长 VISIBLE_DEVICES（当前 ${n_dev}），以便卡一一对应"
    exit 1
fi
if [[ ${n_mc} -gt 1 && ${n_port} -eq 1 ]]; then
    log "ERROR: 多 MAX_CONCURRENCIES (${n_mc}) 需要等长 PORTS（当前 ${n_port}），以便端口一一对应"
    exit 1
fi
if [[ ${n_mc} -ne ${n_dev} || ${n_mc} -ne ${n_port} ]]; then
    log "ERROR: MAX_CONCURRENCIES/VISIBLE_DEVICES/PORTS 长度须一致: ${n_mc}/${n_dev}/${n_port}"
    log "  MAX_CONCURRENCIES=${MAX_CONCURRENCIES}"
    log "  VISIBLE_DEVICES=${VISIBLE_DEVICES}"
    log "  PORTS=${PORTS}"
    exit 1
fi

if [[ "${PARALLEL}" == "1" && ${n_mc} -gt 1 ]]; then
    for ((i = 0; i < n_mc; i++)); do
        for ((j = i + 1; j < n_mc; j++)); do
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

if ! [[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]]; then
    log "ERROR: ROUNDS 须为正整数，当前=${ROUNDS}"
    exit 1
fi
expand_io_lens_for_rounds

if [[ "${ENABLE_PROFILE}" == "1" ]]; then
    if [[ -n "${PROFILE_DELAY_SEC}" ]] && ! [[ "${PROFILE_DELAY_SEC}" =~ ^[0-9]+$ ]]; then
        log "ERROR: PROFILE_DELAY_SEC 须为非负整数，当前=${PROFILE_DELAY_SEC}"
        exit 1
    fi
    if [[ -n "${PROFILE_DURATION_SEC}" ]]; then
        if [[ -z "${PROFILE_DELAY_SEC}" ]]; then
            log "ERROR: PROFILE_DURATION_SEC 需配合 PROFILE_DELAY_SEC 使用（否则请用 bench --profile 模式）"
            exit 1
        fi
        if ! [[ "${PROFILE_DURATION_SEC}" =~ ^[1-9][0-9]*$ ]]; then
            log "ERROR: PROFILE_DURATION_SEC 须为正整数，当前=${PROFILE_DURATION_SEC}"
            exit 1
        fi
    fi
fi

if [[ "${ENABLE_TQ}" != "0" && "${ENABLE_TQ}" != "1" ]]; then
    log "ERROR: ENABLE_TQ 须为 0 或 1，当前=${ENABLE_TQ}"
    exit 1
fi

if [[ "${AB_COMPARE}" == "1" ]]; then
    if [[ "${AB_NO_PROFILE}" == "1" ]]; then
        ENABLE_PROFILE=0
        AB_SKIP_ANALYSIS=1
        # Still apply one-wave / mode defaults (prompts, etc.) without profiler knobs.
        apply_ab_profile_mode
        if [[ -z "${BENCH_NUM_WARMUPS}" ]]; then
            BENCH_NUM_WARMUPS=16
        fi
        if [[ -z "${BENCH_SAVE_DETAILED}" ]]; then
            BENCH_SAVE_DETAILED=1
        fi
        log "AB_NO_PROFILE=1: e2e-only AB (no torch/Ascend profiler)"
    else
        ENABLE_PROFILE=1
        PROFILE_DELAY_SEC=""
        PROFILE_DURATION_SEC=""
        apply_ab_profile_mode
        apply_profile_bench_defaults
    fi
fi

apply_profile_bench_defaults

if [[ "${AB_COMPARE}" == "1" ]]; then
    log "AB_COMPARE=1 mode=${AB_PROFILE_MODE}; ENABLE_TQ will run 0 then 1; single ENABLE_TQ=${ENABLE_TQ} ignored for suite"
fi

log "WORK_ROOT=${WORK_ROOT}"
log "LOG_ROOT=${LOG_ROOT}"
log "BLOCK_SIZE=${BLOCK_SIZE} FIA=${_FIA}/${_DECODE_FIA}/${_NOCACHE_FIA} ROUNDS=${ROUNDS} PARALLEL=${PARALLEL}"
log "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-<default>} enforce_eager=${ENFORCE_EAGER}"
log "bench: num_prompts=${BENCH_NUM_PROMPTS} request_rate=${BENCH_REQUEST_RATE} num_warmups=${BENCH_NUM_WARMUPS:-0} save_detailed=${BENCH_SAVE_DETAILED:-0}"
if [[ "${ENABLE_PROFILE}" == "1" ]]; then
    if [[ -n "${PROFILE_DELAY_SEC}" ]]; then
        log "profile: ENABLE=1 mode=timed delay=${PROFILE_DELAY_SEC}s duration=${PROFILE_DURATION_SEC:-until-bench-end}"
    else
        log "profile: ENABLE=1 mode=bench --profile (warmup后start / 结束后stop)"
    fi
    log "  ignore_frontend=${PROFILE_IGNORE_FRONTEND} with_stack=${PROFILE_WITH_STACK} active_iterations=${PROFILE_ACTIVE_ITERATIONS:-<default>} delay_iterations=${PROFILE_DELAY_ITERATIONS:-0}"
    log "  PROFILE_DIR=${PROFILE_DIR:-<per-worker ${LOG_ROOT}/.../profiler>}"
else
    log "profile: ENABLE=0"
fi
if [[ "${AB_COMPARE}" == "1" ]]; then
    log "  AB_PROFILE_MODE=${AB_PROFILE_MODE} AB_ONE_WAVE=${AB_ONE_WAVE}"
fi
for ((r = 1; r <= ROUNDS; r++)); do
    log "  round[${r}]: input_len=$(round_input_len "${r}") output_len=$(round_output_len "${r}")"
done
for ((i = 0; i < n_mc; i++)); do
    log "  slot[${i}]: max_concurrency=${MC_ARR[i]} device=${DEV_ARR[i]} port=${PORT_ARR[i]}"
done

if [[ "${AB_COMPARE}" == "1" ]]; then
    run_ab_compare_suite
    exit $?
fi

worker_pids=()
if [[ "${PARALLEL}" == "1" && ${n_mc} -gt 1 ]]; then
    for ((i = 0; i < n_mc; i++)); do
        run_worker "${MC_ARR[i]}" "${DEV_ARR[i]}" "${PORT_ARR[i]}" &
        wpid=$!
        worker_pids+=("${wpid}")
        log "spawned worker pid=${wpid} mc=${MC_ARR[i]} dev=${DEV_ARR[i]} port=${PORT_ARR[i]}"
    done
    for pid in "${worker_pids[@]}"; do
        wait "${pid}" || true
    done
else
    for ((i = 0; i < n_mc; i++)); do
        run_worker "${MC_ARR[i]}" "${DEV_ARR[i]}" "${PORT_ARR[i]}" || true
    done
fi

fail_count=0
for ((i = 0; i < n_mc; i++)); do
    fc_file="${LOG_ROOT}/mc${MC_ARR[i]}_dev${DEV_ARR[i]}_port${PORT_ARR[i]}/fail_count"
    if [[ -f "${fc_file}" ]]; then
        fail_count=$((fail_count + $(cat "${fc_file}")))
    else
        fail_count=$((fail_count + 1))
        log "WARN: missing fail_count for mc${MC_ARR[i]}"
    fi
done

log "全部结束 fail_count=${fail_count} logs=${LOG_ROOT}"
if [[ ${fail_count} -gt 0 ]]; then
    exit 1
fi
exit 0
