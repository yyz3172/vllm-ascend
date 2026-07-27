#!/usr/bin/env bash
# Commit merge gate: run BitResidual K8V4 smoke suite and print PASS/FAIL verdict.
#
# Cases (correctness first, then serving):
#   1. xrx_bit_residual_k8v4_key1_unaligned.py
#   2. xrx_bit_residual_k8v4_batch_tiers.py
#   3. xrx_bit_residual_k8v4_golden.py
#   4. xrx_bit_residual_fia_paged_k8v4_smoke.py
#   5. xrx_bit_residual_k8v4_smoke.py
#   6. xrx_bit_residual_k8v4_long_query_profile.py
#   7. xrx_bit_residual_k8v4_bs1_long_profile.py
#
# Usage (inside container, after activate + infoenvs):
#   bash tools/bit_residual_fia_paged_k8v4/run_merge_smoke.sh
#   bash tools/bit_residual_fia_paged_k8v4/run_merge_smoke.sh --skip-build
#   bash tools/bit_residual_fia_paged_k8v4/run_merge_smoke.sh --continue-on-fail
#   bash tools/bit_residual_fia_paged_k8v4/run_merge_smoke.sh --with-fia-serving
#   bash tools/bit_residual_fia_paged_k8v4/run_merge_smoke.sh --profile
#
# Env:
#   XRX_K8V4_MODEL_PATH   default /root/cyl/model/Qwen3-0.6B
#   XRX_K8V4_PROFILE_DIR  default /root/cyl/perflog2/merge_smoke
#   SKIP_BUILD            0|1 (or --skip-build)
#   CONTINUE_ON_FAIL      0|1 (or --continue-on-fail)
#   WITH_FIA_SERVING      0|1 (or --with-fia-serving) — set FIA gates on serving cases
#   ENABLE_TORCH_PROFILE  0|1 (or --profile) — default off for merge gate speed
set -euo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${TOOLS}/../.." && pwd)"
TEST_DIR="${ROOT_DIR}/tests/e2e/singlecard"

MODEL_PATH="${XRX_K8V4_MODEL_PATH:-/root/cyl/model/Qwen3-0.6B}"
PROFILE_ROOT="${XRX_K8V4_PROFILE_DIR:-/root/cyl/perflog2/merge_smoke}"
SKIP_BUILD="${SKIP_BUILD:-0}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
WITH_FIA_SERVING="${WITH_FIA_SERVING:-0}"
ENABLE_TORCH_PROFILE="${ENABLE_TORCH_PROFILE:-0}"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${PROFILE_ROOT}/_logs/${STAMP}"

for arg in "$@"; do
    case "${arg}" in
        --skip-build) SKIP_BUILD=1 ;;
        --continue-on-fail) CONTINUE_ON_FAIL=1 ;;
        --with-fia-serving) WITH_FIA_SERVING=1 ;;
        --profile) ENABLE_TORCH_PROFILE=1 ;;
        -h|--help)
            sed -n '2,28p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            exit 2
            ;;
    esac
done

mkdir -p "${LOG_DIR}"

source_env() {
    local cann_candidates=(
        "${ASCEND_HOME_PATH:-}"
        /usr/local/Ascend/cann/8.5.1
        /usr/local/Ascend/cann-8.5.1
        /usr/local/Ascend/ascend-toolkit/latest
    )
    local cand
    for cand in "${cann_candidates[@]}"; do
        [[ -z "${cand}" ]] && continue
        if [[ -f "${cand}/set_env.sh" ]]; then
            # shellcheck disable=SC1090
            source "${cand}/set_env.sh"
            break
        elif [[ -f "${cand}/bin/setenv.bash" ]]; then
            # shellcheck disable=SC1090
            source "${cand}/bin/setenv.bash" || true
            break
        fi
    done

    if [[ -f /root/cyl/venv/vllmdev/.venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source /root/cyl/venv/vllmdev/.venv/bin/activate
    elif [[ -f /root/l00856060/venv/lcyvllm/.venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source /root/l00856060/venv/lcyvllm/.venv/bin/activate
    fi
    if [[ -f "${ROOT_DIR}/infoenvs" ]]; then
        # shellcheck disable=SC1090
        source "${ROOT_DIR}/infoenvs"
    fi

    if [[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}/opp" ]]; then
        export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
    fi
}

check_cust_ops() {
    local lib="${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so"
    if [[ ! -f "${lib}" ]]; then
        echo "[ERROR] missing ${lib}; run without --skip-build" >&2
        exit 1
    fi
    if ! nm -D "${lib}" | grep -q 'aclnnBitResidualFiaPagedK8v4'; then
        echo "[ERROR] libcust_opapi.so missing aclnnBitResidualFiaPagedK8v4" >&2
        exit 1
    fi
    if ! nm -D "${lib}" | grep -q 'aclnnBitResidualPackK8v4'; then
        echo "[ERROR] libcust_opapi.so missing aclnnBitResidualPackK8v4" >&2
        exit 1
    fi
    # Serving needs more than BR ops.
    if ! nm -D "${lib}" | grep -q 'aclnnAddRmsNormBias'; then
        echo "[ERROR] libcust_opapi.so missing aclnnAddRmsNormBias;" \
            "slim rebuild stripped serving ops. Run without --skip-build." >&2
        exit 1
    fi
}

declare -a CASE_NAMES=()
declare -a CASE_RCS=()
declare -a CASE_SECS=()
declare -a CASE_LOGS=()

run_case() {
    local name="$1"
    shift
    local log_file="${LOG_DIR}/${name}.log"
    local t0 t1 rc elapsed
    CASE_NAMES+=("${name}")
    CASE_LOGS+=("${log_file}")

    echo
    echo "========== ${name} =========="
    echo "cmd: $*"
    echo "log: ${log_file}"
    t0=$(date +%s)
    set +e
    "$@" 2>&1 | tee "${log_file}"
    rc=${PIPESTATUS[0]}
    set -e
    t1=$(date +%s)
    elapsed=$((t1 - t0))
    CASE_RCS+=("${rc}")
    CASE_SECS+=("${elapsed}")

    if [[ "${rc}" -eq 0 ]]; then
        echo "[PASS] ${name} (${elapsed}s)"
    else
        echo "[FAIL] ${name} rc=${rc} (${elapsed}s) — see ${log_file}" >&2
        if [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
            print_summary
            exit "${rc}"
        fi
    fi
}

print_summary() {
    local i n pass fail
    n=${#CASE_NAMES[@]}
    pass=0
    fail=0
    echo
    echo "========================================"
    echo " BitResidual K8V4 merge smoke summary"
    echo " stamp=${STAMP}"
    echo " model=${MODEL_PATH}"
    echo " logs=${LOG_DIR}"
    echo " skip_build=${SKIP_BUILD} with_fia_serving=${WITH_FIA_SERVING} profile=${ENABLE_TORCH_PROFILE}"
    echo "----------------------------------------"
    printf "%-4s %-42s %6s %8s\n" "#" "case" "rc" "time_s"
    for ((i = 0; i < n; i++)); do
        printf "%-4d %-42s %6s %8s\n" \
            "$((i + 1))" "${CASE_NAMES[$i]}" "${CASE_RCS[$i]}" "${CASE_SECS[$i]}"
        if [[ "${CASE_RCS[$i]}" -eq 0 ]]; then
            pass=$((pass + 1))
        else
            fail=$((fail + 1))
        fi
    done
    echo "----------------------------------------"
    echo "passed=${pass} failed=${fail} total=${n}"
    if [[ "${fail}" -eq 0 && "${n}" -gt 0 ]]; then
        echo
        echo "MERGE GATE: PASS — OK to land the commit (smoke suite green)."
        echo "Verdict file: ${LOG_DIR}/VERDICT_PASS.txt"
        {
            echo "MERGE_GATE=PASS"
            echo "stamp=${STAMP}"
            echo "passed=${pass}"
            echo "failed=0"
        } > "${LOG_DIR}/VERDICT_PASS.txt"
        return 0
    else
        echo
        echo "MERGE GATE: FAIL — do not merge until failures are fixed."
        echo "Verdict file: ${LOG_DIR}/VERDICT_FAIL.txt"
        {
            echo "MERGE_GATE=FAIL"
            echo "stamp=${STAMP}"
            echo "passed=${pass}"
            echo "failed=${fail}"
            for ((i = 0; i < n; i++)); do
                if [[ "${CASE_RCS[$i]}" -ne 0 ]]; then
                    echo "fail_case=${CASE_NAMES[$i]} rc=${CASE_RCS[$i]} log=${CASE_LOGS[$i]}"
                fi
            done
        } > "${LOG_DIR}/VERDICT_FAIL.txt"
        return 1
    fi
}

echo "ROOT=${ROOT_DIR}"
echo "MODEL=${MODEL_PATH}"
echo "LOG_DIR=${LOG_DIR}"
echo "SKIP_BUILD=${SKIP_BUILD}"
echo "WITH_FIA_SERVING=${WITH_FIA_SERVING}"
echo "ENABLE_TORCH_PROFILE=${ENABLE_TORCH_PROFILE}"

cd "${ROOT_DIR}"
source_env

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] model not found: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
    echo
    echo "========== build_so (full custom ops) =========="
    bash "${ROOT_DIR}/build_so.sh" 2>&1 | tee "${LOG_DIR}/build_so.log"
else
    echo "[skip] build_so.sh (--skip-build)"
fi
check_cust_ops

export XRX_K8V4_MODEL_PATH="${MODEL_PATH}"
export MODEL_PATH="${MODEL_PATH}"
export XRX_K8V4_PROFILE_DIR="${PROFILE_ROOT}/${STAMP}"
export PROFILE_DIR="${XRX_K8V4_PROFILE_DIR}"
mkdir -p "${PROFILE_DIR}"

# Merge gate: functional correctness, not torch profiler dumps.
if [[ "${ENABLE_TORCH_PROFILE}" == "1" ]]; then
    export XRX_K8V4_PROFILE=1
    export XRX_K8V4_ENABLE_PROFILE=1
else
    export XRX_K8V4_PROFILE=0
    export XRX_K8V4_ENABLE_PROFILE=0
fi

# Scheme A is the commit default; keep golden/tests aligned.
export BR_KEY_UNIFORM_SCHEME="${BR_KEY_UNIFORM_SCHEME:-1}"

# --- Op / pack correctness (no LLM) ---
run_case "key1_unaligned" \
    python "${TEST_DIR}/xrx_bit_residual_k8v4_key1_unaligned.py"

run_case "batch_tiers" \
    python "${TEST_DIR}/xrx_bit_residual_k8v4_batch_tiers.py"

run_case "golden" \
    python "${TEST_DIR}/xrx_bit_residual_k8v4_golden.py"

run_case "fia_paged_smoke" \
    python "${TEST_DIR}/xrx_bit_residual_fia_paged_k8v4_smoke.py"

# --- Serving smoke ---
# Clear FIA gates unless explicitly requested.
unset VLLM_ASCEND_BIT_RESIDUAL_FIA VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA || true
if [[ "${WITH_FIA_SERVING}" == "1" ]]; then
    export VLLM_ASCEND_BIT_RESIDUAL_FIA=1
    export VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA=1
fi

run_case "serving_smoke" \
    python "${TEST_DIR}/xrx_bit_residual_k8v4_smoke.py"

# Long-query: keep defaults modest for gate; override via env if needed.
export XRX_K8V4_LONG_NUM_PROMPTS="${XRX_K8V4_LONG_NUM_PROMPTS:-16}"
export XRX_K8V4_LONG_PROMPT_TOKENS="${XRX_K8V4_LONG_PROMPT_TOKENS:-2000}"
export XRX_K8V4_LONG_OUTPUT_TOKENS="${XRX_K8V4_LONG_OUTPUT_TOKENS:-8}"

run_case "long_query" \
    python "${TEST_DIR}/xrx_bit_residual_k8v4_long_query_profile.py"

# bs1 long: same runner with different defaults; force profile dir under gate root.
export XRX_K8V4_LONG_NUM_PROMPTS=1
export XRX_K8V4_LONG_PROMPT_TOKENS="${XRX_K8V4_LONG_PROMPT_TOKENS:-2000}"
export XRX_K8V4_LONG_OUTPUT_TOKENS="${XRX_K8V4_LONG_OUTPUT_TOKENS:-16}"
export XRX_K8V4_PROFILE_DIR="${PROFILE_ROOT}/${STAMP}/bs1_long"
export PROFILE_DIR="${XRX_K8V4_PROFILE_DIR}"
mkdir -p "${PROFILE_DIR}"

run_case "bs1_long" \
    python "${TEST_DIR}/xrx_bit_residual_k8v4_bs1_long_profile.py"

print_summary
exit $?
