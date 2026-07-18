#!/usr/bin/env bash
# One-shot: full custom-op build + 2 serving torch profiles (vector attn vs FIA) + summarize.
#
# Profile dumps are keyed by operator name for easy browsing:
#   ${XRX_K8V4_PROFILE_DIR}/BitResidualAttentionPagedK8v4/<stamp>/rank0_*_ascend_pt
#   ${XRX_K8V4_PROFILE_DIR}/BitResidualFiaPagedK8v4/<stamp>/rank0_*_ascend_pt
#   + .../<op>/latest -> latest stamp
#
# Usage (inside container vllm-ascend-lcy-v018):
#   bash tools/bit_residual_fia_paged_k8v4/run_serving_profile_ab.sh
#   bash tools/bit_residual_fia_paged_k8v4/run_serving_profile_ab.sh --skip-build
#   ENABLE_DECODE_FIA=0 bash tools/bit_residual_fia_paged_k8v4/run_serving_profile_ab.sh
#
# Env overrides:
#   XRX_K8V4_MODEL_PATH   default /root/l00856060/model/Qwen3-0.6B
#   XRX_K8V4_PROFILE_DIR  default /root/l00856060/perflog2
#   ENABLE_DECODE_FIA     default 1 (FIA leg also sets DECODE_FIA)
#   SKIP_BUILD            default 0 (or pass --skip-build)
set -euo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${TOOLS}/../.." && pwd)"
PROFILE_PY="${ROOT_DIR}/tests/e2e/singlecard/xrx_bit_residual_k8v4_long_query_profile.py"
SUMMARIZE_PY="${ROOT_DIR}/tests/e2e/singlecard/summarize_k8v4_profiler.py"

OP_ATTN="BitResidualAttentionPagedK8v4"
OP_FIA="BitResidualFiaPagedK8v4"

MODEL_PATH="${XRX_K8V4_MODEL_PATH:-/root/l00856060/model/Qwen3-0.6B}"
PROFILE_ROOT="${XRX_K8V4_PROFILE_DIR:-/root/l00856060/perflog2}"
ENABLE_DECODE_FIA="${ENABLE_DECODE_FIA:-1}"
SKIP_BUILD="${SKIP_BUILD:-0}"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

for arg in "$@"; do
    case "${arg}" in
        --skip-build) SKIP_BUILD=1 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            exit 2
            ;;
    esac
done

ATTN_DIR="${PROFILE_ROOT}/${OP_ATTN}/${STAMP}"
FIA_DIR="${PROFILE_ROOT}/${OP_FIA}/${STAMP}"
LOG_DIR="${PROFILE_ROOT}/_logs/${STAMP}"
mkdir -p "${ATTN_DIR}" "${FIA_DIR}" "${LOG_DIR}"

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

    if [[ -f /root/l00856060/venv/lcyvllm/.venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source /root/l00856060/venv/lcyvllm/.venv/bin/activate
    fi
    if [[ -f "${ROOT_DIR}/infoenvs" ]]; then
        # shellcheck disable=SC1090
        source "${ROOT_DIR}/infoenvs"
    fi

    # Serving must use CANN builtin opp; custom kernels via ASCEND_CUSTOM_OPP_PATH
    # (vllm_ascend.platform sets the latter). Do not point ASCEND_OPP_PATH at custom.
    if [[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}/opp" ]]; then
        export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
    fi
}

link_latest() {
    local op_root="$1"
    local stamp_dir="$2"
    ln -sfn "$(basename "${stamp_dir}")" "${op_root}/latest"
}

run_one_profile() {
    local label="$1"
    local out_dir="$2"
    local log_file="$3"
    shift 3

    echo
    echo "========== profile: ${label} =========="
    echo "profile_dir=${out_dir}"
    echo "log=${log_file}"
    mkdir -p "${out_dir}"

    # Clear BR FIA gates then apply caller overrides (passed as env assignments).
    unset VLLM_ASCEND_BIT_RESIDUAL_FIA VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA || true
    export XRX_K8V4_MODEL_PATH="${MODEL_PATH}"
    export XRX_K8V4_PROFILE_DIR="${out_dir}"
    export XRX_K8V4_ENABLE_PROFILE=1
    # Apply extra env from remaining args: KEY=VAL ...
    local kv
    for kv in "$@"; do
        # shellcheck disable=SC2163
        export "${kv}"
    done

    set +e
    python "${PROFILE_PY}" 2>&1 | tee "${log_file}"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ ${rc} -ne 0 ]]; then
        echo "[ERROR] profile ${label} failed rc=${rc}; see ${log_file}" >&2
        exit "${rc}"
    fi
}

echo "ROOT=${ROOT_DIR}"
echo "MODEL=${MODEL_PATH}"
echo "PROFILE_ROOT=${PROFILE_ROOT}"
echo "STAMP=${STAMP}"
echo "ATTN_DIR=${ATTN_DIR}"
echo "FIA_DIR=${FIA_DIR}"
echo "SKIP_BUILD=${SKIP_BUILD}"
echo "ENABLE_DECODE_FIA=${ENABLE_DECODE_FIA}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] model not found: ${MODEL_PATH}" >&2
    exit 1
fi

cd "${ROOT_DIR}"
source_env

if [[ "${SKIP_BUILD}" != "1" ]]; then
    echo
    echo "========== build (full custom ops via build_so.sh) =========="
    # Full package: keeps AddRmsNormBias etc. required by serving.
    bash "${ROOT_DIR}/build_so.sh" 2>&1 | tee "${LOG_DIR}/build_so.log"
else
    echo "[skip] build_so.sh (--skip-build)"
fi

# Sanity: serving needs more than the 3 BR ops.
if ! nm -D "${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so" \
    | grep -q 'aclnnAddRmsNormBias'; then
    echo "[ERROR] libcust_opapi.so missing aclnnAddRmsNormBias;" \
        "do not use slim rebuild_op.sh for serving. Run without --skip-build." >&2
    exit 1
fi
if ! nm -D "${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so" \
    | grep -q 'aclnnBitResidualFiaPagedK8v4'; then
    echo "[ERROR] libcust_opapi.so missing aclnnBitResidualFiaPagedK8v4" >&2
    exit 1
fi

# --- Profile 1: vector BitResidualAttentionPagedK8v4 ---
run_one_profile \
    "${OP_ATTN}" \
    "${ATTN_DIR}" \
    "${LOG_DIR}/profile_attn.log"
link_latest "${PROFILE_ROOT}/${OP_ATTN}" "${ATTN_DIR}"

# --- Profile 2: BitResidualFiaPagedK8v4 ---
fia_env=( "VLLM_ASCEND_BIT_RESIDUAL_FIA=1" )
if [[ "${ENABLE_DECODE_FIA}" == "1" ]]; then
    fia_env+=( "VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA=1" )
fi
run_one_profile \
    "${OP_FIA}" \
    "${FIA_DIR}" \
    "${LOG_DIR}/profile_fia.log" \
    "${fia_env[@]}"
link_latest "${PROFILE_ROOT}/${OP_FIA}" "${FIA_DIR}"

# Write a small manifest for this stamp.
MANIFEST="${PROFILE_ROOT}/_runs/${STAMP}.txt"
mkdir -p "${PROFILE_ROOT}/_runs"
cat > "${MANIFEST}" <<EOF
stamp=${STAMP}
model=${MODEL_PATH}
attn_dir=${ATTN_DIR}
fia_dir=${FIA_DIR}
attn_latest=${PROFILE_ROOT}/${OP_ATTN}/latest
fia_latest=${PROFILE_ROOT}/${OP_FIA}/latest
ENABLE_DECODE_FIA=${ENABLE_DECODE_FIA}
EOF
echo
echo "manifest: ${MANIFEST}"

echo
echo "========== summarize + compare =========="
python "${SUMMARIZE_PY}" --compare --labels attn,fia --latest \
    "${ATTN_DIR}" \
    "${FIA_DIR}" \
    | tee "${LOG_DIR}/summarize_compare.log"

echo
echo "Done."
echo "  attn dump : ${ATTN_DIR}  (symlink ${PROFILE_ROOT}/${OP_ATTN}/latest)"
echo "  fia  dump : ${FIA_DIR}  (symlink ${PROFILE_ROOT}/${OP_FIA}/latest)"
echo "  compare   : ${LOG_DIR}/summarize_compare.log"
