#!/usr/bin/env bash
# Build / install BitResidualFiaPagedK8v4 custom op into vllm_ascend/_cann_ops_custom.
#
# Usage:
#   bash script/lcy/bit_residual_fia_paged_k8v4/rebuild_op.sh
#   bash script/lcy/bit_residual_fia_paged_k8v4/rebuild_op.sh --soc=ascend910b
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SOC_ARG="${SOC_VERSION:-ascend910b}"
OP_DEBUG_CONFIG="${OP_DEBUG_CONFIG:-}"
for arg in "$@"; do
    case "${arg}" in
        --soc=*) SOC_ARG="${arg#*=}" ;;
        --op-debug-config=*) OP_DEBUG_CONFIG="${arg#*=}" ;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
    esac
done

case "${SOC_ARG}" in
    ascend910b*|910b*) SOC_ARG="ascend910b" ;;
    ascend910_93*) SOC_ARG="ascend910_93" ;;
esac

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}
if [[ ! -d "${ASCEND_HOME_PATH}" && -d /usr/local/Ascend/cann ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann
fi
if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_HOME_PATH}/bin/setenv.bash" || true
elif [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_HOME_PATH}/set_env.sh" || true
fi

# Include add_rms_norm_bias so slim rebuilds do not overwrite libcust_opapi.so
# without aclnnAddRmsNormBias (breaks LLM serving / layernorm).
CUSTOM_OPS="bit_residual_fia_paged_k8v4;bit_residual_attention_paged_k8v4;bit_residual_pack_k8v4;add_rms_norm_bias"
INSTALL_PATH="${ROOT_DIR}/vllm_ascend/_cann_ops_custom"
# 1=A asymmetric uniform (default), 2=B symmetric, 3=C legacy LSB-sign+q7.
# Pack + FIA + attn kernels all read this via CMake; golden uses same env.
export BR_KEY_UNIFORM_SCHEME="${BR_KEY_UNIFORM_SCHEME:-1}"

echo "[rebuild] ROOT=${ROOT_DIR}"
echo "[rebuild] SOC=${SOC_ARG}"
echo "[rebuild] OPS=${CUSTOM_OPS}"
echo "[rebuild] BR_KEY_UNIFORM_SCHEME=${BR_KEY_UNIFORM_SCHEME}"
echo "[rebuild] INSTALL=${INSTALL_PATH}"

cd "${ROOT_DIR}/csrc"
rm -rf build output
BUILD_ARGS=(-n "${CUSTOM_OPS}" -c "${SOC_ARG}")
if [[ -n "${OP_DEBUG_CONFIG}" ]]; then
    BUILD_ARGS+=(--op-debug-config "${OP_DEBUG_CONFIG}")
fi
bash build.sh "${BUILD_ARGS[@]}"

RUN_PKG=$(ls -1 output/CANN-custom_ops*.run 2>/dev/null | head -1)
if [[ -z "${RUN_PKG}" ]]; then
    echo "[rebuild] ERROR: no CANN-custom_ops*.run under csrc/output"
    exit 1
fi
mkdir -p "${INSTALL_PATH}"
bash "${RUN_PKG}" --install-path="${INSTALL_PATH}"
echo "[rebuild] Installed. Header check:"
ls -la "${INSTALL_PATH}/vendors/vllm-ascend/op_api/include/aclnn_bit_residual_fia_paged_k8v4.h"
echo "[rebuild] Done."
