#!/usr/bin/env bash
# Compile tools/turboquant_fia_mse8bit/test_aclnn_tq_fia_mse8bit.cpp
#
# Usage:
#   bash tools/turboquant_fia_mse8bit/build_test.sh
#   bash tools/turboquant_fia_mse8bit/build_test.sh --rebuild-op
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOOLS_DIR="${ROOT_DIR}/tools/turboquant_fia_mse8bit"
OUT="${TOOLS_DIR}/output/test_aclnn_tq_fia_mse8bit"
SRC="${TOOLS_DIR}/test_aclnn_tq_fia_mse8bit.cpp"

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}
if [[ ! -d "${ASCEND_HOME_PATH}" && -d /usr/local/Ascend/cann ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann
fi

CUSTOM_OPP_PATH="${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
CUSTOM_OP_LIB="${CUSTOM_OPP_PATH}/op_api/lib"
CUSTOM_OP_INC="${CUSTOM_OPP_PATH}/op_api/include"

if [[ "${1:-}" == "--rebuild-op" ]]; then
    bash "${ROOT_DIR}/script/lcy/tq_fia_mse8bit/rebuild_op.sh"
fi

if [[ ! -f "${CUSTOM_OP_INC}/aclnn_turboquant_fia_mse8bit.h" ]]; then
    echo "[build_test] ERROR: missing ${CUSTOM_OP_INC}/aclnn_turboquant_fia_mse8bit.h"
    echo "[build_test] Run: bash ${ROOT_DIR}/script/lcy/tq_fia_mse8bit/rebuild_op.sh"
    exit 1
fi

ARCH_SUBDIR=""
case "$(uname -m)" in
    aarch64) ARCH_SUBDIR="aarch64-linux" ;;
    x86_64) ARCH_SUBDIR="x86_64-linux" ;;
esac

mkdir -p "$(dirname "${OUT}")"
echo "[build_test] Compiling ${SRC}"
echo "[build_test] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "[build_test] CUSTOM_OP_INC=${CUSTOM_OP_INC}"

g++ -std=c++17 -O2 -g \
    -I"${ASCEND_HOME_PATH}/include" \
    ${ARCH_SUBDIR:+-I"${ASCEND_HOME_PATH}/${ARCH_SUBDIR}/include"} \
    -I"${CUSTOM_OP_INC}" \
    -DVLLM_ASCEND_CUSTOM_OPP_PATH=\"${CUSTOM_OPP_PATH}\" \
    "${SRC}" \
    -L"${ASCEND_HOME_PATH}/lib64" \
    -L"${CUSTOM_OP_LIB}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64:${CUSTOM_OP_LIB}" \
    -lcust_opapi -lopapi -lnnopbase -lascendcl -ldl \
    -o "${OUT}"

echo "[build_test] Built ${OUT}"
echo "[build_test] Run:"
echo "  ASCEND_CUSTOM_OPP_PATH=${CUSTOM_OPP_PATH} ${OUT}"
