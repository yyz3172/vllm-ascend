#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT="${ROOT_DIR}/test_tq4bit"

# ---- Detect Ascend CANN path ----
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -d "${ASCEND_HOME_PATH}" && -d /usr/local/Ascend/cann-8.5.1 ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
fi

CUSTOM_OPP_PATH="${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
CUSTOM_OP_LIB="${CUSTOM_OPP_PATH}/op_api/lib"
CUSTOM_OP_INC="${CUSTOM_OPP_PATH}/op_api/include"

# ---- Handle --rebuild-op: rebuild custom op kernels before compiling ----
REBUILD_OP=false
if [[ "${1:-}" == "--rebuild-op" ]]; then
    REBUILD_OP=true
    shift || true
fi

if $REBUILD_OP; then
    echo "[build_test_tq4bit] Rebuilding custom op kernels..."
    # Determine SOC version
    SOC_VERSION=${SOC_VERSION:-ascend910b3}
    echo "[build_test_tq4bit]   SOC_VERSION = ${SOC_VERSION}"

    # Source CANN environment (needed for bisheng compiler, cmake, etc.)
    source "${ASCEND_HOME_PATH}/bin/setenv.bash" || true

    # Rebuild only the turboquant 4-bit ops (and their dependencies)
    CUSTOM_OPS="turboquant_pack_kv_for_cache4bit;turboquant_attention_paged4bit"

    # Map SOC version to build.sh -c argument
    case "${SOC_VERSION}" in
        ascend910b*)  SOC_ARG="ascend910b" ;;
        ascend910_93) SOC_ARG="ascend910_93" ;;
        ascend310p*)  SOC_ARG="ascend310p" ;;
        *)            SOC_ARG="${SOC_VERSION}" ;;
    esac

    echo "[build_test_tq4bit]   Building ops: ${CUSTOM_OPS} for ${SOC_ARG}"
    cd "${ROOT_DIR}/csrc"
    rm -rf build output
    bash build.sh -n "${CUSTOM_OPS}" -c "${SOC_ARG}"

    # Install rebuilt ops into the source tree
    echo "[build_test_tq4bit]   Installing custom ops..."
    ./output/CANN-custom_ops*.run --install-path="${ROOT_DIR}/vllm_ascend/_cann_ops_custom"
    cd "${ROOT_DIR}"
    echo "[build_test_tq4bit]   Custom ops rebuilt and installed."
fi

# ---- Check that custom op headers exist ----
if [[ ! -f "${CUSTOM_OP_INC}/aclnn_turboquant_pack_kv_for_cache4bit.h" ]]; then
    echo "[build_test_tq4bit] ERROR: Custom op header not found at"
    echo "[build_test_tq4bit]   ${CUSTOM_OP_INC}/aclnn_turboquant_pack_kv_for_cache4bit.h"
    echo "[build_test_tq4bit] Run with --rebuild-op to build the custom op kernels first:"
    echo "[build_test_tq4bit]   bash ${0} --rebuild-op"
    exit 1
fi

echo "[build_test_tq4bit] Compiling test_tq4bit.cpp ..."
echo "[build_test_tq4bit]   ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}"
echo "[build_test_tq4bit]   CUSTOM_OP_INC    = ${CUSTOM_OP_INC}"
echo "[build_test_tq4bit]   CUSTOM_OP_LIB    = ${CUSTOM_OP_LIB}"
echo "[build_test_tq4bit]   Output           = ${OUT}"

# Determine ARCH include path (aarch64-linux on ARM, x86_64-linux on x86)
ARCH_SUBDIR=""
if [[ "$(uname -m)" == "aarch64" ]]; then
    ARCH_SUBDIR="aarch64-linux"
elif [[ "$(uname -m)" == "x86_64" ]]; then
    ARCH_SUBDIR="x86_64-linux"
fi

g++ -std=c++17 -O2 -g \
    -I"${ASCEND_HOME_PATH}/include" \
    ${ARCH_SUBDIR:+-I"${ASCEND_HOME_PATH}/${ARCH_SUBDIR}/include"} \
    -I"${CUSTOM_OP_INC}" \
    -DVLLM_ASCEND_CUSTOM_OPP_PATH=\"${CUSTOM_OPP_PATH}\" \
    "${ROOT_DIR}/test_tq4bit.cpp" \
    -L"${ASCEND_HOME_PATH}/lib64" \
    -L"${CUSTOM_OP_LIB}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64:${CUSTOM_OP_LIB}" \
    -lcust_opapi -lopapi -lnnopbase -lascendcl -ldl \
    -o "${OUT}"

echo "[build_test_tq4bit] Built ${OUT}"
echo "[build_test_tq4bit] Run with:"
echo "  ASCEND_CUSTOM_OPP_PATH=${CUSTOM_OPP_PATH} ${OUT}"
echo ""
echo "[build_test_tq4bit] Configurable env vars:"
echo "  TQ4BIT_SEQ_LEN=32        TQ4BIT_QUERY_TOKENS=1"
echo "  TQ4BIT_NUM_HEADS=4       TQ4BIT_NUM_KV_HEADS=2"
echo "  TQ4BIT_BLOCK_SIZE=16"
