#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
Z_TMP="${ROOT_DIR}/ztmp"
OUT="${1:-"${Z_TMP}/bit_residual_k8v4_perf/bit_residual_k8v4_perf"}"

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -d "${ASCEND_HOME_PATH}" && -d /usr/local/Ascend/cann-8.5.1 ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
fi

CUSTOM_OP_LIB="${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib"

mkdir -p "$(dirname "${OUT}")"

echo "[build_k8v4_perf] root=${ROOT_DIR}"
echo "[build_k8v4_perf] out=${OUT}"
echo "[build_k8v4_perf] ascend_home=${ASCEND_HOME_PATH}"
echo "[build_k8v4_perf] custom_op_lib=${CUSTOM_OP_LIB}"

g++ -std=c++17 -O2 -g -fno-omit-frame-pointer -rdynamic \
    -I"${ASCEND_HOME_PATH}/include" \
    -I"${ROOT_DIR}/csrc" \
    "${ROOT_DIR}/tools/bit_residual_pack_k8v4_perf.cpp" \
    -L"${ASCEND_HOME_PATH}/lib64" \
    -L"${CUSTOM_OP_LIB}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64:${CUSTOM_OP_LIB}" \
    -Wl,-rpath-link,"${ASCEND_HOME_PATH}/lib64:${CUSTOM_OP_LIB}" \
    -lcust_opapi -lopapi -lascendcl -ldl \
    -o "${OUT}"

echo "Built ${OUT}"
echo "Debug symbols: $(file "${OUT}")"
echo ""
echo "Usage:"
echo "  # Build with default output path"
echo "  bash tools/build_bit_residual_k8v4_perf.sh"
echo ""
echo "  # Build with custom output path"
echo "  bash tools/build_bit_residual_k8v4_perf.sh /path/to/output_binary"
echo ""
echo "Run in docker container:"
echo "  docker exec -it vllm.x00827378 bash"
echo "  cd ."
echo "  bash tools/run_bit_residual_k8v4_perf.sh [scenario]"