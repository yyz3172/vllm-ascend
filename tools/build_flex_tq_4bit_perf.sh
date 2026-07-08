#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-"${ROOT_DIR}/mytmp/flex_tq_4bit_perf/flex_tq_4bit_perf"}

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -d "${ASCEND_HOME_PATH}" && -d /usr/local/Ascend/cann-8.5.1 ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
fi

CUSTOM_OP_LIB="${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib"

mkdir -p "$(dirname "${OUT}")"

g++ -std=c++17 -O2 -g -fno-omit-frame-pointer -rdynamic \
    -I"${ASCEND_HOME_PATH}/include" \
    -I"${ROOT_DIR}/csrc" \
    "${ROOT_DIR}/tools/flex_tq_4bit_perf.cpp" \
    -L"${ASCEND_HOME_PATH}/lib64" \
    -L"${CUSTOM_OP_LIB}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64:${CUSTOM_OP_LIB}" \
    -Wl,-rpath-link,"${ASCEND_HOME_PATH}/lib64:${CUSTOM_OP_LIB}" \
    -lcust_opapi -lopapi -lascendcl -ldl \
    -o "${OUT}"

echo "Built ${OUT}"
echo "Debug symbols: $(file "${OUT}")"
