#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-"${ROOT_DIR}/ztmp/turboquant4bit_perf"}

PYTHON_BIN=${PYTHON_BIN:-python3}
TORCH_INFO=$("${PYTHON_BIN}" - <<'PY'
import os
import torch
import torch_npu

torch_root = os.path.dirname(torch.__file__)
torch_npu_root = os.path.dirname(torch_npu.__file__)
print(torch_root)
print(torch_npu_root)
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)

TORCH_ROOT=$(echo "${TORCH_INFO}" | sed -n '1p')
TORCH_NPU_ROOT=$(echo "${TORCH_INFO}" | sed -n '2p')
TORCH_CXX11_ABI=$(echo "${TORCH_INFO}" | sed -n '3p')
TORCH_LIBS_ROOT=$(cd "${TORCH_ROOT}/../torch.libs" && pwd)
PY_INCLUDE=$("${PYTHON_BIN}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["include"])
PY
)
PY_LIBDIR=$("${PYTHON_BIN}" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("LIBDIR"))
PY
)
PY_LDLIBRARY=$("${PYTHON_BIN}" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("LDLIBRARY"))
PY
)
PY_LIB=${PY_LDLIBRARY#lib}
PY_LIB=${PY_LIB%.so}

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -d "${ASCEND_HOME_PATH}" && -d /usr/local/Ascend/cann-8.5.1 ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
fi

mkdir -p "$(dirname "${OUT}")"

g++ -std=c++17 -O2 \
    -D_GLIBCXX_USE_CXX11_ABI="${TORCH_CXX11_ABI}" \
    -I"${PY_INCLUDE}" \
    -I"${TORCH_ROOT}/include" \
    -I"${TORCH_ROOT}/include/torch/csrc/api/include" \
    -I"${TORCH_NPU_ROOT}/include" \
    -I"${ASCEND_HOME_PATH}/include" \
    "${ROOT_DIR}/tools/turboquant4bit_perf.cpp" \
    -L"${TORCH_ROOT}/lib" \
    -L"${TORCH_LIBS_ROOT}" \
    -L"${TORCH_NPU_ROOT}/lib" \
    -L"${PY_LIBDIR}" \
    -L"${ASCEND_HOME_PATH}/lib64" \
    -Wl,-rpath,"${TORCH_ROOT}/lib:${TORCH_LIBS_ROOT}:${TORCH_NPU_ROOT}/lib:${PY_LIBDIR}:${ASCEND_HOME_PATH}/lib64:${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib" \
    -Wl,-rpath-link,"${TORCH_LIBS_ROOT}" \
    -ltorch -ltorch_cpu -ltorch_global_deps -lc10 -ltorch_npu -l"${PY_LIB}" -lascendcl -ldl \
    -o "${OUT}"

echo "Built ${OUT}"
