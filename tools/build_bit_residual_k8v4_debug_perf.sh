#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -d "${ASCEND_HOME_PATH}" && -d /usr/local/Ascend/cann-8.5.1 ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
fi

SOC_DIR=${SOC_DIR:-ascend910b}
RUNTIME_OPP=${ASCEND_CUSTOM_OPP_PATH:-"${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"}
OP_NAME=bit_residual_pack_k8v4
OP_CLASS=BitResidualPackK8v4

export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-${RUNTIME_OPP}}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_HOME_PATH}/set_env.sh"
fi
if [[ -f "${ROOT_DIR}/xrx_infoenvs" ]]; then
    # shellcheck disable=SC1090
    source "${ROOT_DIR}/xrx_infoenvs"
fi
if [[ -f "${RUNTIME_OPP}/bin/set_env.bash" ]]; then
    # shellcheck disable=SC1090
    source "${RUNTIME_OPP}/bin/set_env.bash"
fi

echo "[build_k8v4_debug] root=${ROOT_DIR}"
echo "[build_k8v4_debug] soc=${SOC_DIR}"
echo "[build_k8v4_debug] runtime_opp=${RUNTIME_OPP}"

echo "[build_k8v4_debug] Building debug kernels via build_debug_kernel.sh..."
BUILD_DIR=$(bash "${ROOT_DIR}/tools/build_debug_kernel.sh")

echo "[build_k8v4_debug] Verifying ${OP_NAME} kernel debug line section..."
kernel_dir="${BUILD_DIR}/binary/${SOC_DIR}/bin/${OP_NAME}"
if [[ ! -d "${kernel_dir}" ]]; then
    echo "[build_k8v4_debug] ERROR: missing kernel output directory: ${kernel_dir}" >&2
    exit 1
fi

found_obj=0
while IFS= read -r kernel_obj; do
    found_obj=1
    if ! readelf -SW "${kernel_obj}" 2>/dev/null | grep -q '\.debug_line'; then
        echo "[build_k8v4_debug] ERROR: ${kernel_obj} has no .debug_line" >&2
        exit 1
    fi
done < <(find "${kernel_dir}" -maxdepth 1 -type f -name '*.o' | sort)
if [[ ${found_obj} -eq 0 ]]; then
    echo "[build_k8v4_debug] ERROR: no kernel objects found under ${kernel_dir}" >&2
    exit 1
fi

echo "[build_k8v4_debug] Verifying runtime ${OP_NAME} kernel debug line section..."
dst_bin="${RUNTIME_OPP}/op_impl/ai_core/tbe/kernel/${SOC_DIR}/${OP_NAME}"
found_obj=0
while IFS= read -r kernel_obj; do
    found_obj=1
    if ! readelf -SW "${kernel_obj}" 2>/dev/null | grep -q '\.debug_line'; then
        echo "[build_k8v4_debug] ERROR: runtime object has no .debug_line: ${kernel_obj}" >&2
        exit 1
    fi
done < <(find "${dst_bin}" -maxdepth 1 -type f -name '*.o' | sort)
if [[ ${found_obj} -eq 0 ]]; then
    echo "[build_k8v4_debug] ERROR: no runtime kernel objects found under ${dst_bin}" >&2
    exit 1
fi

echo "[build_k8v4_debug] done"
