#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -d "${ASCEND_HOME_PATH}" && -d /usr/local/Ascend/cann-8.5.1 ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
fi

SOC_DIR=${SOC_DIR:-ascend910b}

if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_HOME_PATH}/set_env.sh"
fi
if [[ -f "${ROOT_DIR}/xrx_infoenvs" ]]; then
    # shellcheck disable=SC1090
    source "${ROOT_DIR}/xrx_infoenvs"
fi

export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend}
if [[ -f "${ASCEND_CUSTOM_OPP_PATH}/bin/set_env.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_CUSTOM_OPP_PATH}/bin/set_env.bash"
fi

BUILD_DIR="${ROOT_DIR}/ztmp/build_libcust_opapi_nodebug_debug_line"
FLEX_RUNNER="${ROOT_DIR}/ztmp/flex_tq_4bit_perf/flex_tq_4bit_perf"

echo "[build_debug_perf] Building debug kernels..."
BUILD_DIR=$(bash "${ROOT_DIR}/tools/build_debug_kernel.sh")

echo "[build_debug_perf] Building flex 4bit C++ runner..."
mkdir -p "$(dirname "${FLEX_RUNNER}")"
bash "${ROOT_DIR}/tools/build_flex_tq_4bit_perf.sh" "${FLEX_RUNNER}"

echo "[build_debug_perf] Verifying flex C++ runner debug line section..."
if ! readelf -SW "${FLEX_RUNNER}" 2>/dev/null | grep -q '\.debug_line'; then
    echo "[build_debug_perf] ERROR: flex C++ runner has no .debug_line section: ${FLEX_RUNNER}" >&2
    exit 1
fi

echo "[build_debug_perf] ============================================"
echo "[build_debug_perf] Build completed successfully!"
echo "[build_debug_perf] ============================================"
echo "[build_debug_perf] Custom ops build dir: ${BUILD_DIR}"
echo "[build_debug_perf] Flex C++ runner: ${FLEX_RUNNER}"
echo "[build_debug_perf] MindStudio source requires runtime OPP kernel .o debug lines,"
echo "[build_debug_perf] not only host runner/libcust_opapi.so debug info."
echo "[build_debug_perf] ============================================"
