#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}
COMPUTE_UNIT=${COMPUTE_UNIT:-ascend910b}
VENDOR_NAME=${VENDOR_NAME:-vllm-ascend}
KENEL_DEBUG_LINE=${KENEL_DEBUG_LINE:-${KERNEL_DEBUG_LINE:-OFF}}
JOBS=${JOBS:-$(nproc)}
SYNC_TO_VENDOR=${SYNC_TO_VENDOR:-1}
echo "ROOT_DIR ${ROOT_DIR}"
echo "KENEL_DEBUG_LINE ${KENEL_DEBUG_LINE}"
rm -rf ${ROOT_DIR}/build
rm -rf ${ROOT_DIR}/csrc/build
case "${KENEL_DEBUG_LINE^^}" in
    ON)
        build_type="RelWithDebInfo"
        build_suffix="debug"
        ;;
    OFF)
        build_type="Release"
        build_suffix="nodebug"
        ;;
    *)
        echo "KENEL_DEBUG_LINE must be ON or OFF, got: ${KENEL_DEBUG_LINE}" >&2
        exit 1
        ;;
esac

BUILD_DIR=${BUILD_DIR:-${ROOT_DIR}/csrc/build_libcust_opapi_${build_suffix}}

if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_HOME_PATH}/bin/setenv.bash"
fi

export KENEL_DEBUG_LINE="${KENEL_DEBUG_LINE^^}"
export KERNEL_DEBUG_LINE="${KERNEL_DEBUG_LINE:-${KENEL_DEBUG_LINE}}"

echo "[build_libcust_opapi] KENEL_DEBUG_LINE=${KENEL_DEBUG_LINE}"
echo "[build_libcust_opapi] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "[build_libcust_opapi] COMPUTE_UNIT=${COMPUTE_UNIT}"
echo "[build_libcust_opapi] build_dir=${BUILD_DIR}"

cmake_args=(
    -S "${ROOT_DIR}/csrc"
    -B "${BUILD_DIR}"
    -DBUILD_OPEN_PROJECT=ON
    -DASCEND_COMPUTE_UNIT="${COMPUTE_UNIT}"
    -DASCEND_OP_NAME=ALL
    -DVENDOR_NAME="${VENDOR_NAME}"
    -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
    -DCHECK_COMPATIBLE=true
    -DCMAKE_BUILD_TYPE="${build_type}"
    "-DCMAKE_C_FLAGS:STRING="
    "-DCMAKE_CXX_FLAGS:STRING="
    "-DCMAKE_C_FLAGS_RELEASE:STRING=-O3 -DNDEBUG"
    "-DCMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG"
    "-DCMAKE_C_FLAGS_RELWITHDEBINFO:STRING=-O2 -g -DNDEBUG"
    "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=-O2 -g -DNDEBUG"
)

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" --target opapi -j "${JOBS}"

lib_path=$(find "${BUILD_DIR}" -maxdepth 3 -type f -name 'libcust_opapi.so' | sort | tail -n 1)
if [[ -z ${lib_path} ]]; then
    echo "[build_libcust_opapi] libcust_opapi.so was not generated under ${BUILD_DIR}" >&2
    exit 1
fi

has_debug_line=0
if command -v readelf >/dev/null 2>&1; then
    readelf_sections=$(readelf -S "${lib_path}" 2>/dev/null || true)
    if grep -q '\.debug_line' <<<"${readelf_sections}"; then
        has_debug_line=1
    fi
else
    echo "[build_libcust_opapi] readelf not found; skipping .debug_line validation" >&2
fi

if [[ ${KENEL_DEBUG_LINE} == "ON" && ${has_debug_line} -ne 1 ]]; then
    echo "[build_libcust_opapi] expected .debug_line, but it is missing in ${lib_path}" >&2
    exit 1
fi
if [[ ${KENEL_DEBUG_LINE} == "OFF" && ${has_debug_line} -ne 0 ]]; then
    echo "[build_libcust_opapi] expected no .debug_line, but it is present in ${lib_path}" >&2
    exit 1
fi

if [[ ${SYNC_TO_VENDOR} == "1" ]]; then
    opp_root=${ASCEND_CUSTOM_OPP_PATH:-${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/${VENDOR_NAME}}
    opp_root=${opp_root%%:*}
    dst_dir="${opp_root}/op_api/lib"
    mkdir -p "${dst_dir}"
    cp -f "${lib_path}" "${dst_dir}/libcust_opapi.so"
    echo "[build_libcust_opapi] installed: ${dst_dir}/libcust_opapi.so"
fi

echo "[build_libcust_opapi] built: ${lib_path}"
echo "[build_libcust_opapi] debug_line=$([[ ${has_debug_line} -eq 1 ]] && echo present || echo absent)"
