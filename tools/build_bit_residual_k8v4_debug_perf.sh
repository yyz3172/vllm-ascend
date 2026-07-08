#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -d "${ASCEND_HOME_PATH}" && -d /usr/local/Ascend/cann-8.5.1 ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
fi

SOC_DIR=${SOC_DIR:-ascend910b}
BUILD_DIR=${BR_K8V4_BUILD_DIR:-"${ROOT_DIR}/mytmp/build_bit_residual_k8v4_debug_line"}
RUNTIME_OPP=${ASCEND_CUSTOM_OPP_PATH:-"${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"}
OP_NAME=bit_residual_pack_k8v4
OP_CLASS=BitResidualPackK8v4

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

export BIT_RESIDUAL_PACK_K8V4_DEBUG_LINE=ON

echo "[build_k8v4_debug] root=${ROOT_DIR}"
echo "[build_k8v4_debug] build=${BUILD_DIR}"
echo "[build_k8v4_debug] soc=${SOC_DIR}"
echo "[build_k8v4_debug] runtime_opp=${RUNTIME_OPP}"

rm -f \
    "${BUILD_DIR}/autogen/aclnn_${OP_NAME}.cpp" \
    "${BUILD_DIR}/autogen/aclnn_${OP_NAME}.h" \
    "${BUILD_DIR}/autogen/${OP_NAME}_proto.h" \
    "${BUILD_DIR}/autogen/aic-${SOC_DIR}-ops-info.json" \
    "${BUILD_DIR}/custom/op_impl/ai_core/tbe/config/${SOC_DIR}/aic-${SOC_DIR}-ops-info.json" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/${OP_NAME}/${OP_CLASS}.py" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/${OP_CLASS}-${OP_NAME}-0.sh" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/${OP_CLASS}-${OP_NAME}-1.sh"
rm -rf \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/${OP_NAME}/__pycache__" \
    "${BUILD_DIR}/binary/${SOC_DIR}/bin/${OP_NAME}"

cmake \
    -S csrc \
    -B "${BUILD_DIR}" \
    -DBUILD_OPEN_PROJECT=ON \
    -DASCEND_COMPUTE_UNIT="${SOC_DIR}" \
    -DASCEND_OP_NAME="${OP_NAME}" \
    -DVENDOR_NAME=vllm-ascend \
    -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
    -DCHECK_COMPATIBLE=true \
    -DBIT_RESIDUAL_PACK_K8V4_DEBUG_LINE=ON \
    -DCMAKE_BUILD_TYPE=Release

if ! grep -Eq "^${OP_CLASS},,.*(^|;)-g(;|$)" \
    "${BUILD_DIR}/autogen/custom_compile_options.ini"; then
    echo "[build_k8v4_debug] ERROR: ${OP_CLASS} compile options do not contain -g" >&2
    grep -E "^${OP_CLASS}," "${BUILD_DIR}/autogen/custom_compile_options.ini" >&2 || true
    exit 1
fi

cmake --build "${BUILD_DIR}" \
    --target opbuild_gen_default \
    --target "${OP_NAME}_${SOC_DIR}_py_copy" \
    -j "${BUILD_JOBS:-64}"

impl_py="${BUILD_DIR}/binary/${SOC_DIR}/src/${OP_NAME}/${OP_CLASS}.py"
if ! grep -Eq -- "'-g'" "${impl_py}"; then
    echo "[build_k8v4_debug] ERROR: regenerated impl Python does not contain -g: ${impl_py}" >&2
    exit 1
fi

rm -f \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/${OP_NAME}/${OP_NAME}_${SOC_DIR}_src_copy.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/${OP_NAME}_${SOC_DIR}_0.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/${OP_NAME}_${SOC_DIR}_1.done"
rm -rf "${BUILD_DIR}/binary/${SOC_DIR}/bin/${OP_NAME}"

cmake --build "${BUILD_DIR}" \
    --target "${OP_NAME}_${SOC_DIR}" \
    --target opapi \
    --target opsproto \
    -j "${BUILD_JOBS:-64}"

if ! command -v readelf >/dev/null 2>&1; then
    echo "[build_k8v4_debug] ERROR: readelf is required" >&2
    exit 1
fi

kernel_dir="${BUILD_DIR}/binary/${SOC_DIR}/bin/${OP_NAME}"
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

mkdir -p \
    "${RUNTIME_OPP}/op_api/lib" \
    "${RUNTIME_OPP}/op_api/include" \
    "${RUNTIME_OPP}/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64" \
    "${RUNTIME_OPP}/op_proto/inc" \
    "${RUNTIME_OPP}/op_proto/lib/linux/aarch64"

cp -a "${BUILD_DIR}/libcust_opapi.so" \
    "${RUNTIME_OPP}/op_api/lib/libcust_opapi.so"
cp -a "${BUILD_DIR}/libcust_opmaster_rt2.0.so" \
    "${RUNTIME_OPP}/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
if [[ -f "${BUILD_DIR}/compat/liboptiling.so" ]]; then
    cp -a "${BUILD_DIR}/compat/liboptiling.so" \
        "${RUNTIME_OPP}/op_impl/ai_core/tbe/op_tiling/liboptiling.so"
fi
cp -a "${BUILD_DIR}/libcust_opsproto_rt2.0.so" \
    "${RUNTIME_OPP}/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"
cp -a "${BUILD_DIR}/autogen/aclnn_${OP_NAME}.h" \
    "${RUNTIME_OPP}/op_api/include/aclnn_${OP_NAME}.h"
cp -a "${BUILD_DIR}/autogen/${OP_NAME}_proto.h" \
    "${RUNTIME_OPP}/op_proto/inc/${OP_NAME}_proto.h"

src_bin="${BUILD_DIR}/binary/${SOC_DIR}/bin/${OP_NAME}"
dst_bin="${RUNTIME_OPP}/op_impl/ai_core/tbe/kernel/${SOC_DIR}/${OP_NAME}"
src_impl="${BUILD_DIR}/binary/${SOC_DIR}/src/${OP_NAME}"
dst_impl="${RUNTIME_OPP}/op_impl/ai_core/tbe/vllm-ascend_impl/ascendc/${OP_NAME}"
src_dynamic="${BUILD_DIR}/impl/dynamic/${OP_NAME}.py"
dst_dynamic="${RUNTIME_OPP}/op_impl/ai_core/tbe/vllm-ascend_impl/dynamic/${OP_NAME}.py"

rm -rf "${dst_bin}" "${dst_impl}"
mkdir -p "${dst_bin}" "${dst_impl}" "$(dirname "${dst_dynamic}")"
cp -a "${src_bin}/." "${dst_bin}/"
cp -a "${src_impl}/." "${dst_impl}/"
cp -a "${src_dynamic}" "${dst_dynamic}"

while IFS= read -r kernel_obj; do
    if ! readelf -SW "${kernel_obj}" 2>/dev/null | grep -q '\.debug_line'; then
        echo "[build_k8v4_debug] ERROR: runtime object has no .debug_line: ${kernel_obj}" >&2
        exit 1
    fi
done < <(find "${dst_bin}" -maxdepth 1 -type f -name '*.o' | sort)

echo "[build_k8v4_debug] done"
