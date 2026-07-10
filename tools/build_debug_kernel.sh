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

export TURBOQUANT_PACK_KV_FOR_CACHE4BIT_DEBUG_LINE=ON
export TURBOQUANT_ATTENTION_PAGED4BIT_DEBUG_LINE=ON
export BIT_RESIDUAL_PACK_K8V4_DEBUG_LINE=ON
export BIT_RESIDUAL_ATTENTION_PAGED_K8V4_DEBUG_LINE=ON

BUILD_DIR="${ROOT_DIR}/ztmp/build_libcust_opapi_nodebug_debug_line"
RUNTIME_OPP="${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"

echo "[build_debug_kernel] Building custom ops with debug line switches..."
echo "[build_debug_kernel] Build directory: ${BUILD_DIR}"

rm -f \
    "${BUILD_DIR}/autogen/aclnn_turboquant_pack_kv_for_cache4bit.cpp" \
    "${BUILD_DIR}/autogen/aclnn_turboquant_pack_kv_for_cache4bit.h" \
    "${BUILD_DIR}/autogen/aclnn_turboquant_attention_paged4bit.cpp" \
    "${BUILD_DIR}/autogen/aclnn_turboquant_attention_paged4bit.h" \
    "${BUILD_DIR}/autogen/aclnn_bit_residual_pack_k8v4.cpp" \
    "${BUILD_DIR}/autogen/aclnn_bit_residual_pack_k8v4.h" \
    "${BUILD_DIR}/autogen/aclnn_bit_residual_attention_paged_k8v4.cpp" \
    "${BUILD_DIR}/autogen/aclnn_bit_residual_attention_paged_k8v4.h" \
    "${BUILD_DIR}/autogen/aic-${SOC_DIR}-ops-info.json" \
    "${BUILD_DIR}/custom/op_impl/ai_core/tbe/config/${SOC_DIR}/aic-${SOC_DIR}-ops-info.json" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/turboquant_pack_kv_for_cache4bit/TurboquantPackKvForCache4bit.py" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/turboquant_attention_paged4bit/TurboquantAttentionPaged4bit.py" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/bit_residual_pack_k8v4/BitResidualPackK8v4.py" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/bit_residual_attention_paged_k8v4/BitResidualAttentionPagedK8v4.py" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/TurboquantPackKvForCache4bit-turboquant_pack_kv_for_cache4bit-0.sh" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/TurboquantPackKvForCache4bit-turboquant_pack_kv_for_cache4bit-1.sh" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/TurboquantAttentionPaged4bit-turboquant_attention_paged4bit-0.sh" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/TurboquantAttentionPaged4bit-turboquant_attention_paged4bit-1.sh" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/BitResidualPackK8v4-bit_residual_pack_k8v4-0.sh" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/BitResidualPackK8v4-bit_residual_pack_k8v4-1.sh" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/BitResidualAttentionPagedK8v4-bit_residual_attention_paged_k8v4-0.sh" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/BitResidualAttentionPagedK8v4-bit_residual_attention_paged_k8v4-1.sh"
rm -rf \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/turboquant_pack_kv_for_cache4bit/__pycache__" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/turboquant_attention_paged4bit/__pycache__" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/bit_residual_pack_k8v4/__pycache__" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/bit_residual_attention_paged_k8v4/__pycache__"
find "${BUILD_DIR}/binary/${SOC_DIR}/gen" -maxdepth 1 -type f \
    \( -name 'TurboquantPackKvForCache4bit_*' -o \
       -name 'TurboquantPackKvForCache4bit-turboquant_pack_kv_for_cache4bit-*' -o \
       -name 'BitResidualPackK8v4_*' -o \
       -name 'BitResidualPackK8v4-bit_residual_pack_k8v4-*' -o \
       -name 'BitResidualAttentionPagedK8v4_*' -o \
       -name 'BitResidualAttentionPagedK8v4-bit_residual_attention_paged_k8v4-*' \) \
    -delete 2>/dev/null || true

echo "[build_debug_kernel] Configuring CMake..."
cmake \
    -S csrc \
    -B "${BUILD_DIR}" \
    -DBUILD_OPEN_PROJECT=ON \
    -DASCEND_COMPUTE_UNIT="${SOC_DIR}" \
    -DASCEND_OP_NAME=ALL \
    -DVENDOR_NAME=vllm-ascend \
    -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
    -DCHECK_COMPATIBLE=true \
    -DTURBOQUANT_PACK_KV_FOR_CACHE4BIT_DEBUG_LINE=ON \
    -DTURBOQUANT_ATTENTION_PAGED4BIT_DEBUG_LINE=ON \
    -DBIT_RESIDUAL_PACK_K8V4_DEBUG_LINE=ON \
    -DBIT_RESIDUAL_ATTENTION_PAGED_K8V4_DEBUG_LINE=ON \
    -DCMAKE_BUILD_TYPE=Release

if ! grep -Eq '^Turboquant(PackKvForCache4bit|AttentionPaged4bit),,.*(^|;)-g(;|$)' \
    "${BUILD_DIR}/autogen/custom_compile_options.ini"; then
    echo "[build_debug_kernel] ERROR: CMake configure did not add -g to 4bit custom compile options" >&2
    grep -E '^Turboquant(PackKvForCache4bit|AttentionPaged4bit),' \
        "${BUILD_DIR}/autogen/custom_compile_options.ini" >&2 || true
    exit 1
fi

if ! grep -Eq '^BitResidual(PackK8v4|AttentionPagedK8v4),,.*(^|;)-g(;|$)' \
    "${BUILD_DIR}/autogen/custom_compile_options.ini"; then
    echo "[build_debug_kernel] ERROR: CMake configure did not add -g to bit_residual custom compile options" >&2
    grep -E '^BitResidual(PackK8v4|AttentionPagedK8v4),' \
        "${BUILD_DIR}/autogen/custom_compile_options.ini" >&2 || true
    exit 1
fi

echo "[build_debug_kernel] Building Python wrappers and generated files..."
cmake --build "${BUILD_DIR}" \
    --target opbuild_gen_default \
    --target turboquant_pack_kv_for_cache4bit_${SOC_DIR}_py_copy \
    --target turboquant_attention_paged4bit_${SOC_DIR}_py_copy \
    --target bit_residual_pack_k8v4_${SOC_DIR}_py_copy \
    --target bit_residual_attention_paged_k8v4_${SOC_DIR}_py_copy \
    -j "${BUILD_JOBS:-64}"

if ! grep -Eq -- "'-g'" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/turboquant_pack_kv_for_cache4bit/TurboquantPackKvForCache4bit.py"; then
    echo "[build_debug_kernel] ERROR: regenerated pack4bit impl Python does not contain -g" >&2
    exit 1
fi
if ! grep -Eq -- "'-g'" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/turboquant_attention_paged4bit/TurboquantAttentionPaged4bit.py"; then
    echo "[build_debug_kernel] ERROR: regenerated attention4bit impl Python does not contain -g" >&2
    exit 1
fi
if ! grep -Eq -- "'-g'" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/bit_residual_pack_k8v4/BitResidualPackK8v4.py"; then
    echo "[build_debug_kernel] ERROR: regenerated bit_residual_pack_k8v4 impl Python does not contain -g" >&2
    exit 1
fi
if ! grep -Eq -- "'-g'" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/bit_residual_attention_paged_k8v4/BitResidualAttentionPagedK8v4.py"; then
    echo "[build_debug_kernel] ERROR: regenerated bit_residual_attention_paged_k8v4 impl Python does not contain -g" >&2
    exit 1
fi

echo "[build_debug_kernel] Cleaning intermediate files and building kernels..."
rm -f \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/turboquant_pack_kv_for_cache4bit/turboquant_pack_kv_for_cache4bit_${SOC_DIR}_src_copy.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/turboquant_pack_kv_for_cache4bit_${SOC_DIR}_0.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/turboquant_pack_kv_for_cache4bit_${SOC_DIR}_1.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/turboquant_attention_paged4bit/turboquant_attention_paged4bit_${SOC_DIR}_src_copy.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/turboquant_attention_paged4bit_${SOC_DIR}_0.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/turboquant_attention_paged4bit_${SOC_DIR}_1.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/bit_residual_pack_k8v4/bit_residual_pack_k8v4_${SOC_DIR}_src_copy.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/bit_residual_pack_k8v4_${SOC_DIR}_0.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/bit_residual_pack_k8v4_${SOC_DIR}_1.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/src/bit_residual_attention_paged_k8v4/bit_residual_attention_paged_k8v4_${SOC_DIR}_src_copy.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/bit_residual_attention_paged_k8v4_${SOC_DIR}_0.done" \
    "${BUILD_DIR}/binary/${SOC_DIR}/gen/bit_residual_attention_paged_k8v4_${SOC_DIR}_1.done"
rm -rf \
    "${BUILD_DIR}/binary/${SOC_DIR}/bin/turboquant_pack_kv_for_cache4bit" \
    "${BUILD_DIR}/binary/${SOC_DIR}/bin/turboquant_attention_paged4bit" \
    "${BUILD_DIR}/binary/${SOC_DIR}/bin/bit_residual_pack_k8v4" \
    "${BUILD_DIR}/binary/${SOC_DIR}/bin/bit_residual_attention_paged_k8v4"

echo "[build_debug_kernel] Building 4bit kernels with debug line..."
cmake --build "${BUILD_DIR}" \
    --target turboquant_pack_kv_for_cache4bit_${SOC_DIR} \
    --target turboquant_attention_paged4bit_${SOC_DIR} \
    --target bit_residual_pack_k8v4_${SOC_DIR} \
    --target bit_residual_attention_paged_k8v4_${SOC_DIR} \
    -j "${BUILD_JOBS:-64}"

echo "[build_debug_kernel] Successfully built custom ops with debug line"

echo "[build_debug_kernel] Building and syncing ACLNN host libraries..."
cmake --build "${BUILD_DIR}" \
    --target opapi \
    --target opsproto \
    -j "${BUILD_JOBS:-64}"

for required_file in \
    "${BUILD_DIR}/libcust_opapi.so" \
    "${BUILD_DIR}/libcust_opmaster_rt2.0.so" \
    "${BUILD_DIR}/libcust_opsproto_rt2.0.so" \
    "${BUILD_DIR}/autogen/aclnn_turboquant_pack_kv_for_cache4bit.h" \
    "${BUILD_DIR}/autogen/turboquant_pack_kv_for_cache4bit_proto.h" \
    "${BUILD_DIR}/autogen/aclnn_turboquant_attention_paged4bit.h" \
    "${BUILD_DIR}/autogen/turboquant_attention_paged4bit_proto.h" \
    "${BUILD_DIR}/autogen/aclnn_bit_residual_pack_k8v4.h" \
    "${BUILD_DIR}/autogen/bit_residual_pack_k8v4_proto.h" \
    "${BUILD_DIR}/autogen/aclnn_bit_residual_attention_paged_k8v4.h" \
    "${BUILD_DIR}/autogen/bit_residual_attention_paged_k8v4_proto.h"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "[build_debug_kernel] ERROR: missing generated ACLNN artifact: ${required_file}" >&2
        exit 1
    fi
done

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
cp -a "${BUILD_DIR}/autogen/aclnn_turboquant_pack_kv_for_cache4bit.h" \
    "${RUNTIME_OPP}/op_api/include/aclnn_turboquant_pack_kv_for_cache4bit.h"
cp -a "${BUILD_DIR}/autogen/turboquant_pack_kv_for_cache4bit_proto.h" \
    "${RUNTIME_OPP}/op_proto/inc/turboquant_pack_kv_for_cache4bit_proto.h"
cp -a "${BUILD_DIR}/autogen/aclnn_turboquant_attention_paged4bit.h" \
    "${RUNTIME_OPP}/op_api/include/aclnn_turboquant_attention_paged4bit.h"
cp -a "${BUILD_DIR}/autogen/turboquant_attention_paged4bit_proto.h" \
    "${RUNTIME_OPP}/op_proto/inc/turboquant_attention_paged4bit_proto.h"
cp -a "${BUILD_DIR}/autogen/aclnn_bit_residual_pack_k8v4.h" \
    "${RUNTIME_OPP}/op_api/include/aclnn_bit_residual_pack_k8v4.h"
cp -a "${BUILD_DIR}/autogen/bit_residual_pack_k8v4_proto.h" \
    "${RUNTIME_OPP}/op_proto/inc/bit_residual_pack_k8v4_proto.h"
cp -a "${BUILD_DIR}/autogen/aclnn_bit_residual_attention_paged_k8v4.h" \
    "${RUNTIME_OPP}/op_api/include/aclnn_bit_residual_attention_paged_k8v4.h"
cp -a "${BUILD_DIR}/autogen/bit_residual_attention_paged_k8v4_proto.h" \
    "${RUNTIME_OPP}/op_proto/inc/bit_residual_attention_paged_k8v4_proto.h"

echo "[build_debug_kernel] Verifying final debug line sections..."
if ! command -v readelf >/dev/null 2>&1; then
    echo "[build_debug_kernel] ERROR: readelf is required to verify debug line sections" >&2
    exit 1
fi

for op_dir in \
    "${BUILD_DIR}/binary/${SOC_DIR}/bin/turboquant_pack_kv_for_cache4bit" \
    "${BUILD_DIR}/binary/${SOC_DIR}/bin/turboquant_attention_paged4bit" \
    "${BUILD_DIR}/binary/${SOC_DIR}/bin/bit_residual_pack_k8v4" \
    "${BUILD_DIR}/binary/${SOC_DIR}/bin/bit_residual_attention_paged_k8v4"; do
    if [[ ! -d "${op_dir}" ]]; then
        echo "[build_debug_kernel] ERROR: missing kernel output directory: ${op_dir}" >&2
        exit 1
    fi
    found_obj=0
    while IFS= read -r kernel_obj; do
        found_obj=1
        if ! readelf -SW "${kernel_obj}" 2>/dev/null | grep -q '\.debug_line'; then
            echo "[build_debug_kernel] ERROR: ${kernel_obj} has no .debug_line section" >&2
            exit 1
        fi
    done < <(find "${op_dir}" -maxdepth 1 -type f -name '*.o' | sort)
    if [[ ${found_obj} -eq 0 ]]; then
        echo "[build_debug_kernel] ERROR: no kernel objects found under ${op_dir}" >&2
        exit 1
    fi
done

echo "[build_debug_kernel] Syncing debug-line kernels into runtime custom OPP..."
for op in turboquant_pack_kv_for_cache4bit turboquant_attention_paged4bit bit_residual_pack_k8v4 bit_residual_attention_paged_k8v4; do
    src_bin="${BUILD_DIR}/binary/${SOC_DIR}/bin/${op}"
    dst_bin="${RUNTIME_OPP}/op_impl/ai_core/tbe/kernel/${SOC_DIR}/${op}"
    src_impl="${BUILD_DIR}/binary/${SOC_DIR}/src/${op}"
    dst_impl="${RUNTIME_OPP}/op_impl/ai_core/tbe/vllm-ascend_impl/ascendc/${op}"
    src_dynamic="${BUILD_DIR}/impl/dynamic/${op}.py"
    dst_dynamic="${RUNTIME_OPP}/op_impl/ai_core/tbe/vllm-ascend_impl/dynamic/${op}.py"
    if [[ ! -d "${src_bin}" || ! -d "${src_impl}" ]]; then
        echo "[build_debug_kernel] ERROR: missing debug build output for ${op}" >&2
        exit 1
    fi
    if [[ ! -f "${src_dynamic}" ]]; then
        echo "[build_debug_kernel] ERROR: missing debug dynamic impl for ${op}: ${src_dynamic}" >&2
        exit 1
    fi
    rm -rf "${dst_bin}" "${dst_impl}"
    mkdir -p "${dst_bin}" "${dst_impl}"
    cp -a "${src_bin}/." "${dst_bin}/"
    cp -a "${src_impl}/." "${dst_impl}/"
    mkdir -p "$(dirname "${dst_dynamic}")"
    cp -a "${src_dynamic}" "${dst_dynamic}"
done

echo "[build_debug_kernel] Verifying runtime custom OPP kernel debug line sections..."
for op in turboquant_pack_kv_for_cache4bit turboquant_attention_paged4bit bit_residual_pack_k8v4 bit_residual_attention_paged_k8v4; do
    runtime_bin="${RUNTIME_OPP}/op_impl/ai_core/tbe/kernel/${SOC_DIR}/${op}"
    if [[ ! -d "${runtime_bin}" ]]; then
        echo "[build_debug_kernel] ERROR: missing runtime kernel output directory: ${runtime_bin}" >&2
        exit 1
    fi
    found_obj=0
    while IFS= read -r kernel_obj; do
        found_obj=1
        if ! readelf -SW "${kernel_obj}" 2>/dev/null | grep -q '\.debug_line'; then
            echo "[build_debug_kernel] ERROR: runtime kernel ${kernel_obj} has no .debug_line section" >&2
            exit 1
        fi
    done < <(find "${runtime_bin}" -maxdepth 1 -type f -name '*.o' | sort)
    if [[ ${found_obj} -eq 0 ]]; then
        echo "[build_debug_kernel] ERROR: no runtime kernel objects found under ${runtime_bin}" >&2
        exit 1
    fi
done

ops_info_src="${BUILD_DIR}/custom/op_impl/ai_core/tbe/config/${SOC_DIR}/aic-${SOC_DIR}-ops-info.json"
if [[ ! -f "${ops_info_src}" ]]; then
    ops_info_src="${BUILD_DIR}/autogen/aic-${SOC_DIR}-ops-info.json"
fi
if [[ ! -f "${ops_info_src}" ]]; then
    echo "[build_debug_kernel] ERROR: missing generated ops info json" >&2
    exit 1
fi
mkdir -p "${RUNTIME_OPP}/op_impl/ai_core/tbe/config/${SOC_DIR}"
cp -a "${ops_info_src}" \
    "${RUNTIME_OPP}/op_impl/ai_core/tbe/config/${SOC_DIR}/aic-${SOC_DIR}-ops-info.json"

tmp_config_root="${ROOT_DIR}/ztmp/debug_line_runtime_config_root"
rm -rf "${tmp_config_root}"
mkdir -p "${tmp_config_root}/${SOC_DIR}/bin"
cp -a "${RUNTIME_OPP}/op_impl/ai_core/tbe/kernel/${SOC_DIR}/." \
    "${tmp_config_root}/${SOC_DIR}/bin/"
python "${ASCEND_HOME_PATH}/tools/op_project_templates/ascendc/customize/cmake/util/ascendc_ops_config.py" \
    -p "${tmp_config_root}/${SOC_DIR}/bin" \
    -s "${SOC_DIR}"
mkdir -p "${RUNTIME_OPP}/op_impl/ai_core/tbe/kernel/config/${SOC_DIR}"
cp -a "${tmp_config_root}/${SOC_DIR}/bin/binary_info_config.json" \
    "${RUNTIME_OPP}/op_impl/ai_core/tbe/kernel/config/${SOC_DIR}/binary_info_config.json"

for op in turboquant_pack_kv_for_cache4bit turboquant_attention_paged4bit bit_residual_pack_k8v4 bit_residual_attention_paged_k8v4; do
    src_json="${tmp_config_root}/${SOC_DIR}/bin/${op}.json"
    dst_json="${RUNTIME_OPP}/op_impl/ai_core/tbe/kernel/config/${SOC_DIR}/${op}.json"
    if [[ -f "${src_json}" ]]; then
        cp -a "${src_json}" "${dst_json}"
    fi
done

echo "[build_debug_kernel] ============================================"
echo "[build_debug_kernel] Build completed successfully!"
echo "[build_debug_kernel] ============================================"
echo "[build_debug_kernel] Custom ops build dir: ${BUILD_DIR}"
echo "[build_debug_kernel] Runtime custom OPP: ${RUNTIME_OPP}"
echo "[build_debug_kernel] ============================================"

echo "${BUILD_DIR}"
