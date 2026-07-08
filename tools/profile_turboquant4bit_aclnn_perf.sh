#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}
SIM_SOC=${SIM_SOC:-Ascend910B3}
SOC_DIR=${SOC_DIR:-ascend910b}
PERF_OUT=${PERF_OUT:-${ROOT_DIR}/mytmp/perf_out_tq4bit_aclnn}
RUN_ID=${RUN_ID:-$(date +%Y%m%d%H%M%S)}
MSPROF_LOG=${MSPROF_LOG:-${PERF_OUT}/msprof_op_simulator_tq4bit_aclnn_${RUN_ID}.log}
LAUNCH_COUNT=${LAUNCH_COUNT:-1}
REQUIRE_DEBUG_LINE=${REQUIRE_DEBUG_LINE:-1}
AUTO_BUILD_DEBUG_LINE=${AUTO_BUILD_DEBUG_LINE:-1}
RUNNER=${RUNNER:-${ROOT_DIR}/mytmp/flex_tq_4bit_perf/flex_tq_4bit_perf}
AUTO_BUILD_RUNNER=${AUTO_BUILD_RUNNER:-1}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    cat <<'EOF'
Usage: ./profile_turboquant4bit_aclnn_perf.sh [runner options]

Profiles tools/flex_tq_4bit_perf with msprof op simulator.
The no-argument default is a short simulator smoke profile. Pass runner
options explicitly for longer performance runs.

Common runner options:
  --attention-only
  --pack-only
  --q-lens L[:L...]
  --kv-lens L[:L...]
  --heads N
  --kv-heads N
  --block-size N
  --pack-tokens N
  --warmup N
  --repeat N

Environment:
  REQUIRE_DEBUG_LINE=1       Require kernel .debug_line sections, default 1.
  AUTO_BUILD_DEBUG_LINE=1    Rebuild 4bit kernels with per-op debug switches
                             when .debug_line is missing, default 1.
  AUTO_BUILD_RUNNER=1        Build the C++ runner when missing, default 1.
  PERF_OUT=DIR               Output directory, default perf_out_tq4bit_aclnn.
  SIM_SOC=Ascend910B3        msprof simulator SoC, default Ascend910B3.
EOF
    exit 0
fi

mkdir -p "${PERF_OUT}"

export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend}
if [[ -f "${ASCEND_CUSTOM_OPP_PATH}/bin/set_env.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_CUSTOM_OPP_PATH}/bin/set_env.bash"
fi

# Let msprof op simulator --soc-version own the simulator runtime setup.
unset CAMODEL_LOG_PATH
unset CAMODEL_CONFIG_PATH
if [[ ${LD_PRELOAD:-} == *"/tools/simulator/"* ]]; then
    echo "[profile_tq4bit] unsetting simulator LD_PRELOAD from caller environment"
    unset LD_PRELOAD
fi

if ! command -v msprof >/dev/null 2>&1; then
    if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
        # shellcheck disable=SC1090
        source "${ASCEND_HOME_PATH}/set_env.sh"
    fi
fi

collect_kernel_objects() {
    local kernel_dir
    kernel_objects=()
    for kernel_dir in "${kernel_roots[@]}"; do
        if [[ ! -d "${kernel_dir}" ]]; then
            return 1
        fi
        while IFS= read -r obj; do
            kernel_objects+=("${obj}")
        done < <(find "${kernel_dir}" -maxdepth 1 -type f -name '*.o' | sort)
    done
    [[ ${#kernel_objects[@]} -gt 0 ]]
}

kernel_objects_have_debug_line() {
    local kernel_obj section_headers
    for kernel_obj in "${kernel_objects[@]}"; do
        section_headers=$(readelf -SW "${kernel_obj}" 2>/dev/null || true)
        if [[ ${section_headers} != *".debug_line"* ]]; then
            missing_debug_line_obj=${kernel_obj}
            return 1
        fi
    done
    return 0
}

sync_4bit_runtime_artifacts() {
    local src_bin dst_bin src_impl dst_impl op

    for op in turboquant_pack_kv_for_cache4bit turboquant_attention_paged4bit; do
        src_bin="${ROOT_DIR}/mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/bin/${op}"
        dst_bin="${opp_root}/op_impl/ai_core/tbe/kernel/${SOC_DIR}/${op}"
        src_impl="${ROOT_DIR}/mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/src/${op}"
        dst_impl="${opp_root}/op_impl/ai_core/tbe/vllm-ascend_impl/ascendc/${op}"
        rm -rf "${dst_bin}" "${dst_impl}"
        mkdir -p "${dst_bin}" "${dst_impl}"
        cp -a "${src_bin}/." "${dst_bin}/"
        cp -a "${src_impl}/." "${dst_impl}/"
    done

    local ops_info_src
    ops_info_src="${ROOT_DIR}/mytmp/build_libcust_opapi_nodebug/custom/op_impl/ai_core/tbe/config/${SOC_DIR}/aic-${SOC_DIR}-ops-info.json"
    if [[ ! -f "${ops_info_src}" ]]; then
        ops_info_src="${ROOT_DIR}/mytmp/build_libcust_opapi_nodebug/autogen/aic-${SOC_DIR}-ops-info.json"
    fi
    cp -a "${ops_info_src}" \
        "${opp_root}/op_impl/ai_core/tbe/config/${SOC_DIR}/aic-${SOC_DIR}-ops-info.json"
    cp -a "${ROOT_DIR}/mytmp/build_libcust_opapi_nodebug/libcust_opapi.so" \
        "${opp_root}/op_api/lib/libcust_opapi.so"
    cp -a "${ROOT_DIR}/mytmp/build_libcust_opapi_nodebug/libcust_opsproto_rt2.0.so" \
        "${opp_root}/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"
    cp -a "${ROOT_DIR}/mytmp/build_libcust_opapi_nodebug/autogen/aclnn_turboquant_pack_kv_for_cache4bit.h" \
        "${opp_root}/op_api/include/aclnn_turboquant_pack_kv_for_cache4bit.h"
    cp -a "${ROOT_DIR}/mytmp/build_libcust_opapi_nodebug/autogen/turboquant_pack_kv_for_cache4bit_proto.h" \
        "${opp_root}/op_proto/inc/turboquant_pack_kv_for_cache4bit_proto.h"
    cp -a "${ROOT_DIR}/mytmp/build_libcust_opapi_nodebug/autogen/aclnn_turboquant_attention_paged4bit.h" \
        "${opp_root}/op_api/include/aclnn_turboquant_attention_paged4bit.h"
    cp -a "${ROOT_DIR}/mytmp/build_libcust_opapi_nodebug/autogen/turboquant_attention_paged4bit_proto.h" \
        "${opp_root}/op_proto/inc/turboquant_attention_paged4bit_proto.h"

    rm -rf /tmp/xrx_tq4bit_runtime_config_root
    mkdir -p /tmp/xrx_tq4bit_runtime_config_root/${SOC_DIR}/bin
    cp -a "${opp_root}/op_impl/ai_core/tbe/kernel/${SOC_DIR}/." \
        "/tmp/xrx_tq4bit_runtime_config_root/${SOC_DIR}/bin/"
    python "${ASCEND_HOME_PATH}/tools/op_project_templates/ascendc/customize/cmake/util/ascendc_ops_config.py" \
        -p "/tmp/xrx_tq4bit_runtime_config_root/${SOC_DIR}/bin" \
        -s "${SOC_DIR}"
    cp -a "/tmp/xrx_tq4bit_runtime_config_root/${SOC_DIR}/bin/binary_info_config.json" \
        "${opp_root}/op_impl/ai_core/tbe/kernel/config/${SOC_DIR}/binary_info_config.json"
    cp -a "/tmp/xrx_tq4bit_runtime_config_root/${SOC_DIR}/bin/turboquant_pack_kv_for_cache4bit.json" \
        "${opp_root}/op_impl/ai_core/tbe/kernel/config/${SOC_DIR}/turboquant_pack_kv_for_cache4bit.json"
    cp -a "/tmp/xrx_tq4bit_runtime_config_root/${SOC_DIR}/bin/turboquant_attention_paged4bit.json" \
        "${opp_root}/op_impl/ai_core/tbe/kernel/config/${SOC_DIR}/turboquant_attention_paged4bit.json"
}

build_4bit_debug_line_kernels() {
    echo "[profile_tq4bit] rebuilding 4bit custom ops with per-op debug-line switches"
    (
        cd "${ROOT_DIR}"
        source xrx_infoenvs
        export TURBOQUANT_PACK_KV_FOR_CACHE4BIT_DEBUG_LINE=ON
        export TURBOQUANT_ATTENTION_PAGED4BIT_DEBUG_LINE=ON
        rm -f \
            "mytmp/build_libcust_opapi_nodebug/autogen/aclnn_turboquant_pack_kv_for_cache4bit.cpp" \
            "mytmp/build_libcust_opapi_nodebug/autogen/aclnn_turboquant_pack_kv_for_cache4bit.h" \
            "mytmp/build_libcust_opapi_nodebug/autogen/aclnn_turboquant_attention_paged4bit.cpp" \
            "mytmp/build_libcust_opapi_nodebug/autogen/aclnn_turboquant_attention_paged4bit.h" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/src/turboquant_pack_kv_for_cache4bit/TurboquantPackKvForCache4bit.py" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/src/turboquant_attention_paged4bit/TurboquantAttentionPaged4bit.py" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/gen/TurboquantPackKvForCache4bit-turboquant_pack_kv_for_cache4bit-0.sh" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/gen/TurboquantPackKvForCache4bit-turboquant_pack_kv_for_cache4bit-1.sh" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/gen/TurboquantAttentionPaged4bit-turboquant_attention_paged4bit-0.sh" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/gen/TurboquantAttentionPaged4bit-turboquant_attention_paged4bit-1.sh"
        rm -rf \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/src/turboquant_pack_kv_for_cache4bit/__pycache__" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/src/turboquant_attention_paged4bit/__pycache__"
        cmake \
            -S csrc \
            -B mytmp/build_libcust_opapi_nodebug \
            -DBUILD_OPEN_PROJECT=ON \
            -DASCEND_COMPUTE_UNIT="${SOC_DIR}" \
            -DASCEND_OP_NAME=ALL \
            -DVENDOR_NAME=vllm-ascend \
            -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
            -DCHECK_COMPATIBLE=true \
            -DTURBOQUANT_PACK_KV_FOR_CACHE4BIT_DEBUG_LINE=ON \
            -DTURBOQUANT_ATTENTION_PAGED4BIT_DEBUG_LINE=ON \
            -DCMAKE_BUILD_TYPE=Release
        if ! grep -Eq '^Turboquant(PackKvForCache4bit|AttentionPaged4bit),,.*(^|;)-g(;|$)' \
            mytmp/build_libcust_opapi_nodebug/autogen/custom_compile_options.ini; then
            echo "[profile_tq4bit] CMake configure did not add -g to 4bit custom compile options" >&2
            grep -E '^Turboquant(PackKvForCache4bit|AttentionPaged4bit),' \
                mytmp/build_libcust_opapi_nodebug/autogen/custom_compile_options.ini >&2 || true
            exit 1
        fi
        cmake --build mytmp/build_libcust_opapi_nodebug \
            --target opbuild_gen_default \
            --target turboquant_pack_kv_for_cache4bit_${SOC_DIR}_py_copy \
            --target turboquant_attention_paged4bit_${SOC_DIR}_py_copy \
            -j "${BUILD_JOBS:-64}"
        if ! grep -Eq -- "'-g'" \
            mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/src/turboquant_pack_kv_for_cache4bit/TurboquantPackKvForCache4bit.py; then
            echo "[profile_tq4bit] regenerated pack4bit impl Python does not contain -g" >&2
            exit 1
        fi
        if ! grep -Eq -- "'-g'" \
            mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/src/turboquant_attention_paged4bit/TurboquantAttentionPaged4bit.py; then
            echo "[profile_tq4bit] regenerated attention4bit impl Python does not contain -g" >&2
            exit 1
        fi
        rm -f \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/src/turboquant_pack_kv_for_cache4bit/turboquant_pack_kv_for_cache4bit_${SOC_DIR}_src_copy.done" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/gen/turboquant_pack_kv_for_cache4bit_${SOC_DIR}_0.done" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/gen/turboquant_pack_kv_for_cache4bit_${SOC_DIR}_1.done" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/src/turboquant_attention_paged4bit/turboquant_attention_paged4bit_${SOC_DIR}_src_copy.done" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/gen/turboquant_attention_paged4bit_${SOC_DIR}_0.done" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/gen/turboquant_attention_paged4bit_${SOC_DIR}_1.done"
        rm -rf \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/bin/turboquant_pack_kv_for_cache4bit" \
            "mytmp/build_libcust_opapi_nodebug/binary/${SOC_DIR}/bin/turboquant_attention_paged4bit"
        cmake --build mytmp/build_libcust_opapi_nodebug \
            --target turboquant_pack_kv_for_cache4bit_${SOC_DIR} \
            --target turboquant_attention_paged4bit_${SOC_DIR} \
            -j "${BUILD_JOBS:-64}"
    )
    sync_4bit_runtime_artifacts
}

if [[ ! -x "${RUNNER}" ]]; then
    if [[ ${AUTO_BUILD_RUNNER} != "1" ]]; then
        echo "[profile_tq4bit] missing executable runner: ${RUNNER}" >&2
        echo "[profile_tq4bit] run: bash tools/build_flex_tq_4bit_perf.sh" >&2
        exit 1
    fi
    echo "[profile_tq4bit] building flex C++ runner: ${RUNNER}"
    bash "${ROOT_DIR}/tools/build_flex_tq_4bit_perf.sh" "${RUNNER}"
fi

opp_root=${ASCEND_CUSTOM_OPP_PATH%%:*}
kernel_roots=(
    "${opp_root}/op_impl/ai_core/tbe/kernel/${SOC_DIR}/turboquant_pack_kv_for_cache4bit"
    "${opp_root}/op_impl/ai_core/tbe/kernel/${SOC_DIR}/turboquant_attention_paged4bit"
)

if ! collect_kernel_objects; then
    echo "[profile_tq4bit] missing 4bit kernel objects under ${opp_root}" >&2
    if [[ ${AUTO_BUILD_DEBUG_LINE} == "1" ]]; then
        build_4bit_debug_line_kernels
        collect_kernel_objects
    else
        exit 1
    fi
fi

if [[ ${REQUIRE_DEBUG_LINE} == "1" ]]; then
    if ! command -v readelf >/dev/null 2>&1; then
        echo "[profile_tq4bit] readelf is required when REQUIRE_DEBUG_LINE=1" >&2
        exit 1
    fi
    missing_debug_line_obj=""
    if ! kernel_objects_have_debug_line; then
        echo "[profile_tq4bit] ${missing_debug_line_obj} has no .debug_line"
        if [[ ${AUTO_BUILD_DEBUG_LINE} == "1" ]]; then
            build_4bit_debug_line_kernels
            collect_kernel_objects
            missing_debug_line_obj=""
            if ! kernel_objects_have_debug_line; then
                echo "[profile_tq4bit] ${missing_debug_line_obj} still has no .debug_line after rebuild" >&2
                exit 1
            fi
        else
            echo "[profile_tq4bit] rebuild the custom op with TURBOQUANT_PACK_KV_FOR_CACHE4BIT_DEBUG_LINE=ON and TURBOQUANT_ATTENTION_PAGED4BIT_DEBUG_LINE=ON before profiling" >&2
            exit 1
        fi
    fi
fi

if [[ $# -gt 0 ]]; then
    runner_args=("$@")
else
    runner_args=(
        --device 0
        --q-lens 1
        --kv-lens 8
        --heads 2
        --kv-heads 2
        --block-size 16
        --pack-tokens 1
        --warmup 0
        --repeat 1
    )
fi

runner_script="${PERF_OUT}/run_flex_tq_4bit_perf_${RUN_ID}.sh"
{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'cd %q\n' "${ROOT_DIR}"
    printf 'export ASCEND_CUSTOM_OPP_PATH=%q\n' "${ASCEND_CUSTOM_OPP_PATH}"
    printf 'exec %q' "${RUNNER}"
    for arg in "${runner_args[@]}"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
} > "${runner_script}"
chmod +x "${runner_script}"

echo "[profile_tq4bit] simulator soc=${SIM_SOC}"
echo "[profile_tq4bit] runner=${RUNNER}"
echo "[profile_tq4bit] runner args: ${runner_args[*]}"
echo "[profile_tq4bit] output=${PERF_OUT}"
echo "[profile_tq4bit] kernel objects:"
for kernel_obj in "${kernel_objects[@]}"; do
    echo "  ${kernel_obj}"
done

set +e
set -x
msprof op simulator \
    --soc-version="${SIM_SOC}" \
    --launch-count="${LAUNCH_COUNT}" \
    --output="${PERF_OUT}" \
    --application="${runner_script}" \
    2>&1 | tee "${MSPROF_LOG}"
msprof_status=${PIPESTATUS[0]}
set +x
set -e

latest_opprof=""
for dir in "${PERF_OUT}"/OPPROF_*; do
    [[ -d "${dir}" ]] || continue
    if [[ -z ${latest_opprof} || ${dir} -nt ${latest_opprof} ]]; then
        latest_opprof=${dir}
    fi
done

if [[ ${msprof_status} -ne 0 ]]; then
    echo "[profile_tq4bit] msprof op simulator failed with ${msprof_status}" >&2
    echo "[profile_tq4bit] msprof log: ${MSPROF_LOG}" >&2
    exit "${msprof_status}"
fi

bad_pattern='Segmentation fault|Kernel missed debug_line|Code call stack is empty|Lack of code info'
if grep -Eq "${bad_pattern}" "${MSPROF_LOG}"; then
    echo "[profile_tq4bit] profiler log contains an error/debug-line warning" >&2
    grep -En "${bad_pattern}" "${MSPROF_LOG}" >&2 || true
    exit 2
fi

if [[ ${REQUIRE_DEBUG_LINE} == "1" ]] && ! grep -Eq 'Parse [0-9]+ addr2line relations' "${MSPROF_LOG}"; then
    echo "[profile_tq4bit] profiler log did not report addr2line parsing" >&2
    echo "[profile_tq4bit] msprof log: ${MSPROF_LOG}" >&2
    exit 3
fi

echo "[profile_tq4bit] msprof log: ${MSPROF_LOG}"
if [[ -n ${latest_opprof} ]]; then
    echo "[profile_tq4bit] op simulator output: ${latest_opprof}"
fi
