#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}
SIM_SOC=${SIM_SOC:-Ascend910B3}
SOC_DIR=${SOC_DIR:-ascend910b}
PERF_OUT=${PERF_OUT:-${ROOT_DIR}/perf_out_pytest_tq_decode}
RUN_ID=${RUN_ID:-$(date +%Y%m%d%H%M%S)}
MSPROF_LOG=${MSPROF_LOG:-${PERF_OUT}/msprof_op_simulator_pytest_${RUN_ID}.log}
PYTHON_BIN=${PYTHON_BIN:-python3}
PYTEST_NODEID=${PYTEST_NODEID:-tests/ut/ops/test_turboquant_decode_paged_8bit_op.py::test_decode_paged_op_matches_pytorch_golden[0]}
LAUNCH_COUNT=${LAUNCH_COUNT:-1}
REQUIRE_DEBUG_LINE=${REQUIRE_DEBUG_LINE:-1}
PYTEST_SIMULATOR_BACKEND=${PYTEST_SIMULATOR_BACKEND:-acl_runner}
CPP_RUNNER=${CPP_RUNNER:-${ROOT_DIR}/test_turboquant_decode_paged8bit}
AUTO_BUILD_CPP_RUNNER=${AUTO_BUILD_CPP_RUNNER:-1}

mkdir -p "${PERF_OUT}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend}
if [[ -f "${ASCEND_CUSTOM_OPP_PATH}/bin/set_env.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_CUSTOM_OPP_PATH}/bin/set_env.bash"
fi

# Let msprof op simulator --soc-version own the simulator runtime setup.
unset CAMODEL_LOG_PATH
unset CAMODEL_CONFIG_PATH
if [[ ${LD_PRELOAD:-} == *"/tools/simulator/"* ]]; then
    echo "[profile_tq_decode] unsetting simulator LD_PRELOAD from caller environment"
    unset LD_PRELOAD
fi

opp_root=${ASCEND_CUSTOM_OPP_PATH%%:*}
kernel_dir="${opp_root}/op_impl/ai_core/tbe/kernel/${SOC_DIR}/turboquant_decode_paged8bit"
kernel_obj=$(find "${kernel_dir}" -maxdepth 1 -type f -name 'TurboquantDecodePaged8bit_*.o' | sort | tail -n 1)
if [[ -z ${kernel_obj} ]]; then
    echo "[profile_tq_decode] missing kernel object under ${kernel_dir}" >&2
    echo "[profile_tq_decode] run: KENEL_DEBUG_LINE=ON bash build_turboquant_decode_paged8bit_fast.sh" >&2
    exit 1
fi

if [[ ${REQUIRE_DEBUG_LINE} == "1" ]]; then
    debug_sections=""
    if command -v readelf >/dev/null 2>&1; then
        debug_sections=$(readelf -S "${kernel_obj}" 2>/dev/null || true)
    fi
    if ! grep -q '\.debug_line' <<<"${debug_sections}"; then
        echo "[profile_tq_decode] ${kernel_obj} has no .debug_line" >&2
        echo "[profile_tq_decode] run: KENEL_DEBUG_LINE=ON bash build_turboquant_decode_paged8bit_fast.sh" >&2
        exit 1
    fi
fi

pytest_plugin_args=()
if [[ ${PYTEST_SIMULATOR_BACKEND} == "acl_runner" ]]; then
    if [[ ! -x "${CPP_RUNNER}" ]]; then
        if [[ ${AUTO_BUILD_CPP_RUNNER} != "1" ]]; then
            echo "[profile_tq_decode] missing executable runner: ${CPP_RUNNER}" >&2
            echo "[profile_tq_decode] run: bash cpp_test.sh" >&2
            exit 1
        fi
        echo "[profile_tq_decode] building C++ ACL runner: ${CPP_RUNNER}"
        bash "${ROOT_DIR}/cpp_test.sh"
    fi

    plugin_dir="${PERF_OUT}/pytest_plugin_${RUN_ID}"
    mkdir -p "${plugin_dir}"
    cat > "${plugin_dir}/tq_decode_profile_patch.py" <<'PYEOF'
import os
import subprocess


def pytest_collection_modifyitems(config, items):
    target = os.environ["TQ_DECODE_PYTEST_NODEID"]
    runner = os.environ["TQ_DECODE_ACL_RUNNER"]

    def _run_acl_runner(mode):
        env = os.environ.copy()
        env.setdefault("TQ_DECODE_BLOCK_SIZE", "128")
        env.setdefault("TQ_DECODE_NUM_KV_HEADS", "2")
        env.setdefault("TQ_DECODE_NUM_BLOCKS", "6")
        env.setdefault("TQ_DECODE_GATHER_BLOCKS", "1")
        env.setdefault("TQ_DECODE_MODE", "0")
        subprocess.run([runner], env=env, check=True)

    for item in items:
        if item.nodeid == target:
            item.obj = _run_acl_runner
PYEOF
    export PYTHONPATH="${plugin_dir}:${PYTHONPATH}"
    export TQ_DECODE_PYTEST_NODEID="${PYTEST_NODEID}"
    export TQ_DECODE_ACL_RUNNER="${CPP_RUNNER}"
    pytest_plugin_args=(-p tq_decode_profile_patch)
elif [[ ${PYTEST_SIMULATOR_BACKEND} != "python" ]]; then
    echo "PYTEST_SIMULATOR_BACKEND must be acl_runner or python, got: ${PYTEST_SIMULATOR_BACKEND}" >&2
    exit 1
fi

runner="${PERF_OUT}/run_pytest_${RUN_ID}.sh"
if [[ $# -gt 0 ]]; then
    pytest_cmd=("$@")
else
    pytest_cmd=("${PYTHON_BIN}" -m pytest -v -s "${pytest_plugin_args[@]}" "${PYTEST_NODEID}")
fi

{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'cd %q\n' "${ROOT_DIR}"
    printf 'exec'
    for arg in "${pytest_cmd[@]}"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
} > "${runner}"
chmod +x "${runner}"

echo "[profile_tq_decode] simulator soc=${SIM_SOC}"
echo "[profile_tq_decode] pytest backend=${PYTEST_SIMULATOR_BACKEND}"
echo "[profile_tq_decode] pytest command: ${pytest_cmd[*]}"
echo "[profile_tq_decode] kernel object: ${kernel_obj}"
echo "[profile_tq_decode] output=${PERF_OUT}"

set +e
set -x
msprof op simulator \
   --soc-version="${SIM_SOC}" \
   --launch-count="${LAUNCH_COUNT}" \
    --output="${PERF_OUT}" \
    --application="${runner}" \
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
    echo "[profile_tq_decode] msprof op simulator failed with ${msprof_status}" >&2
    echo "[profile_tq_decode] msprof log: ${MSPROF_LOG}" >&2
    exit "${msprof_status}"
fi

bad_pattern='Segmentation fault|Kernel missed debug_line|Code call stack is empty|Lack of code info'
if grep -Eq "${bad_pattern}" "${MSPROF_LOG}"; then
    echo "[profile_tq_decode] profiler log contains an error/debug-line warning" >&2
    grep -En "${bad_pattern}" "${MSPROF_LOG}" >&2 || true
    exit 2
fi

if [[ ${REQUIRE_DEBUG_LINE} == "1" ]] && ! grep -Eq 'Parse [0-9]+ addr2line relations' "${MSPROF_LOG}"; then
    echo "[profile_tq_decode] profiler log did not report addr2line parsing" >&2
    echo "[profile_tq_decode] msprof log: ${MSPROF_LOG}" >&2
    exit 3
fi

echo "[profile_tq_decode] msprof log: ${MSPROF_LOG}"
if [[ -n ${latest_opprof} ]]; then
    echo "[profile_tq_decode] op simulator output: ${latest_opprof}"
fi
