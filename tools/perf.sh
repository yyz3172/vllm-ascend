#!/usr/bin/env bash
set -euo pipefail


ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}
SIM_SOC=${SIM_SOC:-Ascend910B3}
SIM_MODE=${SIM_MODE:-pv}
PERF_OUT=${PERF_OUT:-${ROOT_DIR}/mytmp/perf_out}
RUN_ID=${RUN_ID:-$(date +%Y%m%d%H%M%S)}
MSPROF_LOG=${MSPROF_LOG:-${PERF_OUT}/msprof_op_simulator_${RUN_ID}.log}

if [[ ${SIM_MODE} != "pv" ]]; then
    echo "Only SIM_MODE=pv is supported by this perf.sh wrapper" >&2
    exit 1
fi

mkdir -p "${PERF_OUT}"

# Keep direct cpp_test.sh validation at full size, but default simulator profiling to
# a smaller workload so the software simulator can finish reliably.
export TQ_DECODE_NUM_BLOCKS=${TQ_DECODE_NUM_BLOCKS:-1}
export TQ_DECODE_BLOCK_SIZE=${TQ_DECODE_BLOCK_SIZE:-1}
export TQ_DECODE_NUM_KV_HEADS=${TQ_DECODE_NUM_KV_HEADS:-1}

echo "[perf.sh] simulator soc=${SIM_SOC}"
echo "[perf.sh] profiler output=${PERF_OUT}"
echo "[perf.sh] workload: blocks=${TQ_DECODE_NUM_BLOCKS}, block_size=${TQ_DECODE_BLOCK_SIZE}, kv_heads=${TQ_DECODE_NUM_KV_HEADS}"

set +e
msprof op simulator \
    --soc-version="${SIM_SOC}" \
    --output="${PERF_OUT}" \
    --launch-count=1 \
    --application=${ROOT_DIR}/mytmp/test_turboquant_decode_paged8bit \
    2>&1 | tee "${MSPROF_LOG}"
msprof_status=${PIPESTATUS[0]}
set -e

latest_opprof=""
for dir in "${PERF_OUT}"/OPPROF_*; do
    [[ -d "${dir}" ]] || continue
    if [[ -z ${latest_opprof} || ${dir} -nt ${latest_opprof} ]]; then
        latest_opprof=${dir}
    fi
done

if [[ ${msprof_status} -ne 0 ]]; then
    echo "[perf.sh] msprof op simulator failed with ${msprof_status}" >&2
    echo "[perf.sh] msprof log: ${MSPROF_LOG}" >&2
    exit "${msprof_status}"
fi

echo "[perf.sh] msprof log: ${MSPROF_LOG}"
if [[ -n ${latest_opprof} ]]; then
    echo "[perf.sh] op simulator output: ${latest_opprof}"
fi
