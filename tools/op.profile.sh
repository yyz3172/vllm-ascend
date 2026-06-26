#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)
echo "root dir is " $ROOT_DIR
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/mytmp"
export TMPDIR="${ROOT_DIR}/mytmp/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

APP=${TQ4BIT_OP_PROFILE_APP:-tools/run_attention_512.sh}
METRICS=${TQ4BIT_OP_PROFILE_METRICS:-Source,PipeUtilization}
#METRICS=${TQ4BIT_OP_PROFILE_METRICS:-Source,PipeUtilization,TimelineDetail}
LOG_FILE=${TQ4BIT_OP_PROFILE_LOG:-"${ROOT_DIR}/mytmp/log.op.timeline"}
CHECK_SOURCE=${TQ4BIT_OP_PROFILE_CHECK_SOURCE:-1}
MARKER="${TMPDIR}/op_profile_start.$$.marker"
touch "${MARKER}"

if [[ "${CHECK_SOURCE}" == "1" && ",${METRICS}," != *",Source,"* ]]; then
    echo "[op.profile] ERROR: TQ4BIT_OP_PROFILE_CHECK_SOURCE=1 requires Source in --aic-metrics" >&2
    echo "[op.profile] Current metrics: ${METRICS}" >&2
    exit 1
fi

msprof op \
    --output="${ROOT_DIR}/mytmp" \
    --application="${APP}" \
    --aic-metrics="${METRICS}" \
    --dump=on \
    --launch-count="${TQ4BIT_OP_PROFILE_LAUNCH_COUNT:-1}" \
    --warm-up="${TQ4BIT_OP_PROFILE_WARM_UP:-0}" \
    --kill=off \
    2>&1 | tee "${LOG_FILE}"

if [[ "${CHECK_SOURCE}" == "1" ]]; then
    opprof=""
    while IFS= read -r dir; do
        if [[ -z "${opprof}" || "${dir}" -nt "${opprof}" ]]; then
            opprof="${dir}"
        fi
    done < <(find "${ROOT_DIR}/mytmp" -maxdepth 1 -type d -name 'OPPROF_*' -newer "${MARKER}")

    if [[ -z "${opprof}" ]]; then
        echo "[op.profile] ERROR: no new OPPROF output found for debug-source validation" >&2
        exit 1
    fi

    if ! python "${ROOT_DIR}/tools/check_opprof_debug_source.py" "${opprof}" --require-insight-source; then
        echo "[op.profile] ERROR: ${opprof} does not carry MindStudio-visible kernel source data" >&2
        echo "[op.profile] Rebuild and sync debug-line kernels with: bash tools/build_debug_perf.sh" >&2
        echo "[op.profile] Host runner/libcust_opapi.so debug info is not enough; the runtime OPP kernel .o must have .debug_line." >&2
        exit 1
    fi
fi
