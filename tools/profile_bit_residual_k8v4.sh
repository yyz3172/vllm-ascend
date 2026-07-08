#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/mytmp"
export TMPDIR="${ROOT_DIR}/mytmp/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

APP=${BR_K8V4_PROFILE_APP:-"python tests/e2e/singlecard/xrx_bit_residual_k8v4_key1_unaligned.py"}
METRICS=${BR_K8V4_PROFILE_METRICS:-Source,PipeUtilization}
LOG_FILE=${BR_K8V4_PROFILE_LOG:-"${ROOT_DIR}/mytmp/log.bit_residual_k8v4.op.profile"}
CHECK_SOURCE=${BR_K8V4_PROFILE_CHECK_SOURCE:-1}
MARKER="${TMPDIR}/bit_residual_k8v4_profile_start.$$.marker"
touch "${MARKER}"

if [[ "${CHECK_SOURCE}" == "1" && ",${METRICS}," != *",Source,"* ]]; then
    echo "[profile_k8v4] ERROR: BR_K8V4_PROFILE_CHECK_SOURCE=1 requires Source in --aic-metrics" >&2
    echo "[profile_k8v4] Current metrics: ${METRICS}" >&2
    exit 1
fi

echo "[profile_k8v4] app=${APP}"
echo "[profile_k8v4] metrics=${METRICS}"

msprof op \
    --output="${ROOT_DIR}/mytmp" \
    --application="${APP}" \
    --aic-metrics="${METRICS}" \
    --dump=on \
    --launch-count="${BR_K8V4_PROFILE_LAUNCH_COUNT:-1}" \
    --warm-up="${BR_K8V4_PROFILE_WARM_UP:-0}" \
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
        echo "[profile_k8v4] ERROR: no new OPPROF output found" >&2
        exit 1
    fi

    if ! python "${ROOT_DIR}/tools/check_opprof_debug_source.py" "${opprof}" --require-insight-source; then
        echo "[profile_k8v4] ERROR: ${opprof} has no MindStudio-visible kernel source data" >&2
        echo "[profile_k8v4] Rebuild and sync debug-line kernels with:" >&2
        echo "[profile_k8v4]   bash tools/build_bit_residual_k8v4_debug_perf.sh" >&2
        exit 1
    fi
fi
