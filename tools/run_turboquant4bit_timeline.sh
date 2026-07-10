#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -d "${ASCEND_HOME_PATH}" && -d /usr/local/Ascend/cann-8.5.1 ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
fi
export ASCEND_HOME_PATH

if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
    # shellcheck source=/dev/null
    source "${ASCEND_HOME_PATH}/bin/setenv.bash" >/dev/null 2>&1 || true
fi
if [[ -f "${ROOT_DIR}/xrx_infoenvs" ]]; then
    # shellcheck source=/dev/null
    source "${ROOT_DIR}/xrx_infoenvs"
fi
if [[ -f "${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash" ]]; then
    export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    # shellcheck source=/dev/null
    source "${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash"
fi

TMP_ROOT="${ROOT_DIR}/ztmp/turboquant4bit_timeline"
mkdir -p "${TMP_ROOT}/tmp"
export TMPDIR="${TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

exec python3 "${ROOT_DIR}/tools/profile_turboquant4bit_timeline.py" "$@"
