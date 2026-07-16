#!/usr/bin/env bash
# Build / run / msprof for BitResidualFiaPagedK8v4 (same scale as TQ FIA L6).
#
# Modes:
#   build       Rebuild custom op + compile C++ test
#   run         Run test binary (requires prior build)
#   msprof      msprof op profiling (+ CSV summarize)
#   analyze     Summarize latest OPPROF dump
#   all         build + run
#
# Usage:
#   bash tools/bit_residual_fia_paged_k8v4/prof_tnd_pa_bit_residual.sh build
#   bash tools/bit_residual_fia_paged_k8v4/prof_tnd_pa_bit_residual.sh run --kv=1000
#   bash tools/bit_residual_fia_paged_k8v4/prof_tnd_pa_bit_residual.sh msprof --kv=1000 --skip-build
#   bash tools/bit_residual_fia_paged_k8v4/prof_tnd_pa_bit_residual.sh msprof --kv=1000 --source
set -euo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${TOOLS}/../.." && pwd)"
OUTPUT_BIN="${TOOLS}/output/test_aclnn_bit_residual_fia_paged_k8v4"
PROF_ROOT="${TOOLS}/prof_output/tnd_pa_bit_residual"
LOG_FILE="${LOG_FILE:-${TOOLS}/br_fia_run.log}"
CUSTOM_OPP_PATH="${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
KERNEL_NAME="BitResidualFiaPagedK8v4"

PROF_MODE="all"
SKIP_OP_BUILD=false
SKIP_EXAMPLE_BUILD=false
WANT_SOURCE=false
WANT_ANALYZE=true
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"
WARM_UP="${WARM_UP:-0}"
OP_AIC_METRICS="${OP_AIC_METRICS:-PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,ResourceConflictRatio}"
KV_SEQ_LEN="${KV_SEQ_LEN:-1000}"
ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-1}"
OP_DEBUG_CONFIG="${OP_DEBUG_CONFIG:-}"

usage() {
    sed -n '2,18p' "$0"
    exit 1
}

for arg in "$@"; do
    case "${arg}" in
        build|run|msprof|analyze|all) PROF_MODE="${arg}" ;;
        --skip-build) SKIP_OP_BUILD=true; SKIP_EXAMPLE_BUILD=true ;;
        --skip-op-build) SKIP_OP_BUILD=true ;;
        --skip-example-build) SKIP_EXAMPLE_BUILD=true ;;
        --log=*) LOG_FILE="${arg#*=}" ;;
        --output=*) PROF_ROOT="${arg#*=}" ;;
        --launch-count=*) LAUNCH_COUNT="${arg#*=}" ;;
        --warm-up=*) WARM_UP="${arg#*=}" ;;
        --kv=*) KV_SEQ_LEN="${arg#*=}" ;;
        --device=*) ASCEND_DEVICE_ID="${arg#*=}" ;;
        --source) WANT_SOURCE=true ;;
        --no-analyze) WANT_ANALYZE=false ;;
        --op-debug-config=*) OP_DEBUG_CONFIG="${arg#*=}" ;;
        -h|--help) usage ;;
        *)
            echo "Unknown argument: ${arg}"
            usage
            ;;
    esac
done

if [[ "${WANT_SOURCE}" == "true" ]]; then
    if [[ "${OP_AIC_METRICS}" != *"Source"* ]]; then
        OP_AIC_METRICS="${OP_AIC_METRICS},Source"
    fi
    if [[ -z "${OP_DEBUG_CONFIG}" ]]; then
        OP_DEBUG_CONFIG="ccec_g"
    fi
fi

PROF_ROOT="${PROF_ROOT}/kv${KV_SEQ_LEN}"

source_cann_env() {
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        for env_script in \
            "/usr/local/Ascend/cann/set_env.sh" \
            "/usr/local/Ascend/cann-8.5.1/set_env.sh" \
            "/usr/local/Ascend/cann-8.5.1/bin/setenv.bash" \
            "${HOME}/Ascend/cann/set_env.sh"; do
            if [[ -f "${env_script}" ]]; then
                # shellcheck disable=SC1090
                source "${env_script}"
                echo "CANN env sourced from: ${env_script}"
                break
            fi
        done
    fi
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        echo "Error: ASCEND_HOME_PATH not set."
        exit 1
    fi
    echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
}

setup_runtime_env() {
    export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
    export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_PATH}"
    export LD_LIBRARY_PATH="${CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH:-}"
    export ASCEND_DEVICE_ID
    export KV_SEQ_LEN
    if [[ -f "${CUSTOM_OPP_PATH}/bin/set_env.bash" ]]; then
        # shellcheck disable=SC1090
        source "${CUSTOM_OPP_PATH}/bin/set_env.bash"
    fi
    echo "KV_SEQ_LEN=${KV_SEQ_LEN} ASCEND_DEVICE_ID=${ASCEND_DEVICE_ID}"
    echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
}

write_run_wrapper() {
    local run_wrapper="$1"
    mkdir -p "$(dirname "${run_wrapper}")"
    cat > "${run_wrapper}" <<EOF
#!/bin/bash
export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="${CUSTOM_OPP_PATH}/op_api/lib:\${LD_LIBRARY_PATH}"
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID}"
export KV_SEQ_LEN="${KV_SEQ_LEN}"
EOF
    for v in Q_PATH PI_PATH GOLDEN_OUT_PATH WS_DUMP_PATH; do
        if [[ -n "${!v:-}" ]]; then
            printf 'export %s=%q\n' "${v}" "${!v}" >> "${run_wrapper}"
        fi
    done
    cat >> "${run_wrapper}" <<EOF
exec "${OUTPUT_BIN}" "\$@"
EOF
    chmod +x "${run_wrapper}"
}

do_build() {
    source_cann_env
    if [[ "${SKIP_OP_BUILD}" != "true" ]]; then
        local rebuild_args=(--soc=ascend910b)
        if [[ -n "${OP_DEBUG_CONFIG}" ]]; then
            rebuild_args+=(--op-debug-config="${OP_DEBUG_CONFIG}")
            echo "Op rebuild with OP_DEBUG_CONFIG=${OP_DEBUG_CONFIG}"
        fi
        bash "${ROOT_DIR}/script/lcy/bit_residual_fia_paged_k8v4/rebuild_op.sh" "${rebuild_args[@]}"
    else
        echo "Skip op rebuild."
    fi
    if [[ "${SKIP_EXAMPLE_BUILD}" != "true" ]]; then
        bash "${TOOLS}/build_test.sh"
    else
        echo "Skip example rebuild."
    fi
    echo "Build complete. Binary: ${OUTPUT_BIN}"
}

do_run() {
    source_cann_env
    setup_runtime_env
    if [[ ! -x "${OUTPUT_BIN}" ]]; then
        echo "Error: executable not found: ${OUTPUT_BIN}"
        echo "Hint: bash prof_tnd_pa_bit_residual.sh build"
        exit 1
    fi
    echo "Running ${OUTPUT_BIN} -> ${LOG_FILE}"
    set +e
    "${OUTPUT_BIN}" >"${LOG_FILE}" 2>&1
    local exit_code=$?
    set -e
    echo "Exit code: ${exit_code}"
    grep -E 'workspace|ERROR|failed|success|BitResidual|mode|golden|KV_SEQ|tiling|Output check' "${LOG_FILE}" | head -40 || true
    return ${exit_code}
}

latest_opprof_dump() {
    local out_dir="${PROF_ROOT}/op"
    local latest
    latest="$(ls -1dt "${out_dir}"/OPPROF_* 2>/dev/null | head -1 || true)"
    if [[ -z "${latest}" ]]; then
        echo ""
        return 0
    fi
    if [[ -d "${latest}/dump" ]]; then
        echo "${latest}/dump"
    else
        echo "${latest}"
    fi
}

summarize_msprof_csv() {
    local opprof_dir="$1"
    python3 - <<'PY' "${opprof_dir}"
import csv, os, sys

d = sys.argv[1]
print(f"\n=== Op metrics summary: {d} ===")

obi = os.path.join(d, "OpBasicInfo.csv")
if os.path.isfile(obi):
    with open(obi) as f:
        rows = list(csv.DictReader(f))
    if rows:
        r = rows[0]
        print(f"Op Name       : {r.get('Op Name','')}")
        print(f"Task Duration : {r.get('Task Duration(us)','')} us")
        print(f"Block Dim     : {r.get('Block Dim','')}  Mix Block Dim: {r.get('Mix Block Dim','')}")
        print(f"Freq          : {r.get('Current Freq','')} / rated {r.get('Rated Freq','')} MHz")

pu = os.path.join(d, "PipeUtilization.csv")
if os.path.isfile(pu):
    with open(pu) as f:
        rows = list(csv.DictReader(f))
    aic = [r for r in rows if r.get("sub_block_id","").startswith("cube")]
    aiv = [r for r in rows if r.get("sub_block_id","").startswith("vector")]
    def fnum(x):
        try: return float(x)
        except: return None
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals)/len(vals)) if vals else None
    def fmt(v, s=""):
        return f"{v:.3f}{s}" if v is not None else "NA"
    aic_t = avg([fnum(r.get("aic_time(us)")) for r in aic])
    aic_c = avg([fnum(r.get("aic_total_cycles")) for r in aic])
    aiv_t = avg([fnum(r.get("aiv_time(us)")) for r in aiv])
    aiv_c = avg([fnum(r.get("aiv_total_cycles")) for r in aiv])
    aiv_vec = avg([fnum(r.get("aiv_vec_ratio")) for r in aiv])
    aiv_mte2 = avg([fnum(r.get("aiv_mte2_ratio")) for r in aiv])
    aiv_mte3 = avg([fnum(r.get("aiv_mte3_ratio")) for r in aiv])
    aic_cube = avg([fnum(r.get("aic_cube_ratio")) for r in aic])
    aic_mte2 = avg([fnum(r.get("aic_mte2_ratio")) for r in aic])
    print(f"AIC cores     : {len(aic)}  avg time={fmt(aic_t,' us')}  avg cycles={fmt(aic_c,'')}")
    print(f"  cube_ratio  : {fmt(aic_cube)}  mte2_ratio={fmt(aic_mte2)}")
    print(f"AIV cores     : {len(aiv)}  avg time={fmt(aiv_t,' us')}  avg cycles={fmt(aiv_c,'')}")
    print(f"  vec_ratio   : {fmt(aiv_vec)}  mte2_ratio={fmt(aiv_mte2)}  mte3_ratio={fmt(aiv_mte3)}")
    if aic_t and aiv_t:
        print(f"Pipe bound    : {'AIV' if aiv_t > aic_t else 'AIC'} (max avg {max(aic_t,aiv_t):.3f} us)")

au = os.path.join(d, "ArithmeticUtilization.csv")
if os.path.isfile(au):
    with open(au) as f:
        rows = list(csv.DictReader(f))
    aic = [r for r in rows if r.get("sub_block_id","").startswith("cube")]
    aiv = [r for r in rows if r.get("sub_block_id","").startswith("vector")]
    def fnum(x):
        try: return float(x)
        except: return None
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals)/len(vals)) if vals else None
    cube_instr = avg([fnum(r.get("aic_cube_total_instr_number")) for r in aic])
    cube_fops = avg([fnum(r.get("aic_cube_fops")) for r in aic])
    vec_fops = avg([fnum(r.get("aiv_vec_fops")) for r in aiv])
    print(f"Cube instr#   : avg={cube_instr}")
    print(f"Cube FOPS     : avg={cube_fops}")
    print(f"Vec FOPS      : avg={vec_fops}")
PY
}

do_analyze() {
    local dump_dir="${1:-}"
    if [[ -z "${dump_dir}" ]]; then
        dump_dir="$(latest_opprof_dump)"
    fi
    if [[ -z "${dump_dir}" || ! -d "${dump_dir}" ]]; then
        echo "Error: no OPPROF dump found under ${PROF_ROOT}/op"
        exit 1
    fi
    local opprof_dir
    opprof_dir="$(dirname "${dump_dir}")"
    if [[ "$(basename "${dump_dir}")" != "dump" ]]; then
        opprof_dir="${dump_dir}"
        if [[ -d "${dump_dir}/dump" ]]; then
            dump_dir="${dump_dir}/dump"
        fi
    fi
    summarize_msprof_csv "${opprof_dir}"
    if [[ ! -f "${dump_dir}/fdata" ]]; then
        echo ""
        echo "No Source fdata in ${dump_dir} (use --source / ccec_g for per-line exec counts)."
    fi
}

do_msprof_op() {
    source_cann_env
    if ! command -v msprof >/dev/null 2>&1; then
        echo "Error: msprof not found."
        exit 1
    fi
    if [[ "${WANT_SOURCE}" == "true" && "${SKIP_OP_BUILD}" == "true" ]]; then
        echo "WARN: --source with --skip-build: fdata only if op was built with ccec_g."
    fi
    if [[ "${SKIP_OP_BUILD}" != "true" || "${SKIP_EXAMPLE_BUILD}" != "true" ]]; then
        do_build
    fi
    setup_runtime_env
    local out_dir="${PROF_ROOT}/op"
    local run_wrapper="${out_dir}/run_with_env.sh"
    mkdir -p "${out_dir}"
    write_run_wrapper "${run_wrapper}"
    cd "${out_dir}"
    echo "Output     : ${out_dir}"
    echo "KV_SEQ_LEN : ${KV_SEQ_LEN}"
    echo "Kernel     : ${KERNEL_NAME}"
    echo "aic-metrics: ${OP_AIC_METRICS}"
    echo "OP_DEBUG   : ${OP_DEBUG_CONFIG:-<none>}"
    msprof op \
        --output="${out_dir}" \
        --kernel-name="${KERNEL_NAME}" \
        --aic-metrics="${OP_AIC_METRICS}" \
        --launch-count="${LAUNCH_COUNT}" \
        --warm-up="${WARM_UP}" \
        --application="${run_wrapper}"
    echo "Profiling done: ${out_dir}/OPPROF_*"
    if [[ "${WANT_ANALYZE}" == "true" ]]; then
        do_analyze
    fi
}

echo "=== BitResidualFiaPagedK8v4 mode=${PROF_MODE} kv=${KV_SEQ_LEN} ==="
case "${PROF_MODE}" in
    build) do_build ;;
    run) do_run ;;
    msprof) do_msprof_op ;;
    analyze) do_analyze ;;
    all) do_build; do_run ;;
    *) usage ;;
esac
