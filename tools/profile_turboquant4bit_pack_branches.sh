#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

OUT_DIR=${OUT_DIR:-"${ROOT_DIR}/mytmp/pack4bit_branch_profiles"}
RUNNER=${RUNNER:-"${ROOT_DIR}/mytmp/flex_tq_4bit_perf/flex_tq_4bit_perf"}
DEBUG_BUILD_DIR=${DEBUG_BUILD_DIR:-"${ROOT_DIR}/mytmp/build_libcust_opapi_nodebug_debug_line"}
SOC_DIR=${SOC_DIR:-ascend910b}
PACK_TOKENS=${PACK_TOKENS:-512}
HEADS=${HEADS:-16}
KV_HEADS=${KV_HEADS:-8}
BLOCK_SIZE=${BLOCK_SIZE:-128}
SEQ_LEN=${SEQ_LEN:-512}
WARMUP=${WARMUP:-1}
REPEAT=${REPEAT:-5}
LAUNCH_COUNT=${LAUNCH_COUNT:-1}
SKIP_BUILD=${SKIP_BUILD:-0}

mkdir -p "${OUT_DIR}"

if [[ "${SKIP_BUILD}" == "1" ]]; then
    echo "[pack_branch_profile] skipping debug-line build because SKIP_BUILD=1"
else
    echo "[pack_branch_profile] building debug-line kernels and runner"
    bash "${ROOT_DIR}/tools/build_debug_perf.sh" 2>&1 | tee "${OUT_DIR}/build_debug_perf.log"
fi

if [[ ! -x "${RUNNER}" ]]; then
    echo "[pack_branch_profile] missing runner: ${RUNNER}" >&2
    exit 1
fi

debug_pack_objs=(
    "${DEBUG_BUILD_DIR}/binary/${SOC_DIR}/bin/turboquant_pack_kv_for_cache4bit"/TurboquantPackKvForCache4bit_*.o
)
if [[ ! -e "${debug_pack_objs[0]}" ]]; then
    echo "[pack_branch_profile] missing debug pack kernel objects under ${DEBUG_BUILD_DIR}" >&2
    exit 1
fi

for obj in "${debug_pack_objs[@]}"; do
    if ! readelf -SW "${obj}" | grep -q '\.debug_line'; then
        echo "[pack_branch_profile] ${obj} has no .debug_line" >&2
        exit 1
    fi
done

run_case() {
    local name=$1
    local pack_mode=$2
    local slot_pattern=$3
    local case_dir="${OUT_DIR}/${name}"
    local app="${case_dir}/run.sh"
    local log="${case_dir}/op.profile.log"
    local marker="${case_dir}/profile_start.marker"

    mkdir -p "${case_dir}"
    cat > "${app}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT_DIR}"
source xrx_infoenvs
export ASCEND_CUSTOM_OPP_PATH="${ROOT_DIR}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
exec "${RUNNER}" \\
    --pack-only \\
    --pack-mode "${pack_mode}" \\
    --slot-pattern "${slot_pattern}" \\
    --q-lens 1 \\
    --kv-lens "${SEQ_LEN}" \\
    --pack-tokens "${PACK_TOKENS}" \\
    --heads "${HEADS}" \\
    --kv-heads "${KV_HEADS}" \\
    --block-size "${BLOCK_SIZE}" \\
    --warmup "${WARMUP}" \\
    --repeat "${REPEAT}"
EOF
    chmod +x "${app}"

    echo "[pack_branch_profile] profiling ${name}: mode=${pack_mode} slot=${slot_pattern}"
    rm -f "${marker}"
    touch "${marker}"
    TQ4BIT_OP_PROFILE_APP="${app}" \
    TQ4BIT_OP_PROFILE_LOG="${log}" \
    TQ4BIT_OP_PROFILE_LAUNCH_COUNT="${LAUNCH_COUNT}" \
        bash "${ROOT_DIR}/tools/op.profile.sh"

    local opprof=""
    while IFS= read -r dir; do
        if [[ -z "${opprof}" || "${dir}" -nt "${opprof}" ]]; then
            opprof="${dir}"
        fi
    done < <(find "${ROOT_DIR}/mytmp" -maxdepth 1 -type d -name 'OPPROF_*' -newer "${marker}")

    if [[ -z "${opprof}" ]]; then
        while IFS= read -r dir; do
            if [[ -z "${opprof}" || "${dir}" -nt "${opprof}" ]]; then
                opprof="${dir}"
            fi
        done < <(find "${ROOT_DIR}/mytmp" -maxdepth 1 -type d -name 'OPPROF_*')
    fi

    if [[ -z "${opprof}" ]]; then
        echo "[pack_branch_profile] no OPPROF output found for ${name}" >&2
        exit 1
    fi

    local saved="${case_dir}/$(basename "${opprof}")"
    rm -rf "${saved}"
    mv "${opprof}" "${saved}"
    echo "${saved}" > "${case_dir}/opprof_path.txt"
    python "${ROOT_DIR}/tools/check_opprof_debug_source.py" "${saved}" \
        --require-insight-source \
        2>&1 | tee "${case_dir}/debug_source_check.log"

    local op_name
    op_name=$(sed -n 's/^Op Name=//p' "${saved}/dump/op_basic_info.txt" | head -1)
    local hash_name=${op_name%_0_mix_aic}
    local debug_elf="${DEBUG_BUILD_DIR}/binary/${SOC_DIR}/bin/turboquant_pack_kv_for_cache4bit/${hash_name}.o"
    if [[ ! -f "${debug_elf}" ]]; then
        echo "[pack_branch_profile] debug elf not found for ${op_name}: ${debug_elf}" >&2
        exit 1
    fi

    python "${ROOT_DIR}/tools/msopprof_hotspot_lines.py" "${saved}" \
        --debug-elf "${debug_elf}" \
        --top-lines 40 \
        --top-repo-lines 40 \
        --top-bb 40 \
        2>&1 | tee "${case_dir}/hotspot_lines.log"
}

run_case general_contiguous 0 contiguous
run_case direct_contiguous 1 contiguous
run_case contiguous_fast_contiguous 2 contiguous
run_case contiguous_fast_scatter_groups 2 scatter-groups

echo "[pack_branch_profile] results written to ${OUT_DIR}"
