#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

source xrx_infoenvs

RUNNER=${RUNNER:-"${ROOT_DIR}/ztmp/flex_tq_4bit_perf/flex_tq_4bit_perf"}
DEVICE=${DEVICE:-0}
SEQ_LEN=${SEQ_LEN:-512}
DECODE_SEQ_LEN=${DECODE_SEQ_LEN:-2048}
HEADS=${HEADS:-16}
KV_HEADS=${KV_HEADS:-8}
BLOCK_SIZE=${BLOCK_SIZE:-128}
WARMUP=${WARMUP:-1}
REPEAT=${REPEAT:-5}
SLOT_PATTERN=${SLOT_PATTERN:-contiguous}
SKIP_CACHE_FILL=${SKIP_CACHE_FILL:-1}

scenario=${1:-${SCENARIO:-prefill_pack}}
if [[ "${scenario}" == "-h" || "${scenario}" == "--help" || "${scenario}" == "help" ]]; then
    cat <<EOF
Usage: $0 [scenario]

Scenarios:
  prefill_pack              Pack one prefill/chunked-prefill request. Default.
  prefill_attention         Attention for one many-Q prefill request.
  single_decode_pack        Pack one decode request with one new token.
  single_decode_attention   Attention for one decode request.
  batch_decode_pack         Pack one decode token per request.
  batch_decode_attention    Attention for batched decode.
  mixed_pack                True mixed prefill+decode pack via --q-lens/--kv-lens.
  mixed_attention           True mixed prefill+decode attention via --q-lens/--kv-lens.
  mixed_both                True mixed pack+attention in one runner invocation.

Common env overrides:
  RUNNER=${RUNNER}
  SEQ_LEN=${SEQ_LEN}
  DECODE_SEQ_LEN=${DECODE_SEQ_LEN}
  HEADS=${HEADS}
  KV_HEADS=${KV_HEADS}
  BLOCK_SIZE=${BLOCK_SIZE}
  WARMUP=${WARMUP}
  REPEAT=${REPEAT}
  SKIP_CACHE_FILL=${SKIP_CACHE_FILL}
  Q_LENS, KV_LENS, BATCH_SIZE, PREFILL_TOKENS, DECODE_BATCH, PACK_MODE, SLOT_PATTERN
EOF
    exit 0
fi

if [[ ! -x "${RUNNER}" ]]; then
    echo "[run_attention_512] missing runner: ${RUNNER}" >&2
    echo "[run_attention_512] build it with: bash tools/build_flex_tq_4bit_perf.sh" >&2
    exit 1
fi

repeat_lens() {
    local value=$1
    local count=$2
    local out=""
    local i
    for ((i = 0; i < count; ++i)); do
        if [[ -n "${out}" ]]; then
            out+=":"
        fi
        out+="${value}"
    done
    printf '%s\n' "${out}"
}

mixed_q_lens() {
    local decode_batch=${DECODE_BATCH:-7}
    local prefill_tokens=${PREFILL_TOKENS:-128}
    local decode_lens
    decode_lens=$(repeat_lens 1 "${decode_batch}")
    if [[ "${decode_batch}" -eq 0 ]]; then
        printf '%s\n' "${prefill_tokens}"
    else
        printf '%s:%s\n' "${prefill_tokens}" "${decode_lens}"
    fi
}

mixed_kv_lens() {
    local decode_batch=${DECODE_BATCH:-7}
    local prefill_kv_len=${PREFILL_KV_LEN:-${SEQ_LEN}}
    local decode_kv_len=${DECODE_KV_LEN:-${DECODE_SEQ_LEN}}
    local decode_lens
    decode_lens=$(repeat_lens "${decode_kv_len}" "${decode_batch}")
    if [[ "${decode_batch}" -eq 0 ]]; then
        printf '%s\n' "${prefill_kv_len}"
    else
        printf '%s:%s\n' "${prefill_kv_len}" "${decode_lens}"
    fi
}

common_args() {
    local args=(
        --device "${DEVICE}" \
        --heads "${HEADS}" \
        --kv-heads "${KV_HEADS}" \
        --block-size "${BLOCK_SIZE}" \
        --slot-pattern "${SLOT_PATTERN}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}"
    )
    if [[ "${SKIP_CACHE_FILL}" == "1" ]]; then
        args+=(--skip-cache-fill)
    fi
    printf '%s\n' "${args[@]}"
}

run_flex() {
    exec "${RUNNER}" "$@"
}

prefill_pack() {
    local q_lens=${Q_LENS:-${SEQ_LEN}}
    local kv_lens=${KV_LENS:-${SEQ_LEN}}
    run_flex \
        --pack-only \
        --q-lens "${q_lens}" \
        --kv-lens "${kv_lens}" \
        --pack-mode "${PACK_MODE:-contiguous-group-fast}" \
        $(common_args)
}

prefill_attention() {
    local q_lens=${Q_LENS:-${SEQ_LEN}}
    local kv_lens=${KV_LENS:-${SEQ_LEN}}
    run_flex \
        --attention-only \
        --q-lens "${q_lens}" \
        --kv-lens "${kv_lens}" \
        $(common_args)
}

single_decode_pack() {
    run_flex \
        --pack-only \
        --q-lens "${Q_LENS:-1}" \
        --kv-lens "${KV_LENS:-${DECODE_SEQ_LEN}}" \
        --pack-mode "${PACK_MODE:-decode-vec-tasks}" \
        $(common_args)
}

single_decode_attention() {
    run_flex \
        --attention-only \
        --q-lens "${Q_LENS:-1}" \
        --kv-lens "${KV_LENS:-${DECODE_SEQ_LEN}}" \
        $(common_args)
}

batch_decode_pack() {
    local batch_size=${BATCH_SIZE:-32}
    local q_lens=${Q_LENS:-$(repeat_lens 1 "${batch_size}")}
    local kv_lens=${KV_LENS:-$(repeat_lens "${DECODE_SEQ_LEN}" "${batch_size}")}
    run_flex \
        --pack-only \
        --q-lens "${q_lens}" \
        --kv-lens "${kv_lens}" \
        --pack-mode "${PACK_MODE:-decode-vec-tasks}" \
        $(common_args)
}

batch_decode_attention() {
    local batch_size=${BATCH_SIZE:-32}
    local q_lens=${Q_LENS:-$(repeat_lens 1 "${batch_size}")}
    local kv_lens=${KV_LENS:-$(repeat_lens "${DECODE_SEQ_LEN}" "${batch_size}")}
    run_flex \
        --attention-only \
        --q-lens "${q_lens}" \
        --kv-lens "${kv_lens}" \
        $(common_args)
}

mixed_pack() {
    local q_lens=${Q_LENS:-$(mixed_q_lens)}
    local kv_lens=${KV_LENS:-$(mixed_kv_lens)}
    run_flex \
        --pack-only \
        --q-lens "${q_lens}" \
        --kv-lens "${kv_lens}" \
        --pack-mode "${PACK_MODE:-slot-mapping-group-owner}" \
        $(common_args)
}

mixed_attention() {
    local q_lens=${Q_LENS:-$(mixed_q_lens)}
    local kv_lens=${KV_LENS:-$(mixed_kv_lens)}
    run_flex \
        --attention-only \
        --q-lens "${q_lens}" \
        --kv-lens "${kv_lens}" \
        $(common_args)
}

mixed_both() {
    local q_lens=${Q_LENS:-$(mixed_q_lens)}
    local kv_lens=${KV_LENS:-$(mixed_kv_lens)}
    run_flex \
        --q-lens "${q_lens}" \
        --kv-lens "${kv_lens}" \
        --pack-mode "${PACK_MODE:-slot-mapping-group-owner}" \
        $(common_args)
}

case "${scenario}" in
    prefill_pack)
        prefill_pack
        ;;
    prefill_attention)
        prefill_attention
        ;;
    single_decode_pack)
        single_decode_pack
        ;;
    single_decode_attention)
        single_decode_attention
        ;;
    batch_decode_pack)
        batch_decode_pack
        ;;
    batch_decode_attention)
        batch_decode_attention
        ;;
    mixed_pack)
        mixed_pack
        ;;
    mixed_attention)
        mixed_attention
        ;;
    mixed_both)
        mixed_both
        ;;
    *)
        echo "[run_attention_512] unknown scenario: ${scenario}" >&2
        "$0" --help >&2
        exit 1
        ;;
esac
