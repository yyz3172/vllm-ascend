#!/usr/bin/env bash
# Verify TurboQuant Π path (Q@Π^T + acc@Π) against QT+O golden.
#
# Baseline for csrc/turboquant_fia_mse8bit (NOT the old fused_infer...8bit baseline).
#
# Usage:
#   bash tools/turboquant_fia_mse8bit/verify_pi_ortho.sh
#   bash tools/turboquant_fia_mse8bit/verify_pi_ortho.sh --skip-build
#   bash tools/turboquant_fia_mse8bit/verify_pi_ortho.sh --kv=1000
#   bash tools/turboquant_fia_mse8bit/verify_pi_ortho.sh --regen
set -euo pipefail

EXAMPLES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF_SH="${EXAMPLES}/prof_tnd_pa_turboquant.sh"
GOLDEN_PY="${EXAMPLES}/golden_tnd_pa_turboquant.py"
BIN_DIR="${EXAMPLES}/golden_ortho_random"
VERIFY_DIR="${EXAMPLES}/verify_pi_ortho"
KERNEL_OUT="${VERIFY_DIR}/kernel_out.bin"
RUN_LOG="${VERIFY_DIR}/run.log"

SKIP_BUILD=0
REGEN_GOLDEN=0
KV_SEQ_LEN=512
PI_MODE=rotation
INPUT_MODE=random
SEED=42
MAX_PASS=5e-5

for arg in "$@"; do
    case "${arg}" in
        --skip-build) SKIP_BUILD=1 ;;
        --regen) REGEN_GOLDEN=1 ;;
        --kv=*) KV_SEQ_LEN="${arg#*=}" ;;
        --pi=*) PI_MODE="${arg#*=}" ;;
        --seed=*) SEED="${arg#*=}" ;;
        -h|--help)
            sed -n '2,14p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: ${arg}"
            exit 1
            ;;
    esac
done

mkdir -p "${VERIFY_DIR}" "${BIN_DIR}"

need_bins() {
    local f
    for f in golden_query.bin golden_key.bin golden_value.bin golden_gamma.bin golden_pi.bin; do
        [[ -f "${BIN_DIR}/${f}" ]] || return 1
    done
    return 0
}

gen_golden() {
    echo "=== Generate golden (${PI_MODE}, ${INPUT_MODE}, seed=${SEED}) ==="
    python3 "${GOLDEN_PY}" \
        --pi "${PI_MODE}" --input "${INPUT_MODE}" --seed "${SEED}" --dump-qbin \
        --out-dir "${BIN_DIR}"
}

echo "=== verify_pi_ortho: Π=${PI_MODE} input=${INPUT_MODE} kv=${KV_SEQ_LEN} ==="
echo "=== op: TurboquantFiaMse8bit ==="

if [[ "${REGEN_GOLDEN}" -eq 1 ]] || ! need_bins; then
    gen_golden
else
    echo "Reuse bins in ${BIN_DIR} (pass --regen to refresh)"
fi

if [[ "${SKIP_BUILD}" -eq 1 ]]; then
    echo "=== Build: skip ==="
else
    echo "=== Build via prof_tnd_pa_turboquant.sh ==="
    bash "${PROF_SH}" build
fi

echo "=== Run via prof_tnd_pa_turboquant.sh ==="
export Q_PATH="${BIN_DIR}/golden_query.bin"
export KV_KEY_PATH="${BIN_DIR}/golden_key.bin"
export KV_VALUE_PATH="${BIN_DIR}/golden_value.bin"
export GAMMA_PATH="${BIN_DIR}/golden_gamma.bin"
export PI_PATH="${BIN_DIR}/golden_pi.bin"
export KV_SEQ_LEN
export GOLDEN_OUT_PATH="${KERNEL_OUT}"
unset WS_DUMP_PATH || true

bash "${PROF_SH}" run --skip-build --log="${RUN_LOG}"

if ! grep -q "Run success" "${RUN_LOG}"; then
    echo "ERROR: binary did not report Run success. See ${RUN_LOG}"
    grep -E 'ERROR|failed|507046|161002|361001' "${RUN_LOG}" | head -20 || true
    exit 1
fi
if [[ ! -f "${KERNEL_OUT}" ]]; then
    echo "ERROR: missing kernel dump ${KERNEL_OUT}"
    exit 1
fi

echo "=== Compare kernel vs QT+O / I+O (kv=${KV_SEQ_LEN}) ==="
python3 - <<PY
import numpy as np
from pathlib import Path

bin_dir = Path("${BIN_DIR}")
ker_path = Path("${KERNEL_OUT}")
kvlen = int("${KV_SEQ_LEN}")
max_pass = float("${MAX_PASS}")

q = np.fromfile(bin_dir / "golden_query.bin", dtype=np.float16).reshape(16, 16, 128).astype(np.float32)
k_i8 = np.fromfile(bin_dir / "golden_key.bin", dtype=np.int8).reshape(120, 128, 1024)
v_i8 = np.fromfile(bin_dir / "golden_value.bin", dtype=np.int8).reshape(120, 128, 1024)
gamma = np.fromfile(bin_dir / "golden_gamma.bin", dtype=np.float16).reshape(15360, 8).astype(np.float32)
pi = np.fromfile(bin_dir / "golden_pi.bin", dtype=np.float16).astype(np.float32).reshape(128, 128)

BATCH, H, KVH, D = 16, 16, 8, 128
CENTER, S, SCALE = 127.5, 0.0026, 1.0 / np.sqrt(D)
bt = np.zeros((BATCH, 16), np.int32)
bid = 0
for b in range(BATCH):
    for i in range(16):
        bt[b, i] = bid % 120
        bid += 1


def gather(b, cache):
    out = np.zeros((kvlen, KVH, D), np.float32)
    flat = cache.reshape(-1)
    gflat = gamma.reshape(-1)
    for s in range(kvlen):
        blk, pos = s // 128, s % 128
        phys = int(bt[b, blk])
        tok = phys * 128 + pos
        for n2 in range(KVH):
            off = tok * (KVH * D) + n2 * D
            go = phys * 128 * KVH + pos * KVH + n2
            u8 = flat[off : off + D].view(np.uint8).astype(np.float32)
            out[s, n2] = (u8 - CENTER) * S * float(gflat[go])
    return out


def make(qs: str, os_: str):
    out = np.zeros((BATCH, H, D), np.float32)
    for m in range(BATCH):
        kb, vb = gather(m, k_i8), gather(m, v_i8)
        for h in range(H):
            qv = q[m, h]
            qr = qv @ pi.T if qs == "QT" else qv
            sc = (kb[:, h // 2] @ qr) * SCALE
            sc = sc - sc.max()
            e = np.exp(sc)
            acc = (e / e.sum()) @ vb[:, h // 2]
            out[m, h] = acc @ pi if os_ == "O" else acc
    return out


def report(name, a, b):
    d = np.abs(a - b)
    cos = np.mean(
        [
            float(np.dot(a[m, h], b[m, h]) / (np.linalg.norm(a[m, h]) * np.linalg.norm(b[m, h]) + 1e-30))
            for m in range(BATCH)
            for h in range(H)
        ]
    )
    print(f"  vs {name:8s}  max={d.max():.6e}  mean={d.mean():.6e}  cos={cos:.8f}")
    return float(d.max()), float(cos)


ker = np.fromfile(ker_path, dtype=np.float16).astype(np.float32).reshape(BATCH, H, D)
print(f"kernel finite={bool(np.isfinite(ker).all())}  |o|mean={np.abs(ker).mean():.6f}")
if not np.isfinite(ker).all():
    print("FAIL: NaN/Inf in kernel output")
    raise SystemExit(2)

qt_o = make("QT", "O")
i_o = make("I", "O")
max_qt, cos_qt = report("QT+O", ker, qt_o)
max_io, cos_io = report("I+O", ker, i_o)

pi_is_i = bool(np.allclose(pi, np.eye(D), atol=1e-3))
ok_qt = max_qt <= max_pass
ok_sep = True if pi_is_i else (max_qt < max_io * 0.5 and cos_qt > cos_io)
print("")
print(f"Π is identity: {pi_is_i}")
print(f"threshold QT+O: max_abs <= {max_pass:g}  -> {'PASS' if ok_qt else 'FAIL'} (got {max_qt:.6e})")
if not pi_is_i:
    print(f"Q-rotation active (QT+O << I+O): {'PASS' if ok_sep else 'FAIL'}")

if ok_qt and ok_sep:
    print("VERDICT: PASS")
    raise SystemExit(0)
print("VERDICT: FAIL")
raise SystemExit(1)
PY
