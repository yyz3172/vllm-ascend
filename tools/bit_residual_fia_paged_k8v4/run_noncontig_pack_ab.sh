#!/usr/bin/env bash
# A/B: noncontig pack smoke — fixed tree vs HEAD (buggy firstSlot+rowOff).
#
# Usage (inside NPU container, after venv+infoenvs):
#   bash tools/bit_residual_fia_paged_k8v4/run_noncontig_pack_ab.sh
#   bash tools/bit_residual_fia_paged_k8v4/run_noncontig_pack_ab.sh --with-fia
#   bash tools/bit_residual_fia_paged_k8v4/run_noncontig_pack_ab.sh --skip-build
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

WITH_FIA=0
SKIP_BUILD=0
for arg in "$@"; do
  case "${arg}" in
    --with-fia) WITH_FIA=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    *) echo "unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

KERNEL_FILES=(
  csrc/bit_residual_pack_k8v4/op_kernel/bit_residual_pack_k8v4_common.h
  csrc/bit_residual_pack_k8v4/op_kernel/bit_residual_pack_k8v4_aiv.h
  csrc/bit_residual_pack_k8v4/op_kernel/bit_residual_pack_k8v4_aic.h
)
TEST_PY=tests/e2e/singlecard/xrx_bit_residual_k8v4_noncontig_blocks.py
STASH_DIR="$(mktemp -d /tmp/noncontig_pack_ab.XXXXXX)"
REBUILD_SH=script/lcy/bit_residual_fia_paged_k8v4/rebuild_op.sh

cleanup() {
  # Always try to restore fixed sources if we stashed them.
  if [[ -f "${STASH_DIR}/.have_fixed" ]]; then
    for f in "${KERNEL_FILES[@]}"; do
      cp -f "${STASH_DIR}/$(basename "${f}")" "${ROOT_DIR}/${f}"
    done
  fi
  rm -rf "${STASH_DIR}"
}
trap cleanup EXIT

run_test() {
  local label="$1"
  local expect="$2" # pass|fail
  local cmd=(python "${TEST_PY}")
  if [[ "${WITH_FIA}" == "1" ]]; then
    cmd+=(--with-fia)
  fi
  echo ""
  echo "===== RUN ${label} (expect=${expect}) ====="
  set +e
  "${cmd[@]}"
  local rc=$?
  set -e
  if [[ "${expect}" == "pass" ]]; then
    if [[ "${rc}" -eq 0 ]]; then
      echo "[AB] ${label}: PASS (as expected)"
      return 0
    fi
    echo "[AB] ${label}: UNEXPECTED FAIL rc=${rc}"
    return 1
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "[AB] ${label}: FAIL (as expected for buggy build)"
    return 0
  fi
  echo "[AB] ${label}: UNEXPECTED PASS (bug not reproduced)"
  return 1
}

rebuild() {
  if [[ "${SKIP_BUILD}" == "1" ]]; then
    echo "[AB] skip rebuild"
    return 0
  fi
  bash "${REBUILD_SH}" --soc=ascend910b
}

# Save fixed working-tree kernels.
mkdir -p "${STASH_DIR}"
for f in "${KERNEL_FILES[@]}"; do
  cp -f "${ROOT_DIR}/${f}" "${STASH_DIR}/$(basename "${f}")"
done
touch "${STASH_DIR}/.have_fixed"

echo "===== A: FIXED (working tree) ====="
rebuild
run_test "FIXED" pass
FIXED_RC=$?

echo ""
echo "===== B: BUGGY (git HEAD kernels) ====="
for f in "${KERNEL_FILES[@]}"; do
  git show "HEAD:${f}" > "${ROOT_DIR}/${f}"
done
rebuild
run_test "BUGGY_HEAD" fail
BUGGY_RC=$?

echo ""
echo "===== RESTORE FIXED + rebuild ====="
for f in "${KERNEL_FILES[@]}"; do
  cp -f "${STASH_DIR}/$(basename "${f}")" "${ROOT_DIR}/${f}"
done
rebuild
run_test "FIXED_RESTORED" pass
REST_RC=$?

echo ""
echo "===== AB SUMMARY ====="
echo "  FIXED:          $([[ ${FIXED_RC} -eq 0 ]] && echo OK || echo BAD)"
echo "  BUGGY_HEAD:     $([[ ${BUGGY_RC} -eq 0 ]] && echo OK_reproduced || echo BAD_no_repro)"
echo "  FIXED_RESTORED: $([[ ${REST_RC} -eq 0 ]] && echo OK || echo BAD)"

if [[ ${FIXED_RC} -eq 0 && ${BUGGY_RC} -eq 0 && ${REST_RC} -eq 0 ]]; then
  echo "CONCLUSION: noncontig smoke distinguishes buggy firstSlot+rowOff vs ResolvePackChunk fix."
  exit 0
fi
echo "CONCLUSION: A/B incomplete — check logs above."
exit 1
