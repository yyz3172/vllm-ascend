#!/bin/bash
# FIA NaN 单元测试入口（容器内、需 NPU）
# 产物默认写入 /root/yyz/fia_nan_unit/YYYYMMDD_HHMMSS/（不写入仓库）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-5}"
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-${REPO_ROOT}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend}"
export VLLM_ASCEND_BIT_RESIDUAL_FIA_FORCE="${VLLM_ASCEND_BIT_RESIDUAL_FIA_FORCE:-1}"

BASE="${FIA_NAN_UNIT_BASE:-/root/yyz/fia_nan_unit}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="${FIA_NAN_UNIT_OUT:-${BASE}/${TS}}"
mkdir -p "$OUT"
export FIA_NAN_UNIT_OUT="$OUT"
LOG="$OUT/run.log"

echo "OUT=$OUT" | tee "$LOG"
echo "REPO_ROOT=$REPO_ROOT" | tee -a "$LOG"
echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES" | tee -a "$LOG"
python "${SCRIPT_DIR}/test_fia_nan_unit.py" -v 2>&1 | tee -a "$LOG"
echo "results: $OUT/fia_nan_unit_results.jsonl"
echo "summary: $OUT/fia_nan_unit_summary.json"
echo "log:     $OUT/run.log"
