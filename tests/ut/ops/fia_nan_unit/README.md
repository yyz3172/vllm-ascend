# FIA NaN / 内核 verify

检测 `bit_residual_fia_paged_k8v4` 在敏感形状下是否产出 **NaN/Inf**，并做 rem 矩阵 / 长 KV 回归（与 serve c16 TopK remain 同源）。

需 **NPU + 已 rebuild 自定义算子**；不适合纯 CPU CI。

## 文件

| 文件 | 用途 |
|------|------|
| `test_fia_nan_unit.py` | unittest：短 rem 矩阵 + serve-like 短 KV |
| `verify_rem_matrix.py` | rem 冒烟（原 `dual_aiv_smoke.sh`） |
| `verify_cum_matrix.py` | 长 KV 抽测 + FD smoke（原 `verify_cum_matrix.py`） |
| `verify_full_fia.py` | 完整矩阵 + fp16 多 batch + smoke（原 `verify_full_fia.py`） |
| `common.py` | 共享 setup / `run_once` / `run_case` |

## 覆盖（rem / unittest）

| 维度 | 取值 |
|------|------|
| batch | 1, 16 |
| kv | 16…128（含 rem0/8/16 与 serve 短 KV 20/24/40） |
| trials | 矩阵 3 次；serve-like 短 KV 各 5 次 |
| dtype | bf16，H=16，KV heads=8 |

失败条件：任一 trial 的 FIA 输出非有限，或 maxdiff 超阈值（rem 冒烟默认 `>1.0`）。

## 运行

unittest：

```bash
cd /vllm-workspace/vllm-ascend   # 或本仓库根目录
bash tests/ut/ops/fia_nan_unit/run.sh
# 或
ASCEND_RT_VISIBLE_DEVICES=5 \
python tests/ut/ops/fia_nan_unit/test_fia_nan_unit.py -v
```

内核 verify（纯 Python，无 sh）：

```bash
cd /vllm-workspace/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=5 \
python tests/ut/ops/fia_nan_unit/verify_rem_matrix.py
python tests/ut/ops/fia_nan_unit/verify_cum_matrix.py
python tests/ut/ops/fia_nan_unit/verify_full_fia.py
```

环境变量：

- `FIA_NAN_UNIT_BASE`：产物父目录（默认=`/root/yyz/fia_nan_unit`）
- `FIA_NAN_UNIT_OUT`：单次运行目录（默认=`$BASE/YYYYMMDD_HHMMSS[_suite]`）
- `ASCEND_RT_VISIBLE_DEVICES`（常用 `5`）
- `ASCEND_CUSTOM_OPP_PATH`（默认仓库内 `_cann_ops_custom`）
- `VLLM_ASCEND_BIT_RESIDUAL_FIA_FORCE`（common 默认置 `1`）

产物（**不写入 git 树**）：

```text
/root/yyz/fia_nan_unit/YYYYMMDD_HHMMSS_*/
  *_results.jsonl
  *_summary.json
```
