# FIA NaN 单元测试

专门检测 `bit_residual_fia_paged_k8v4` 是否在已知敏感形状下产出 **NaN/Inf**（与 serve c16 TopK remain 同源）。

需 **NPU + 已 rebuild 自定义算子**；不适合纯 CPU CI。

## 覆盖

| 维度 | 取值 |
|------|------|
| batch | 1, 16 |
| kv | 16…128（含 rem0/8/16 与 serve 短 KV 20/24/40） |
| trials | 矩阵 3 次；serve-like 短 KV 各 5 次 |
| dtype | bf16，H=16，KV heads=8 |

失败条件：任一 trial 的 FIA 输出非有限（Paged 也应有限，否则提示环境/算子异常）。

## 运行

```bash
cd /vllm-workspace/vllm-ascend   # 或本仓库根目录
bash tests/ut/ops/fia_nan_unit/run.sh
```

环境变量：

- `FIA_NAN_UNIT_BASE`：产物父目录（默认=`/root/yyz/fia_nan_unit`）
- `FIA_NAN_UNIT_OUT`：单次运行目录（默认=`$BASE/YYYYMMDD_HHMMSS`）
- `ASCEND_RT_VISIBLE_DEVICES`（默认 `5`）
- `ASCEND_CUSTOM_OPP_PATH`（默认仓库内 `_cann_ops_custom`）

产物（**不写入 git 树**）：

```text
/root/yyz/fia_nan_unit/YYYYMMDD_HHMMSS/
  run.log
  fia_nan_unit_results.jsonl
  fia_nan_unit_summary.json
```

或直接调 Python：

```bash
cd /vllm-workspace/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=5 \
ASCEND_CUSTOM_OPP_PATH=$PWD/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend \
python tests/ut/ops/fia_nan_unit/test_fia_nan_unit.py -v
```
