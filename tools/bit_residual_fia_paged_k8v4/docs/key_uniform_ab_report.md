# Key 均匀量化 A/B 报告

> 环境：`k8s_POD_ascend-device-agent-tool-0.18` + `/root/cyl/venv/vllmdev/.venv`  
> 对照基线：[`/root/cyl/perflog2/BASELINE_CURRENT.md`](/root/cyl/perflog2/BASELINE_CURRENT.md)（LSB-sign Key，P18）  
> 合入门槛：L6 decode **≥3%** 且 longquery FIA kernel 同向；golden / UT / smoke PASS。

## 结论

**选用方案 A（非对称均匀）作为默认**（`BR_KEY_UNIFORM_SCHEME=1`）。

| 对比 | 结论 |
|------|------|
| A vs 基线（Scheme C / LSB-sign） | L6 decode **−19.1%**，longquery FIA decode med **−21.4%**，远超 3% |
| B vs A | L6 / longquery 增量均在 **±1%** 内（噪声），不达「保留 B」的 ~1–2% 增量门槛 |
| 数值 | A 按行 min/max 重建，对近零非对称分布更稳；B 仅作宏开关 spike 保留 |
| Scheme C | 原 LSB-sign+q7 路径已以 `BR_KEY_UNIFORM_SCHEME=3` 保留；**默认与测试仍用 A** |

方案 B/C 仍可通过改宏对比；文档与 golden 默认跟 A。

---

## 方案与改动

| | 方案 A（默认） | 方案 B（spike） |
|--|--|--|
| Encode | `base=min`, `step=(max-min)/255`, `q∈[0,255]` | `s=max(\|y\|,ε)/127.5`, `q=round(y/s+127.5)` |
| Decode | `y=base+q·step` | `y=s·(q−127.5)` ≡ `meta0+q·meta1`（meta0=`−127.5·s`） |
| Meta 布局 | base+step（2×2B） | 双槽兼容，stride 不变 |
| Key scratch | **2×fp32**（去 C） | 同 A 保守账 |

主要文件：`EncodeKeyBatch`、`BrDecodeKeyTile`、`kFp32ScratchCount=2`、`golden_dequant.py`。

UB 预期：Key `maxSub` **23→29**（见 `ub_layout.md`）。

---

## 正确性

| 项 | 方案 A | 方案 B |
|----|--------|--------|
| UT `test_bit_residual_fia_dequant.py` | PASS | PASS |
| `key1_unaligned` QDQ | PASS | PASS |
| L6 Fair 跑通（数值路径） | PASS | PASS |

---

## 性能（Fair 口径）

### L6 msprof（kv=2000, warm-up=5, launch=20, device=0）

| 配置 | decode Task Duration | prefill | vs 基线 decode |
|------|---------------------|---------|----------------|
| 基线 P18 | **626.673 µs** | 664.353 µs | — |
| 方案 A | **506.850 µs** | 534.231 µs | **−19.1%** |
| 方案 B | **509.430 µs** | 529.911 µs | **−18.7%** |
| B vs A | +0.5% | −0.8% | — |

产物：

- A: `/root/cyl/perflog2/l6_schemeA_20260727_011255`
- B: `/root/cyl/perflog2/l6_schemeB_20260727_012727`

AIV 仍 bound（decode vec_ratio≈0.49）；相对基线主要来自去 bit-extract + `maxSub`↑。

### longquery serving（FIA kernel med）

| 配置 | decode Q=16 med | prefill Q=241 med | vs 基线 decode |
|------|-----------------|-------------------|----------------|
| 基线 | **607.9 µs** | 641.4 µs | — |
| 方案 A | **478.0 µs** | 519.2 µs | **−21.4%** |
| 方案 B | **474.7 µs** | 516.8 µs | **−21.9%** |
| B vs A | −0.7% | −0.5% | — |

产物：

- A: `/root/cyl/perflog2/BitResidualFiaPagedK8v4/20260727_011453`
- B: `/root/cyl/perflog2/BitResidualFiaPagedK8v4/20260727_012727_lq`
- compare：`/root/cyl/perflog2/_logs/<stamp>/summarize_compare.log`

---

## 择优规则应用

1. A、B 相对基线均 **≥3%** 且 longquery 同向 → 均达标。  
2. B 相对 A 增量 **&lt;~1–2%** → **保留 A**。  
3. 默认宏与 golden：`BR_KEY_UNIFORM_SCHEME=1` / `KEY_UNIFORM_SCHEME=1`。

---

## 文档

- [`br_dequant_decode_flow.md`](../../../csrc/bit_residual_fia_paged_k8v4/docs/br_dequant_decode_flow.md)：Key 公式改为均匀 affine  
- [`ub_layout.md`](../../../csrc/bit_residual_fia_paged_k8v4/docs/ub_layout.md)：Key scratch 2×fp32、`maxSub≈29`
