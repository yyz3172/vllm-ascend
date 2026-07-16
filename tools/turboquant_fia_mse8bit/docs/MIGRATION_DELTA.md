# ops-transformer → vllm-ascend：TurboQuant FIA 迁移差异（最新）

> 更新时间：2026-07-16  
> 源：ops-transformer FIA V4 TQ P0（`antiquantMode=6`）  
> 目标：vllm-ascend `TurboquantFiaMse8bit`（`csrc/turboquant_fia_mse8bit/`）  
> 相关文档：[设计文档](docs/turboquant_fia_mse8bit_design.md) · [开发计划](DEVELOPMENT_PLAN.md)

本文只记 **与源仓仍不同或已对齐** 的点，不重复算法公式。

---

## 0. 一句话结论

| 层 | 状态 |
|----|------|
| **核内 TQ 数学**（Dequant / Π@Q / Softmax / Πᵀ / FD SyncAll） | **一致**（vendor + 少量编译 shim） |
| **接口 / KV 绑定 / Host tiling** | **有意适配**（standalone op），见下文 |
| **P0 验收** | `verify_pi_ortho` kv=512 / kv=1000 **PASS**（含 sparseMode=3） |

---

## 1. 已对齐（相对早期迁移偏差，现已修掉）

| 项 | 早期本仓 | 当前本仓 | ops-transformer |
|----|----------|----------|-----------------|
| `mBase`（TND） | 误写死 **256** | **512**（`kMBaseSize`） | TND `CalcMBaseSize` → 512 |
| `s2Base`（TND） | 512 | 512 | TND 默认 512 |
| FlashDecode | 曾串行 workaround | **SplitCore 真 FD**（key=1） | `numOfFdHead>0` → FD |
| atten_mask / sparse | tiling **强制关** | **已接通**（可选 mask + `sparse_mode` 0..4） | 完整 FIA mask 路径 |

当前 tiling 日志示例（kv=1000）：

```text
mask=1 sparse=3 pre=2147483647 next=0 fd=1 fdHeads=10 key=1
```

（`next=0` = right-down causal 的 FIA `GetPreNextToken` 改写，行为对齐。）

---

## 2. 有意适配（standalone，非 bug）

### 2.1 Π / γ 命名与 mode

| 数学量 | ops-transformer | vllm-ascend |
|--------|-----------------|-------------|
| Π `[D,D]` | `antiquantScale` | `rotation` |
| γ_k / γ_v | `keyAntiquantScale` / `valueAntiquantScale` | `key_scale` / `value_scale` |
| TQ 模式 | attr `antiquantMode/key/value=6` | 模板 `FIA_TQ_MSE_8BIT_MODE`，无 runtime mode attr |

核内仍走原 `Init(..., antiquantScale, ..., keyAntiquantScale, valueAntiquantScale, ...)` 形参；入口把新名字塞回旧槽位。**公式不变。**

### 2.2 KV：ListTensorDesc → raw GM

| | ops-transformer | vllm-ascend |
|--|-----------------|-------------|
| ACLNN | `aclTensorList` key/value | 单 tensor `key_cache` / `value_cache` |
| 核绑定 | `ListTensorDesc.GetDataPtr(...)` | `#else` of `TQ_FIA_KV_TENSOR_LIST`：整块 GM |
| 布局 | PA + `batchContinuous=1` | 同：`[BN,BS,KV*D]` + `block_table` |

等价于 FIA PA「list 只有一项、永远取 base」；**不做** `batchContinuous=0` 的 per-batch TensorList。

### 2.3 算子形态 / tiling key

| | ops-transformer | vllm-ascend |
|--|-----------------|-------------|
| 算子 | 挂在 `FusedInferAttentionScore` | 独立 `TurboquantFiaMse8bit` |
| ACLNN | `aclnnFusedInferAttentionScoreV4*` | `aclnnTurboquantFiaMse8bit*` |
| Tiling key | FIA 长编码（含 HP/BRCB/…） | **0** = 无 FD，**1** = FD |
| Host tiling | 完整 `FiaTilingNonQuant` | 精简 `turboquant_fia_mse8bit_tiling.cpp` + vendored `SplitCore` |

### 2.4 Attrs（本仓）

```text
num_heads, num_kv_heads, head_size, block_size, scale_value
pre_tokens, next_tokens, sparse_mode   # 已对齐 FIA 语义
```

相对 FIA 仍 **没有**：`inner_precise`、`input_layout`、`antiquant_mode*`、`softmax_lse_flag` 等大盘属性（P0 写死 TND+PA+fp16）。

---

## 3. 仍存在的差异 / 缺口

| 项 | ops-transformer | vllm-ascend | 风险 |
|----|-----------------|-------------|------|
| **mm1/mm2ResSize** | `min(g×s1, mBase) × …` | **`mBase × …`（满块上界）** | workspace **偏大**，一般更安全 |
| **normal/TQ WS 核数** | 常用 `aicNum` 开满 | 按 **`usedCoreNum`** | 与核 `InitWorkspace` 更贴；数值可小于 FIA |
| **FD tiling 字段** | 常写满 FD size | `!enableFd` 时 FD size **置 0** | key=0 不读则 OK |
| **BRCB** | 有独立 tiling key（`…, true`） | 入口写死 **`BRCB=false`** | TND+mBase=512 时 FIA 本身也不开 BRCB |
| **bf16** | 部分路径 | **仅 fp16** | 覆盖面 |
| **headDim** | 多档 / rope | **仅 D=128** | 覆盖面 |
| **layout** | BSH/BNSD/TND/… | **仅 TND+PA** | 覆盖面 |
| **4bit / prod / QJL / MLA** | 设计有路线 | **不做（P0）** | 范围 |
| **atten_mask dtype** | int8 / fp16 等 | OpDef **仅 int8** | 够用 TQ 用例；fp16 mask 未开 |
| **inner_precise → isRowInvalid** | `innerPrecise>>1` | **固定 0**（无该 attr） | 行无效细节未暴露 |
| **Compile shim** | — | `fia_kernel_common` 改名、`RowMuls`/`Max`/`Min` 等 | 非算法 |

---

## 4. 文件映射（速查）

| 角色 | ops-transformer | vllm-ascend |
|------|-----------------|-------------|
| Kernel | `attention/common/op_kernel/arch32/fia_kernel_turboquant_p0.h` 等 | `csrc/turboquant_fia_mse8bit/op_kernel/vendored/arch32/` |
| Entry | `fused_infer_attention_score_v3.cpp` | `op_kernel/turboquant_fia_mse8bit.cpp` |
| Host tiling | `fia_tiling_nonquant.*` | `op_host/turboquant_fia_mse8bit_tiling.cpp` |
| SplitCore | `attention/common/op_host/split_core.*` | `op_host/vendored/split_core.*` |
| 测试 / golden | `examples/*turboquant*` | `tools/turboquant_fia_mse8bit/` |

旧 baseline `csrc/turboquant_fused_infer_attention_score8bit/` **故意不动**（packed/codebook），与本 op 无关。

---

## 5. 验收基线（当前）

| 用例 | 期望 |
|------|------|
| `verify_pi_ortho.sh --kv=512` | PASS（QT+O ≤ 5e-5） |
| `verify_pi_ortho.sh --kv=1000` | PASS + FD（key=1） |
| 测试默认 | `sparseMode=3` + 2048×2048 全 0 mask（同 FIA TQ example） |
| Profile | `prof_tnd_pa_turboquant.sh msprof --kv=1000` |

---

## 6. 后续若要「更贴 FIA」

优先级建议（非必须）：

1. `mm*ResSize` 改为 `min(g×s1, mBase) × …`（减 workspace）
2. 可选：`inner_precise` → `isRowInvalid`
3. 可选：atten_mask fp16 dtype 组合
4. Phase 2：torch binding（见 DEVELOPMENT_PLAN）

**不必**再为 TND 改 `mBase`（已 512）；**不必**为对齐而恢复 ListTensorDesc（raw GM 是产品形态）。
