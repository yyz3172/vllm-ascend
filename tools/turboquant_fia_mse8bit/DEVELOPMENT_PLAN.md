# TurboQuant FIA MSE-8bit 迁入 vllm-ascend 开发计划

> 基线：ops-transformer `verify_pi_ortho.sh`（QT+O golden）  
> 源实现：`fia_kernel_turboquant_p0.h` + cube/vec block  
> 目标仓：vllm-ascend `csrc/` 独立 custom op，先 C++ 验收，再 Python 集成

---

## 0. 背景与目标

### 0.1 要解决什么

将 ops-transformer 中已验证的 **FIA TurboQuant P0（MSE 8bit, TND+PA+GQA）** 迁到 vllm-ascend，作为可独立编译、调用、profile 的 custom op，便于：

1. 不依赖 CANN 内置 `FusedInferAttentionScoreV4` 是否带 TQ
2. 用同一套 golden / msprof 工具做正确性与性能回归
3. 后续挂到 `torch.ops._C_ascend` 供 attention 路径调用

### 0.2 明确不做（P0）

- 不以现有 scalar `turboquant_fused_infer_attention_score8bit` baseline 为正确性参考
- P0 不改 serving 热路径强制切换（可保留 decode+FIA fallback）
- P0 不先做 packed（idx‖norm）布局；先对齐 FIA「idx + 独立 γ + Π」
- 不做 4bit / prod / QJL / MLA

### 0.3 成功标准（P0 验收）

| 项 | 标准 |
|----|------|
| 正确性 | `verify_pi_ortho.sh`：Π=rotation、input=random、kv=512 → **VERDICT: PASS**（QT+O max_abs ≤ 5e-5，且明显优于 I+O） |
| FD | `--kv=1000` 能 Run success，且与 QT+O 在约定阈值内（可单独放宽） |
| 性能工具 | `prof_tnd_pa_turboquant.sh msprof` 能采到本 op；`analyze_source_exec.py` 可解析 Source dump |
| 工程 | `csrc` 单 op 可 `--rebuild-op` 编过；C++ 用例不依赖 Python/torch |

---

## 1. 接口约定（P0 冻结）

语义对齐 FIA TQ P0，命名对齐 vllm-ascend custom op：

```text
Inputs
  query            fp16  [T, H, D]                 # TND
  key_cache        int8  [BN, BS, KV_H * D]        # PA idx
  value_cache      int8  同上
  block_table      int32 [B, max_blocks]
  actual_seq_len_q int64 [B]   (ValueDepend / cumsum for TND)
  actual_seq_len_kv int64 [B]
  atten_mask       optional (int8/fp16)
  key_scale        fp16  [phys_tokens, KV_H]       # γ_k  (= FIA keyAntiquantScale)
  value_scale      fp16  同上                      # γ_v
  rotation         fp16  [D, D]                    # Π   (= FIA antiquantScale)

Output
  attention_out    fp16  [T, H, D]

Attrs
  num_heads, num_kv_heads, head_size, block_size, scale_value
```

算子名（**新建目录，不改原 baseline op**）：

| 项 | 名称 |
|----|------|
| 目录 | `csrc/turboquant_fia_mse8bit/` |
| GE/OpDef | `TurboquantFiaMse8bit` |
| aclnn | `aclnnTurboquantFiaMse8bit` |
| kernel entry | `turboquant_fia_mse8bit` |
| torch（Phase 2） | `torch.ops._C_ascend.turboquant_fia_mse8bit` |

原 `csrc/turboquant_fused_infer_attention_score8bit/` **保持不动**（旧 scalar/codebook baseline）。

**与旧 baseline 差异（刻意）：** 无 codebook；无 packed；γ/Π 显式 Tensor。

---

## 2. 总体架构

```text
ops-transformer TQ P0                    vllm-ascend
─────────────────────                    ──────────────────────────────
fia_block_*_turboquant_p0.h    ──vendor──►  csrc/turboquant_fia_mse8bit/op_kernel/vendored/
fia_tiling_nonquant (TQ 分支)  ──精简───►  csrc/turboquant_fia_mse8bit/op_host/*_tiling.cpp
aclnnFusedInferAttentionScoreV4 ──新建──►  aclnnTurboquantFiaMse8bit
test_aclnn_...turboquant.cpp   ──改写───►  tools/.../test_*.cpp
verify_pi_ortho / prof / golden ──迁路径─►  tools/turboquant_fia_mse8bit/

（并行保留）csrc/turboquant_fused_infer_attention_score8bit/  # 旧 baseline，本计划不碰
```

核内算法不变：

```text
y = formula(uint8(idx)) * γ          # 旋转域
S = (Π @ Q) @ y_k^T / √D
O = Π^T @ (P @ y_v)
```

仅适配层改动：

- KV：custom op 传 **raw GM**（非 FIA ListTensorDesc）
- Tiling：自包含结构，字段布局与核 `InitTilingData` 读取一致
- 入口：按 tiling key 实例化 `FLASH_DECODE=false/true`

---

## 3. 目录与产物

```text
csrc/turboquant_fia_mse8bit/                    # ★ 新建独立算子（勿改 fused_infer...8bit）
  op_host/
    turboquant_fia_mse8bit_def.cpp
    turboquant_fia_tiling_data.h
    turboquant_fia_mse8bit_tiling.{h,cpp}
    aclnn_turboquant_fia_mse8bit.h
    CMakeLists.txt
  op_kernel/
    vendored/arch32/          # TQ 核 + flashdecode
    vendored/*.h              # fia_public_define / vector_common / ...
    turboquant_fia_mse8bit.cpp
  turboquant_fia_mse8bit_torch_adpt.h
  MANIFEST.txt

csrc/turboquant_fused_infer_attention_score8bit/  # 原 baseline，本计划不修改

tools/turboquant_fia_mse8bit/
  verify_pi_ortho.sh
  prof_tnd_pa_turboquant.sh
  golden_tnd_pa_turboquant.py
  compare_golden.py
  analyze_source_exec.py
  DEVELOPMENT_PLAN.md
  test_aclnn_tq_fia_mse8bit.cpp      # Phase 1 新增/改写
  build_test.sh                      # Phase 1

script/lcy/tq_fia_mse8bit/
  rebuild_op.sh                      # 对齐 tq4bit；ASCEND_OP_NAME=turboquant_fia_mse8bit
```

---

## 4. 分阶段计划

### Phase 1 — C++ 打通（优先，本阶段主目标）

**目标：** custom op 可编译安装；C++ 用例跑通；`verify_pi_ortho` PASS。

| 序号 | 任务 | 说明 | 产出 |
|------|------|------|------|
| 1.1 | 冻结 IO / tiling key | key0=无 FD，key1=FD；D=128 only | 本文档 §1 |
| 1.2 | 补齐 vendored 可编译性 | include 路径、缺头、raw KV 补丁回归检查 | 可过 op kernel 编译 |
| 1.3 | Host tiling 对齐核 | workspace = libapi \| normal \| FD \| tqDequant；usedCoreNum=AIC | tiling 与 InitWorkspace 一致 |
| 1.4 | Kernel 入口 | `INVOKE` FiaKernelTurboQuantP0；空指针 optional 安全 | `.o` / opp 包 |
| 1.5 | C++ 用例 | 从 `test_aclnn_...turboquant.cpp` 改调本仓 aclnn；保留 env：Q/KV/GAMMA/PI/GOLDEN_OUT/KV_SEQ_LEN | `test_aclnn_tq_fia_mse8bit` |
| 1.6 | build / prof 脚本 | 参考 `script/lcy/tq4bit` + 原 `prof_tnd_pa_turboquant.sh` | `rebuild_op.sh` + `prof_*.sh` |
| 1.7 | 工具路径适配 | verify/prof 指向本仓 binary 与 CUSTOM_OPP | `verify_pi_ortho.sh` 可一键跑 |
| 1.8 | 正确性验收 | kv=512 rotation；必要时 `--regen` | VERDICT: PASS |
| 1.9 | FD 冒烟 | `--kv=1000` | Run success + 精度记录 |

**Phase 1 退出标准：** `bash tools/turboquant_fia_mse8bit/verify_pi_ortho.sh` 退出码 0。

---

### Phase 2 — Torch / Python 接线

| 序号 | 任务 | 说明 |
|------|------|------|
| 2.1 | **新增** `torch_binding.cpp` 注册（不改旧 8bit fused 绑定） | `turboquant_fia_mse8bit` |
| 2.2 | 使用 `turboquant_fia_mse8bit_torch_adpt.h` | 与 aclnn 一致 |
| 2.3 | Python wrapper | 新 API，例如 `turboquant_fia_mse8bit(...)` |
| 2.4 | UT | mock 或 NPU：shape/dtype/可选 mask |
| 2.5 | 文档注释 | 与旧 packed baseline 的差异、并存说明 |

**退出标准：** `torch.ops._C_ascend.turboquant_fia_mse8bit(...)` 与 C++ 同输入数值一致（同阈值）。

---

### Phase 3 — vllm attention 集成

| 序号 | 任务 | 说明 |
|------|------|------|
| 3.1 | KV 布局策略二选一 | A：pack 改为 idx+独立 γ Tensor；B：核支持 packed（改 Dequant 寻址） |
| 3.2 | `attention_v1` 开关 | 环境变量 / ascend_config 控制走新 op |
| 3.3 | fallback | 不满足 D=128 / 无 γ / 无 Π 时回退 decode+FIA |
| 3.4 | e2e smoke | 单卡短请求；对比非 TQ / 旧路径 |

**推荐默认：3.1-A**（少改核，与 verify 契约一致）。

---

### Phase 4 — 性能与清理

| 序号 | 任务 | 说明 |
|------|------|------|
| 4.1 | msprof + Source | `prof_*.sh msprof`；`analyze_source_exec.py` |
| 4.2 | 与 ops-transformer 同 shape 对比 | PipeUtilization / 时长 |
| 4.3 | 旧 scalar op 可继续保留；文档标明勿混用 | 两套算子并存直到 serving 切换完成 |
| 4.4 | tiling 硬化 | 从「verify 形状可用」扩到通用 prefill/混部；补齐 FD 负载均衡与 outer split |

---

## 5. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| FIA tiling 依赖面大，精简 tiling 漏字段 | 挂死 / 错结果 | 先只保证 verify 形状；对照 `InitTilingData` 字段清单逐项填 |
| Workspace 与 usedCoreNum 不一致 | FD SyncAll 异常 | 严格用 AIC `usedCoreNum`，禁止误用 AIV `GetBlockNum()` |
| ListTensorDesc → raw GM | MTE 越界 | 已用宏区分；仅 custom op 走 raw 路径 |
| Π 路径回归 | Pi=I 蒙混过关 | **必须以 verify_pi_ortho（非 I）为门禁** |
| 旧 Python 仍传 codebook/packed | 集成失败 | Phase 2 显式改签名；Phase 3 再切 attention |
| 编译 include / 算子体积 | 编不过 | CMake `-I vendored`；缺头按 MANIFEST 补 vendor |

---

## 6. 验证矩阵

| Case | 配置 | 期望 |
|------|------|------|
| V1 | Π=I, uniform, kv=512 | 非全 0；接近 I+O |
| V2 | Π=rotation, random, kv=512 | **verify_pi_ortho PASS**（主门禁） |
| V3 | Π=rotation, random, kv=1000 | FD 路径；精度记录 |
| V4 | msprof Source | analyze 产出 per_file/per_line |
| V5（Phase2） | 同 V2 输入经 torch | 与 C++ 输出一致 |

---

## 7. 里程碑与大致工作量

| 里程碑 | 内容 | 粗估 |
|--------|------|------|
| M1 | Phase 1.1–1.7：可编译 + C++ 可跑 | 3–5 人日 |
| M2 | Phase 1.8–1.9：verify PASS + FD 冒烟 | 2–4 人日（含编不过/精度迭代） |
| M3 | Phase 2 Torch 接线 | 1–2 人日 |
| M4 | Phase 3 attention 集成 | 3–5 人日（含 pack/γ 方案） |
| M5 | Phase 4 性能与清理 | 2–3 人日 |

---

## 8. 当前进度（截至计划落盘）

已完成（骨架 + Phase1 工具）：

- [x] **新建** `csrc/turboquant_fia_mse8bit/`（原 fused_infer...8bit 不修改）
- [x] TQ 核 vendoring + raw KV 补丁
- [x] OpDef / tiling / kernel 入口草稿
- [x] C++ 用例 `tools/.../test_aclnn_tq_fia_mse8bit.cpp`
- [x] `rebuild_op.sh` / `build_test.sh` / `prof_*.sh` / `verify_pi_ortho.sh`
- [x] 首次容器内编译：opbuild 已通过；**kernel CCE 编译失败**（见下）

编译阻塞（当前）：

- vendored `kernel_common.h` / `vector_common.h` 在独立 AscendC 环境下：
  - `ConstInfo` / `FIA_LAYOUT` / `ListTensorDesc` 可见性与 include 顺序
  - `Max`/`Min` 与 AscendC API 冲突
- 需继续修 vendored 头文件适配（Phase 1.2）

待做：

- [ ] 修通 kernel 编译并安装 opp
- [ ] 跑 `verify_pi_ortho.sh` PASS
- [ ] torch_binding 新增注册（Phase 2）

---

## 9. 下一步（立即执行项）

1. 补 `test_aclnn_tq_fia_mse8bit.cpp`（从 ops-transformer example 改 aclnn 目标与参数列表）
2. 补 `script/lcy/tq_fia_mse8bit/rebuild_op.sh` + `tools/.../prof_tnd_pa_turboquant.sh`
3. `bash rebuild_op.sh` 编 `TurboquantFiaMse8bit`（目录 `csrc/turboquant_fia_mse8bit`）
4. 跑 `verify_pi_ortho.sh`；按编译/精度错误迭代 tiling 与入口
5. M2 通过后再开 Phase 2

---

## 10. 参考

- ops-transformer：`.cursor/rules/fia-turboquant.mdc`
- 设计：`attention/fused_infer_attention_score/docs/fia_turboquant_design.md`
- 主测 tiling key（原 FIA）：`103000000016300303`（HP+FD+TQ）；本仓简化为 key `0/1`
- 现有工程模板：`script/lcy/tq4bit/`、`test_tq4bit.cpp`
