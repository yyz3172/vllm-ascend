# TurboQuant 反量化与 FIA 融合：特性优化设计

> 适用范围：`vllm_ascend/ops/turboquant_kv_cache.py` + `csrc/turboquant_fused_infer_attention_score8bit/` + `attention_v1.py` 的 turboquant 分支
> 分支：`v0.18.0-branch`
> 前置阅读：《attention_v1_mechanism.md》

本文档面向后续开发，给出 TurboQuant 量化融入 Attention 的优化方案，包含三部分：
1. 纯 PyTorch 实现说明 + 不支持的状态 + 如何支持；
2. 反量化与 FIA 融合 —— **方案 A（解码算子化 + 复用 FIA）**；
3. 反量化与 FIA 融合 —— **方案 B（片上真正融合）**。
并给出开发计划与测试用例设计。

---

## 0. 背景与目标

TurboQuant 低比特 KV cache 的注意力当前走"**先解码到紧凑 buffer，再调 FIA**"的两段式（纯 PyTorch）。相对非量化路径，它额外引入：

- **block 压缩重映射** `unique/searchsorted/index_select`（host 侧，含数据相关 shape，见机制文档 §5.3）；
- **整份被引用 KV 的解码物化**（HBM 读写放大）。

优化目标：在保证数值正确的前提下，**逐步消除上述额外开销**，并支持更多状态。

### 范围约定：仅 8-bit，暂不考虑 4-bit

> **本文档只针对 8-bit TurboQuant。** 4-bit 路径（含已有的 `turboquant_decode_packed_blocks_compact` 自定义算子）**正确性同样未经验证**，因此 **不作为参考模板、不在本期开发范围内**。
>
> 优先级：**先把 8-bit 的全部能力（方案 A → 方案 B → 扩展状态）做对做稳**；待 8-bit 完整可用并通过验证后，再单独评估是否复用其经验推进 4-bit。下文所有自定义算子/测试默认 `bits_key == bits_value == 8`。

三种实现的关系：

```
纯 PyTorch(现状) ──A──> 解码算子化 + 复用 FIA ──B──> 片上真正融合(解码内联进 FlashAttention)
   多 kernel        单解码 kernel + FIA(已验证)      单 kernel，无 compaction、无解码物化
```

---

## 1. 纯 PyTorch 实现说明

### 1.1 数据流

入口：`forward_fused_infer_attention`（`attention_v1.py:827`）→ `turboquant_decode_kv_cache_compact`（`turboquant_kv_cache.py:710`）→ `npu_fused_infer_attention_score`。

```
① bt = block_tables.to(int32)
② valid = bt >= 0
③ any(valid) + Python 分支            # D2H 同步
④ used = bt[valid]                    # 布尔 gather，数据相关 shape
⑤ used_sorted = used.unique()         # 排序去重，数据相关 shape
⑥ key_packed = key_cache.index_select(0, used_sorted)   # gather U 个块 [U,BS,H,P]
   value_packed = ...
   ─ 解码：codebook[idx] → @R → ×norm  →  k,v: [U,BS,H,D] fp16
⑦ bt_compact = bt.clone()
⑧ bt_compact[valid] = searchsorted(used_sorted, used)   # 反向查表 + 散射
→ npu_fused_infer_attention_score(query, key.flatten, value.flatten, block_table=bt_compact, ...)
```

解码语义（`turboquant_kv_cache.py:347` / `:532`）：
```
y_hat = codebook[idx]      # 8-bit: 256 项码本
x_hat = y_hat @ R          # 反 Haar 旋转
x_hat = x_hat * norm       # fp16 norm
```

> 注：**8-bit 解码目前走纯 PyTorch 参考路径**（`turboquant_dequantize_from_packed_bytes`，`turboquant_kv_cache.py:532`），这是本期要替换/加速的对象。
> （仓库里另有 4-bit 的 `turboquant_decode_packed_blocks_compact`，但其正确性未验证，本文档不依赖、不参考。）

### 1.2 当前支持矩阵

| 状态 | 纯 PyTorch 解码+FIA | 8-bit 融合算子(已加) |
|---|---|---|
| PrefillNoCache | ✅（block_table=None，不进 compact）| ❌（命中门要求 block_table≠None）|
| PrefillCacheHit | ✅ | ❌ 未实现多 token 因果 |
| DecodeOnly | ✅ | ⚠️ 仅此状态正确，但门未收窄 |
| ChunkedPrefill | ✅ | ❌ 未实现多 token 因果 + decode/prefill 混合 |
| SpecDecoding | ✅ | ❌ 未实现多 query token 因果 |

纯 PyTorch 路径 **功能上覆盖全部状态**（因为最终复用通用 FIA），问题只在 **性能**（compaction + 解码物化 + 多 kernel launch）。

### 1.3 纯 PyTorch 算子不支持/待改进项 + 如何支持

| 问题 | 现状 | 如何支持 |
|---|---|---|
| 8-bit 无 compact 解码算子 | 走纯 PyTorch 逐元素解码，慢 | 新增 **独立的 8-bit** compact 解码自定义算子（码本 256、无 nibble 解包、行宽 130），不依赖 4-bit 实现，需独立数值验证 |
| `unique/searchsorted` 数据相关 shape | 每步执行，破坏图捕获 | DecodeOnly 走 `arange` 快路径替换 ⑤⑧（见机制文档 §6 Q2）|
| 解码 KV 物化到 HBM | `index_select`+`cat`+`contiguous` 多次读写 | 方案 A 解码 kernel 减少中间物化；方案 B 彻底不物化 |
| 多 kernel launch | ~8–10 次 | 方案 A 合并解码为 1 次；方案 B 合并解码+attention 为 1 次 |

**纯 PyTorch 层面可立即落地的低风险改进（Phase 0）**：
1. 新增 **8-bit** compact 解码自定义算子（独立实现，不复用未验证的 4-bit 算子）。
2. `turboquant_decode_kv_cache_compact` 按 `attn_state==DecodeOnly` 走 `arange`，其余保留 `unique`。
3. 给所有 8-bit turboquant 算子路径补 env 开关与 `_c_ascend_turboquant_op_available` 双重门控（与现有 `VLLM_ASCEND_TURBOQUANT_DECODE_OP` 风格一致）。

---

## 2. 方案 A —— 解码算子化 + 复用 FIA

### 2.1 思路

把"解码 packed KV → fp16"做成 **一个向量化的 AscendC 解码算子**（codebook 查表 + Cube 做 `y_hat@R` + ×norm），输出 fp16 K/V 到 workspace；随后调 **已验证、已优化的 `npu_fused_infer_attention_score`**。

```
[A] decode_kernel(packed K/V, codebook, R) → fp16 K/V(workspace) → FIA(已验证)
```

### 2.2 与纯 PyTorch 对比的优势

| 维度 | 纯 PyTorch | 方案 A |
|---|---|---|
| 解码 kernel launch | ~8–10 | **1** |
| 中间物化 | `codebook[idx]`、decode、cat、contiguous 多份 | y_hat 留 UB，仅写一次 decoded |
| 解码 HBM 流量 | ~5–7× 最小 | ~2× 最小（约 **3–4× 下降**）|
| 片上复用 | codebook/R 每次从 HBM 读 | codebook 常驻 UB、R 常驻 L1 |
| host 串行点 | unique/searchsorted 卡 stream | 折进 kernel 寻址 |
| 计算单元 | 通用 batched matmul | Cube，R 作 stationary |
| FIA 段 | 同 | **完全相同（复用）** |

> 注意：方案 A **仍保留** block 压缩重映射（除非配合 §1.3 的 `arange` 改进），且 decoded KV 仍落 workspace 再被 FIA 读回——这部分 HBM 往返是方案 B 才消除的增量。

### 2.3 性能预期

- 解码阶段 HBM 流量约降到 1/3~1/2，launch ~10→1。
- 端到端受 **解码:FIA 占比** 制约（Amdahl）：**长上下文、小 batch（典型 decode）收益最大**。
- 不能给确切倍数，需 profiling 实测。

### 2.4 状态支持

方案 A 因复用通用 FIA，可覆盖 **全部状态**（与纯 PyTorch 一致），只是把解码换成快算子。

### 2.5 复杂度/风险

- **无可信模板**：4-bit 算子未验证，不作参考；8-bit 解码算子需从头实现并独立验证。
- 验证相对简单：只需对"解码算子输出"与纯 PyTorch 8-bit 解码（`turboquant_dequantize_from_packed_bytes`）做数值对拍即可，golden 明确。

---

## 3. 方案 B —— 片上真正融合

### 3.1 思路

单个 AscendC kernel 内完成"**解码内联 + FlashAttention**"，KV 不落 GM、不做 host compaction（kernel 内用 **原始 block_table** 寻址，与原生 paged attention 一致）。

```
[B] for tile in kv(按 block_size, via 原始 block_table):
        gather packed K/V tile → 片上解码(查表 + y_hat@R(Cube) + ×norm) → K_tile/V_tile(fp16)
        S = Q·K_tileᵀ * scale → 施 atten_mask/因果 → 在线 softmax → acc += P·V_tile
    O = acc / l
```

### 3.2 PyTorch → kernel 阶段映射

| PyTorch 参考 | 方案 B kernel 阶段 |
|---|---|
| `block_table` 寻址 | kernel 内 MTE gather（原始页表，**无 compaction**）|
| `codebook[idx] * norm` | tile 内查表 + 向量乘（codebook 常驻 UB）|
| `y_hat @ R` | Cube 矩阵乘（R 常驻 L1，stationary）|
| `Q·Kᵀ * scale` | Cube + scale |
| `softmax`（因果/mask）| 在线 softmax + **应用 atten_mask** |
| `P·V` | Cube，在线累加 |

### 3.3 相对方案 A 的增量收益

- **消除 block 压缩重映射**（④⑤⑥⑧ 全无），恢复对 ACL Graph 友好的静态 shape。
- **消除 decoded KV 的 HBM 落盘与回读**（方案 A 仍有）。

### 3.4 当前 kernel 现状与差距

现有 `csrc/turboquant_fused_infer_attention_score8bit/op_kernel/...cpp` 是 **correctness-first 标量基线**，差距：
- 纯标量、`GetValue/SetValue` 逐元素，无 Cube、无向量化、无双缓冲 → 性能不可用；
- 只对 **DecodeOnly** 正确（`lastPos=kvLen-1`，单 query token）；
- `(void)attenMask` 丢弃掩码；
- 无 env 开关，命中门未收窄。

方案 B 需在此基础上重写为 Cube + 向量化在线 softmax，并补齐多 query token 的 per-token 因果掩码。

### 3.5 A→B 切换成本

**中等，且前置去风险**。A 与 B 共享"解码前端"（packed 布局解析、codebook UB、R 的 Cube 乘、norm 解析、host/tiling/build 脚手架、对拍 golden）。A→B 丢弃的是"写 workspace + 单独调 FIA"，新增的是 kernel 内 FlashAttention 后端（标准但需正确数值 + 掩码）。前提：**方案 A 的解码必须写成可复用 device 函数**，而非与写回揉死。

---

## 4. 开发计划

### Phase 0 —— 纯 PyTorch 去风险（低风险，先做）
- [ ] 0.1 命中门收窄到 `attn_state == DecodeOnly`（`attention_v1.py:827`），把多 token 状态挡在融合算子外。
- [ ] 0.2 新增 env 开关（如 `VLLM_ASCEND_TURBOQUANT_FUSED_FIA_8BIT`，默认关）+ `_c_ascend_turboquant_op_available` 门控。
- [ ] 0.3 `turboquant_decode_kv_cache_compact` 增加 DecodeOnly `arange` 快路径（替换 `unique/searchsorted`）。
- [ ] 0.4 新增 8-bit compact 解码自定义算子（独立实现 + 数值验证；不复用未验证的 4-bit 算子）。
- [ ] 0.5 图模式 guard（见下「turboquant 与图模式兼容性」）。
- [ ] 0.6 SWA + turboquant：`_forward_fia_slidingwindow`（`attention_v1.py:767`）依赖永不被赋值的 `_decoded_key_cache` → 当前 **assert 崩溃**；要么实现该 cache，要么对 `turboquant + sliding_window` 显式报错不支持。
- 交付：纯 PyTorch 路径在 DecodeOnly 下更快且图友好，融合算子安全 opt-in，图模式/SWA 边界不再出错。

#### turboquant 与图模式兼容性（Phase 0 必修）

> 背景原理见 `fx_and_acl_graph.md`。结论:**turboquant 只能在 PIECEWISE 或 NONE 下正确运行,不能进 FULL / FULL_DECODE_ONLY。**

| 图模式 | 注意力 | turboquant 正确性 | 说明 |
|---|---|---|---|
| `NONE`（全 eager）| eager | ✅ | 最稳 |
| **`PIECEWISE`（默认）** | eager | ✅ | 注意力 eager → `decode_compact` 动态 shape 在 eager 下合法；piece 不碰 KV cache。**推荐** |
| `FULL_DECODE_ONLY` | 进图 | ❌ | decode 走 `full_graph_pa`（`attention_v1.py:663`）**不解码 packed cache** → 输出错；compact 动态 shape 也不可捕获 |
| `FULL` | 进图 | ❌ | 同上，`full_graph_fia`（`attention_v1.py:570`）不解码 |

两条根因（均见 `fx_and_acl_graph.md` §7）：
1. `turboquant_decode_kv_cache_compact` 的 `unique()` / `bt[valid]` 是**数据相关 shape**，触发 graph break / 不可捕获。
2. 捕获路径 `full_graph_fia` / `full_graph_pa` **没有 turboquant 解码分支**，会直接在 packed uint8 cache 上算注意力 → 结果错误。

**开发项（Phase 0.5）**：
- [ ] 在 `platform.py` 配置阶段加 guard：当 `kv_cache_dtype == "turboquant"` 且 `cudagraph_mode in {FULL, FULL_DECODE_ONLY}` 时，**强制降级为 `PIECEWISE` 并 `logger.warning`**（默认 PIECEWISE 本就安全，仅拦截用户显式开 full 的情况）。
- [ ] （可选加固）在 `full_graph_fia` / `full_graph_pa` 入口对 `kv_cache_dtype == "turboquant"` 直接 `raise`，作为第二道防线。
- [ ] UT：见测试用例 G1（turboquant 下 full 模式被拦截/降级，piecewise 数值正确）。

> 方案 B（片上寻址、静态 shape、kernel 内解码）的目标之一，就是让 turboquant 也能进 **FULL** 图——届时注意力可被捕获、host 开销彻底隐藏，本节 guard 可放开。

### Phase 1 —— 方案 A（解码算子化）
- [ ] 1.1 实现/复用 8-bit 解码 AscendC 算子：Cube 做 `y_hat@R`，codebook UB、R L1。
- [ ] 1.2 Python 侧改为 "解码算子 → FIA"，解码前端封装为可复用 device 函数。
- [ ] 1.3 数值对拍 + profiling（解码阶段、端到端）。
- 交付：DecodeOnly（及可选全状态）下解码加速，FIA 复用。

### Phase 2 —— 方案 B（片上融合，DecodeOnly）
- [ ] 2.1 重写 kernel：tile gather（原始 block_table）+ 片上解码 + Cube QKᵀ/PV + 向量化在线 softmax。
- [ ] 2.2 复用 Phase 1 解码前端 device 函数。
- [ ] 2.3 与方案 A / 纯 PyTorch 三方数值对拍 + profiling。
- 交付：DecodeOnly 无 compaction、无解码物化的真正融合。

### Phase 3 —— 方案 B 扩展状态
- [ ] 3.1 支持多 query token 的 per-token 因果掩码（PrefillCacheHit / SpecDecoding）。
- [ ] 3.2 支持 ChunkedPrefill 的 decode/prefill 混合（参考 `_forward_c8_chunked_prefill` 拆段思路，`attention_v1.py:1277`）。
- [ ] 3.3 放开命中门到对应状态。

### Phase 4 —— 4-bit（仅在 8-bit 全部能力 OK 后启动）
- [ ] 4.1 在 8-bit 方案 A/B 验证通过后，评估是否将经验推广到 4-bit（含先验证现有 `turboquant_decode_packed_blocks_compact` 的正确性）。
- 说明：**4-bit 不在本期范围**，此 Phase 仅为占位，待 8-bit 收尾后单独立项。

> 依赖关系：Phase 0 可独立先行；Phase 1 是 Phase 2 的解码前端来源；Phase 3 依赖 Phase 2；**Phase 4（4-bit）依赖 Phase 0~3 全部完成**。

---

## 5. 测试用例设计

### 5.1 数值正确性（golden = 纯 PyTorch 参考）

参考实现：`_turboquant_fused_infer_attention_score_8bit_impl`（`turboquant_kv_cache.py:811`）与 `turboquant_dequantize_from_packed_bytes`。扩展 `tests/ut/attention/test_attention_v1.py`。

| 用例 | 输入构造 | 校验 |
|---|---|---|
| T1 解码算子对拍（A）| 随机 packed K/V cache + block_table（无共享）| 解码输出 vs PyTorch 解码，`allclose(atol/rtol 按 fp16)` |
| T2 融合输出对拍（A/B）| 随机 Q + packed cache + block_table，DecodeOnly | attn_output vs fallback |
| T3 跨序列块共享 | block_table 含重复物理块（模拟前缀命中）| 共享块解码一致；`arange` 路径与 `unique` 路径结果一致 |
| T4 padding/-1 | block_table 含 -1 尾部填充 | 无越界，结果与 golden 一致 |
| T5 kvLen 非块整除 | kvLen=300（最后块 44 行）| 因果界正确，尾块只用有效行 |
| T6 GQA | num_heads≠num_kv_heads | kv_head 分组正确 |
| T7 mixed/4-bit 不命中融合 | bits≠8 或 bits_key≠bits_value | **不进 8-bit 融合算子**，回落纯 PyTorch，数值对 |

### 5.2 状态门控

| 用例 | 校验 |
|---|---|
| S1 命中门 | DecodeOnly 命中融合算子；PrefillCacheHit/ChunkedPrefill/SpecDecoding 回落（Phase0 后）|
| S2 env 开关 | 开关关闭时走纯 PyTorch；打开且算子存在时走融合 |
| S3 算子缺失回落 | `_C_ascend` 无该 op 时 `turboquant_fused_infer_attention_score_8bit` 正确回落 fallback |
| S4 dtype/head_size 边界 | 非 fp16 / head_size≠128 / sinks≠None 时不命中融合 |

### 5.3 compact 等价性

| 用例 | 校验 |
|---|---|
| C1 arange vs unique | DecodeOnly 下两路径 `k,v,bt_compact` 等价（共享块时 arange 允许重复解码但 attn 结果一致）|
| C2 空/全 -1 block_table | 返回空 decoded，无异常 |

### 5.4 图捕获 / 端到端

| 用例 | 校验 |
|---|---|
| G1 图模式 guard（Phase 0）| turboquant + `FULL/FULL_DECODE_ONLY` 被强制降级为 `PIECEWISE` 并告警；turboquant + `PIECEWISE`（默认）数值正确 |
| G2 图捕获（方案 B 目标）| 方案 B 下 DecodeOnly 注意力可被 full graph 捕获（静态 shape，无数据相关 shape）|
| E1 端到端精度 | 小模型 8-bit turboquant 生成结果与非量化基线在可接受误差内（如 perplexity / 关键 token 一致率）|
| E2 性能回归 | profiling 记录解码阶段耗时、端到端吞吐，长上下文/小 batch 重点观测 |

### 5.5 测试矩阵建议

- bits：**仅 8（A/B 主线）**；4-bit 不在本期范围（仅验证"不命中 8-bit 融合、正确回落"）
- 状态：DecodeOnly（主线）、其余（回落或 Phase3）
- shape：head_size=128；num_heads∈{8,32}，num_kv_heads∈{1,8,32}（含 MQA/GQA）；kvLen∈{1,127,128,300,2048}
- dtype：fp16（融合主线）、bf16（纯 PyTorch）

---

## 6. 参考代码索引

| 主题 | 位置 |
|---|---|
| FIA + turboquant 分支 + 8bit 命中门 | `attention_v1.py:803-899` |
| `turboquant_decode_kv_cache_compact` | `turboquant_kv_cache.py:710-808` |
| 8-bit 融合封装 + fallback | `turboquant_kv_cache.py:811-953` |
| 4-bit compact 解码算子（**未验证，不参考**，仅本期范围外）| `turboquant_kv_cache.py:763-808` |
| 量化/解码语义 | `turboquant_kv_cache.py:336-403, 532-607` |
| AscendC kernel（待重写）| `csrc/turboquant_fused_infer_attention_score8bit/op_kernel/` |
| tiling | `csrc/turboquant_fused_infer_attention_score8bit/op_host/` |
| C8 chunked prefill 拆段参考 | `attention_v1.py:1277-1375` |
| UT | `tests/ut/attention/test_attention_v1.py` |
