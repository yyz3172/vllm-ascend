# BitResidualFiaPagedK8v4 设计文档

> 从计划文档整理。P0 目标：在不动 `turboquant_fia_mse8bit` / 现有 pack / attn 的前提下，新建 FIA 流水算子，仅替换 AIV Dequant 为 bit_residual K8/V4。

## 1. 架构

```text
pack_k8v4 (不变) → uint8 PA cache
query → Q@R_key → Cube MM1 ← dequantKWs (BR K8 dequant)
                → Softmax → Cube MM2 ← dequantVWs (BR V4 dequant)
                → @R_value → attention_out
```

| 项 | 值 |
|----|-----|
| 目录 | `csrc/bit_residual_fia_paged_k8v4/` |
| OpDef | `BitResidualFiaPagedK8v4` |
| aclnn | `aclnnBitResidualFiaPagedK8v4` |

## 2. IO

**Inputs**

- `query` / `attention_out`：fp16 **或 bf16** TND `[T,H,D]`
- `key_cache` / `value_cache`：**uint8**（pack 布局）
- `block_table` int32
- `actual_seq_len_q` / `actual_seq_len_kv` int64（ValueDepend）
- `atten_mask` optional
- `rotation_key` / `rotation_value` `[D,D]`，**与 query 同 dtype**

**去掉**：`key_scale` / `value_scale` / 单一 `rotation`

**Attrs**：`num_heads`, `num_kv_heads`, `head_size(=128)`, `block_size(%16==0)`, `scale_value`, `pre_tokens`, `next_tokens`, `sparse_mode`

**dtype 契约**：pack 写入的 base/step/vmin/vstep 是 2 字节浮点，dtype 跟 pack 输入一致。FIA dequant **必须**按 query dtype 解读 metadata；bf16 pack + 按 half 读 meta 会导致数值炸裂（serving 乱码根因）。

## 3. Cache 布局（pack 真源）

```text
Key  head: [BS*128 codes][BS*2 base][BS*2 step]   // 132B/row, 2112B/16-row
Value head: [BS*64 nibbles][BS*2 vmin][BS*2 vstep] // 68B/row, 1088B/16-row
```

**PA 寻址**（`block_size=BS`, `D=128`）：

```text
head_stride_key   = BS * (128 + 4)
head_stride_value = BS * (64 + 4)
code_off_k        = pos_in_block * 128
base_off_k        = BS*128 + pos_in_block*2
step_off_k        = BS*(128+2) + pos_in_block*2
code_off_v        = pos_in_block * 64
vmin_off_v        = BS*64 + pos_in_block*2
vstep_off_v       = BS*(64+2) + pos_in_block*2
```

## 4. 反量化公式

### Key (K8)

```text
q7   = code & 0x7F
sign = code >> 7     // 0→+1, 1→-1
y    = (base + q7 * step) * (sign ? -1 : +1)
```

### Value (V4)

```text
idx4 = unpack_nibble(stored)   // signed int4 → [0,15]
V    = vmin + idx4 * vstep
```

### 手算例（Key 前 4 维）

```text
y = [+0.80, -0.20, +0.50, -0.10], M=0.80
base=0.10, step≈0.00551
d0: code=127 → +0.80; d1: q7=18,sign=1 → -0.20; ...
```

## 5. Kernel 改造点（vendor TQ FIA）

| 保留 | 替换 |
|------|------|
| Cube MM1/MM2, Softmax, FD, Flag 7–9 | — |
| `DequantKvImpl` + flat int8 offset + γ | **BR Dequant** + pack layout 寻址 |
| `rotation` / γ GM | `rotation_key` / `rotation_value` |

**P1 Dequant 约束**：`s2_sub=1`（对齐 TQ），dequant TBuf ≤3KB；禁止 16-row float **常驻** TBuf。

**P2**：Dequant 时分借 `tmpBuff1(32KB)`，同 PA block 连续 run 批量 GM→UB（codes+meta），`s2_sub` 上限由 UB 动态决定（≤32）；decode 仍按行复用 fp32 scratch。

## 6. UB 账本（AIV ~192KB）

| Buffer | 大小 |
|--------|------|
| Queues + Softmax + tmpBuff1 | ~184KB |
| dequant (1 row) | ~0.9KB |
| **剩余** | **~7KB** |

批量 Dequant 应 batch **同 kv head 的多 token (s2)**，不是多 Q head（GQA 共享 K/V）。

## 7. 测试矩阵

| 层级 | 文件 | 需 NPU |
|------|------|--------|
| L1/L2 | `tools/bit_residual_fia_paged_k8v4/golden_dequant.py` + `tests/ut/ops/test_bit_residual_fia_dequant.py` | 否 |
| L5 | `tests/e2e/singlecard/xrx_bit_residual_fia_paged_k8v4_smoke.py` | 是 |
| L6 | `tools/bit_residual_fia_paged_k8v4/test_aclnn_*.cpp` + `prof_tnd_pa_bit_residual.sh` | 是 |

**E2E 对比**：`pack → bit_residual_attention_paged_k8v4` 为基线，新 op 输出 `max_abs ≤ 2e-3`。

**不做**：与 TQ MSE golden 对比。

## 8. 性能路径与 attention_v1 接入

| 场景 | 路径 |
|------|------|
| Decode Q 小（`DecodeOnly`） | 默认 `bit_residual_attention_paged_k8v4`；`VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA=1` 时试 FIA（A/B） |
| PrefillCacheHit / ChunkedPrefill | `VLLM_ASCEND_BIT_RESIDUAL_FIA=1` → `bit_residual_fia_paged_k8v4`；否则 vector attn |
| PrefillNoCache | **始终** float `key`/`value`（`block_table=None`），不读 KV cache |

**开关**（均默认 `0`）：
- `VLLM_ASCEND_BIT_RESIDUAL_FIA`：PrefillCacheHit / ChunkedPrefill → BR FIA
- `VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA`：Decode → BR FIA（对比 vector paged attn）

```text
AscendAttentionBackendImpl.forward()
  └─ reshape_and_cache → bit_residual_pack_k8v4   # turboquant_kv_bits=[8,4]
  └─ forward_impl
        ├─ DecodeOnly + using_paged_attention
        │     ├─ DECODE_FIA=1 → bit_residual_fia_paged_k8v4 (sparse_mode=0)
        │     └─ else / fallback → bit_residual_attention_paged_k8v4
        └─ else → forward_fused_infer_attention
              ├─ PrefillNoCache → stock FIA (float key/value)
              ├─ PrefillCacheHit/ChunkedPrefill + FIA=1
              │     → bit_residual_fia_paged_k8v4 (atten_mask, sparse_mode=3)
              └─ fallback → bit_residual_attention_paged_k8v4
```

FIA 框架内 **无法** 全面快于 TQ FIA；decode 默认走 vector attn，用 `DECODE_FIA` 做 A/B。

**bf16 serving**：kernel / OpDef / wrapper 原生支持 bf16（query、rotation、pack meta 同 dtype）；勿再对 bf16 做 `→fp16` 边界 cast。

## 9. 实现分期

| 阶段 | 内容 | 状态 |
|------|------|------|
| P0 | golden + pytest + 脚手架 + 单行 Dequant | 完成 |
| P1 | E2E vs attn/golden：identity×GQA；dense×GQA；meta/WS/`Q@Π` | 完成 |
| P1.5 | Prefill/FD 正确性 + L6 aclnn/msprof 基线 | 完成 |
| Serving | Prefill→FIA / Decode→attn（可配 Decode FIA A/B） | 完成 |
| bf16 | 原生 bf16 query/rotation + meta dequant；smoke golden | 完成 |
| P2 | tmpBuff1 时分复用 + s2_sub 批量 dequant（同 PA block run） | **已做**；L6 用 `analyze_opprof.py` 对比 |
| P3 | Queue 合并 + msprof ≤ TQ FIA +15% | **未做（性能）** |

## 10. 构建

整仓（推荐，会进 `vllm_ascend/_cann_ops_custom`）：

```bash
# build_so.sh → COMPILE_CUSTOM_KERNELS=1 → setup.py build_ext
#   → build_aclnn.sh（CUSTOM_OPS 须含 bit_residual_fia_paged_k8v4）
bash build_so.sh
```

仅本算子（调试）：

```bash
bash script/lcy/bit_residual_fia_paged_k8v4/rebuild_op.sh
pytest -sv tests/ut/ops/test_bit_residual_fia_dequant.py
```

确认安装成功：

```bash
nm -D vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so \
  | grep aclnnBitResidualFiaPagedK8v4
```

**注意**：`turboquant_fia_mse8bit` 与本算子都 vendored 了 `split_core.cpp`（同名 `optiling::SplitCore`），不可同时进 `build_aclnn.sh` 的 CUSTOM_OPS；TQ FIA 请单独 `rebuild_op.sh`。
