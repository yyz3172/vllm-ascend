# BitResidual FIA Paged K8V4 — 工具与验收

## P0 交付清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `csrc/bit_residual_fia_paged_k8v4/DESIGN.md` | 完成 | 设计真源 |
| `golden_dequant.py` | 完成 | K8/V4 decode + layout |
| `tests/ut/ops/test_bit_residual_fia_dequant.py` | 完成 | L1/L2 pytest |
| `csrc/bit_residual_fia_paged_k8v4/` | 完成 | vendor TQ FIA + BR Dequant |
| `script/lcy/bit_residual_fia_paged_k8v4/rebuild_op.sh` | 完成 | 构建脚本 |
| `tests/e2e/.../xrx_bit_residual_fia_paged_k8v4_smoke.py` | 完成 | NPU E2E |

## 本地验收（无需 NPU）

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/ops/test_bit_residual_fia_dequant.py --noconftest
```

## NPU 验收

```bash
# 仅本算子（调试）
bash script/lcy/bit_residual_fia_paged_k8v4/rebuild_op.sh
python tests/e2e/singlecard/xrx_bit_residual_fia_paged_k8v4_smoke.py

# 正式进包（build_so.sh → build_aclnn.sh，CUSTOM_OPS 含 bit_residual_fia_paged_k8v4）
bash build_so.sh
nm -D vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so \
  | grep aclnnBitResidualFiaPagedK8v4
```

P1.5 smoke 覆盖：

1. `manual_single_kv`：FIA vs CPU closed-form + vs attn
2. `decode_multi_kv_gqa`：kv=32 跨 block、GQA 8/2、identity rotation
3. `pack_fia_dense_rotation`：dense R × GQA 4/8，FIA/attn/golden 三角对比
4. `prefill_causal_gqa`：q=8/kv=32，`sparse_mode=3` + 2048 下三角 compress mask
5. `prefill_full_seq_dense_rotation`：q=kv=16 全序列 prefill + dense R
6. `flash_decode_long_kv`：kv=1000（> s2Base=512）三角对比
7. `flash_decode_multibatch_vs_attn`：8×kv=1000，压 FD reduce（FIA vs attn）

Prefill mask 约定（FIA）：`atten_mask` 为 **2048×2048 下三角** int8，**0=保留 / 1=屏蔽**；全 0 mask 在多 Q 下等于不做 causal。

## L6 aclnn 性能基线（对齐 TQ FIA）

同规模：`B=16, T=16, H=16, NKV=8, D=128, BS=128, kv=1000`（FD 目标 workload）。

```bash
# 编译 C++ harness（算子已安装则可 --skip-build）
bash tools/bit_residual_fia_paged_k8v4/build_test.sh

# 冒烟运行
bash tools/bit_residual_fia_paged_k8v4/prof_tnd_pa_bit_residual.sh run --kv=1000 --device=1 --skip-build

# msprof：时延 + Pipe/Arithmetic（指令开销）
bash tools/bit_residual_fia_paged_k8v4/prof_tnd_pa_bit_residual.sh msprof --kv=1000 --device=1 --skip-build

# Decode / Prefill + Source（ccec_g；勿与 --skip-build 同用首次）
bash tools/bit_residual_fia_paged_k8v4/prof_tnd_pa_bit_residual.sh msprof \
  --kv=2000 --workload=decode --source
bash tools/bit_residual_fia_paged_k8v4/prof_tnd_pa_bit_residual.sh msprof \
  --kv=2000 --workload=prefill --source   # T=240 (16×15), ≈ serving Q=241
```

输出目录：`tools/.../prof_output/tnd_pa_bit_residual/kv<KV>/<decode|prefill>/op/OPPROF_*`  
对照 TQ：`tools/turboquant_fia_mse8bit/prof_tnd_pa_turboquant.sh msprof --kv=1000`

### 解读指标（固化脚本）

```bash
# 最新 OPPROF，或显式传目录
python3 tools/bit_residual_fia_paged_k8v4/analyze_opprof.py --latest \
  tools/bit_residual_fia_paged_k8v4/prof_output/tnd_pa_bit_residual/kv1000/decode/op

# 指标含义全文
python3 tools/bit_residual_fia_paged_k8v4/analyze_opprof.py --explain
```

关注点摘要：

| 指标 | 含义 | 用途 |
|------|------|------|
| `Task Duration` | 单次 kernel 墙钟时延 | P2/P3 主 KPI；P3：`BR ≤ TQ×1.15` |
| `Pipe bound` (AIC vs AIV) | Cube / Vector 谁更慢 | AIV-bound → 值得做 P2 批量 dequant |
| `aiv_mte2_ratio` / `aiv_vec_ratio` | AIV 搬数 vs 向量算 | mte2>vec 常指向 dequant 趟数 |
| `aic_cube_ratio` / `aic_mte2_ratio` | Cube MAC vs 载入 | AIC-bound 时看 tiling/FD |

`prof_tnd_pa_bit_residual.sh msprof|analyze` 结束后会自动调用该脚本。

## Serving（attention_v1）

`cache_dtype=turboquant` + `turboquant_kv_bits=[8,4]`：

| `AscendAttentionState` | Op | 开关（默认均为 0） |
|------------------------|----|-------------------|
| `DecodeOnly` | `bit_residual_attention_paged_k8v4` | 默认；`VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA=1` → 试 FIA |
| PrefillCacheHit / ChunkedPrefill | `bit_residual_fia_paged_k8v4` | `VLLM_ASCEND_BIT_RESIDUAL_FIA=1` |
| PrefillNoCache | stock FIA（float key/value） | 不读 KV cache |

Pack 仍走 `bit_residual_pack_k8v4`（`reshape_and_cache`）。

Decode A/B 示例：

```bash
# vector paged attn（默认）
unset VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA

# BitResidual FIA decode
export VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA=1
```

## 实现步骤（Agent 模式执行）

1. 从 `csrc/turboquant_fia_mse8bit/` 复制为 `csrc/bit_residual_fia_paged_k8v4/`
2. 全局重命名 `TurboquantFiaMse8bit` → `BitResidualFiaPagedK8v4`
3. 修改 OpDef：uint8 cache，`rotation_key`/`rotation_value`，去掉 γ
4. 在 `fia_block_vec_*` 中替换 `DequantKvImpl`（pack layout + K8/V4 公式）
5. 注册 torch binding（若需要 serving 路由）

## Dequant 核心伪代码（P1, s2_sub=1）

```cpp
// per token si, kv head n2Idx
uint64_t headBase = physBlock * numKvHeads * headStride + n2Idx * headStride;
DataCopy(codes, keyCache[headBase + pos*128], 128);
base = read_fp16(keyCache[headBase + BS*128 + pos*2]);
step = read_fp16(keyCache[headBase + BS*(128+2) + pos*2]);
// vector: sign/q7 → fp32 → fp16 → dequantKWs
```

Value 同理：`64B nibble + vmin/vstep`。
