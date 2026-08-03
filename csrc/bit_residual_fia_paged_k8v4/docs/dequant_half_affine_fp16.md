# Dequant half affine（仅 fp16）

## 策略

| `Q_T` / `WS_T` | Dequant 路径 |
|----------------|--------------|
| **half (fp16)** | `u8/int4 → half` → `BrApplyRowAffine(half)` → **直写 half WS**（无 CAST #1） |
| **bfloat16** | **无法**做原生 bf16 affine：910B AscendC `Mul`/`Add`/`Muls`/`Adds` 不支持 `bfloat16_t`，且无 `uint8/int4→bf16` Cast。保持 **fp32 affine + Cast→bf16** |
| **Scheme C legacy** | 强制 fp32 affine（sign+q7，3×fp32 scratch） |

编译期分支：`IsSameType<WS_T, half>::value`（Scheme C legacy 除外，仍走 fp32）。

**不要**在 bf16 上做 half affine 再写 WS（无 `half↔bf16` Cast；且 bit 解释错误）。

## 实现要点

- `BrApplyRowAffine(LocalTensor<half>, …)`：`FP16_BLOCK=16`，`REPEAT=128`；Brcb 仍按每 repeat **8** 个标量（与 fp32 相同）。
- Meta：**全程 half**。`hoistTile` 为 16 的倍数，使 `metaHalf[j0]` 天然 32B 对齐，**直接**喂给 `Brcb`（无 `half→fp32→half`，也无 `Adds` staging——两者在未对齐 half src 上都会 VEC fault）。
- Key/Value decode tile：half 用 **`BR_*_DECODE_TILE_MAX_HALF=16`**（Brcb 对齐）；bf16/Scheme C 仍用 **8/13**（避免 fp32 scratch 翻倍挤爆 `maxSub`）。
- Value fold：`BrFoldValueMetaHalf`（`vmin' = vmin + 8·vstep`）在 half 上一次完成。
- 大头省下的是：`q/s` 的 `half→fp32`（tile×128）与出口 **CAST #1**，以及 meta 往返 Cast。

## Longquery：fp16 vs bf16 对比

模型默认 `torch_dtype: bfloat16`。测 fp16 降精度时：

```bash
# 备份后改 dtype，加载会按 float16 转权重
sed -i 's/"torch_dtype": "bfloat16"/"torch_dtype": "float16"/' \
  /root/cyl/model/Qwen3-0.6B/config.json
```

跑完 longquery 后改回 `bfloat16`，再跑一轮作对照。

L6 harness 本身已是 fp16，可直接覆盖 half affine。

## 验收

- Golden：`xrx_bit_residual_k8v4_golden.py`（fp16 + bf16）
- L6：`prof_tnd_pa_bit_residual.sh` decode/prefill Task Duration
- Longquery：改 `config.json` 后对比 bf16 基线
