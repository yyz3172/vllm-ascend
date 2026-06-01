# Attention（attention_v1.py）实现机制说明

> 适用范围：`vllm_ascend/attention/attention_v1.py` 的 `AscendAttentionBackendImpl`
> 分支：`v0.18.0-branch`
> 用途：作为后续 TurboQuant 融合算子开发的背景输入文档。

本文档说明当前昇腾 Attention 后端的实现机制，按 **5 个注意力状态** 拆解处理流程，并把 **非量化（fp16/bf16）路径** 与 **TurboQuant 量化路径** 分开讲清楚。文末附带若干常见问题的举例说明。

---

## 1. 总体结构

| 组件 | 位置 | 角色 |
|---|---|---|
| `AscendAttentionBackend` | `attention_v1.py:85` | 后端工厂：注册 `CUSTOM` 后端、KV cache 形状、block 拷贝/交换 |
| `AscendAttentionState` | `attention_v1.py:158` | 5 种注意力状态枚举 |
| `AscendMetadata` | `attention_v1.py:166` | 每步注意力元数据（mask、seq_lens、block_tables、slot_mapping…）|
| `AscendAttentionMetadataBuilder` | `attention_v1.py:229` | 从通用元数据构建 `AscendMetadata` |
| `AscendAttentionBackendImpl` | `attention_v1.py:373` | 计算实现（forward / KV 写入 / 各 kernel 路径）|
| `AscendC8AttentionBackendImpl` | `attention_v1.py:1082` | INT8 C8（per-channel）KV cache 子类 |

KV cache 形状（`get_kv_cache_shape`，`attention_v1.py:112`）：

- fp16/bf16：`(2, num_blocks, block_size, num_kv_heads, head_size)`
- TurboQuant：`(2, num_blocks, block_size, num_kv_heads, packed)`，`packed = max(P_k, P_v)`，8-bit 时 `P = head_size + 2`（128 字节索引 + 2 字节 fp16 norm）。
- `block_size` 固定为 `128`（`get_supported_kernel_block_sizes`）。

---

## 2. 五个注意力状态

判定逻辑：`model_runner_v1.py:891` `_build_attn_state`，核心信号是 `num_computed_tokens`（已算/已命中）与 `num_scheduled_tokens`（本步排了几个 query token）。

```python
if   num_computed_tokens 全为 0:          → PrefillNoCache
elif num_scheduled_tokens 全为 1:         → DecodeOnly (mtp 时 → SpecDecoding)
elif num_valid_tokens 全为 1:             → SpecDecoding / ChunkedPrefill
elif enable_chunked_prefill:             → ChunkedPrefill
else:                                    → PrefillCacheHit
```

| 状态 | 含义 | query 形状 | 历史 KV | chunked prefill 配置 | 跨序列块共享 |
|---|---|---|---|---|---|
| **PrefillNoCache** | 全新 prefill，无缓存 | 多 token | 无 | — | 无 |
| **PrefillCacheHit** | 命中前缀，一次性 prefill 剩余 | 多 token | 有 | 关 | 可能有 |
| **DecodeOnly** | 纯自回归生成 | `q_len=1`/seq | 有 | — | 稳态下基本无 |
| **ChunkedPrefill** | decode+prefill 分块混合批 | 混合 | 有 | 开 | 可能有 |
| **SpecDecoding** | 投机解码（draft 多 token 验证）| 1~16/seq | 有 | — | 稳态下基本无 |

要点：
- `PrefillCacheHit` 与 `ChunkedPrefill` 是"带历史的 prefill"在 **chunked prefill 关/开** 两种配置下的分支。
- `SpecDecoding` 形状像 decode，但一个序列有多个 query token（`≤16`，受 `decode_threshold` 约束，`attention_v1.py:263`）。

### 一个覆盖全状态的推理时间线

两个请求共享 128-token 系统提示 `S`；Req-A=`S(128)+Qa(200)`，Req-B=`S(128)+Qb(40)`。

| Step | 事件 | 状态 | 触发关键 |
|---|---|---|---|
| 0 | A 第一个 prefill 分块（缓存空）| **PrefillNoCache** | computed 全 0 |
| 1 | A 续算 prefill 分块 | **ChunkedPrefill** | computed≠0，scheduled 非全 1，chunked 开 |
| 2 | A decode + B 命中 S 后 prefill 剩余（混合批）| **ChunkedPrefill** | scheduled=[1,40]，chunked 开 |
| 2′ | 同 Step2 但 chunked **关** | **PrefillCacheHit** | else 分支 |
| 3 | A、B 都生成 1 token | **DecodeOnly** | scheduled 全 1，无 spec |
| 4 | A、B 生成，开 MTP 投机 | **SpecDecoding** | valid 全 1 + spec |

### 各状态的开启方式

5 个状态里 **只有 2 个是任何推理必经的基线状态**，另外 3 个需开启对应特性/配置。判定按 batch 形状 + 配置决定（`model_runner_v1.py:893-911`）。

| 状态 | 是否必经 | 触发所需特性/配置 |
|---|---|---|
| **PrefillNoCache** | ✅ 基线必经 | 无——任何新请求首次 prefill（无缓存命中）|
| **DecodeOnly** | ✅ 基线必经 | 无——任何请求进入自回归生成 |
| **PrefillCacheHit** | ❌ 条件性 | 前缀缓存开 **且** chunked prefill 关 **且** 命中前缀 |
| **ChunkedPrefill** | ❌ 条件性 | `enable_chunked_prefill=True` |
| **SpecDecoding** | ❌ 条件性 | 配 `speculative_config` |

互斥点：
- **PrefillCacheHit 与 ChunkedPrefill 互斥**：由 `enable_chunked_prefill` 决定走哪个；开了 chunked prefill 就永远看不到 PrefillCacheHit。
- **SpecDecoding 完全依赖** `speculative_config`。

#### 请求生命周期（按开启的特性）

| 配置 | 状态序列 |
|---|---|
| 最朴素（无前缀缓存/无 chunked/无投机）| `PrefillNoCache` → `DecodeOnly` |
| + 前缀缓存（chunked 关）| 命中：`PrefillCacheHit` → `DecodeOnly`；未命中：`PrefillNoCache` → `DecodeOnly` |
| + chunked prefill | `PrefillNoCache`（首块 computed=0）→ `ChunkedPrefill`（后续块/混合批）→ `DecodeOnly` |
| + 投机解码(MTP) | …prefill… → `SpecDecoding`（生成阶段验证 draft）|

> 即使开了 chunked prefill，长 prompt 的**第一块**因 `computed==0` 仍命中 `PrefillNoCache`，从第二块起才是 `ChunkedPrefill`。

#### ChunkedPrefill —— 开 `enable_chunked_prefill`

```bash
vllm serve <model> --enable-chunked-prefill --max-num-batched-tokens 2048
```
```python
LLM(model="<model>", enable_chunked_prefill=True, max_num_batched_tokens=2048)
```
- vLLM V1 下多为**默认开启**；`max_num_batched_tokens` 决定每步 token 预算（分块大小）。
- **Ascend 特例**：开启动态批（`ascend_config.SLO_limits_for_dynamic_batch != -1`）时 `platform.py:480` 会**强制** `enable_chunked_prefill=True`。

#### PrefillCacheHit —— 开前缀缓存 **且** 关 chunked prefill

```bash
vllm serve <model> --enable-prefix-caching --no-enable-chunked-prefill
```
```python
LLM(model="<model>", enable_prefix_caching=True, enable_chunked_prefill=False)
```
- 三条件缺一不可：前缀缓存开 + chunked prefill 关 + 请求真的命中前缀（共享相同 system prompt 等）。
- **最挑配置**：只要 chunked prefill 开着（默认），就只会走 ChunkedPrefill。

#### SpecDecoding —— 配 `speculative_config`

支持的 `method`：`draft_model / eagle / eagle3 / medusa / mtp / ngram / suffix`。

```bash
# ngram（无需额外模型）
vllm serve <model> --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}'
# mtp
vllm serve <model> --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
# eagle3（需 draft 模型）
vllm serve <model> --speculative-config '{"method":"eagle3","model":"<draft_model>","num_speculative_tokens":3}'
```
```python
LLM(model="<model>", speculative_config={"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4})
```
- `decode_threshold = 1 + num_speculative_tokens`，**断言 ≤16**（`attention_v1.py:263`，FIA TND 布局限制）。
- `mtp` 在 DecodeOnly 分支即转 SpecDecoding（`model_runner_v1.py:897`）；非 `mtp` 方法某些路径会回退为 ChunkedPrefill（`model_runner_v1.py:915`）。

#### 开启方式小结

| 状态 | 必需配置 | 关键约束 |
|---|---|---|
| ChunkedPrefill | `enable_chunked_prefill=True` | V1 常默认开；Ascend 动态批强制开 |
| PrefillCacheHit | `enable_prefix_caching=True` 且 `enable_chunked_prefill=False` 且命中前缀 | 三条件缺一不可 |
| SpecDecoding | `speculative_config={...}` | `decode_threshold≤16`；mtp 与其它方法行为不同 |

---

## 3. forward 主流程

`forward`（`attention_v1.py:1026`）→ `forward_impl`（`attention_v1.py:1005`）二分派：

```
forward()
  ├─ 断言 output 非空、k/v scale==1.0
  ├─ reshape_and_cache(query,key,value)      # 把当前 KV 写入分页 cache
  ├─ pooling 且非因果 → _forward_encoder_attention
  └─ forward_impl()
        ├─ DecodeOnly + using_paged_attention + 无SWA → forward_paged_attention  (_npu_paged_attention)
        └─ 其余                                       → forward_fused_infer_attention (FIA, TND)
```

**两个核心 kernel**：
- `_npu_paged_attention`：decode 专用分页注意力。
- `npu_fused_infer_attention_score`（FIA）：通用路径，TND 布局，`sparse_mode=3`（因果），prefill / decode 统一处理。

---

## 4. 非量化路径（fp16/bf16）

### 4.1 取数与寻址

`_get_fia_params`（`attention_v1.py:744`，DecodeOnly 为例）：

```python
num_block, block_size, _, _ = self.key_cache.shape   # [num_blocks, 128, num_kv_heads, head_size]
key   = self.key_cache.view(num_block, block_size, -1)   # 仅 reshape，零拷贝
value = self.value_cache.view(num_block, block_size, -1)
block_table = attn_metadata.block_tables                 # 原始页表，不重映射
actual_seq_lengths_kv = attn_metadata.seq_lens_list
```

> **注意参数命名**：传给 `npu_fused_infer_attention_score` 的 `key` / `value` 参数,在 decode/cache 命中场景下**就是整份分页 KV cache 池**(`self.key_cache.view(...)`),不是当前 token 的 K/V——FIA 沿用 paged attention 惯例,用 `key`/`value` 承载 `key_cache`/`value_cache` 的角色。当前 token 的 KV 已在本次调用前被 `reshape_and_cache` 写入 cache,故无需单独传入。`block_table` 寻址的正是"在这个 KV cache 池里,某序列第 p 个历史(及当前)token 落在哪个物理块、块内第几行"。
> 唯一例外是 **PrefillNoCache**:`block_table=None`,`key`/`value` 传入的是当前 prefill chunk 自己的连续 K/V,FIA 不做分页寻址。

传进 FIA 的是 **整份分页 cache 的视图**，`block_table` 是 **原始页表**。逻辑位置 → 物理地址在 **kernel 内部** 完成：

```
逻辑块下标 = p // block_size
物理块 b   = block_table[s, p // block_size]
块内偏移   = p % block_size
K 行       = key_cache[b, off, kv_head, :]
kv_head    = q_head // (num_heads / num_kv_heads)   # GQA 分组
```

### 4.2 算分（paged FlashAttention）

对每个 `(query token, head)`，在线 softmax，KV 按 block tile 即取即算：

```
m=-inf; l=0; acc=0
for tile in kv 按 block_size 切片:
    K_tile = MTE gather(key_cache via block_table)   # 即取即算
    S = (q · K_tileᵀ) * scale
    施因果掩码 (sparse_mode=3)
    m_new=max(m, max S); α=exp(m-m_new); P=exp(S-m_new)
    acc = acc*α + P·V_tile ; l = l*α + ΣP ; m = m_new
O = acc / l
```

**特点**：无 host 侧寻址、无 KV 物化，block_size=128 既是分页粒度也是 tile 粒度。

### 4.3 各状态在非量化下的处理

| 状态 | kernel | block_table | 备注 |
|---|---|---|---|
| PrefillNoCache | FIA | None | kv 长度 = q 长度，无历史 |
| PrefillCacheHit | FIA | 原始 | kv = 命中前缀 + 新算，因果 |
| DecodeOnly | paged / FIA | 原始 | q_len=1，attend 全历史 |
| ChunkedPrefill | FIA | 原始 | decode+prefill 混合，builder 用 `split_decodes_and_prefills` 拆段 |
| SpecDecoding | FIA | 原始 | 多 q token / seq，因果 |

---

## 5. TurboQuant 量化路径

TurboQuant 采用 **写时压缩、读时解码** 的低比特 KV cache：Haar 旋转 + 标量码本，每个 head 向量压成 `head_size 个索引 + 2 字节 fp16 norm`。详见 `vllm_ascend/ops/turboquant_kv_cache.py`。

### 5.1 写入（reshape_and_cache）

`attention_v1.py:971`：`turboquant_pack_kv_for_cache` 量化打包成 uint8，再 `_npu_reshape_and_cache` 写入。

量化语义（`TurboQuantMSE.quantize`，`turboquant_kv_cache.py:336`）：
```
norm  = ‖x‖
y     = (x / norm) @ Rᵀ                  # Haar 旋转
idx   = argmin |y - codebook|            # 最近码本(8-bit: 256 项)
存储   = [idx(128B) | norm(fp16, 2B)]
```

### 5.2 读取/解码 + 算分

`forward_fused_infer_attention`（`attention_v1.py:827`）。**关键差异：TurboQuant 必须先解码、并做 block_table 压缩重映射，再喂 FIA**：

```python
if kv_cache_dtype == "turboquant" and block_table is not None:
    # (新增) 8-bit 融合算子快路径（见 5.4）
    if 满足 8bit/head128/fp16/无sink:
        return turboquant_fused_infer_attention_score_8bit(...)
    # 通用：解码 + 压缩重映射
    key_dec, value_dec, block_table = turboquant_decode_kv_cache_compact(...)
    key = key_dec.flatten(2,3).contiguous(); value = value_dec.flatten(2,3).contiguous()
# 之后走与非量化相同的 FIA 调用
```

解码语义（`TurboQuantMSE.dequantize`，`turboquant_kv_cache.py:347`）：
```
y_hat = codebook[idx]          # 查表
x_hat = y_hat @ R              # 反旋转(注意是 R，不是 Rᵀ)
x_hat = x_hat * norm
```

### 5.3 block 压缩重映射（compact）—— 量化新增的开销

`turboquant_decode_kv_cache_compact`（`turboquant_kv_cache.py:710`）。因为只解码"被引用的块"到一个紧凑小 buffer（重新编号 `0..U-1`），必须把 block_table 也重映射：

```python
bt = block_tables.to(int32)
valid = bt >= 0
used = bt[valid]                         # ④ 布尔 gather，数据相关 shape
used_sorted = used.unique()              # ⑤ 排序去重
key_packed = key_cache.index_select(0, used_sorted)   # ⑥ gather U 个块
bt_compact[valid] = searchsorted(used_sorted, used)   # ⑧ 反向查表 + 散射
```

这套（④⑤⑥⑧）是 **TurboQuant 相对非量化新增的 host 侧开销**——非量化路径直接把原始 block_table 传给 FIA，由 kernel 内寻址，根本没有这些操作。其中 ④⑤ 引入 **数据相关 shape + 同步点**，与 ACL Graph 静态捕获冲突。

### 5.4 已新增的 8-bit 融合算子（未验证）

`turboquant_fused_infer_attention_score_8bit`（`turboquant_kv_cache.py:891`）+ AscendC kernel（`csrc/turboquant_fused_infer_attention_score8bit/`）。
- 命中门（`attention_v1.py:827`）：`turboquant + block_table≠None + sinks=None + bits_key=bits_value=8 + head_size=128 + query.dtype=fp16`。
- 拿不到 C++ 算子时回落 `_turboquant_fused_infer_attention_score_8bit_impl`（`turboquant_kv_cache.py:811`，即"解码 + FIA"）。
- **现状问题**：kernel 为纯标量基线，且只对 **DecodeOnly** 正确（假设单 query token attend 全 kv、忽略 atten_mask）；命中门未收窄到 DecodeOnly，也无 env 开关。详见《attention_turboquant_fusion_design.md》。

### 5.5 FIA 路径对照：turboquant vs 非量化

两者入口都是 `forward_fused_infer_attention`（`attention_v1.py:803`），公共前缀:`forward → reshape_and_cache → forward_impl → forward_fused_infer_attention → _get_fia_params`。**最终的 `npu_fused_infer_attention_score` 调用两者完全相同**;唯一区别是 turboquant 在 `_get_fia_params` 与 FIA 之间**插入了"解码 + block 压缩重映射"**。

**非量化 FIA 链路**：
```python
_get_fia_params: key/value = self.key_cache/value_cache.view(...)   # fp16 cache 视图,零拷贝
                 block_table = 原始页表
# (跳过 turboquant 分支)
npu_fused_infer_attention_score(query, key, value, block_table=原始页表, TND, sparse_mode=3, ...)
```

**turboquant FIA 链路**：
```python
_get_fia_params: key/value = packed uint8 cache 视图   # 还没解码
                 block_table = 原始页表
if kv_cache_dtype=="turboquant" and block_table is not None:        # :830
    # [8bit 命中门] → turboquant_fused_infer_attention_score_8bit(纯PyTorch回落=decode_compact+FIA)
    # [否则] turboquant_decode_kv_cache_compact(...)                # :857  ★ 解码 + 重映射
    #   ├─ unique/index_select(gather U块)/searchsorted             （④⑤⑥⑧）
    #   ├─ codebook[idx] → @R → ×norm                                解码成 fp16
    #   └─ key=key_dec.flatten; value=value_dec.flatten; block_table=bt_compact
npu_fused_infer_attention_score(query, 解码后fp16, 解码后fp16, block_table=bt_compact, TND, sparse_mode=3, ...)
```

| 维度 | 不开 turboquant | 开 turboquant |
|---|---|---|
| `_get_fia_params` 出的 key/value | fp16 cache 视图(零拷贝) | **packed uint8 cache 视图** |
| key/value → FIA 之前 | 不动,直接用 | **decode_compact 解码成 fp16**(codebook+@R+×norm) |
| block_table | 原始页表,直接传 | **unique/index_select/searchsorted 重映射成 bt_compact** |
| host 额外开销 | 无 | **④⑤⑥⑧ + KV 物化**(动态 shape、同步点) |
| 寻址在哪做 | kernel 内即取即算 | host 先解码物化,kernel 再寻址紧凑 buffer |
| 最终 FIA 调用 | — | **与非量化完全相同**(query / 解码KV / bt_compact) |
| PrefillNoCache(block_table=None) | fp16 在手 KV,直接算 | **:830 不成立 → 跳过解码,与非量化一致** |
| 图捕获(capturing) | `full_graph_fia` 正常捕获 | `full_graph_fia` **不解码 → 错误**(应 eager/piecewise，见 `fx_and_acl_graph.md`) |

**一句话**：FIA 算分本身两者同算子、同参数语义;turboquant 只是在"取 cache"和"进 FIA"之间塞了一段 `decode_compact`(反量化 + block 重映射),使下游 FIA 看到的输入与非量化等价。这段插入就是 turboquant 的全部额外开销,也是方案 A/B 的优化对象。

---

## 6. 常见问题举例说明

### Q1：`⑧ searchsorted + 散射` 在干什么？

构造"重映射后的 block_table"，是 ⑥ 的逆映射。举例：cache 有 10 个物理块，batch 两序列共享物理块 2：

```
block_tables:  seq0:[5,2,-1]  seq1:[2,7,-1]
used        = [5,2,2,7]
used_sorted = [2,5,7]                       # U=3
index_select(0,[2,5,7]): 槽位0←块2, 槽位1←块5, 槽位2←块7
searchsorted([2,5,7],[5,2,2,7]) = [1,0,0,2]
bt_compact:    seq0:[1,0,-1]  seq1:[0,2,-1]   # 改成索引小 buffer
```
共享的物理块 2 在 seq0/seq1 都指向紧凑槽位 0（只解码一次）。

### Q2：`⑤ unique` 能省吗？

- **单序列内**块号不重复；**跨序列**（前缀缓存）可能重复 → 不能无脑省。
- 若接受"共享块重复解码一次"，可把 **⑤+⑧** 一起换成 `arange`（`index_select(0, used)` + `bt_compact[valid]=arange`），省掉排序与二分，结果仍正确。
- **DecodeOnly 稳态**跨序列基本不共享块 → `unique` 近似空操作 → 推荐换 `arange`；**PrefillCacheHit / ChunkedPrefill** 共享概率高 → 保留 `unique`。
- 注意 ④ `bt[valid]` 的动态 shape 与 `unique`/`searchsorted` 无关，仍存在。

### Q3：哪些状态当前融合算子能覆盖？

只有 **DecodeOnly**。`ChunkedPrefill`/`PrefillCacheHit`/`SpecDecoding` 都是多 query token + 因果掩码，当前 kernel 未实现 per-token 掩码逻辑。

---

## 7. 参考代码索引

| 主题 | 位置 |
|---|---|
| forward / forward_impl | `attention_v1.py:1005-1079` |
| FIA 路径 + turboquant 分支 | `attention_v1.py:803-899` |
| paged attention 路径 | `attention_v1.py:901-932` |
| `_get_fia_params` 寻址 | `attention_v1.py:726-765` |
| reshape_and_cache（写入）| `attention_v1.py:954-1003` |
| 状态判定 | `model_runner_v1.py:891` |
| TurboQuant 量化/解码/compact | `turboquant_kv_cache.py:336-808` |
| 8-bit 融合算子封装 + fallback | `turboquant_kv_cache.py:811-953` |
| AscendC kernel | `csrc/turboquant_fused_infer_attention_score8bit/` |
