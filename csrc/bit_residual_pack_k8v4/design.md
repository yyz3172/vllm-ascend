# BitResidual Pack KV For Cache

`bit_residual_pack_kv_for_cache` packs token-major K/V tensors into a
field-major paged cache for the 1-bit sign + 7-bit residual quantizer.

Scope:

- `head_size=128`
- K/V input dtype: fp16 or bf16
- `rotation_t` dtype: fp16, shape `[128, 128]`
- cache dtype: uint8, shape `[num_blocks, num_heads, block_size * 134]`
- pack pipeline follows `turboquant_pack_kv_for_cache_v2_to_cache`: linear row
  batching, normalize, KFC rotate matmul, encode, slot-mapped write-out

Algorithm per vector:

1. `norm = ||x||`
2. `n = x / (norm + eps)`
3. `y = n @ R^T`
4. `sign = y >= 0`
5. `err = y - (+/- 1/sqrt(128))`
6. `base = min(err)`, `step = (max(err) - base) / 127`
7. `q7 = round((err - base) / step)`
8. code byte per dim: `(q7 << 1) | sign`, with sign in the lowest bit

Cache layout inside each `(block, head)` pageblock:

```text
[ block_size * 128 code bytes ][ block_size fp16 norms ]
[ block_size fp16 bases ][ block_size fp16 steps ]
```

The first implementation intentionally avoids the grouped 4-bit slab memory
management used by `turboquant_pack_kv_for_cache4bit`.
