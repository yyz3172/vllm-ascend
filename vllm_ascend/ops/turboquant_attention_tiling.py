#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-side mirror of TurboQuant Scheme B host tiling (SplitBN / SplitBNS)."""

from __future__ import annotations

FLASH_DECODE_BN_RATIO = 0.5
FLASH_DECODE_MIN_KV_LEN = 2048
FLASH_DECODE_MIN_SEGMENT = 512
TQ_ATTN_KV_TILE_ROWS = 32
TQ_ATTN_UB_KV_TILE_CAP = 32
TQ_ATTN_UB_GQA_CAP = 8
TQ_ATTN_CUBE_MIN_G = 8
TQ_ATTN_CUBE_MIN_TILE = 32
TQ_ATTN_SPLIT_BN = 0
TQ_ATTN_SPLIT_BNS = 1
TQ_ATTN_QKPV_VECTOR = 0
TQ_ATTN_QKPV_CUBE = 1


def _align_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def _align_down(x: int, align: int) -> int:
    return (x // align) * align


def pick_kv_tile_rows(block_size: int) -> int:
    tile = min(TQ_ATTN_KV_TILE_ROWS, TQ_ATTN_UB_KV_TILE_CAP)
    if block_size == 0:
        return tile
    if block_size <= tile and tile % block_size != 0:
        tile = _align_down(tile, block_size)
        if tile == 0:
            tile = block_size
    tile = min(tile, TQ_ATTN_UB_KV_TILE_CAP)
    cube_min = min(TQ_ATTN_CUBE_MIN_TILE, TQ_ATTN_UB_KV_TILE_CAP)
    return max(tile, cube_min)


def is_flash_decode(num_tokens: int, num_kv_heads: int, max_kv_len: int, aic_num: int) -> bool:
    bn = num_tokens * num_kv_heads
    if max_kv_len < FLASH_DECODE_MIN_KV_LEN:
        return False
    if bn > int(FLASH_DECODE_BN_RATIO * aic_num):
        return False
    return True


def pick_qk_pv_mode(gqa_group: int, kv_tile_rows: int) -> int:
    if gqa_group >= TQ_ATTN_CUBE_MIN_G and kv_tile_rows >= TQ_ATTN_CUBE_MIN_TILE:
        return TQ_ATTN_QKPV_CUBE
    return TQ_ATTN_QKPV_VECTOR


def split_bn(bn: int, core_num: int) -> tuple[int, list[int]]:
    used = min(bn, core_num) if bn > 0 else 0
    if used == 0:
        return 0, []
    former = bn % used
    if former == 0:
        block_range = bn // used
        tail_range = block_range
    else:
        block_range = bn // used + 1
        tail_range = block_range - 1
    starts = []
    former_base = former * block_range
    for i in range(used):
        if i < former:
            starts.append(block_range * i)
        else:
            starts.append(former_base + tail_range * (i - former))
    return used, starts


def split_bns(
    bn: int, max_kv_len: int, block_size: int, core_num: int
) -> tuple[int, int, int, list[int]]:
    kv_split = core_num // bn if bn > 0 else 1
    if kv_split == 0:
        kv_split = 1
    while (max_kv_len // kv_split) < FLASH_DECODE_MIN_SEGMENT and kv_split > 1:
        kv_split -= 1
    used = bn * kv_split
    seg_len = _align_up((max_kv_len + kv_split - 1) // kv_split, block_size) if block_size else (
        (max_kv_len + kv_split - 1) // kv_split
    )
    starts = list(range(used))
    return used, kv_split, seg_len, starts


def pick_tiling_key(split_mode: int, qk_pv_mode: int) -> int:
    if split_mode == TQ_ATTN_SPLIT_BN:
        return 1 if qk_pv_mode == TQ_ATTN_QKPV_CUBE else 0
    return 3 if qk_pv_mode == TQ_ATTN_QKPV_CUBE else 2


def compute_turboquant_attention_tiling(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    block_size: int,
    max_kv_len: int,
    aic_num: int = 24,
    max_actual_seq_len: int | None = None,
) -> dict:
    """Return host tiling decisions for Scheme B attention."""
    if max_actual_seq_len is None:
        max_actual_seq_len = max_kv_len
    gqa = num_heads // num_kv_heads
    bn = num_tokens * num_kv_heads
    kv_tile = pick_kv_tile_rows(block_size)
    split_mode = (
        TQ_ATTN_SPLIT_BNS
        if is_flash_decode(num_tokens, num_kv_heads, max_actual_seq_len, aic_num)
        else TQ_ATTN_SPLIT_BN
    )
    qk_pv = pick_qk_pv_mode(gqa, kv_tile)
    if split_mode == TQ_ATTN_SPLIT_BNS:
        used, kv_split, seg_len, starts = split_bns(
            bn, max_actual_seq_len, block_size, aic_num
        )
    else:
        used, starts = split_bn(bn, aic_num)
        kv_split = 1
        seg_len = max_actual_seq_len
    return {
        "num_tokens": num_tokens,
        "gqa_group": gqa,
        "bn": bn,
        "max_kv_len": max_kv_len,
        "max_actual_seq_len": max_actual_seq_len,
        "split_mode": split_mode,
        "qk_pv_mode": qk_pv,
        "kv_tile_rows": kv_tile,
        "used_core_num": used,
        "kv_split_part": kv_split,
        "kv_segment_len": seg_len,
        "core_task_start": starts,
        "tiling_key": pick_tiling_key(split_mode, qk_pv),
    }
