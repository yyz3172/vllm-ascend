"""BitResidual K8V4 batch-tier guard test.

Run on an NPU machine after building ``bit_residual_pack_k8v4`` (and the
``bit_residual_attention_paged_k8v4`` decode op):

    python tests/e2e/singlecard/xrx_bit_residual_k8v4_batch_tiers.py

This test guards the batch-tier tiling renovation (encode batch M grows from
16 to 16/32/48/64). It does NOT assume a particular NPU model: cases that
exercise cross-tier byte-for-byte equality (Case A) are skipped until the host
select-BATCH-M logic and the ``BR_K8V4_FORCE_BATCH_M`` knob land; the layout
and decode-transparency cases run on any tier and act as a regression baseline
so the renovation cannot silently corrupt the cache.

Cache physical layout (ground truth, no 16-row sub-block):
  A page (block) holds ``blockSize`` tokens; ``blockSize % 16 == 0``.
  Within a page, all tokens of one head are contiguous:
    key   head region = [blockSize rows of code (128B each)]
                         [blockSize bases (2B each)]
                         [blockSize steps (2B each)]
    value head region = [blockSize rows of code (64B each)]
                         [blockSize vmins (2B each)]
                         [blockSize vsteps (2B each)]
  Multiple heads are laid out sequentially inside the page; a block larger
  than one page is split across pages (block_table lookup).

Cases:
  A. Cross-tier byte-for-byte equality (skip until tiers exist).
  B. Per-head contiguous layout + QDQ round-trip + range/sign validity,
     across blockSizes 16/32/64 and aligned/unaligned slot offsets, including
     cross-page (token_count > blockSize) to exercise block_table addressing.
  C. Read-modify-write correctness for base/step (token-shared scalars) under
     overlapping writes, within the active tier.
  D. End-to-end decode transparency: each tier's cache feeds
     ``bit_residual_attention_paged_k8v4`` and matches the golden attention.
  E. Dual-AIV dispatch: numHeads>=2 so AIV#0 / AIV#1 each get a head slice;
     the two heads' caches stay independent and correct.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Set ASCEND_CUSTOM_OPP_PATH before importing torch/torch_npu. Otherwise CANN
# may only search the system opapi library and miss this repo's custom ops.
_CANN_OPP = REPO_ROOT / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "vllm-ascend"
if _CANN_OPP.is_dir():
    existing_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        str(_CANN_OPP) if not existing_opp else f"{_CANN_OPP}:{existing_opp}"
    )

import torch

from vllm_ascend.utils import enable_custom_op

HEAD_SIZE = 128
# Per-row byte costs (match kernel + golden test).
KEY_ROW_CODE_BYTES = HEAD_SIZE
VALUE_ROW_CODE_BYTES = HEAD_SIZE // 2
ROW_META_BYTES = 2  # base/step/vmin/vstep are uint16 scalars
KEY_BYTES_PER_ROW = KEY_ROW_CODE_BYTES + 2 * ROW_META_BYTES
VALUE_BYTES_PER_ROW = VALUE_ROW_CODE_BYTES + 2 * ROW_META_BYTES

SEED = 2026

# Batch tiers the renovation will expose. Currently only 16 exists; the rest
# become reachable once host SelectBatchM + BR_K8V4_FORCE_BATCH_M land.
ALL_TIERS = (16, 32, 48, 64)


def _active_tiers() -> list[int]:
    """Tiers the test should attempt to exercise.

    Honors BR_K8V4_FORCE_BATCH_M if the host knob exists; otherwise reports
    only the implemented 16 tier (cross-tier Case A auto-skips).
    """
    forced = os.getenv("BR_K8V4_FORCE_BATCH_M", "")
    if forced:
        try:
            m = int(forced)
        except ValueError:
            raise RuntimeError(f"BR_K8V4_FORCE_BATCH_M must be int, got {forced!r}")
        return [m]
    return [16]


def _require_ops() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    if not enable_custom_op():
        raise RuntimeError("vllm_ascend_C is not loaded; build/install custom ops first")
    if not hasattr(torch.ops, "_C_ascend"):
        raise RuntimeError("torch.ops._C_ascend is not registered")
    for op_name in ("bit_residual_pack_k8v4", "bit_residual_attention_paged_k8v4"):
        if not hasattr(torch.ops._C_ascend, op_name):
            raise RuntimeError(f"torch.ops._C_ascend.{op_name} is not registered")


def _key_head_stride(block_size: int) -> int:
    return block_size * KEY_BYTES_PER_ROW


def _value_head_stride(block_size: int) -> int:
    return block_size * VALUE_BYTES_PER_ROW


def _build_block_table(actual_seq_lens_kv: list[int]) -> tuple[torch.Tensor, int]:
    blocks_per_seq = [
        max(1, (seq_len + 16 - 1) // 16) for seq_len in actual_seq_lens_kv
    ]
    max_blocks = max(blocks_per_seq)
    block_table = torch.zeros((len(actual_seq_lens_kv), max_blocks), dtype=torch.int32)
    next_block = 0
    for seq_idx, block_count in enumerate(blocks_per_seq):
        block_table[seq_idx, :block_count] = torch.arange(
            next_block, next_block + block_count, dtype=torch.int32
        )
        next_block += block_count
    return block_table, next_block


def _make_rotation(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Fixed dense orthogonal R^T the kernel consumes directly (mirrors golden)."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 17)
    matrix = torch.randn((HEAD_SIZE, HEAD_SIZE), dtype=torch.float32, generator=generator)
    rotation, _ = torch.linalg.qr(matrix)
    return rotation.to(device=device, dtype=dtype).contiguous()


def _u8_to_dtype_float(x: torch.Tensor, dtype: torch.dtype) -> float:
    return x.contiguous().view(dtype)[0].float().item()


def _unpack_value_idx4(packed: torch.Tensor) -> torch.Tensor:
    code = packed.to(torch.int32)
    low = code & 0x0F
    high = (code >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low) + 8
    high = torch.where(high >= 8, high - 16, high) + 8
    idx4 = torch.empty(HEAD_SIZE, dtype=torch.uint8)
    idx4[0::2] = low.to(torch.uint8)
    idx4[1::2] = high.to(torch.uint8)
    return idx4


def _decode_key_cache(
    cache: torch.Tensor, slots: torch.Tensor, block_size: int, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Decode key cache using the per-head-per-block contiguous layout."""
    cache_cpu = cache.cpu()
    slots_cpu = slots.cpu().to(torch.int64)
    t = slots_cpu.numel()
    h = cache_cpu.shape[1]
    code = torch.empty(t, h, HEAD_SIZE, dtype=torch.uint8)
    base = torch.empty(t, h, dtype=torch.float32)
    step = torch.empty(t, h, dtype=torch.float32)
    head_stride = _key_head_stride(block_size)
    for token_idx, slot in enumerate(slots_cpu.tolist()):
        block_idx = slot // block_size
        pos_in_block = slot % block_size
        for head in range(h):
            slab = cache_cpu[block_idx, head]
            code_off = pos_in_block * KEY_ROW_CODE_BYTES
            code[token_idx, head] = slab[code_off : code_off + HEAD_SIZE]
            base_off = block_size * KEY_ROW_CODE_BYTES + pos_in_block * ROW_META_BYTES
            step_off = block_size * (KEY_ROW_CODE_BYTES + ROW_META_BYTES) + pos_in_block * ROW_META_BYTES
            base[token_idx, head] = _u8_to_dtype_float(
                slab[base_off : base_off + ROW_META_BYTES], dtype
            )
            step[token_idx, head] = _u8_to_dtype_float(
                slab[step_off : step_off + ROW_META_BYTES], dtype
            )
    return {"code": code, "base": base, "step": step}


def _decode_value_cache(
    cache: torch.Tensor, slots: torch.Tensor, block_size: int, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Decode value cache using the per-head-per-block contiguous layout."""
    cache_cpu = cache.cpu()
    slots_cpu = slots.cpu().to(torch.int64)
    t = slots_cpu.numel()
    h = cache_cpu.shape[1]
    idx4 = torch.empty(t, h, HEAD_SIZE, dtype=torch.uint8)
    vmin = torch.empty(t, h, dtype=torch.float32)
    vstep = torch.empty(t, h, dtype=torch.float32)
    for token_idx, slot in enumerate(slots_cpu.tolist()):
        block_idx = slot // block_size
        pos_in_block = slot % block_size
        for head in range(h):
            slab = cache_cpu[block_idx, head]
            code_off = pos_in_block * VALUE_ROW_CODE_BYTES
            idx4[token_idx, head] = _unpack_value_idx4(
                slab[code_off : code_off + VALUE_ROW_CODE_BYTES]
            )
            vmin_off = block_size * VALUE_ROW_CODE_BYTES + pos_in_block * ROW_META_BYTES
            vstep_off = block_size * (VALUE_ROW_CODE_BYTES + ROW_META_BYTES) + pos_in_block * ROW_META_BYTES
            vmin[token_idx, head] = _u8_to_dtype_float(
                slab[vmin_off : vmin_off + ROW_META_BYTES], dtype
            )
            vstep[token_idx, head] = _u8_to_dtype_float(
                slab[vstep_off : vstep_off + ROW_META_BYTES], dtype
            )
    return {"idx4": idx4, "vmin": vmin, "vstep": vstep}


def _assert_qdq_and_ranges(
    name: str,
    dtype: torch.dtype,
    key_dec: dict[str, torch.Tensor],
    value_dec: dict[str, torch.Tensor],
) -> None:
    """QDQ round-trip exactness + sign/range validity (mirrors key1_unaligned)."""
    # Key QDQ: err_recon = base + q7*step; (err-base)/step == q7.
    q7 = (key_dec["code"].to(torch.int32) & 0x7F).to(torch.float32)
    err_recon = key_dec["base"].unsqueeze(-1) + q7 * key_dec["step"].unsqueeze(-1)
    step_valid = key_dec["step"] > 1e-6
    if step_valid.any():
        q7_rt = (
            (err_recon - key_dec["base"].unsqueeze(-1))
            / key_dec["step"].unsqueeze(-1).clamp(min=1e-6)
        ).nan_to_num(0.0)
        q7_err = (q7_rt - q7).abs()
        max_q7_err = q7_err[step_valid.unsqueeze(-1).expand_as(q7_err)].max().item()
        assert max_q7_err < 0.5, f"{name}: key QDQ violated, max dev {max_q7_err:.4f}"

    # Value QDQ: y = vmin + idx4*vstep; (y-vmin)/vstep == idx4.
    idx4 = value_dec["idx4"].to(torch.float32)
    y_recon = value_dec["vmin"].unsqueeze(-1) + idx4 * value_dec["vstep"].unsqueeze(-1)
    vstep_valid = value_dec["vstep"] > 1e-6
    if vstep_valid.any():
        idx4_rt = (
            (y_recon - value_dec["vmin"].unsqueeze(-1))
            / value_dec["vstep"].unsqueeze(-1).clamp(min=1e-6)
        ).nan_to_num(0.0)
        idx4_err = (idx4_rt - idx4).abs()
        max_idx4_err = idx4_err[vstep_valid.unsqueeze(-1).expand_as(idx4_err)].max().item()
        assert max_idx4_err < 0.5, f"{name}: value QDQ violated, max dev {max_idx4_err:.4f}"

    # Sign-bit consistency: code == (code&0x7F) | (sign<<7).
    q7_int = key_dec["code"].to(torch.int32) & 0x7F
    sign_int = key_dec["code"].to(torch.int32) >> 7
    assert (
        (q7_int | (sign_int << 7)) == key_dec["code"].to(torch.int32)
    ).all(), f"{name}: sign-bit consistency violated"

    # Range validity.
    assert (q7_int >= 0).all() and (q7_int <= 127).all(), f"{name}: q7 out of [0,127]"
    assert (value_dec["idx4"] <= 15).all(), f"{name}: idx4 out of [0,15]"
    assert (key_dec["step"] >= 0).all(), f"{name}: negative key step"
    assert (value_dec["vstep"] >= 0).all(), f"{name}: negative value vstep"

    # Non-degenerate: most groups must have a real range.
    key_ratio = step_valid.sum().item() / key_dec["step"].numel()
    val_ratio = vstep_valid.sum().item() / value_dec["vstep"].numel()
    assert key_ratio >= 0.5, f"{name}: too many degenerate key groups ({key_ratio:.2f})"
    assert val_ratio >= 0.5, f"{name}: too many degenerate value groups ({val_ratio:.2f})"

    # Sign balance for random input.
    sign1_frac = (sign_int == 1).sum().item() / sign_int.numel()
    assert 0.25 <= sign1_frac <= 0.75, f"{name}: sign skew {sign1_frac:.3f}"
    print(
        f"PASS {name}: dtype={dtype}, qdq=OK, sign=OK({sign1_frac:.2f}), "
        f"nondeg=OK(key={key_ratio:.2f},val={val_ratio:.2f})"
    )


def _alloc_caches(
    num_blocks: int, num_heads: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache = torch.zeros(
        num_blocks, num_heads, _key_head_stride(block_size), dtype=torch.uint8, device=device
    )
    value_cache = torch.zeros(
        num_blocks,
        num_heads,
        _value_head_stride(block_size),
        dtype=torch.uint8,
        device=device,
    )
    return key_cache, value_cache


def _run_pack(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    query_start_loc: torch.Tensor,
    rotation_t: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_reqs: int,
    block_size: int,
) -> None:
    torch.ops._C_ascend.bit_residual_pack_k8v4(
        key, value, slots, query_start_loc, rotation_t,
        key_cache, value_cache, num_reqs, block_size,
    )
    torch.npu.synchronize()


# ---------------------------------------------------------------------------
# Case B: per-head layout + QDQ across blockSizes / offsets / cross-page.
# ---------------------------------------------------------------------------
SHORT_OFFSETS = (0, 1, 2, 3, 4, 5)
BLOCK_SIZES = (16, 32, 64)


def _run_case_b(
    name: str,
    token_count: int,
    slot_offset: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    num_heads = 8
    torch.manual_seed(SEED + token_count + slot_offset + block_size)
    key = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots = torch.arange(
        slot_offset, slot_offset + token_count, dtype=torch.int32, device=device
    )
    qsl = torch.tensor([0, token_count], dtype=torch.int32, device=device)
    num_blocks = (slot_offset + token_count + block_size - 1) // block_size + 1
    key_cache, value_cache = _alloc_caches(num_blocks, num_heads, block_size, device)
    _run_pack(
        key=key, value=value, slots=slots, query_start_loc=qsl, rotation_t=rotation_t,
        key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
    )
    key_dec = _decode_key_cache(key_cache, slots, block_size, dtype)
    value_dec = _decode_value_cache(value_cache, slots, block_size, dtype)
    _assert_qdq_and_ranges(name, dtype, key_dec, value_dec)


def _run_cross_page(dtype: torch.dtype, device: torch.device) -> None:
    """token_count > block_size forces block_table multi-page addressing."""
    block_size = 16
    num_heads = 8
    token_count = 40  # spans 3 pages of 16
    torch.manual_seed(SEED + 7777)
    key = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots = torch.arange(token_count, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, token_count], dtype=torch.int32, device=device)
    num_blocks = (token_count + block_size - 1) // block_size + 1
    key_cache, value_cache = _alloc_caches(num_blocks, num_heads, block_size, device)
    _run_pack(
        key=key, value=value, slots=slots, query_start_loc=qsl, rotation_t=rotation_t,
        key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
    )
    key_dec = _decode_key_cache(key_cache, slots, block_size, dtype)
    value_dec = _decode_value_cache(value_cache, slots, block_size, dtype)
    _assert_qdq_and_ranges("cross_page", dtype, key_dec, value_dec)


# ---------------------------------------------------------------------------
# Case C: base/step RMW correctness under overlapping writes (active tier).
# ---------------------------------------------------------------------------
def _run_case_c(dtype: torch.dtype, device: torch.device, block_size: int) -> None:
    num_heads = 8
    torch.manual_seed(SEED + 2000)
    t1, off1 = 10, 1
    key1 = torch.randn(t1, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value1 = torch.randn(t1, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots1 = torch.arange(off1, off1 + t1, dtype=torch.int32, device=device)
    qsl1 = torch.tensor([0, t1], dtype=torch.int32, device=device)
    t2, off2 = 10, 4
    key2 = torch.randn(t2, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value2 = torch.randn(t2, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    slots2 = torch.arange(off2, off2 + t2, dtype=torch.int32, device=device)
    qsl2 = torch.tensor([0, t2], dtype=torch.int32, device=device)
    num_blocks = 2
    key_cache, value_cache = _alloc_caches(num_blocks, num_heads, block_size, device)

    _run_pack(
        key=key1, value=value1, slots=slots1, query_start_loc=qsl1, rotation_t=rotation_t,
        key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
    )
    preserved = torch.arange(off1, off1 + 3, dtype=torch.int32, device=device)
    key_pres_1 = _decode_key_cache(key_cache, preserved, block_size, dtype)
    val_pres_1 = _decode_value_cache(value_cache, preserved, block_size, dtype)

    _run_pack(
        key=key2, value=value2, slots=slots2, query_start_loc=qsl2, rotation_t=rotation_t,
        key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
    )
    key_pres_2 = _decode_key_cache(key_cache, preserved, block_size, dtype)
    val_pres_2 = _decode_value_cache(value_cache, preserved, block_size, dtype)
    for field in ("code", "base", "step"):
        torch.testing.assert_close(key_pres_1[field], key_pres_2[field], rtol=0, atol=0)
    for field in ("idx4", "vmin", "vstep"):
        torch.testing.assert_close(val_pres_1[field], val_pres_2[field], rtol=0, atol=0)

    key_p2 = _decode_key_cache(key_cache, slots2, block_size, dtype)
    val_p2 = _decode_value_cache(value_cache, slots2, block_size, dtype)
    _assert_qdq_and_ranges(f"rmw_bs{block_size}_phase2", dtype, key_p2, val_p2)
    print(f"PASS rmw_bs{block_size}: preserved slots unchanged, phase2 correct, dtype={dtype}")


# ---------------------------------------------------------------------------
# Case D: decode transparency — feed cache to attention, match golden.
# ---------------------------------------------------------------------------
def _decode_key_row(cache, block_table, seq_idx, kv_head, abs_pos, block_size, dtype):
    block_id = int(block_table[seq_idx, abs_pos // block_size])
    pos_in_block = abs_pos % block_size
    block = cache[block_id, kv_head]
    code = block[pos_in_block * KEY_ROW_CODE_BYTES : pos_in_block * KEY_ROW_CODE_BYTES + HEAD_SIZE].to(torch.int32)
    q7 = (code & 0x7F).float()
    sign = torch.where((code >> 7) == 0, 1.0, -1.0)
    base = _u8_to_dtype_float(
        block[block_size * KEY_ROW_CODE_BYTES + pos_in_block * ROW_META_BYTES :
              block_size * KEY_ROW_CODE_BYTES + pos_in_block * ROW_META_BYTES + ROW_META_BYTES], dtype)
    step = _u8_to_dtype_float(
        block[block_size * (KEY_ROW_CODE_BYTES + ROW_META_BYTES) + pos_in_block * ROW_META_BYTES :
              block_size * (KEY_ROW_CODE_BYTES + ROW_META_BYTES) + pos_in_block * ROW_META_BYTES + ROW_META_BYTES], dtype)
    return (base + q7 * step) * sign


def _unpack_value_idx4_code(code: torch.Tensor) -> torch.Tensor:
    low = code & 0x0F
    high = (code >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low) + 8
    high = torch.where(high >= 8, high - 16, high) + 8
    idx4 = torch.empty(HEAD_SIZE, dtype=torch.float32)
    idx4[0::2] = low.float()
    idx4[1::2] = high.float()
    return idx4


def _decode_value_row(cache, block_table, seq_idx, kv_head, abs_pos, block_size, dtype):
    block_id = int(block_table[seq_idx, abs_pos // block_size])
    pos_in_block = abs_pos % block_size
    block = cache[block_id, kv_head]
    code = block[pos_in_block * VALUE_ROW_CODE_BYTES :
                 pos_in_block * VALUE_ROW_CODE_BYTES + VALUE_ROW_CODE_BYTES].to(torch.int32)
    idx4 = _unpack_value_idx4_code(code)
    vmin = _u8_to_dtype_float(
        block[block_size * VALUE_ROW_CODE_BYTES + pos_in_block * ROW_META_BYTES :
              block_size * VALUE_ROW_CODE_BYTES + pos_in_block * ROW_META_BYTES + ROW_META_BYTES], dtype)
    vstep = _u8_to_dtype_float(
        block[block_size * (VALUE_ROW_CODE_BYTES + ROW_META_BYTES) + pos_in_block * ROW_META_BYTES :
              block_size * (VALUE_ROW_CODE_BYTES + ROW_META_BYTES) + pos_in_block * ROW_META_BYTES + ROW_META_BYTES], dtype)
    return vmin + idx4 * vstep


def _golden_attention(
    *, query, key_cache, value_cache, block_table, block_size,
    actual_seq_lens_q, actual_seq_lens_kv, rotation_key, rotation_value,
    num_heads, num_kv_heads, scale,
):
    dtype = query.dtype
    gqa = num_heads // num_kv_heads
    out = torch.zeros_like(query, dtype=torch.float32)
    for ti in range(query.shape[0]):
        seq_idx = next(i for i, e in enumerate(actual_seq_lens_q) if ti < e)
        q_start = 0 if seq_idx == 0 else actual_seq_lens_q[seq_idx - 1]
        nq = actual_seq_lens_q[seq_idx] - q_start
        qp = ti - q_start
        ce = actual_seq_lens_kv[seq_idx] - nq + qp + 1
        ce = max(0, min(ce, actual_seq_lens_kv[seq_idx]))
        if ce == 0:
            continue
        for kh in range(num_kv_heads):
            keys = torch.stack(
                [_decode_key_row(key_cache, block_table, seq_idx, kh, p, block_size, dtype)
                 for p in range(ce)], dim=0)
            vals = torch.stack(
                [_decode_value_row(value_cache, block_table, seq_idx, kh, p, block_size, dtype)
                 for p in range(ce)], dim=0)
            for gi in range(gqa):
                hi = kh * gqa + gi
                qr = query[ti, hi].float() @ rotation_key.float()
                scores = (keys * qr).sum(-1) * scale
                attn = torch.softmax(scores, -1) @ vals
                # Post-rotate: out = attn @ R (matches kernel's out @ rotation_value).
                out[ti, hi] = (attn.to(dtype).float() @ rotation_value.float()).to(dtype).float()
    return out


def _run_attention_op(
    *, query, key_cache, value_cache, block_table, block_size,
    actual_seq_lens_q, actual_seq_lens_kv, rotation_key, rotation_value,
    num_heads, num_kv_heads, scale,
):
    return torch.ops._C_ascend.bit_residual_attention_paged_k8v4(
        query.contiguous(), key_cache.contiguous(), value_cache.contiguous(),
        block_table.contiguous(), actual_seq_lens_q, actual_seq_lens_kv,
        rotation_key.contiguous(), rotation_value.contiguous(),
        num_heads, num_kv_heads, HEAD_SIZE, block_size,
        max(1, max(actual_seq_lens_kv)), float(scale),
    )


def _run_case_d(dtype: torch.dtype, device: torch.device, block_size: int) -> None:
    torch.manual_seed(SEED + 9000 + block_size)
    num_kv_tokens = 20  # > block_size when block_size=16 → cross-page
    num_query_tokens = 3
    num_kv_heads = 8
    num_heads = 2 * num_kv_heads
    scale = HEAD_SIZE ** -0.5
    block_table, num_blocks = _build_block_table([num_kv_tokens])
    key = torch.randn(num_kv_tokens, num_kv_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value = torch.randn(num_kv_tokens, num_kv_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    query = torch.randn(num_query_tokens, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    rotation = rotation_t.transpose(0, 1).contiguous()
    slots = torch.arange(num_kv_tokens, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, num_kv_tokens], dtype=torch.int32, device=device)
    key_cache, value_cache = _alloc_caches(num_blocks, num_kv_heads, block_size, device)
    _run_pack(
        key=key, value=value, slots=slots, query_start_loc=qsl, rotation_t=rotation_t,
        key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
    )
    actual = _run_attention_op(
        query=query, key_cache=key_cache, value_cache=value_cache, block_table=block_table.to(device),
        block_size=block_size, actual_seq_lens_q=[num_query_tokens],
        actual_seq_lens_kv=[num_kv_tokens], rotation_key=rotation_t, rotation_value=rotation,
        num_heads=num_heads, num_kv_heads=num_kv_heads, scale=scale,
    )
    torch.npu.synchronize()
    expected = _golden_attention(
        query=query.cpu(), key_cache=key_cache.cpu(), value_cache=value_cache.cpu(),
        block_table=block_table, block_size=block_size,
        actual_seq_lens_q=[num_query_tokens], actual_seq_lens_kv=[num_kv_tokens],
        rotation_key=rotation_t.cpu(), rotation_value=rotation.cpu(),
        num_heads=num_heads, num_kv_heads=num_kv_heads, scale=scale,
    )
    actual_cpu = actual.float().cpu()
    atol = 1e-3 if dtype == torch.float16 else 4e-2
    torch.testing.assert_close(actual_cpu, expected.float(), atol=atol, rtol=atol)
    print(
        f"PASS decode_transparency_bs{block_size}: dtype={dtype}, "
        f"maxdiff={(actual_cpu - expected).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Case E: dual-AIV head dispatch (num_heads>=2, two head slices).
# ---------------------------------------------------------------------------
def _run_case_e(dtype: torch.dtype, device: torch.device, block_size: int) -> None:
    num_heads = 4  # >= HEADS_PER_TILE(=2) so AIV#0/AIV#1 split heads
    torch.manual_seed(SEED + 5000 + block_size)
    token_count = 9
    key = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots = torch.arange(token_count, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, token_count], dtype=torch.int32, device=device)
    num_blocks = (token_count + block_size - 1) // block_size + 1
    key_cache, value_cache = _alloc_caches(num_blocks, num_heads, block_size, device)
    _run_pack(
        key=key, value=value, slots=slots, query_start_loc=qsl, rotation_t=rotation_t,
        key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
    )
    key_dec = _decode_key_cache(key_cache, slots, block_size, dtype)
    value_dec = _decode_value_cache(value_cache, slots, block_size, dtype)
    _assert_qdq_and_ranges(f"dual_aiv_bs{block_size}", dtype, key_dec, value_dec)


# ---------------------------------------------------------------------------
# Case A: cross-tier byte-for-byte equality (skipped until tiers exist).
# ---------------------------------------------------------------------------
def _run_case_a(dtype: torch.dtype, device: torch.device) -> None:
    tiers = _active_tiers()
    if len(tiers) < 2 and 16 in tiers and not os.getenv("BR_K8V4_FORCE_BATCH_M"):
        print(
            "SKIP case_a (cross-tier equality): only tier 16 implemented; "
            "set BR_K8V4_FORCE_BATCH_M once the select-batch-M host knob lands."
        )
        return
    num_heads = 8
    token_count = 32
    torch.manual_seed(SEED + 31)
    key = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots = torch.arange(token_count, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, token_count], dtype=torch.int32, device=device)
    block_size = 16
    num_blocks = (token_count + block_size - 1) // block_size + 1
    ref_key = ref_val = None
    for m in tiers:
        key_cache, value_cache = _alloc_caches(num_blocks, num_heads, block_size, device)
        _run_pack(
            key=key, value=value, slots=slots, query_start_loc=qsl, rotation_t=rotation_t,
            key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
        )
        if ref_key is None:
            ref_key, ref_val = key_cache.cpu(), value_cache.cpu()
        else:
            assert torch.equal(key_cache.cpu(), ref_key), (
                f"case_a: key cache differs across tiers (tier {m})"
            )
            assert torch.equal(value_cache.cpu(), ref_val), (
                f"case_a: value cache differs across tiers (tier {m})"
            )
    print(f"PASS case_a: cross-tier byte-for-byte equality over tiers {tiers}, dtype={dtype}")


def main() -> None:
    _require_ops()
    device = torch.device("npu:0")
    for dtype in (torch.float16, torch.bfloat16):
        for bs in BLOCK_SIZES:
            for off in SHORT_OFFSETS:
                _run_case_b(f"layout_bs{bs}_off{off}", 9, off, bs, dtype, device)
            _run_cross_page(dtype, device)
            _run_case_c(dtype, device, bs)
            _run_case_d(dtype, device, bs)
            _run_case_e(dtype, device, bs)
        _run_case_a(dtype, device)
    print("bit residual k8v4 batch-tier guard: all cases passed")


if __name__ == "__main__":
    main()
