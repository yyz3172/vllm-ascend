#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Non-contiguous physical block_table pack regression.
#
# Reclaim often hands out non-monotonic block ids (e.g. [2, 0]). Pack must
# resolve each chunk via slot_mapping[token], NOT firstSlot+rowOff.
#
# Bug (pre-fix): firstSlot+rowOff writes token 128.. to phys block
#   firstSlot/bs+1 (here block 3) while FIA reads block_table[1]=0.
# Fix: ResolvePackChunk re-reads slot_mapping at each PA-block boundary.
#
# Run on NPU after building bit_residual_pack_k8v4:
#   python tests/e2e/singlecard/xrx_bit_residual_k8v4_noncontig_blocks.py
#
# Optional FIA check (needs bit_residual_fia_paged_k8v4):
#   python .../xrx_bit_residual_k8v4_noncontig_blocks.py --with-fia

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

_CANN_OPP = REPO_ROOT / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "vllm-ascend"
if _CANN_OPP.is_dir():
    existing_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        str(_CANN_OPP) if not existing_opp else f"{_CANN_OPP}:{existing_opp}"
    )

import torch

from vllm_ascend.utils import enable_custom_op

HEAD_SIZE = 128
KEY_ROW_CODE_BYTES = 128
VAL_ROW_CODE_BYTES = 64
BLOCK_SIZE = 128
NUM_KV_HEADS = 2
NUM_TOKENS = 200  # 128 + 72 → two PA blocks
# Reclaim-style non-monotonic ids (LIFO free list).
PHYS_BLOCKS = (2, 0)
WRONG_DEST_BLOCK = PHYS_BLOCKS[0] + 1  # firstSlot+128 lands here under the bug
PAINT = 0xA5
SEED = 20260801
SCALE = HEAD_SIZE**-0.5
COMPRESS_MASK_SIZE = 2048
SPARSE_MODE_RIGHT_DOWN = 3
INT_MAX = 2147483647


def _key_stride(bs: int = BLOCK_SIZE) -> int:
    return bs * (KEY_ROW_CODE_BYTES + 4)


def _val_stride(bs: int = BLOCK_SIZE) -> int:
    return bs * (VAL_ROW_CODE_BYTES + 4)


def _require_npu(*, need_fia: bool) -> torch.device:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    if not enable_custom_op():
        raise RuntimeError("vllm_ascend_C is not loaded; build/install custom ops first")
    ops = ["bit_residual_pack_k8v4"]
    if need_fia:
        ops.append("bit_residual_fia_paged_k8v4")
    for op_name in ops:
        if not hasattr(torch.ops._C_ascend, op_name):
            raise RuntimeError(f"missing custom op: {op_name}")
    return torch.device("npu")


def _identity(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.eye(HEAD_SIZE, dtype=dtype, device=device).contiguous()


def _slots_for_phys(phys: tuple[int, ...], num_tokens: int, bs: int) -> torch.Tensor:
    """Build slot_mapping for a paged sequence with the given physical blocks."""
    slots: list[int] = []
    for t in range(num_tokens):
        bid = phys[t // bs]
        slots.append(bid * bs + (t % bs))
    return torch.tensor(slots, dtype=torch.int32)


def _extract_used_key(head: torch.Tensor, used: int, bs: int = BLOCK_SIZE) -> torch.Tensor:
    codes = head[: used * KEY_ROW_CODE_BYTES]
    base_off = bs * KEY_ROW_CODE_BYTES
    step_off = base_off + bs * 2
    return torch.cat(
        [
            codes,
            head[base_off : base_off + used * 2],
            head[step_off : step_off + used * 2],
        ]
    )


def _extract_used_val(head: torch.Tensor, used: int, bs: int = BLOCK_SIZE) -> torch.Tensor:
    codes = head[: used * VAL_ROW_CODE_BYTES]
    vmin_off = bs * VAL_ROW_CODE_BYTES
    vstep_off = vmin_off + bs * 2
    return torch.cat(
        [
            codes,
            head[vmin_off : vmin_off + used * 2],
            head[vstep_off : vstep_off + used * 2],
        ]
    )


def _pack(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    rotation_t: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> None:
    n = int(key.shape[0])
    qsl = torch.tensor([0, n], dtype=torch.int32, device=key.device)
    torch.ops._C_ascend.bit_residual_pack_k8v4(
        key,
        value,
        slots.to(device=key.device, dtype=torch.int32).contiguous(),
        qsl,
        rotation_t.contiguous(),
        key_cache,
        value_cache,
        1,
        block_size,
    )
    torch.npu.synchronize()


def _compress_causal_mask(device: torch.device) -> torch.Tensor:
    ones = torch.ones(
        (COMPRESS_MASK_SIZE, COMPRESS_MASK_SIZE), dtype=torch.int8, device=device
    )
    return torch.triu(ones, diagonal=1)


def test_pack_noncontig_phys_blocks_match_contig_ref(
    device: torch.device, *, with_fia: bool
) -> None:
    """Pack with phys=[2,0] must land on those blocks (same bytes as contig ref)."""
    dtype = torch.float16
    bs = BLOCK_SIZE
    n = NUM_TOKENS
    used1 = n - bs  # 72
    assert used1 > 0
    torch.manual_seed(SEED)
    key = torch.randn(n, NUM_KV_HEADS, HEAD_SIZE, dtype=dtype, device=device)
    value = torch.randn(n, NUM_KV_HEADS, HEAD_SIZE, dtype=dtype, device=device)
    rotation_t = _identity(dtype, device)

    k_stride = _key_stride(bs)
    v_stride = _val_stride(bs)
    # Need blocks 0,2 for correct dest and block 3 as wrong-dest canary.
    num_blocks = max(PHYS_BLOCKS) + 2  # 0..3

    # --- Reference: contiguous phys [0,1], firstSlot+rowOff is correct ---
    ref_k = torch.zeros(num_blocks, NUM_KV_HEADS, k_stride, dtype=torch.uint8, device=device)
    ref_v = torch.zeros(num_blocks, NUM_KV_HEADS, v_stride, dtype=torch.uint8, device=device)
    ref_slots = torch.arange(n, dtype=torch.int32, device=device)
    _pack(
        key=key,
        value=value,
        slots=ref_slots,
        rotation_t=rotation_t,
        key_cache=ref_k,
        value_cache=ref_v,
    )

    # --- Under test: reclaim-style phys [2,0] ---
    test_k = torch.full(
        (num_blocks, NUM_KV_HEADS, k_stride), PAINT, dtype=torch.uint8, device=device
    )
    test_v = torch.full(
        (num_blocks, NUM_KV_HEADS, v_stride), PAINT, dtype=torch.uint8, device=device
    )
    test_slots = _slots_for_phys(PHYS_BLOCKS, n, bs).to(device)
    print(
        f"noncontig_pack: tokens={n} bs={bs} phys={list(PHYS_BLOCKS)} "
        f"slots0={int(test_slots[0])} slots128={int(test_slots[bs])} "
        f"wrong_dest_canary=block{WRONG_DEST_BLOCK}",
        flush=True,
    )
    _pack(
        key=key,
        value=value,
        slots=test_slots,
        rotation_t=rotation_t,
        key_cache=test_k,
        value_cache=test_v,
    )

    # Map: ref block0 ↔ test PHYS_BLOCKS[0]; ref block1 used ↔ test PHYS_BLOCKS[1]
    mismatches: list[str] = []
    for h in range(NUM_KV_HEADS):
        # Full first block
        if not torch.equal(
            _extract_used_key(ref_k[0, h], bs),
            _extract_used_key(test_k[PHYS_BLOCKS[0], h], bs),
        ):
            mismatches.append(f"head{h}:key block0↔phys{PHYS_BLOCKS[0]}")
        if not torch.equal(
            _extract_used_val(ref_v[0, h], bs),
            _extract_used_val(test_v[PHYS_BLOCKS[0], h], bs),
        ):
            mismatches.append(f"head{h}:val block0↔phys{PHYS_BLOCKS[0]}")
        # Partial second block
        if not torch.equal(
            _extract_used_key(ref_k[1, h], used1),
            _extract_used_key(test_k[PHYS_BLOCKS[1], h], used1),
        ):
            mismatches.append(f"head{h}:key block1↔phys{PHYS_BLOCKS[1]}")
        if not torch.equal(
            _extract_used_val(ref_v[1, h], used1),
            _extract_used_val(test_v[PHYS_BLOCKS[1], h], used1),
        ):
            mismatches.append(f"head{h}:val block1↔phys{PHYS_BLOCKS[1]}")

    # Canary: old bug writes firstSlot+128 into WRONG_DEST_BLOCK; paint must remain.
    canary = test_k[WRONG_DEST_BLOCK]
    canary_clean = bool(torch.all(canary == PAINT))
    if not canary_clean:
        nz = int((canary != PAINT).sum().item())
        mismatches.append(
            f"wrong_dest_canary: block{WRONG_DEST_BLOCK} polluted "
            f"({nz} bytes != 0x{PAINT:02X}); firstSlot+rowOff still active"
        )

    # True second block must have been written (not left as paint).
    second = test_k[PHYS_BLOCKS[1], 0]
    if bool(torch.all(second[: used1 * KEY_ROW_CODE_BYTES] == PAINT)):
        mismatches.append(
            f"phys{PHYS_BLOCKS[1]} still paint on used codes — "
            "second block never received pack writes"
        )

    if mismatches:
        print("FAIL noncontig_pack:", "; ".join(mismatches), flush=True)
        raise AssertionError(
            "non-contiguous block_table pack mismatch: " + "; ".join(mismatches)
        )
    print(
        "PASS pack_noncontig_phys_blocks_match_contig_ref "
        f"(phys={list(PHYS_BLOCKS)}, canary_block{WRONG_DEST_BLOCK}_clean=1)",
        flush=True,
    )

    if not with_fia:
        return

    # FIA: same Q against contig vs noncontig caches must match.
    num_q = min(32, n)  # shorter Q keeps smoke fast
    num_heads = NUM_KV_HEADS * 2
    query = torch.randn(num_q, num_heads, HEAD_SIZE, dtype=dtype, device=device)
    mask = _compress_causal_mask(device)
    bt_ref = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    bt_test = torch.tensor([list(PHYS_BLOCKS)], dtype=torch.int32, device=device)
    rot = _identity(dtype, device)
    actual_q = [num_q]
    actual_kv = [n]
    out_ref = torch.ops._C_ascend.bit_residual_fia_paged_k8v4(
        query.contiguous(),
        ref_k.contiguous(),
        ref_v.contiguous(),
        bt_ref.contiguous(),
        actual_q,
        actual_kv,
        mask,
        rot,
        rot,
        num_heads,
        NUM_KV_HEADS,
        HEAD_SIZE,
        bs,
        float(SCALE),
        INT_MAX,
        INT_MAX,
        SPARSE_MODE_RIGHT_DOWN,
    )
    out_test = torch.ops._C_ascend.bit_residual_fia_paged_k8v4(
        query.contiguous(),
        test_k.contiguous(),
        test_v.contiguous(),
        bt_test.contiguous(),
        actual_q,
        actual_kv,
        mask,
        rot,
        rot,
        num_heads,
        NUM_KV_HEADS,
        HEAD_SIZE,
        bs,
        float(SCALE),
        INT_MAX,
        INT_MAX,
        SPARSE_MODE_RIGHT_DOWN,
    )
    torch.npu.synchronize()
    if out_ref is None or out_test is None:
        raise AssertionError("FIA returned None")
    if not torch.allclose(out_ref, out_test, rtol=0, atol=0):
        max_diff = float((out_ref.float() - out_test.float()).abs().max())
        raise AssertionError(
            f"FIA contig vs noncontig phys mismatch max_diff={max_diff}"
        )
    print("PASS fia_noncontig_phys_matches_contig_ref", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-fia",
        action="store_true",
        help="Also compare FIA outputs for contig vs noncontig block_table",
    )
    args = parser.parse_args()
    device = _require_npu(need_fia=args.with_fia)
    test_pack_noncontig_phys_blocks_match_contig_ref(device, with_fia=args.with_fia)
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
