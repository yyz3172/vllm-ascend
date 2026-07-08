"""BitResidual k8v4 manual-Mmad unaligned-slot test.

Run on an NPU machine after building ``bit_residual_pack_k8v4``:

    python tests/e2e/singlecard/xrx_bit_residual_k8v4_key1_unaligned.py

The test checks:
  1. two identical NPU calls produce bit-exact cache bytes;
  2. decoded Key/Value cache matches the CPU reference quantizer;
  3. non-group-aligned first slots are handled with read-modify-write.
"""

from __future__ import annotations

import os

import torch

from vllm_ascend.utils import enable_custom_op

D = 128
H = 8
BS = 128
GROUP_STRIDE = 288
KEY_GROUP_ROWS = 2
VALUE_GROUP_ROWS = 4
SEED = 42
SHORT_SLOT_OFFSETS = (0, 1, 2, 3, 5)
LONG_SLOT_OFFSETS = (0, 3)


def _require_op() -> None:
    if not enable_custom_op():
        raise RuntimeError("vllm_ascend_C is not loaded; build/install custom ops first")
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(
        torch.ops._C_ascend, "bit_residual_pack_k8v4"
    ):
        raise RuntimeError("torch.ops._C_ascend.bit_residual_pack_k8v4 is not registered")


def _make_rotation(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Deterministic orthogonal matrix. The kernel consumes rotation_t directly.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + 17)
    q, _ = torch.linalg.qr(torch.randn(D, D, generator=gen, dtype=torch.float32))
    return q.to(device=device, dtype=dtype).contiguous()


def _cpu_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    rotation_t: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    def _rotate(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_f32 = x.cpu().float()
        dtype = x.dtype
        rot_f32 = rotation_t.cpu().to(dtype).float()
        norm = x_f32.norm(dim=-1)
        a = (x_f32 / (norm.unsqueeze(-1) + 1e-10)).to(dtype).float()
        y = (a @ rot_f32).to(dtype).float()
        return y, norm

    key_y, key_norm = _rotate(key)
    sign = key_y > -0.0
    sign_val = torch.where(sign, 1.0 / (D**0.5), -1.0 / (D**0.5))
    err = key_y - sign_val
    base = err.amin(dim=-1)
    maxv = err.amax(dim=-1)
    key_range = maxv - base
    step = torch.where(key_range > 0, key_range / 127.0, torch.ones_like(key_range))
    inv_step = torch.where(key_range > 0, 1.0 / step, torch.zeros_like(step))
    q7 = torch.round((err - base.unsqueeze(-1)) * inv_step.unsqueeze(-1))
    q7 = q7.clamp(0, 127).to(torch.uint8)
    code = ((q7.to(torch.int16) << 1) | sign.to(torch.int16)).to(torch.uint8)

    value_y, _ = _rotate(value)
    vmin = value_y.amin(dim=-1)
    vmax = value_y.amax(dim=-1)
    value_range = vmax - vmin
    vstep = torch.where(value_range > 0, value_range / 15.0, torch.ones_like(value_range))
    inv_vstep = torch.where(value_range > 0, 1.0 / vstep, torch.zeros_like(vstep))
    idx4 = torch.round((value_y - vmin.unsqueeze(-1)) * inv_vstep.unsqueeze(-1))
    idx4 = idx4.clamp(0, 15).to(torch.uint8)

    return (
        {"code": code, "norm": key_norm.to(key.dtype).float(), "base": base, "step": step},
        {"idx4": idx4, "vmin": vmin, "vstep": vstep},
    )


def _u8_to_float32(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.float32).clone()


def _decode_key_cache(
    cache: torch.Tensor,
    slots: torch.Tensor,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    cache_cpu = cache.cpu()
    slots_cpu = slots.cpu().to(torch.int64)
    t = slots_cpu.numel()
    h = cache_cpu.shape[1]
    code = torch.empty(t, h, D, dtype=torch.uint8)
    norm = torch.empty(t, h, dtype=torch.float32)
    base = torch.empty(t, h, dtype=torch.float32)
    step = torch.empty(t, h, dtype=torch.float32)
    norm_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16

    for token_idx, slot in enumerate(slots_cpu.tolist()):
        block_idx = slot // BS
        block_off = slot % BS
        group_idx = block_off // KEY_GROUP_ROWS
        group_row = block_off % KEY_GROUP_ROWS
        group_base = group_idx * GROUP_STRIDE
        for head in range(h):
            group = cache_cpu[block_idx, head, group_base : group_base + GROUP_STRIDE]
            words = group[:256].contiguous().view(torch.uint16)
            if group_row == 0:
                code[token_idx, head] = (words & 0x00FF).to(torch.uint8)
            else:
                code[token_idx, head] = ((words >> 8) & 0x00FF).to(torch.uint8)
            norm[token_idx, head] = (
                group[256 + group_row * 2 : 258 + group_row * 2]
                .contiguous()
                .view(norm_dtype)
                .float()
                .item()
            )
            base[token_idx, head] = _u8_to_float32(
                group[260 + group_row * 4 : 264 + group_row * 4]
            ).item()
            step[token_idx, head] = _u8_to_float32(
                group[268 + group_row * 4 : 272 + group_row * 4]
            ).item()

    return {"code": code, "norm": norm, "base": base, "step": step}


def _decode_value_cache(cache: torch.Tensor, slots: torch.Tensor) -> dict[str, torch.Tensor]:
    cache_cpu = cache.cpu()
    slots_cpu = slots.cpu().to(torch.int64)
    t = slots_cpu.numel()
    h = cache_cpu.shape[1]
    idx4 = torch.empty(t, h, D, dtype=torch.uint8)
    vmin = torch.empty(t, h, dtype=torch.float32)
    vstep = torch.empty(t, h, dtype=torch.float32)

    for token_idx, slot in enumerate(slots_cpu.tolist()):
        block_idx = slot // BS
        block_off = slot % BS
        group_idx = block_off // VALUE_GROUP_ROWS
        group_row = block_off % VALUE_GROUP_ROWS
        group_base = group_idx * GROUP_STRIDE
        shift = group_row * 4
        for head in range(h):
            group = cache_cpu[block_idx, head, group_base : group_base + GROUP_STRIDE]
            words = group[:256].contiguous().view(torch.uint16)
            idx4[token_idx, head] = ((words >> shift) & 0x000F).to(torch.uint8)
            vmin[token_idx, head] = _u8_to_float32(
                group[256 + group_row * 4 : 260 + group_row * 4]
            ).item()
            vstep[token_idx, head] = _u8_to_float32(
                group[272 + group_row * 4 : 276 + group_row * 4]
            ).item()

    return {"idx4": idx4, "vmin": vmin, "vstep": vstep}


def _assert_match(
    name: str,
    dtype: torch.dtype,
    key_ref: dict[str, torch.Tensor],
    value_ref: dict[str, torch.Tensor],
    key_dec: dict[str, torch.Tensor],
    value_dec: dict[str, torch.Tensor],
) -> None:
    torch.testing.assert_close(key_dec["code"], key_ref["code"], rtol=0, atol=0)
    torch.testing.assert_close(value_dec["idx4"], value_ref["idx4"], rtol=0, atol=0)

    norm_atol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(
        key_dec["norm"], key_ref["norm"], rtol=2e-2, atol=norm_atol
    )
    torch.testing.assert_close(key_dec["base"], key_ref["base"], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(key_dec["step"], key_ref["step"], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(value_dec["vmin"], value_ref["vmin"], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(value_dec["vstep"], value_ref["vstep"], rtol=2e-3, atol=2e-3)
    print(f"PASS {name}: dtype={dtype}")


def _run_case(
    name: str,
    token_count: int,
    slot_offset: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    torch.manual_seed(SEED + token_count + slot_offset)
    key = torch.randn(token_count, H, D, dtype=dtype, device=device).contiguous()
    value = torch.randn(token_count, H, D, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots = torch.arange(slot_offset, slot_offset + token_count, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, token_count], dtype=torch.int32, device=device)
    num_blocks = int((slot_offset + token_count + BS - 1) // BS + 1)

    key_ref, value_ref = _cpu_reference(key, value, rotation_t)

    key_cache_1 = torch.zeros(
        num_blocks, H, (BS // KEY_GROUP_ROWS) * GROUP_STRIDE,
        dtype=torch.uint8, device=device,
    )
    value_cache_1 = torch.zeros(
        num_blocks, H, (BS // VALUE_GROUP_ROWS) * GROUP_STRIDE,
        dtype=torch.uint8, device=device,
    )
    key_cache_2 = torch.zeros_like(key_cache_1)
    value_cache_2 = torch.zeros_like(value_cache_1)

    torch.ops._C_ascend.bit_residual_pack_k8v4(
        key, value, slots, qsl, rotation_t, key_cache_1, value_cache_1, 1, BS
    )
    torch.npu.synchronize()
    torch.ops._C_ascend.bit_residual_pack_k8v4(
        key, value, slots, qsl, rotation_t, key_cache_2, value_cache_2, 1, BS
    )
    torch.npu.synchronize()

    torch.testing.assert_close(key_cache_1, key_cache_2, rtol=0, atol=0)
    torch.testing.assert_close(value_cache_1, value_cache_2, rtol=0, atol=0)

    key_dec = _decode_key_cache(key_cache_1, slots, dtype)
    value_dec = _decode_value_cache(value_cache_1, slots)
    _assert_match(name, dtype, key_ref, value_ref, key_dec, value_dec)


def main() -> None:
    _require_op()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    device = torch.device("npu:0")
    long_tokens = int(os.getenv("BR_K8V4_LONG_TOKENS", "0"))
    for dtype in (torch.float16, torch.bfloat16):
        for offset in SHORT_SLOT_OFFSETS:
            _run_case(f"short_offset_{offset}", 9, offset, dtype, device)
        if long_tokens > 0:
            for offset in LONG_SLOT_OFFSETS:
                _run_case(f"long_offset_{offset}", long_tokens, offset, dtype, device)


if __name__ == "__main__":
    main()
