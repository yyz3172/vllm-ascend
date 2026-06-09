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

from __future__ import annotations

from functools import lru_cache
import os

import torch
import vllm_ascend.envs as envs_ascend

# Optional: torch_npu is only present in NPU runtime.
try:
    import torch_npu  # type: ignore
except Exception:  # pragma: no cover
    torch_npu = None

# Use the vLLM root logger so messages always follow vLLM log config.
from vllm.logger import logger

from vllm_ascend.ops.turboquant_lloyd_max import LloydMaxCodebook
from vllm_ascend.utils import enable_custom_op


def _sync_npu_if_needed(device: torch.device) -> None:
    """Wait for in-flight NPU work; avoids reading packed bytes before AscendC encode finishes."""
    if device.type not in ("npu", "privateuseone"):
        return
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def _c_ascend_turboquant_op_available(op_name: str) -> bool:
    """Return True once ``vllm_ascend_C`` is loaded and ``op_name`` exists on ``torch.ops._C_ascend``.

    Custom ops are registered when the extension module is imported; ``hasattr`` alone
    after ``build_ext`` is not enough until ``enable_custom_op()`` runs in this process.
    """
    if not enable_custom_op():
        return False
    lib = getattr(torch.ops, "_C_ascend", None)
    return lib is not None and hasattr(lib, op_name)


def pack_uint4(indices: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit indices into uint8 bytes (nibble packing)."""
    if indices.dtype != torch.uint8:
        raise ValueError(f"indices must be uint8, got {indices.dtype}")
    if indices.shape[-1] % 2 != 0:
        raise ValueError(f"Last dim must be even, got {indices.shape[-1]}")
    high = indices[..., 0::2] << 4
    low = indices[..., 1::2] & 0x0F
    return (high | low).to(torch.uint8)


def unpack_uint4(packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
    """Unpack uint8 bytes into 4-bit indices (uint8 values 0..15)."""
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed must be uint8, got {packed.dtype}")
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    out = torch.empty(
        packed.shape[:-1] + (orig_dim,), dtype=torch.uint8, device=packed.device
    )
    out[..., 0::2] = high
    out[..., 1::2] = low
    return out


def _validate_turboquant_bits(bits: int) -> None:
    if bits not in (4, 8):
        raise NotImplementedError(f"TurboQuant KV supports bits 4 or 8 only, got {bits}.")


def turboquant_indices_byte_len(head_size: int, bits: int) -> int:
    """Byte length of the packed index segment (excluding the 2-byte fp16 norm slot)."""
    _validate_turboquant_bits(bits)
    if bits == 4:
        if head_size % 2 != 0:
            raise ValueError(f"4-bit TurboQuant requires even head_size, got {head_size}.")
        return head_size // 2
    return head_size


def _pack_turboquant_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Flatten last dim to packed uint8 bytes (nibbles for 4-bit, raw uint8 for 8-bit)."""
    _validate_turboquant_bits(bits)
    if bits == 4:
        return pack_uint4(indices)
    return indices


def _unpack_turboquant_indices(idx_packed: torch.Tensor, head_size: int, bits: int) -> torch.Tensor:
    _validate_turboquant_bits(bits)
    if bits == 4:
        return unpack_uint4(idx_packed, head_size)
    if idx_packed.shape[-1] != head_size:
        raise ValueError(
            f"8-bit indices last dim must be head_size={head_size}, got {idx_packed.shape[-1]}."
        )
    return idx_packed.view(dtype=torch.uint8)


def _turboquant_u32_tuple_from_fp32_le_hex(*hex_chunks: str) -> tuple[int, ...]:
    """uint32 words from contiguous hex = little-endian fp32 IEEE bit patterns per word."""
    h = "".join(s.strip() for s in hex_chunks)
    raw = bytes.fromhex(h)
    if len(raw) % 4:
        raise ValueError("TurboQuant codebook hex must have length multiple of 8 (4 bytes/word).")
    return tuple(int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw), 4))


# IEEE-754 float32 bit patterns for fixed scalar codebooks (4-bit: 16 entries; 8-bit: 256).
# Stored as LE hex for a short module; to refresh from tensors use ``dump_turboquant_*_u32_literal``
# and convert the printed ``0x..`` tuple to hex via the same packing as this helper.
_TURBOQUANT_4BIT_CODEBOOK_FP32_U32: tuple[int, ...] = _turboquant_u32_tuple_from_fp32_le_hex(
    "8ff666becf1129be561000bed7bdc0bdd5c48cbd49a43fbde5d9ddbc0b110cbc"
    "07c5183ca23ae13cb1f13f3d4fe48c3d92b5c03dbd46fe3dcda8263e59ee613e",
)

_TURBOQUANT_8BIT_CODEBOOK_FP32_U32: tuple[int, ...] = _turboquant_u32_tuple_from_fp32_le_hex(
    "d1959abea19584be2cf66dbe86ec5abe64984bbef7c83fbe41a536be22dc2ebe"
    "937d28beac0d23bebeb41ebe99961abe54d116be345213beb7f60fbeebbe0cbe"
    "74e309bec72c07bec68e04bea6fc01be7123ffbd226ffabdf0b4f5bdca3bf1bd"
    "ab0cedbd8bf6e8bd81ffe4bdb75ee1bda8b4ddbdad0cdabd7d94d6bd80fcd2bd"
    "127ccfbd5718ccbdb7b6c8bd453cc5bdad08c2bdafdabebd5cbbbbbd7d94b8bd"
    "d592b5bda0adb2bd1ebcafbd3ccdacbdaf11aabdb75ca7bd9da0a4bd97d6a1bd"
    "22599fbd90f49cbd6a999abd991c98bdfdb295bd395a93bda9e790bdfb6c8ebd"
    "bcef8bbdce9189bdbc4787bd11fc84bdda8682bd6b4480bdaa477cbd4a9677bd"
    "8bed72bd669c6ebddc196abd2bb665bd988761bda1735dbd4d6959bdca7655bd"
    "625651bdb0104dbd64eb48bd169444bd174d40bd192b3cbdad4138bdbd2134bd"
    "e47330bdae9c2cbd56e828bdcae724bd38e720bdd9cd1cbdf2d918bd8b1c15bd"
    "e75811bd9d740dbd47b109bd08e505bdca1b02bdf5effcbcdb83f5bc80d7edbc"
    "0ce0e5bc4ff3ddbc2026d6bcb68acebc5217c7bc6d31c0bc3c07b9bc45c5b1bc"
    "9272aabcd104a3bc43679bbc4bb793bc9af68bbce57b84bcf28379bc0a2a6abc"
    "9def5bbc23f64dbc97623fbcb16c31bc3f3523bc297615bcec0e08bce2a1f3bb"
    "7729d6bb1a90b9bb67eb9cbb890a7fbb929a46bb93f30ebb2b5caebaf059f5b9"
    "a26eb7398bdaa53a13ff0e3b8d61483b8039813b3c249d3bcbbeba3b4a06d83b"
    "4da6f33bea12083cf392163cbe9e253ca207343c6990423cccb5513ce36b603c"
    "64e16e3c5b247d3c5ae1853c9f4c8d3cb587943cdb989b3cffe1a23cf72daa3c"
    "177eb13c1d9ab83c6e56c03c7598c73c5414cf3c3499d63cfb4bde3c33c2e53c"
    "3263ed3c7fbff43cd566fc3cc9ee013dc1b0053dc54c093d5dc30c3d4741103d"
    "68da133dfea4173da1891b3dbd821f3def87233db5b4273df2f42b3d2a42303d"
    "3b40343d1504383d54d53b3d109e3f3d4e87433dadb9473d00ec4b3da156503d"
    "2a85543d31e2583d6e7b5d3dbe1d623d0e71663dc6fa6a3de2266f3d8669733d"
    "b9ef773d007c7c3d3787803dc8d1823d9bfd843d6c34873d1174893d17d78b3d"
    "ae438e3d57b1903de831933da4a3953dfe20983df6b49a3d844b9d3da5f69f3d"
    "be8aa23dc218a53d89e4a73df67eaa3da31fad3d82f9af3db6c7b23d58eab53d"
    "530bb93d8b0dbc3d6b15bf3d0f44c23df774c53df9a6c83d00e2cb3d5a12cf3d"
    "bb77d23d7b00d63daf98d93d1130dd3d0ce5e03dfc99e43dda94e83db151ec3d"
    "a466f03dabbef43da12af93dd8f2fd3df761013e9ede033e52a2063e7d7c093e"
    "dc8f0c3e57b30f3e210d133eceb2163e8a9e1a3ebecc1e3e257b233e2dee283e"
    "141b2f3e0758363e43293f3e981b4a3e0dde573e1ab0693efa1d813e3728953e",
)

def dump_turboquant_codebook_u32_literal(codebook: torch.Tensor, *, bits: int) -> None:
    """Print a copy-paste block for ``_turboquant_u32_tuple_from_fp32_le_hex`` (LE fp32 bits).

    Output matches the module form:
    ``_TURBOQUANT_{bits}BIT_CODEBOOK_FP32_U32 = _turboquant_u32_tuple_from_fp32_le_hex(...)``
    with 64 hex characters per string line (32 bytes). ``bits`` = 4 → 16 words; 8 → 256.

    Pass the float32 tensor from ``_beta_codebook_via_sampling`` (shape ``[2**bits]``).
    Do not copy values from ``print(tensor)``; use this helper so float32 bits are exact.
    """
    k = 1 << bits
    t = codebook.detach().float().contiguous().reshape(-1)
    if t.numel() != k:
        raise ValueError(
            f"expected {k} codebook entries for bits={bits}, got {t.numel()}"
        )
    u32_list = [int(x) & 0xFFFFFFFF for x in t.cpu().view(torch.int32).reshape(-1).tolist()]
    raw = b"".join(w.to_bytes(4, "little") for w in u32_list)
    hexs = raw.hex()
    chunk_w = 64  # 32 bytes per line; same as literals in this module
    chunks = [hexs[i : i + chunk_w] for i in range(0, len(hexs), chunk_w)]
    name = f"_TURBOQUANT_{bits}BIT_CODEBOOK_FP32_U32"
    print(f"{name}: tuple[int, ...] = _turboquant_u32_tuple_from_fp32_le_hex(")
    for i, c in enumerate(chunks):
        if i < len(chunks) - 1:
            print(f'    "{c}"')
        else:
            print(f'    "{c}",')
    print(")")


def dump_turboquant_4bit_codebook_u32_literal(codebook: torch.Tensor) -> None:
    """Print a lossless hex literal block for ``_TURBOQUANT_4BIT_CODEBOOK_FP32_U32``."""
    dump_turboquant_codebook_u32_literal(codebook, bits=4)


def dump_turboquant_8bit_codebook_u32_literal(codebook: torch.Tensor) -> None:
    """Print a lossless hex literal block for ``_TURBOQUANT_8BIT_CODEBOOK_FP32_U32``."""
    dump_turboquant_codebook_u32_literal(codebook, bits=8)


def _beta_codebook_via_sampling(
    *,
    dim: int,
    bits: int,
    device: torch.device,
    seed: int,
    n_samples: int = 200_000,
    n_iters: int = 30,
) -> torch.Tensor:
    """Approximate the 1D scalar quantizer codebook for TurboQuantMSE (sample mode)."""
    if bits < 1 or bits > 8:
        raise NotImplementedError("TurboQuant bits must be in [1, 8].")
    k = 1 << bits
    alpha = (dim - 1) / 2.0

    beta = torch.distributions.Beta(alpha, alpha)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        u = beta.sample((n_samples,))  # [0,1]
    x = (2.0 * u - 1.0).to(dtype=torch.float32, device=device).clamp(-1.0, 1.0)

    # Seed centroids with quantiles.
    probs = (torch.arange(1, k + 1, device=device, dtype=torch.float32) - 0.5) / k
    try:
        c = torch.quantile(x, probs).contiguous()
    except Exception:
        c = torch.quantile(x.cpu(), probs.cpu()).to(device=device).contiguous()

    for _ in range(n_iters):
        d = (x[:, None] - c[None, :]).abs()
        a = d.argmin(dim=1)
        new_c = torch.empty_like(c)
        for j in range(k):
            mask = a == j
            new_c[j] = x[mask].mean() if torch.any(mask) else c[j]
        new_c, _ = torch.sort(new_c)
        c = new_c
    if bits == 4:
        dump_turboquant_4bit_codebook_u32_literal(c)
    elif bits == 8:
        dump_turboquant_8bit_codebook_u32_literal(c)
    return c


def _fast_codebook_4bit(*, device: torch.device) -> torch.Tensor:
    """Fast 4-bit scalar codebook: exact float32 values from bit patterns.

    See `_TURBOQUANT_4BIT_CODEBOOK_FP32_U32` and `dump_turboquant_4bit_codebook_u32_literal`.
    """
    t = torch.tensor(_TURBOQUANT_4BIT_CODEBOOK_FP32_U32, dtype=torch.uint32)
    return t.view(torch.float32).to(device=device)


def _v1_fast_codebook(*, dim: int, bits: int, device: torch.device) -> torch.Tensor:
    """Deterministic V1 codebook: fixed 4-bit table or Lloyd–Max for 8-bit."""
    _validate_turboquant_bits(bits)
    if bits == 4:
        return _fast_codebook_4bit(device=device)
    return LloydMaxCodebook(dim, bits, use_exact=False).centroids.to(
        device=device, dtype=torch.float32
    )


def _generate_haar_orthogonal_matrix(
    dim: int, seed: int, device: torch.device
) -> torch.Tensor:
    """Haar random orthogonal matrix (same construction as turboquant-npu ``generate_rotation_matrix``)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    g = torch.randn((dim, dim), generator=gen, dtype=torch.float32, device="cpu")
    q, r = torch.linalg.qr(g)
    s = torch.sign(torch.diag(r))
    s[s == 0] = 1.0
    q = q * s.unsqueeze(0)
    return q.to(device=device)


def _normalize_turboquant_mse_impl(name: str) -> str:
    n = name.lower().strip()
    return n if n in ("v1", "v2", "v3") else "v1"


def _turboquant_v3_y_hat_from_indices(indices: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Per-coordinate reconstruction in rotated space (V3 dequant), ``y_hat`` before ``@ R``."""
    x = indices.to(dtype=torch.float32)
    if bits == 8:
        # Ki ∈ [0, 255]
        return 0.0026 * (x - 127.5)
    if bits == 4:
        # Vi ∈ [0, 15]
        d = x - 7.5
        return 0.0001926 * (d**3) + 0.020799 * d
    raise ValueError(f"TurboQuantMSEV3 supports bits 4 or 8 only, got {bits}.")


class TurboQuantMSE:
    """Turbo ``QuantMSE`` (V1): Haar rotation + scalar codebook; **4 or 8 bits** per coordinate."""

    def __init__(self, *, dim: int, bits: int = 4, device: torch.device, seed: int = 42):
        _validate_turboquant_bits(bits)
        if bits == 4 and dim % 2 != 0:
            raise ValueError(f"TurboQuant 4-bit requires even dim for packing, got {dim}.")

        self.dim = dim
        self.bits = bits
        self.device = device
        self.seed = seed

        self.rotation = _generate_haar_orthogonal_matrix(dim, seed, device)
        self.rotation_t = self.rotation.T

        if envs_ascend.VLLM_ASCEND_TURBOQUANT_CODEBOOK_METHOD == "sample":
            self.codebook = _beta_codebook_via_sampling(
                dim=dim, bits=bits, device=torch.device("cpu"), seed=seed + 1234
            ).to(device=device)
        else:
            self.codebook = _v1_fast_codebook(dim=dim, bits=bits, device=device)
        print("TurboQuantMSE codebook: ", self.codebook)
        # Cached fp16 views for custom op fast path (initialized lazily).
        self._codebook_fp16: torch.Tensor | None = None
        self._rotation_fp16: torch.Tensor | None = None
        self._rotation_t_fp16: torch.Tensor | None = None
        self._rotation_batched_fp16: dict[tuple[str, int], torch.Tensor] = {}
        self._codebook_fp16 = self.codebook.to(dtype=torch.float16)
        self._rotation_fp16 = self.rotation.to(dtype=torch.float16)
        self._rotation_t_fp16 = self.rotation_t.to(dtype=torch.float16)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")
        x_f32 = x.to(dtype=torch.float16)
        norms = torch.linalg.vector_norm(x_f32, dim=-1, keepdim=True)
        x_unit = x_f32 / (norms + 1e-10)
        y = x_unit @ self._rotation_t_fp16
        d = (y.unsqueeze(-1) - self._codebook_fp16.view(1, 1, -1)).abs()
        idx = d.argmin(dim=-1).to(torch.uint8)
        return idx, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {indices.shape[-1]}")
        y_hat = self._codebook_fp16[indices.to(torch.int64)]
        x_hat = y_hat @ self._rotation_fp16
        return x_hat * norms.to(dtype=torch.float16)


class TurboQuantMSEV2:
    """Lloyd–Max scalar codebook (same tables as ``turboquant-npu``), **same KV pack semantics as V1**.

    - ``quantize``: ``y = (x / ‖x‖) @ R.T``, nearest Lloyd–Max centroid per coord; returns **true** ``‖x‖``.
    - ``dequantize``: ``x_hat = centroids[idx] @ R * ‖x‖`` (norm read from the 2-byte fp16 slot).

    V1 differs mainly in **codebook construction** (fixed 4-bit table / env-driven sample) and
    4-bit path options; V2 always uses :class:`LloydMaxCodebook` centroids for the given ``dim``.

    Select via ``VLLM_ASCEND_TURBOQUANT_MSE_IMPL=v2``.
    """

    def __init__(self, *, dim: int, bits: int = 4, device: torch.device, seed: int = 42):
        _validate_turboquant_bits(bits)
        if bits == 4 and dim % 2 != 0:
            raise ValueError(f"TurboQuant 4-bit requires even dim for packing, got {dim}.")

        self.dim = dim
        self.bits = bits
        self.device = device
        self.seed = seed
        self.rotation = _generate_haar_orthogonal_matrix(dim, seed, device)
        self.rotation_t = self.rotation.T
        self.codebook = LloydMaxCodebook(dim, bits, use_exact=False).centroids.to(
            device=device, dtype=torch.float32
        )
        print("TurboQuantMSEV2 codebook: ", self.codebook)
        self._codebook_fp16 = self.codebook.to(dtype=torch.float16)
        self._rotation_fp16 = self.rotation.to(dtype=torch.float16)
        self._rotation_t_fp16 = self.rotation_t.to(dtype=torch.float16)
        self._rotation_batched_fp16: dict[tuple[str, int], torch.Tensor] = {}

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")
        x_f32 = x.to(dtype=torch.float32)
        norms = torch.linalg.vector_norm(x_f32, dim=-1, keepdim=True)
        x_unit = x_f32 / (norms + 1e-10)
        y = x_unit @ self.rotation_t
        d = (y.unsqueeze(-1) - self.codebook.view(1, 1, -1)).abs()
        idx = d.argmin(dim=-1).to(torch.uint8)
        return idx, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {indices.shape[-1]}")
        y_hat = self.codebook[indices.to(torch.int64)]
        x_hat = y_hat @ self.rotation
        return x_hat * norms.to(dtype=torch.float32)


class TurboQuantMSEV3(TurboQuantMSEV2):
    """Same **quantize** path as :class:`TurboQuantMSEV2` (Lloyd–Max indices + ‖x‖).

    **Dequantize** replaces codebook lookup with closed-form maps from indices
    in rotated space:

    - **8-bit**: ``y_i = 0.0026 * (K_i - 127.5)`` for ``K_i ∈ [0, 255]``.
    - **4-bit**: ``y_i = 0.0001926 * (V_i - 7.5)^3 + 0.020799 * (V_i - 7.5)`` for ``V_i ∈ [0, 15]``.

    Then ``x_hat = y_hat @ R * ‖x‖`` as in V2. Quantize still uses Lloyd–Max
    centroids to pick indices (same as V2); only the reconstruction from index
    to scalar differs.

    NPU TurboQuant **decode custom ops** assume codebook lookup; use
    ``VLLM_ASCEND_TURBOQUANT_DECODE_OP=0`` when ``MSE_IMPL=v3``, or decoding may
    disagree with this class.

    Select via ``VLLM_ASCEND_TURBOQUANT_MSE_IMPL=v3``.
    """

    def __init__(self, *, dim: int, bits: int = 4, device: torch.device, seed: int = 42):
        super().__init__(dim=dim, bits=bits, device=device, seed=seed)
        print("TurboQuantMSEV3 (V2 quantize + formula dequant), bits=", self.bits)

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {indices.shape[-1]}")
        y_hat = _turboquant_v3_y_hat_from_indices(indices, bits=self.bits)
        x_hat = y_hat @ self.rotation
        return x_hat * norms.to(dtype=torch.float32)


@lru_cache(maxsize=64)
def _get_quantizer(
    dim: int, bits: int, device: torch.device
) -> TurboQuantMSE:
    return TurboQuantMSE(dim=dim, bits=bits, device=device, seed=42)


def _current_mse_impl() -> str:
    return _normalize_turboquant_mse_impl(envs_ascend.VLLM_ASCEND_TURBOQUANT_MSE_IMPL)


_turboquant_kv_banner_logged: bool = False


def log_turboquant_kv_banner_once(
    *,
    cache_dtype: str,
    turboquant_kv_bits_key: int,
    turboquant_kv_bits_value: int,
) -> None:
    """Log whether TurboQuant KV and MSE v1/v2/v3 are in use (once per process)."""
    global _turboquant_kv_banner_logged
    if _turboquant_kv_banner_logged:
        return
    _turboquant_kv_banner_logged = True
    if cache_dtype == "turboquant":
        mse = _current_mse_impl()
        if mse == "v3":
            quan = "TurboQuantMSEV3 (formula dequant)"
        elif mse == "v2":
            quan = "TurboQuantMSEV2 (Lloyd–Max)"
        else:
            quan = "TurboQuantMSE (v1)"
        bits_msg = (
            f"key={turboquant_kv_bits_key}, value={turboquant_kv_bits_value}"
            if turboquant_kv_bits_key != turboquant_kv_bits_value
            else f"{turboquant_kv_bits_key}"
        )
        logger.info(
            "[vllm-ascend] KV TurboQuant: on, bits(%s), quantizer=%s (MSE_IMPL=%s). "
            "Override: VLLM_ASCEND_TURBOQUANT_MSE_IMPL=v1, v2, or v3.",
            bits_msg,
            quan,
            mse,
        )
    else:
        logger.info(
            "[vllm-ascend] KV TurboQuant: off (cache_dtype=%r).",
            cache_dtype,
        )


def turboquant_packed_bytes_per_vector(head_size: int, bits: int = 4) -> int:
    """Bytes per head vector: packed indices + 2-byte fp16 norm slot."""
    return turboquant_indices_byte_len(head_size, bits) + 2


def _pad_packed_to_slot_width(packed: torch.Tensor, slot_width: int) -> torch.Tensor:
    """Right-pad last dim so ``shape[-1] == slot_width`` (for wider shared cache rows)."""
    w = packed.shape[-1]
    if w == slot_width:
        return packed
    if w > slot_width:
        raise ValueError(
            f"packed width {w} exceeds cache slot width {slot_width}"
        )
    return torch.nn.functional.pad(packed, (0, slot_width - w))


def turboquant_quantize_to_packed_bytes(x: torch.Tensor, *, bits: int = 4) -> torch.Tensor:
    head_size = x.shape[-1]
    packed_bytes = turboquant_packed_bytes_per_vector(head_size, bits=bits)
    quantizer = _get_quantizer(head_size, bits, x.device)

    x_flat = x.reshape(-1, head_size)
    indices, norms = quantizer.quantize(x_flat)  # [N,D], [N,1]
    idx_storage = _pack_turboquant_indices(indices, bits)
    idx_w = idx_storage.shape[-1]

    norms_fp16 = norms.to(dtype=torch.float16)
    norm_bytes = norms_fp16.view(torch.uint8).view(-1, 2)

    packed = torch.empty((x_flat.shape[0], packed_bytes), dtype=torch.uint8, device=x.device)
    packed[:, :idx_w] = idx_storage.reshape(x_flat.shape[0], idx_w)
    packed[:, idx_w : idx_w + 2] = norm_bytes
    return packed.reshape(*x.shape[:-1], packed_bytes)


def turboquant_dequantize_from_packed_bytes(
    packed: torch.Tensor, *, head_size: int, dtype: torch.dtype, bits: int = 4
) -> torch.Tensor:
    packed_bytes = turboquant_packed_bytes_per_vector(head_size, bits=bits)
    row_w = packed.shape[-1]
    if row_w < packed_bytes:
        raise ValueError(
            f"packed last dim {row_w} smaller than logical width {packed_bytes} "
            f"(bits={bits}, head_size={head_size})."
        )
    packed_logical = packed[..., :packed_bytes]
    if row_w != packed_bytes:
        packed_logical = packed_logical.contiguous()

    # Fast path: use compiled NPU custom op when available.
    # NOTE: Some torch/torch_npu builds do not expose Tensor.is_privateuseone().
    # Use device.type as a stable fallback to detect NPU tensors.
    dev_type = getattr(getattr(packed, "device", None), "type", None)
    is_npu_tensor = (dev_type in ("npu", "privateuseone"))
    if (envs_ascend.VLLM_ASCEND_TURBOQUANT_DECODE_OP
            and is_npu_tensor
            and bits == 4
            and row_w == packed_bytes
            and head_size % 2 == 0
            and _current_mse_impl() != "v3"
            and dtype in (torch.float16, torch.bfloat16)
            and _c_ascend_turboquant_op_available("turboquant_decode_packed_blocks")):
        # codebook: [16] fp16 on NPU, rotation: [D, D] fp16 on NPU
        quantizer = _get_quantizer(head_size, bits, packed.device)
        if quantizer._codebook_fp16 is None or quantizer._codebook_fp16.device != packed.device:
            quantizer._codebook_fp16 = quantizer.codebook.to(device=packed.device, dtype=torch.float16)
        if quantizer._rotation_fp16 is None or quantizer._rotation_fp16.device != packed.device:
            rot = quantizer.rotation.to(device=packed.device, dtype=torch.float16)
            # Matmul performance on Ascend is sensitive to internal format, but
            # forcing NZ can also backfire depending on torch_npu/CANN version
            # and matmul kernel selection. Only enable NZ when explicitly asked.
            if (envs_ascend.VLLM_ASCEND_ENABLE_NZ == 2 and torch_npu is not None
                    and hasattr(torch_npu, "npu_format_cast")):
                try:
                    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

                    rot = torch_npu.npu_format_cast(rot, ACL_FORMAT_FRACTAL_NZ)
                except Exception:
                    pass
            quantizer._rotation_fp16 = rot
        codebook = quantizer._codebook_fp16
        rotation = quantizer._rotation_fp16
        # Custom decode kernel assumes packed is contiguous 2D row-major.
        # packed can be a view with non-trivial strides (e.g. from slicing/indexing).
        packed_view = packed_logical.reshape(-1, packed_bytes).contiguous()
        out = torch.ops._C_ascend.turboquant_decode_packed_blocks(
            packed_view,
            codebook,
            rotation,
            head_size,
            0 if dtype == torch.float16 else 1,
        )
        out = out.reshape(*packed.shape[:-1], head_size)
        return out

    # Reference path (PyTorch ops).
    quantizer = _get_quantizer(head_size, bits, packed.device)
    packed_flat = packed_logical.reshape(-1, packed_bytes)
    idx_len = turboquant_indices_byte_len(head_size, bits)
    idx_packed = packed_flat[:, :idx_len]
    norm_bytes = packed_flat[:, idx_len : idx_len + 2]

    indices = _unpack_turboquant_indices(idx_packed, head_size, bits)
    # norms_fp16 = norm_bytes.contiguous().view(torch.float16).view(-1, 1)
    # norms = norms_fp16.to(dtype=torch.float32)
    norms = norm_bytes.contiguous().view(torch.float16).view(-1, 1)
    if not torch.isfinite(norms).all():
        norms = torch.nan_to_num(norms, nan=0.0, posinf=65504.0, neginf=0.0)

    x_hat = quantizer.dequantize(indices, norms).to(dtype=dtype)
    return x_hat.reshape(*packed.shape[:-1], head_size)


def _ensure_quantizer_fp16_views(
    quantizer: TurboQuantMSE | TurboQuantMSEV2 | TurboQuantMSEV3,
    _device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (codebook_fp16, rotation_t_fp16) eager-cached at init time."""
    return quantizer._codebook_fp16, quantizer._rotation_t_fp16


def turboquant_pack_kv_for_cache(
    *,
    key: torch.Tensor,  # [T, H, D]
    value: torch.Tensor,  # [T, H, D]
    bits_key: int,
    bits_value: int,
    slot_w_k: int,
    slot_w_v: int,
    codebook: torch.Tensor | None = None,
    rotation: torch.Tensor | None = None,
    codebook_value: torch.Tensor | None = None,
    rotation_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K/V to packed rows padded to cache slot width (no cache scatter).

    Intended for ``torch_npu._npu_reshape_and_cache`` on Ascend, which performs the
    paged layout write when the last dim matches ``key_cache`` / ``value_cache``.

    When ``VLLM_ASCEND_TURBOQUANT_ENCODE_OP=1`` and the fused NPU op is built,
    uses ``turboquant_pack_kv_for_cache``: norm + ``@ R^T`` + nearest-neighbor
    encode. If ``bits_key == bits_value``, K and V are encoded in one batched
    kernel launch.

    Optional ``codebook`` / ``rotation`` (``R^T``, fp16) override the per-head
    quantizer tables; ``codebook_value`` / ``rotation_value`` apply to V when
    K/V bit widths differ.
    """
    if key.numel() == 0:
        return key, value
    if value.dtype != key.dtype:
        raise ValueError("Key/value dtypes must match for turboquant.")

    head_size = key.shape[-1]
    if envs_ascend.VLLM_ASCEND_TURBOQUANT_ENCODE_OP:
        qk = _get_quantizer(head_size, bits_key, key.device)
        qv = qk if bits_key == bits_value else _get_quantizer(
            head_size, bits_value, key.device
        )
        if codebook is None or rotation is None:
            cb_k, rot_k = _ensure_quantizer_fp16_views(qk, key.device)
        else:
            cb_k = codebook.to(device=key.device, dtype=torch.float16)
            rot_k = rotation.to(device=key.device, dtype=torch.float16)
        if bits_key == bits_value:
            cb_v, rot_v = cb_k, rot_k
        elif codebook_value is None or rotation_value is None:
            cb_v, rot_v = _ensure_quantizer_fp16_views(qv, key.device)
        else:
            cb_v = codebook_value.to(device=key.device, dtype=torch.float16)
            rot_v = rotation_value.to(device=key.device, dtype=torch.float16)

        # The fused AscendC kernel currently accepts fp16 K/V only. Cast here so
        # callers with bf16/fp32 KV tensors still use the fused path consistently.
        key_fused = key.to(dtype=torch.float16).contiguous()
        value_fused = value.to(dtype=torch.float16).contiguous()
        packed_k, packed_v = torch.ops._C_ascend.turboquant_pack_kv_for_cache(
            key_fused,
            value_fused,
            cb_k,
            rot_k,
            cb_v,
            rot_v,
            bits_key,
            bits_value,
            slot_w_k,
            slot_w_v,
        )
        # Match ENCODE_OP=0: int8 view of byte storage for _npu_reshape_and_cache.
        # C++ already returns contiguous kChar tensors; keep the explicit view for parity.
        return (
            packed_k.view(dtype=torch.int8),
            packed_v.view(dtype=torch.int8),
        )

    packed_k = _pad_packed_to_slot_width(
        turboquant_quantize_to_packed_bytes(key, bits=bits_key), slot_w_k
    )
    packed_v = _pad_packed_to_slot_width(
        turboquant_quantize_to_packed_bytes(value, bits=bits_value), slot_w_v
    )
    return packed_k.view(dtype=torch.int8), packed_v.view(dtype=torch.int8)


def turboquant_decode_kv_cache_compact(
    *,
    key_cache: torch.Tensor,  # [B, BS, H, P] uint8
    value_cache: torch.Tensor,  # [B, BS, H, P] uint8
    block_tables: torch.Tensor,  # [num_seqs, max_blocks_per_seq] int32/int64 (may contain -1)
    head_size: int,
    dtype: torch.dtype,
    bits: int = 4,
    bits_key: int | None = None,
    bits_value: int | None = None,
    decode_only_arange_fast_path: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode only blocks referenced by `block_tables` and remap block_tables to a
    compact 0..U-1 range to avoid decoding the full KV cache.

    When ``decode_only_arange_fast_path`` is True (DecodeOnly state), this function
    skips the ``unique``/``searchsorted`` compaction (which has data-dependent shape
    and breaks ACL Graph capture) and uses an ``arange`` mapping instead. The cost
    is that physical blocks shared across sequences are decoded once per reference
    rather than once total; the benefit is fully static shapes and no host sync.

    Padding slots in ``block_tables`` (vLLM v1 fills unused slots with 0, the
    reserved ``null_block``) are kept in the compact mapping rather than re-marked
    as -1: a negative compact index would be unsafe for NPU paged-attention
    kernels that prefetch entire block_table rows. The arange path prepends a
    null_block sentinel as workspace slot 0 so every padding entry can map to a
    valid index pointing at all-zero content.

    Returns:
      key_cache_decoded: shape ``[U, BS, H, D]`` (unique path) or
        ``[N + 1, BS, H, D]`` (arange path, with slot 0 as null_block sentinel)
      value_cache_decoded: same shape as ``key_cache_decoded``
      block_tables_compact: same shape as block_tables; every entry is a valid
        non-negative index into the decoded workspace.
    """
    if block_tables.numel() == 0:
        # Degenerate case: return empty decoded caches.
        empty = torch.empty((0,) + key_cache.shape[1:-1] + (head_size,), dtype=dtype, device=key_cache.device)
        return empty, empty, block_tables

    bt = block_tables.to(torch.int32)

    bk = bits if bits_key is None else bits_key
    bv = bits if bits_value is None else bits_value

    # if decode_only_arange_fast_path:
    #     # Static-shape decode path for DecodeOnly:
    #     #
    #     # Workspace layout (size N + 1, where N = bt.numel()):
    #     #   slot 0      : decode(cache[0]) — null_block sentinel for padding sinks
    #     #   slot 1..N   : decode(cache[bt.flatten()[i].clamp(min=0)]) for i in [0, N)
    #     #
    #     # ``bt_compact`` mapping:
    #     #   bt[i] >  0 (valid)   : i + 1   (own dedicated workspace slot)
    #     #   bt[i] <= 0 (padding) : 0       (points at the null_block sentinel)
    #     #
    #     # vLLM v1 reserves physical block id 0 as ``null_block`` and zero-fills
    #     # unused slots of ``block_table``. Routing padding to compact index 0
    #     # mirrors the unique-path convention (where searchsorted maps padding to
    #     # the null_block's compact index) — every entry of ``bt_compact`` is a
    #     # valid workspace index, no negative offsets.
    #     n = bt.numel()
    #     flat_bt = bt.flatten()
    #     gather_idx = torch.cat(
    #         (
    #             torch.zeros(1, device=bt.device, dtype=torch.int64),
    #             flat_bt.clamp(min=0).to(torch.int64),
    #         )
    #     )  # [N + 1]
    #     key_packed = key_cache.index_select(0, gather_idx)  # [N+1, BS, H, P]
    #     value_packed = value_cache.index_select(0, gather_idx)

    #     # Decode via the PyTorch reference path (always correct). An 8-bit
    #     # dedicated decode op is the subject of design doc Phase 0.4 / Phase 1.
    #     k = turboquant_dequantize_from_packed_bytes(
    #         key_packed, head_size=head_size, dtype=dtype, bits=bk
    #     )
    #     v = turboquant_dequantize_from_packed_bytes(
    #         value_packed, head_size=head_size, dtype=dtype, bits=bv
    #     )

    #     # bt_compact = (arange(1..N+1) reshaped) * (bt > 0). Padding → 0;
    #     # arithmetic mask avoids ``torch.where`` and its bool dispatch.
    #     arange_plus_one = torch.arange(1, n + 1, device=bt.device, dtype=torch.int32)
    #     bt_compact = arange_plus_one.view_as(bt) * (bt > 0).to(torch.int32)
    #     return k, v, bt_compact.to(block_tables.dtype)

    valid = bt >= 0
    if not torch.any(valid):
        empty = torch.empty((0,) + key_cache.shape[1:-1] + (head_size,), dtype=dtype, device=key_cache.device)
        return empty, empty, block_tables

    used = bt[valid]
    used_sorted = used.unique()
    logger.debug(
        "TurboQuant(ascend) decode_compact: block_tables=%s used_blocks=%d",
        tuple(block_tables.shape),
        int(used_sorted.numel()),
    )

    # Gather packed blocks and decode only those.
    key_packed = key_cache.index_select(0, used_sorted)    # [U, BS, H, P]
    value_packed = value_cache.index_select(0, used_sorted)  # [U, BS, H, P]

    # 8-bit only: decode via PyTorch reference path. The legacy 4-bit
    # ``turboquant_decode_packed_blocks_compact`` custom op was removed because
    # its correctness has not been verified (see design doc §0/§1.3); an 8-bit
    # dedicated decode op is the subject of Phase 0.4 / Phase 1.
    k = turboquant_dequantize_from_packed_bytes(
        key_packed, head_size=head_size, dtype=dtype, bits=bk
    )
    v = turboquant_dequantize_from_packed_bytes(
        value_packed, head_size=head_size, dtype=dtype, bits=bv
    )

    # Remap block tables to compact indices.
    # block_tables_compact = searchsorted(used_sorted, bt) for valid entries.
    bt_compact = bt.clone()
    bt_compact[valid] = torch.searchsorted(used_sorted, used, out_int32=True)
    return k, v, bt_compact.to(block_tables.dtype)


def turboquant_paged_decode_host_indices(
    block_table: torch.Tensor,
    actual_seq_lengths_kv,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Host-side ceil+cumsum for the 8-bit paged decode op (design doc §2.6.7).

    Replaces the ``unique``/``searchsorted`` compaction with a compact physical
    block-id list (in seq order) plus a remapped block table. Pure tensor ops
    (no custom op), so it is CPU-testable.

    Args:
      block_table: ``[batch, max_blocks_per_seq]`` physical block ids; the first
        ``ceil(seq_len / block_size)`` entries of each row are the live blocks.
      actual_seq_lengths_kv: per-seq KV lengths (list or tensor).
      block_size: KV block size (BS).

    Returns:
      gather_block_ids: ``[total_blocks]`` int32, ``block_table`` entries of every
        seq concatenated in order (the workspace block order).
      bt_compact: same shape/dtype as ``block_table``; valid slot ``[s, j]`` maps to
        ``block_offsets[s] + j`` and padding slots collapse to ``block_offsets[s]``
        (a valid workspace slot, so FIA prefetch never reads out of range).
      total_blocks: ``int`` number of referenced blocks (one D2H sync).
    """
    device = block_table.device
    _, max_bps = block_table.shape
    if isinstance(actual_seq_lengths_kv, torch.Tensor):
        actual_kv = actual_seq_lengths_kv.to(device=device, dtype=torch.int64)
    else:
        actual_kv = torch.tensor(actual_seq_lengths_kv, dtype=torch.int64, device=device)

    num_blocks_per_seq = (actual_kv + block_size - 1) // block_size          # [batch]
    block_offsets = num_blocks_per_seq.cumsum(0) - num_blocks_per_seq        # [batch]
    total_blocks = int(num_blocks_per_seq.sum().item())
    arange_j = torch.arange(max_bps, device=device, dtype=torch.int64)
    j_mask = arange_j.unsqueeze(0) < num_blocks_per_seq.unsqueeze(1)         # [batch, max_bps]
    gather_block_ids = block_table[j_mask].to(torch.int32)                   # [total_blocks]
    bt_compact = block_offsets.unsqueeze(1) + arange_j.unsqueeze(0)          # [batch, max_bps]
    bt_compact = torch.where(j_mask, bt_compact, block_offsets.unsqueeze(1))
    bt_compact = bt_compact.to(block_table.dtype)
    return gather_block_ids, bt_compact, total_blocks


def _try_8bit_decode_paged(
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_kv,
    head_size: int,
    dtype: torch.dtype,
    bits_key: int,
    bits_value: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """8-bit packed KV cache -> compact fp16 K/V via the fused paged decode op.

    Returns ``(key_ws, value_ws, bt_compact)`` on hit (``key_ws``/``value_ws`` are
    ``[total_blocks, BS, H, head_size]`` fp16) or ``None`` to fall back to the
    PyTorch ``turboquant_decode_kv_cache_compact`` path. Two gates: the env switch
    and op availability (design doc §2.6.7). Kept as a separate switch from the
    4-bit ``VLLM_ASCEND_TURBOQUANT_DECODE_OP`` because the 4-bit op is unverified.
    """
    if not envs_ascend.VLLM_ASCEND_TURBOQUANT_DECODE_OP_8BIT:
        return None
    if block_table.numel() == 0:
        return None
    block_size = int(key_cache.shape[1])
    device = key_cache.device
    gather_block_ids, bt_compact, total_blocks = turboquant_paged_decode_host_indices(
        block_table, actual_seq_lengths_kv, block_size
    )
    if total_blocks == 0:
        empty = torch.empty(
            (0,) + tuple(key_cache.shape[1:-1]) + (head_size,),
            dtype=dtype,
            device=device,
        )
        return empty, empty, bt_compact

    quantizer = _get_quantizer(head_size, 8, device)
    codebook = quantizer._codebook_fp16
    if codebook is None or codebook.device != device:
        codebook = quantizer.codebook.to(device=device, dtype=torch.float16)
        quantizer._codebook_fp16 = codebook
    rotation = quantizer._rotation_fp16
    if rotation is None or rotation.device != device:
        rotation = quantizer.rotation.to(device=device, dtype=torch.float16)
        quantizer._rotation_fp16 = rotation

    mode = 1 if envs_ascend.VLLM_ASCEND_TURBOQUANT_DECODE_OP_8BIT_MODE == 1 else 0
    key_ws, value_ws = torch.ops._C_ascend.turboquant_decode_paged_8bit(
        key_cache.view(torch.uint8).contiguous(),
        value_cache.view(torch.uint8).contiguous(),
        gather_block_ids.contiguous(),
        codebook,
        rotation,
        head_size,
        block_size,
        0,  # out_dtype = fp16
        mode,
    )
    return key_ws.to(torch.bfloat16), value_ws.to(torch.bfloat16), bt_compact


def _turboquant_fused_infer_attention_score_8bit_impl(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    atten_mask: torch.Tensor | None,
    actual_seq_lengths_q: list[int],
    actual_seq_lengths_kv: list[int],
    head_size: int,
    num_heads: int,
    num_key_value_heads: int,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """Fallback path (non-fused): decode 8-bit TurboQuant KV cache then run FIA.

    True fused implementation is provided by `_C_ascend.turboquant_fused_infer_attention_score_8bit`.
    This function is only used when the compiled custom op is unavailable.
    """
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for TurboQuant FIA fallback.")
    if atten_mask is None:
        raise RuntimeError("atten_mask is required for TurboQuant FIA fallback.")

    key_dec, value_dec, block_tables_compact = turboquant_decode_kv_cache_compact(
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        head_size=head_size,
        dtype=query.dtype,
        bits_key=8,
        bits_value=8,
    )
    key = key_dec.flatten(2, 3).contiguous()
    value = value_dec.flatten(2, 3).contiguous()

    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        atten_mask=atten_mask,
        block_table=block_tables_compact,
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_key_value_heads=num_key_value_heads,
        num_heads=num_heads,
        scale=scale,
        sparse_mode=3,
    )
    return attn_output.view(query.shape[0], num_heads, head_size)


def _turboquant_fused_8bit_decode_tables(
    device: torch.device,
    head_size: int,
    bits_key: int,
    bits_value: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (codebook_k, rotation_k, codebook_v, rotation_v) fp16 on device."""
    qk = _get_quantizer(head_size, bits_key, device)
    if bits_key == bits_value:
        qv = qk
    else:
        qv = _get_quantizer(head_size, bits_value, device)
    if qk._codebook_fp16 is None or qk._codebook_fp16.device != device:
        qk._codebook_fp16 = qk.codebook.to(device=device, dtype=torch.float16)
    if qk._rotation_fp16 is None or qk._rotation_fp16.device != device:
        qk._rotation_fp16 = qk.rotation.to(device=device, dtype=torch.float16)
    cb_k, rot_k = qk._codebook_fp16, qk._rotation_fp16
    if qv is qk:
        return cb_k, rot_k, cb_k, rot_k
    if qv._codebook_fp16 is None or qv._codebook_fp16.device != device:
        qv._codebook_fp16 = qv.codebook.to(device=device, dtype=torch.float16)
    if qv._rotation_fp16 is None or qv._rotation_fp16.device != device:
        qv._rotation_fp16 = qv.rotation.to(device=device, dtype=torch.float16)
    return cb_k, rot_k, qv._codebook_fp16, qv._rotation_fp16


def turboquant_fused_infer_attention_score_8bit(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    atten_mask: torch.Tensor | None,
    actual_seq_lengths_q: list[int],
    actual_seq_lengths_kv: list[int],
    head_size: int,
    num_heads: int,
    num_key_value_heads: int,
    block_size: int,
    scale: float,
    bits_key: int = 8,
    bits_value: int = 8,
) -> torch.Tensor:
    """Call the fused `_C_ascend` op when available; otherwise decode+FIA fallback."""
    if atten_mask is None:
        raise RuntimeError("atten_mask is required for TurboQuant fused FIA.")

    fused = getattr(getattr(torch.ops, "_C_ascend", None), "turboquant_fused_infer_attention_score_8bit", None)
    if fused is None:
        return _turboquant_fused_infer_attention_score_8bit_impl(
            query,
            key_cache,
            value_cache,
            block_tables,
            atten_mask,
            actual_seq_lengths_q,
            actual_seq_lengths_kv,
            head_size,
            num_heads,
            num_key_value_heads,
            block_size,
            scale,
        )

    cb_k, rot_k, cb_v, rot_v = _turboquant_fused_8bit_decode_tables(
        query.device, head_size, bits_key, bits_value
    )
    seq_q = torch.tensor(actual_seq_lengths_q, device=query.device, dtype=torch.int32)
    seq_kv = torch.tensor(actual_seq_lengths_kv, device=query.device, dtype=torch.int32)
    out = fused(
        query,
        key_cache,
        value_cache,
        block_tables.to(torch.int32),
        atten_mask,
        seq_q,
        seq_kv,
        cb_k,
        rot_k,
        cb_v,
        rot_v,
        int(num_heads),
        int(num_key_value_heads),
        int(head_size),
        int(block_size),
        float(scale),
    )
    return out.view(query.shape[0], num_heads, head_size)


def _seq_lens_tensor_and_max(
    seq_lens,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, int]:
    if isinstance(seq_lens, torch.Tensor):
        seq_lens_tensor = seq_lens.to(device=device, dtype=dtype)
        max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
        return seq_lens_tensor, max_seq_len

    max_seq_len = max(seq_lens) if seq_lens else 0
    seq_lens_tensor = torch.tensor(seq_lens, device=device, dtype=dtype)
    return seq_lens_tensor, int(max_seq_len)


def turboquant_attention_paged8bit(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seq_lengths_q: list[int],
    actual_seq_lengths_kv: list[int],
    head_size: int,
    num_heads: int,
    num_key_value_heads: int,
    block_size: int,
    scale: float,
    bits_key: int = 8,
    bits_value: int = 8,
) -> torch.Tensor | None:
    """Call the fused packed 8-bit paged attention op when supported.

    Returns ``None`` when the custom op is unavailable or the request falls
    outside the initial kernel constraints, so callers can keep existing
    decode+FIA fallbacks.
    """
    if (
        bits_key != 8
        or bits_value != 8
        or head_size != 128
        or block_tables.numel() == 0
        or num_key_value_heads <= 0
        or num_heads % num_key_value_heads != 0
    ):
        return None
    gqa_group = num_heads // num_key_value_heads
    # Kernel UB buffers are sized for G <= 8 (see turboquant_attention_paged8bit.cpp).
    if gqa_group > 8:
        return None
    if not _c_ascend_turboquant_op_available("turboquant_attention_paged8bit"):
        print("op not available")
        return None

    cb_k, rot_k, cb_v, rot_v = _turboquant_fused_8bit_decode_tables(
        query.device, head_size, bits_key, bits_value
    )
    seq_q, _ = _seq_lens_tensor_and_max(
        actual_seq_lengths_q, device=query.device, dtype=torch.int64
    )
    seq_kv, max_actual_seq_len = _seq_lens_tensor_and_max(
        actual_seq_lengths_kv, device=query.device, dtype=torch.int64
    )
    if max_actual_seq_len <= 0:
        return None
    fused = torch.ops._C_ascend.turboquant_attention_paged8bit
    out = fused(
        query.to(torch.float16).contiguous(),
        key_cache.view(torch.uint8).contiguous(),
        value_cache.view(torch.uint8).contiguous(),
        block_tables.to(torch.int32).contiguous(),
        seq_q.contiguous(),
        seq_kv.contiguous(),
        cb_k,
        rot_k,
        cb_v,
        rot_v,
        int(num_heads),
        int(num_key_value_heads),
        int(head_size),
        int(block_size),
        int(max_actual_seq_len),
        float(scale),
    )
    return out.view(query.shape[0], num_heads, head_size)

