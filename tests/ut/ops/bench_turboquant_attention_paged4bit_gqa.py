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

"""Direct-op benchmark for TurboQuant paged 4-bit attention GQA scaling.

Run from repo root:

    source xrx_infoenvs
    python tests/ut/ops/bench_turboquant_attention_paged4bit_gqa.py

The benchmark intentionally bypasses vLLM scheduling and calls the custom op
directly.  ``scalar`` cases use a small token count that should not select the
qTile tiling key.  ``qtile`` cases use enough token tasks to select qTile.
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch

try:
    import torch_npu  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    torch_npu = None

from vllm_ascend.ops.turboquant_kv_cache import (
    _c_ascend_turboquant_op_available,
    _turboquant_fused_8bit_decode_tables,
    _turboquant_pack_tables,
)

HEAD_SIZE = 128
BLOCK_SIZE = 16
ROW_BYTES = HEAD_SIZE // 2 + 2
UB_GQA_CAP = 8
MAX_PARALLEL_CORES = 16


def _build_block_table(
    actual_seq_lens_kv: list[int],
    block_size: int,
) -> tuple[torch.Tensor, int]:
    blocks_per_seq = [
        (length + block_size - 1) // block_size
        for length in actual_seq_lens_kv
    ]
    max_blocks = max(blocks_per_seq)
    total_blocks = sum(blocks_per_seq)
    block_table = torch.zeros((len(actual_seq_lens_kv), max_blocks), dtype=torch.int32)
    next_block = 0
    for seq_idx, block_count in enumerate(blocks_per_seq):
        block_table[seq_idx, :block_count] = torch.arange(
            next_block, next_block + block_count, dtype=torch.int32
        )
        next_block += block_count
    return block_table, total_blocks


def _build_slab_cache(
    *,
    num_blocks: int,
    num_kv_heads: int,
    seed: int,
    norm_dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        0,
        16,
        (num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_SIZE),
        generator=generator,
        dtype=torch.uint8,
    )
    norms = (
        torch.rand(
            (num_blocks, num_kv_heads, BLOCK_SIZE),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.25
        + 0.25
    ).to(norm_dtype)
    groups_per_block = BLOCK_SIZE // 4
    group_bytes = ROW_BYTES * 4
    cache_group = torch.zeros(
        (num_blocks, num_kv_heads, groups_per_block, group_bytes),
        dtype=torch.uint8,
    )
    words = torch.zeros(
        (num_blocks, num_kv_heads, groups_per_block, HEAD_SIZE),
        dtype=torch.int32,
    )
    for group_row in range(4):
        idx = indices[:, group_row::4, :, :].permute(0, 2, 1, 3).to(torch.int32)
        words |= (idx & 0x0F) << (group_row * 4)
    cache_group[..., : HEAD_SIZE * 2 : 2] = (words & 0xFF).to(torch.uint8)
    cache_group[..., 1 : HEAD_SIZE * 2 : 2] = ((words >> 8) & 0xFF).to(torch.uint8)

    norm_bytes = norms.view(torch.uint8).reshape(num_blocks, num_kv_heads, BLOCK_SIZE, 2)
    for group_row in range(4):
        cache_group[
            ...,
            HEAD_SIZE * 2 + group_row * 2 : HEAD_SIZE * 2 + (group_row + 1) * 2,
        ] = norm_bytes[:, :, group_row::4, :]
    return cache_group.reshape(num_blocks, num_kv_heads, BLOCK_SIZE * ROW_BYTES)


def _expected_qtile(
    *,
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    parallel_cores: int,
) -> tuple[bool, int, int, int, int]:
    gqa_group = num_heads // num_kv_heads
    gqa_chunks = math.ceil(gqa_group / UB_GQA_CAP)
    head_chunk_scale = num_kv_heads * gqa_chunks
    task_count = num_tokens * head_chunk_scale
    used_cores = min(task_count, parallel_cores)
    block_split_range = math.ceil(task_count / used_cores) if used_cores else 0
    return (
        block_split_range > head_chunk_scale,
        gqa_group,
        gqa_chunks,
        task_count,
        block_split_range,
    )


def _bench_case(
    *,
    label: str,
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    kv_len: int,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
    parallel_cores: int,
) -> dict[str, float | int | str]:
    device = torch.device("npu")
    actual_seq_lens_q = [num_tokens]
    actual_seq_lens_kv = [kv_len]
    block_table_cpu, total_blocks = _build_block_table(actual_seq_lens_kv, BLOCK_SIZE)
    key_cache = _build_slab_cache(
        num_blocks=total_blocks,
        num_kv_heads=num_kv_heads,
        seed=17,
        norm_dtype=dtype,
    ).to(device).contiguous()
    value_cache = _build_slab_cache(
        num_blocks=total_blocks,
        num_kv_heads=num_kv_heads,
        seed=117,
        norm_dtype=dtype,
    ).to(device).contiguous()
    block_table = block_table_cpu.to(device).contiguous()

    torch.manual_seed(1000 + num_heads * 17 + num_kv_heads)
    query = torch.randn((num_tokens, num_heads, HEAD_SIZE), dtype=dtype, device=device)
    cb_k, rot_t_k = _turboquant_pack_tables(
        device=device,
        head_size=HEAD_SIZE,
        bits=4,
        dtype=dtype,
    )
    _, _, _, rot_v = _turboquant_fused_8bit_decode_tables(
        device=device,
        head_size=HEAD_SIZE,
        bits_key=4,
        bits_value=4,
        dtype=dtype,
    )
    cb_k = cb_k.contiguous()
    rot_t_k = rot_t_k.contiguous()
    rot_v = rot_v.contiguous()

    op = torch.ops._C_ascend.turboquant_attention_paged4bit
    scale = 1.0 / math.sqrt(HEAD_SIZE)

    def call() -> torch.Tensor:
        return op(
            query,
            key_cache,
            value_cache,
            block_table,
            actual_seq_lens_q,
            actual_seq_lens_kv,
            cb_k,
            rot_t_k,
            cb_k,
            rot_v,
            num_heads,
            num_kv_heads,
            HEAD_SIZE,
            BLOCK_SIZE,
            kv_len,
            float(scale),
        )

    for _ in range(warmup):
        out = call()
    torch.npu.synchronize()

    durations_us: list[float] = []
    for _ in range(repeat):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        out = call()
        end.record()
        torch.npu.synchronize()
        durations_us.append(start.elapsed_time(end) * 1000.0)

    expected_qtile, gqa_group, gqa_chunks, task_count, block_split_range = _expected_qtile(
        num_tokens=num_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        parallel_cores=parallel_cores,
    )
    checksum = float(out.float().mean().cpu())
    return {
        "label": label,
        "tokens": num_tokens,
        "kv_len": kv_len,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "gqa_group": gqa_group,
        "gqa_chunks": gqa_chunks,
        "task_count": task_count,
        "expected_qtile": int(expected_qtile),
        "block_split_range": block_split_range,
        "avg_us": statistics.mean(durations_us),
        "median_us": statistics.median(durations_us),
        "min_us": min(durations_us),
        "max_us": max(durations_us),
        "checksum": checksum,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("both", "scalar", "qtile"), default="both")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--kv-len", type=int, default=128)
    parser.add_argument("--scalar-tokens", type=int, default=4)
    parser.add_argument("--qtile-tokens", type=int, default=40)
    parser.add_argument("--parallel-cores", type=int, default=MAX_PARALLEL_CORES)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if torch_npu is None or not torch.npu.is_available():
        raise RuntimeError("Ascend NPU and torch_npu are required")
    if not _c_ascend_turboquant_op_available("turboquant_attention_paged4bit"):
        raise RuntimeError("turboquant_attention_paged4bit custom op is unavailable")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    modes: list[tuple[str, int]] = []
    if args.mode in ("both", "scalar"):
        modes.append(("scalar", args.scalar_tokens))
    if args.mode in ("both", "qtile"):
        modes.append(("qtile", args.qtile_tokens))

    shapes = [(8, 1), (16, 1), (32, 1), (32, 4)]
    header = (
        "label,tokens,kv_len,num_heads,num_kv_heads,gqa_group,gqa_chunks,"
        "expected_qtile,task_count,block_split_range,avg_us,median_us,min_us,max_us,checksum"
    )
    print(header)
    for mode_name, tokens in modes:
        for num_heads, num_kv_heads in shapes:
            row = _bench_case(
                label=f"{mode_name}:h{num_heads}_kv{num_kv_heads}",
                num_tokens=tokens,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                kv_len=args.kv_len,
                dtype=dtype,
                warmup=args.warmup,
                repeat=args.repeat,
                parallel_cores=args.parallel_cores,
            )
            print(
                f"{row['label']},{row['tokens']},{row['kv_len']},"
                f"{row['num_heads']},{row['num_kv_heads']},{row['gqa_group']},"
                f"{row['gqa_chunks']},{row['expected_qtile']},{row['task_count']},"
                f"{row['block_split_range']},{row['avg_us']:.2f},"
                f"{row['median_us']:.2f},{row['min_us']:.2f},{row['max_us']:.2f},"
                f"{row['checksum']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
