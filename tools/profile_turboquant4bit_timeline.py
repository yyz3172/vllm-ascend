#!/usr/bin/env python3
"""Launch TurboquantAttentionPaged4bit through mskpp for msprof TimelineDetail."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np


HEAD_SIZE = 128
CODEBOOK_SIZE = 16
ROW_BYTES_4BIT = HEAD_SIZE // 2 + 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TurboquantAttentionPaged4bit from its AscendC .o with explicit "
            "kernel args so msprof op can collect TimelineDetail/Source."
        ))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--query-tokens", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--soc-version",
                        default=os.getenv("TQ4BIT_TIMELINE_SOC_VERSION",
                                          "Ascend910B3"))
    parser.add_argument("--dtype",
                        choices=("float16", ),
                        default="float16",
                        help="Kernel dtype to launch. Current smoke path is fp16.")
    parser.add_argument(
        "--kernel-o",
        default=os.getenv("TQ4BIT_TIMELINE_KERNEL_O", ""),
        help="Optional TurboquantAttentionPaged4bit_*.o override.")
    parser.add_argument(
        "--tiling-lib",
        default=os.getenv("TQ4BIT_TIMELINE_TILING_LIB", ""),
        help="Optional custom op tiling .so override.")
    parser.add_argument(
        "--tmp-dir",
        default=os.getenv("TQ4BIT_TIMELINE_TMP_DIR",
                          "mytmp/turboquant4bit_timeline"))
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_tiling_lib(root: Path, override: str) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"tiling lib not found: {path}")
        return path

    candidates = [
        root / "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
        root / "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_impl/ai_core/tbe/op_tiling/liboptiling.so",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "cannot find custom op tiling library; run "
        "tools/build_debug_perf.sh first or set "
        "TQ4BIT_TIMELINE_TILING_LIB")


def resolve_kernel_o(root: Path, dtype: str, override: str) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"kernel .o not found: {path}")
        return path

    kernel_dir = (
        root /
        "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_impl/ai_core/tbe/kernel/ascend910b/turboquant_attention_paged4bit"
    )
    if dtype == "float16":
        preferred = kernel_dir / "TurboquantAttentionPaged4bit_5d4907a4205cb9daa18c855ed63ec34d.o"
        if preferred.is_file():
            return preferred.resolve()

    matches = sorted(kernel_dir.glob("TurboquantAttentionPaged4bit_*.o"))
    if matches:
        return matches[-1].resolve()
    raise FileNotFoundError(
        "cannot find TurboquantAttentionPaged4bit kernel .o; run "
        "tools/build_debug_perf.sh first or set "
        "TQ4BIT_TIMELINE_KERNEL_O")


def check_debug_line(kernel_o: Path) -> bool:
    import subprocess

    result = subprocess.run(["readelf", "-S", str(kernel_o)],
                            check=False,
                            capture_output=True,
                            text=True)
    return result.returncode == 0 and ".debug_line" in result.stdout


def make_inputs(args: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if args.query_tokens % args.batch_size != 0:
        raise ValueError("--query-tokens must be divisible by --batch-size")

    rng = np.random.default_rng(args.seed)
    blocks_per_seq = (args.seq_len + args.block_size - 1) // args.block_size
    total_blocks = args.batch_size * blocks_per_seq
    cache_row_span = args.block_size * ROW_BYTES_4BIT

    query = rng.normal(
        0.0, 0.02,
        size=(args.query_tokens, args.heads, HEAD_SIZE)).astype(np.float16)
    key_cache = np.zeros((total_blocks, args.kv_heads, cache_row_span),
                         dtype=np.uint8)
    value_cache = np.zeros_like(key_cache)

    block_table = np.empty((args.batch_size, blocks_per_seq), dtype=np.int32)
    for seq in range(args.batch_size):
        for block in range(blocks_per_seq):
            block_table[seq, block] = seq * blocks_per_seq + block

    query_tokens_per_seq = args.query_tokens // args.batch_size
    actual_seq_len_q = np.array(
        [(seq + 1) * query_tokens_per_seq for seq in range(args.batch_size)],
        dtype=np.int64)
    actual_seq_len_kv = np.full((args.batch_size, ),
                                args.seq_len,
                                dtype=np.int64)

    codebook = np.array(
        [
            -0.2255,
            -0.1651,
            -0.1251,
            -0.0941,
            -0.0687,
            -0.0468,
            -0.0271,
            -0.0085,
            0.0093,
            0.0275,
            0.0469,
            0.0688,
            0.0941,
            0.1242,
            0.1628,
            0.2206,
        ],
        dtype=np.float16)
    rotation = np.eye(HEAD_SIZE, dtype=np.float16)
    out = np.zeros((args.query_tokens, args.heads, HEAD_SIZE),
                   dtype=np.float16)

    inputs = [
        query,
        key_cache,
        value_cache,
        block_table,
        actual_seq_len_q,
        actual_seq_len_kv,
        codebook,
        rotation,
        codebook,
        rotation,
    ]
    outputs = [out]
    return inputs, outputs


def main() -> int:
    args = parse_args()
    root = repo_root()
    tmp_dir = (root / args.tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TMPDIR"] = str(tmp_dir / "tmp")
    os.environ["TMP"] = os.environ["TMPDIR"]
    os.environ["TEMP"] = os.environ["TMPDIR"]
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    tiling_lib = resolve_tiling_lib(root, args.tiling_lib)
    kernel_o = resolve_kernel_o(root, args.dtype, args.kernel_o)
    has_debug_line = check_debug_line(kernel_o)
    if not has_debug_line:
        raise RuntimeError(
            f"{kernel_o} has no .debug_line; rebuild with "
            "tools/build_debug_perf.sh")

    # mskpp writes mindstudio_mskpp_gen under cwd.
    os.chdir(tmp_dir)

    from mskpp import get_kernel_from_binary, tiling_func

    inputs, outputs = make_inputs(args)
    attrs = [
        {
            "name": "num_heads",
            "dtype": "int",
            "value": args.heads
        },
        {
            "name": "num_kv_heads",
            "dtype": "int",
            "value": args.kv_heads
        },
        {
            "name": "head_size",
            "dtype": "int",
            "value": HEAD_SIZE
        },
        {
            "name": "block_size",
            "dtype": "int",
            "value": args.block_size
        },
        {
            "name": "max_actual_seq_len",
            "dtype": "int",
            "value": args.seq_len
        },
        {
            "name": "scale_value",
            "dtype": "float",
            "value": 1.0 / math.sqrt(HEAD_SIZE)
        },
    ]

    tiling = tiling_func("TurboquantAttentionPaged4bit",
                         inputs=inputs,
                         outputs=outputs,
                         lib_path=str(tiling_lib),
                         attr=attrs,
                         soc_version=args.soc_version)
    print("TurboquantAttentionPaged4bit timeline launch")
    print(f"  kernel_o={kernel_o}")
    print(f"  tiling_lib={tiling_lib}")
    print(f"  soc_version={args.soc_version}")
    print(f"  blockdim={tiling.blockdim}")
    print(f"  tiling_key={tiling.tiling_key}")
    print(f"  workspace_size={tiling.workspace_size}")
    print(f"  debug_line={has_debug_line}")

    kernel = get_kernel_from_binary(str(kernel_o), kernel_type="mix")
    kernel[tiling.blockdim](
        *inputs,
        outputs[0],
        tiling.workspace,
        tiling.tiling_data,
        device_id=args.device,
        repeat=args.repeat,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
