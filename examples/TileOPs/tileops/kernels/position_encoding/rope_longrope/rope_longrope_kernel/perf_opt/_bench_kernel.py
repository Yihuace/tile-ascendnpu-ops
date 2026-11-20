"""Standalone msprof op probe for the RoPE (neox-family) kernels (one workload/process).

Loads the kernel factory from an arbitrary candidate file so the measured
artifact is exactly the file named on the command line.  Only the kernel is
launched -- the Stage 5 wrapper / benchmark is never used.

1d: kernel(x (S,d), cos (S,d/2), sin (S,d/2)) -> y (S,d)
2d: kernel(x_flat (B*S*H*d,), cos (S,d/2), sin (S,d/2)) -> y_flat

Usage:
  python _bench_kernel.py --impl <candidate.py> --layout 1d --S 2048 --d 64 \
      --dtype float16 [--iters 10] [--tile <int>] [--core_num <int>]
  python _bench_kernel.py --impl <candidate.py> --layout 2d --B 2 --S 2048 \
      --H 32 --d 128 --dtype float16 [--iters 10] [--tile <int>]
"""

import argparse
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401


_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _load_impl(path: str, layout: str):
    name = f"rope_neox_{layout}_candidate_mod"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tables(seq_len, head_dim, dtype_str, base=10000.0):
    half = head_dim // 2
    td = _TORCH_DTYPE[dtype_str]
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), freqs)
    return torch.cos(angles).to(td).npu(), torch.sin(angles).to(td).npu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--layout", required=True, choices=("1d", "2d"))
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--block_size", type=int, default=None,
                    help="override the block_size passed to the compiled callable "
                         "(default: the candidate module's _DEFAULT_BLOCK_SIZE[dtype])")
    ap.add_argument("--tile", type=int, default=None,
                    help="override the tuned tile (rows for 1d / tokens for 2d); "
                         "only honoured by candidates exposing the kwarg")
    ap.add_argument("--core_num", type=int, default=None)
    args = ap.parse_args()

    mod = _load_impl(args.impl, args.layout)
    td = _TORCH_DTYPE[args.dtype]
    cos, sin = _tables(args.S, args.d, args.dtype)

    blk = args.block_size if args.block_size is not None else mod._DEFAULT_BLOCK_SIZE[args.dtype]

    if args.layout == "1d":
        factory = mod._make_rope_neox_1d
        kwargs = {}
        if args.tile is not None:
            kwargs["tile_rows"] = args.tile
        if args.core_num is not None:
            kwargs["core_num"] = args.core_num
        kernel = factory(args.S, args.d, args.dtype, **kwargs)(blk)
        x = torch.randn(args.S, args.d, dtype=td, device="npu")
    else:
        factory = mod._make_rope_neox_2d
        kwargs = {}
        if args.tile is not None:
            kwargs["block_tokens"] = args.tile
        if args.core_num is not None:
            kwargs["core_num"] = args.core_num
        kernel = factory(args.B, args.S, args.H, args.d, args.dtype, **kwargs)(blk)
        x = torch.randn(args.B * args.S * args.H * args.d, dtype=td, device="npu")

    for _ in range(args.iters):
        kernel(x, cos, sin)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
