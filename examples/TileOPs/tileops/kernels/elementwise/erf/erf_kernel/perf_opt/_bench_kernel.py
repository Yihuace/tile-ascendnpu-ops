"""Standalone msprof op probe for the erf kernel (one workload per process).

Loads the kernel factory from an arbitrary candidate file so that the exact
artifact under measurement is the one named on the command line.  Only the
kernel is launched here -- the Stage 5 wrapper / benchmark is never used.

Usage:
  python _bench_kernel.py --impl <candidate.py> --N <int> --dtype <str>
                          [--block_size <int>] [--iters <int>]
"""

import argparse
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401


def _load_impl(path: str):
    spec = importlib.util.spec_from_file_location("erf_candidate_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["erf_candidate_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--clamp", type=float, default=None)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    mod = _load_impl(args.impl)
    factory = mod.erf_fwd_kernel

    N = args.N
    dtype_str = args.dtype
    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype_str]
    x = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()

    kwargs = {}
    if args.block_size is not None:
        kwargs["block_size"] = args.block_size
    if args.variant is not None:
        kwargs["variant"] = args.variant
    if args.degree is not None:
        kwargs["degree"] = args.degree
    if args.mode is not None:
        kwargs["mode"] = args.mode
    if args.clamp is not None:
        kwargs["clamp"] = args.clamp
    kernel = factory(N, dtype_str, **kwargs)

    for _ in range(args.iters):
        kernel(x, N)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
