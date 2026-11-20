"""Standalone msprof op probe for the expm1 kernel (one workload per process).

Loads the kernel factory from an arbitrary candidate file so that the exact
artifact under measurement is the one named on the command line.  Only the
kernel is launched here -- the Stage 5 wrapper / benchmark is never used.

Usage:
  python _bench_kernel.py --impl <candidate.py> --N <int> --dtype <str>
                          [--block_size <int>] [--kw k=v ...] [--iters <int>]
"""

import argparse
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401


def _load_impl(path: str):
    spec = importlib.util.spec_from_file_location("expm1_candidate_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["expm1_candidate_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--kw", action="append", default=[], help="extra factory kwarg key=value")
    ap.add_argument("--input", default="randn", choices=("randn", "linspace"))
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    mod = _load_impl(args.impl)
    factory = mod.expm1_fwd_kernel

    N = args.N
    dtype_str = args.dtype
    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype_str]
    if args.input == "linspace":
        x = torch.linspace(-9.0, 9.0, N, dtype=torch.float32).to(torch_dtype)
    else:
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype)
    x = x.npu()

    kwargs = {}
    if args.block_size is not None:
        kwargs["block_size"] = args.block_size
    for item in args.kw:
        key, _, val = item.partition("=")
        try:
            parsed = int(val)
        except ValueError:
            try:
                parsed = float(val)
            except ValueError:
                parsed = val
        kwargs[key] = parsed
    kernel = factory(N, dtype_str, **kwargs)

    for _ in range(args.iters):
        kernel(x, N)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
