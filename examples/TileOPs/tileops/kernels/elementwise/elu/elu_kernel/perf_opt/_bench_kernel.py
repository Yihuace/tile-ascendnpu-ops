"""Standalone msprof op probe for the ELU kernel (one workload per process).

Loads the kernel factory from an arbitrary candidate file so that the exact
artifact under measurement is the one named on the command line.  Only the
kernel is launched here -- the Stage 5 wrapper / benchmark is never used.

Usage:
  python _bench_kernel.py --impl <candidate.py> --N <int> --dtype <str>
                          [--block_size <int>] [--alpha <float>] [--kw k=v ...]
                          [--iters <int>]
"""

import argparse
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401


def _load_impl(path: str):
    spec = importlib.util.spec_from_file_location("elu_candidate_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["elu_candidate_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument(
        "--impl-defaults",
        action="store_true",
        help="use the impl module's per-dtype block_size table (wrapper parity)",
    )
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--kw", action="append", default=[], help="extra factory kwarg key=value")
    ap.add_argument("--input", default="randn", choices=("randn", "linspace"))
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    mod = _load_impl(args.impl)
    factory = mod._make_elu_kernel

    N = args.N
    dtype_str = args.dtype
    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype_str]
    if args.input == "linspace":
        x = torch.linspace(-9.0, 9.0, N, dtype=torch.float32).to(torch_dtype)
    else:
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype)
    x = x.npu()

    kwargs = {"alpha": args.alpha}
    if args.block_size is not None:
        kwargs["block_size"] = args.block_size
    elif args.impl_defaults:
        # Mirror the Stage-5 wrapper dispatch: it passes
        # ``block_size = _DEFAULT_BLOCK_SIZE[dtype]`` (baseline) /
        # ``TUNED_BLOCK_SIZE[dtype]`` (perf_opt) to the factory.
        table = getattr(mod, "TUNED_BLOCK_SIZE", None) or getattr(
            mod, "_DEFAULT_BLOCK_SIZE", None
        )
        if table is not None:
            kwargs["block_size"] = table[dtype_str]
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
