"""Standalone msprof op probe for the Tensor-bound clamp kernel (one workload
per process).

Loads the kernel factory from an arbitrary candidate file so that the exact
artifact under measurement is the one named on the command line.  Only the
kernel is launched here -- the Stage 5 wrapper / benchmark is never used.

Factory contract measured:
    _make_clamp_tensor_kernel(N, dtype, has_min, has_max) -> kernel(block_size)
    compiled(x[, lo][, hi]) -> y

Usage:
  python _bench_kernel.py --impl <candidate.py> --N <int> --dtype <str>
                          [--block_size <int>] [--impl-defaults]
                          [--has-min] [--has-max] [--kw k=v ...] [--iters <int>]
"""

import argparse
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401


def _parse_kw(items):
    out = {}
    for item in items:
        key, _, val = item.partition("=")
        try:
            parsed = int(val)
        except ValueError:
            try:
                parsed = float(val)
            except ValueError:
                parsed = val
        out[key] = parsed
    return out


def _load_impl(path: str):
    spec = importlib.util.spec_from_file_location("clamp_candidate_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clamp_candidate_mod"] = mod
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
    ap.add_argument("--no-min", action="store_true", help="omit the lower bound")
    ap.add_argument("--no-max", action="store_true", help="omit the upper bound")
    ap.add_argument("--kw", action="append", default=[], help="extra inner-callable kwarg key=value")
    ap.add_argument("--fkw", action="append", default=[], help="extra factory kwarg key=value")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    has_min = not args.no_min
    has_max = not args.no_max
    if not (has_min or has_max):
        print("at least one bound required")
        return 2

    mod = _load_impl(args.impl)
    factory = mod._make_clamp_tensor_kernel

    N = args.N
    dtype_str = args.dtype
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(N, dtype=torch.float32, generator=gen).to(torch_dtype)
    lo = (torch.randn(N, dtype=torch.float32, generator=gen) - 0.5).to(torch_dtype)
    hi = (torch.randn(N, dtype=torch.float32, generator=gen) + 0.5).to(torch_dtype)
    x = x.npu()

    kwargs = {}
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

    kernel = factory(N, dtype_str, has_min, has_max, **_parse_kw(args.fkw))
    compiled = kernel(**kwargs)

    call_args = [x]
    if has_min:
        call_args.append(lo.npu())
    if has_max:
        call_args.append(hi.npu())

    for _ in range(args.iters):
        compiled(*call_args)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
