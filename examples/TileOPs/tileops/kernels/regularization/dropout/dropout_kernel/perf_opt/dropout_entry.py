#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Workload entry point for ``msprof op`` profiling of the dropout kernel.

Loads a dropout kernel module (baseline Stage 3 file or a ``perf_opt``
candidate) and runs a single benchmark workload so that ``msprof op`` can
capture the TileLang kernel launch.  No wrapper / op layer is involved.

Usage:
    python dropout_entry.py --kernel <path.py> --N 4194304 --dtype float16 \
        --p 0.5 --seed 42 --bs 2048 --warmup 5 --runs 20

The tensor is built with ``torch.randn`` (matching the benchmark workload
generator) so the profiled kernel is the exact benchmark path.
"""

import argparse
import importlib.util


def _load_module(path, name="dropout_candidate"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    args, _ = ap.parse_known_args()

    import torch  # noqa: E402
    import torch_npu  # noqa: F401,E402

    mod = _load_module(args.kernel)
    dtype = mod._TORCH_DTYPE[args.dtype]
    bs = args.bs if args.bs is not None else mod._DEFAULT_BLOCK_SIZE[args.dtype]

    factory = mod._make_dropout_kernel(args.N, args.dtype, float(args.p), int(args.seed))
    fn = factory(bs)
    x = torch.randn(args.N, dtype=dtype).npu()

    with torch.no_grad():
        for _ in range(args.warmup):
            fn(x)
        for _ in range(args.runs):
            fn(x)
    print(f"[entry] done N={args.N} dtype={args.dtype} bs={bs} "
          f"p={args.p} seed={args.seed} warmup={args.warmup} runs={args.runs}")


if __name__ == "__main__":
    main()
