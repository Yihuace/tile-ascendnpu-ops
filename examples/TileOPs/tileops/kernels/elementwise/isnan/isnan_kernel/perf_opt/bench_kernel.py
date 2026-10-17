#!/usr/bin/env python3
"""Standalone msprof driver for IsnanFwdOp Stage 4 (unary, bool via int8).

    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
        python perf_opt/bench_kernel.py --module <kernel.py> \
        --workload 16m --dtype float16 --launches 25
"""

import argparse
import importlib.util
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

WORKLOADS = {
    "smoke-1m": [1048576],
    "16m": [4096, 4096],
    "256m": [16384, 16384],
    "16m-half": [4096, 4096],
}
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_module(path):
    name = "perf_candidate_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument("--block-size", type=int, default=8192)
    ap.add_argument("--launches", type=int, default=25)
    args = ap.parse_args()

    shape = WORKLOADS[args.workload]
    N = 1
    for d in shape:
        N *= d
    dtype = DTYPES[args.dtype]

    mod = load_module(args.module)
    prog = mod.isnan_fwd_kernel(N, args.dtype, block_size=args.block_size)
    # Inject a few NaNs so the predicate path is exercised (not a perf factor).
    x = torch.randn(shape, dtype=dtype, device="npu").view(-1)
    x[::4096] = float("nan")

    print(f"[driver] module={os.path.abspath(args.module)} workload={args.workload} "
          f"dtype={args.dtype} N={N} launches={args.launches}", file=sys.stderr)
    with torch.no_grad():
        for _ in range(args.launches):
            prog(x, N)
        torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
