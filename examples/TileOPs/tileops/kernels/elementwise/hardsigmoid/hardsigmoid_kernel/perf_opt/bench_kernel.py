#!/usr/bin/env python3
"""Standalone driver for msprof (HardsigmoidFwdOp Stage 4).

Runs ONE benchmark workload directly against the Stage-3 kernel file or a
perf_opt candidate; no Stage-5 wrapper, no benchmark harness.

    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
        python perf_opt/bench_kernel.py --module <kernel.py> \
        --workload fc-wide --dtype float16 --launches 25
"""

import argparse
import importlib.util
import os
import sys
from math import prod

os.environ["TILELANG_ASCEND_MODE"] = "Developer"  # force

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# label -> (input_shape,)   (unary elementwise)
WORKLOADS = {
    "smoke-1m": ([1048576],),
    "mbv3-se-gate": ([32, 240, 1, 1],),
    "mbv3-se-gate-deep": ([32, 960, 1, 1],),
    "fc-wide": ([2048, 4096],),
}
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# dtype-aware default block_size (mirrors the wrapper's _DEFAULT_BLOCK_SIZE).
DEFAULT_BLOCK = {"float32": 4096, "float16": 8192, "bfloat16": 8192}


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
    ap.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--launches", type=int, default=25)
    args = ap.parse_args()

    shape = WORKLOADS[args.workload][0]
    N = prod(shape)
    dtype = DTYPES[args.dtype]
    block_size = args.block_size if args.block_size else DEFAULT_BLOCK[args.dtype]

    mod = load_module(args.module)
    prog = mod.hardsigmoid_fwd_kernel(N, args.dtype, block_size=block_size)

    x = torch.randn(shape, dtype=dtype, device="npu").contiguous().view(-1)

    print(f"[driver] module={os.path.abspath(args.module)} workload={args.workload} "
          f"dtype={args.dtype} N={N} block_size={block_size} launches={args.launches}",
          file=sys.stderr)
    with torch.no_grad():
        for _ in range(args.launches):
            prog(x, N)
        torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
