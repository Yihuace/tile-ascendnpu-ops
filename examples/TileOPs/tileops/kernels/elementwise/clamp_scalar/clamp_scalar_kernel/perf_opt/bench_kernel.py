#!/usr/bin/env python3
"""Standalone driver for ``msprof op`` on one clamp workload (Stage 4, R2).

Runs one benchmark workload directly against a Stage-3 / perf_opt / branch
kernel module -- no Stage-5 wrapper, no benchmark harness::

    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
        python perf_opt/bench_kernel.py --module <kernel.py> \
        --workload 16M --dtype float16 --launches 25
"""

import argparse
import importlib.util
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# label -> (N, min_val, max_val); N comes from the manifest input_shape product.
WORKLOADS = {
    "16M": (4096 * 4096, -0.5, 0.5),
    "256M": (16384 * 16384, -0.5, 0.5),
    "16M-min-only": (4096 * 4096, -0.5, None),
    "16M-max-only": (4096 * 4096, None, 0.5),
    "64M-min-only": (8192 * 8192, -0.5, None),
    "256M-min-only": (16384 * 16384, -0.5, None),
}
DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


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
    ap.add_argument("--block-size", type=int, default=8192)
    ap.add_argument("--launches", type=int, default=25)
    args = ap.parse_args()

    N, min_val, max_val = WORKLOADS[args.workload]
    dtype = DTYPES[args.dtype]

    mod = load_module(args.module)
    prog = mod.clamp_scalar_fwd_kernel(
        N, args.dtype, min_val=min_val, max_val=max_val,
        block_size=args.block_size,
    )
    x = torch.randn(N, dtype=dtype, device="npu")

    print(f"[driver] module={os.path.abspath(args.module)} "
          f"workload={args.workload} dtype={args.dtype} N={N} "
          f"block_size={args.block_size} launches={args.launches}",
          file=sys.stderr)
    with torch.no_grad():
        for _ in range(args.launches):
            prog(x, N)
        torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
