#!/usr/bin/env python3
"""Standalone msprof driver for FusedAddLayerNormFwdOp Stage 4.

    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
        python perf_opt/bench_kernel.py --module <kernel.py> \
        --workload llama-8b-prefill --dtype float16 --launches 25
"""

import argparse
import importlib.util
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# label -> [M, N], from tileops/manifest/norm.yaml FusedAddLayerNormFwdOp
WORKLOADS = {
    "smoke-llama": [64, 4096],
    "llama-8b-prefill": [2048, 4096],
    "llama-8b-decode": [1, 4096],
    "llama-70b-prefill": [2048, 8192],
    "llama-70b-decode": [1, 8192],
    "llama-13b-prefill": [2048, 5120],
}
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
EPS = 1e-5


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
    ap.add_argument("--block-size", type=int, default=0, help="unused (M/N tiling)")
    ap.add_argument("--block-m", type=int, default=0, help="override block_m (0 = factory default)")
    ap.add_argument("--launches", type=int, default=25)
    args = ap.parse_args()

    M, N = WORKLOADS[args.workload]
    dtype = DTYPES[args.dtype]

    mod = load_module(args.module)
    factory = getattr(mod, "_fused_add_layer_norm_kernel")
    prog = factory(M, N, EPS, args.dtype)(args.block_m) if args.block_m > 0 else factory(M, N, EPS, args.dtype)()

    x = torch.randn(M, N, dtype=dtype, device="npu")
    residual = torch.randn(M, N, dtype=dtype, device="npu")
    weight = torch.randn(N, dtype=dtype, device="npu")
    bias = torch.randn(N, dtype=dtype, device="npu")

    print(f"[driver] module={os.path.abspath(args.module)} workload={args.workload} "
          f"dtype={args.dtype} M={M} N={N} launches={args.launches}", file=sys.stderr)
    with torch.no_grad():
        for _ in range(args.launches):
            prog(x, residual, weight, bias)
        torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
