#!/usr/bin/env python3
"""Standalone driver for msprof (GeluAndMulFwdOp Stage 4).

Runs ONE benchmark workload directly against a kernel module (baseline or a
perf_opt candidate); no Stage-5 wrapper, no benchmark harness.

    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
        python perf_opt/bench_kernel.py --module <kernel.py> \
        --workload prefill --dtype float16 --launches 25
"""

import argparse
import importlib.util
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Developer"  # force; matches wrapper

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# label -> (M, N); input x is [M, 2N], output [M, N].
WORKLOADS = {
    "llama-8b-ffn-prefill":  (2048, 14336),
    "llama-8b-ffn-decode":   (1, 14336),
    "qwen3-8b-swiglu-b32":   (32, 12288),
    "smoke":                 (16, 128),
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
    ap.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--num-cores", type=int, default=None)
    ap.add_argument("--launches", type=int, default=25)
    args = ap.parse_args()

    M, N = WORKLOADS[args.workload]
    dtype = DTYPES[args.dtype]

    mod = load_module(args.module)
    kwargs = {}
    if args.block_size is not None:
        kwargs["block_N"] = args.block_size
    if args.num_cores is not None:
        kwargs["num_cores"] = args.num_cores
    prog = mod.gelu_and_mul_fwd_kernel(M, N, args.dtype, **kwargs)

    x = torch.randn(M, 2 * N, dtype=dtype, device="npu")

    print(f"[driver] module={os.path.abspath(args.module)} workload={args.workload} "
          f"dtype={args.dtype} M={M} N={N} launches={args.launches}", file=sys.stderr)
    with torch.no_grad():
        for _ in range(args.launches):
            prog(x)
        torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
