#!/usr/bin/env python3
"""Standalone msprof driver for SubFwdOp Stage 4.

Runs ONE manifest workload directly against the Stage-3 kernel file or a
perf_opt candidate; no Stage-5 wrapper, no benchmark harness.

    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
        python perf_opt/bench_kernel.py --module <kernel.py> \
        --workload linear-bias --dtype float16 --launches 25
"""

import argparse
import importlib.util
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# label -> (input_shape, other_shape), from tileops/manifest/elementwise.yaml SubFwdOp
WORKLOADS = {
    "hidden-state-prefill": ([2048, 4096], [2048, 4096]),
    "cnn-feat-broadcast":   ([16, 256, 56, 56], [256, 1, 1]),
    "llama-7b-ffn":         ([1024, 11008], [1024, 11008]),
    "linear-bias":          ([1024, 4096], [1, 4096]),
    "llama-7b-hidden":      ([1024, 4096], [1024, 4096]),
    "phi-2-ffn":            ([1024, 10240], [1024, 10240]),
    "phi-2-ffn-bias":       ([1024, 10240], [1, 10240]),
    "llama-7b-ffn-bias":    ([1024, 11008], [1, 11008]),
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
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--block-size", type=int, default=8192)
    ap.add_argument("--launches", type=int, default=25)
    args = ap.parse_args()

    in_shape, other_shape = WORKLOADS[args.workload]
    out_shape = tuple(torch.broadcast_shapes(in_shape, other_shape))
    N = 1
    for d in out_shape:
        N *= d
    dtype = DTYPES[args.dtype]

    mod = load_module(args.module)
    alpha = int(args.alpha) if float(args.alpha).is_integer() else float(args.alpha)
    prog = mod.sub_fwd_kernel(N, args.dtype, alpha=alpha, block_size=args.block_size)

    a = torch.randn(in_shape, dtype=dtype, device="npu").expand(out_shape).contiguous().view(-1)
    b = torch.randn(other_shape, dtype=dtype, device="npu").expand(out_shape).contiguous().view(-1)

    print(f"[driver] module={os.path.abspath(args.module)} workload={args.workload} "
          f"dtype={args.dtype} N={N} launches={args.launches}", file=sys.stderr)
    with torch.no_grad():
        for _ in range(args.launches):
            prog(a, b, N)
        torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
