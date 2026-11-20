#!/usr/bin/env python3
"""R9 diagnostic driver: time an arbitrary (M, N, dtype, block_m) with the
msprof `op` launcher.  Used to read the block_m scaling curve at a shape where
block_m=3 already fits, so the value of the R9 byte work is known before it is
written.

    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
      python bench_shape.py --module <kernel.py> --m 2048 --n 2048 \
             --dtype float16 --block-m 3 --launches 25
"""
import argparse
import importlib.util
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
EPS = 1e-5


def load_module(path):
    name = "r9probe_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    ap.add_argument("--block-m", type=int, default=0)
    ap.add_argument("--launches", type=int, default=25)
    args = ap.parse_args()

    mod = load_module(args.module)
    factory = getattr(mod, "_fused_add_layer_norm_kernel")
    prog = (factory(args.m, args.n, EPS, args.dtype)(args.block_m)
            if args.block_m > 0 else factory(args.m, args.n, EPS, args.dtype)())
    dtype = DTYPES[args.dtype]
    x = torch.randn(args.m, args.n, dtype=dtype, device="npu")
    residual = torch.randn(args.m, args.n, dtype=dtype, device="npu")
    weight = torch.randn(args.n, dtype=dtype, device="npu")
    bias = torch.randn(args.n, dtype=dtype, device="npu")
    print(f"[driver] M={args.m} N={args.n} dtype={args.dtype} bm={args.block_m}",
          file=sys.stderr)
    with torch.no_grad():
        for _ in range(args.launches):
            prog(x, residual, weight, bias)
        torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
