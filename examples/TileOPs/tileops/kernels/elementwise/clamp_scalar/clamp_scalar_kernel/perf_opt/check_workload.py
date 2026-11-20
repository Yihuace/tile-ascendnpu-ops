#!/usr/bin/env python3
"""Workload-level precision check (one benchmark shape/dtype/block_size).

Complements the aggregated ``--level L0`` gate: it validates exactly the
configuration that ``msprof op`` is about to measure (shape, dtype, bounds and
the wrapper's block_size), against the torch golden.  Used for candidates whose
aggregated L0 hits the known Stage-3 UB-overflow case (fp32 at block_size 8192).

    python check_workload.py --module <kernel.py> --workload 16M --dtype float16 \
        --block-size 8192
"""

import argparse
import importlib.util
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

from bench_kernel import DTYPES, WORKLOADS  # noqa: E402

TOL = {"float16": 1e-3, "float32": 1e-5, "bfloat16": 1.6e-2}


def load_module(path):
    name = "prec_candidate_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--dtype", required=True, choices=sorted(DTYPES))
    ap.add_argument("--block-size", type=int, default=8192)
    args = ap.parse_args()

    N, lo, hi = WORKLOADS[args.workload]
    dtype = DTYPES[args.dtype]
    mod = load_module(args.module)
    prog = mod.clamp_scalar_fwd_kernel(N, args.dtype, min_val=lo, max_val=hi,
                                       block_size=args.block_size)
    torch.manual_seed(1234)
    x = torch.randn(N, dtype=dtype, device="npu")
    y = prog(x, N).cpu()
    ref = torch.clamp(x.cpu().float(), lo, hi).to(dtype)
    eq = (y == ref) | (torch.isnan(y) & torch.isnan(ref))
    diff = torch.where(eq, torch.zeros((), dtype=torch.float32),
                       (y.float() - ref.float()).abs())
    max_diff = diff.max().item()
    ok = max_diff <= TOL[args.dtype]
    mism = int((~eq).sum().item())
    print(f"[workload-precision] {'PASS' if ok else 'FAIL'} N={N} "
          f"dtype={args.dtype} min={lo} max={hi} block_size={args.block_size} "
          f"max_diff={max_diff:.3e} mismatched={mism}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
