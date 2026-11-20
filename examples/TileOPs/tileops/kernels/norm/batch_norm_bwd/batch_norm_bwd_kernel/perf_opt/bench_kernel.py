#!/usr/bin/env python3
"""Standalone msprof driver for BatchNormBwdOp Stage 4.

Measures one of the three migrated kernels directly (no Stage-5 wrapper):

    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
        --aic-metrics=Default,Roofline \
        python perf_opt/bench_kernel.py --module <kernel.py> --kernel wide \
        --workload resnet-stage1 --dtype float16 --launches 25

For the split kernel (`--kernel split`) the driver runs stats -> finalize ->
apply in sequence (exactly as the wrapper does); msprof is invoked WITHOUT a
kernel-name filter and the three kernel names are summed by the runner.
"""

import argparse
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# label -> [N, C, S], from tileops/manifest/norm.yaml BatchNormBwdOp
WORKLOADS = {
    "smoke-fc": [32, 64, 1],
    "resnet-stage1": [8, 64, 1024],
    "resnet-stage2": [4, 128, 1024],
    "resnet-stage3": [4, 256, 784],
    "large-spatial": [4, 128, 262144],
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
    ap.add_argument("--kernel", required=True, choices=["wide", "split", "general"])
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument("--launches", type=int, default=25)
    args = ap.parse_args()

    N, C, S = WORKLOADS[args.workload]
    dtype = DTYPES[args.dtype]
    dtype_str = args.dtype

    mod = load_module(args.module)
    torch.manual_seed(0)
    grad_out = (torch.randn(N, C, S) * 0.5).to(dtype).npu()
    x = torch.randn(N, C, S).to(dtype).npu()
    weight = torch.randn(C).npu()
    mean = torch.randn(C).npu()
    rstd = (torch.rand(C).abs() + 0.5).npu()

    print(f"[driver] module={os.path.abspath(args.module)} kernel={args.kernel} "
          f"workload={args.workload} dtype={args.dtype} N={N} C={C} S={S} "
          f"launches={args.launches}", file=sys.stderr)

    if args.kernel == "split":
        stats, finalize, apply_ = getattr(mod, "_batch_norm_bwd_split_kernel")(
            N, C, S, dtype_str
        )
        # recompute splits locally (mirrors _plan_splits in the split kernel)
        L = N * S
        splits = max(1, min(L, -(-512 // C)))
        splits = max(1, min(splits, S))
        partial_sum = torch.empty(C, splits, device="npu", dtype=torch.float32)
        partial_sum_xhat = torch.empty(C, splits, device="npu", dtype=torch.float32)
        grad_weight = torch.empty(C, device="npu", dtype=torch.float32)
        grad_bias = torch.empty(C, device="npu", dtype=torch.float32)
        scale = torch.empty(C, device="npu", dtype=torch.float32)
        shift = torch.empty(C, device="npu", dtype=torch.float32)
        centered = torch.empty(C, device="npu", dtype=torch.float32)
        s = stats()
        f = finalize()
        a = apply_()
        with torch.no_grad():
            for _ in range(args.launches):
                s(grad_out, x, mean, rstd, partial_sum, partial_sum_xhat)
                f(partial_sum, partial_sum_xhat, weight, rstd,
                  grad_weight, grad_bias, scale, shift, centered)
                a(grad_out, x, mean, scale, shift, centered)
            torch.npu.synchronize()
    else:
        factory_name = (
            "_batch_norm_bwd_wide_kernel" if args.kernel == "wide"
            else "_batch_norm_bwd_kernel"
        )
        factory = getattr(mod, factory_name)
        prog = factory(N, C, S, dtype_str)()
        with torch.no_grad():
            if args.kernel == "wide":
                for _ in range(args.launches):
                    prog(grad_out, x, weight, mean, rstd)
            else:
                grad_weight = torch.empty(C, device="npu", dtype=torch.float32)
                grad_bias = torch.empty(C, device="npu", dtype=torch.float32)
                for _ in range(args.launches):
                    prog(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)
            torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
