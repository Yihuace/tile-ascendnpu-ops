"""Stage 4 measurement entry for FusedAddRMSNormFwdOp under `msprof op`.

Launches the target TileLang kernel directly with benchmark workload parameters
(the Stage 5 wrapper path is deliberately NOT used).

Family-specific calling convention: the factory takes `(M, N, eps, dtype)` and
returns ANOTHER jitted function called with `(block_m)`, so

    y, residual_out = _fused_add_rms_norm_kernel(m, n, eps, dtype)(1)(x, residual, weight)

Both outputs are compared against the module's own `golden_fused_add_rms_norm`.

Usage (from perf_opt/):
    msprof op --kernel-name=<name> --output=<dir> --launch-count=15 --warm-up=5 \
        --dump=off --aic-metrics=... \
        python bench.py --impl <kernel.py> --case <workload_id>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: E402  (registers the "npu" device)

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

FACTORY_NAME = "_fused_add_rms_norm_kernel"
GOLDEN_NAME = "golden_fused_add_rms_norm"
BLOCK_M = 1  # the factory's inner _func takes block_m (Stage 3 harness uses 1)


def load_impl(path: str):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("stage4_candidate", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4_candidate"] = module
    spec.loader.exec_module(module)
    return module


def load_case(case_id: str, inventory_path: str) -> dict:
    with open(inventory_path, encoding="utf-8") as handle:
        inventory = json.load(handle)
    for item in inventory["workloads"]:
        if item["workload_id"] == case_id:
            return item
    raise SystemExit(f"unknown case {case_id!r} in {inventory_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--inventory", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "workload_inventory.json"))
    ap.add_argument("--launch-count", type=int, default=15)
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--num-cores", type=int, help="override the factory's grid cap")
    args = ap.parse_args()

    case = load_case(args.case, args.inventory)
    dtype = DTYPES[case["dtype"]]
    m = int(case["params"]["m"])
    n = int(case["params"]["n"])
    eps = float(case["params"].get("eps", 1e-6))

    module = load_impl(args.impl)
    factory = getattr(module, FACTORY_NAME)
    dtype_str = str(dtype).replace("torch.", "")
    if args.num_cores:
        kernel = factory(m, n, eps, dtype_str, num_cores=args.num_cores)(BLOCK_M)
    else:
        kernel = factory(m, n, eps, dtype_str)(BLOCK_M)

    x = torch.randn(m, n, dtype=dtype, device="npu")
    residual = torch.randn(m, n, dtype=dtype, device="npu")
    weight = torch.randn(n, dtype=dtype, device="npu")

    if args.check:
        golden = getattr(module, GOLDEN_NAME)
        ref_y, ref_r = golden(x.cpu(), residual.cpu(), weight.cpu(), eps)
        y, ro = kernel(x, residual, weight)
        tol = module.TOLERANCES[dtype]
        torch.testing.assert_close(y, ref_y.to("npu"), atol=tol["atol"], rtol=tol["rtol"])
        torch.testing.assert_close(ro, ref_r.to("npu"), atol=tol["atol"], rtol=tol["rtol"])
        print(f"[check] PASS {args.case} (both outputs)")

    for _ in range(args.warm_up):
        kernel(x, residual, weight)
    torch.npu.synchronize()
    for _ in range(args.launch_count):
        kernel(x, residual, weight)
    torch.npu.synchronize()
    print(f"[bench] {args.case}: {args.warm_up + args.launch_count} launches of {args.impl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
