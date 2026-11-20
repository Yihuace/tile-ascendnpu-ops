"""Stage 4 measurement entry for the AdaLayerNorm family under `msprof op`.

Launches the target TileLang kernel directly with benchmark workload parameters
(the Stage 5 wrapper path is deliberately NOT used).

Family-specific calling convention: the factory takes
`(M, N, eps, dtype, has_gate=...)` and returns ANOTHER jitted function called
with `(block_m)`, so

    y = _ada_layer_norm_kernel(m, n, eps, dtype, has_gate=hg)(block_m)(x, sc, sh, g)

`g` is always passed (the Stage 3 harness passes all four tensors and the
has_gate=False path ignores it).  One kernel covers TWO manifest Ops:
AdaLayerNormFwdOp (has_gate=False) and AdaLayerNormZeroFwdOp (has_gate=True).

Usage (from perf_opt/):
    msprof op --kernel-name=main --output=<dir> --launch-count=15 --warm-up=5 \
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

FACTORY_NAME = "_ada_layer_norm_kernel"
GOLDEN_NAME = "golden_ada_layer_norm"


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
    ap.add_argument("--block-m", type=int, help="override the factory's _default_block_m(N)")
    args = ap.parse_args()

    case = load_case(args.case, args.inventory)
    dtype = DTYPES[case["dtype"]]
    m = int(case["params"]["m"])
    n = int(case["params"]["n"])
    eps = float(case["params"].get("eps", 1e-5))
    has_gate = bool(case["params"]["has_gate"])

    module = load_impl(args.impl)
    factory = getattr(module, FACTORY_NAME)
    dtype_str = str(dtype).replace("torch.", "")
    block_m = args.block_m or module._default_block_m(n)
    kernel = factory(m, n, eps, dtype_str, has_gate=has_gate)(block_m)

    x = torch.randn(m, n, dtype=dtype, device="npu")
    sc = torch.randn(m, n, dtype=dtype, device="npu")
    sh = torch.randn(m, n, dtype=dtype, device="npu")
    g = torch.randn(m, n, dtype=dtype, device="npu")

    if args.check:
        golden = getattr(module, GOLDEN_NAME)
        ref = golden(x.cpu(), sc.cpu(), sh.cpu(), g.cpu() if has_gate else None, eps)
        y = kernel(x, sc, sh, g)
        tol = module.TOLERANCES[dtype]
        torch.testing.assert_close(y, ref.to("npu"), atol=tol["atol"], rtol=tol["rtol"])
        print(f"[check] PASS {args.case} (block_m={block_m}, has_gate={has_gate})")

    for _ in range(args.warm_up):
        kernel(x, sc, sh, g)
    torch.npu.synchronize()
    for _ in range(args.launch_count):
        kernel(x, sc, sh, g)
    torch.npu.synchronize()
    print(f"[bench] {args.case}: {args.warm_up + args.launch_count} launches of {args.impl} (block_m={block_m})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
