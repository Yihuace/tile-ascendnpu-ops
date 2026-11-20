"""Stage 4 measurement entry for the vector-norm family under `msprof op`.

This is the Stage 4 counterpart of the Stage 5 wrapper benchmark: it launches the
target TileLang kernel directly with benchmark workload parameters, so that
`msprof op --kernel-name=...` captures the target kernel and nothing else.  The
Stage 5 wrapper path is deliberately NOT used.

Family-specific calling convention (differs from the elementwise ops): the
factory takes `(M, N, op_kind, dtype)` and returns ANOTHER jitted function which
is then called with `(block_m)` -- i.e.

    kernel = _vector_norm_kernel(m, n, op_kind, dtype)(block_m)
    y = kernel(x)                      # x is (M, N)

`op_kind` is a trace-time Python value ("l1" / "l2" / "inf"), which is why three
manifest Ops share this one kernel.

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

# Developer (Vector) lowering path must be fixed before the target module is
# imported (vector_norm Stage 3 evidence E1).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: E402  (registers the "npu" device)

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

FACTORY_NAME = "_vector_norm_kernel"
GOLDEN_NAME = "golden_vector_norm"


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
    ap.add_argument("--block-m", type=int, help="override pick_block_m()")
    ap.add_argument("--num-cores", type=int, help="override the factory's grid cap")
    args = ap.parse_args()

    case = load_case(args.case, args.inventory)
    dtype = DTYPES[case["dtype"]]
    m = int(case["params"]["m"])
    n = int(case["params"]["n"])
    op_kind = case["params"]["op_kind"]

    module = load_impl(args.impl)
    factory = getattr(module, FACTORY_NAME)

    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    block_m = args.block_m or module.pick_block_m(m, n, elem_bytes)
    if args.num_cores:
        kernel = factory(m, n, op_kind, str(dtype).replace("torch.", ""), num_cores=args.num_cores)(block_m)
    else:
        kernel = factory(m, n, op_kind, str(dtype).replace("torch.", ""))(block_m)

    x = torch.randn(m, n, dtype=dtype, device="npu")

    if args.check:
        golden = getattr(module, GOLDEN_NAME)(x.cpu(), op_kind).to("npu")
        y = kernel(x)
        tol = module.TOLERANCES[dtype]
        torch.testing.assert_close(y, golden, atol=tol["atol"], rtol=tol["rtol"])
        print(f"[check] PASS {args.case} (block_m={block_m})")

    for _ in range(args.warm_up):
        kernel(x)
    torch.npu.synchronize()
    for _ in range(args.launch_count):
        kernel(x)
    torch.npu.synchronize()
    print(f"[bench] {args.case}: {args.warm_up + args.launch_count} launches of {args.impl} (block_m={block_m})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
