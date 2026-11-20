"""Stage 4 measurement entry for AbsFwdOp -- direct kernel launch under `msprof op`.

This is the Stage 4 counterpart of the Stage 5 wrapper benchmark: it launches the
target TileLang kernel directly with benchmark workload parameters, so that
`msprof op --kernel-name=main` captures the target kernel and nothing else.
The Stage 5 wrapper path is deliberately NOT used (see
`tilelang-op-optimize/references/profile-collection.md` §3).

Usage (from perf_opt/):
    msprof op --kernel-name=main --output=<dir> --launch-count=15 --warm-up=5 \
        --dump=off --aic-metrics=... \
        python bench.py --impl <kernel.py> --case elementwise-16m-float16

Without msprof it is also runnable directly, which is what the batch runner does
when it needs a plain latency smoke check.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

# The kernel family compiles through the Developer (Vector) path.  This must be
# in the environment BEFORE the target module is imported, otherwise the jit
# compiles against the wrong lowering path and the measured kernel differs from
# the delivered one (vector_norm Stage 3 evidence E1).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: E402  (registers the "npu" device)

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# ---- per-op adaptation -------------------------------------------------------
# FACTORY_NAME: the `@tilelang.jit` factory in the candidate module.
# GOLDEN_NAME: the candidate module's own CPU reference (reused verbatim, so the
#   Stage 4 precision gate matches the Stage 3 gate exactly).
# PASS_N_SCALAR: ceil's prim_func carries `shape: T.int32` and its Stage 3 harness
#   calls `kernel(x, n)`; abs's takes the tensor alone.
FACTORY_NAME = "exp_fwd_kernel"
GOLDEN_NAME = "golden_exp"
PASS_N_SCALAR = True
# -----------------------------------------------------------------------------


def load_impl(path: str):
    """Import a candidate kernel file as a standalone module."""
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
    ap.add_argument("--impl", required=True, help="candidate kernel file to measure")
    ap.add_argument("--case", required=True, help="benchmark workload_id")
    ap.add_argument("--inventory", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "workload_inventory.json"))
    ap.add_argument("--launch-count", type=int, default=15)
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="also run the golden comparison")
    # Sweep knobs; when omitted the candidate module's own defaults apply.
    ap.add_argument("--block-size", type=int)
    ap.add_argument("--num-cores", type=int)
    args = ap.parse_args()

    case = load_case(args.case, args.inventory)
    dtype = DTYPES[case["dtype"]]
    n_total = int(case["params"]["n_total"])

    module = load_impl(args.impl)
    factory = getattr(module, FACTORY_NAME)

    # `block_size` / `num_cores` are the kernel's exposed tunables; an explicit
    # flag wins, otherwise the candidate module's own default applies.
    kwargs = {}
    if args.block_size:
        kwargs["block_size"] = args.block_size
    if args.num_cores:
        kwargs["num_cores"] = args.num_cores
    kernel = factory(n_total, str(dtype).replace("torch.", ""), **kwargs)

    x = torch.randn(n_total, dtype=dtype, device="npu")

    # Some kernels in this family take an explicit runtime extent (the prim_func
    # signature carries `shape: T.int32` and the Stage 3 harness calls
    # `kernel(x, n)`); abs does not.  PASS_N_SCALAR pins which convention applies.
    def launch():
        return kernel(x, n_total) if PASS_N_SCALAR else kernel(x)

    if args.check:
        # Reuse the candidate module's own CPU reference verbatim.
        golden = getattr(module, GOLDEN_NAME)(x.cpu()).to("npu")
        y = launch()
        atol = {"float16": 1e-3, "bfloat16": 1.6e-2, "float32": 1e-5}[case["dtype"]]
        torch.testing.assert_close(y, golden, atol=atol, rtol=atol)
        print(f"[check] PASS {args.case}")

    for _ in range(args.warm_up):
        launch()
    torch.npu.synchronize()
    for _ in range(args.launch_count):
        launch()
    torch.npu.synchronize()
    print(f"[bench] {args.case}: {args.warm_up + args.launch_count} launches of {args.impl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
