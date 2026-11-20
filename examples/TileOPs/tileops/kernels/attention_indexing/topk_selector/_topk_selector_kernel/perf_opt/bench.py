"""Stage 4 measurement entry for TopkSelectorOp -- direct kernel launch under `msprof op`.

Why this file is not the generic single-kernel bench
----------------------------------------------------
The delivered op is TWO kernels plus a host-side bisection loop:

  * count kernel  -- Stage 3 design launches it `ROUNDS` (=40) times per op,
                     once per bisection step;
  * select kernel -- one launch, after the threshold converges.

Both prim_funcs are literally named `main`, and `msprof op` profiles one kernel
name at a time, so a single `--kernel-name=main` capture cannot be attributed to
count-vs-select.  This bench therefore measures the two kernels SEPARATELY
(`--mode count|select`) and `run_experiments.py` accounts for the number of
launches the delivered algorithm needs:

    op_latency = n_count_launches * count_us + 1 * select_us

`--mode op` runs the whole delivered path and exists only for the correctness
gate (L0); it is never used for the perf metric.

Metric discipline (tilelang-op-optimize): `msprof op Task Duration(us)` is the
only kernel-latency metric.  Host launch gaps are out of scope by construction.

Usage (from perf_opt/):
    msprof op --kernel-name=main --output=<dir> --launch-count=15 --warm-up=5 \
        --dump=off --aic-metrics=... \
        python bench.py --impl ../_topk_selector_kernel.py --case topk-1024 --mode count
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
# the delivered one.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: E402  (registers the "npu" device)


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


def make_input(m: int, n: int) -> torch.Tensor:
    """Distinct-valued rows (same construction as the Stage 3 harness).

    Distinct values keep the set comparison unambiguous: with ties at the
    boundary, which of the tied elements is chosen is not defined by the
    contract, so a tie input would test an undefined choice.
    """
    base = torch.randperm(n, dtype=torch.float32) * 7.0
    return base.unsqueeze(0).repeat(m, 1) + torch.arange(m, dtype=torch.float32).unsqueeze(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, help="candidate kernel file to measure")
    ap.add_argument("--case", required=True, help="benchmark workload_id")
    ap.add_argument("--mode", default="count", choices=["count", "select", "op"])
    ap.add_argument("--inventory", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "workload_inventory.json"))
    ap.add_argument("--launch-count", type=int, default=15)
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=0, help="0 = the candidate module's own ROUNDS")
    ap.add_argument("--check", action="store_true", help="also run the module's own golden comparison")
    args = ap.parse_args()

    case = load_case(args.case, args.inventory)
    m = int(case["params"]["m"])
    n = int(case["params"]["n"])
    k = int(case["params"]["k"])
    tile = case["params"].get("tile")

    module = load_impl(args.impl)
    rounds = args.rounds or int(getattr(module, "ROUNDS", 40))

    x_cpu = make_input(m, n)
    xn = x_cpu.npu()

    if args.mode == "count":
        if getattr(module, "_COUNT_IN_KERNEL", False):
            # Fused-bisection variant: one launch carries all `rounds` steps.
            kernel = module._topk_count_kernel(m, n, k, tile=tile)
            launch = lambda: kernel(xn)  # noqa: E731
            n_launches = 1
        else:
            kernel = module._topk_count_kernel(m, n, tile=tile)
            # Representative threshold: the exact k-th largest value per row, so
            # the mask density equals k/N exactly as it does mid-bisection.
            thr = torch.topk(x_cpu, k, dim=-1).values[:, -1].contiguous().npu()
            launch = lambda: kernel(xn, thr)  # noqa: E731
            n_launches = rounds
    elif args.mode == "select":
        kernel = module._topk_select_kernel(m, n, k, tile=tile)
        thr = torch.topk(x_cpu, k, dim=-1).values[:, -1].contiguous().npu()
        launch = lambda: kernel(xn, thr)  # noqa: E731
        n_launches = 1
    else:  # op: full delivered path, correctness only
        launch = lambda: module.topk_indices(x_cpu, k, rounds=rounds, tile=tile)  # noqa: E731
        n_launches = 1

    if args.check:
        got = launch()
        want = module.golden_topk_indices(x_cpu, k)
        ok = module.set_equal(got.cpu(), want)
        print(f"[check] {args.case} {args.mode} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1

    for _ in range(args.warm_up):
        launch()
    for _ in range(args.launch_count):
        launch()
    torch.npu.synchronize()
    print(
        json.dumps(
            {
                "case": args.case,
                "mode": args.mode,
                "impl": args.impl,
                "n_launches_per_op": n_launches,
                "launches": args.warm_up + args.launch_count,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
