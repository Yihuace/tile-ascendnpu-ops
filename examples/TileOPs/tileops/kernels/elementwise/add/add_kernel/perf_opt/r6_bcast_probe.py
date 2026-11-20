#!/usr/bin/env python3
"""R6 broadcast fast-path probe: correctness + msprof timing for the row/scalar
reuse kernels (``add_bcast_row_kernel`` / ``add_bcast_scalar_kernel``).

Correctness arm (default): for one workload / dtype / alpha, run
  1. the flat path exactly as ``AddFwdOp`` does today (``expand`` + flatten +
     ``add_fwd_kernel``),
  2. the broadcast fast path selected by ``plan_broadcast``,
and compare both against the torch golden, elementwise BITWISE.

Timing arm (``--output DIR``): run under ``msprof op`` and print the Task
Duration median, so ``.agents/tools/ab_test.py`` can interleave the two arms.

    python3 r6_bcast_probe.py --workload linear-bias --dtype bfloat16 --check
    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
        python3 r6_bcast_probe.py --workload linear-bias --dtype bfloat16 \
        --impl bcast --launches 25
"""

import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

_HERE = os.path.dirname(os.path.abspath(__file__))
# perf_opt -> add_kernel -> add -> elementwise -> kernels -> tileops -> TileOPs
_TILEOPS_ROOT = os.path.abspath(os.path.join(_HERE, "../../../../../.."))
sys.path.insert(0, _TILEOPS_ROOT)

from tileops.kernels.elementwise.add.add_kernel.perf_opt.AddFwdOp import (  # noqa: E402
    add_bcast_row_kernel,
    add_bcast_scalar_kernel,
    add_fwd_kernel,
    plan_broadcast,
)

WORKLOADS = {
    "hidden-state-prefill": ([2048, 4096], [2048, 4096]),
    "cnn-feat-broadcast": ([16, 256, 56, 56], [256, 1, 1]),
    "llama-7b-ffn": ([1024, 11008], [1024, 11008]),
    "linear-bias": ([1024, 4096], [1, 4096]),
    # extra plan-coverage shapes (tests/ops/test_add.py)
    "bias-1-1-768": ([2, 512, 768], [1, 1, 768]),
    "row-2-512-1": ([2, 512, 768], [2, 512, 1]),
    "scalar-1-1": ([1024, 1024], [1, 1]),
    "col-bcast": ([1024, 4096], [1024, 1]),
}
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_INT_VIEW = {"float16": torch.int16, "bfloat16": torch.int16, "float32": torch.int32}


def _shapes(label):
    in_shape, other_shape = WORKLOADS[label]
    out_shape = tuple(torch.broadcast_shapes(in_shape, other_shape))
    n_total = 1
    for d in out_shape:
        n_total *= d
    return tuple(in_shape), tuple(other_shape), out_shape, n_total


def flat_arm(a, b, out_shape, n_total, dtype_str, alpha):
    """Current production path: materialize both operands, one flat kernel."""
    a_flat = a.expand(out_shape).contiguous().view(-1)
    b_flat = b.expand(out_shape).contiguous().view(-1)
    prog = add_fwd_kernel(n_total, dtype_str, alpha=alpha)
    return prog(a_flat, b_flat, n_total).view(out_shape)


def bcast_arm(a, b, plan, dtype_str, alpha):
    if plan["mode"] == "row":
        prog = add_bcast_row_kernel(
            plan["N"], plan["row"], dtype_str, alpha=alpha, block_size=plan["tile"]
        )
        return prog(a.view(-1), b.view(-1), plan["N"]).view(plan["out_shape"])
    prog = add_bcast_scalar_kernel(
        plan["R"],
        plan["L"],
        plan["b_len"],
        dtype_str,
        alpha=alpha,
        planes=plan["planes"],
    )
    return prog(
        a.view(plan["R"], plan["L"]),
        b.view(plan["chunk"], plan["planes"], 1),
        plan["R"],
    ).view(plan["out_shape"])


def describe(plan):
    if plan is None:
        return "flat (no plan)"
    if plan["mode"] == "row":
        return f"row: N={plan['N']} row={plan['row']} tile={plan['tile']} reps={plan['reps']}"
    return (
        f"scalar: N={plan['N']} R={plan['R']} L={plan['L']} b_len={plan['b_len']} "
        f"planes={plan['planes']} chunk={plan['chunk']} tile={plan['planes'] * plan['L']}"
    )


def check(args):
    dtype = DTYPES[args.dtype]
    in_shape, other_shape, out_shape, n_total = _shapes(args.workload)
    alpha = int(args.alpha) if float(args.alpha).is_integer() else float(args.alpha)
    plan = plan_broadcast(in_shape, other_shape, dtype.itemsize, alpha=alpha)
    print(f"[plan] {args.workload} {args.dtype} alpha={alpha}: {describe(plan)}")
    if plan is None:
        raise SystemExit("[plan] no fast path for this shape (expected for some cases)")

    a = torch.randn(in_shape, dtype=dtype, device="npu")
    b = torch.randn(other_shape, dtype=dtype, device="npu")
    if args.special:
        # NaN / +-inf / signed zero mix in both operands
        a = a.view(-1)
        b = b.view(-1)
        a[0::16] = float("inf")
        a[1::16] = float("-inf")
        a[2::16] = float("nan")
        a[3::16] = 0.0
        a[4::16] = -0.0
        b[5::16] = float("inf")
        b[6::16] = float("-inf")
        b[7::16] = float("nan")
        a = a.view(in_shape)
        b = b.view(other_shape)

    with torch.no_grad():
        y_bcast = bcast_arm(a, b, plan, args.dtype, alpha)
        y_flat = flat_arm(a, b, out_shape, n_total, args.dtype, alpha)
        golden = torch.add(a.float(), b.float(), alpha=alpha).to(dtype)

    iv = _INT_VIEW[args.dtype]
    yb = y_bcast.cpu().view(iv)
    yf = y_flat.cpu().view(iv)
    eq_flat = torch.equal(yb, yf)
    eq_gold = torch.equal(yb, golden.cpu().view(iv))
    md_flat = (y_bcast.float() - y_flat.float()).abs().max().item()
    nan_eq = bool((torch.isnan(y_bcast) == torch.isnan(golden)).all().item())
    if not eq_flat:
        bad = (yb != yf).view(-1)
        bad_nan = (torch.isnan(y_bcast).view(-1) & torch.isnan(y_flat).view(-1)).cpu()
        nan_cnt = int((bad & bad_nan).sum().item())
        print(
            f"[check] mismatches={int(bad.sum().item())} "
            f"nan_vs_nan={nan_cnt} real={int(bad.sum().item()) - nan_cnt}"
        )
    print(
        f"[check] {args.workload} {args.dtype} alpha={alpha} "
        f"bitwise_vs_flat={eq_flat} bitwise_vs_torch={eq_gold} "
        f"max_diff_flat={md_flat:.3e} nan_pattern_eq={nan_eq}"
    )
    if not (eq_flat and eq_gold and nan_eq):
        raise SystemExit("[check] FAIL")


def measure(args):
    out_dir = args.output
    if os.path.isdir(out_dir):
        subprocess.run(["rm", "-rf", out_dir], check=False)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof",
        "op",
        "--kernel-name=main",
        f"--launch-count={args.launch_count}",
        "--warm-up=5",
        "--aic-metrics=Default,Roofline",
        f"--output={out_dir}",
        sys.executable,
        os.path.abspath(__file__),
        "--workload",
        args.workload,
        "--dtype",
        args.dtype,
        "--alpha",
        str(args.alpha),
        "--impl",
        args.impl,
        "--launches",
        str(args.launches),
        "--no-msprof",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    print(proc.stdout[-2000:])
    print(proc.stderr[-2000:], file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"msprof failed rc={proc.returncode}")
    durations = []
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("Task Duration(us)"):
                    durations.append(float(row["Task Duration(us)"]))
    if not durations:
        raise SystemExit("no OpBasicInfo rows")
    print(
        f"[probe] n={len(durations)} median={statistics.median(durations):.3f}us "
        f"min={min(durations):.3f} max={max(durations):.3f}"
    )


def run(args):
    dtype = DTYPES[args.dtype]
    in_shape, other_shape, out_shape, n_total = _shapes(args.workload)
    alpha = int(args.alpha) if float(args.alpha).is_integer() else float(args.alpha)
    plan = plan_broadcast(in_shape, other_shape, dtype.itemsize, alpha=alpha)
    a = torch.randn(in_shape, dtype=dtype, device="npu")
    b = torch.randn(other_shape, dtype=dtype, device="npu")
    print(f"[plan] {args.workload} {args.dtype} alpha={alpha}: {describe(plan)}", file=sys.stderr)
    with torch.no_grad():
        for _ in range(args.launches):
            if args.impl == "flat" or plan is None:
                flat_arm(a, b, out_shape, n_total, args.dtype, alpha)
            else:
                bcast_arm(a, b, plan, args.dtype, alpha)
        torch.npu.synchronize()
    print("[probe] done", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="linear-bias", choices=sorted(WORKLOADS))
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--impl", default="bcast", choices=["bcast", "flat"])
    ap.add_argument("--launches", type=int, default=25)
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--special", action="store_true")
    ap.add_argument("--output", default="")
    ap.add_argument("--no-msprof", action="store_true")
    args = ap.parse_args()

    if args.output:
        measure(args)
    elif args.check:
        check(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
