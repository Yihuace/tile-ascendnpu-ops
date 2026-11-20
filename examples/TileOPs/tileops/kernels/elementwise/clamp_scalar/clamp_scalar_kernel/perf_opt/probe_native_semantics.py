#!/usr/bin/env python3
"""R2 NaN / bound-rounding probe for the native-dtype clamp path.

Re-runs the R1 falsification experiment (586 NaN points x 3 bound forms) on
the native path, plus a bound-rounding scan for bounds that are *not*
representable in the compute dtype, and a min>max scan.

    python probe_native_semantics.py --modules ClampScalarFwdOp_r2_combined.py

Every row prints ``nan_ok`` (NaN mask identical to the golden) and
``max_diff`` (NaN/Inf aware absolute difference).  Exit code is non-zero if any
row fails.
"""

import argparse
import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401

_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
_TOL = {"float16": 1e-3, "float32": 1e-5, "bfloat16": 1.6e-2}
NAN_POINTS = 586


def load_module(path):
    name = "probe_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def golden(x, lo, hi):
    return torch.clamp(x.detach().cpu().float(), lo, hi).to(x.dtype)


def max_diff(out, ref):
    eq = (out == ref) | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(eq, torch.zeros((), dtype=torch.float32),
                       (out.float() - ref.float()).abs())
    return diff.max().item()


def make_special_input(dtype, n=1048576, nan_points=NAN_POINTS):
    x = torch.randn(n, dtype=torch.float32)
    x[1::5] = float("inf")
    x[2::5] = float("-inf")
    x[3::5] = 0.0
    x[4::5] = -0.0
    # NaN is written LAST so exactly ``nan_points`` positions hold NaN
    step = n // (nan_points + 1)
    x[step // 2 :: step][:nan_points] = float("nan")
    return x.to(dtype).npu()


def check(mod, x, lo, hi, dtype, tag):
    kernel = mod.clamp_scalar_fwd_kernel(x.numel(), dtype, min_val=lo, max_val=hi)
    y = kernel(x, x.numel()).cpu()
    ref = golden(x, lo, hi)
    nd = max_diff(y, ref)
    nan_ok = bool(torch.equal(torch.isnan(y), torch.isnan(ref)))
    ok = nan_ok and (nd <= _TOL[dtype] or nd != nd)
    n_in = int(torch.isnan(x.cpu()).sum().item())
    n_out = int(torch.isnan(y).sum().item())
    print(f"{tag:<34} dtype={dtype:<8} min={lo!r:>10} max={hi!r:>10} "
          f"nan_in={n_in:>4} nan_out={n_out:>4} nan_ok={nan_ok} "
          f"max_diff={nd:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok, nd, nan_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="+", required=True)
    args = ap.parse_args()

    failures = 0
    for path in args.modules:
        mod = load_module(path)
        print(f"\n=== module: {os.path.basename(path)} ===")
        for dtype in ("float32", "float16", "bfloat16"):
            x = make_special_input(_TORCH_DTYPE[dtype])
            for lo, hi, name in [(-0.5, 0.5, "both"), (-0.5, None, "min-only"),
                                 (None, 0.5, "max-only")]:
                ok, _, _ = check(mod, x, lo, hi, dtype, f"nan586-{name}")
                failures += 0 if ok else 1
            # min > max and min == max
            for lo, hi, name in [(0.5, -0.5, "min>max"), (0.25, 0.25, "min==max")]:
                ok, _, _ = check(mod, x, lo, hi, dtype, name)
                failures += 0 if ok else 1
            # bounds not representable on the dtype grid
            torch.manual_seed(7)
            for k in range(3):
                lo = float(torch.empty(()).uniform_(-1.0, -0.1))
                hi = float(torch.empty(()).uniform_(0.1, 1.0))
                ok, _, _ = check(mod, x, lo, hi, dtype, f"round-bound-{k}")
                failures += 0 if ok else 1
            # dense grid around a non-representable bound
            if dtype == "float16":
                xg = torch.arange(-65504, 65504, dtype=torch.float32)[:1048576]
                xg = xg.to(torch.float16).npu()
                lo = float(torch.tensor(-0.3333333, dtype=torch.float32))
                hi = float(torch.tensor(1.0000001, dtype=torch.float32))
                ok, _, _ = check(mod, xg, lo, hi, dtype, "sweep-grid")
                failures += 0 if ok else 1

    print(f"\nprobe failures: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    sys.exit(main())
