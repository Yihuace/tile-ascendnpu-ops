"""Device accuracy probe: run a candidate factory and compare vs torch.erf.

Usage: python _check_acc.py --impl probe_tanh.py --degree 4 [--block_size 2048]
Prints max abs diff (fp64 comparison) per dtype and for corner cases.
"""

import argparse
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401

_DT = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_TOL = {"float32": 1e-5, "float16": 1e-3, "bfloat16": 1.6e-2}


def _load(path):
    spec = importlib.util.spec_from_file_location("cand_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cand_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _diff(y, g):
    return (y.cpu().to(torch.float64) - g.cpu().to(torch.float64)).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--clamp", type=float, default=None)
    args = ap.parse_args()
    mod = _load(args.impl)
    kw = {}
    if args.degree is not None:
        kw["degree"] = args.degree
    if args.mode is not None:
        kw["mode"] = args.mode
    if args.clamp is not None:
        kw["clamp"] = args.clamp

    def run(x_cpu, dt):
        N = x_cpu.numel()
        kern = mod.erf_fwd_kernel(N, dt, **kw)
        y = kern(x_cpu.npu().to(_DT[dt]), N).cpu()
        g = torch.erf(x_cpu.to(torch.float32)).to(_DT[dt])
        return _diff(y, g), _TOL[dt]

    worst = 0.0
    for dt in ("float32", "float16", "bfloat16"):
        x = torch.randn(1 << 20)
        d, tol = run(x, dt)
        print(f"  {dt} randn-1M", f"max_diff={d:.4e} tol={tol:.1e} margin={tol/max(d,1e-30):.1f}x")
        worst = max(worst, d / tol)
    # dense linspace full domain
    x = torch.linspace(-8, 8, 1 << 20)
    d, tol = run(x, "float32")
    print(f"  fp32 linspace[-8,8]", f"max_diff={d:.4e} margin={tol/d:.1f}x")
    worst = max(worst, d / tol)
    # corner
    for dt in ("float32", "float16", "bfloat16"):
        special = torch.tensor([0.0, 1e-8, -1e-8, 3.0, -3.0, 3.9, -3.9, 5.0, 100.0, -100.0])
        x = special.repeat(4000)[:4096]
        d, tol = run(x, dt)
        print(f"  {dt} corner", f"max_diff={d:.4e} margin={tol/max(d,1e-30):.1f}x")
        worst = max(worst, d / tol)
    # inf/nan
    for dt in ("float32", "float16", "bfloat16"):
        td = _DT[dt]
        x = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0], dtype=torch.float64).repeat(1024)
        xb = x.to(td)
        N = xb.numel()
        kern = mod.erf_fwd_kernel(N, dt, **kw)
        y = kern(xb.npu(), N).cpu()
        g = torch.erf(x.to(torch.float32)).to(td)
        ok_nan = bool(torch.isnan(y).sum() == torch.isnan(g).sum())
        print(f"  {dt} inf/nan", "nan-match" if ok_nan else "NAN MISMATCH", y[:4].tolist(), g[:4].tolist())
    print(f"worst gate-ratio (should be <1): {worst:.3f}")


if __name__ == "__main__":
    sys.exit(main())
