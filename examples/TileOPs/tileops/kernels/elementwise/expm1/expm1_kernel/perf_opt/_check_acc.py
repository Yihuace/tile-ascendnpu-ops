"""Device accuracy probe for an expm1 candidate factory (non-gate diagnostic).

Loads ``--impl`` (a module), builds the factory ``--factory`` at the given
extra kwargs, and compares against ``torch.expm1`` in fp64 on the gate-critical
distributions (randn 1M, small-x band rel-check, full linspace, corners,
inf/nan).  Prints max abs / max rel and the gate margin.

Usage:
  python _check_acc.py --impl probe_struct.py --factory expm1_struct \
      --kw mode=deg3_blend --kw block_size=6144
"""

import argparse
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401

_DT = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_TOL = {"float32": (1e-5, 1e-5), "float16": (1e-3, 1e-3), "bfloat16": (1.6e-2, 1.6e-2)}


def _load(path):
    spec = importlib.util.spec_from_file_location("expm1_cand_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["expm1_cand_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_kw(items):
    kw = {}
    for item in items:
        k, _, v = item.partition("=")
        try:
            kw[k] = int(v)
        except ValueError:
            try:
                kw[k] = float(v)
            except ValueError:
                kw[k] = v
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--factory", default="expm1_fwd_kernel")
    ap.add_argument("--kw", action="append", default=[])
    args = ap.parse_args()
    mod = _load(args.impl)
    factory = getattr(mod, args.factory)
    kw = _parse_kw(args.kw)

    def run(x_cpu, dt, rel_check=False):
        N = x_cpu.numel()
        kern = factory(N, dt, **kw)
        # Quantize to the target dtype FIRST so kernel and golden see the same
        # input (the real harness builds the input directly in the target dtype).
        xq = x_cpu.to(_DT[dt])
        y = kern(xq.npu(), N).cpu()
        g = torch.expm1(xq.to(torch.float32)).to(_DT[dt])
        y64, g64 = y.to(torch.float64), g.to(torch.float64)
        fin = torch.isfinite(g64)
        abs_d = (y64[fin] - g64[fin]).abs().max().item() if bool(fin.any()) else 0.0
        nz = torch.isfinite(g64) & (g64 != 0)
        rel_d = ((y64[nz] - g64[nz]).abs() / g64[nz].abs()).max().item() if bool(nz.any()) else 0.0
        atol, rtol = _TOL[dt]
        ok = torch.allclose(y64, g64, rtol=rtol, atol=atol, equal_nan=True)
        rflag = ""
        if rel_check:
            rflag = f" REL{'PASS' if rel_d <= rtol else 'FAIL(>rtol)'}"
        return abs_d, rel_d, ok, rflag, atol, rtol

    worst_ratio = 0.0
    print(f"== {args.impl} factory={args.factory} kw={kw} ==")
    for dt in ("float32", "float16", "bfloat16"):
        x = torch.randn(1 << 20)
        a, r, ok, _, atol, rtol = run(x, dt)
        print(f"  {dt:8s} randn-1M    max_abs={a:.4e} max_rel={r:.4e} gate={'PASS' if ok else 'FAIL'}")
        if dt == "float32":
            worst_ratio = max(worst_ratio, r / rtol)
    a, r, ok, rf, _, rtol = run(torch.linspace(-1e-3, 1e-3, 1 << 20), "float32", rel_check=True)
    print(f"  fp32 small-x-band max_abs={a:.4e} max_rel={r:.4e} gate={'PASS' if ok else 'FAIL'}{rf}")
    worst_ratio = max(worst_ratio, r / rtol)
    a, r, ok, rf, _, rtol = run(torch.tensor([0.0, 88.8, -88.8, 1e-7]).repeat(1024), "float32", rel_check=True)
    print(f"  fp32 corner       max_abs={a:.4e} max_rel={r:.4e} gate={'PASS' if ok else 'FAIL'}{rf}")
    worst_ratio = max(worst_ratio, r / rtol)
    a, r, ok, _, _, _ = run(torch.linspace(-9.0, 9.0, 1 << 20), "float32")
    print(f"  fp32 linspace[-9,9] max_abs={a:.4e} max_rel={r:.4e} gate={'PASS' if ok else 'FAIL'}")
    for dt in ("float32", "float16", "bfloat16"):
        xb = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0]).repeat(1024).to(_DT[dt])
        N = xb.numel()
        y = factory(N, dt, **kw)(xb.npu(), N).cpu()
        g = torch.expm1(xb.to(torch.float32)).to(_DT[dt])
        nan_ok = int(torch.isnan(y).sum()) == int(torch.isnan(g).sum())
        inf_ok = bool(torch.equal(torch.nan_to_num(y, nan=7.0), torch.nan_to_num(g, nan=7.0)))
        print(f"  {dt:8s} inf/nan     nan-match={nan_ok} value-match={inf_ok} y={y[:2].tolist()}")
    print(f"  worst gate ratio (rel/rtol, should be <<1): {worst_ratio:.3f}")


if __name__ == "__main__":
    sys.exit(main())
