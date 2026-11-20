#!/usr/bin/env python3
"""Numeric debug for a wide-kernel candidate: prints gb/gw/grad_x vs golden."""

import importlib.util
import sys

import torch
import torch_npu  # noqa: F401


def load(path):
    spec = importlib.util.spec_from_file_location("dbg_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load(sys.argv[1])
    N, C, S = (int(v) for v in sys.argv[2].split(","))
    dtype = sys.argv[3]
    td = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    torch.manual_seed(0)
    go = (torch.randn(N, C, S) * 0.5).to(td)
    x = torch.randn(N, C, S).to(td)
    w = torch.randn(C)
    mean = torch.randn(C)
    rstd = torch.rand(C).abs() + 0.5

    gx_g, gw_g, gb_g = mod.golden_batch_norm_bwd(go, x, w, mean, rstd)

    fac = mod._batch_norm_bwd_wide_kernel(N, C, S, dtype)
    gx, gw, gb = fac()(go.npu(), x.npu(), w.npu(), mean.npu(), rstd.npu())
    gx = gx.cpu().float()
    gw = gw.cpu().float()
    gb = gb.cpu().float()

    print("gw max_abs_err", (gw - gw_g).abs().max().item())
    print("gb max_abs_err", (gb - gb_g).abs().max().item())
    print("gx max_abs_err", (gx - gx_g.float()).abs().max().item())
    print("gw got ", gw[:4].tolist())
    print("gw ref ", gw_g[:4].tolist())
    print("gb got ", gb[:4].tolist())
    print("gb ref ", gb_g[:4].tolist())




def solve_coeffs():
    """Recover the effective A/B/C the kernel applied for one channel."""
    mod = load(sys.argv[1])
    N, C, S = 2, 8, 3
    td = torch.float16
    torch.manual_seed(0)
    go = (torch.randn(N, C, S) * 0.5).to(td)
    x = torch.randn(N, C, S).to(td)
    w = torch.randn(C)
    mean = torch.randn(C)
    rstd = torch.rand(C).abs() + 0.5
    gx_g, gw_g, gb_g = mod.golden_batch_norm_bwd(go, x, w, mean, rstd)
    gx, gw, gb = mod._batch_norm_bwd_wide_kernel(N, C, S, "float16")()(
        go.npu(), x.npu(), w.npu(), mean.npu(), rstd.npu())
    gx = gx.cpu().float()
    L = N * S
    c = 0
    gf = go[:, c, :].float().reshape(-1)
    xf = x[:, c, :].float().reshape(-1)
    out = gx[:, c, :].reshape(-1)
    A = (w[c] * rstd[c]) / L
    B = -A * gw_g[c] * rstd[c]
    Cc = A * gw_g[c] * rstd[c] * mean[c] - A * gb_g[c]
    print("expected A,B,C", A.item(), B.item(), Cc.item())
    # solve with two equations (rows 0 and 1 of the flattened plane)
    a11, a12 = gf[0].item(), xf[0].item()
    a21, a22 = gf[1].item(), xf[1].item()
    b1, b2 = out[0].item() - Cc.item(), out[1].item() - Cc.item()
    det = a11 * a22 - a12 * a21
    A_h = (b1 * a22 - a12 * b2) / det
    B_h = (a11 * b2 - b1 * a21) / det
    print("kernel A,B (C assumed)", A_h, B_h)
    # full 3-unknown solve from three rows
    import numpy as np
    M = np.array([[gf[i].item(), xf[i].item(), 1.0] for i in range(3)], dtype=np.float64)
    rhs = np.array([out[i].item() for i in range(3)], dtype=np.float64)
    A3, B3, C3 = np.linalg.solve(M, rhs)
    print("kernel A,B,C", A3, B3, C3)
    per_ = (w[c] * rstd[c]) / L
    xc_ = per_ * gw_g[c]
    sh_ = per_ * gb_g[c]
    print("coef", (w[c] * rstd[c]).item(), "xc*rstd", (xc_ * rstd[c]).item(),
          "-xc*rstd", (-xc_ * rstd[c]).item(),
          "xc*m*rstd", (xc_ * rstd[c] * mean[c]).item(),
          "shift", sh_.item(), "C_minus", (xc_ * rstd[c] * mean[c] - sh_).item(),
          "C_plus", (xc_ * rstd[c] * mean[c] + sh_).item())
    print("out", out[:3].tolist(), "ref", (gx_g[:, c, :].reshape(-1)[:3]).tolist())
    print("g", gf[:3].tolist(), "x", xf[:3].tolist())


if __name__ == "__main__":
    main()
    solve_coeffs()
