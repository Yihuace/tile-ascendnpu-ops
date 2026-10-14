# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro: TRAP-copy-ub-offset-stride  (pattern-library/traps-compiler.md)

Phenomenon: In Expert mode, a 2-D ``T.copy`` GM->UB whose UB destination is a
  *column-offset slice* (``ub[0:R, PAD:PAD+N]`` with ``ub`` allocated wider than
  ``N``) silently copies only the first row correctly -- rows ``1..R-1`` land at
  the wrong stride (the generic copy flattens the region instead of honouring
  the UB row stride).  The result is silently wrong (no compile error), so it
  demonstrates as a precision failure, not a crash.

Fix: route the column shift through the *GM* side instead -- copy the source
  contiguously into a same-shape UB buffer and write it to a GM destination
  slice ``gm[0:R, PAD:PAD+N]`` (a GM offset destination is honoured correctly).
  This is exactly the ``x_pad`` construction used by ``_conv1d_kernel``.

Assertion: the GM-offset workaround matches a CPU reference; the UB-offset
  variant is executed *separately* by ``--show-bug`` only (kept out of the
  default path because it is the broken form).

First verified: 2026-10-14 conv1d _conv1d_kernel Stage 3.
Toolchain: tilelang 0.1.2+417dae99 + Ascend910B3 (get_aicore_num=20).
Run: python repro/TRAP-copy-ub-offset-stride.py [--show-bug]
"""
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

DT = "float16"
M, N, PAD = 16, 64, 16
LP = N + PAD


@tilelang.jit(out_idx=[1], target="npuir")
def build_gm_offset():
    @T.prim_func
    def main(a: T.Tensor((M, N), DT), b: T.Tensor((M, LP), DT)):
        with T.Kernel(1, is_npu=True) as (kid, _):
            with T.Scope("Vector"):
                ub = T.alloc_ub([M, N], DT)
                T.copy(a[0:M, 0:N], ub)
                T.copy(ub, b[0:M, PAD:PAD + N])  # GM offset destination: correct
    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_ub_offset():
    @T.prim_func
    def main(a: T.Tensor((M, N), DT), b: T.Tensor((M, LP), DT)):
        with T.Kernel(1, is_npu=True) as (kid, _):
            with T.Scope("Vector"):
                ub = T.alloc_ub([M, LP], DT)
                T.clear(ub)
                T.copy(a[0:M, 0:N], ub[0:M, PAD:PAD + N])  # BROKEN: UB offset dest
                T.copy(ub, b)
    return main


def _ref():
    torch.manual_seed(0)
    a = torch.randn(M, N, dtype=torch.float16)
    ref = torch.zeros(M, LP, dtype=torch.float16)
    ref[:, PAD:PAD + N] = a
    return a, ref


def run():
    a, ref = _ref()
    out = build_gm_offset()(a.to("npu"))
    torch.npu.synchronize()
    err = float((out.cpu().float() - ref.float()).abs().max())
    print(f"GM-offset workaround max_abs_err = {err:.4e}")
    assert err < 1e-2, f"GM-offset workaround wrong: {err}"
    if "--show-bug" in sys.argv:
        bad = build_ub_offset()(a.to("npu"))
        torch.npu.synchronize()
        berr = float((bad.cpu().float() - ref.float()).abs().max())
        badrows = int(((bad.cpu().float() - ref.float()).abs() > 1e-3).any(1).sum())
        print(f"[broken UB-offset variant] max_abs_err = {berr:.4e}, bad rows = {badrows}")
    print("[TRAP-copy-ub-offset-stride] PASS: GM-offset column shift correct")


if __name__ == "__main__":
    run()
