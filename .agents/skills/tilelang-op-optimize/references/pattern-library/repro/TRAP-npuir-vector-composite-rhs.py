# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""TRAP-npuir-vector-composite-rhs: a multi-operation RHS in a T.Parallel body
silently produces zeros on this toolchain (npuir, Expert mode).

Observed (2026-10-31, Ascend910_9362, tilelang 0.1.2+9738284):

  * ``for i in T.Parallel(h): dst[0,0,i] = (a[i]-b[i])*wr[i] - (c[i]-d[i])*wi[i]``
    (seven operations in one RHS) compiles, verdict "AscendNPU IR compile
    success", and writes **0.0** into ``dst`` for every element.
  * The same arithmetic written as eight one-operation statements, each with
    its own user-declared 3-D UB temporary, is correct to fp32.

Root cause (inferred): the RHS is split by ``DecomposeExpression`` inside
``NpuLoopVectorize`` (src/transform/npu_loop_vectorize.cc, the ``tmp_bufs``
path reached from ``VectorizeSingleStatement``); the generated temporaries are
then placed by the UB planner on top of live buffers, so the multiply chain
reads zeros.  Keeping every RHS to ONE operation and every temporary a
buffer the kernel declares itself avoids the decomposed path entirely.

This is the reason fft_c2c_vec.py spells the complex multiply out as
``tr/ti -> pv/qv -> dif`` instead of one expression (DESIGN.md section 3.3
"RHS fallback").

run: python TRAP-npuir-vector-composite-rhs.py

First verified: 2026-10-31, tilelang 0.1.2+9738284 / CANN 9.0.0 /
Ascend910_9362 (Expert mode).  origin_task (provenance, may go stale):
fft_c2c_vec-fft_c2c_vec-20261031T103133Z; file lives in
traps-compiler.md.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import math

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

DT = "float32"


@tilelang.jit(out_idx=[4], target="npuir")
def k_composite(n):
    """Complex multiply written as one composite RHS -> zeros."""
    nh = n // 2

    @T.prim_func
    def main(
        xr: T.Tensor((1, n), DT),
        xi: T.Tensor((1, n), DT),
        wr: T.Tensor((nh,), DT),
        wi: T.Tensor((nh,), DT),
        y: T.Tensor((2, nh), DT),
    ) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            sr = T.alloc_ub((2, 1, nh), DT)
            si = T.alloc_ub((2, 1, nh), DT)
            wr_ub = T.alloc_ub((nh,), DT)
            wi_ub = T.alloc_ub((nh,), DT)
            dr = T.alloc_ub((2, 1, nh), DT)
            di = T.alloc_ub((2, 1, nh), DT)
            T.copy(xr[cid, 0:nh], sr[0, 0, 0:nh])
            T.copy(xr[cid, nh:n], sr[1, 0, 0:nh])
            T.copy(xi[cid, 0:nh], si[0, 0, 0:nh])
            T.copy(xi[cid, nh:n], si[1, 0, 0:nh])
            T.copy(wr[0:nh], wr_ub[0:nh])
            T.copy(wi[0:nh], wi_ub[0:nh])
            T.pipe_barrier("PIPE_ALL")
            for i in T.Parallel(nh):
                dr[1, 0, i] = (sr[0, 0, i] - sr[1, 0, i]) * wr_ub[i] - (
                    si[0, 0, i] - si[1, 0, i]) * wi_ub[i]
            for i in T.Parallel(nh):
                di[1, 0, i] = (sr[0, 0, i] - sr[1, 0, i]) * wi_ub[i] + (
                    si[0, 0, i] - si[1, 0, i]) * wr_ub[i]
            T.pipe_barrier("PIPE_V")
            T.copy(dr[1, 0, 0:nh], y[0, 0:nh])
            T.copy(di[1, 0, 0:nh], y[1, 0:nh])

    return main


@tilelang.jit(out_idx=[4], target="npuir")
def k_single_op(n):
    """Same arithmetic, one operation per statement -> correct."""
    nh = n // 2

    @T.prim_func
    def main(
        xr: T.Tensor((1, n), DT),
        xi: T.Tensor((1, n), DT),
        wr: T.Tensor((nh,), DT),
        wi: T.Tensor((nh,), DT),
        y: T.Tensor((2, nh), DT),
    ) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            sr = T.alloc_ub((2, 1, nh), DT)
            si = T.alloc_ub((2, 1, nh), DT)
            wr_ub = T.alloc_ub((nh,), DT)
            wi_ub = T.alloc_ub((nh,), DT)
            dr = T.alloc_ub((2, 1, nh), DT)
            di = T.alloc_ub((2, 1, nh), DT)
            tr = T.alloc_ub((2, 1, nh), DT)
            ti = T.alloc_ub((2, 1, nh), DT)
            pv = T.alloc_ub((2, 1, nh), DT)
            qv = T.alloc_ub((2, 1, nh), DT)
            T.copy(xr[cid, 0:nh], sr[0, 0, 0:nh])
            T.copy(xr[cid, nh:n], sr[1, 0, 0:nh])
            T.copy(xi[cid, 0:nh], si[0, 0, 0:nh])
            T.copy(xi[cid, nh:n], si[1, 0, 0:nh])
            T.copy(wr[0:nh], wr_ub[0:nh])
            T.copy(wi[0:nh], wi_ub[0:nh])
            T.pipe_barrier("PIPE_ALL")
            for i in T.Parallel(nh):
                tr[0, 0, i] = sr[0, 0, i] - sr[1, 0, i]
            for i in T.Parallel(nh):
                ti[0, 0, i] = si[0, 0, i] - si[1, 0, i]
            for i in T.Parallel(nh):
                pv[0, 0, i] = tr[0, 0, i] * wr_ub[i]
            for i in T.Parallel(nh):
                qv[0, 0, i] = ti[0, 0, i] * wi_ub[i]
            for i in T.Parallel(nh):
                dr[1, 0, i] = pv[0, 0, i] - qv[0, 0, i]
            for i in T.Parallel(nh):
                pv[0, 0, i] = tr[0, 0, i] * wi_ub[i]
            for i in T.Parallel(nh):
                qv[0, 0, i] = ti[0, 0, i] * wr_ub[i]
            for i in T.Parallel(nh):
                di[1, 0, i] = pv[0, 0, i] + qv[0, 0, i]
            T.pipe_barrier("PIPE_V")
            T.copy(dr[1, 0, 0:nh], y[0, 0:nh])
            T.copy(di[1, 0, 0:nh], y[1, 0:nh])

    return main


def main():
    n = 8
    nh = n // 2
    torch.manual_seed(0)
    xr = torch.randn(1, n)
    xi = torch.randn(1, n)
    ang = -2.0 * math.pi * torch.arange(nh, dtype=torch.float64) / n
    wr = torch.cos(ang).to(torch.float32)
    wi = torch.sin(ang).to(torch.float32)

    s0 = torch.complex(xr[0, :nh], xi[0, :nh])
    s1 = torch.complex(xr[0, nh:], xi[0, nh:])
    ref = (s0 - s1) * torch.complex(wr, wi)

    out = k_single_op(n)(xr.npu(), xi.npu(), wr.npu(), wi.npu())
    torch.npu.synchronize()
    good = torch.complex(out[0].cpu(), out[1].cpu())
    err_good = (good - ref).abs().max().item()
    print(f"[single-op RHS ] max_abs_err = {err_good:.3e}")
    assert err_good < 1e-5, f"single-operation chain must be exact, got {err_good}"

    out = k_composite(n)(xr.npu(), xi.npu(), wr.npu(), wi.npu())
    torch.npu.synchronize()
    bad = torch.complex(out[0].cpu(), out[1].cpu())
    err_bad = (bad - ref).abs().max().item()
    print(f"[composite RHS ] max_abs_err = {err_bad:.3e}  (got={bad.tolist()})")
    if err_bad > 1e-5:
        print("TRAP CONFIRMED: the composite RHS produced wrong values "
              "(zeros on the observed toolchain) - use one operation per "
              "T.Parallel body with explicit UB temporaries.")
    else:
        print("TRAP NO LONGER REPRODUCES on this toolchain (fixed); the "
              "single-operation form is still the safe one.")
    print("ok")


if __name__ == "__main__":
    main()
