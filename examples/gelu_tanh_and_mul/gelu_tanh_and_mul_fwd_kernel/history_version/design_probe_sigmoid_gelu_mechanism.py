# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Design-time probe (D-3, probe #2): validate the explicit GELU-tanh mechanism.

Purpose
-------
After probe #1 showed ``T.vtanh`` is a truncated series (not a faithful tanh),
Stage 1 must confirm that an explicit, robust mechanism compiles and is
numerically correct on the NPU before freezing it into DESIGN.md.

Adopted mechanism (DESIGN.md §0.6 / §1.6.1)
-------------------------------------------
Using the identity ``1 + tanh(u) = 2 * sigmoid(2u)``, the tanh-approximate GELU
becomes a single well-conditioned sigmoid:

    gelu_tanh(g) = 0.5 * g * (1 + tanh(u)) = g * sigmoid(2u),
    u = sqrt(2/pi) * (g + 0.044715 * g^3)

with ``sigmoid(v) = 1 / (1 + exp(-v))`` evaluated in fp32 via explicit
``vcast / vmul / vadd / vexp / vdiv`` (the reciprocal dividend is a ``vbrc``-filled
ones buffer because ``vdiv`` only allows the *second* operand to be scalar).

This probe exercises the full per-element chain (gate/value in -> activation*value
out) for fp16 / bf16 / fp32 on a single tile, against the fp64 reference
``0.5*g*(1+tanh(u))*v``, and checks the ±inf / NaN boundary semantics.

Result (2026-10-13, tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B3)
----------------------------------------------------------------------
    dtype    finite max abs err   gelu(+inf)  gelu(-inf)  nan->nan
    fp16     9.4e-4  (< 1e-2)     +inf        nan         True
    bf16     6.8e-3  (< 1.6e-2)    +inf        nan         True
    fp32     1.9e-7  (< 1e-5)      +inf        nan         True

Saturation is exact (g=+50 -> +50, g=-50 -> -0.0) and NaN propagates.
Trap found while probing: ``T.vcast(fp32 -> fp32)`` corrupts the data on this
toolchain — the fp32 input/output path must use ``T.copy`` (same-dtype), exactly
as the sibling gelu_and_mul design does.  With that fix the fp32 chain is 1.9e-7.

Run:  python history_version/design_probe_sigmoid_gelu_mechanism.py
"""

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

SQ = 0.7978845608028654  # sqrt(2/pi)
C = 0.044715

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@tilelang.jit(out_idx=[1], target="npuir")
def _gelu_tanh_mul(B, dtype):
    f32 = dtype == "float32"

    @T.prim_func
    def main(x: T.Tensor((2 * B,), dtype), y: T.Tensor((B,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            gub = T.alloc_ub((B,), dtype)
            vub = T.alloc_ub((B,), dtype)
            g32 = T.alloc_ub((B,), "float32")
            v32 = T.alloc_ub((B,), "float32")
            g2 = T.alloc_ub((B,), "float32")
            g3 = T.alloc_ub((B,), "float32")
            m1 = T.alloc_ub((B,), "float32")
            u = T.alloc_ub((B,), "float32")
            vi = T.alloc_ub((B,), "float32")
            ex = T.alloc_ub((B,), "float32")
            den = T.alloc_ub((B,), "float32")
            one = T.alloc_ub((B,), "float32")
            s = T.alloc_ub((B,), "float32")
            res = T.alloc_ub((B,), dtype)

            T.copy(x[0:B], gub[0:B])
            T.copy(x[B:2 * B], vub[0:B])
            if f32:
                T.copy(gub, g32)
                T.copy(vub, v32)
            else:
                T.vcast(gub, g32, round_mode="rint")
                T.vcast(vub, v32, round_mode="rint")

            T.vmul(g32, g32, g2)          # g^2
            T.vmul(g2, g32, g3)           # g^3
            T.vmul(g3, C, m1)             # c g^3
            T.vadd(g32, m1, u)            # g + c g^3
            T.vmul(u, SQ, u)              # u
            T.vmul(u, -2.0, vi)           # -2u
            T.vexp(vi, ex)                # exp(-2u)
            T.vadd(ex, 1.0, den)          # 1 + exp(-2u)
            T.vbrc(T.cast(1.0, "float32"), one)
            T.vdiv(one, den, s)           # sigmoid(2u)
            T.vmul(g32, s, s)             # gelu_tanh(g)
            T.vmul(s, v32, s)             # * value
            if f32:
                T.copy(s, res)
            else:
                T.vcast(s, res, round_mode="rint")
            T.copy(res[0:B], y[0:B])

    return main


def run(dtname):
    td = DT[dtname]
    B = 512
    torch.manual_seed(0)
    g = torch.randn(B, dtype=torch.float32).to(td)
    v = torch.ones(B, dtype=torch.float32).to(td)
    vals = [float("inf"), float("-inf"), float("nan"), 6.0, -6.0, 50.0, -50.0,
            0.0, -0.0, 1.5, -1.5, 0.001, -0.001]
    for i, val in enumerate(vals):
        g[i] = val
    x = torch.cat([g, v])
    y = _gelu_tanh_mul(B, dtname)(x.npu()).cpu().float()

    gt = g.double()
    inner = SQ * (gt + C * gt ** 3)
    ref = 0.5 * gt * (1 + torch.tanh(inner))
    fin = torch.isfinite(ref)
    d = (y.double() - ref).abs()
    print(f"[{dtname}] finite max_abs_err = {float(d[fin].max()):.3e}")
    print(f"   gelu(+inf)={float(y[0])!r}  gelu(-inf)={float(y[1])!r}  "
          f"nan->nan={bool(torch.isnan(y[2]))}")
    print(f"   g=+50 -> {float(y[5])!r} (ref {float(ref[5])!r})   "
          f"g=-50 -> {float(y[6])!r} (ref {float(ref[6])!r})")


if __name__ == "__main__":
    for dt in ("float16", "bfloat16", "float32"):
        run(dt)
