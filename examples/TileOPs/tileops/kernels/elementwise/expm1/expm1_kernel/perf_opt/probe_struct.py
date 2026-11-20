# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Structural candidate probe for expm1 (diagnostic; not part of the gate).

Tests branchless-select / reduced-degree / tanh-identity re-formulations of the
piecewise expm1 chain at a given ``block_size``.  Each mode is a complete
persistent grid-stride kernel sharing the baseline structure; only the vector
chain differs, so ``msprof op`` deltas are comparable.

Modes:
  base        -- baseline: T=0.3466, Taylor deg 5, vcmp+vselect
  deg3_sel    -- T=0.11,    Taylor deg 3, vcmp+vselect
  deg5_blend  -- T=0.3466, Taylor deg 5, arithmetic clamp-mask blend (no vselect)
  deg3_blend  -- T=0.11,    Taylor deg 3, arithmetic clamp-mask blend (no vselect)
  tanh_id     -- expm1(x) = tanh(x/2)*(exp(x)+1)  (single formula, no branch)

Usage:
  python probe_struct.py --mode deg3_blend --N 16777216 --dtype float16 --block_size 6144
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import math
import sys

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

_CORE_NUM = 40
_F32 = "float32"
_BIG = 1.0e4

# Taylor coefficients C_n = 1/(n+1)! (ascending), threshold per mode.
_MODES = {
    "base":       {"T": 0.5 * math.log(2.0), "deg": 5, "sel": "vselect"},
    "deg3_sel":   {"T": 0.11, "deg": 3, "sel": "vselect"},
    "deg2mm_sel": {"T": 0.05, "deg": 2, "sel": "vselect",
                   "coef": [0.9999999955, 0.5000625040, 0.1666845248]},
    "deg2_sel":   {"T": 0.042, "deg": 2, "sel": "vselect"},
    "deg2mm_nosel": {"T": 0.05, "deg": 2, "sel": "nosel",
                   "coef": [0.9999999955, 0.5000625040, 0.1666845248]},
    "deg3_nosel": {"T": 0.11, "deg": 3, "sel": "nosel"},
    "deg5_blend": {"T": 0.5 * math.log(2.0), "deg": 5, "sel": "blend"},
    "deg3_blend": {"T": 0.11, "deg": 3, "sel": "blend"},
    "deg3_blendc": {"T": 0.11, "deg": 3, "sel": "blendc"},
    "deg2_blendc": {"T": 0.042, "deg": 2, "sel": "blendc"},
    "tanh_id":    {"T": 0.0, "deg": 0, "sel": "tanh"},
}
_DEFAULT_C = [1.0 / math.factorial(n + 1) for n in range(6)]
_GCLAMP = 0.5  # clamp the (otherwise divergent) Taylor branch before blending

tilelang.cache.clear_cache()


@tilelang.jit(out_idx=[1], target="npuir")
def expm1_struct(N, dtype, block_size=2048, mode="base", cores=_CORE_NUM, layout="stride"):
    cfg = _MODES[mode]
    thr_val = cfg["T"]
    deg = cfg["deg"]
    sel = cfg["sel"]
    coef = cfg.get("coef", _DEFAULT_C)
    xmax = 50.0  # tanh argument clamp (|x/2|<=50 < fp32 exp domain)
    f32 = _F32
    need_cast = dtype != f32
    n_num = T.ceildiv(N, block_size)

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
        with T.Kernel(cores, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), dtype)
            xf = T.alloc_ub((block_size,), f32)
            xa = T.alloc_ub((block_size,), f32)
            g = T.alloc_ub((block_size,), f32)
            e = T.alloc_ub((block_size,), f32)
            thr = T.alloc_ub((block_size,), f32)
            cond = T.alloc_ub((block_size,), "bool")
            T.fill(thr, T.cast(thr_val, f32))
            xc = xf if need_cast else x_ub
            _iters = T.ceildiv(n_num, cores)
            # task = i*A + cid*B  (stride: A=cores,B=1 ; contig: A=1,B=iters)
            _A = 1 if layout == "contig" else cores
            _B = _iters if layout == "contig" else 1
            for i in T.serial(_iters):
                task = i * _A + cid * _B
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    T.copy(x[off : off + tail], x_ub[0:tail])
                    if need_cast:
                        T.vcast(x_ub, xf, round_mode="rint")

                    if sel == "tanh":
                        # expm1(x) = tanh(x/2) * (exp(x) + 1)
                        T.vmul(xc, T.cast(0.5, f32), g)      # g = x/2
                        T.vmin(g, T.cast(xmax, f32), g)      # clamp +arg (vmin NaN-safe)
                        T.vmax(g, T.cast(-xmax, f32), g)     # clamp -arg
                        T.vtanh(g, g)                        # g = tanh(x/2)
                        T.vexp(xc, e)                        # e = exp(x)
                        T.vadd(e, T.cast(1.0, f32), e)       # e = exp(x)+1
                        T.vmul(g, e, xc)                     # y = tanh(x/2)*(exp(x)+1)
                    else:
                        # mask = (|x| > T)
                        T.vabs(xc, xa)
                        if sel in ("blend", "blendc"):
                            T.vsub(xa, T.cast(thr_val, f32), xa)
                            T.vmul(xa, T.cast(_BIG, f32), xa)
                            T.vclamp(xa, xa, T.cast(0.0, f32), T.cast(1.0, f32))
                        else:
                            T.vcmp(xa, thr, cond, "lt")      # cond = |x|<T

                        # Horner Taylor of expm1(x)/x, degree `deg` (hand-unrolled;
                        # compile-time guards select the terms).
                        T.vmul(xc, coef[deg], g)               # g = C_deg * x
                        if deg >= 5:
                            T.vadd(g, T.cast(coef[4], f32), g)
                            T.vmul(g, xc, g)
                        if deg >= 4:
                            T.vadd(g, T.cast(coef[3], f32), g)
                            T.vmul(g, xc, g)
                        if deg >= 3:
                            T.vadd(g, T.cast(coef[2], f32), g)
                            T.vmul(g, xc, g)
                        if deg >= 2:
                            T.vadd(g, T.cast(coef[1], f32), g)
                            T.vmul(g, xc, g)
                        T.vadd(g, T.cast(coef[0], f32), g)
                        T.vmul(xc, g, g)                     # g = small branch

                        T.vexp(xc, e)
                        T.vsub(e, T.cast(1.0, f32), e)       # e = large branch

                        if sel == "blendc":
                            # Clamp the divergent Taylor branch so that the
                            # blend stays finite and cancellation-free for
                            # large |x| and for +-inf (algebraic identity:
                            # (g + (e-g)) == e for any finite g).
                            T.vclamp(g, g, T.cast(-_GCLAMP, f32), T.cast(_GCLAMP, f32))
                        if sel in ("blend", "blendc"):
                            T.vsub(e, g, e)                  # e = large-small
                            T.vmul(xa, e, e)                 # e = mask*(large-small)
                            T.vadd(g, e, xc)                 # y = small+mask*(large-small)
                        elif sel == "nosel":
                            T.vadd(g, e, xc)                 # diagnostic: no select
                        else:
                            T.vselect(cond, g, e, xc)        # y = |x|<T ? small : large

                    if need_cast:
                        T.vcast(xc, y_ub, round_mode="rint")
                        T.copy(y_ub[0:tail], y[off : off + tail])
                    else:
                        T.copy(xc[0:tail], y[off : off + tail])

    return main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="base")
    ap.add_argument("--N", type=int, default=16777216)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--cores", type=int, default=_CORE_NUM)
    ap.add_argument("--layout", default="stride")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    td = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[args.dtype]
    x = torch.randn(args.N, dtype=torch.float32).to(td).npu()
    kern = expm1_struct(args.N, args.dtype, block_size=args.block_size, mode=args.mode, cores=args.cores, layout=args.layout)
    for _ in range(args.iters):
        kern(x, args.N)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
