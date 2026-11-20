# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Diagnostic op-cost ablation probe for the expm1 chain (16m fp16).

NOT part of the gate: this file is a same-structure ablation probe used only to
locate the dominant op in the baseline 16-op piecewise chain.  Each ``mode``
produces a kernel with the same persistent grid-stride structure but a
different vector chain, so the per-op marginal cost can be read off the
``msprof op`` deltas.

Modes:
  full          -- baseline 16-op piecewise chain (correct)
  no_select     -- vselect replaced by a plain vadd (1 op; WRONG result)
  short_horner  -- Horner 10 ops collapsed to 2 (WRONG result)
  no_exp        -- vexp replaced by vmul (WRONG result)
  exp_only      -- only the large branch (vexp + vsub), no small branch/select
  copy_only     -- pure copy in/out (floor)

Usage (one mode per process; msprof wraps this):
  python probe_chain.py --mode full --N 16777216 --dtype float16
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import importlib.util
import sys

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

_CORE_NUM = 40
_F32 = "float32"
_T = 0.34657359027997264
_C = [1.0, 1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0]

tilelang.cache.clear_cache()


@tilelang.jit(out_idx=[1], target="npuir")
def expm1_probe(N, dtype, block_size=2048, mode="full"):
    f32 = _F32
    need_cast = dtype != f32
    n_num = T.ceildiv(N, block_size)

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), dtype)
            xf = T.alloc_ub((block_size,), f32)
            xa = T.alloc_ub((block_size,), f32)
            g = T.alloc_ub((block_size,), f32)
            e = T.alloc_ub((block_size,), f32)
            thr = T.alloc_ub((block_size,), f32)
            cond = T.alloc_ub((block_size,), "bool")
            T.fill(thr, T.cast(_T, f32))
            xc = xf if need_cast else x_ub
            for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                task = i * _CORE_NUM + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    T.copy(x[off : off + tail], x_ub[0:tail])
                    if need_cast:
                        T.vcast(x_ub, xf, round_mode="rint")

                    if mode == "copy_only":
                        if need_cast:
                            T.vcast(xc, y_ub, round_mode="rint")
                            T.copy(y_ub[0:tail], y[off : off + tail])
                        else:
                            T.copy(xc[0:tail], y[off : off + tail])
                    else:
                        T.vexp(xc, e)
                        T.vsub(e, T.cast(1.0, f32), e)
                        if mode != "exp_only":
                            T.vabs(xc, xa)
                            T.vcmp(xa, thr, cond, "lt")
                            if mode == "short_horner":
                                T.vmul(xc, _C[0], g)
                                T.vadd(g, _C[5], g)
                            else:
                                T.vmul(xc, _C[5], g)
                                T.vadd(g, T.cast(_C[4], f32), g)
                                T.vmul(g, xc, g)
                                T.vadd(g, T.cast(_C[3], f32), g)
                                T.vmul(g, xc, g)
                                T.vadd(g, T.cast(_C[2], f32), g)
                                T.vmul(g, xc, g)
                                T.vadd(g, T.cast(_C[1], f32), g)
                                T.vmul(g, xc, g)
                                T.vadd(g, T.cast(_C[0], f32), g)
                            T.vmul(xc, g, xa)
                            if mode == "no_exp":
                                T.vmul(e, T.cast(1.0, f32), e)
                            if mode == "no_select":
                                T.vadd(xa, e, xc)
                            else:
                                T.vselect(cond, xa, e, xc)
                        else:
                            T.vadd(e, T.cast(0.0, f32), xc)
                        if need_cast:
                            T.vcast(xc, y_ub, round_mode="rint")
                            T.copy(y_ub[0:tail], y[off : off + tail])
                        else:
                            T.copy(xc[0:tail], y[off : off + tail])

    return main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full")
    ap.add_argument("--N", type=int, default=16777216)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    td = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[args.dtype]
    x = torch.randn(args.N, dtype=torch.float32).to(td).npu()
    kern = expm1_probe(args.N, args.dtype, block_size=args.block_size, mode=args.mode)
    for _ in range(args.iters):
        kern(x, args.N)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
