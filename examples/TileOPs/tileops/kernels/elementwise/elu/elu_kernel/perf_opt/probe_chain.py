# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Diagnostic chain-cost probe for the ELU fp32-transit path (NOT gate evidence).

Builds the same kernel skeleton with different fp32-domain vector chains so the
marginal cost of each op group can be read off a single fixed block_size.  Used
to decide between the mask chain (vcmp+vselect) and arithmetic chains
(vmin/vmax/vadd or vrelu) once the kernel has left the DMA-bound regime.

Chains (all operate on the fp32 transit buffer `xc`, output back into `xc`):
  mask      : vcmp, vexp, vsub, vselect                     (Stage 3 baseline)
  noselect  : vcmp, vexp, vsub, vadd                        (vselect -> cheap op)
  arith     : vmin, vexp, vsub, vmax, vadd                  (mask-free identity)
  relu      : vrelu, vsub, vexp, vsub, vadd                 (mask-free, no z buf)
  exp3      : vexp, vsub, vadd                              (drop min/max group)
  copyonly  : cast-in + cast-out only (no chain)            (DMA floor)

run: python probe_chain.py --impl <file> --chain mask --bs 10240 --N 16777216 --dtype float16
"""

import argparse
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

F32 = "float32"


def build(N, dtype, chain, block_size, core_num=40):
    need_cast = dtype != F32

    @tilelang.jit(out_idx=[1], target="npuir")
    def factory(N_, dtype_, chain_, block_size_):
        n_num = T.ceildiv(N_, block_size_)

        @T.prim_func
        def main(
            x: T.Tensor((N_,), dtype_),
            y: T.Tensor((N_,), dtype_),
            shape: T.int32,
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size_,), dtype_)
                y_ub = T.alloc_ub((block_size_,), dtype_)
                xf = T.alloc_ub((block_size_,), F32)
                e = T.alloc_ub((block_size_,), F32)
                z = T.alloc_ub((block_size_,), F32)
                cond = T.alloc_ub((block_size_,), "bool")

                T.fill(z, T.cast(0.0, F32))
                xc = xf if need_cast else x_ub

                for i in T.serial(T.ceildiv(n_num, core_num)):
                    task = i * core_num + cid
                    if task < n_num:
                        off = task * block_size_
                        tail = T.min(block_size_, shape - off)
                        T.copy(x[off : off + tail], x_ub[0:tail])
                        if need_cast:
                            T.vcast(x_ub, xf, round_mode="rint")

                        if chain_ == "mask":
                            T.vcmp(xc, z, cond, "gt")
                            T.vexp(xc, e)
                            T.vsub(e, T.cast(1.0, F32), e)
                            T.vselect(cond, xc, e, xc)
                        elif chain_ == "noselect":
                            T.vcmp(xc, z, cond, "gt")
                            T.vexp(xc, e)
                            T.vsub(e, T.cast(1.0, F32), e)
                            T.vadd(e, xc, xc)
                        elif chain_ == "arith":
                            T.vmin(xc, z, e)
                            T.vexp(e, e)
                            T.vsub(e, T.cast(1.0, F32), e)
                            T.vmax(xc, z, xc)
                            T.vadd(xc, e, xc)
                        elif chain_ == "relu":
                            T.vrelu(xc, e)
                            T.vsub(xc, e, xc)
                            T.vexp(xc, xc)
                            T.vsub(xc, T.cast(1.0, F32), xc)
                            T.vadd(xc, e, xc)
                        elif chain_ == "exp3":
                            T.vexp(xc, e)
                            T.vsub(e, T.cast(1.0, F32), e)
                            T.vadd(e, xc, xc)
                        elif chain_ == "copyonly":
                            T.vadd(xc, z, xc)
                        else:
                            raise ValueError(chain_)

                        if need_cast:
                            T.vcast(xc, y_ub, round_mode="rint")
                            T.copy(y_ub[0:tail], y[off : off + tail])
                        else:
                            T.copy(xc[0:tail], y[off : off + tail])

        return main

    return factory(N, dtype, chain, block_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default=None, help="ignored (probe_sweep compatibility)")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--chain", required=True)
    ap.add_argument("--bs", "--block_size", dest="bs", type=int, required=True)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    td = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[args.dtype]
    x = torch.randn(args.N, dtype=torch.float32).to(td).npu()
    k = build(args.N, args.dtype, args.chain, args.bs)
    for _ in range(args.iters):
        k(x, args.N)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
