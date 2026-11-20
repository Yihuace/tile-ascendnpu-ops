# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""H1 probes: kernel launch floor + per-core / aggregate MTE2 GM-read bandwidth.

Data-only measurement helpers for the Stage 4 H1 (grid = batch -> core
under-utilization) reverse-solve.  These are *not* candidate kernels; they only
bound what a rework of ``_engram_decode_kernel`` could reach.

Modes (``--mode``):
  floor   : one 512 B GM->UB->GM copy per block (launch / teardown floor)
  read    : ``--nchunk`` x 32 KB contiguous GM->UB copies (dtype preserved)
  convert : same but GM fp16 -> UB fp32 (dtype-mixing copy, as the real kernel)
  copy    : ``--nchunk`` x 32 KB GM->UB + UB->GM (streaming read+write)
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()



@tilelang.jit(out_idx=[1], target="npuir")
def _probe(nblk, nchunk, mode, gm_dtype, CH):
    ub_dtype = "float32" if mode in ("convert", "floor") else gm_dtype
    total = max(nblk * nchunk, 1) * CH

    @T.prim_func
    def main(
        a: T.Tensor((total,), gm_dtype),
        b: T.Tensor((nblk * CH,), gm_dtype),
    ):
        with T.Kernel(nblk, is_npu=True) as (bid, _):
            s = T.alloc_shared((CH,), ub_dtype)
            if mode == "floor":
                T.copy(a[bid * CH:(bid + 1) * CH], s)
                T.copy(s, b[bid * CH:(bid + 1) * CH])
            elif mode == "copy":
                for c in T.serial(nchunk):
                    off = (bid * nchunk + c) * CH
                    T.copy(a[off:off + CH], s)
                    T.copy(s, b[off:off + CH])
            else:
                for c in T.serial(nchunk):
                    off = (bid * nchunk + c) * CH
                    T.copy(a[off:off + CH], s)
                T.copy(s, b[bid * CH:(bid + 1) * CH])

    return main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["floor", "read", "convert", "copy"])
    ap.add_argument("--nblk", type=int, default=1)
    ap.add_argument("--nchunk", type=int, default=1)
    ap.add_argument("--chunk-elem", type=int, default=8192)
    ap.add_argument("--gm-dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    nchunk = 1 if args.mode == "floor" else args.nchunk
    gm_dtype = args.gm_dtype
    if args.mode == "floor":
        gm_dtype = "float32"
    if args.mode == "convert":
        gm_dtype = "float16"
    ub_dtype = "float32" if args.mode in ("convert", "floor") else gm_dtype
    elem = 4 if gm_dtype == "float32" else 2

    kernel = _probe(args.nblk, nchunk, args.mode, gm_dtype, args.chunk_elem)
    CH = args.chunk_elem
    total = args.nblk * nchunk * CH
    b_len = args.nblk * CH if args.mode != "copy" else total
    a = torch.randn(total, dtype=getattr(torch, gm_dtype)).npu()
    for _ in range(args.iters):
        kernel(a)
    torch.npu.synchronize()
    read_b = total * elem
    write_b = b_len * elem
    print(f"[done] mode={args.mode} nblk={args.nblk} nchunk={nchunk} "
          f"gm={gm_dtype} ub={ub_dtype} chunk_elem={CH} read_bytes={read_b} write_bytes={write_b}")


if __name__ == "__main__":
    main()
