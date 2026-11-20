#!/usr/bin/env python3
"""R9 micro-probe 2: bisect which vector ops accept a *column-half* view
`buf[0:BM, 0:N//2]` of a (BM, N) fp32 UB plane.

Each feature compiles (and, if it compiles, runs + checks numerics) in its own
try/except so one failure does not hide the others.

    TILELANG_ASCEND_MODE=Developer python colslice_probe2.py
"""
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

N = 4096
H = N // 2
BM = 3
M = 96
DT = "float16"


@tilelang.jit(out_idx=[1], target="npuir")
def build_mul():
    @T.prim_func
    def main(x: T.Tensor[(M, N), DT], out: T.Tensor[(M, N), DT]):
        with T.Kernel(1, is_npu=True) as (cid, _):
            xf = T.alloc_shared((BM, N), "float32")
            half = T.alloc_shared((BM, H), "float32")
            for i in T.serial(M // BM):
                off = i * BM
                T.copy(x[off:off + BM, 0:N], xf[0:BM, 0:N])
                T.vmul(xf[0:BM, 0:H], xf[0:BM, 0:H], half)
                T.copy(half[0:BM, 0:H], out[off:off + BM, 0:H])
                T.vmul(xf[0:BM, H:N], xf[0:BM, H:N], half)
                T.copy(half[0:BM, 0:H], out[off:off + BM, H:N])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_reduce():
    @T.prim_func
    def main(x: T.Tensor[(M, N), DT], out: T.Tensor[(M, 1), "float32"]):
        with T.Kernel(1, is_npu=True) as (cid, _):
            xf = T.alloc_shared((BM, N), "float32")
            s = T.alloc_shared((BM, 1), "float32")
            t = T.alloc_shared((BM, 1), "float32")
            for i in T.serial(M // BM):
                off = i * BM
                T.copy(x[off:off + BM, 0:N], xf[0:BM, 0:N])
                T.reduce_sum(xf[0:BM, 0:H], s[0:BM, 0:1], dim=1)
                T.reduce_sum(xf[0:BM, H:N], t[0:BM, 0:1], dim=1)
                T.vadd(s, t, s)
                T.copy(s[0:BM, 0:1], out[off:off + BM, 0:1])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_vcast():
    @T.prim_func
    def main(x: T.Tensor[(M, N), DT], out: T.Tensor[(M, H), DT]):
        with T.Kernel(1, is_npu=True) as (cid, _):
            xf = T.alloc_shared((BM, N), "float32")
            half = T.alloc_shared((BM, H), DT)
            for i in T.serial(M // BM):
                off = i * BM
                T.copy(x[off:off + BM, 0:N], xf[0:BM, 0:N])
                T.vcast(xf[0:BM, 0:H], half, round_mode="rint")
                T.copy(half[0:BM, 0:H], out[off:off + BM, 0:H])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_vbrc():
    @T.prim_func
    def main(w: T.Tensor[(N,), DT], out: T.Tensor[(M, H), DT]):
        with T.Kernel(1, is_npu=True) as (cid, _):
            pl = T.alloc_shared((BM, H), "float32")
            wh = T.alloc_shared((1, H), "float32")
            T.copy(w[0:H], wh[0:1, 0:H])
            for i in T.serial(M // BM):
                off = i * BM
                T.vbrc(wh[0:1, 0:H], pl)
                T.vcast(pl, pl, round_mode="rint")
                T.copy(pl[0:BM, 0:H], out[off:off + BM, 0:H])

    return main


def run(name, fn, *args):
    try:
        prog = fn()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).strip().splitlines()
        print(f"[{name}] COMPILE=FAIL :: {msg[-1] if msg else exc}")
        return False
    print(f"[{name}] COMPILE=OK")
    try:
        prog(*args)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"[{name}] RUN=FAIL :: {type(exc).__name__}: {str(exc)[:300]}")
        return False
    print(f"[{name}] RUN=OK")
    return True


def main():
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    w = torch.randn(N, dtype=torch.float16, device="npu")
    xf = x.float()

    o = torch.zeros(M, N, dtype=torch.float16, device="npu")
    if run("mul", build_mul, x, o):
        d = (o.float() - xf * xf).abs().max().item()
        print(f"[mul] max_diff={d:.3e} -> {'OK' if d < 2e-3 else 'MISMATCH'}")

    o2 = torch.zeros(M, 1, dtype=torch.float32, device="npu")
    if run("reduce", build_reduce, x, o2):
        ref = xf.sum(dim=1, keepdim=True)
        d = (o2 - ref).abs().max().item()
        print(f"[reduce] max_diff={d:.3e} -> {'OK' if d < 0.05 else 'MISMATCH'}")

    o3 = torch.zeros(M, H, dtype=torch.float16, device="npu")
    if run("vcast", build_vcast, x, o3):
        d = (o3.float() - xf[:, :H]).abs().max().item()
        print(f"[vcast] max_diff={d:.3e} -> {'OK' if d < 2e-3 else 'MISMATCH'}")

    o4 = torch.zeros(M, H, dtype=torch.float16, device="npu")
    if run("vbrc", build_vbrc, w, o4):
        ref = w.float()[:H].unsqueeze(0).expand(M, H)
        d = (o4.float() - ref).abs().max().item()
        print(f"[vbrc] max_diff={d:.3e} -> {'OK' if d < 2e-3 else 'MISMATCH'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
