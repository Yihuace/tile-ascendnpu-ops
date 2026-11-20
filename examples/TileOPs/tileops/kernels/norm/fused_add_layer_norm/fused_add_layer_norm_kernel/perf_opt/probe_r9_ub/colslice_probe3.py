#!/usr/bin/env python3
"""R9 micro-probe 3: the two primitives the sq-half layout needs.

  F1  in-place *column-half* destination:
        vmul(x[0:BM, 0:H], y_h, x[0:BM, 0:H])   (dst is a strided half)
  F2  reduce of a full (BM, H) plane into (BM, 1) + scalar combine
  F3  vadd in-place on the *upper* half (offset slice, dst)

    TILELANG_ASCEND_MODE=Developer python colslice_probe3.py
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


@tilelang.jit(out_idx=[2], target="npuir")
def build_inplace():
    @T.prim_func
    def main(x: T.Tensor[(M, N), DT], w: T.Tensor[(N,), DT],
             out: T.Tensor[(M, N), DT]):
        with T.Kernel(1, is_npu=True) as (cid, _):
            xf = T.alloc_shared((BM, N), "float32")
            sh = T.alloc_shared((BM, H), "float32")
            wa = T.alloc_shared((1, H), "float32")
            wb = T.alloc_shared((1, H), "float32")
            T.copy(w[0:H], wa[0:1, 0:H])
            T.copy(w[H:N], wb[0:1, 0:H])
            for i in T.serial(M // BM):
                off = i * BM
                T.copy(x[off:off + BM, 0:N], xf[0:BM, 0:N])
                # lower half: in-place vmul with a broadcast plane
                T.vbrc(wa[0:1, 0:H], sh)
                T.vmul(xf[0:BM, 0:H], sh, xf[0:BM, 0:H])
                # upper half: in-place vadd with a broadcast plane (offset src)
                T.vbrc(wb[0:1, 0:H], sh)
                T.vadd(xf[0:BM, H:N], sh, xf[0:BM, H:N])
                T.vcast(xf[0:BM, 0:N], out[off:off + BM, 0:N])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_halfreduce():
    @T.prim_func
    def main(x: T.Tensor[(M, N), DT], out: T.Tensor[(M, 1), "float32"]):
        with T.Kernel(1, is_npu=True) as (cid, _):
            xf = T.alloc_shared((BM, N), "float32")
            sh = T.alloc_shared((BM, H), "float32")
            sa = T.alloc_shared((BM, 1), "float32")
            sb = T.alloc_shared((BM, 1), "float32")
            for i in T.serial(M // BM):
                off = i * BM
                T.copy(x[off:off + BM, 0:N], xf[0:BM, 0:N])
                T.vmul(xf[0:BM, 0:H], xf[0:BM, 0:H], sh)
                T.reduce_sum(sh, sa, dim=1)
                T.vmul(xf[0:BM, H:N], xf[0:BM, H:N], sh)
                T.reduce_sum(sh, sb, dim=1)
                T.vadd(sa, sb, sa)
                T.copy(sa[0:BM, 0:1], out[off:off + BM, 0:1])

    return main


def run(name, fn, *args):
    try:
        prog = fn()
    except Exception as exc:  # noqa: BLE001
        msg = [ln for ln in str(exc).splitlines() if ln.strip()]
        print(f"[{name}] COMPILE=FAIL :: {msg[-1] if msg else exc}")
        return None
    print(f"[{name}] COMPILE=OK")
    try:
        res = prog(*args)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"[{name}] RUN=FAIL :: {type(exc).__name__}: {str(exc)[:300]}")
        return None
    print(f"[{name}] RUN=OK")
    return res


def main():
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    w = torch.randn(N, dtype=torch.float16, device="npu")
    xf = x.float()
    ref = torch.cat([xf[:, :H] * w.float()[:H], xf[:, H:] + w.float()[H:]], dim=1)

    o = run("inplace", build_inplace, x, w)
    if o is not None:
        d = (o.float() - ref).abs().max().item()
        print(f"[inplace] max_diff={d:.3e} -> {'OK' if d < 3e-3 else 'MISMATCH'}")

    o2 = run("halfreduce", build_halfreduce, x)
    if o2 is not None:
        ref2 = (xf * xf).sum(dim=1, keepdim=True)
        d = (o2 - ref2).abs().max().item()
        print(f"[halfreduce] max_diff={d:.3e} -> {'OK' if d < 0.1 else 'MISMATCH'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
