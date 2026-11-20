#!/usr/bin/env python3
"""R9 micro-probe 4: per-row affine (row-slice dst, no `vbrc`).

If `T.vmul(x[0:1, 0:N], w1, x[0:1, 0:N])` (row slice as v-op *dst*) is legal, the
affine needs no `(bm,N)` broadcast plane at all: w1/b1 are `(1,N)` and the row
slice is `(1,N)`, so the shapes match without `vbrc`.  That both removes 2 full
plane-passes per tile and (hypothesis) frees the backend's broadcast scratch,
which is the last 4 B/elem standing between `block_m=3` and UB capacity.

    TILELANG_ASCEND_MODE=Developer python rowslice_probe.py
"""
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

N = 4096
BM = 3
M = 96
DT = "float16"


@tilelang.jit(out_idx=[3], target="npuir")
def build():
    @T.prim_func
    def main(x: T.Tensor[(M, N), DT], w: T.Tensor[(N,), DT],
             b: T.Tensor[(N,), DT], out: T.Tensor[(M, N), DT]):
        with T.Kernel(1, is_npu=True) as (cid, _):
            xf = T.alloc_shared((BM, N), "float32")
            w1 = T.alloc_shared((1, N), "float32")
            b1 = T.alloc_shared((1, N), "float32")
            T.copy(w[0:N], w1[0:1, 0:N])
            T.copy(b[0:N], b1[0:1, 0:N])
            for i in T.serial(M // BM):
                off = i * BM
                T.copy(x[off:off + BM, 0:N], xf[0:BM, 0:N])
                for r in range(BM):
                    # per-row affine: (1,N) row slice dst, (1,N) factor src
                    T.vmul(xf[r:r + 1, 0:N], w1, xf[r:r + 1, 0:N])
                    T.vadd(xf[r:r + 1, 0:N], b1, xf[r:r + 1, 0:N])
                T.vcast(xf[0:BM, 0:N], out[off:off + BM, 0:N])

    return main


def main():
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    w = torch.randn(N, dtype=torch.float16, device="npu")
    b = torch.randn(N, dtype=torch.float16, device="npu")
    ref = (x.float() * w.float() + b.float()).to(torch.float16)
    try:
        prog = build()
    except Exception as exc:  # noqa: BLE001
        msg = [ln for ln in str(exc).splitlines() if ln.strip()]
        print(f"[rowaffine] COMPILE=FAIL :: {msg[-1] if msg else exc}")
        return 1
    print("[rowaffine] COMPILE=OK")
    try:
        o = prog(x, w, b)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"[rowaffine] RUN=FAIL :: {type(exc).__name__}: {str(exc)[:300]}")
        return 2
    d = (o.float() - ref.float()).abs().max().item()
    print(f"[rowaffine] RUN=OK max_diff={d:.3e} -> {'OK' if d < 3e-3 else 'MISMATCH'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
