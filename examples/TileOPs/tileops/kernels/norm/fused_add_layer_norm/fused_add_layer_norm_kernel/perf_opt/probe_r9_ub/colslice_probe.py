#!/usr/bin/env python3
"""R9 micro-probe: are *column-half slices* of a (block_m, N) fp32 UB plane
legal operands for vector ops (vmul / vcast / vbrc / reduce_sum) and for the
GM copy path?

Motivation: the R9 candidate is to halve the d^2 scratch plane (`sq_f32`) along
N so that `block_m=3` fits in UB for N=4096.  That is only expressible if the
full-width plane can be addressed half by half, i.e.

    x_f32[0:bm, 0:N//2]          (row-major, rows strided by N)
    x_f32[0:bm, N//2:N]

is a valid v-op operand.  This probe answers that before any kernel rewrite.

Run (NPU, inside npu_lock):
    TILELANG_ASCEND_MODE=Developer python colslice_probe.py
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


@tilelang.jit(out_idx=[6], target="npuir")
def build(dtype="float32"):
    @T.prim_func
    def main(
        x: T.Tensor[(M, N), dtype],
        y_mul: T.Tensor[(M, N), dtype],     # x*x, computed in column halves
        y_sum: T.Tensor[(M, 2), "float32"],  # per-half row sums
        y_brc: T.Tensor[(M, N), dtype],     # half-broadcast affine check
        w: T.Tensor[(N,), dtype],
        acc: T.Tensor[(M, 2), "float32"],
        out: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            xf = T.alloc_shared((BM, N), "float32")
            half = T.alloc_shared((BM, N // 2), "float32")   # halved scratch
            wh = T.alloc_shared((1, N // 2), "float32")
            s = T.alloc_shared((BM, 2), "float32")
            for i in T.serial(M // BM):
                off = i * BM
                T.copy(x[off:off + BM, 0:N], xf[0:BM, 0:N])
                T.copy(w[0:N // 2], wh[0:1, 0:N // 2])
                # (1) column-half vmul, dst a half-width plane
                T.vmul(xf[0:BM, 0:N // 2], xf[0:BM, 0:N // 2], half)
                T.copy(half[0:BM, 0:N // 2], y_mul[off:off + BM, 0:N // 2])
                T.vmul(xf[0:BM, N // 2:N], xf[0:BM, N // 2:N], half)
                T.copy(half[0:BM, 0:N // 2], y_mul[off:off + BM, N // 2:N])
                # (2) column-half reduce
                T.reduce_sum(xf[0:BM, 0:N // 2], s[0:BM, 0:1], dim=1)
                T.reduce_sum(xf[0:BM, N // 2:N], s[0:BM, 1:2], dim=1)
                T.copy(s[0:BM, 0:2], y_sum[off:off + BM, 0:2])
                # (3) half-width broadcast + in-place column-half vmul
                T.vbrc(wh[0:1, 0:N // 2], half)
                T.vmul(xf[0:BM, N // 2:N], half, xf[0:BM, N // 2:N])
                T.copy(xf[0:BM, 0:N], out[off:off + BM, 0:N])
                # (4) accumulate half planes into a full-width dst via halves
                T.vcast(xf[0:BM, 0:N // 2], half)
                T.copy(half[0:BM, 0:N // 2], acc[off:off + BM, 0:1])

    return main


def main():
    x = torch.randn(M, N, dtype=torch.float32).to(torch.float16).npu()
    w = torch.randn(N, dtype=torch.float16).npu()
    y_mul = torch.empty(M, N, dtype=torch.float16, device="npu")
    y_sum = torch.empty(M, 2, dtype=torch.float32, device="npu")
    y_brc = torch.empty(M, N, dtype=torch.float16, device="npu")
    acc = torch.empty(M, 2, dtype=torch.float32, device="npu")
    out = torch.empty(M, N, dtype=torch.float16, device="npu")
    try:
        prog = build("float16")
    except Exception as exc:  # noqa: BLE001
        print(f"[colslice] COMPILE=FAIL {type(exc).__name__}: {exc}")
        return 1
    print("[colslice] COMPILE=OK")
    prog(x, w, y_mul, y_sum, y_brc, acc, out)
    torch.npu.synchronize()

    xf = x.float()
    ref_mul = x * x
    ref_sum = torch.stack([
        xf[:, : N // 2].sum(dim=1), xf[:, N // 2:].sum(dim=1)
    ], dim=1)
    wh = w.float()[: N // 2]
    ref_out = torch.cat([xf, xf[:, N // 2:] * wh], dim=1).to(torch.float16)
    d_mul = (y_mul.float() - ref_mul.float()).abs().max().item()
    d_sum = (y_sum - ref_sum).abs().max().item()
    d_out = (out.float() - ref_out.float()).abs().max().item()
    print(f"[colslice] max_diff mul={d_mul:.3e} sum={d_sum:.3e} out={d_out:.3e}")
    ok = d_mul < 2e-3 and d_sum < 0.05 and d_out < 2e-3
    print(f"[colslice] RESULT={'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
