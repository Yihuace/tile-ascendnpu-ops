#!/usr/bin/env python3
"""Minimal probe: is ``T.vcast`` with identical src/dst dtype an identity?

R6 found this while building the broadcast kernels: an fp32 kernel that ran the
"cast to compute dtype" chain (``vcast(fp32_ub -> fp32_ub)``) produced wrong
values for **every** element (12.8 M/12.8 M mismatches, max_diff 1.0), while
the same chain with fp16/bf16 sources was bit-exact.  Switching fp32 to a
direct add (no cast) fixed it -- the fast path is now bitwise equal to the flat
path for all dtypes.

This probe isolates dtype vs shape:

    A  (n,)    fp32 -> fp32   (flat tile, the shape the kernel would use)
    B  (p,1)   fp32 -> fp32   (the tiny per-plane scalar buffer)
    C  (p,L)   fp32 -> fp32   (2D tile)
    D  (n,)    fp16 -> fp32   (control: the documented path)

    python3 r6_vcast_probe.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


@tilelang.jit(out_idx=[1], target="npuir")
def cast_flat(n, src_dtype, dst_dtype):
    @T.prim_func
    def main(
        a: T.Tensor((n,), src_dtype),
        y: T.Tensor((n,), dst_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((n,), src_dtype)
            y_ub = T.alloc_ub((n,), dst_dtype)
            T.copy(a, a_ub)
            T.vcast(a_ub, y_ub, round_mode="rint")
            T.copy(y_ub, y)

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def cast_2d(p, l, src_dtype, dst_dtype):
    @T.prim_func
    def main(
        a: T.Tensor((p, l), src_dtype),
        y: T.Tensor((p, l), dst_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((p, l), src_dtype)
            y_ub = T.alloc_ub((p, l), dst_dtype)
            T.copy(a, a_ub)
            T.vcast(a_ub, y_ub, round_mode="rint")
            T.copy(y_ub, y)

    return main


def main():
    torch.manual_seed(0)
    report = []

    # A: flat fp32 -> fp32
    n = 8192
    a = torch.randn(n, dtype=torch.float32, device="npu")
    y = cast_flat(n, "float32", "float32")(a)
    same = torch.equal(y.cpu().view(torch.int32), a.cpu().view(torch.int32))
    md = (y.float() - a.float()).abs().max().item()
    report.append(("A flat fp32->fp32", n, same, md))

    # B: tiny per-plane scalar buffer (p,1) fp32 -> fp32
    p = 2
    a = torch.randn(p, 1, dtype=torch.float32, device="npu")
    y = cast_2d(p, 1, "float32", "float32")(a)
    same = torch.equal(y.cpu().view(torch.int32), a.cpu().view(torch.int32))
    md = (y.float() - a.float()).abs().max().item()
    report.append(("B (p,1) fp32->fp32", p, same, md))

    # C: 2D tile (p,L) fp32 -> fp32
    p, l = 2, 3136
    a = torch.randn(p, l, dtype=torch.float32, device="npu")
    y = cast_2d(p, l, "float32", "float32")(a)
    same = torch.equal(y.cpu().view(torch.int32), a.cpu().view(torch.int32))
    md = (y.float() - a.float()).abs().max().item()
    report.append(("C (p,L) fp32->fp32", p * l, same, md))

    # D: control, fp16 -> fp32
    n = 8192
    a = torch.randn(n, dtype=torch.float16, device="npu")
    y = cast_flat(n, "float16", "float32")(a)
    md = (y.float() - a.float()).abs().max().item()
    report.append(("D flat fp16->fp32", n, md == 0.0, md))

    bad = 0
    for name, elems, same, md in report:
        print(f"[vcast] {name:22s} elems={elems:6d} identity={same} max_diff={md:.3e}")
        if not same:
            bad += 1
    print(f"[vcast] non-identity cases: {bad}/{len(report)}")
    if bad:
        raise SystemExit("[vcast] reproduced: same-dtype vcast is not an identity")


if __name__ == "__main__":
    main()
