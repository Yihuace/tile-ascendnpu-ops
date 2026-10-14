"""Scratch v2: probe individual npuir constructs needed by BatchNormBwd.

Fixed: per-channel outputs (C,N,S) so blocks don't race on one tensor.
Run: python _scratch_constructs.py
"""
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import traceback
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

N, C, S = 4, 8, 16
L = N * S
DT = "float32"


def run(name, fn):
    print(f"\n===== {name} =====", flush=True)
    try:
        fn()
        print(f"[{name}] OK", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


def t1_strided_plane_load():
    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(
            x: T.Tensor((N, C, S), DT),
            y: T.Tensor((C, N, S), DT),
        ):
            with T.Kernel(C, is_npu=True) as (cid, _):
                plane = T.alloc_shared((N, S), DT)
                T.copy(x[0:N, cid, 0:S], plane)
                T.copy(plane, y[cid, 0:N, 0:S])
        return main
    k = _f()
    x = torch.randn(N, C, S).npu()
    y = k(x).cpu()
    ref = x.cpu().permute(1, 0, 2)
    err = (y - ref).abs().max().item()
    print("  plane gather err:", err)
    assert err < 1e-5


def t2_scalar_slice_load_vbrc():
    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(
            w: T.Tensor((C,), DT),
            y: T.Tensor((C, N, S), DT),
        ):
            with T.Kernel(C, is_npu=True) as (cid, _):
                w1 = T.alloc_shared((1, 1), DT)
                plane = T.alloc_shared((N, S), DT)
                T.copy(w[cid : cid + 1], w1)
                T.vbrc(w1, plane)
                T.copy(plane, y[cid, 0:N, 0:S])
        return main
    k = _f()
    w = torch.randn(C).npu()
    y = k(w).cpu()
    ref = w.cpu().view(C, 1, 1).expand(C, N, S)
    err = (y - ref).abs().max().item()
    print("  scalar load + vbrc(1,1)->(N,S) err:", err)
    assert err < 1e-5


def t3_two_axis_reduce():
    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(
            x: T.Tensor((N, C, S), DT),
            w: T.Tensor((C,), DT),
            gb: T.Tensor((C,), DT),
        ):
            with T.Kernel(C, is_npu=True) as (cid, _):
                plane = T.alloc_shared((N, S), DT)
                red = T.alloc_shared((N, 1), DT)
                tot = T.alloc_shared((1, 1), DT)
                T.copy(x[0:N, cid, 0:S], plane)
                T.reduce(plane, red, dims=[1], reduce_mode="sum", clear=True)
                T.reduce(red, tot, dims=[0], reduce_mode="sum", clear=True)
                T.copy(tot, gb[cid : cid + 1])
        return main
    k = _f()
    x = torch.randn(N, C, S).npu()
    w = torch.randn(C).npu()
    gb = k(x, w).cpu()
    ref = x.cpu().sum(dim=(0, 2))
    err = (gb - ref).abs().max().item()
    print("  two-axis reduce err:", err)
    assert err < 1e-3


def t5_epilogue_broadcast():
    # plane [N,S]; per-channel scalars m,r,w.  xh=(plane-m)*r; out=w*(L*plane - tot) ...
    @tilelang.jit(out_idx=[5], target="npuir")
    def _f():
        @T.prim_func
        def main(
            x: T.Tensor((N, C, S), DT),
            weight: T.Tensor((C,), DT),
            mean: T.Tensor((C,), DT),
            rstd: T.Tensor((C,), DT),
            gb: T.Tensor((C,), DT),
            y: T.Tensor((C, N, S), DT),
        ):
            with T.Kernel(C, is_npu=True) as (cid, _):
                plane = T.alloc_shared((N, S), DT)
                prod = T.alloc_shared((N, S), DT)
                brc = T.alloc_shared((N, S), DT)
                red = T.alloc_shared((N, 1), DT)
                tot = T.alloc_shared((1, 1), DT)
                w1 = T.alloc_shared((1, 1), DT)
                m1 = T.alloc_shared((1, 1), DT)
                r1 = T.alloc_shared((1, 1), DT)
                T.copy(x[0:N, cid, 0:S], plane)
                T.copy(weight[cid : cid + 1], w1)
                T.copy(mean[cid : cid + 1], m1)
                T.copy(rstd[cid : cid + 1], r1)
                # xh = (plane - m)*r
                T.vbrc(m1, brc)
                T.vsub(plane, brc, plane)
                T.vbrc(r1, brc)
                T.vmul(plane, brc, plane)
                # tot = sum(plane)
                T.reduce(plane, red, dims=[1], reduce_mode="sum", clear=True)
                T.reduce(red, tot, dims=[0], reduce_mode="sum", clear=True)
                # out = w * (L*plane - tot)
                T.vmul(plane, float(L), prod)
                T.vbrc(tot, brc)
                T.vsub(prod, brc, prod)
                T.vbrc(w1, brc)
                T.vmul(prod, brc, prod)
                T.copy(prod, y[cid, 0:N, 0:S])
        return main
    k = _f()
    x = torch.randn(N, C, S).npu()
    weight = torch.randn(C).npu()
    mean = torch.randn(C).npu()
    rstd = torch.randn(C).npu()
    gb = torch.randn(C).npu()
    y = k(x, weight, mean, rstd, gb).cpu()
    xc = x.cpu()
    xh = (xc - mean.cpu().view(1, C, 1)) * rstd.cpu().view(1, C, 1)
    ref = weight.cpu().view(1, C, 1) * (L * xh - xh.sum(dim=(0, 2)).view(1, C, 1))
    ref = ref.permute(1, 0, 2).contiguous()
    err = (y - ref).abs().max().item()
    print("  epilogue broadcast err:", err)
    assert err < 1e-3


if __name__ == "__main__":
    print("device:", torch.npu.get_device_name(0))
    results = {}
    results["t1_strided_plane_load"] = run("t1_strided_plane_load", t1_strided_plane_load)
    results["t2_scalar_slice_load_vbrc"] = run("t2_scalar_slice_load_vbrc", t2_scalar_slice_load_vbrc)
    results["t3_two_axis_reduce"] = run("t3_two_axis_reduce", t3_two_axis_reduce)
    results["t5_epilogue_broadcast"] = run("t5_epilogue_broadcast", t5_epilogue_broadcast)
    print("\n===== SUMMARY =====")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
