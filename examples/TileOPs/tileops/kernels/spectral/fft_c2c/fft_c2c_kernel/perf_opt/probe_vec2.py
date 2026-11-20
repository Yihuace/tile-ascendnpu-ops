"""Probe: which loop form does NpuLoopVectorize accept? (single-store variants)"""
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import argparse
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa

N = 4096
REP = 12
H = N // 2


@tilelang.jit(out_idx=[2], target="npuir")
def k_flat(rep=REP):
    """flat 1-D buffers, affine store index."""
    @T.prim_func
    def main(x: T.Tensor((1, N), "float32"), lut: T.Tensor((N - 1,), "float32"),
             y: T.Tensor((1, N), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((N,), "float32")
            Bc = T.alloc_shared((N,), "float32")
            T.copy(x[cid, 0:N], A)
            for _r in T.serial(rep):
                for i in T.Parallel(H):
                    Bc[i] = A[i] + A[i + H]
                for i in T.Parallel(H):
                    Bc[i + H] = (A[i] - A[i + H]) * lut[i]
                for i in T.Parallel(H):
                    A[i] = Bc[i]
                for i in T.Parallel(H):
                    A[i + H] = Bc[i + H]
            T.copy(A, y[cid, 0:N])
    return main


@tilelang.jit(out_idx=[2], target="npuir")
def k_view(rep=REP):
    """3-D views, plain store index, one store per loop."""
    @T.prim_func
    def main(x: T.Tensor((1, N), "float32"), lut: T.Tensor((N - 1,), "float32"),
             y: T.Tensor((1, N), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((N,), "float32")
            Bc = T.alloc_shared((N,), "float32")
            T.copy(x[cid, 0:N], A)
            S2 = T.view(A, [1, 2 * H])
            D3 = T.view(Bc, [2, 1, H])
            for _r in T.serial(rep):
                for g, i in T.Parallel(1, H):
                    D3[0, g, i] = S2[g, i] + S2[g, H + i]
                for g, i in T.Parallel(1, H):
                    D3[1, g, i] = (S2[g, i] - S2[g, H + i]) * lut[i]
                for g, i in T.Parallel(1, H):
                    S2[g, i] = D3[0, g, i]
                for g, i in T.Parallel(1, H):
                    S2[g, H + i] = D3[1, g, i]
            T.copy(A, y[cid, 0:N])
    return main


@tilelang.jit(out_idx=[2], target="npuir")
def k_view_vop(rep=REP):
    """3-D views + explicit v-prefix ops on 1-D segment views."""
    @T.prim_func
    def main(x: T.Tensor((1, N), "float32"), lut: T.Tensor((N - 1,), "float32"),
             y: T.Tensor((1, N), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((N,), "float32")
            Bc = T.alloc_shared((N,), "float32")
            T.copy(x[cid, 0:N], A)
            A0 = T.view(A, [2, H])
            B0 = T.view(Bc, [2, H])
            for _r in T.serial(rep):
                T.vadd(A0[0, 0:H], A0[1, 0:H], B0[0, 0:H])
                T.vsub(A0[0, 0:H], A0[1, 0:H], B0[1, 0:H])
                T.vadd(B0[0, 0:H], B0[1, 0:H], A0[0, 0:H])
                T.vsub(B0[0, 0:H], B0[1, 0:H], A0[1, 0:H])
            T.copy(A, y[cid, 0:N])
    return main


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--p", default="flat")
    a = ap.parse_args()
    kf = {"flat": k_flat, "view": k_view, "view_vop": k_view_vop}[a.p]
    k = kf()
    x = torch.randn(1, N).npu(); lut = torch.randn(N - 1).npu()
    y = k(x, lut); torch.npu.synchronize()
    for _ in range(20):
        y = k(x, lut)
    torch.npu.synchronize()
    print("ok", a.p, tuple(y.shape))
