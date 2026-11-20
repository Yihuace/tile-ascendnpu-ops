"""Minimal probe: which loop form does NpuLoopVectorize accept?

Runs one synthetic 'stage' (12 times) over a 4096-length row.
  P1: flat buffer, affine store index, 4 stores in one loop   (baseline style)
  P2: 3-D views, plain store index, 4 stores in one loop
  P3: 3-D views, plain store index, 1 store per loop (4 loops)
"""
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import sys, math, argparse
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa

N = 4096
REP = 12


@tilelang.jit(out_idx=[2], target="npuir")
def k_p1(rep=REP):
    @T.prim_func
    def main(x: T.Tensor((1, N), "float32"), lut: T.Tensor((N - 1,), "float32"),
             y: T.Tensor((1, N), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((N,), "float32")
            Bc = T.alloc_shared((N,), "float32")
            T.copy(x[cid, 0:N], A)
            for _r in T.serial(rep):
                for i in T.Parallel(N // 2):
                    Bc[i] = A[i] + A[i + N // 2]
                for i in T.Parallel(N // 2):
                    Bc[i + N // 2] = (A[i] - A[i + N // 2]) * lut[i]
                for i in T.Parallel(N // 2):
                    A[i] = Bc[i]
                for i in T.Parallel(N // 2):
                    A[i + N // 2] = Bc[i + N // 2]
            T.copy(A, y[cid, 0:N])
    return main


@tilelang.jit(out_idx=[2], target="npuir")
def k_p3(rep=REP):
    @T.prim_func
    def main(x: T.Tensor((1, N), "float32"), lut: T.Tensor((N - 1,), "float32"),
             y: T.Tensor((1, N), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((N,), "float32")
            Bc = T.alloc_shared((N,), "float32")
            T.copy(x[cid, 0:N], A)
            for _r in T.serial(rep):
                S2 = T.view(A, [N // (2 * (N // 2)), 2 * (N // 2)])
                D3 = T.view(Bc, [2, 1, N // 2])
                for g, i in T.Parallel(1, N // 2):
                    D3[0, g, i] = S2[g, i] + S2[g, g + i]
                for g, i in T.Parallel(1, N // 2):
                    D3[1, g, i] = (S2[g, i] - S2[g, i]) * lut[i]
                for g, i in T.Parallel(1, N // 2):
                    S2[0 * g, i] = D3[0, g, i]
                for g, i in T.Parallel(1, N // 2):
                    S2[0, i] = D3[1, g, i]
            T.copy(A, y[cid, 0:N])
    return main


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--p", default="p1")
    a = ap.parse_args()
    k = k_p1() if a.p == "p1" else k_p3()
    x = torch.randn(1, N).npu(); lut = torch.randn(N - 1).npu()
    y = k(x, lut)
    torch.npu.synchronize()
    for _ in range(20):
        y = k(x, lut)
    torch.npu.synchronize()
    print("ok", a.p, tuple(y.shape))
