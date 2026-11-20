"""Find a working buffer-view form on npuir."""
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import argparse
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa
N = 4096
H = N // 2


@tilelang.jit(out_idx=[1], target="npuir")
def k_v1():
    """view assigned at kernel top, no serial loop."""
    @T.prim_func
    def main(x: T.Tensor((N,), "float32"), y: T.Tensor((N,), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((N,), "float32")
            Bc = T.alloc_shared((N,), "float32")
            T.copy(x[0:N], A)
            S2 = T.view(A, [1, N])
            D3 = T.view(Bc, [2, 1, H])
            for g, i in T.Parallel(1, H):
                D3[0, g, i] = S2[g, i] + S2[g, H + i]
            for g, i in T.Parallel(1, H):
                D3[1, g, i] = S2[g, i] - S2[g, H + i]
            T.copy(Bc, y[0:N])
    return main


@tilelang.jit(out_idx=[1], target="npuir")
def k_v2():
    """alloc_shared declared directly in the 3-D shape used by the loop."""
    @T.prim_func
    def main(x: T.Tensor((N,), "float32"), y: T.Tensor((N,), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((1, N), "float32")
            Bc = T.alloc_shared((2, 1, H), "float32")
            T.copy(x[0:N], A[0, 0:N])
            for g, i in T.Parallel(1, H):
                Bc[0, g, i] = A[g, i] + A[g, H + i]
            for g, i in T.Parallel(1, H):
                Bc[1, g, i] = A[g, i] - A[g, H + i]
            T.copy(Bc, y[0:N])
    return main


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--p", default="v1")
    a = ap.parse_args()
    k = {"v1": k_v1, "v2": k_v2}[a.p]()
    x = torch.randn(N).npu()
    y = k(x); torch.npu.synchronize()
    for _ in range(10):
        y = k(x)
    torch.npu.synchronize()
    print("ok", a.p, float((y[:5] - (x[:5] + x[H:H + 5])).abs().max()))
