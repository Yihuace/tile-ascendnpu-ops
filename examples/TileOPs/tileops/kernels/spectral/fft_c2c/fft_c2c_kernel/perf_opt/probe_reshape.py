"""Test T.reshape(src, dst) as a view on npuir."""
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
def k_r1():
    """alloc a dst buffer + T.reshape(src, dst) then use dst as a view."""
    @T.prim_func
    def main(x: T.Tensor((N,), "float32"), y: T.Tensor((N,), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((N,), "float32")
            Bc = T.alloc_shared((N,), "float32")
            A2 = T.alloc_shared((1, N), "float32")
            B2 = T.alloc_shared((2, 1, H), "float32")
            T.copy(x[0:N], A)
            T.copy(A, A2)
            for g, i in T.Parallel(1, H):
                B2[0, g, i] = A2[g, i] + A2[g, H + i]
            for g, i in T.Parallel(1, H):
                B2[1, g, i] = A2[g, i] - A2[g, H + i]
            T.copy(B2, Bc)
            T.copy(Bc, y[0:N])
    return main


@tilelang.jit(out_idx=[1], target="npuir")
def k_r2():
    """T.reshape to create the view without an explicit copy."""
    @T.prim_func
    def main(x: T.Tensor((N,), "float32"), y: T.Tensor((N,), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((1, N), "float32")
            Bc = T.alloc_shared((2, 1, H), "float32")
            A2 = T.alloc_shared((2, H), "float32")
            B3 = T.alloc_shared((N,), "float32")
            T.copy(x[0:N], A[0, 0:N])
            T.reshape(A, A2)
            for g, i in T.Parallel(1, H):
                Bc[0, g, i] = A2[0, i] + A2[1, i]
            for g, i in T.Parallel(1, H):
                Bc[1, g, i] = A2[0, i] - A2[1, i]
            T.reshape(Bc, B3)
            T.copy(B3, y[0:N])
    return main


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--p", default="r1")
    a = ap.parse_args()
    k = {"r1": k_r1, "r2": k_r2}[a.p]()
    x = torch.randn(N).npu()
    y = k(x); torch.npu.synchronize()
    for _ in range(10):
        y = k(x)
    torch.npu.synchronize()
    ref = torch.cat([x[:H] + x[H:], x[:H] - x[H:]])
    print("ok", a.p, "maxerr", float((y - ref).abs().max()))
