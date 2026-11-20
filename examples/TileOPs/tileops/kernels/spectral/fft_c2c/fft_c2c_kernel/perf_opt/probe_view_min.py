"""Minimal localisation of the T.view/T.reshape SSA failure.

G1: one T.reshape on a shared buffer, one T.Parallel loop store.
G2: one T.view on a shared buffer, one T.Parallel loop store.
G3: two different-shaped views of the SAME buffer (unique names), two loops.
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T

N = 8


def g1():
    @tilelang.jit(out_idx=[1], target="npuir")
    def f():
        @T.prim_func
        def main(x: T.Tensor((N,), "float32"),
                 y: T.Tensor((N,), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                T.copy(x[0:N], A)
                A2 = T.reshape(A, [N // 2, 2])
                for i, j in T.Parallel(N // 2, 2):
                    A2[i, j] = A2[i, j] + 1.0
                T.copy(A, y[0:N])
        return main
    return f()


def g2():
    @tilelang.jit(out_idx=[1], target="npuir")
    def f():
        @T.prim_func
        def main(x: T.Tensor((N,), "float32"),
                 y: T.Tensor((N,), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                T.copy(x[0:N], A)
                A2 = T.view(A, [N // 2, 2])
                for i, j in T.Parallel(N // 2, 2):
                    A2[i, j] = A2[i, j] + 1.0
                T.copy(A, y[0:N])
        return main
    return f()


def g3():
    @tilelang.jit(out_idx=[1], target="npuir")
    def f():
        @T.prim_func
        def main(x: T.Tensor((N,), "float32"),
                 y: T.Tensor((N,), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                B = T.alloc_shared((N,), "float32")
                T.copy(x[0:N], A)
                A_v0 = T.view(A, [N // 2, 2])
                A_v1 = T.view(A, [2, N // 2])
                B_v0 = T.view(B, [N // 2, 2])
                for i, j in T.Parallel(N // 2, 2):
                    B_v0[i, j] = A_v0[i, j] + 1.0
                for i, j in T.Parallel(2, N // 2):
                    A_v1[i, j] = B_v0[j, i]
                T.copy(A, y[0:N])
        return main
    return f()


for name, fn in (("G1_reshape", g1), ("G2_view", g2), ("G3_two_views", g3)):
    try:
        k = fn()
        ir = k.mlir_content
        print(f"{name}: COMPILE OK scf.for={ir.count('scf.for')} "
              f"npuir_add={ir.count('npuir_add')}")
    except Exception as e:  # noqa: BLE001
        print(f"{name}: FAIL {str(e).splitlines()[-1][:200]}")
