"""Last alias-form probe before concluding: are ANY buffer aliases usable on
npuir (shared/UB) so a per-stage [ng,2,h] view can be built?

H1: official view pattern  A2 = T.view(A,[N//2,2]); T.copy(A2, y2)
H2: explicit T.Buffer(..., data=A.data, scope="shared")
H3: explicit T.Buffer(..., data=A.data, scope="local")
H4: single T.view + T.Parallel loop, one store, Var index
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T

N = 8


def h1():
    @tilelang.jit(out_idx=[1], target="npuir")
    def f():
        @T.prim_func
        def main(x: T.Tensor((N,), "float32"),
                 y: T.Tensor((N // 2, 2), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                T.copy(x[0:N], A)
                A2 = T.view(A, [N // 2, 2])
                T.copy(A2, y[0:N // 2, 0:2])
        return main
    return f()


def h2(scope):
    @tilelang.jit(out_idx=[1], target="npuir")
    def f(scope=scope):
        @T.prim_func
        def main(x: T.Tensor((N,), "float32"),
                 y: T.Tensor((N,), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                T.copy(x[0:N], A)
                A2 = T.Buffer([N // 2, 2], "float32", A.data, scope=scope)
                for i, j in T.Parallel(N // 2, 2):
                    A2[i, j] = A2[i, j] + 1.0
                T.copy(A, y[0:N])
        return main
    return f()


def h4():
    @tilelang.jit(out_idx=[1], target="npuir")
    def f():
        @T.prim_func
        def main(x: T.Tensor((N,), "float32"),
                 y: T.Tensor((N,), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                B = T.alloc_shared((N,), "float32")
                T.copy(x[0:N], A)
                A2 = T.view(A, [N // 2, 2])
                B2 = T.view(B, [N // 2, 2])
                for i, j in T.Parallel(N // 2, 2):
                    B2[i, j] = A2[i, j] * 2.0
                T.copy(B, y[0:N])
        return main
    return f()


for name, fn in (("H1_view_copy", h1),
                 ("H2_shared_scope", lambda: h2("shared")),
                 ("H3_local_scope", lambda: h2("local")),
                 ("H4_view_parallel", h4)):
    try:
        k = fn()
        print(f"{name}: COMPILE OK")
    except Exception as e:  # noqa: BLE001
        print(f"{name}: FAIL {str(e).splitlines()[-1][:180]}")
