"""Isolate the T.view SSA failure: how many / where can T.view be declared?"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import sys

import tilelang
import tilelang.language as T

N = 64


def build(mode):
    @tilelang.jit(out_idx=[1], target="npuir")
    def f(mode=mode):
        @T.prim_func
        def main(x: T.Tensor((N,), "float32"), y: T.Tensor((N,), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                Bc = T.alloc_shared((N,), "float32")
                T.copy(x[0:N], A)
                if mode == "one_view":
                    A1 = T.view(A, [N // 2, 2])
                    for i, j in T.Parallel(N // 2, 2):
                        Bc[i * 2 + j] = A1[i, j]
                elif mode == "two_views_same":
                    A1 = T.view(A, [N // 2, 2])
                    B1 = T.view(Bc, [N // 2, 2])
                    for i, j in T.Parallel(N // 2, 2):
                        B1[i, j] = A1[j, i] + 0.0
                elif mode == "view_in_two_seq_loops":
                    A1 = T.view(A, [N // 2, 2])
                    B1 = T.view(Bc, [N // 2, 2])
                    for i, j in T.Parallel(N // 2, 2):
                        B1[0, j] = A1[0, j]
                    A2 = T.view(A, [N // 4, 4])
                    for i, j in T.Parallel(N // 4, 4):
                        B1[0, j] = A2[0, j]
                T.copy(Bc, y[0:N])

        return main

    return f()


for mode in ("one_view", "two_views_same", "view_in_two_seq_loops"):
    try:
        k = build(mode)
        print(f"{mode}: COMPILE_OK")
    except Exception as e:  # noqa: BLE001
        print(f"{mode}: ERR {str(e).splitlines()[-1][:200]}")
