#!/usr/bin/env python3
"""Compile-only probe: which (1,1) scalar-plane combination ops are legal?

R3-W1 hit `hivm.hir.vadd op failed to verify that operand at index 0 is
vector-only`.  This probe isolates the (1,1) forms.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import tilelang
import tilelang.language as T


def build(mode):
    @tilelang.jit(target="npuir")
    def _f():
        @T.prim_func
        def main(
            a: T.Tensor((8, 64), "float32"),
            b: T.Tensor((8, 64), "float32"),
            c: T.Tensor((8, 64), "float32"),
            s0: T.Tensor((1,), "float32"),
            s1: T.Tensor((1,), "float32"),
            o: T.Tensor((8, 64), "float32"),
            o2: T.Tensor((1,), "float32"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, subid):
                with T.Scope("Vector"):
                    A = T.alloc_ub((8, 64), "float32")
                    B = T.alloc_ub((8, 64), "float32")
                    C = T.alloc_ub((8, 64), "float32")
                    P = T.alloc_ub((1, 1), "float32")
                    Q = T.alloc_ub((1, 1), "float32")
                    R = T.alloc_ub((1, 1), "float32")
                    BR = T.alloc_ub((8, 64), "float32")
                    T.copy(a, A)
                    T.copy(b, B)
                    T.copy(s0, P)
                    T.copy(s1, Q)
                    T.vmul(P, Q, R)
                    if mode == "vadd11":
                        T.vadd(R, P, R)
                    elif mode == "vadd11_noalias":
                        T.vadd(R, P, Q)
                        T.copy(Q, o2)
                    elif mode == "vsub11":
                        T.vsub(R, P, R)
                    elif mode == "vmul11":
                        T.vmul(R, P, R)
                    elif mode == "vadd_full":
                        T.vbrc(P, BR)
                        T.vadd(A, BR, C)
                        T.copy(C, o)
                    T.copy(R, o2)
        return main

    return _f


for mode in ("vmul11", "vadd11", "vadd11_noalias", "vsub11", "vadd_full"):
    try:
        build(mode)()
        print(f"{mode}: COMPILE OK")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).splitlines()
        print(f"{mode}: FAIL -> {msg[0][:160]}")
