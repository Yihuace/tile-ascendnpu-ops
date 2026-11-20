"""Probe: can we get plain-Var store indices with 3-D buffers, and does T.copy
work between differently-shaped buffers of equal total size?"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import collections
import re
import traceback

import tilelang
import tilelang.language as T

N = 4096
H = N // 2


def report(name, fn):
    try:
        k = fn()
        ir = k.mlir_content
        ops = collections.Counter(re.findall(r"hivm\.hir\.[a-z_0-9]+", ir))
        print(f"### {name}: OK scf.for={ir.count('scf.for')} "
              f"addf={ir.count('arith.addf')} mulf={ir.count('arith.mulf')} "
              f"hivm={dict(ops)}")
    except Exception as e:  # noqa: BLE001
        tail = str(e).splitlines()[-1][:400]
        print(f"### {name}: ERR {tail}")


def v_3d_single_stage():
    """ng=1,h=H: 3-D allocs, 1 store per loop, plain-Var store indices."""

    @tilelang.jit(out_idx=[2], target="npuir")
    def f():
        @T.prim_func
        def main(x: T.Tensor((1, N), "float32"), lut: T.Tensor((N - 1,), "float32"),
                 y: T.Tensor((1, N), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A3 = T.alloc_shared((1, 2, H), "float32")
                D3 = T.alloc_shared((2, 1, H), "float32")
                T.copy(x[cid, 0:H], A3[0, 0, 0:H])
                T.copy(x[cid, H:N], A3[0, 1, 0:H])
                for g, i in T.Parallel(1, H):
                    D3[0, g, i] = A3[g, 0, i] + A3[g, 1, i]
                for g, i in T.Parallel(1, H):
                    D3[1, g, i] = (A3[g, 0, i] - A3[g, 1, i]) * lut[i]
                T.copy(D3, y[cid, 0:N])
        return main

    return f()


def v_copy_3d_to_3d_diffshape():
    """T.copy between [2,4,8] and [4,2,8] (same total)."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def f():
        @T.prim_func
        def main(x: T.Tensor((64,), "float32"), y: T.Tensor((64,), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((2, 4, 8), "float32")
                B = T.alloc_shared((4, 2, 8), "float32")
                T.copy(x[0:64], A)
                T.copy(A, B)
                T.copy(B, y[0:64])
        return main

    return f()


def v_copy_1d_to_3d():
    """T.copy flat [64] -> 3-D [2,4,8]."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def f():
        @T.prim_func
        def main(x: T.Tensor((64,), "float32"), y: T.Tensor((64,), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((64,), "float32")
                B = T.alloc_shared((2, 4, 8), "float32")
                T.copy(x[0:64], A)
                T.copy(A, B)
                T.copy(B, y[0:64])
        return main

    return f()


def v_2stage_chain_3d():
    """Two stages, each with its own 3-D src/dst, cross-shape copies."""

    @tilelang.jit(out_idx=[2], target="npuir")
    def f():
        @T.prim_func
        def main(x: T.Tensor((1, N), "float32"), lut: T.Tensor((N - 1,), "float32"),
                 y: T.Tensor((1, N), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A3 = T.alloc_shared((1, 2, H), "float32")
                D3 = T.alloc_shared((2, 1, H), "float32")
                S3 = T.alloc_shared((2, 2, H // 2), "float32")
                E3 = T.alloc_shared((2, 2, H // 2), "float32")
                T.copy(x[cid, 0:H], A3[0, 0, 0:H])
                T.copy(x[cid, H:N], A3[0, 1, 0:H])
                for g, i in T.Parallel(1, H):
                    D3[0, g, i] = A3[g, 0, i] + A3[g, 1, i]
                for g, i in T.Parallel(1, H):
                    D3[1, g, i] = (A3[g, 0, i] - A3[g, 1, i]) * lut[i]
                T.copy(D3, S3)
                for g, i in T.Parallel(2, H // 2):
                    E3[0, g, i] = S3[g, 0, i] + S3[g, 1, i]
                for g, i in T.Parallel(2, H // 2):
                    E3[1, g, i] = (S3[g, 0, i] - S3[g, 1, i]) * lut[H // 2 + i]
                T.copy(E3, y[cid, 0:N])
        return main

    return f()


report("3d_single_stage", v_3d_single_stage)
report("copy_3d_to_3d_diffshape", v_copy_3d_to_3d_diffshape)
report("copy_1d_to_3d", v_copy_1d_to_3d)
report("2stage_chain_3d", v_2stage_chain_3d)
