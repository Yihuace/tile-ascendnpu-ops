"""Probe: create all per-stage views ONCE at prim_func top level as lists
(unique names via list binding), then pass them through the macro chain and
index them inline (no per-stage T.view binding, so no name collision).

P1: single component, n=8, t=3 -- compile + correctness.
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T

N = 8
T_STAGES = 3
SRC_SHAPES = [[N // (2 * (1 << (T_STAGES - 1 - s))), 2, (1 << (T_STAGES - 1 - s))]
              for s in range(T_STAGES)]
DST_SHAPES = [[2, N // (2 * (1 << (T_STAGES - 1 - s))), (1 << (T_STAGES - 1 - s))]
              for s in range(T_STAGES)]


def view_lists(buf, shapes):
    return [T.view(buf, sh) for sh in shapes]


def pick(even_list, odd_list, s):
    return even_list if s % 2 == 0 else odd_list


def build_chain(s):
    if s == T_STAGES:
        @T.macro
        def done(srcA, srcB, dstA, dstB):
            pass
        return done
    nxt = build_chain(s + 1)
    h = 1 << (T_STAGES - 1 - s)
    ng = N // (2 * h)

    @T.macro
    def stage(srcA, srcB, dstA, dstB):
        for g, i in T.Parallel(ng, h):
            pick(dstB, dstA, s)[s][0, g, i] = (
                pick(srcA, srcB, s)[s][g, 0, i]
                + pick(srcA, srcB, s)[s][g, 1, i])
        for g, i in T.Parallel(ng, h):
            pick(dstB, dstA, s)[s][1, g, i] = (
                pick(srcA, srcB, s)[s][g, 0, i]
                - pick(srcA, srcB, s)[s][g, 1, i])
        nxt(dstB, dstA, srcB, srcA)
    return stage


@tilelang.jit(out_idx=[1], target="npuir")
def f():
    chain = build_chain(0)

    @T.prim_func
    def main(x: T.Tensor((1, N), "float32"),
             y: T.Tensor((1, N), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((N,), "float32")
            B = T.alloc_shared((N,), "float32")
            T.copy(x[cid, 0:N], A)
            srcA = view_lists(A, SRC_SHAPES)
            dstA = view_lists(A, DST_SHAPES)
            srcB = view_lists(B, SRC_SHAPES)
            dstB = view_lists(B, DST_SHAPES)
            chain(srcA, srcB, dstA, dstB)
            if T_STAGES % 2 == 1:
                T.copy(B, y[cid, 0:N])
            else:
                T.copy(A, y[cid, 0:N])

    return main


try:
    k = f()
    ir = k.mlir_content
    print(f"P1: COMPILE OK  scf.for={ir.count('scf.for')} "
          f"vector={ir.count('vector')} npuir_add={ir.count('npuir_add')} "
          f"npuir_sub={ir.count('npuir_sub')}")
    import torch
    import torch_npu  # noqa: F401
    x = torch.arange(N, dtype=torch.float32).reshape(1, N)
    y = k(x.npu())[0].cpu()
    print("P1 out:", y.tolist())
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    print(f"P1: FAIL {str(e).splitlines()[-1][:300]}")
