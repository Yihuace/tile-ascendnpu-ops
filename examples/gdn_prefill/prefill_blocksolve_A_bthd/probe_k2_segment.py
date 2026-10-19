"""K2 split probe: one Cube-only segment with two independent GEMMs.

Segment computes:
    tmp = P @ T
    P2  = P @ P
Both are single-GEMM, independent L0C outputs; no chained initC=False.
A separate Vector kernel can do T2 = T + tmp.
"""

import argparse

import tilelang
import tilelang.language as T
from tilelang.jit import compile as tl_compile


def build(TT=4, C=64, dtype="float32"):
    @T.prim_func
    def main(
        T_in: T.Tensor((TT, C, C), dtype),
        P_in: T.Tensor((TT, C, C), dtype),
        tmp_out: T.Tensor((TT, C, C), dtype),
        p2_out: T.Tensor((TT, C, C), dtype),
    ):
        with T.Kernel(TT, is_npu=True) as (t, _), T.Scope("Cube"):
            p_l1 = T.alloc_L1([C, C], dtype)
            t_l1 = T.alloc_L1([C, C], dtype)
            tmp_l0 = T.alloc_L0C([C, C], dtype)
            p2_l0 = T.alloc_L0C([C, C], dtype)
            T.copy(T_in[t, 0:C, 0:C], t_l1)
            T.copy(P_in[t, 0:C, 0:C], p_l1)
            T.gemm(p_l1, t_l1, tmp_l0, size=[C, C, C], initC=True)
            T.gemm(p_l1, p_l1, p2_l0, size=[C, C, C], initC=True)
            with T.rs("PIPE_FIX"):
                T.copy(tmp_l0, tmp_out[t, 0:C, 0:C])
                T.copy(p2_l0, p2_out[t, 0:C, 0:C])

    return main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    func = build(args.tasks, args.C, args.dtype)
    kernel = tl_compile(func, target="npuir")
    print(f"[K2-seg] COMPILE_OK tasks={args.tasks} C={args.C} dtype={args.dtype} "
          f"backend={type(kernel).__name__}")


if __name__ == "__main__":
    main()
