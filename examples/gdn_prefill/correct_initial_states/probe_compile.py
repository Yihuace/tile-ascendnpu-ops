"""K4 `correct_initial_states` compile probe (CANN 9.0, no NPU execution).

Purpose: check whether the K4 shape family (Cube GEMM + GM/L1 ping-pong,
state_v_first=False) can be lowered by the 9.0 closed-source hivmc 0.2.0.

Design used by the probe (matches K4's math, minus varlen/mask branches):

    h_0 = ht[0]
    for s in 0..N-2:
        h_prev = cp_h0[s]              # previous corrected state
        h     = ht[s+1]
        h     = I * h                  # seed L0C (NPU: L0C only takes a GEMM)
        h     = h + M[s+1] * h_prev    # correction
        cp_h0[s+1] = h

All GM stores are L0C -> GM (`PIPE_FIX`); no L1 -> GM `store`, which the
9.0 HIR pipeline rejects.
"""

import argparse

import tilelang
import tilelang.language as T
from tilelang.jit import compile as tl_compile


def build(H=2, DK=64, DV=64, CP=4, num_iters=4, dtype="float16"):
    @T.prim_func
    def main(
        ht_buffer: T.Tensor((CP, H, DK, DV), dtype),
        mt_buffer: T.Tensor((CP, H, DK, DK), dtype),
        h_in: T.Tensor((CP, H, DK, DV), dtype),
        cp_h0: T.Tensor((CP, H, DK, DV), "float32"),
    ):
        with T.Kernel(H, is_npu=True) as (bh, _), T.Scope("Cube"):
            hd_l1 = T.alloc_L1([DK, DV], dtype)
            m_l1 = T.alloc_L1([DK, DK], dtype)
            h_l0 = T.alloc_L0C([DK, DV], "float32")

            for s in T.serial(num_iters):
                with T.rs("PIPE_MTE2"):
                    T.copy(h_in[s, bh, 0:DK, 0:DV], hd_l1)
                    T.copy(mt_buffer[s, bh, 0:DK, 0:DK], m_l1)
                T.gemm(m_l1, hd_l1, h_l0, size=[DK, DK, DV], initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(h_l0, cp_h0[s, bh, 0:DK, 0:DV])

    return main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--DK", type=int, default=64)
    ap.add_argument("--DV", type=int, default=64)
    ap.add_argument("--CP", type=int, default=4)
    ap.add_argument("--iters", type=int, default=4)
    args = ap.parse_args()

    func = build(args.H, args.DK, args.DV, args.CP, args.iters)
    kernel = tl_compile(func, target="npuir")
    print(f"[K4-probe] COMPILE_OK H={args.H} DK={args.DK} DV={args.DV} "
          f"CP={args.CP} iters={args.iters} backend={type(kernel).__name__}")


if __name__ == "__main__":
    main()
