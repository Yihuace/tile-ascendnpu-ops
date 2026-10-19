"""K4 real-shape probe: seed L0C with h, then accumulate M @ h_prev."""

import argparse

import tilelang
import tilelang.language as T
from tilelang.jit import compile as tl_compile


def build(H=2, DK=64, DV=64, CP=4, num_iters=4, dtype="float16"):
    @T.prim_func
    def main(
        ht_buffer: T.Tensor((CP, H, DK, DV), dtype),
        mt_buffer: T.Tensor((CP, H, DK, DK), dtype),
        h_prev: T.Tensor((CP, H, DK, DV), dtype),
        eye: T.Tensor((DK, DK), dtype),
        cp_h0: T.Tensor((CP, H, DK, DV), "float32"),
    ):
        with T.Kernel(H, is_npu=True) as (bh, _), T.Scope("Cube"):
            h_l1 = T.alloc_L1([DK, DV], dtype)
            hd_l1 = T.alloc_L1([DK, DV], dtype)
            m_l1 = T.alloc_L1([DK, DK], dtype)
            e_l1 = T.alloc_L1([DK, DK], dtype)
            h_l0 = T.alloc_L0C([DK, DV], "float32")

            T.copy(eye[0:DK, 0:DK], e_l1)
            for s in T.serial(num_iters):
                with T.rs("PIPE_MTE2"):
                    T.copy(ht_buffer[s, bh, 0:DK, 0:DV], h_l1)
                    T.copy(h_prev[s, bh, 0:DK, 0:DV], hd_l1)
                    T.copy(mt_buffer[s, bh, 0:DK, 0:DK], m_l1)
                T.gemm(e_l1, h_l1, h_l0, size=[DK, DK, DV], initC=True)
                T.gemm(m_l1, hd_l1, h_l0, size=[DK, DK, DV], initC=False)
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
    print(f"[K4-two-gemm] COMPILE_OK H={args.H} DK={args.DK} DV={args.DV} "
          f"CP={args.CP} iters={args.iters} backend={type(kernel).__name__}")


if __name__ == "__main__":
    main()
