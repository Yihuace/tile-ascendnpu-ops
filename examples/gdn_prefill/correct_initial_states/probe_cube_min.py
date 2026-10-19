"""Minimal Cube GEMM compile probe: A[64,64] @ B[64,64] -> C[64,64]."""

import argparse

import tilelang
import tilelang.language as T
from tilelang.jit import compile as tl_compile


def build(M=64, N=64, K=64, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, is_npu=True), T.Scope("Cube"):
            a = T.alloc_L1([M, K], dtype)
            b = T.alloc_L1([K, N], dtype)
            c = T.alloc_L0C([M, N], "float32")
            T.copy(A[0:M, 0:K], a)
            T.copy(B[0:K, 0:N], b)
            T.gemm(a, b, c, size=[M, K, N], initC=True)
            with T.rs("PIPE_FIX"):
                T.copy(c, C[0:M, 0:N])

    return main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--K", type=int, default=64)
    args = ap.parse_args()
    func = build(args.M, args.N, args.K)
    kernel = tl_compile(func, target="npuir")
    print(f"[cube-min] COMPILE_OK M={args.M} N={args.N} K={args.K} backend={type(kernel).__name__}")


if __name__ == "__main__":
    main()
