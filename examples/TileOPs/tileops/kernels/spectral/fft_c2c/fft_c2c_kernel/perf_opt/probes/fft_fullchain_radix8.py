"""FFT v3: v2's per-stage parallel kernels for stages 1..9, plus a FUSED
radix-8 kernel that replaces the last three radix-2 stages.

Why fuse the tail
-----------------
v2 per-stage timings (T.Kernel(nb), 40 AIV cores):

    stage  1..5 : 1.7 - 2.7 us      (nb = 1..16)
    stage  6..9 : 4.2 - 20.4 us     (nb = 32..256)
    stage 10..12: 42.4 / 73.4 / 129.9 us  (nb = 512 / 1024 / 2048, m = 4 / 2 / 1)
                                     ^ 245 us = 82 % of the 299.9 us chain

The tail stages pair elements inside each contiguous block of 8, so their 3
butterfly levels are exactly an 8-point DFT with **compile-time constants**
(W8^i, W4^i, W2^0) -- no runtime twiddle table at all.  fusing them into one
kernel means 512 blocks x 8 elements instead of 512+1024+2048 tiny blocks.

The fused kernel writes exactly what the three separate stages would write, so
the global DIF bit-reversal still applies to the final output (validated
against torch.fft.fft).

Run: python fft_fullchain_radix8.py
"""

import math
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

N = 4096
LOG2N = 12
HEAD = 9                     # stages 1..9 as separate kernels
NB8 = N // 8                 # 512 blocks for the fused radix-8 tail
DT = "float32"


def build_stage(nb, m):
    @tilelang.jit(target="npuir")
    def factory():
        @T.prim_func
        def main(ar: T.Tensor((nb, 2, m), DT), ai: T.Tensor((nb, 2, m), DT),
                 wr: T.Tensor((2, m), DT), wi: T.Tensor((2, m), DT),
                 br: T.Tensor((nb, 2, m), DT), bi: T.Tensor((nb, 2, m), DT)) -> None:
            with T.Kernel(nb, is_npu=True) as (bid, _v):
                Ar = T.alloc_ub((2, m), DT)
                Ai = T.alloc_ub((2, m), DT)
                Br = T.alloc_ub((2, m), DT)
                Bi = T.alloc_ub((2, m), DT)
                Wr = T.alloc_ub((2, m), DT)
                Wi = T.alloc_ub((2, m), DT)
                T.copy(ar[bid, 0, 0], Ar)
                T.copy(ai[bid, 0, 0], Ai)
                T.copy(wr, Wr)
                T.copy(wi, Wi)
                for i in T.Parallel(m):
                    Br[0, i] = Ar[0, i] + Ar[1, i]
                for i in T.Parallel(m):
                    Bi[0, i] = Ai[0, i] + Ai[1, i]
                for i in T.Parallel(m):
                    Br[1, i] = ((Ar[0, i] - Ar[1, i]) * Wr[0, i]
                                - (Ai[0, i] - Ai[1, i]) * Wi[0, i])
                for i in T.Parallel(m):
                    Bi[1, i] = ((Ar[0, i] - Ar[1, i]) * Wi[0, i]
                                + (Ai[0, i] - Ai[1, i]) * Wr[0, i])
                T.copy(Br, br[bid, 0, 0])
                T.copy(Bi, bi[bid, 0, 0])

        return main

    return factory()


def build_tail():
    """8-point DFT = the last three radix-2 DIF stages of an 8-element block,
    written as explicit scalar code (parser only accepts range/T.serial inside
    prim_func, so the stage structure is spelled out). Twiddles come from UB
    tables: W8^i (i<4), W4^i (i<2); the last stage has no twiddle."""

    @tilelang.jit(target="npuir")
    def factory():
        @T.prim_func
        def tail_radix8(ar: T.Tensor((NB8, 8), DT), ai: T.Tensor((NB8, 8), DT),
                        w8r: T.Tensor((4,), DT), w8i: T.Tensor((4,), DT),
                        w4r: T.Tensor((2,), DT), w4i: T.Tensor((2,), DT),
                        br: T.Tensor((NB8, 8), DT), bi: T.Tensor((NB8, 8), DT)) -> None:
            with T.Kernel(NB8, is_npu=True) as (bid, _v):
                zr = T.alloc_ub((8,), DT)
                zi = T.alloc_ub((8,), DT)
                yr = T.alloc_ub((8,), DT)
                yi = T.alloc_ub((8,), DT)
                W8r = T.alloc_ub((4,), DT)
                W8i = T.alloc_ub((4,), DT)
                W4r = T.alloc_ub((2,), DT)
                W4i = T.alloc_ub((2,), DT)
                T.copy(ar[bid, 0], zr)
                T.copy(ai[bid, 0], zi)
                T.copy(w8r, W8r)
                T.copy(w8i, W8i)
                T.copy(w4r, W4r)
                T.copy(w4i, W4i)

                # stage A (bs=8, m=4): z -> y
                for i in T.serial(4):
                    yr[i] = zr[i] + zr[i + 4]
                    yi[i] = zi[i] + zi[i + 4]
                    yr[i + 4] = (zr[i] - zr[i + 4]) * W8r[i] - (zi[i] - zi[i + 4]) * W8i[i]
                    yi[i + 4] = (zr[i] - zr[i + 4]) * W8i[i] + (zi[i] - zi[i + 4]) * W8r[i]

                # stage B (bs=4, m=2, two blocks): y -> z
                for blk in T.serial(2):
                    for i in T.serial(2):
                        zr[blk * 4 + i] = yr[blk * 4 + i] + yr[blk * 4 + i + 2]
                        zi[blk * 4 + i] = yi[blk * 4 + i] + yi[blk * 4 + i + 2]
                        zr[blk * 4 + i + 2] = (
                            (yr[blk * 4 + i] - yr[blk * 4 + i + 2]) * W4r[i]
                            - (yi[blk * 4 + i] - yi[blk * 4 + i + 2]) * W4i[i])
                        zi[blk * 4 + i + 2] = (
                            (yr[blk * 4 + i] - yr[blk * 4 + i + 2]) * W4i[i]
                            + (yi[blk * 4 + i] - yi[blk * 4 + i + 2]) * W4r[i])

                # stage C (bs=2, m=1, four blocks, twiddle = 1): z -> y
                for blk in T.serial(4):
                    yr[blk * 2] = zr[blk * 2] + zr[blk * 2 + 1]
                    yi[blk * 2] = zi[blk * 2] + zi[blk * 2 + 1]
                    yr[blk * 2 + 1] = zr[blk * 2] - zr[blk * 2 + 1]
                    yi[blk * 2 + 1] = zi[blk * 2] - zi[blk * 2 + 1]

                T.copy(yr, br[bid, 0])
                T.copy(yi, bi[bid, 0])

        return tail_radix8

    return factory()


def bit_reverse_perm(n, log2n):
    idx = torch.arange(n)
    rev = torch.zeros(n, dtype=torch.long)
    for bit in range(log2n):
        rev |= ((idx >> bit) & 1) << (log2n - 1 - bit)
    return rev


def main():
    torch.manual_seed(0)
    dev = "npu"
    n_iter = int(os.environ.get("ITERS", "3"))

    x = torch.randn(N, dtype=torch.float32, device=dev)
    ref = torch.fft.fft(x.double().cpu())

    re = torch.zeros((HEAD + 2, N), dtype=torch.float32, device=dev)
    im = torch.zeros((HEAD + 2, N), dtype=torch.float32, device=dev)
    re[0] = x

    head = []
    for s in range(HEAD):
        m = N >> (s + 1)
        nb = 1 << s
        ang = torch.arange(m, dtype=torch.float32) * (-2.0 * math.pi / (2 * m))
        head.append((build_stage(nb, m), nb, m,
                     torch.cos(ang).repeat(2, 1).to(dev),
                     torch.sin(ang).repeat(2, 1).to(dev)))
    tail = build_tail()
    a8 = torch.arange(4, dtype=torch.float32) * (-2.0 * math.pi / 8)
    a4 = torch.arange(2, dtype=torch.float32) * (-2.0 * math.pi / 4)
    W8R, W8I = torch.cos(a8).to(dev), torch.sin(a8).to(dev)
    W4R, W4I = torch.cos(a4).to(dev), torch.sin(a4).to(dev)

    def run_chain():
        for s, (k, nb, m, w_r, w_i) in enumerate(head):
            k(re[s].view(nb, 2, m), im[s].view(nb, 2, m), w_r, w_i,
              re[s + 1].view(nb, 2, m), im[s + 1].view(nb, 2, m))
        tail(re[HEAD].view(NB8, 8), im[HEAD].view(NB8, 8), W8R, W8I, W4R, W4I,
             re[HEAD + 1].view(NB8, 8), im[HEAD + 1].view(NB8, 8))
        torch.npu.synchronize()

    run_chain()

    rev = bit_reverse_perm(N, LOG2N)
    got = torch.complex(re[HEAD + 1].cpu(), im[HEAD + 1].cpu())
    natural = torch.empty_like(got)
    natural[rev] = got
    err = (natural - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"[numeric] max_abs_err={err:.3e} ref_peak={scale:.1f} rel={err/scale:.2e} "
          f"-> {'PASS' if err/scale < 1e-4 else 'FAIL'}  ({HEAD}+1 kernels/chain)")

    for _ in range(n_iter):
        run_chain()
    print(f"[app] {n_iter} chains done")


if __name__ == "__main__":
    main()
