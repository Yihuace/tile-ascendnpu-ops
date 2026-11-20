"""FFT v5: same group-parallel form as v3, but the last SIX radix-2 stages are
fused into two kernels by a generic "3-stage fusion over bs0-sized blocks"
template.

v3 timings (10 kernels / chain, 59.3 us):
    stage 1..6 : 1.6 - 4.1 us   (nb = 1..32)
    stage 7..9 : 5.6 / 10.4 / 23.0 us   (nb = 64..256, m = 32..8)  <-- 39 us
    fused 10-12: 2.2 us

The last three stages operating inside a bs0-sized block are just a radix-8 DFT
with compile-time twiddles, for any bs0.  So the same template fuses:
    bs0 = 64  -> stages 7,8,9   (64 blocks x 64 elements)
    bs0 = 8   -> stages 10,11,12 (512 blocks x 8 elements)
leaving stages 1..6 as individual kernels: 6 + 1 + 1 = 8 kernels per chain.

Run: python fft_fullchain_v4.py
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
HEAD = 3                    # stages 1..3 as individual kernels
# stage 1-3 have nb = 1/2/4 blocks; fusing them would put all the scalar work
# on one core, so they stay separate.  Stages 4-6 use fused(512), 7-9 fused(64),
# 10-12 fused(8) -> 6 kernels per chain instead of v4's 8.
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


def build_fused(bs0):
    """Fuse the three radix-2 DIF stages inside a bs0-element block.

    Stage list: (bs0, m0=bs0/2), (bs0/2, m1=bs0/4), (bs0/4, m2=bs0/8).
    Twiddles are passed in as flat tables (they only depend on the position
    inside the block, never on the block index).
    """
    m0, m1, m2 = bs0 // 2, bs0 // 4, bs0 // 8
    nb = N // bs0

    @tilelang.jit(target="npuir")
    def factory():
        @T.prim_func
        def fused(ar: T.Tensor((nb, bs0), DT), ai: T.Tensor((nb, bs0), DT),
                  w0r: T.Tensor((m0,), DT), w0i: T.Tensor((m0,), DT),
                  w1r: T.Tensor((m1,), DT), w1i: T.Tensor((m1,), DT),
                  w2r: T.Tensor((m2,), DT), w2i: T.Tensor((m2,), DT),
                  br: T.Tensor((nb, bs0), DT), bi: T.Tensor((nb, bs0), DT)) -> None:
            with T.Kernel(nb, is_npu=True) as (bid, _v):
                zr = T.alloc_ub((bs0,), DT)
                zi = T.alloc_ub((bs0,), DT)
                yr = T.alloc_ub((bs0,), DT)
                yi = T.alloc_ub((bs0,), DT)
                W0r = T.alloc_ub((m0,), DT)
                W0i = T.alloc_ub((m0,), DT)
                W1r = T.alloc_ub((m1,), DT)
                W1i = T.alloc_ub((m1,), DT)
                W2r = T.alloc_ub((m2,), DT)
                W2i = T.alloc_ub((m2,), DT)
                T.copy(ar[bid, 0], zr)
                T.copy(ai[bid, 0], zi)
                T.copy(w0r, W0r)
                T.copy(w0i, W0i)
                T.copy(w1r, W1r)
                T.copy(w1i, W1i)
                T.copy(w2r, W2r)
                T.copy(w2i, W2i)

                # stage 1 (bs = bs0): z -> y
                for i in T.serial(m0):
                    yr[i] = zr[i] + zr[i + m0]
                    yi[i] = zi[i] + zi[i + m0]
                    yr[i + m0] = (zr[i] - zr[i + m0]) * W0r[i] - (zi[i] - zi[i + m0]) * W0i[i]
                    yi[i + m0] = (zr[i] - zr[i + m0]) * W0i[i] + (zi[i] - zi[i + m0]) * W0r[i]

                # stage 2 (bs = bs0/2): y -> z
                for blk in T.serial(2):
                    for i in T.serial(m1):
                        zr[blk * (2 * m1) + i] = yr[blk * (2 * m1) + i] + yr[blk * (2 * m1) + i + m1]
                        zi[blk * (2 * m1) + i] = yi[blk * (2 * m1) + i] + yi[blk * (2 * m1) + i + m1]
                        zr[blk * (2 * m1) + i + m1] = (
                            (yr[blk * (2 * m1) + i] - yr[blk * (2 * m1) + i + m1]) * W1r[i]
                            - (yi[blk * (2 * m1) + i] - yi[blk * (2 * m1) + i + m1]) * W1i[i])
                        zi[blk * (2 * m1) + i + m1] = (
                            (yr[blk * (2 * m1) + i] - yr[blk * (2 * m1) + i + m1]) * W1i[i]
                            + (yi[blk * (2 * m1) + i] - yi[blk * (2 * m1) + i + m1]) * W1r[i])

                # stage 3 (bs = bs0/4): z -> y
                for blk in T.serial(4):
                    for i in T.serial(m2):
                        yr[blk * (2 * m2) + i] = zr[blk * (2 * m2) + i] + zr[blk * (2 * m2) + i + m2]
                        yi[blk * (2 * m2) + i] = zi[blk * (2 * m2) + i] + zi[blk * (2 * m2) + i + m2]
                        yr[blk * (2 * m2) + i + m2] = (
                            (zr[blk * (2 * m2) + i] - zr[blk * (2 * m2) + i + m2]) * W2r[i]
                            - (zi[blk * (2 * m2) + i] - zi[blk * (2 * m2) + i + m2]) * W2i[i])
                        yi[blk * (2 * m2) + i + m2] = (
                            (zr[blk * (2 * m2) + i] - zr[blk * (2 * m2) + i + m2]) * W2i[i]
                            + (zi[blk * (2 * m2) + i] - zi[blk * (2 * m2) + i + m2]) * W2r[i])

                T.copy(yr, br[bid, 0])
                T.copy(yi, bi[bid, 0])

        return fused

    return factory(), nb, (m0, m1, m2)


def bit_reverse_perm(n, log2n):
    idx = torch.arange(n)
    rev = torch.zeros(n, dtype=torch.long)
    for bit in range(log2n):
        rev |= ((idx >> bit) & 1) << (log2n - 1 - bit)
    return rev


def tables(m, bs, dev):
    ang = torch.arange(m, dtype=torch.float32) * (-2.0 * math.pi / bs)
    return torch.cos(ang).to(dev), torch.sin(ang).to(dev)


def main():
    torch.manual_seed(0)
    dev = "npu"
    n_iter = int(os.environ.get("ITERS", "3"))

    x = torch.randn(N, dtype=torch.float32, device=dev)
    ref = torch.fft.fft(x.double().cpu())

    re = torch.zeros((HEAD + 4, N), dtype=torch.float32, device=dev)
    im = torch.zeros((HEAD + 4, N), dtype=torch.float32, device=dev)
    re[0] = x

    head = []
    for s in range(HEAD):                       # stages 1..6
        m = N >> (s + 1)
        nb = 1 << s
        wr, wi = tables(m, 2 * m, dev)
        head.append((build_stage(nb, m), nb, m, wr.repeat(2, 1), wi.repeat(2, 1)))

    fuse512, nb512, (h0, h1, h2) = build_fused(512)  # stages 4,5,6
    T512 = (tables(h0, 512, dev) + tables(h1, 256, dev) + tables(h2, 128, dev))
    fuse64, nb64, (a0, a1, a2) = build_fused(64)     # stages 7,8,9
    T64 = (tables(a0, 64, dev) + tables(a1, 32, dev) + tables(a2, 16, dev))
    fuse8, nb8, (b0, b1, b2) = build_fused(8)        # stages 10,11,12
    T8 = (tables(b0, 8, dev) + tables(b1, 4, dev) + tables(b2, 2, dev))

    def run_chain():
        for s, (k, nb, m, wr, wi) in enumerate(head):
            k(re[s].view(nb, 2, m), im[s].view(nb, 2, m), wr, wi,
              re[s + 1].view(nb, 2, m), im[s + 1].view(nb, 2, m))
        fuse512(re[HEAD].view(nb512, 512), im[HEAD].view(nb512, 512), *T512,
                re[HEAD + 1].view(nb512, 512), im[HEAD + 1].view(nb512, 512))
        fuse64(re[HEAD + 1].view(nb64, 64), im[HEAD + 1].view(nb64, 64), *T64,
               re[HEAD + 2].view(nb64, 64), im[HEAD + 2].view(nb64, 64))
        fuse8(re[HEAD + 2].view(nb8, 8), im[HEAD + 2].view(nb8, 8), *T8,
              re[HEAD + 3].view(nb8, 8), im[HEAD + 3].view(nb8, 8))
        torch.npu.synchronize()

    run_chain()

    rev = bit_reverse_perm(N, LOG2N)
    got = torch.complex(re[HEAD + 3].cpu(), im[HEAD + 3].cpu())
    natural = torch.empty_like(got)
    natural[rev] = got
    err = (natural - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"[numeric] max_abs_err={err:.3e} ref_peak={scale:.1f} rel={err/scale:.2e} "
          f"-> {'PASS' if err/scale < 1e-4 else 'FAIL'}  ({HEAD + 3} kernels/chain)")

    for _ in range(n_iter):
        run_chain()
    print("[app] done")


if __name__ == "__main__":
    main()
