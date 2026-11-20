"""FFT v2: parallelise each stage over its groups (T.Kernel(nb)) instead of
serialising all groups inside one block.

Why: the v1 prototype (fft_fullchain_proto.py) ran every stage in `T.Kernel(1)`
with `T.serial(nb)` over groups.  Its per-stage timings showed the tail stages
dominating -- stage 10 (m=4, nb=512) took 239 us, stage 11 (m=2, nb=1024) took
305 us -- because a serial loop of 512-2048 tiny vector ops is scalar work on
one core.  The vector width (m) is fixed by the FFT's radix-2 structure and
cannot grow, but the GROUP count nb can be spread over the 40 AIV cores.

v2: one block per group.  Same foldable form as v1 (buffer viewed as (2, m)
per block, stores `B[0,i]` / `B[1,i]`, twiddle duplicated to (2, m) so the
operand ranks match), so vectorisation behaviour is unchanged -- only the
parallelism changes.

Numerics are validated against torch.fft.fft with the DIF bit-reversal applied.

Run: python fft_fullchain_par.py
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
DT = "float32"


def build_stage(nb, m):
    """One DIF stage; block b owns the contiguous group of 2*m elements."""

    @tilelang.jit(target="npuir")
    def factory():
        @T.prim_func
        def main(
            ar: T.Tensor((nb, 2, m), DT),
            ai: T.Tensor((nb, 2, m), DT),
            wr: T.Tensor((2, m), DT),
            wi: T.Tensor((2, m), DT),
            br: T.Tensor((nb, 2, m), DT),
            bi: T.Tensor((nb, 2, m), DT),
        ) -> None:
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

    re = torch.zeros((LOG2N + 1, N), dtype=torch.float32, device=dev)
    im = torch.zeros((LOG2N + 1, N), dtype=torch.float32, device=dev)
    re[0] = x

    stages = []
    for s in range(LOG2N):
        m = N >> (s + 1)
        nb = 1 << s
        ang = torch.arange(m, dtype=torch.float32) * (-2.0 * math.pi / (2 * m))
        w_r = torch.cos(ang).repeat(2, 1).to(dev)   # (2, m), rows duplicated
        w_i = torch.sin(ang).repeat(2, 1).to(dev)
        stages.append((build_stage(nb, m), nb, m, w_r, w_i))

    def run_chain():
        for s, (k, nb, m, w_r, w_i) in enumerate(stages):
            k(re[s].view(nb, 2, m), im[s].view(nb, 2, m), w_r, w_i,
              re[s + 1].view(nb, 2, m), im[s + 1].view(nb, 2, m))
        torch.npu.synchronize()

    run_chain()

    rev = bit_reverse_perm(N, LOG2N)
    got = torch.complex(re[LOG2N].cpu(), im[LOG2N].cpu())
    natural = torch.empty_like(got)
    natural[rev] = got
    err = (natural - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"[numeric] max_abs_err={err:.3e}  ref_peak={scale:.1f}  "
          f"rel={err/scale:.2e} -> {'PASS' if err/scale < 1e-4 else 'FAIL'}")

    for _ in range(n_iter):
        run_chain()
    print(f"[app] {n_iter} chains x {LOG2N} stages done")


if __name__ == "__main__":
    main()
