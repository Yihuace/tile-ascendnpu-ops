"""Route A prototype: full 12-stage radix-2 DIF FFT using the foldable form.

Key idea (validated at stage level by `butterfly_stage_vecprobe.py`): never write
an affine store index.  Keep every stage in the shape `(nb, 2, m)` so a butterfly
becomes `Y[b,0,i] = X[b,0,i] + X[b,1,i]` / `Y[b,1,i] = (X[b,0,i]-X[b,1,i]) * W[i]`
-- indices are (serial Var, IntImm, parallel Var) or (IntImm, IntImm, parallel Var),
which satisfies `npu_loop_vectorize.cc` (store idx must be a bare Var or an
immediate; loads must be unit-stride in the parallel var).

Layout arithmetic: nb * 2 * m == n for every stage, so all stages share one flat
n-element buffer; only the *view shape* changes per stage (host-side view only,
no TIR aliasing needed -- that is what blocked the old implementation).

Stage s (0-based, DIF):  m = n >> (s+1), nb = 1 << s, bs = 2*m
    Y[p]     = X[p] + X[q]              p = b*bs + i, q = p + m
    Y[q]     = (X[p] - X[q]) * exp(-2j*pi*i/bs)

DIF leaves the output in bit-reversed order -> the reference applies the same
permutation before comparing with torch.fft.fft.

Run: python fft_fullchain_proto.py            # numerics + msprof summary
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
    """One DIF stage over buffers viewed as (nb, 2, m)."""

    @tilelang.jit(target="npuir")
    def factory():
        @T.prim_func
        def main(
            ar: T.Tensor((nb, 2, m), DT),
            ai: T.Tensor((nb, 2, m), DT),
            wr: T.Tensor((1, 1, m), DT),
            wi: T.Tensor((1, 1, m), DT),
            br: T.Tensor((nb, 2, m), DT),
            bi: T.Tensor((nb, 2, m), DT),
        ) -> None:
            with T.Kernel(1, is_npu=True) as (_cid, _v):
                Ar = T.alloc_ub((nb, 2, m), DT)
                Ai = T.alloc_ub((nb, 2, m), DT)
                Br = T.alloc_ub((nb, 2, m), DT)
                Bi = T.alloc_ub((nb, 2, m), DT)
                Wr = T.alloc_ub((1, 1, m), DT)
                Wi = T.alloc_ub((1, 1, m), DT)
                T.copy(ar, Ar)
                T.copy(ai, Ai)
                T.copy(wr, Wr)
                T.copy(wi, Wi)

                for b in T.serial(nb):
                    # upper half: a + c  (one store per loop body)
                    for i in T.Parallel(m):
                        Br[b, 0, i] = Ar[b, 0, i] + Ar[b, 1, i]
                    for i in T.Parallel(m):
                        Bi[b, 0, i] = Ai[b, 0, i] + Ai[b, 1, i]
                    # lower half: (a - c) * W
                    for i in T.Parallel(m):
                        Br[b, 1, i] = ((Ar[b, 0, i] - Ar[b, 1, i]) * Wr[0, 0, i]
                                       - (Ai[b, 0, i] - Ai[b, 1, i]) * Wi[0, 0, i])
                    for i in T.Parallel(m):
                        Bi[b, 1, i] = ((Ar[b, 0, i] - Ar[b, 1, i]) * Wi[0, 0, i]
                                       + (Ai[b, 0, i] - Ai[b, 1, i]) * Wr[0, 0, i])

                T.copy(Br, br)
                T.copy(Bi, bi)

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

    x = torch.randn(N, dtype=torch.float32, device=dev)  # real input
    ref = torch.fft.fft(x.double().cpu())                # natural-order reference

    # flat ping-pong buffers: stage s reads buf[s], writes buf[s+1]
    re = torch.zeros((LOG2N + 1, N), dtype=torch.float32, device=dev)
    im = torch.zeros((LOG2N + 1, N), dtype=torch.float32, device=dev)
    re[0] = x
    im[0] = 0.0

    kernels = []
    for s in range(LOG2N):
        m = N >> (s + 1)
        nb = 1 << s
        bs = 2 * m
        ang = torch.arange(m, dtype=torch.float32) * (-2.0 * math.pi / bs)
        wr = torch.cos(ang).reshape(1, 1, m).to(dev)
        wi = torch.sin(ang).reshape(1, 1, m).to(dev)
        kernels.append((build_stage(nb, m), nb, m, wr, wi))

    def run_chain():
        for s, (k, nb, m, wr, wi) in enumerate(kernels):
            k(re[s].view(nb, 2, m), im[s].view(nb, 2, m), wr, wi,
              re[s + 1].view(nb, 2, m), im[s + 1].view(nb, 2, m))
        torch.npu.synchronize()

    run_chain()  # warm up / compile all

    # ---- numerics: DIF output is bit-reversed ----
    rev = bit_reverse_perm(N, LOG2N)
    got = torch.complex(re[LOG2N].cpu(), im[LOG2N].cpu())
    got_natural = torch.empty_like(got)
    got_natural[rev] = got
    err = (got_natural - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"[numeric] max_abs_err={err:.3e}  (ref peak={scale:.1f})  "
          f"rel={err/scale:.2e} -> {'PASS' if err/scale < 1e-4 else 'FAIL'}")

    for _ in range(n_iter):
        run_chain()
    print(f"[app] ran {n_iter} full chains x {LOG2N} stages "
          f"(msprof: sum Task Duration over {LOG2N} kernels = one chain)")


if __name__ == "__main__":
    main()
