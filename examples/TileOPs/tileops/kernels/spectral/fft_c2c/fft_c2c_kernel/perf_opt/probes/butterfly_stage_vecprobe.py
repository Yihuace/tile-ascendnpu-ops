"""Route A probe: can ONE radix-2 butterfly stage be vectorized on npuir?

Why this probe exists
---------------------
FFT is 99.9 % scalar today.  The vectorizer (`src/transform/npu_loop_vectorize.cc`)
accepts a statement only if:
  * the innermost `T.Parallel` body is a single `BufferStore`, and
  * EVERY store index is a bare `Var` or an `IntImm`/`FloatImm`  (IsScalar),
  * every load index simplifies to loop-invariant or `v ± invariant` (NO `stride*v`).

The old FFT wrote `D[g*h + i]` / `D[g*h + i + nh]` -> affine store indices -> never
vectorized, and satisfying the rule needed a per-stage relayout which npuir cannot
alias (opt_log §2.1).

KEY IDEA TESTED HERE: fold the stage geometry into the buffer SHAPE instead of the
index expression.  With a `(2, nh)` buffer, a butterfly becomes
    Dr[0, i] = Ar[0, i] + Ar[1, i]        # indices (0, i): IntImm + Var  -> legal
    Dr[1, i] = (Ar[0,i]-Ar[1,i])*Wr[i] - (Ai[0,i]-Ai[1,i])*Wi[i]
so no affine index appears and no aliasing is needed.  One store per `T.Parallel`
body is required, hence four separate loops per stage.

PASS criterion (measured, not asserted here): `aiv_vec_ratio` >> 1e-3 (today's FFT
is 1.2e-4) with the same arithmetic.  Read it from the msprof artifact:

  cd /tmp/opwork && mkdir -p a && chmod 700 a && cd a
  msprof op --kernel-name=main --launch-count=5 --warm-up=2 --output=./out \
      python <this file>
  # then: ArithmeticUtilization.csv -> aiv_vec_ratio / aiv_time(us)

Run: python butterfly_stage_vecprobe.py
"""

import torch
import torch_npu  # noqa: F401
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import tilelang
import tilelang.language as T

N = 4096
NH = N // 2
DT = "float32"


@tilelang.jit(target="npuir")
def build():
    @T.prim_func
    def main(
        ar: T.Tensor((2, NH), DT),
        ai: T.Tensor((2, NH), DT),
        wr: T.Tensor((2, NH), DT),
        wi: T.Tensor((2, NH), DT),
        br: T.Tensor((2, NH), DT),
        bi: T.Tensor((2, NH), DT),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _v):
            Ar = T.alloc_ub((2, NH), DT)
            Ai = T.alloc_ub((2, NH), DT)
            Br = T.alloc_ub((2, NH), DT)
            Bi = T.alloc_ub((2, NH), DT)
            Wr = T.alloc_ub((2, NH), DT)
            Wi = T.alloc_ub((2, NH), DT)
            T.copy(ar, Ar)
            T.copy(ai, Ai)
            T.copy(wr, Wr)
            T.copy(wi, Wi)

            # upper half: a = x0 + x1  (one store per loop body)
            for i in T.Parallel(NH):
                Br[0, i] = Ar[0, i] + Ar[1, i]
            for i in T.Parallel(NH):
                Bi[0, i] = Ai[0, i] + Ai[1, i]
            # lower half: b = (x0 - x1) * W
            for i in T.Parallel(NH):
                Br[1, i] = (Ar[0, i] - Ar[1, i]) * Wr[0, i] - (Ai[0, i] - Ai[1, i]) * Wi[0, i]
            for i in T.Parallel(NH):
                Bi[1, i] = (Ar[0, i] - Ar[1, i]) * Wi[0, i] + (Ai[0, i] - Ai[1, i]) * Wr[0, i]

            T.copy(Br, br)
            T.copy(Bi, bi)

    return main


def main():
    k = build()
    dev = "npu"
    ar = torch.randn(2, NH, dtype=torch.float32, device=dev)
    ai = torch.randn(2, NH, dtype=torch.float32, device=dev)
    ang = torch.arange(NH, dtype=torch.float32, device=dev) * (-3.14159265 / NH)
    wr = torch.cos(ang).unsqueeze(0).repeat(2, 1)
    wi = torch.sin(ang).unsqueeze(0).repeat(2, 1)
    br = torch.empty(2, NH, dtype=torch.float32, device=dev)
    bi = torch.empty(2, NH, dtype=torch.float32, device=dev)

    for _ in range(30):
        k(ar, ai, wr, wi, br, bi)
    torch.npu.synchronize()

    # numeric sanity (fp32 reference on CPU)
    a_r, a_i = ar.cpu(), ai.cpu()
    w_r, w_i = wr.cpu()[0], wi.cpu()[0]
    exp_br = torch.stack([a_r[0] + a_r[1], (a_r[0] - a_r[1]) * w_r - (a_i[0] - a_i[1]) * w_i])
    exp_bi = torch.stack([a_i[0] + a_i[1], (a_r[0] - a_r[1]) * w_i + (a_i[0] - a_i[1]) * w_r])
    g_br, g_bi = br.cpu(), bi.cpu()
    for name, got, exp in (("br[0] add ", g_br[0], exp_br[0]), ("bi[0] add ", g_bi[0], exp_bi[0]),
                           ("br[1] butterfly", g_br[1], exp_br[1]), ("bi[1] butterfly", g_bi[1], exp_bi[1])):
        d = (got - exp).abs().max().item()
        print(f"  {name}: max_diff={d:.3e}")
    d = max((g_br - exp_br).abs().max().item(), (g_bi - exp_bi).abs().max().item())
    print(f"OVERALL max_diff={d:.3e} -> {'PASS' if d < 1e-4 else 'FAIL (diagnose per-half above)'}")


if __name__ == "__main__":
    main()
