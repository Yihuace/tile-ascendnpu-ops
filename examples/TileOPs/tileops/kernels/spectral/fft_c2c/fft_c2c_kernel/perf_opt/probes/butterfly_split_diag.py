"""A2 diagnostic: at which expression size does the vectorizer break?

The single-stage probe showed:
    br[0] = Ar[0]+Ar[1]        -> diff 0        (2 loads, 1 add)
    br[1] = full butterfly     -> diff ~17      (6 loads, 1 sub+1 sub+2 mul+1 add)

This probe splits that expression into three depth levels, each in its own
`T.Parallel` loop (single store per body, which is what the vectorizer wants):

    o1[i] = Ar[0,i] - Ar[1,i]                      (2 loads, 1 sub)
    o2[i] = (Ar[0,i] - Ar[1,i]) * Wr[i]            (3 loads, 1 sub + 1 mul)
    o3[i] = (Ar[0,i]-Ar[1,i])*Wr[i] - (Ai[0,i]-Ai[1,i])*Wi[i]   (6 loads, full)

Whichever level first shows a non-zero diff marks the failure point: it says
whether the vectorizer miscompiles (a) multi-load subtraction, (b) a product
with a third operand, or (c) only the widest form.

Run: python butterfly_split_diag.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import torch_npu  # noqa: F401
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
        wr: T.Tensor((NH,), DT),
        wi: T.Tensor((NH,), DT),
        o1: T.Tensor((NH,), DT),
        o2: T.Tensor((NH,), DT),
        o3: T.Tensor((NH,), DT),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _v):
            Ar = T.alloc_ub((2, NH), DT)
            Ai = T.alloc_ub((2, NH), DT)
            Wr = T.alloc_ub((NH,), DT)
            Wi = T.alloc_ub((NH,), DT)
            O1 = T.alloc_ub((NH,), DT)
            O2 = T.alloc_ub((NH,), DT)
            O3 = T.alloc_ub((NH,), DT)
            T.copy(ar, Ar)
            T.copy(ai, Ai)
            T.copy(wr, Wr)
            T.copy(wi, Wi)

            for i in T.Parallel(NH):
                O1[i] = Ar[0, i] - Ar[1, i]
            for i in T.Parallel(NH):
                O2[i] = (Ar[0, i] - Ar[1, i]) * Wr[i]
            for i in T.Parallel(NH):
                O3[i] = (Ar[0, i] - Ar[1, i]) * Wr[i] - (Ai[0, i] - Ai[1, i]) * Wi[i]

            T.copy(O1, o1)
            T.copy(O2, o2)
            T.copy(O3, o3)

    return main


def main():
    k = build()
    dev = "npu"
    torch.manual_seed(0)
    ar = torch.randn(2, NH, dtype=torch.float32, device=dev)
    ai = torch.randn(2, NH, dtype=torch.float32, device=dev)
    wr = torch.randn(NH, dtype=torch.float32, device=dev)
    wi = torch.randn(NH, dtype=torch.float32, device=dev)
    o1 = torch.empty(NH, dtype=torch.float32, device=dev)
    o2 = torch.empty(NH, dtype=torch.float32, device=dev)
    o3 = torch.empty(NH, dtype=torch.float32, device=dev)
    k(ar, ai, wr, wi, o1, o2, o3)
    torch.npu.synchronize()

    a_r, a_i, w_r, w_i = ar.cpu(), ai.cpu(), wr.cpu(), wi.cpu()
    exp1 = a_r[0] - a_r[1]
    exp2 = exp1 * w_r
    exp3 = exp2 - (a_i[0] - a_i[1]) * w_i
    for name, got, exp in (("L1 sub(2 loads)      ", o1.cpu(), exp1),
                           ("L2 sub*mul(3 loads)  ", o2.cpu(), exp2),
                           ("L3 full(6 loads)     ", o3.cpu(), exp3)):
        d = (got - exp).abs().max().item()
        print(f"  {name} max_diff={d:.3e}  {'OK' if d < 1e-4 else 'BROKEN'}")

    # how many elements are wrong, and whether they cluster
    bad = (o3.cpu() - exp3).abs() > 1e-4
    if bad.any():
        idx = bad.nonzero().flatten()
        print(f"  L3 wrong elements: {bad.sum().item()}/{NH}, first={idx[:5].tolist()}, "
              f"last={idx[-3:].tolist()}")


if __name__ == "__main__":
    main()
