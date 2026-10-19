"""K6 single-chunk prototype: split Cube GEMMs + host elementwise.

Chunk math (matches the GPU fused_gdr_fwd stages):
    KS = k S
    W  = v - exp(gc) * KS
    Ae = A * G * beta_col
    Vd = Ae W
    Vp = exp(gc_last - gc) * Vd
    QS = q S
    QK = q k^T
    o  = scale*exp(gc)*QS + scale*(G⊙QK) Vd
    S' = exp(gc_last)*S + k^T Vp

Elementwise stays on the host for this prototype; the Cube GEMM shapes and
the formulas are what is being validated.
"""

import argparse

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[2], target="npuir")
def mm(M: int, N: int, K: int, ta: bool = False, tb: bool = False, dtype: str = "float32"):
    a_shape = (K, M) if ta else (M, K)
    b_shape = (N, K) if tb else (K, N)

    @T.prim_func
    def main(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_, __), T.Scope("Cube"):
            a_l1 = T.alloc_L1(a_shape, dtype)
            b_l1 = T.alloc_L1(b_shape, dtype)
            c_l0 = T.alloc_L0C([M, N], "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, c_l0, size=[M, K, N],
                   a_transpose=ta, b_transpose=tb, initC=True)
            with T.rs("PIPE_FIX"):
                T.copy(c_l0, C)

    return main


def chunk_golden(q, k, v, g_raw, beta, S_in, scale):
    """Token-sequential delta rule, independent of the chunk formula."""
    C, K = q.shape
    V = v.shape[1]
    S = S_in.clone().double()
    o = torch.zeros(C, V, dtype=torch.float64)
    for t in range(C):
        dec = torch.exp(g_raw[t].double())
        kt = k[t].double()
        vt = v[t].double()
        corr = vt - kt @ S
        S = dec * S + beta[t].double() * torch.outer(kt, corr)
        o[t] = scale * (q[t].double() @ S)
    return o, S


def k6_chunk(q, k, v, A, gc, beta, S_in, scale):
    C, K = q.shape
    V = v.shape[1]
    A = A.contiguous()
    gprev = torch.cat([torch.zeros(1, device=q.device), gc[:-1]])
    Gout = torch.exp(gc[:, None] - gc[None, :]) * torch.tril(
        torch.ones(C, C, device=q.device), 0
    )
    KS = mm(C, V, K)(k, S_in).clone(); torch.npu.synchronize()
    rhs = beta[:, None] * (v - torch.exp(gprev)[:, None] * KS)
    W = mm(C, V, C)(A, rhs).clone(); torch.npu.synchronize()
    Vp = torch.exp(gc[-1] - gc)[:, None] * W
    QS = mm(C, V, K)(q, S_in).clone(); torch.npu.synchronize()
    QK = mm(C, C, K, False, True)(q, k).clone(); torch.npu.synchronize()
    O1 = scale * torch.exp(gc)[:, None] * QS
    P = scale * (Gout * QK)
    O2 = mm(C, V, C)(P, W).clone(); torch.npu.synchronize()
    o = O1 + O2
    S_scale = torch.exp(gc[-1]) * S_in
    S_delta = mm(K, V, C, True, False)(k, Vp).clone(); torch.npu.synchronize()
    S_out = S_scale + S_delta
    return o, S_out


def run_case(C=64, K=64, V=64, seed=0):
    torch.manual_seed(seed)
    dev = "npu"
    q = torch.randn(C, K, dtype=torch.float32, device=dev)
    k = torch.randn(C, K, dtype=torch.float32, device=dev)
    v = torch.randn(C, V, dtype=torch.float32, device=dev)
    g_raw = -torch.rand(C, dtype=torch.float32, device=dev) * 0.2
    gc = torch.cumsum(g_raw, dim=0)
    beta = torch.rand(C, dtype=torch.float32, device=dev) * 0.2
    S_in = torch.randn(K, V, dtype=torch.float32, device=dev) * 0.1
    scale = K ** -0.5

    # A = (I + beta_i * exp(gc_i-gc_j) * k_i k_j^T)^{-1}, strict lower
    gprev = torch.cat([torch.zeros(1, device=dev), gc[:-1]])
    Gp = torch.exp(gprev[:, None] - gc[None, :]) * torch.tril(
        torch.ones(C, C, device=dev), -1
    )
    Ssys = beta[:, None] * Gp * (k @ k.T)
    A = torch.linalg.solve(
        torch.eye(C, dtype=torch.float32, device=dev) + Ssys,
        torch.eye(C, dtype=torch.float32, device=dev),
    )
    torch.npu.synchronize()
    o, S_out = k6_chunk(q, k, v, A, gc, beta, S_in, scale)
    torch.npu.synchronize()
    ref_o, ref_S = chunk_golden(q.cpu(), k.cpu(), v.cpu(), g_raw.cpu(), beta.cpu(), S_in.cpu(), scale)
    eo = float((o.cpu().double() - ref_o).abs().max())
    es = float((S_out.cpu().double() - ref_S).abs().max())
    print(f"[K6-chunk] C={C} K={K} V={V} o_err={eo:.3e} S_err={es:.3e}")
    assert eo < 5e-3 and es < 5e-3, (eo, es)
    return eo, es


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--V", type=int, default=64)
    args = ap.parse_args()
    run_case(args.C, args.K, args.V, seed=0)
    run_case(args.C, args.K, args.V, seed=1)
    print("[K6-chunk] PASS")


if __name__ == "__main__":
    main()
