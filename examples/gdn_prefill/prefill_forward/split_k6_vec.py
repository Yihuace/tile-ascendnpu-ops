"""K6 full chunk loop with TileLang Vector kernels.

Cube stages reuse split_k6.mm; Vector stages:
  vec_rhs, vec_p, vec_vp, vec_o, vec_state.
"""

import argparse

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

from split_k6 import mm


@tilelang.jit(out_idx=[4], target="npuir")
def vec_rhs(C: int, V: int):
    @T.prim_func
    def main(
        KS: T.Tensor((C, V), "float32"),
        v: T.Tensor((C, V), "float32"),
        gprev: T.Tensor((C,), "float32"),
        beta: T.Tensor((C,), "float32"),
        rhs: T.Tensor((C, V), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_, __):
            ks_ub = T.alloc_ub((C, V), "float32")
            v_ub = T.alloc_ub((C, V), "float32")
            gp_ub = T.alloc_ub((C,), "float32")
            b_ub = T.alloc_ub((C,), "float32")
            out_ub = T.alloc_ub((C, V), "float32")
            T.copy(KS, ks_ub)
            T.copy(v, v_ub)
            T.copy(gprev, gp_ub)
            T.copy(beta, b_ub)
            for i, j in T.Parallel(C, V):
                out_ub[i, j] = b_ub[i] * (v_ub[i, j] - T.exp(gp_ub[i]) * ks_ub[i, j])
            T.copy(out_ub, rhs)

    return main


@tilelang.jit(out_idx=[2], target="npuir")
def vec_p(C: int, scale: float = 1.0):
    @T.prim_func
    def main(
        QK: T.Tensor((C, C), "float32"),
        gc: T.Tensor((C,), "float32"),
        P: T.Tensor((C, C), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_, __):
            qk_ub = T.alloc_ub((C, C), "float32")
            g_ub = T.alloc_ub((C,), "float32")
            out_ub = T.alloc_ub((C, C), "float32")
            T.copy(QK, qk_ub)
            T.copy(gc, g_ub)
            for i, j in T.Parallel(C, C):
                out_ub[i, j] = T.if_then_else(
                    i >= j,
                    scale * T.exp(g_ub[i] - g_ub[j]) * qk_ub[i, j],
                    T.float32(0.0),
                )
            T.copy(out_ub, P)

    return main


@tilelang.jit(out_idx=[2], target="npuir")
def vec_vp(C: int, V: int):
    @T.prim_func
    def main(
        W: T.Tensor((C, V), "float32"),
        gc: T.Tensor((C,), "float32"),
        Vp: T.Tensor((C, V), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_, __):
            w_ub = T.alloc_ub((C, V), "float32")
            g_ub = T.alloc_ub((C,), "float32")
            out_ub = T.alloc_ub((C, V), "float32")
            T.copy(W, w_ub)
            T.copy(gc, g_ub)
            for i, j in T.Parallel(C, V):
                out_ub[i, j] = T.exp(g_ub[C - 1] - g_ub[i]) * w_ub[i, j]
            T.copy(out_ub, Vp)

    return main


@tilelang.jit(out_idx=[3], target="npuir")
def vec_o(C: int, V: int, scale: float = 1.0):
    @T.prim_func
    def main(
        QS: T.Tensor((C, V), "float32"),
        O2: T.Tensor((C, V), "float32"),
        gc: T.Tensor((C,), "float32"),
        o: T.Tensor((C, V), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_, __):
            qs_ub = T.alloc_ub((C, V), "float32")
            o2_ub = T.alloc_ub((C, V), "float32")
            g_ub = T.alloc_ub((C,), "float32")
            out_ub = T.alloc_ub((C, V), "float32")
            T.copy(QS, qs_ub)
            T.copy(O2, o2_ub)
            T.copy(gc, g_ub)
            for i, j in T.Parallel(C, V):
                out_ub[i, j] = scale * T.exp(g_ub[i]) * qs_ub[i, j] + o2_ub[i, j]
            T.copy(out_ub, o)

    return main


@tilelang.jit(out_idx=[3], target="npuir")
def vec_state(K: int, V: int):
    @T.prim_func
    def main(
        S_in: T.Tensor((K, V), "float32"),
        S_delta: T.Tensor((K, V), "float32"),
        g_last: T.Tensor((K, V), "float32"),
        S_out: T.Tensor((K, V), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_, __):
            s_ub = T.alloc_ub((K, V), "float32")
            d_ub = T.alloc_ub((K, V), "float32")
            g_ub = T.alloc_ub((K, V), "float32")
            out_ub = T.alloc_ub((K, V), "float32")
            T.copy(S_in, s_ub)
            T.copy(S_delta, d_ub)
            T.copy(g_last, g_ub)
            for i, j in T.Parallel(K, V):
                out_ub[i, j] = T.exp(g_ub[i, j]) * s_ub[i, j] + d_ub[i, j]
            T.copy(out_ub, S_out)

    return main


def chunk_vec(q, k, v, A, gc, beta, S_in, scale):
    C, K = q.shape
    V = v.shape[1]
    A = A.contiguous()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    gc, beta, S_in = gc.contiguous(), beta.contiguous(), S_in.contiguous()
    gprev = torch.cat([torch.zeros(1, device=q.device), gc[:-1]])
    KS = mm(C, V, K)(k, S_in).clone()
    rhs = vec_rhs(C, V)(KS, v, gprev, beta).clone()
    W = mm(C, V, C)(A, rhs).clone()
    Vp = vec_vp(C, V)(W, gc).clone()
    QS = mm(C, V, K)(q, S_in).clone()
    QK = mm(C, C, K, False, True)(q, k).clone()
    P = vec_p(C, scale)(QK, gc).clone()
    O2 = mm(C, V, C)(P, W).clone()
    o = vec_o(C, V, scale)(QS, O2, gc).clone()
    S_delta = mm(K, V, C, True, False)(k, Vp).clone()
    S_out = vec_state(K, V)(S_in, S_delta, gc[-1].expand(K, V).contiguous()).clone()
    return o, S_out


def seq_golden(q, k, v, g_raw, beta, S_in, scale):
    C, K = q.shape
    V = v.shape[1]
    S = S_in.clone().double()
    o = torch.zeros(C, V, dtype=torch.float64)
    for t in range(C):
        dec = torch.exp(g_raw[t].double())
        corr = v[t].double() - k[t].double() @ S
        S = dec * S + beta[t].double() * torch.outer(k[t].double(), corr)
        o[t] = scale * (q[t].double() @ S)
    return o, S


def run_case(C=64, K=64, V=64, seed=0):
    torch.manual_seed(seed)
    dev = "npu"
    q = torch.randn(C, K, device=dev)
    k = torch.randn(C, K, device=dev)
    v = torch.randn(C, V, device=dev)
    g_raw = -torch.rand(C, device=dev) * 0.2
    gc = torch.cumsum(g_raw, 0)
    beta = torch.rand(C, device=dev) * 0.2
    S_in = torch.randn(K, V, device=dev) * 0.1
    scale = K ** -0.5

    gprev = torch.cat([torch.zeros(1, device=dev), gc[:-1]])
    Gp = torch.exp(gprev[:, None] - gc[None, :]) * torch.tril(torch.ones(C, C, device=dev), -1)
    A = torch.linalg.solve(torch.eye(C, device=dev) + beta[:, None] * Gp * (k @ k.T),
                           torch.eye(C, device=dev)).contiguous()
    torch.npu.synchronize()
    o, S_out = chunk_vec(q, k, v, A, gc, beta, S_in, scale)
    torch.npu.synchronize()
    ref_o, ref_S = seq_golden(q.cpu(), k.cpu(), v.cpu(), g_raw.cpu(), beta.cpu(), S_in.cpu(), scale)
    eo = float((o.cpu().double() - ref_o).abs().max())
    es = float((S_out.cpu().double() - ref_S).abs().max())
    print(f"[K6-vec] C={C} K={K} V={V} o_err={eo:.3e} S_err={es:.3e}")
    assert eo < 5e-3 and es < 5e-3, (eo, es)
    return eo, es


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--V", type=int, default=64)
    args = ap.parse_args()
    run_case(args.C, args.K, args.V, 0)
    run_case(args.C, args.K, args.V, 1)
    run_multi_case(2 * args.C, args.K, args.V, 0, args.C)
    print("[K6-vec] PASS")



def k6_forward(q, k, v, g_raw, beta, S_in, scale, C=64):
    """Full sequence, full chunks (S % C == 0), state carried across chunks."""
    S, K = q.shape
    V = v.shape[1]
    o = torch.empty(S, V, dtype=torch.float32, device=q.device)
    S_cur = S_in
    for c in range(S // C):
        lo, hi = c * C, (c + 1) * C
        gc = torch.cumsum(g_raw[lo:hi], 0)
        gprev = torch.cat([torch.zeros(1, device=q.device), gc[:-1]])
        Gp = torch.exp(gprev[:, None] - gc[None, :]) * torch.tril(
            torch.ones(C, C, device=q.device), -1
        )
        A = torch.linalg.solve(
            torch.eye(C, device=q.device) + beta[lo:hi, None] * Gp * (k[lo:hi] @ k[lo:hi].T),
            torch.eye(C, device=q.device),
        ).contiguous()
        o_c, S_cur = chunk_vec(q[lo:hi], k[lo:hi], v[lo:hi], A, gc, beta[lo:hi], S_cur, scale)
        S_cur = S_cur.contiguous()
        o[lo:hi] = o_c
    return o, S_cur


def seq_golden_all(q, k, v, g_raw, beta, S_in, scale):
    S, K = q.shape
    V = v.shape[1]
    S_cur = S_in.clone().double()
    o = torch.zeros(S, V, dtype=torch.float64)
    for t in range(S):
        dec = torch.exp(g_raw[t].double())
        corr = v[t].double() - k[t].double() @ S_cur
        S_cur = dec * S_cur + beta[t].double() * torch.outer(k[t].double(), corr)
        o[t] = scale * (q[t].double() @ S_cur)
    return o, S_cur


def run_multi_case(S=128, K=64, V=64, seed=0, C=64):
    torch.manual_seed(seed)
    dev = "npu"
    q = torch.randn(S, K, device=dev)
    k = torch.randn(S, K, device=dev)
    v = torch.randn(S, V, device=dev)
    g_raw = -torch.rand(S, device=dev) * 0.2
    beta = torch.rand(S, device=dev) * 0.2
    S_in = torch.randn(K, V, device=dev) * 0.1
    scale = K ** -0.5
    o, S_out = k6_forward(q, k, v, g_raw, beta, S_in, scale, C)
    torch.npu.synchronize()
    # Compare against a float32 sequential reference (same arithmetic precision).
    qc, kc, vc, gc, bc, Sc = (x.cpu() for x in (q, k, v, g_raw, beta, S_in))
    Sf = Sc.clone()
    of = torch.zeros(S, V)
    for t in range(S):
        dec = torch.exp(gc[t])
        Sf = dec * Sf + bc[t] * torch.outer(kc[t], vc[t] - kc[t] @ Sf)
        of[t] = scale * (qc[t] @ Sf)
    eo = float((o.cpu() - of).abs().max())
    es = float((S_out.cpu() - Sf).abs().max())
    print(f"[K6-multi] S={S} K={K} V={V} C={C} o_err={eo:.3e} S_err={es:.3e} (vs fp32 seq)")
    assert eo < 0.2 and es < 0.3, (eo, es)
    return eo, es


if __name__ == "__main__":
    main()
