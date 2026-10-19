"""Batched K6 stages: one kernel launch per stage across all heads.

Per chunk:
  bmm: KS, W, QS, QK, O2, Sd
  vec: rhs, P, Vp, o, state
The chunk loop stays on the host (recurrence is sequential), but all H heads
are processed together inside each kernel.
"""

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[2], target="npuir")
def bmm(H: int, M: int, N: int, K: int, ta: bool = False, tb: bool = False,
        dtype: str = "float32"):
    a_shape = (H, K, M) if ta else (H, M, K)
    b_shape = (H, N, K) if tb else (H, K, N)

    @T.prim_func
    def main(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor((H, M, N), "float32"),
    ):
        with T.Kernel(H, is_npu=True) as (h, _), T.Scope("Cube"):
            a_l1 = T.alloc_L1(a_shape[1:], dtype)
            b_l1 = T.alloc_L1(b_shape[1:], dtype)
            c_l0 = T.alloc_L0C([M, N], "float32")
            T.copy(A[h, 0:a_shape[1], 0:a_shape[2]], a_l1)
            T.copy(B[h, 0:b_shape[1], 0:b_shape[2]], b_l1)
            T.gemm(a_l1, b_l1, c_l0, size=[M, K, N],
                   a_transpose=ta, b_transpose=tb, initC=True)
            with T.rs("PIPE_FIX"):
                T.copy(c_l0, C[h, 0:M, 0:N])

    return main


@tilelang.jit(out_idx=[4], target="npuir")
def vec_rhs_b(H: int, C: int, V: int):
    @T.prim_func
    def main(
        KS: T.Tensor((H, C, V), "float32"),
        v: T.Tensor((H, C, V), "float32"),
        gprev: T.Tensor((H, 1, C), "float32"),
        beta: T.Tensor((H, 1, C), "float32"),
        rhs: T.Tensor((H, C, V), "float32"),
    ):
        with T.Kernel(H, is_npu=True) as (h, _):
            ks_ub = T.alloc_ub((C, V), "float32")
            v_ub = T.alloc_ub((C, V), "float32")
            gp_ub = T.alloc_ub((1, C), "float32")
            b_ub = T.alloc_ub((1, C), "float32")
            out_ub = T.alloc_ub((C, V), "float32")
            T.copy(KS[h, 0:C, 0:V], ks_ub)
            T.copy(v[h, 0:C, 0:V], v_ub)
            T.copy(gprev[h, 0:1, 0:C], gp_ub)
            T.copy(beta[h, 0:1, 0:C], b_ub)
            for i, j in T.Parallel(C, V):
                out_ub[i, j] = b_ub[0, i] * (v_ub[i, j] - T.exp(gp_ub[0, i]) * ks_ub[i, j])
            T.copy(out_ub, rhs[h, 0:C, 0:V])

    return main


@tilelang.jit(out_idx=[2], target="npuir")
def vec_p_b(H: int, C: int, scale: float = 1.0):
    @T.prim_func
    def main(
        QK: T.Tensor((H, C, C), "float32"),
        gc: T.Tensor((H, 1, C), "float32"),
        P: T.Tensor((H, C, C), "float32"),
    ):
        with T.Kernel(H, is_npu=True) as (h, _):
            qk_ub = T.alloc_ub((C, C), "float32")
            g_ub = T.alloc_ub((1, C), "float32")
            out_ub = T.alloc_ub((C, C), "float32")
            T.copy(QK[h, 0:C, 0:C], qk_ub)
            T.copy(gc[h, 0:1, 0:C], g_ub)
            for i, j in T.Parallel(C, C):
                out_ub[i, j] = T.if_then_else(
                    i >= j, scale * T.exp(g_ub[0, i] - g_ub[0, j]) * qk_ub[i, j],
                    T.float32(0.0),
                )
            T.copy(out_ub, P[h, 0:C, 0:C])

    return main


@tilelang.jit(out_idx=[2], target="npuir")
def vec_vp_b(H: int, C: int, V: int):
    @T.prim_func
    def main(
        W: T.Tensor((H, C, V), "float32"),
        gc: T.Tensor((H, 1, C), "float32"),
        Vp: T.Tensor((H, C, V), "float32"),
    ):
        with T.Kernel(H, is_npu=True) as (h, _):
            w_ub = T.alloc_ub((C, V), "float32")
            g_ub = T.alloc_ub((1, C), "float32")
            out_ub = T.alloc_ub((C, V), "float32")
            T.copy(W[h, 0:C, 0:V], w_ub)
            T.copy(gc[h, 0:1, 0:C], g_ub)
            for i, j in T.Parallel(C, V):
                out_ub[i, j] = T.exp(g_ub[0, C - 1] - g_ub[0, i]) * w_ub[i, j]
            T.copy(out_ub, Vp[h, 0:C, 0:V])

    return main


@tilelang.jit(out_idx=[3], target="npuir")
def vec_o_b(H: int, C: int, V: int, scale: float = 1.0):
    @T.prim_func
    def main(
        QS: T.Tensor((H, C, V), "float32"),
        O2: T.Tensor((H, C, V), "float32"),
        gc: T.Tensor((H, 1, C), "float32"),
        o: T.Tensor((H, C, V), "float32"),
    ):
        with T.Kernel(H, is_npu=True) as (h, _):
            qs_ub = T.alloc_ub((C, V), "float32")
            o2_ub = T.alloc_ub((C, V), "float32")
            g_ub = T.alloc_ub((1, C), "float32")
            out_ub = T.alloc_ub((C, V), "float32")
            T.copy(QS[h, 0:C, 0:V], qs_ub)
            T.copy(O2[h, 0:C, 0:V], o2_ub)
            T.copy(gc[h, 0:1, 0:C], g_ub)
            for i, j in T.Parallel(C, V):
                out_ub[i, j] = scale * T.exp(g_ub[0, i]) * qs_ub[i, j] + o2_ub[i, j]
            T.copy(out_ub, o[h, 0:C, 0:V])

    return main


@tilelang.jit(out_idx=[2], target="npuir")
def vec_state_b(H: int, K: int, V: int):
    @T.prim_func
    def main(
        S_scale: T.Tensor((H, K, V), "float32"),
        S_delta: T.Tensor((H, K, V), "float32"),
        S_out: T.Tensor((H, K, V), "float32"),
    ):
        with T.Kernel(H, is_npu=True) as (h, _):
            s_ub = T.alloc_ub((K, V), "float32")
            d_ub = T.alloc_ub((K, V), "float32")
            T.copy(S_scale[h, 0:K, 0:V], s_ub)
            T.copy(S_delta[h, 0:K, 0:V], d_ub)
            T.vadd(s_ub, d_ub, s_ub)
            T.copy(s_ub, S_out[h, 0:K, 0:V])

    return main


def k6_forward_batched(q, k, v, g_raw, beta, S_in, scale, C=64, A_all=None):
    """q [S,H,K], k [S,H,K], v [S,H,V], g [S,H], beta [S,H], S_in [H,K,V]."""
    S, H, K = q.shape
    V = v.shape[2]
    o = torch.zeros(S, H, V, dtype=torch.float32, device=q.device)
    S_cur = S_in.contiguous()
    nch = (S + C - 1) // C
    for c in range(nch):
        lo = c * C
        valid = min(C, S - lo)
        qc = torch.zeros(H, C, K, device=q.device); qc[:, :valid] = q[lo:lo + valid].transpose(0, 1)
        kc = torch.zeros(H, C, K, device=q.device); kc[:, :valid] = k[lo:lo + valid].transpose(0, 1)
        vc = torch.zeros(H, C, V, device=q.device); vc[:, :valid] = v[lo:lo + valid].transpose(0, 1)
        bc = torch.zeros(H, C, device=q.device); bc[:, :valid] = beta[lo:lo + valid].transpose(0, 1)
        gc = torch.zeros(H, C, device=q.device)
        if valid > 0:
            gc[:, :valid] = torch.cumsum(g_raw[lo:lo + valid], dim=0).transpose(0, 1)
            gc[:, valid:] = gc[:, valid - 1:valid]
        gprev = torch.zeros(H, C, device=q.device)
        if valid > 1:
            gprev[:, 1:valid] = gc[:, :valid - 1]
        gc3, gp3, b3 = gc.unsqueeze(1), gprev.unsqueeze(1), bc.unsqueeze(1)
        if A_all is None:
            Gp = torch.exp(gp3[:, :, :, None] - gc3[:, :, None, :]) * torch.tril(
                torch.ones(C, C, device=q.device), -1
            )
            A = torch.linalg.solve(
                torch.eye(C, device=q.device) + b3[:, :, :, None] * Gp * (kc @ kc.transpose(1, 2)),
                torch.eye(C, device=q.device).expand(H, C, C),
            ).contiguous()
        else:
            A = A_all[lo:lo + C, :, :].transpose(0, 1).contiguous()
        KS = bmm(H, C, V, K)(kc, S_cur).clone()
        rhs = vec_rhs_b(H, C, V)(KS, vc, gp3, b3).clone()
        W = bmm(H, C, V, C)(A, rhs).clone()
        Vp = vec_vp_b(H, C, V)(W, gc3).clone()
        QS = bmm(H, C, V, K)(qc, S_cur).clone()
        QK = bmm(H, C, C, K, False, True)(qc, kc).clone()
        P = vec_p_b(H, C, scale)(QK, gc3).clone()
        O2 = bmm(H, C, V, C)(P, W).clone()
        o_c = vec_o_b(H, C, V, scale)(QS, O2, gc3).clone()
        Sd = bmm(H, K, V, C)(kc.transpose(1, 2).contiguous(), Vp).clone()
        S_scale = torch.exp(gc[:, -1])[:, None, None] * S_cur
        S_cur = vec_state_b(H, K, V)(S_scale, Sd).clone()
        o[lo:lo + valid] = o_c[:, :valid].transpose(0, 1)
    return o, S_cur
