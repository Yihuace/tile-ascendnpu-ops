"""K2 split blocksolve prototype: Gram + build_N + 6-level Neumann inverse.

This is the compile-friendly alternative to the fused blocksolve:
  Cube   gram_kernel     G = k k^T
  Vector build_n_kernel  N = -beta_i * exp(gc_i-gc_j) * G * mask
  Cube   seg_cube        tmp = P T ; P2 = P P
  Vector seg_add         T2 = T + tmp
"""

import argparse

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

from .gdn_split_inverse import seg_cube, seg_add


@tilelang.jit(out_idx=[1], target="npuir")
def gram_kernel(tasks: int, C: int = 64, K: int = 64, dtype: str = "float16"):
    @T.prim_func
    def gram_main(
        k_tasks: T.Tensor((tasks, C, K), dtype),
        G: T.Tensor((tasks, C, C), "float32"),
    ):
        with T.Kernel(tasks, is_npu=True) as (t, _), T.Scope("Cube"):
            k_l1 = T.alloc_L1([C, K], dtype)
            g_l0 = T.alloc_L0C([C, C], "float32")
            T.copy(k_tasks[t, 0:C, 0:K], k_l1)
            T.gemm(k_l1, k_l1, g_l0, size=[C, K, C], b_transpose=True, initC=True)
            with T.rs("PIPE_FIX"):
                T.copy(g_l0, G[t, 0:C, 0:C])

    return gram_main


@tilelang.jit(out_idx=[5], target="npuir")
def build_n_kernel(tasks: int, C: int = 64):
    @T.prim_func
    def build_n_main(
        G: T.Tensor((tasks, C, C), "float32"),
        gprev: T.Tensor((tasks, 1, C), "float32"),
        gc: T.Tensor((tasks, 1, C), "float32"),
        beta: T.Tensor((tasks, 1, C), "float32"),
        mask: T.Tensor((tasks, C, C), "float32"),
        N: T.Tensor((tasks, C, C), "float32"),
    ):
        with T.Kernel(tasks, is_npu=True) as (t, _):
            g_ub = T.alloc_ub((C, C), "float32")
            m_ub = T.alloc_ub((C, C), "float32")
            n_ub = T.alloc_ub((C, C), "float32")
            gp_ub = T.alloc_ub((1, C), "float32")
            gc_ub = T.alloc_ub((1, C), "float32")
            b_ub = T.alloc_ub((1, C), "float32")
            T.copy(G[t, 0:C, 0:C], g_ub)
            T.copy(mask[t, 0:C, 0:C], m_ub)
            T.copy(gprev[t, 0:1, 0:C], gp_ub)
            T.copy(gc[t, 0:1, 0:C], gc_ub)
            T.copy(beta[t, 0:1, 0:C], b_ub)
            for i, j in T.Parallel(C, C):
                n_ub[i, j] = -b_ub[0, i] * T.exp(gp_ub[0, i] - gc_ub[0, j]) * g_ub[i, j] * m_ub[i, j]
            T.copy(n_ub, N[t, 0:C, 0:C])

    return build_n_main


def split_blocksolve(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, C: int = 64, dtype="float16"):
    """k [1,S,H,K], g [1,S,H], beta [1,S,H] (already cumsum) -> A [tasks,C,C]."""
    _, S, H, K = k.shape
    tasks = ((S + C - 1) // C) * H
    k_tasks = torch.zeros(tasks, C, K, dtype=k.dtype, device=k.device)
    gc = torch.zeros(tasks, C, dtype=torch.float32, device=k.device)
    gprev = torch.zeros(tasks, C, dtype=torch.float32, device=k.device)
    bt = torch.zeros(tasks, C, dtype=torch.float32, device=k.device)
    mask = torch.zeros(tasks, C, C, dtype=torch.float32, device=k.device)
    for c in range((S + C - 1) // C):
        base = c * C
        valid = min(C, S - base)
        for h in range(H):
            t = c * H + h
            k_tasks[t, 0:valid, :] = k[0, base:base + valid, h, :]
            gc[t, 0:valid] = g[0, base:base + valid, h].float()
            if valid > 1:
                gprev[t, 1:valid] = gc[t, 0:valid - 1]
            bt[t, 0:valid] = beta[0, base:base + valid, h].float()
            mask[t, 0:valid, 0:valid] = torch.tril(
                torch.ones(valid, valid, device=k.device), -1
            )
    G = gram_kernel(tasks, C, K, dtype)(k_tasks)
    N = build_n_kernel(tasks, C)(G, gprev.unsqueeze(1), gc.unsqueeze(1), bt.unsqueeze(1), mask)
    eye = torch.eye(C, dtype=torch.float32, device=k.device)
    Tmat = eye.unsqueeze(0).repeat(tasks, 1, 1).contiguous()
    P = N.contiguous()
    kc = seg_cube(tasks, C, "float32")
    ka = seg_add(tasks, C, "float32")
    for _ in range(6):
        tmp, P2 = kc(Tmat, P)
        Tmat = ka(Tmat, tmp)
        P = P2
    return Tmat


def golden(k, g, beta, C=64):
    """fp64 direct solve, matching golden_blocksolve_A_bthd."""
    _, S, H, K = k.shape
    out = torch.zeros(S, H, C, dtype=torch.float64)
    strict = torch.tril(torch.ones(C, C, dtype=torch.bool), -1)
    for c in range((S + C - 1) // C):
        base = c * C
        valid = min(C, S - base)
        for h in range(H):
            kk = k[0, base:base + valid, h, :].double()
            gg = g[0, base:base + valid, h].double()
            gpp = torch.cat([torch.zeros(1, dtype=torch.float64), gg[:-1]])
            bb = beta[0, base:base + valid, h].double()
            Ssys = bb[:, None] * torch.exp(gpp[:, None] - gg[None, :]) * (kk @ kk.T)
            Ssys = torch.where(strict[:valid, :valid], Ssys, torch.zeros((), dtype=torch.float64))
            A = torch.linalg.solve(torch.eye(valid, dtype=torch.float64) + Ssys,
                                   torch.eye(valid, dtype=torch.float64))
            out[base:base + valid, h, 0:valid] = A
    return out


def run_case(S=64, H=1, K=64, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    dev = "npu"
    k = torch.randn(1, S, H, K, dtype=dtype, device=dev)
    g = -torch.rand(1, S, H, dtype=torch.float32, device=dev) * 0.2
    beta = torch.rand(1, S, H, dtype=torch.float32, device=dev) * 0.1
    # cumsum per chunk
    for c in range((S + 63) // 64):
        lo = c * 64
        hi = min(S, lo + 64)
        g[0, lo:hi] = torch.cumsum(g[0, lo:hi], dim=0)
    A = split_blocksolve(k, g, beta, 64, str(dtype).split(".")[-1])
    torch.npu.synchronize()
    ref = golden(k.cpu(), g.cpu(), beta.cpu(), 64)
    out = torch.zeros_like(ref)
    tasks = ((S + 63) // 64) * H
    for c in range((S + 63) // 64):
        base = c * 64
        valid = min(64, S - base)
        for h in range(H):
            out[base:base + valid, h, 0:valid] = A[c * H + h, 0:valid, 0:valid].cpu().double()
    err = float((out - ref).abs().max())
    print(f"[K2] S={S} H={H} K={K} dtype={str(dtype).split('.')[-1]} max_abs_err={err:.3e}")
    assert err < 5e-3, f"blocksolve split mismatch {err:.3e}"
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--K", type=int, default=64)
    args = ap.parse_args()
    run_case(64, 1, 64, torch.float16, seed=0)
    run_case(100, 1, 64, torch.float16, seed=1)
    run_case(128, 2, 64, torch.float16, seed=2)
    run_wrapper_case(100, 1, 64, torch.float16, seed=7)
    print("[K2] PASS")



def prefill_blocksolve_A_bthd_split(
    k, g, beta, cu_seqlens=None, chunk_size=64,
    use_gate=False, l2norm=False, beta_sigmoid=False, allow_neg_eigval=False,
    uniform_seq_len=0, a_pad=None, return_padded=False,
):
    """Drop-in signature for the in-scope B=1 equal-length path.

    Returns ``(A, k_rnorm)`` with A ``[1,S,H,C]`` and k_rnorm ``[1,1,key_head]``.
    The split path is correctness-first: host loops over the 6 Neumann levels and
    launches Cube/Vector kernels; no fused MixCV kernel.
    """
    S, H, K = k.shape[1], k.shape[2], k.shape[3]
    C = chunk_size
    A = split_blocksolve(k, g, beta, C, str(k.dtype).split(".")[-1])
    s_pad = ((S + C - 1) // C) * C
    if a_pad is None:
        a_pad = torch.zeros((s_pad, H, C), dtype=torch.float32, device=k.device)
    for c in range((S + C - 1) // C):
        base = c * C
        valid = min(C, S - base)
        for h in range(H):
            a_pad[base:base + valid, h, 0:valid] = A[c * H + h, 0:valid, 0:valid].to(k.dtype)
            a_pad[base:base + valid, h, valid:C] = 0
    k_rnorm = torch.zeros((1, 1, k.shape[2]), dtype=torch.float32, device=k.device)
    if return_padded:
        return a_pad, k_rnorm
    return a_pad[:S].view(1, S, H, C), k_rnorm


def run_wrapper_case(S=100, H=1, K=64, dtype=torch.float16, seed=7):
    torch.manual_seed(seed)
    dev = "npu"
    k = torch.randn(1, S, H, K, dtype=dtype, device=dev)
    g = -torch.rand(1, S, H, dtype=torch.float32, device=dev) * 0.2
    beta = torch.rand(1, S, H, dtype=torch.float32, device=dev) * 0.1
    for c in range((S + 63) // 64):
        lo = c * 64
        hi = min(S, lo + 64)
        g[0, lo:hi] = torch.cumsum(g[0, lo:hi], dim=0)
    a, kr = prefill_blocksolve_A_bthd_split(k, g, beta, None, 64)
    torch.npu.synchronize()
    ref = golden(k.cpu(), g.cpu(), beta.cpu(), 64)
    out = a.cpu().double().reshape(S, H, 64)
    err = float((out[:, :, 0:64] - ref).abs().max())
    print(f"[K2-wrapper] S={S} H={H} K={K} dtype={str(dtype).split('.')[-1]} "
          f"A={tuple(a.shape)} k_rnorm={tuple(kr.shape)} max_abs_err={err:.3e}")
    assert err < 5e-3
    return err


if __name__ == "__main__":
    main()
