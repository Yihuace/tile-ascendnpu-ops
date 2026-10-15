# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Feasibility probe: Gated DeltaNet (GDN) chunked PREFILL, ``target="npuir"``.

Stage-0 pre-migration FEASIBILITY PROBE for migrating ``GatedDeltaNetPrefillFwdOp``
(baseline op key ``GDNFwdOp``; ref_api ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``)
to npuir (Ascend910B3, CANN 8.5.2).

This is NOT the migrated operator. It is a minimal but *representative* kernel that
exercises the hard capabilities the prefill migration depends on:

  (a) intra-chunk delta-rule        ``W = (I + S_sys)^{-1} rhs``   (Cube T.gemm)
  (b) gate/beta transform           ``exp(gc)``, ``exp(gc_l-gc_s)``, causal masks,
                                    ``beta`` row scaling              (Vector)
  (c) cross-chunk recurrence carry  ``S <- exp(g_last) S + (K*grev)^T W``
                                                                     (Cube T.gemm + Vector)
  (d) output combine                ``o = scale*exp(gc)*(Q@S) + (scale*G*(QK^T))@W``
                                                                     (Cube T.gemm + Vector)

Math (chunk-local, chunk size C; state S is [D, DV] fp32):
    gc[l]      = sum_{s<=l} g[s]                        (chunk-local cumulative log-gate)
    G[l,s]     = exp(gc[l]-gc[s]) if l>=s else 0
    S_sys[l,s] = beta[l]*G[l,s]*(k_l . k_s) if l>s else 0     (strictly lower)
    rhs[l]     = beta[l]*(v_l - exp(gc[l])*(k_l @ S))
    W          = (I + S_sys)^{-1} rhs
    o[l]       = scale*exp(gc[l])*(q_l @ S) + sum_s scale*G[l,s]*(q_l.k_s)*W[s]
    S_new      = exp(gc[C-1])*S + sum_l exp(gc[C-1]-gc[l]) * k_l (x) W[l]

The strictly-lower inverse is evaluated by the *exact finite Neumann series*
``(I+S_sys)^{-1} = sum_j (-S_sys)^j`` (nilpotent: ``S_sys^C = 0``), which uses only
``T.gemm`` and is a stand-in for the GPU's blocked forward-substitution
(``prefill_prepare.py``); both rely solely on Cube matmul.

The golden is an INDEPENDENT pure-torch CPU implementation of the *sequential*
delta-rule recurrence (token by token), NOT the chunked algorithm above.

Run: python gdn_prefill_probe.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import traceback

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

__all__ = ["gdn_prefill_probe", "golden_gdn_sequential"]

_IN = "float16"
_ACC = "float32"
_TOL = 5e-2  # fp16 operands + exp/decay + fp32 accumulate; see PROBE.md


# ---------------------------------------------------------------------------
# Independent pure-torch CPU golden: the SEQUENTIAL delta-rule recurrence.
# ---------------------------------------------------------------------------
def golden_gdn_sequential(q, k, v, g, beta, state0, scale):
    """Token-by-token gated delta rule (fp32 opmath on CPU).

    q,k: ``(B,H,T,D)``; v: ``(B,H,T,DV)``; g,beta: ``(B,H,T)``;
    state0: ``(B,H,D,DV)``. Returns ``(o: (B,H,T,DV), final_state: (B,H,D,DV))``.
    """
    B, H, Tn, D = q.shape
    DV = v.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    gf, bf, S = g.float(), beta.float(), state0.float().clone()
    o = torch.zeros(B, H, Tn, DV)
    for t in range(Tn):
        S = S * torch.exp(gf[:, :, t])[:, :, None, None]
        pred = torch.einsum("bhd,bhdv->bhv", kf[:, :, t], S)
        err = bf[:, :, t][:, :, None] * (vf[:, :, t] - pred)
        S = S + kf[:, :, t][:, :, :, None] * err[:, :, None, :]
        o[:, :, t] = scale * torch.einsum("bhd,bhdv->bhv", qf[:, :, t], S)
    return o, S


# ---------------------------------------------------------------------------
# Probe kernel
# ---------------------------------------------------------------------------
def gdn_prefill_probe(B, H, Tn, D, DV, chunk=16, scale=None):
    """Chunked GDN prefill kernel factory (one NPU core per (b,h)).

    Returns a JIT callable ``main(q,k,v,g,beta,state0) -> (o, final_state)``
    with ``out_idx=[6, 7]``.
    """
    assert Tn % chunk == 0, "T must be a multiple of chunk"
    nch = Tn // chunk
    C = chunk
    if scale is None:
        scale = D ** -0.5
    scale = float(scale)
    pairs = (C + 1) // 2  # Neumann pairs: 2*pairs >= C terms (trailing term is exact 0)

    @tilelang.jit(out_idx=[6, 7], target="npuir")
    def _func():
        @T.prim_func
        def main(
            q: T.Tensor((B, H, Tn, D), _IN),
            k: T.Tensor((B, H, Tn, D), _IN),
            v: T.Tensor((B, H, Tn, DV), _IN),
            g: T.Tensor((B, H, Tn), _IN),
            beta: T.Tensor((B, H, Tn), _IN),
            state0: T.Tensor((B, H, D, DV), _ACC),
            o: T.Tensor((B, H, Tn, DV), _ACC),            # out_idx=6
            final_state: T.Tensor((B, H, D, DV), _ACC),   # out_idx=7
        ):
            with T.Kernel(B * H, is_npu=True) as (cid, _):
                bid = cid // H
                hid = cid % H

                # -- Cube operands (fp16) --
                k16 = T.alloc_shared((C, D), _IN)
                q16 = T.alloc_shared((C, D), _IN)
                v16 = T.alloc_shared((C, DV), _IN)
                s16 = T.alloc_shared((D, DV), _IN)
                sn16 = T.alloc_shared((C, C), _IN)
                a16 = T.alloc_shared((C, DV), _IN)
                b16 = T.alloc_shared((C, DV), _IN)
                p16 = T.alloc_shared((C, C), _IN)
                kg16 = T.alloc_shared((C, D), _IN)

                # -- fp32 accumulators / vector work --
                S = T.alloc_shared((D, DV), _ACC)
                KK = T.alloc_shared((C, C), _ACC)
                QK = T.alloc_shared((C, C), _ACC)
                Emat = T.alloc_shared((C, C), _ACC)
                Gm = T.alloc_shared((C, C), _ACC)
                sysm = T.alloc_shared((C, C), _ACC)
                Sn = T.alloc_shared((C, C), _ACC)
                betaM = T.alloc_shared((C, C), _ACC)
                bmV = T.alloc_shared((C, DV), _ACC)
                gcL = T.alloc_shared((C, C), _ACC)
                gcS = T.alloc_shared((C, C), _ACC)
                dmat = T.alloc_shared((C, C), _ACC)
                idxL = T.alloc_shared((C, C), _ACC)
                idxS = T.alloc_shared((C, C), _ACC)
                g1 = T.alloc_shared((C, 1), _ACC)
                beta1 = T.alloc_shared((C, 1), _ACC)
                gc = T.alloc_shared((C, 1), _ACC)
                gcRow = T.alloc_shared((1, C), _ACC)
                gcCol = T.alloc_shared((1, C), _ACC)
                gexp = T.alloc_shared((C, 1), _ACC)
                grev = T.alloc_shared((C, 1), _ACC)
                gl = T.alloc_shared((C, 1), _ACC)
                gdb = T.alloc_shared((D, DV), _ACC)
                gexpM = T.alloc_shared((C, DV), _ACC)
                grevM = T.alloc_shared((C, D), _ACC)
                KS = T.alloc_shared((C, DV), _ACC)
                QS = T.alloc_shared((C, DV), _ACC)
                vf = T.alloc_shared((C, DV), _ACC)
                rhs = T.alloc_shared((C, DV), _ACC)
                W = T.alloc_shared((C, DV), _ACC)
                term = T.alloc_shared((C, DV), _ACC)
                tmp = T.alloc_shared((C, DV), _ACC)
                kf = T.alloc_shared((C, D), _ACC)

                # -- init state --
                T.copy(state0[bid, hid, 0:D, 0:DV], S)

                # causal index helpers (built once)
                T.arange(idxL, [1, 0], 0)              # l
                T.arange(idxS, [0, 1], 0)              # s
                T.vsub(idxL, idxS, dmat)               # d = l - s

                for c in T.serial(nch):
                    lo = c * C
                    T.copy(q[bid, hid, lo:lo + C, 0:D], q16)
                    T.copy(k[bid, hid, lo:lo + C, 0:D], k16)
                    T.copy(v[bid, hid, lo:lo + C, 0:DV], v16)
                    T.copy(g[bid, hid, lo:lo + C], g1)
                    T.copy(g[bid, hid, lo:lo + C], gcRow)
                    T.copy(beta[bid, hid, lo:lo + C], beta1)

                    # gates: gc = chunk-local cumsum
                    T.cumsum(g1, gc, dim=0)
                    T.cumsum(gcRow, gcCol, dim=1)
                    T.vexp(gc, gexp)                          # exp(gc_l)
                    T.vbrc(gc, gcL)                           # gc_l over rows
                    T.vbrc(gcCol, gcS)                        # gc_s over cols
                    T.fill(gl, gc[C - 1, 0])
                    T.vsub(gl, gc, grev)
                    T.vexp(grev, grev)                        # exp(g_last - gc_l)
                    T.fill(gdb, gc[C - 1, 0])
                    T.vexp(gdb, gdb)                          # exp(g_last)

                    # G[l,s] = exp(gc_l - gc_s); causal masks from d
                    T.vsub(gcL, gcS, Emat)
                    T.vexp(Emat, Emat)
                    T.vmin(dmat, 1.0, sysm)                   # min(d,1)
                    T.vmax(sysm, 0.0, sysm)                   # strict indicator
                    T.vmul(Emat, sysm, sysm)                  # G strict (no diag)
                    T.vadd(dmat, 1.0, dmat)                   # d+1
                    T.vmin(dmat, 1.0, dmat)
                    T.vmax(dmat, 0.0, dmat)                   # inclusive indicator
                    T.vmul(Emat, dmat, Gm)                    # G inclusive

                    # beta broadcasts
                    T.vbrc(beta1, betaM)
                    T.vbrc(beta1, bmV)

                    # ---- Cube matmuls ----
                    T.gemm(k16, k16, KK, size=[C, D, C], b_transpose=True, initC=True)
                    T.vmul(betaM, sysm, sysm)                 # beta_l * G strict
                    T.vmul(sysm, KK, sysm)                    # S_sys = beta_l*G*k_l.k_s
                    T.vmul(sysm, -1.0, Sn)                    # Sn = -S_sys
                    T.vcast(S, s16, round_mode="rint")
                    T.gemm(k16, s16, KS, size=[C, D, DV], initC=True)
                    T.gemm(q16, s16, QS, size=[C, D, DV], initC=True)
                    T.gemm(q16, k16, QK, size=[C, D, C], b_transpose=True, initC=True)

                    # rhs = beta_l * (v - exp(gc_l)*KS)
                    T.vcast(v16, vf, round_mode="rint")
                    T.vbrc(gexp, gexpM)
                    T.vmul(KS, gexpM, KS)
                    T.vsub(vf, KS, rhs)
                    T.vmul(rhs, bmV, rhs)

                    # ---- Neumann solve W = (I+S_sys)^{-1} rhs ----
                    T.copy(rhs, W)
                    T.copy(rhs, term)
                    T.vcast(Sn, sn16, round_mode="rint")
                    for _j in T.serial(pairs):
                        T.vcast(term, a16, round_mode="rint")
                        T.gemm(sn16, a16, tmp, size=[C, C, DV], initC=True)
                        T.vadd(W, tmp, W)
                        T.vcast(tmp, b16, round_mode="rint")
                        T.gemm(sn16, b16, term, size=[C, C, DV], initC=True)
                        T.vadd(W, term, W)

                    # ---- output: o = scale*exp(gc)*(Q@S) + (scale*G*(QK^T))@W ----
                    T.vbrc(gexp, gexpM)
                    T.vmul(QS, gexpM, QS)
                    T.vmul(QS, scale, QS)
                    T.vmul(QK, scale, QK)
                    T.vmul(QK, Gm, QK)
                    T.vcast(QK, p16, round_mode="rint")
                    T.vcast(W, a16, round_mode="rint")
                    T.gemm(p16, a16, QS, size=[C, C, DV], initC=False)
                    T.copy(QS, o[bid, hid, lo:lo + C, 0:DV])

                    # ---- state carry: S = exp(g_last)*S + (K*grev)^T @ W ----
                    T.vbrc(grev, grevM)                       # exp(g_last-gc_l) over cols
                    T.vcast(k16, kf, round_mode="rint")
                    T.vmul(kf, grevM, kf)
                    T.vcast(kf, kg16, round_mode="rint")
                    T.vmul(S, gdb, S)
                    T.vcast(W, b16, round_mode="rint")
                    T.gemm(kg16, b16, S, size=[D, C, DV], a_transpose=True, initC=False)

                T.copy(S, final_state[bid, hid, 0:D, 0:DV])

        return main

    return _func


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_case(B, H, Tn, D, DV, chunk, seed=0):
    torch.manual_seed(seed)
    scale = D ** -0.5
    q = (torch.randn(B, H, Tn, D) * 0.5).to(torch.float16)
    k = (torch.randn(B, H, Tn, D) * 0.5).to(torch.float16)
    v = (torch.randn(B, H, Tn, DV) * 0.5).to(torch.float16)
    g = (-torch.rand(B, H, Tn) * 0.3).to(torch.float16)     # log decay <= 0
    beta = torch.rand(B, H, Tn).to(torch.float16)           # step size in (0,1)
    state0 = torch.randn(B, H, D, DV, dtype=torch.float32) * 0.1

    kernel = gdn_prefill_probe(B, H, Tn, D, DV, chunk, scale)()
    o, fs = kernel(q.npu(), k.npu(), v.npu(), g.npu(), beta.npu(), state0.npu())
    o, fs = o.cpu(), fs.cpu()

    ref_o, ref_fs = golden_gdn_sequential(q, k, v, g, beta, state0, scale)
    err_o = (o.float() - ref_o).abs().max().item()
    err_fs = (fs.float() - ref_fs).abs().max().item()
    err = max(err_o, err_fs)
    print(f"[PROBE] (B={B},H={H},T={Tn},D={D},DV={DV},C={chunk}) "
          f"max_abs_err o={err_o:.3e} final_state={err_fs:.3e} (tol={_TOL:.1e})")
    return err <= _TOL, err


CASES = [
    (1, 2, 48, 16, 16, 16),
    (2, 1, 48, 16, 16, 16),
    (1, 1, 64, 32, 32, 16),
]


def main():
    print("device:", torch.npu.get_device_name(0))
    all_ok, worst, fails = True, 0.0, []
    for B, H, Tn, D, DV, C in CASES:
        print(f"\n===== case (B={B},H={H},T={Tn},D={D},DV={DV},chunk={C}) =====",
              flush=True)
        try:
            ok, err = _run_case(B, H, Tn, D, DV, C)
            worst = max(worst, err)
            all_ok = all_ok and ok
            if not ok:
                fails.append((B, H, Tn, D, DV, C, err))
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            fails.append((B, H, Tn, D, DV, C, repr(exc)))
            print(f"[PROBE] EXCEPTION: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\nworst max_abs_err across cases: {worst:.3e}")
    print("PROBE PASS" if all_ok else f"PROBE FAIL: {fails}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
