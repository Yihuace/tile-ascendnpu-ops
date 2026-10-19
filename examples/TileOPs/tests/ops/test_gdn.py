"""Correctness tests for GatedDeltaNetPrefillFwdOp (NPU).

Delivered op: ``GatedDeltaNetPrefillFwdOp`` (baseline ``GDNFwdOp``, kernel
``GDNDensePrefillFwdKernel``).  The independent golden below is the pure-torch
**sequential** delta-rule recurrence (one token at a time), which is
mathematically equivalent to the chunked GPU pipeline and shares no code with
the NPU implementation.

CORRECTNESS-FIRST DELIVERY: the op is exposed and validated against the independent
sequential golden.  The six-kernel TileLang pipeline is still being ported; the
current kernel path is a reference-speed NPU implementation (see
``kernels/linear_attention/gdn_prefill/gdn_prefill.py``).
"""

import pytest
import torch



def gdn_prefill_golden(
    q: torch.Tensor,      # [B, S, H,  K]   (fp32 for the reference)
    k: torch.Tensor,      # [B, S, H,  K]
    v: torch.Tensor,      # [B, S, HV, V]
    g: torch.Tensor,      # [B, S, HV]      log-space decay
    beta: torch.Tensor,   # [B, S, HV]
    state: torch.Tensor,  # [B, HV, K, V]   fp32
    scale: float,
):
    """Sequential delta-rule reference.  Returns (o, final_state).

    Per token t:
        S_t = diag(exp(g_t)) @ S_{t-1} + beta_t * k_t (x) (v_t - k_t^T S_{t-1})
        o_t = scale * q_t^T S_t
    """
    B, S, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    group = HV // H

    o = torch.zeros(B, S, HV, V, dtype=torch.float32)
    st = state.clone().float()

    for b in range(B):
        for t in range(S):
            for hv in range(HV):
                h = hv // group
                dec = torch.exp(g[b, t, hv].float())                 # [K]
                kt = k[b, t, h].float()                              # [K]
                vt = v[b, t, hv].float()                             # [V]
                qt = q[b, t, h].float()                              # [K]
                prev = st[b, hv]                                     # [K, V]
                corr = vt - kt @ prev                                # [V]
                st[b, hv] = dec * prev + beta[b, t, hv].float() * torch.outer(kt, corr)
                o[b, t, hv] = scale * (qt @ st[b, hv])
    return o, st


CASES = [
    # label, B, S, H, HV, K, V, has_state
    ("L0-single-chunk", 1, 8, 1, 1, 8, 8, False),
    ("L0b-with-state", 1, 8, 1, 1, 8, 8, True),
    ("L1-multi-chunk", 1, 64, 1, 1, 16, 16, True),
    # Split main path (K=V in {64,128}); 100 rows exercises the tail chunk.
    ("L1b-in-scope", 1, 64, 1, 1, 64, 64, True),
    ("L1c-tail", 1, 100, 1, 1, 64, 64, True),
]


@pytest.mark.parametrize("label,B,S,H,HV,K,V,has_state", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gdn_prefill(label, B, S, H, HV, K, V, has_state, dtype):
    torch.manual_seed(0)
    scale = K ** -0.5
    q = torch.randn(B, S, H, K, dtype=dtype)
    k = torch.randn(B, S, H, K, dtype=dtype)
    v = torch.randn(B, S, HV, V, dtype=dtype)
    g = -torch.rand(B, S, HV) * 0.5
    beta = torch.rand(B, S, HV) * 0.1
    state = (torch.randn(B, HV, K, V) * 0.1) if has_state else torch.zeros(B, HV, K, V)

    ref_o, ref_state = gdn_prefill_golden(q, k, v, g, beta, state, scale)

    from tileops.ops.linear_attention.gdn_prefill import GatedDeltaNetPrefillFwdOp

    op = GatedDeltaNetPrefillFwdOp(B=B, S=S, H=H, HV=HV, K=K, V=V, dtype=dtype, scale=scale)
    out = op(q.npu(), k.npu(), v.npu(), g.npu(), beta.npu(), state.npu() if has_state else None)
    o, st = out[0].cpu().float(), out[1].cpu().float()

    tol = 5e-3 if dtype is torch.float16 else 1.6e-2
    assert torch.allclose(o, ref_o, atol=tol, rtol=0), f"o max_err={float((o - ref_o).abs().max()):.3e}"
    assert torch.allclose(st, ref_state, atol=5e-3, rtol=0), f"state max_err={float((st - ref_state).abs().max()):.3e}"
