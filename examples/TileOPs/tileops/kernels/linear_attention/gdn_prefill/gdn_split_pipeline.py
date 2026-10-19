"""Correctness-first GDN split pipeline: K1 cumsum + K2 split blocksolve + K6.

In-scope: B=1, equal-length, H=HV, K=V in {64,128}, chunk=64.
The CP partition path (K4/K5) is intentionally disabled; K6 walks chunks
sequentially and carries the state.
"""

import torch

from .gdn_split_blocksolve import prefill_blocksolve_A_bthd_split
from .gdn_split_k6_batched import k6_forward_batched


def gdn_prefill_split(q, k, v, g, beta, state, scale, chunk=64):
    B, S, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    assert B == 1 and H == HV and K == V and K in (64, 128)
    C = chunk
    s_pad = ((S + C - 1) // C) * C

    # K1: chunk-local cumulative sum. g is per-token log gate; cast to the
    # kernel dtype (fp16/bf16) and pad the tail with zeros.
    from .gdn_prefill import gdn_chunk_cumsum_kernel
    g_dtype = g.dtype if g.dtype in (torch.float16, torch.bfloat16) else torch.float16
    g_pad = torch.zeros((1, s_pad, H), dtype=g_dtype, device=g.device)
    g_pad[0, :S, :] = g[0].to(g_dtype)
    in_dtype = "bfloat16" if g_dtype == torch.bfloat16 else "float16"
    k1 = gdn_chunk_cumsum_kernel(s_pad, H, C, in_dtype)
    g_cum_pad = k1(g_pad)
    g_cum = g_cum_pad[:, :S, :]

    # K2: split blocksolve, returns padded A [s_pad, H, C] and k_rnorm.
    A_pad, _ = prefill_blocksolve_A_bthd_split(
        k, g_cum, beta, None, C, return_padded=True
    )
    torch.npu.synchronize()

    st = (
        torch.zeros((1, HV, K, V), dtype=torch.float32, device=v.device)
        if state is None
        else state.float().clone()
    )
    o_h, st_h = k6_forward_batched(
        q[0].float(), k[0].float(), v[0].float(), g[0].float(), beta[0].float(),
        st[0].contiguous(), scale, C, A_all=A_pad,
    )
    o = o_h.unsqueeze(0).to(v.dtype)
    st = st_h.unsqueeze(0)
    return o, st
