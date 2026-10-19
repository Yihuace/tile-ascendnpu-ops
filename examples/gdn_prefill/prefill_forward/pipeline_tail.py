"""K1 + K2 + K6 with a tail chunk (S=100, valid=36)."""
import sys
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/mnt/workspace/npuir-migration/examples/TileOPs")
sys.path.insert(0, "/mnt/workspace/npuir-migration/examples/gdn_prefill/prefill_blocksolve_A_bthd")
sys.path.insert(0, ".")

from tileops.kernels.linear_attention.gdn_prefill.gdn_prefill import gdn_chunk_cumsum_kernel
from split_blocksolve import prefill_blocksolve_A_bthd_split
from split_k6_vec import k6_forward, seq_golden


def run(S=100, K=64, V=64, seed=0):
    torch.manual_seed(seed)
    dev = "npu"
    q = torch.randn(S, K, device=dev)
    k = torch.nn.functional.normalize(torch.randn(S, K, device=dev), dim=-1)
    v = torch.randn(S, V, device=dev)
    g_raw = -torch.rand(S, device=dev) * 0.2
    beta = torch.rand(S, device=dev) * 0.1
    S_in = torch.randn(K, V, device=dev) * 0.1
    scale = K ** -0.5

    s_pad = ((S + 63) // 64) * 64
    g_pad = torch.zeros(s_pad, device=dev)
    g_pad[:S] = g_raw
    k1 = gdn_chunk_cumsum_kernel(s_pad, 1, 64, "float16")
    g_cum = k1(g_pad.half().view(1, s_pad, 1))[0, :S, 0].float()
    A_pad, kr = prefill_blocksolve_A_bthd_split(
        k.half().view(1, S, 1, K), g_cum.view(1, S, 1), beta.view(1, S, 1), None, 64,
        return_padded=True,
    )
    torch.npu.synchronize()
    o, S_out = k6_forward(q, k, v, g_raw, beta, S_in, scale, 64, A_all=A_pad)
    torch.npu.synchronize()
    ref_o, ref_S = seq_golden(q.cpu(), k.cpu(), v.cpu(), g_raw.cpu(), beta.cpu(), S_in.cpu(), scale)
    eo = float((o.cpu().double() - ref_o).abs().max())
    es = float((S_out.cpu().double() - ref_S).abs().max())
    print(f"[GDN-tail] S={S} K={K} V={V} A_pad={tuple(A_pad.shape)} o_err={eo:.3e} S_err={es:.3e}")
    assert eo < 5e-3 and es < 5e-3, (eo, es)


if __name__ == "__main__":
    run(100, 64, 64, 0)
    run(100, 64, 64, 1)
    run(3000, 128, 128, 2)
    print("[GDN-tail] PASS")
