"""K1 cumsum + K2 split blocksolve + K6 split forward, S=64 end-to-end."""
import sys
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/mnt/workspace/npuir-migration/examples/TileOPs")
sys.path.insert(0, "/mnt/workspace/npuir-migration/examples/gdn_prefill/prefill_blocksolve_A_bthd")
sys.path.insert(0, "/mnt/workspace/npuir-migration/examples/gdn_prefill/prefill_forward")

from tileops.kernels.linear_attention.gdn_prefill.gdn_prefill import gdn_chunk_cumsum_kernel
from split_blocksolve import prefill_blocksolve_A_bthd_split
from split_k6_vec import k6_forward, seq_golden


def run(S=64, K=64, V=64, seed=0):
    torch.manual_seed(seed)
    dev = "npu"
    q = torch.randn(S, K, device=dev)
    k = torch.randn(S, K, device=dev)
    v = torch.randn(S, V, device=dev)
    g_raw = -torch.rand(S, device=dev) * 0.2
    beta = torch.rand(S, device=dev) * 0.1
    S_in = torch.randn(K, V, device=dev) * 0.1
    scale = K ** -0.5

    k1 = gdn_chunk_cumsum_kernel(S, 1, 64, "float16")
    g_cum = k1(g_raw.half().view(1, S, 1))[0, :, 0].float()
    A, kr = prefill_blocksolve_A_bthd_split(
        k.view(1, S, 1, K), g_cum.view(1, S, 1), beta.view(1, S, 1), None, 64
    )
    o, S_out = k6_forward(q, k, v, g_raw, beta, S_in, scale, 64, A_all=A[0])
    torch.npu.synchronize()
    ref_o, ref_S = seq_golden(q.cpu(), k.cpu(), v.cpu(), g_raw.cpu(), beta.cpu(), S_in.cpu(), scale)
    eo = float((o.cpu().double() - ref_o).abs().max())
    es = float((S_out.cpu().double() - ref_S).abs().max())
    print(f"[GDN-pipeline] S={S} K={K} V={V} A={tuple(A.shape)} kr={tuple(kr.shape)} "
          f"o_err={eo:.3e} S_err={es:.3e}")
    assert eo < 2e-2 and es < 2e-2, (eo, es)


if __name__ == "__main__":
    run(64, 64, 64, 0)
    run(64, 64, 64, 1)
    print("[GDN-pipeline] PASS")
