"""GDN prefill — NPU kernel module (pipeline of 6 TileLang kernels).

Delivered op ``GatedDeltaNetPrefillFwdOp`` (baseline ``GDNFwdOp``).  The GPU
implementation is a six-kernel pipeline; this module grows one kernel per
Stage-3 iteration:

    1. prefill_chunk_local_cumsum_bthd_tl   <-- implemented here (function 1/6)
    2. prefill_blocksolve_A_bthd
    3. get_warmup_chunks
    4. correct_initial_states
    5. fused_gdr_h
    6. fused_gdr_fwd

Design: ``DESIGN.md`` (function 1).  Programming mode: **Developer** (pure
Vector — no Cube / ``T.gemm``), matching ``DESIGN.md`` §2.

Run: ``python gdn_prefill.py --level all``.
"""

import argparse
import os

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
import tilelang
import tilelang.language as T

from tileops.kernels.kernel_base import Kernel
from .gdn_split_pipeline import gdn_prefill_split

__all__ = ["GDNPrefillKernel", "gdn_chunk_cumsum_kernel", "golden_gdn_chunk_cumsum"]


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference — fp64 accumulation, per-chunk restart)
# ---------------------------------------------------------------------------
def golden_gdn_chunk_cumsum(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Per-chunk forward cumulative sum of the log-gate, in fp64 then cast to fp32.

    ``g``: ``[1, total_tokens, head]`` on CPU.  Every ``chunk_size`` rows the
    accumulator restarts (matching the GPU kernel's per-``cid`` reset).
    """
    g64 = g.double()
    total_tokens = g64.shape[1]
    out = torch.zeros_like(g64)
    for c in range(total_tokens // chunk_size):
        lo = c * chunk_size
        out[0, lo:lo + chunk_size] = torch.cumsum(g64[0, lo:lo + chunk_size], dim=0)
    return out.float()


# ---------------------------------------------------------------------------
# Kernel 1/6: chunk-local cumulative sum
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], target="npuir")
def gdn_chunk_cumsum_kernel(
    total_tokens: int,
    head: int,
    chunk_size: int = 64,
    in_dtype: str = "float16",
    head_tile: int = 64,
):
    """One block per (chunk, head_slab); ``T.cumsum`` does the row scan in fp32."""
    assert total_tokens % chunk_size == 0, "kernel 1: total_tokens must be a multiple of chunk_size"
    ht = min(head, head_tile)
    num_chunks = total_tokens // chunk_size
    num_slabs = (head + ht - 1) // ht

    @T.prim_func
    def main(
        g: T.Tensor((1, total_tokens, head), in_dtype),
        out: T.Tensor((1, total_tokens, head), "float32"),
    ):
        with T.Kernel(num_chunks * num_slabs, is_npu=True) as (cid, _):
            c = cid // num_slabs
            slab = cid % num_slabs
            hs = slab * ht
            g_ub = T.alloc_ub((chunk_size, ht), in_dtype)
            f_ub = T.alloc_ub((chunk_size, ht), "float32")
            o_ub = T.alloc_ub((chunk_size, ht), "float32")
            T.copy(g[0, c * chunk_size:(c + 1) * chunk_size, hs:hs + ht], g_ub)
            T.vcast(g_ub, f_ub)
            T.cumsum(f_ub, o_ub, dim=0)
            T.copy(o_ub, out[0, c * chunk_size:(c + 1) * chunk_size, hs:hs + ht])

    return main


# ---------------------------------------------------------------------------
# Layered tests
# ---------------------------------------------------------------------------
def _run_case(total_tokens, head, chunk_size, dtype, tag):
    torch.manual_seed(0)
    in_dtype = str(dtype).replace("torch.", "")
    g = torch.randn(1, total_tokens, head, dtype=dtype) * 0.5
    kernel = gdn_chunk_cumsum_kernel(
        total_tokens, head, chunk_size=chunk_size, in_dtype=in_dtype
    )
    out = kernel(g.npu())
    torch.npu.synchronize()
    ref = golden_gdn_chunk_cumsum(g, chunk_size)
    err = float((out.cpu().float() - ref).abs().max())
    atol = 1e-5 if dtype is torch.float16 else 1e-2
    assert err <= atol, f"[{tag}] FAIL shape=({total_tokens},{head}) {dtype} max_abs_err={err:.3e} > {atol}"
    print(f"[{tag}] PASS: tokens={total_tokens} head={head} chunk={chunk_size} dtype={in_dtype} err={err:.2e}")


def run_L0():
    _run_case(64, 16, 64, torch.float16, "L0")
    _run_case(64, 16, 64, torch.bfloat16, "L0b")


def run_L1():
    _run_case(192, 64, 64, torch.float16, "L1")
    _run_case(256, 64, 64, torch.float16, "L1")


def run_L2():
    _run_case(4096, 16, 64, torch.float16, "L2")
    _run_case(12288, 16, 64, torch.bfloat16, "L2")


def test_boundary():
    """Boundary: chunk restart parity and a single-chunk case."""
    _run_case(64, 64, 64, torch.float16, "boundary-single-chunk")
    _run_case(128, 64, 64, torch.bfloat16, "boundary-two-chunks")


# ---------------------------------------------------------------------------
# Kernel 3/6: warmup chunk count (backward scan of chunk-end gates)
# ---------------------------------------------------------------------------
def golden_gdn_warmup_chunks(g, cu_seqlens, ht_mask, chunk_size=64, threshold=-10.0):
    """Reference: walk chunks backwards, accumulate the chunk-end gate, stop at threshold."""
    B = ht_mask.shape[0]
    heads = g.shape[2]
    n = torch.zeros((B, heads), dtype=torch.int32)
    f = torch.zeros((B, heads), dtype=torch.int32)
    for b in range(B):
        if int(ht_mask[b]) != 0:
            continue
        start, end = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        num_iters = (end - start) // chunk_size
        acc = torch.zeros(heads, dtype=torch.float64)
        nb = torch.full((heads,), num_iters, dtype=torch.int32)
        fb = torch.ones(heads, dtype=torch.int32)
        done = torch.zeros(heads, dtype=torch.bool)
        for i_s in range(num_iters):
            row = end - i_s * chunk_size - 1
            acc = acc + g[0, row].double()
            hit = (acc < threshold) & (~done)
            nb = torch.where(hit, torch.full_like(nb, i_s + 1), nb)
            fb = torch.where(hit, torch.zeros_like(fb), fb)
            done = done | hit
        n[b] = nb
        f[b] = fb
    return n, f


@tilelang.jit(out_idx=[3, 4], target="npuir")
def gdn_warmup_chunks_kernel(batch, num_tokens, heads, chunk_size=64,
                             threshold=-10.0, g_dtype="float16"):
    seq_len = num_tokens // batch
    num_iters = seq_len // chunk_size
    assert num_iters * chunk_size == seq_len, "in-scope: equal-length, seq_len % chunk == 0"

    @T.prim_func
    def main(
        g: T.Tensor((1, num_tokens, heads), g_dtype),
        cu_seqlens: T.Tensor((batch + 1,), "int32"),
        ht_mask: T.Tensor((batch,), "int32"),
        num_warmup: T.Tensor((batch, heads), "int32"),
        fallback: T.Tensor((batch, heads), "int32"),
    ):
        with T.Kernel(batch, is_npu=True) as (bb, _):
            acc = T.alloc_ub((heads,), "float32")
            gr = T.alloc_ub((heads,), g_dtype)
            nr = T.alloc_ub((heads,), "int32")
            fr = T.alloc_ub((heads,), "int32")
            for h in T.Parallel(heads):
                acc[h] = T.float32(0.0)
                nr[h] = num_iters
                fr[h] = 1
            if ht_mask[bb] > 0:
                for h in T.Parallel(heads):
                    nr[h] = 0
            else:
                for i_s in T.serial(num_iters):
                    row = (bb + 1) * seq_len - i_s * chunk_size - 1
                    T.copy(g[0, row, 0:heads], gr)
                    for h in T.Parallel(heads):
                        acc[h] = acc[h] + T.cast(gr[h], "float32")
                        if acc[h] < threshold and nr[h] == num_iters:
                            nr[h] = i_s + 1
                            fr[h] = 0
            for h in T.Parallel(heads):
                num_warmup[bb, h] = nr[h]
                fallback[bb, h] = fr[h]

    return main


def _run_warmup_case(batch, seq_len, heads, dtype, tag, threshold=-10.0):
    torch.manual_seed(0)
    in_dtype = str(dtype).replace("torch.", "")
    num_tokens = batch * seq_len
    g = (-torch.rand(1, num_tokens, heads, dtype=dtype) - 0.5)
    cu = torch.arange(0, batch + 1, dtype=torch.int32) * seq_len
    mask = torch.zeros(batch, dtype=torch.int32)
    k = gdn_warmup_chunks_kernel(batch, num_tokens, heads,
                                 threshold=threshold, g_dtype=in_dtype)
    n, f = k(g.npu(), cu.npu(), mask.npu())
    torch.npu.synchronize()
    ref_n, ref_f = golden_gdn_warmup_chunks(g, cu, mask, threshold=threshold)
    assert torch.equal(n.cpu().int(), ref_n), f"[{tag}] n mismatch: {n.cpu().int()} vs {ref_n}"
    assert torch.equal(f.cpu().int(), ref_f), f"[{tag}] fallback mismatch: {f.cpu().int()} vs {ref_f}"
    print(f"[{tag}] PASS: batch={batch} seq_len={seq_len} heads={heads} {in_dtype} n={ref_n[0].tolist()[:4]}")


def run_L0_warmup():
    _run_warmup_case(2, 1024, 16, torch.float16, "L0-warmup")
    _run_warmup_case(2, 1024, 16, torch.bfloat16, "L0b-warmup")


def run_L1_warmup():
    _run_warmup_case(4, 512, 64, torch.float16, "L1-warmup")


def test_boundary_warmup():
    _run_warmup_case(1, 64, 16, torch.float16, "boundary-warmup-single-chunk")


# ---------------------------------------------------------------------------
# Op-level wrapper (filled at Stage 5 once all six kernels exist)
# ---------------------------------------------------------------------------
class GDNPrefillKernel(Kernel):
    """Chunked gated delta-rule prefill (equal-length, key-major state)."""

    supported_archs = None
    ascend_mode = "Developer"

    def __init__(self, B, S, H, HV, K, V, dtype=torch.bfloat16, scale=None, chunk=64, config=None):
        super().__init__()
        self.B, self.S, self.H, self.HV, self.K, self.V = B, S, H, HV, K, V
        self.dtype = dtype
        self.scale = float(scale) if scale is not None else K ** -0.5
        self.chunk = chunk
        self.init_config(config)

    @property
    def default_config(self):
        return {"chunk": self.chunk}

    def forward(self, q, k, v, g, beta, state=None):
        """Correctness-first fallback: sequential delta-rule reference on NPU.

        The six-kernel TileLang pipeline is still being ported; this path keeps the
        delivered op numerically correct for the in-scope API (B=1, equal length,
        key-major state).  It is intentionally a reference-speed implementation and
        is replaced kernel-by-kernel as the Cube/Vector ports land.
        """
        B, S, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        if (
            os.environ.get("GDN_FORCE_FALLBACK") != "1"
            and B == 1 and S >= 64 and H == HV and K == V and K in (64, 128)
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            return gdn_prefill_split(q, k, v, g, beta, state, self.scale, self.chunk)
        group = HV // H
        if group < 1 or H * group != HV:
            raise ValueError(f"H={H}, HV={HV}: HV must be a multiple of H")
        dev = q.device
        st = (
            torch.zeros((B, HV, K, V), dtype=torch.float32, device=dev)
            if state is None
            else state.to(torch.float32).clone()
        )
        o = torch.empty((B, S, HV, V), dtype=v.dtype, device=dev)
        qq = q if group == 1 else q.repeat_interleave(group, dim=2)
        kk = k if group == 1 else k.repeat_interleave(group, dim=2)
        gf = g.to(torch.float32)
        bf = beta.to(torch.float32)
        for t in range(S):
            dec = torch.exp(gf[:, t])                       # [B, HV]
            kt = kk[:, t].to(torch.float32)                 # [B, HV, K]
            vt = v[:, t].to(torch.float32)                  # [B, HV, V]
            corr = vt - torch.einsum("bhk,bhkv->bhv", kt, st)
            st = dec[..., None, None] * st + bf[:, t, :, None, None] * kt[..., :, None] * corr[..., None, :]
            ot = torch.einsum("bhk,bhkv->bhv", qq[:, t].to(torch.float32), st)
            o[:, t] = (self.scale * ot).to(v.dtype)
        return o, st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "all"])
    args = ap.parse_args()
    if args.level in ("L0", "all"):
        run_L0()
    if args.level in ("L1", "all"):
        run_L1()
    if args.level in ("L2", "all"):
        run_L2()
    if args.level == "all":
        test_boundary()
