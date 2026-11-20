"""Workload definitions for the attention op family (NPU).

Ported from the GPU TileOPs ``workloads/attention/mha.py`` --
``MhaFwdWorkload`` and ``MhaBwdWorkload``.  ``MhaDecodeWorkload`` /
``MhaDecodePagedWorkload`` belong to ops that are not migrated yet.

``MhaBwdWorkload`` calls ``MultiHeadAttentionFwdOp`` to materialize the
forward output ``o`` and needs the per-row square-attention log-sum-exp,
which on the GPU comes from ``workloads/attention/gqa.py``.  The NPU
project has no ``workloads/attention/`` package, so that helper
(``_compute_gqa_square_lse``) is ported verbatim into this module.

The only adaptation (W1) is device placement: ``device='cuda'`` is
resolved through :func:`tileops.device.get_device_backend` instead of the
hard-coded ``"cuda"``.
"""

import math

import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.ops.attention.multi_head_attention import MultiHeadAttentionFwdOp
from tileops.workloads.workload_base import WorkloadBase

__all__ = ["DsaDecodeWorkload", "GqaPagedWorkload", "GroupedQueryAttentionBwdWorkload", "GroupedQueryAttentionSlidingWindowFwdWorkload", "GroupedQueryAttentionSlidingWindowVarlenFwdWorkload", "MhaBwdWorkload", "MhaDecodePagedWorkload", "MhaFwdWorkload"]


class MhaFwdWorkload(WorkloadBase):
    """Workload for MultiHeadAttentionFwdOp.

    Generates the q/k/v triple in BSHD layout:
    ``(batch, seq_len, heads, dim)`` per tensor.
    """

    def __init__(
        self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool, dtype: torch.dtype
    ):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        k = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        v = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        return q, k, v


def _compute_gqa_square_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
) -> torch.Tensor:
    """Per-row log-sum-exp of the square (S x S) attention scores, base-2.

    Ported verbatim from the GPU ``workloads/attention/gqa.py`` (device-agnostic:
    ``q.device`` / ``masked_fill`` / ``logsumexp``).  The fp32 score path and the
    ``* log2(e)`` conversion match the kernel's e-base online-softmax domain.
    """
    groups = heads // heads_kv
    seq_len = q.shape[1]
    q_bhsd = q.transpose(1, 2).float()
    k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * (dim**-0.5)
    if is_causal:
        pos = torch.arange(seq_len, device=q.device)
        mask = pos[None, :] <= pos[:, None]
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), float("-inf"))
    return torch.logsumexp(scores, dim=-1) * math.log2(math.e)


class MhaBwdWorkload(WorkloadBase):
    """Workload for MultiHeadAttentionBwdOp.

    Generates the backward input tuple in BSHD layout:
    ``q``, ``k``, ``v`` (all ``requires_grad=True``, for the autograd
    reference), the forward output ``o`` (produced by
    :class:`MultiHeadAttentionFwdOp`), a random ``grad_output``, and the
    base-2 per-row log-sum-exp ``lse`` (shape ``(B, H, S)``, float32).
    """

    def __init__(
        self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool, dtype: torch.dtype
    ):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device=device,
            requires_grad=True,
        )
        k = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device=device,
            requires_grad=True,
        )
        v = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device=device,
            requires_grad=True,
        )
        grad_output = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device=device
        )

        fwd_op = MultiHeadAttentionFwdOp(
            self.batch, self.heads, self.seq_len, self.dim, self.is_causal, self.dtype
        )
        with torch.no_grad():
            result = fwd_op(q, k, v)
            o = result[0] if isinstance(result, tuple) else result
            lse = _compute_gqa_square_lse(
                q,
                k,
                heads=self.heads,
                heads_kv=self.heads,
                dim=self.dim,
                is_causal=self.is_causal,
            )

        return q, k, v, o, grad_output, lse


class GroupedQueryAttentionBwdWorkload(WorkloadBase):
    """Workload for GroupedQueryAttentionBwdOp.

    Generates the 6 backward inputs, preserving the GPU input count and
    generation logic (``device`` adapted per W1):

    - ``q`` / ``k`` / ``v``: BSHD ``(batch, seq_len, {heads,heads_kv}, dim)``,
      ``requires_grad=True`` (the correctness reference backprops through them)
    - ``o``: forward attention output ``(batch, seq_len, heads, dim)``
    - ``do``: output gradient ``(batch, seq_len, heads, dim)``
    - ``lse``: fp32 base-2 log-sum-exp ``(batch, heads, seq_len)``

    NPU adaptation: the GPU version computes ``o`` with the TileOPs
    ``GroupedQueryAttentionFwdOp``; that forward op is not migrated on NPU, so
    ``o`` is generated here with a torch SDPA reference (``enable_gqa=True``)
    instead.  The downstream ``delta = rowsum(o * do)`` preprocessing requires
    ``o`` to be the true forward output, so a consistent reference is used.
    """

    def __init__(self, batch: int, heads: int, heads_kv: int, seq_len: int, dim: int,
                 is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim,
            dtype=self.dtype, device=device, requires_grad=True)
        k = torch.randn(
            self.batch, self.seq_len, self.heads_kv, self.dim,
            dtype=self.dtype, device=device, requires_grad=True)
        v = torch.randn(
            self.batch, self.seq_len, self.heads_kv, self.dim,
            dtype=self.dtype, device=device, requires_grad=True)
        do = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device=device)

        with torch.no_grad():
            # GPU used the (unmigrated) GroupedQueryAttentionFwdOp here; use a
            # torch SDPA reference so ``o`` matches the correctness reference.
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=self.is_causal,
                enable_gqa=True,
            ).transpose(1, 2).contiguous()
            lse = _compute_gqa_square_lse(
                q,
                k,
                heads=self.heads,
                heads_kv=self.heads_kv,
                dim=self.dim,
                is_causal=self.is_causal,
            )

        return q, k, v, o, do, lse


class DsaDecodeWorkload(WorkloadBase):
    """Workload for DeepSeekSparseAttentionDecodeWithKVCacheFwdOp.

    Generates the q / kv / indices triple in BSHD layout, preserving the
    GPU generation logic verbatim (``device`` only adapted per W1):

    - ``q``: ``(batch, seq_len, heads, dim + dim_tail)``
    - ``kv``: ``(batch, seq_len_kv, heads_kv, dim + dim_tail)``
    - ``indices``: int32 ``(batch, seq_len, heads_kv, topk)`` — per (b, t, h)
      a random permutation of the valid (causal) KV positions, padded with
      ``seq_len_kv`` (the out-of-range sentinel) up to ``topk``.
    """

    def __init__(self, batch: int, heads: int, seq_len: int, seq_len_kv: int, dim: int,
                 dim_tail: int, topk: int, stride_kv: int, heads_kv: int, q_start_index_s: int,
                 sm_scale: float = None, is_causal: bool = True,
                 dtype: torch.dtype = torch.float16) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_tail = dim_tail
        self.topk = topk
        self.stride_kv = stride_kv
        self.heads_kv = heads_kv
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.dtype = dtype
        self.q_start_index_s = q_start_index_s

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim + self.dim_tail,
            device=device,
            dtype=self.dtype)
        kv = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim + self.dim_tail,
            device=device,
            dtype=self.dtype)
        indices = torch.full((self.batch, self.seq_len, self.heads_kv, self.topk),
                             self.seq_len_kv,
                             dtype=torch.int32,
                             device=device)
        for b in range(self.batch):
            for t in range(self.seq_len):
                for h in range(self.heads_kv):
                    i_i = torch.randperm(
                        min(
                            max(1, ((t + int(self.q_start_index_s)) // self.stride_kv)),
                            self.seq_len_kv))[:self.topk]
                    indices[b, t, h, :len(i_i)] = i_i
        return q, kv, indices


class GqaPagedWorkload(WorkloadBase):
    """Workload for GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp.

    Builds the whole paged-attention input set: contiguous new K/V, a page pool,
    the per-request page table, the scalar dequant scales, and ``cu_seqlens_q``.
    The page table is laid out so each request owns a disjoint set of pages, which
    makes a page-indexing mistake visible immediately.
    """

    def __init__(self, spec: dict, dtype: torch.dtype):
        self.spec = dict(spec)
        self.dtype = dtype
        self.batch = int(spec["batch"])
        self.q_lens = [int(v) for v in spec["q_lens"]]
        self.cache_lens = [int(v) for v in spec["cache_lens"]]
        self.heads = int(spec["heads"])
        self.heads_kv = int(spec["heads_kv"])
        self.dim = int(spec["dim"])
        self.page_size = int(spec["page_size"])
        self.max_pages = int(spec["max_pages_per_req"])
        self.total_q = int(sum(self.q_lens))

    @property
    def device(self) -> str:
        return get_device_backend().name

    def gen_inputs(self) -> tuple:
        g = torch.Generator().manual_seed(0)
        H, H_kv, D, PS = self.heads, self.heads_kv, self.dim, self.page_size
        q = torch.randn(self.total_q, H, D, generator=g).to(self.dtype)
        k_new = torch.randn(self.total_q, H_kv, D, generator=g).to(self.dtype)
        v_new = torch.randn(self.total_q, H_kv, D, generator=g).to(self.dtype)

        pages_per_req = [(c + PS - 1) // PS for c in self.cache_lens]
        phys_pages = sum(pages_per_req)
        phys_tokens = phys_pages * PS
        k_pages = torch.randn(phys_tokens, H_kv, D, generator=g).to(self.dtype)
        v_pages = torch.randn(phys_tokens, H_kv, D, generator=g).to(self.dtype)

        bt = torch.zeros(self.batch, self.max_pages, dtype=torch.int32)
        base = 0
        for b, npg in enumerate(pages_per_req):
            bt[b, :npg] = torch.arange(base, base + npg, dtype=torch.int32)
            base += npg

        cu = torch.zeros(self.batch + 1, dtype=torch.int32)
        cu[1:] = torch.tensor(self.q_lens, dtype=torch.int32).cumsum(0)
        cl = torch.tensor(self.cache_lens, dtype=torch.int32)
        ks = torch.tensor([1.0], dtype=torch.float32)
        vs = torch.tensor([1.0], dtype=torch.float32)

        dev = self.device
        return (q.to(dev), k_new.to(dev), v_new.to(dev),
                k_pages.to(dev), v_pages.to(dev), ks.to(dev), vs.to(dev),
                cu.to(dev), cl.to(dev), bt.to(dev))


class MhaDecodePagedWorkload(WorkloadBase):
    """Workload for MultiHeadAttentionDecodePagedWithKVCacheFwdOp.

    Generates the 5 paged-decode inputs, preserving the GPU generation logic
    verbatim (``device`` only adapted per W1):

    - ``q``: BSHD ``(batch, seqlen_q, heads, dim)``
    - ``k`` / ``v``: physical paged cache ``(seqlen_kv, heads, dim)``
    - ``real_seqlen_kv``: int32 ``(batch,)``, all equal to the full cache length
    - ``block_table``: int32 ``(batch, num_pages)`` identity mapping
      (logical page i -> physical page i, i.e. contiguous layout)
    """

    def __init__(self, batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int,
                 page_size: int, is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
            self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        num_pages = self.seqlen_kv // self.page_size
        real_seqlen_kv = torch.ones(
            (self.batch,), dtype=torch.int32, device=device) * self.seqlen_kv
        q = torch.randn(
            self.batch, self.seqlen_q, self.heads, self.dim, device=device, dtype=self.dtype)
        k = torch.randn(self.seqlen_kv, self.heads, self.dim, device=device, dtype=self.dtype)
        v = torch.randn(self.seqlen_kv, self.heads, self.dim, device=device, dtype=self.dtype)
        # Identity block_table: logical page i -> physical page i (contiguous layout)
        block_table = torch.arange(
            num_pages, dtype=torch.int32, device=device).unsqueeze(0).expand(self.batch, -1)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        block_table = block_table.contiguous()
        real_seqlen_kv = real_seqlen_kv.contiguous()

        return q, k, v, real_seqlen_kv, block_table


class GroupedQueryAttentionSlidingWindowFwdWorkload(WorkloadBase):
    """Workload for GroupedQueryAttentionSlidingWindowFwdOp.

    Generates the GQA q/k/v triple in BSHD layout: q ``(batch, seq, heads, dim)``
    and k/v ``(batch, seq, heads_kv, dim)``.  Inputs are scaled by ``0.1`` so the
    sliding-window softmax stays well-conditioned (as in the GPU workload).
    """

    def __init__(
        self,
        batch: int,
        seq: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        wl: int,
        wr: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.seq = seq
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.wl = wl
        self.wr = wr
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        q = torch.randn(
            self.batch, self.seq, self.heads, self.dim, dtype=self.dtype, device=device
        ) * 0.1
        k = torch.randn(
            self.batch, self.seq, self.heads_kv, self.dim, dtype=self.dtype, device=device
        ) * 0.1
        v = torch.randn(
            self.batch, self.seq, self.heads_kv, self.dim, dtype=self.dtype, device=device
        ) * 0.1
        return q, k, v


class GroupedQueryAttentionSlidingWindowVarlenFwdWorkload(WorkloadBase):
    """Workload for GroupedQueryAttentionSlidingWindowVarlenFwdOp (NPU).

    Ported from the GPU ``GroupedQueryAttentionSlidingWindowVarlenFwdWorkload``
    (``workloads/attention/gqa.py``).  Generates packed (no padding) q/k/v --
    q ``(total_q, heads, dim)`` and k/v ``(total_k, heads_kv, dim)`` -- plus the
    int32 cumulative-sequence-length arrays ``cu_seqlens_q`` / ``cu_seqlens_k``
    ``(batch + 1,)`` and the scalar ``max_seqlen_q``.  Inputs are scaled by
    ``0.1`` so the sliding-window softmax stays well-conditioned (as in the GPU
    workload).

    Note the mixed dtypes: q/k/v carry ``dtype`` (fp16/bf16) while the
    ``cu_seqlens_*`` arrays are always ``int32``.
    """

    def __init__(
        self,
        batch: int,
        seqlens_q: list[int],
        seqlens_k: list[int],
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        wl: int,
        wr: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.seqlens_q = seqlens_q
        self.seqlens_k = seqlens_k
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.wl = wl
        self.wr = wr
        self.dtype = dtype

    def gen_inputs(
        self,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int
    ]:
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        total_q = sum(self.seqlens_q)
        total_k = sum(self.seqlens_k)
        q = torch.randn(total_q, self.heads, self.dim, dtype=self.dtype, device=device) * 0.1
        k = torch.randn(total_k, self.heads_kv, self.dim, dtype=self.dtype, device=device) * 0.1
        v = torch.randn(total_k, self.heads_kv, self.dim, dtype=self.dtype, device=device) * 0.1

        # cu_seqlens arrays are int32 (mixed-dtype workload: q/k/v are fp16/bf16).
        cu_seqlens_q = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(self.seqlens_q), 0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        cu_seqlens_k = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(self.seqlens_k), 0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        max_seqlen_q = max(self.seqlens_q)

        return q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q
