from typing import Optional

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase


class TopkSelectorWorkload(WorkloadBase):
    """Workload for TopkSelectorOp: a flat (M, N) row matrix plus k.

    The golden set is generated on the host, so the input is built DISTINCT along
    each row (a random permutation scaled by 7).  That matters: with ties at the
    k-th boundary, which of the tied elements a top-k returns is not defined by
    the contract, and the in-repo comparison is a SET comparison -- a tie would
    test an undefined choice rather than the kernel.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype, k: int = 16):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.k = int(k)
        if len(self.shape) != 2:
            raise ValueError(f"topk_selector expects a 2-D (M, N) shape, got {self.shape}")
        self.m, self.n = self.shape
        if self.k > self.n:
            raise ValueError(f"k={self.k} exceeds N={self.n}")

    @property
    def device(self) -> str:
        return get_device_backend().name

    def gen_inputs(self) -> tuple[torch.Tensor]:
        base = torch.randperm(self.n, dtype=torch.float32) * 7.0
        x = base.unsqueeze(0).repeat(self.m, 1) + torch.arange(
            self.m, dtype=torch.float32
        ).unsqueeze(1)
        return (x.to(self.dtype).to(self.device),)


class FP8LightningIndexerWorkload(WorkloadBase):

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 heads: int,
                 index_dim: int,
                 seq_len_kv: int,
                 kv_group: int,
                 clean_logits: bool = True,
                 config: Optional[dict] = None):
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.clean_logits = clean_logits
        self.config = config
        self.dtype = torch.float8_e4m3fn
        self.accum_dtype = torch.float32
        self.index_dtype = torch.int32

    def cal_seq_idx_for_q(self, cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor,
                          seq_len: int) -> torch.IntTensor:
        seq_idx_for_q = torch.zeros(seq_len, dtype=torch.int32, device=cu_seqlens_qs.device)
        if len(cu_seqlens_qs) > 1:
            seq_idx_for_q[cu_seqlens_qs[1:]] = 1
        return torch.cumsum(seq_idx_for_q, dim=0, dtype=torch.int32)

    def cal_cu_seqlen_ke_for_q(
        self,
        cu_seqlens_qs: torch.LongTensor,
        cu_seqlens_qe: torch.LongTensor,
        cu_seqlens_ks: torch.LongTensor,
        cu_seqlens_ke: torch.LongTensor,
        q_start_idxs: torch.LongTensor,
        seq_len: int,
        kv_stride: int,
    ) -> torch.IntTensor:
        cu_seqlen_ke_for_each_q = torch.gather(
            input=torch.cat(
                [cu_seqlens_ke,
                 torch.zeros(1, dtype=torch.int32, device=cu_seqlens_qs.device)]),
            dim=0,
            index=self.cal_seq_idx_for_q(
                cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long(),
        )
        casual_cu_seqlen_ke_for_each_q = torch.zeros((seq_len,),
                                                     dtype=torch.int32,
                                                     device=cu_seqlens_qs.device)
        for i in range(len(cu_seqlens_qs)):
            casual_cu_seqlen_ke_for_each_q[cu_seqlens_qs[i]:cu_seqlens_qe[i]] = (torch.arange(
                q_start_idxs[i],
                q_start_idxs[i] + cu_seqlens_qe[i] - cu_seqlens_qs[i],
                dtype=torch.int32,
                device=cu_seqlens_qs.device) + 1) // kv_stride + cu_seqlens_ks[i]
        cu_seqlen_ke_for_each_q = torch.minimum(casual_cu_seqlen_ke_for_each_q,
                                                cu_seqlen_ke_for_each_q)
        return cu_seqlen_ke_for_each_q.int()

    def cal_cu_seqlen_ks_for_q(
        self,
        cu_seqlens_qs: torch.LongTensor,
        cu_seqlens_qe: torch.LongTensor,
        cu_seqlens_ks: torch.LongTensor,
        seq_len: int,
    ) -> torch.IntTensor:
        cu_seqlen_ks_for_each_q = torch.gather(
            input=torch.cat([
                cu_seqlens_ks,
                torch.full((1,),
                           torch.iinfo(torch.int32).max,
                           dtype=torch.int32,
                           device=cu_seqlens_qs.device)
            ]),
            dim=0,
            index=self.cal_seq_idx_for_q(
                cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long(),
        )
        return cu_seqlen_ks_for_each_q.int()

    def generate_random_cu_seqlens(self,
                                   cp_size: int = 1,
                                   cp_rank: int = 0,
                                   kv_stride: int = 1,
                                   average_q_len: int = 512):
        # W1: device resolved through the device backend (GPU used ``.cuda()``).
        device = get_device_backend().name
        # W2: single-rank NPU deployment -- the kv window axis is the local
        # ``[0, seq_len_kv)`` cache (the source GPU generator's ``cp_size`` /
        # ``cp_rank`` global-axis slicing is retained for interface parity but
        # defaults to a no-op single rank).
        total_seqlen = self.seq_len * cp_size

        # W2: at least one random segment even when ``average_q_len`` exceeds the
        # axis length (the GPU formula could yield an empty sample set / an empty
        # ``torch.where`` prefix for small shapes, e.g. seq_len=1024).
        num_rand = max(1, total_seqlen // average_q_len * 2)
        cu_seqlens = torch.randint(0, average_q_len * 2, (num_rand,), device=device)
        # Truncate at the first prefix that reaches ``total_seqlen``; if no random
        # prefix reaches it, keep every sample and let the remainder close the gap.
        reached = torch.where(cu_seqlens.cumsum(0) >= total_seqlen)[0]
        if len(reached) > 0:
            cu_seqlens = cu_seqlens[:reached[0]]

        if cu_seqlens.sum() < total_seqlen:
            cu_seqlens = torch.cat(
                [cu_seqlens, torch.tensor([total_seqlen - cu_seqlens.sum()], device=device)])

        cu_seqlens_cumsum = torch.cumsum(cu_seqlens, dim=0)
        cu_seqlens_k_cumsum = torch.cumsum(cu_seqlens // kv_stride, dim=0)
        cu_seqlens_qs = torch.cat([torch.tensor([0], device=device), cu_seqlens_cumsum[:-1]])
        cu_seqlens_ks = torch.cat([torch.tensor([0], device=device), cu_seqlens_k_cumsum[:-1]])
        cu_seqlens_qe = cu_seqlens_cumsum.clone()
        cu_seqlens_ke = cu_seqlens_k_cumsum.clone()

        cu_seqlens_ks_for_each_q = self.cal_cu_seqlen_ks_for_q(
            cu_seqlens_qs,
            cu_seqlens_qe,
            cu_seqlens_ks,
            total_seqlen,
        )

        cu_seqlens_ke_for_each_q = self.cal_cu_seqlen_ke_for_q(
            cu_seqlens_qs=cu_seqlens_qs,
            cu_seqlens_qe=cu_seqlens_qe,
            cu_seqlens_ks=cu_seqlens_ks,
            cu_seqlens_ke=cu_seqlens_ke,
            q_start_idxs=torch.zeros_like(cu_seqlens_qs),
            seq_len=total_seqlen,
            kv_stride=kv_stride,
        )

        assert self.seq_len % 2 == 0
        per_chunk_seqlen = self.seq_len // 2
        slice_short = slice(cp_rank * per_chunk_seqlen, (cp_rank + 1) * per_chunk_seqlen)
        slice_long = slice(
            total_seqlen - (cp_rank + 1) * per_chunk_seqlen,
            total_seqlen - cp_rank * per_chunk_seqlen,
        )

        ks = torch.cat([
            cu_seqlens_ks_for_each_q[slice_short],
            cu_seqlens_ks_for_each_q[slice_long],
        ])
        ke = torch.cat([
            cu_seqlens_ke_for_each_q[slice_short],
            cu_seqlens_ke_for_each_q[slice_long],
        ])
        assert len(ks) == len(ke) == self.seq_len
        # W2: constrain the window bounds to the valid local kv axis.  This is a
        # no-op for the single-rank config (bounds already lie in [0, seq_len]),
        # but enforces the [0, seq_len_kv] invariant for any cp configuration.
        ks = ks.clamp_(0, self.seq_len_kv)
        ke = ke.clamp_(0, self.seq_len_kv)
        return ks, ke

    def gen_inputs(
            self,
            params=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved through the device backend (GPU used ``'cuda'``).
        device = get_device_backend().name
        IndexQ = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.index_dim,
            device=device,
            dtype=torch.bfloat16)
        IndexK = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            device=device,
            dtype=torch.bfloat16)
        Weights = torch.randn(self.seq_len, self.heads, device=device, dtype=self.accum_dtype)
        # W2: single-rank NPU windows on the local kv axis (see
        # ``generate_random_cu_seqlens``).
        CuSeqLenKS, CuSeqLenKE = self.generate_random_cu_seqlens(
            cp_size=1, cp_rank=0, kv_stride=1, average_q_len=512)
        return IndexQ, IndexK, Weights, CuSeqLenKS, CuSeqLenKE
