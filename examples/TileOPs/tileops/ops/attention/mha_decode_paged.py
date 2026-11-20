"""MultiHeadAttentionDecodePagedWithKVCacheFwdOp (spec-conformant interface).

Paged MHA decode with a dynamic KV cache.  Layout: ``Q [batch, seqlen_q, heads,
dim]`` (BSHD); ``K`` / ``V`` physical cache ``[seqlen_kv, heads, dim]``;
``real_seqlen_kv [batch]``; ``block_table [batch, num_pages]``; output
``[batch, seqlen_q, heads, dim]``.

Adaptation from GPU (TileOPs) to NPU:

- O3: ``tune`` parameter removed from ``__init__``; the kernel constructor
  takes no ``tune`` argument (NPU uses heuristic config selection only).
- O4: no CUDA-specific helpers; the kernel class is imported from the NPU
  ``tileops.kernels.attention.mha_decode_paged`` package.
- O5: ``compile_boundary.register_instance`` removed (not used on NPU).
- O6: op-specific flow preserved unchanged -- construction dispatches
  ``mha_decode_paged_kernel`` and installs the built kernel; ``forward`` passes
  the 5 paged inputs (q, k, v, real_seqlen_kv, block_table) straight through.
- ``eval_roofline`` is ported from the GPU manifest's
  ``tileops.perf.formulas.mha_decode_paged_roofline`` (the NPU project has no
  ``tileops.perf`` module; the roofline arithmetic is device-agnostic).
"""

from typing import Dict, Optional

import torch

from tileops.kernels.attention.mha_decode_paged import MHADecodePagedKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["MultiHeadAttentionDecodePagedWithKVCacheFwdOp"]


class MultiHeadAttentionDecodePagedWithKVCacheFwdOp(Op):
    """Paged MHA decode with dynamic KV cache.

    Args:
        batch: Batch size.
        heads: Number of heads (== number of KV heads for MHA).
        seqlen_q: Query sequence length.
        seqlen_kv: Physical KV cache length.
        dim: Head dimension.
        page_size: Number of tokens per paged KV block.
        is_causal: Whether to apply a causal mask.
        dtype: Data type (float16 or bfloat16).
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 seqlen_q: int,
                 seqlen_kv: int,
                 dim: int,
                 page_size: int,
                 is_causal: bool = False,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        # O3: no ``tune`` argument (NPU uses heuristic config selection only).
        self.kernel = self.kernel_map["mha_decode_paged_kernel"](
            batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, self.dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mha_decode_paged_kernel": MHADecodePagedKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        """Run paged MHA decode on the 5 paged inputs; returns the BSHD output."""
        return self.kernel(q, k, v, real_seqlen_kv, block_table)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the configured workload shape.

        Ported verbatim (device-agnostic arithmetic) from the GPU repo's
        ``tileops.perf.formulas.mha_decode_paged_roofline``: 2 GEMMs over the
        ``(S_q x S_kv x D)`` score/output pair = ``4 * B * H * S_q * S_kv * D``
        FLOPs; q/k/v read + o written, plus int32 KV-length / block-table
        metadata bytes.
        """
        elem_bytes = self.dtype.itemsize
        flops = 4 * self.batch * self.heads * self.seqlen_q * self.seqlen_kv * self.dim
        q_elems = self.batch * self.seqlen_q * self.heads * self.dim
        kv_elems = self.seqlen_kv * self.heads * self.dim
        metadata_bytes = (
            self.batch * 4
            + self.batch * max(1, (self.seqlen_kv + self.page_size - 1) // self.page_size) * 4
        )
        nbytes = (q_elems + 2 * kv_elems + q_elems) * elem_bytes + metadata_bytes
        return int(flops), int(nbytes)
