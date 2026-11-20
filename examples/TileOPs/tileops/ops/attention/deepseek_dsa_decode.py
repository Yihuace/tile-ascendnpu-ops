"""DeepSeekSparseAttentionDecodeWithKVCacheFwdOp (sparse MLA decode).

Sparse attention *decode* with a KV cache and per-(b, s, h_kv) top-k index
lists, layout BSHD:

- ``q`` ``(batch, seq_len, heads, dim + dim_tail)``
- ``kv`` ``(batch, seq_len_kv, heads_kv, dim + dim_tail)``
- ``indices`` int32 ``(batch, seq_len, heads_kv, topk)``
- output ``o`` ``(batch, seq_len, heads, dim)``

Adaptation from GPU (TileOPs) to NPU:

- O1/O2: no ``is_cuda``/``device="cuda"`` in this op; device handling is
  inherited from the NPU ``Op`` base (device-backend abstracted).
- O3: ``tune`` parameter removed from ``__init__``; the kernel constructor
  takes no ``tune`` argument (NPU uses heuristic config selection only).
- O4: the GPU ``from tileops.utils import is_hopper`` import (and the
  ``is_hopper()`` runtime gate that made this op Hopper-only) is removed --
  the WGMMA/Hopper restriction has no NPU analogue; the NPU kernel component
  designs the NPU replacement (K5-style arch-gate removal).
- O5: ``compile_boundary.register_instance`` not used on NPU.
- O6: op-specific flow preserved unchanged -- the ``q_start_index_s`` sanity
  validation, the ``cp0`` derivation, the ``sparse_mla_kernel`` dispatch, and
  the 3-input ``forward`` passthrough.
- ``eval_roofline`` is ported from the GPU manifest's
  ``tileops.perf.formulas.deepseek_dsa_decode_roofline`` (the NPU project has
  no ``tileops.perf`` module; the roofline arithmetic is device-agnostic).
"""

from typing import Dict, Optional

import torch

from tileops.kernels.attention.deepseek_dsa_decode import SparseMlaKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["DeepSeekSparseAttentionDecodeWithKVCacheFwdOp"]


class DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(Op):
    """Sparse attention decode with KV cache (DeepSeek sparse MLA).

    The layout of the operation is BSHD.

    Args:
        batch: The batch size.
        heads: The number of attention heads.
        seq_len: The length of the input sequence.
        seq_len_kv: The length of the key-value sequence.
        dim: The dimension of the attention vectors.
        dim_tail: The dimension of the tail portion of the attention vectors.
        topk: The number of top elements to consider in sparse attention.
        stride_kv: The stride for the key-value sequence.
        heads_kv: The number of key-value heads.
        q_start_index_s: The start index for queries in the sequence.
        sm_scale: Scaling factor for the softmax function (default None ->
            ``(1 / (dim + dim_tail)) ** 0.5``).
        is_causal: Whether the attention is causal (True for causal).
        dtype: The data type for the tensors (default float16).
        kernel_map: Optional mapping for custom kernels.
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 seq_len_kv: int,
                 dim: int,
                 dim_tail: int,
                 topk: int,
                 stride_kv: int,
                 heads_kv: int,
                 q_start_index_s: int,
                 sm_scale: Optional[float] = None,
                 is_causal: bool = True,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
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
        self.dtype = dtype
        self.is_causal = is_causal

        if q_start_index_s != 0 and q_start_index_s <= stride_kv:
            raise ValueError(f"Invalid q_start_index_s={q_start_index_s}:"
                             f"must be > stride_kv={stride_kv}. "
                             "This indicates incorrect cp0 masking."
                             "Ensure queries with pos < stride_kv are masked "
                             "to avoid NaNs in early outputs.")

        cp0 = q_start_index_s == 0
        self.q_start_index_s = q_start_index_s

        self.dispatch_kernel(kernel_map)
        # O3: no ``tune`` argument (NPU uses heuristic config selection only).
        self.kernel = self.kernel_map["sparse_mla_kernel"](
            self.batch,
            self.seq_len,
            self.seq_len_kv,
            self.heads,
            self.dim,
            self.dim_tail,
            self.dtype,
            self.topk,
            self.stride_kv,
            self.q_start_index_s,
            self.heads_kv,
            self.sm_scale,
            self.is_causal,
            cp0)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        """Default kernel map.

        O4: the GPU ``is_hopper()`` gate (SM90/WGMMA requirement) is removed --
        the NPU kernel component provides the NPU replacement.
        """
        return {"sparse_mla_kernel": SparseMlaKernel}

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """Run sparse attention decode.

        Args:
            q: Query tensor ``(batch, seq_len, heads, dim + dim_tail)``.
            kv: Key-value tensor ``(batch, seq_len_kv, heads_kv, dim + dim_tail)``.
            indices: int32 indices for sparse attention.

        Returns:
            The output tensor ``(batch, seq_len, heads, dim)``.
        """
        return self.kernel(q, kv, indices)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the configured workload shape.

        Ported verbatim (device-agnostic arithmetic) from the GPU repo's
        ``tileops.perf.formulas.deepseek_dsa_decode_roofline``: two GEMMs over
        the selected top-k columns of each head
        (``flops = 2 * B * S * H * topk * (2 * dim + dim_tail)``); q read +
        kv read + o written (io dtype) + int32 indices.
        """
        elem_bytes = self.dtype.itemsize
        flops = 2 * self.batch * self.seq_len * self.heads * self.topk * (
            2 * self.dim + self.dim_tail)
        q_elems = self.batch * self.seq_len * self.heads * (self.dim + self.dim_tail)
        kv_elems = self.batch * self.seq_len_kv * self.heads_kv * (self.dim + self.dim_tail)
        o_elems = self.batch * self.seq_len * self.heads * self.dim
        index_bytes = self.batch * self.seq_len * self.heads_kv * self.topk * 4
        nbytes = (q_elems + kv_elems + o_elems) * elem_bytes + index_bytes
        return int(flops), int(nbytes)
