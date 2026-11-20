"""GroupedQueryAttentionBwdOp (spec-conformant interface).

GQA backward, BSHD layout.  Given ``q``/``k``/``v``, the forward output ``o``,
the output gradient ``do`` and the log-sum-exp ``lse``, produces
``dq``/``dk``/``dv`` via a three-kernel pipeline (preprocess -> main ->
postprocess).

Adaptation from GPU (TileOPs) to NPU:

- O3: ``tune`` parameter removed from ``__init__``; the kernel constructors
  take no ``tune`` argument.
- O6: op-specific flow preserved unchanged -- ``delta = rowsum(o * do)`` via
  the preprocess kernel; the main kernel recomputes the softmax probabilities
  from ``q``/``k``/``lse`` and accumulates ``dq``/``dk``/``dv``; the
  postprocess kernel casts the fp32 ``dq`` accumulator to the storage dtype
  (K5: the GPU's ``is_hopper()`` branch -- which skips the postprocess kernel
  and casts ``dq`` directly -- is not ported, since NPU has no Hopper/WGMMA
  path).
- ``eval_roofline`` is ported from the GPU manifest's
  ``tileops.perf.formulas.gqa_bwd_roofline`` (the NPU project has no
  tileops.perf module; roofline arithmetic is device-agnostic):
  ``flops = 10 * B * H * S * S * D`` (halved when causal) and
  ``bytes = B * (3 * H + 4 * H_kv) * S * D * elem_bytes``.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.attention.grouped_query_attention_bwd import (
    FlashAttnBwdPostprocessKernel,
    FlashAttnBwdPreprocessKernel,
    GQABwdKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["GroupedQueryAttentionBwdOp"]


class GroupedQueryAttentionBwdOp(Op):
    """Grouped-query attention backward. Layout: BSHD.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        heads_kv: Number of key/value heads (``heads % heads_kv == 0``).
        seq_len: Sequence length (query and kv; ``s_q != s_kv`` is not
            supported yet).
        dim: Head dimension.
        is_causal: Whether the forward pass used a causal mask.
        dtype: Data type (float16 or bfloat16).
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool = True,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        # O3: no ``tune`` argument on any of the three kernel constructors.
        self.prep_kernel = self.kernel_map["gqa_bwd_preprocess_kernel"](
            batch, heads, seq_len, dim, self.dtype
        )
        self.kernel = self.kernel_map["gqa_bwd_kernel"](
            batch, heads, heads_kv, seq_len, dim, is_causal, self.dtype
        )
        # K5: the GPU skips the postprocess kernel on Hopper (WGMMA path casts
        # ``dq`` directly); NPU always runs the postprocess kernel.
        self.post_kernel = self.kernel_map["gqa_bwd_postprocess_kernel"](
            batch, heads, seq_len, dim, self.dtype
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # K5: the GPU selects ``GQABwdWgmmaPipelinedKernel`` (Hopper/WGMMA)
        # when ``is_hopper()`` and ``None`` for the postprocess kernel in that
        # case; NPU uses the architecture-generic three-kernel pipeline.
        return {
            "gqa_bwd_preprocess_kernel": FlashAttnBwdPreprocessKernel,
            "gqa_bwd_kernel": GQABwdKernel,
            "gqa_bwd_postprocess_kernel": FlashAttnBwdPostprocessKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(dq, dk, dv)`` for BSHD-layout ``q``/``k``/``v``/``o``/``do``.

        ``lse`` is the fp32 log-sum-exp (base-2 domain) of shape
        ``[batch, heads, seq_len]``.  ``dq`` starts as an fp32 accumulator and
        is cast back to the storage dtype by the postprocess kernel; ``dk`` /
        ``dv`` are cast back directly.
        """
        do = do.contiguous()
        delta = self.prep_kernel(o, do)
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        self.kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = self.post_kernel(dq)
        dk, dv = dk.to(self.dtype), dv.to(self.dtype)
        return dq, dk, dv

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the configured workload shape.

        Ported verbatim (device-agnostic arithmetic) from the GPU repo's
        ``tileops.perf.formulas.gqa_bwd_roofline``: ``10 * B * H * S * S * D``
        FLOPs (halved when causal) and
        ``B * (3 * H + 4 * H_kv) * S * D * elem_bytes`` bytes.
        """
        elem_bytes = self.dtype.itemsize
        flops = 10 * self.batch * self.heads * self.seq_len * self.seq_len * self.dim
        if self.is_causal:
            flops //= 2
        nbytes = (
            self.batch
            * (3 * self.heads + 4 * self.heads_kv)
            * self.seq_len
            * self.dim
            * elem_bytes
        )
        return int(flops), int(nbytes)
