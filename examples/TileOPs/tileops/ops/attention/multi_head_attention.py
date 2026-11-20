"""MultiHeadAttentionFwdOp (spec-conformant interface).

Adaptation from GPU (TileOPs) to NPU:

- O3: ``tune`` parameter removed from ``__init__``; the kernel
  constructor takes no ``tune`` argument.
- O6: op-specific flow preserved unchanged -- the GPU routes MHA through
  ``GroupedQueryAttentionFwdOp`` (the ``heads_kv == heads``
  specialization), which resolves ``sm_scale = dim ** -0.5`` and
  ``softcap = 0.0`` and instantiates the dense-path GQA prefill kernel at
  construction time, and ``forward`` returns the historical MHA contract
  ``(output, lse)``.  The NPU port inlines that dense-path dispatch (the
  GQA op wrapper itself is not migrated) while keeping the same kernel
  key (``gqa_prefill_fwd_kernel``) and return contract.
- K5: the GPU kernel selection (``_select_gqa_prefill_fwd_kernel_cls`` --
  Hopper warp-specialized variant -- plus the H200 square dense fast
  path) collapses to the architecture-generic ``GQAPrefillFwdKernel``.
- ``eval_roofline`` is ported from the GPU manifest's
  ``tileops.perf.formulas.mha_fwd_roofline`` (the NPU project has no
  tileops.perf module; roofline arithmetic is device-agnostic).

``MultiHeadAttentionBwdOp`` follows the same adaptation pattern for the
backward pass:

- O3/K5: ``tune`` removed; the GPU's Hopper-conditional kernel selection
  (``GQABwdWgmmaPipelinedKernel`` vs ``GQABwdKernel`` and the
  ``is_hopper()``-gated postprocess entry) collapses to the three
  architecture-generic dense backward kernels.
- O6: the GPU op delegates to ``GroupedQueryAttentionBwdOp``; the NPU port
  inlines that dense-path dispatch (the GQA op wrapper itself is not
  migrated) while keeping the same kernel keys (``gqa_bwd_*``), the
  ``forward(q, k, v, o, do, lse)`` contract and the legacy-key rejection.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.attention.multi_head_attention import (
    FlashAttnBwdPostprocessKernel,
    FlashAttnBwdPreprocessKernel,
    GQABwdKernel,
    GQAPrefillFwdKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["MultiHeadAttentionBwdOp", "MultiHeadAttentionFwdOp"]


class MultiHeadAttentionFwdOp(Op):
    """Multi-head attention forward. Layout: BSHD.

    MHA is the ``heads_kv == heads`` specialization of GQA, so the
    maintained forward path runs the GQA prefill kernel while keeping the
    historical MHA return contract ``(output, lse)``.

    Args:
        batch: Batch size.
        heads: Number of heads.
        seq_len: Sequence length (query and kv; ``s_q != s_kv`` is not
            supported yet).
        dim: Head dimension.
        is_causal: Whether to apply a causal mask.
        dtype: Data type (float16 or bfloat16).
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        is_causal: bool = True,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        # GPU: GroupedQueryAttentionFwdOp(batch, heads, heads_kv=heads, ...)
        # resolves sm_scale = dim ** -0.5 and softcap = 0.0, then
        # instantiates the dense-path kernel at construction time (MHA's
        # torch.compile smoke expects forward to call an already-built
        # custom op).  O3: no ``tune`` argument.
        self.kernel: Kernel = self.kernel_map["gqa_prefill_fwd_kernel"](
            batch,
            heads,
            heads,
            seq_len,
            seq_len,
            dim,
            is_causal,
            dtype,
            sm_scale=dim**-0.5,
            softcap=0.0,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # K5: the GPU selects GQAPrefillFwdKernel (generic),
        # GQAPrefillFwdWsPersistentCausalKernel (Hopper) or the H200
        # square GQAFwdWsPersistentCausalKernel via
        # _select_gqa_prefill_fwd_kernel_cls / the square dense fast path;
        # NPU uses the architecture-generic dense prefill kernel.
        return {"gqa_prefill_fwd_kernel": GQAPrefillFwdKernel}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run MHA forward on BSHD-layout q/k/v; returns ``(output, lse)``."""
        return self.kernel(q, k, v)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the configured workload shape.

        Ported verbatim (device-agnostic arithmetic) from the GPU repo's
        ``tileops.perf.formulas.mha_fwd_roofline``: 2 GEMMs per
        ``(S x S x D)`` score/output pair = ``4 * B * H * S * S * D``
        FLOPs (halved when causal); q/k/v read + o written.
        """
        elem_bytes = self.dtype.itemsize
        flops = 4 * self.batch * self.heads * self.seq_len * self.seq_len * self.dim
        if self.is_causal:
            flops //= 2
        q_elems = self.batch * self.seq_len * self.heads * self.dim
        kv_elems = q_elems
        return int(flops), int(2 * (q_elems + kv_elems) * elem_bytes)


class MultiHeadAttentionBwdOp(Op):
    """Multi-head attention backward. Layout: BSHD.

    MHA backward is the ``heads_kv == heads`` specialization of GQA backward,
    matching the forward path's dispatch through GQA.

    The GPU op delegates to ``GroupedQueryAttentionBwdOp``; the NPU port
    inlines that dense-path dispatch (the GQA op wrapper itself is not
    migrated) while keeping the historical kernel keys (``gqa_bwd_*``) and
    the ``forward(q, k, v, o, do, lse)`` contract.

    Args:
        batch: Batch size.
        heads: Number of heads.
        seq_len: Sequence length (query and kv; ``s_q != s_kv`` is not
            supported yet).
        dim: Head dimension.
        is_causal: Whether to apply a causal mask.
        dtype: Data type (float16 or bfloat16).
        kernel_map: Optional override for kernel dispatch.
    """

    _LEGACY_KERNEL_MAP_KEYS = frozenset(
        {
            "mha_bwd_preprocess_kernel",
            "mha_bwd_kernel",
            "mha_bwd_postprocess_kernel",
        }
    )

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        is_causal: bool = True,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.dispatch_kernel(self._gqa_kernel_map(kernel_map))
        # GPU: GroupedQueryAttentionBwdOp(batch, heads, heads_kv=heads, ...)
        # instantiates the preprocess / GQA-backward / postprocess kernels at
        # construction time.  O3: no ``tune`` argument.
        self.prep_kernel: Kernel = self.kernel_map["gqa_bwd_preprocess_kernel"](
            batch, heads, seq_len, dim, self.dtype
        )
        self.kernel: Kernel = self.kernel_map["gqa_bwd_kernel"](
            batch, heads, heads, seq_len, dim, is_causal, self.dtype
        )
        # K5: the GPU postprocess entry is ``is_hopper()``-gated (None on
        # Hopper, where dq is cast inline); NPU always uses the generic
        # postprocess kernel.
        self.post_kernel: Kernel = self.kernel_map["gqa_bwd_postprocess_kernel"](
            batch, heads, seq_len, dim, self.dtype
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # K5: the GPU selects GQABwdWgmmaPipelinedKernel (Hopper) over
        # GQABwdKernel and gates the postprocess entry on ``not is_hopper()``;
        # NPU uses the architecture-generic dense backward kernels
        # unconditionally.
        return {
            "gqa_bwd_preprocess_kernel": FlashAttnBwdPreprocessKernel,
            "gqa_bwd_kernel": GQABwdKernel,
            "gqa_bwd_postprocess_kernel": FlashAttnBwdPostprocessKernel,
        }

    @staticmethod
    def _gqa_kernel_map(
        kernel_map: Optional[Dict[str, Kernel]],
    ) -> Optional[Dict[str, Kernel]]:
        if kernel_map is None:
            return None
        legacy_keys = MultiHeadAttentionBwdOp._LEGACY_KERNEL_MAP_KEYS.intersection(kernel_map)
        if legacy_keys:
            keys = ", ".join(sorted(legacy_keys))
            raise ValueError(
                "MultiHeadAttentionBwdOp delegates to GroupedQueryAttentionBwdOp; "
                f"legacy MHA backward kernel_map keys are not compatible: {keys}. "
                "Use gqa_bwd_* keys with kernels that implement the GQA backward ABI."
            )
        return dict(kernel_map)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run MHA backward on BSHD-layout inputs; returns ``(dq, dk, dv)``.

        Ported from GPU ``GroupedQueryAttentionBwdOp.forward`` with the
        Hopper branch (``dq.to(dtype)``) replaced by the unconditional
        postprocess call.
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
        ``tileops.perf.formulas.mha_bwd_roofline``: ``10 * B * H * S * S * D``
        FLOPs (halved when causal) for the backward score/output work;
        q/k/v/o/do read + dq/dk/dv written = ``7 * B * H * S * D`` elements.
        """
        elem_bytes = self.dtype.itemsize
        flops = 10 * self.batch * self.heads * self.seq_len * self.seq_len * self.dim
        if self.is_causal:
            flops //= 2
        nbytes = self.batch * 7 * self.heads * self.seq_len * self.dim * elem_bytes
        return int(flops), int(nbytes)
