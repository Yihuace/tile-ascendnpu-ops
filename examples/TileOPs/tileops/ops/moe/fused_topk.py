"""MoE fused top-k routing op (NPU-adapted).

Ported from GPU TileOPs ``tileops/ops/moe/fused_topk.py`` to the NPU backend.

Adaptation from GPU (TileOPs) to NPU:

- O1: the device checks are routed through :func:`get_device_backend`
  (``backend.is_device_tensor``), replacing the CUDA ``is_cuda`` checks on
  ``gating_output`` / ``correction_bias``.
- O2: no hard-coded ``device="cuda"`` -- the output buffers are allocated on the
  input's device (inside the kernel's ``forward``).
- O3: no ``tune`` parameter existed in the GPU op, so nothing to remove; the
  kernel constructor is called without ``tune``.
- O4: ``from ..op_base import Op`` becomes the NPU package import
  ``from tileops.ops.op_base import Op``.
- O5: the GPU op did not use ``compile_boundary`` (nothing to remove).
- O6: the op-specific flow (constructor signature, ``default_kernel_map``,
  per-shape kernel cache, ``_get_kernel`` dispatch, ``forward`` validation and
  pass-through) is preserved unchanged.

``eval_roofline()`` implements the manifest-declared roofline inline
(``flops = T*E*2*(1+K)``, ``bytes = T*E*elem_bytes + 2*T*K*4``) -- roofline
arithmetic is device-agnostic.
"""

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.fused_topk import FusedTopKKernel
from tileops.ops.op_base import Op

__all__ = ["FusedTopKOp"]


class FusedTopKOp(Op):
    """MoE top-k routing operator.

    Applies scoring (softmax or sigmoid) to router logits and selects the
    top-k experts per token.

    Args:
        num_tokens: Optional committed number of input tokens T. Preferred
            API infers it from ``gating_output.shape[0]``.
        num_experts: Optional committed number of experts E. Preferred API
            infers it from ``gating_output.shape[1]``.
        top_k: Number of experts to select per token K.
        scoring_func: "softmax" (Qwen3/Qwen2) or "sigmoid" (DeepSeek-V3/GLM-4/Kimi K2).
        renormalize: If True, normalize top-k weights to sum to 1.
        with_correction_bias: If True, forward() accepts a per-expert correction_bias
            tensor.  Bias is added to sigmoid scores for selection only; output
            weights remain the original sigmoid scores.  Requires scoring_func="sigmoid".
        kernel_map: Optional kernel map override.
        config: Optional kernel config dict.

    Example:
        >>> op = FusedTopKOp(top_k=8)
        >>> topk_weights, topk_ids = op(gating_output)
        >>> # topk_weights: [512, 8] float32
        >>> # topk_ids:     [512, 8] int32
    """

    def __init__(
        self,
        num_tokens: Optional[int] = None,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        with_correction_bias: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        config: Optional[dict] = None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self._committed_num_tokens = num_tokens
        self._committed_num_experts = num_experts
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias
        if with_correction_bias and scoring_func != "sigmoid":
            raise ValueError("with_correction_bias=True requires scoring_func='sigmoid'")

        self.dispatch_kernel(kernel_map)
        self.config = config
        self._kernel_cache: Dict[tuple[int, int, int, int | None], Kernel] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fused_topk_kernel": FusedTopKKernel}

    def _get_kernel(
        self, num_tokens: int, num_experts: int, top_k: int,
        device_index: int | None,
    ) -> Kernel:
        key = (num_tokens, num_experts, top_k, device_index)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["fused_topk_kernel"](
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                scoring_func=self.scoring_func,
                renormalize=self.renormalize,
                with_correction_bias=self.with_correction_bias,
                config=self.config,
            )
        return self._kernel_cache[key]

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op (manifest formula, verbatim)."""
        T = self.num_tokens
        E = self.num_experts
        K = self.top_k
        # Defensive: eval_roofline may run before the first forward() on a
        # shape-inferring op; fall back to the smallest float element width.
        elem_bytes = self.dtype.itemsize if self.dtype is not None else 2
        flops = T * E * 2 * (1 + K)
        nbytes = T * E * elem_bytes + 2 * T * K * 4
        return int(flops), int(nbytes)

    def forward(
        self,
        gating_output: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run top-k routing.

        Args:
            gating_output: [T, E] router logits (bf16, fp16, or float32).
            correction_bias: [E] float32 per-expert bias.  Required when
                with_correction_bias=True; must be None otherwise.

        Returns:
            topk_weights: [T, K] float32.
            topk_ids:     [T, K] int32.
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(gating_output):
            raise ValueError("gating_output must be an NPU tensor")
        if correction_bias is not None and not backend.is_device_tensor(correction_bias):
            raise ValueError("correction_bias must be an NPU tensor")
        if correction_bias is not None and gating_output.device != correction_bias.device:
            raise ValueError(
                f"Expected gating_output and correction_bias to be on the same device, "
                f"got {gating_output.device} and {correction_bias.device}"
            )
        if gating_output.ndim != 2:
            raise ValueError(
                f"Expected gating_output to be 2D [T, E], got {gating_output.ndim}D"
            )
        if gating_output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "Expected gating_output.dtype to be torch.float16, "
                f"torch.bfloat16, or torch.float32, got {gating_output.dtype}"
            )
        num_tokens, num_experts = gating_output.shape
        if (
            self._committed_num_tokens is not None
            and num_tokens != self._committed_num_tokens
        ):
            raise ValueError(
                f"Expected num_tokens={self._committed_num_tokens}, got {num_tokens}"
            )
        if (
            self._committed_num_experts is not None
            and num_experts != self._committed_num_experts
        ):
            raise ValueError(
                f"Expected num_experts={self._committed_num_experts}, got {num_experts}"
            )
        if self.top_k is None:
            raise ValueError("top_k must be provided at construction time")
        if self.top_k > num_experts:
            raise ValueError(
                f"top_k={self.top_k} cannot exceed num_experts={num_experts}"
            )
        if self.with_correction_bias:
            if correction_bias is None:
                raise ValueError("correction_bias is required when with_correction_bias=True")
            if correction_bias.shape != (num_experts,):
                raise ValueError(
                    f"Expected correction_bias shape {(num_experts,)}, "
                    f"got {tuple(correction_bias.shape)}"
                )
            if correction_bias.dtype != torch.float32:
                raise ValueError(
                    f"Expected correction_bias.dtype torch.float32, got {correction_bias.dtype}"
                )
        elif correction_bias is not None:
            raise ValueError("correction_bias must be None when with_correction_bias=False")

        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.dtype = gating_output.dtype
        kernel = self._get_kernel(
            num_tokens, num_experts, self.top_k, gating_output.device.index,
        )
        return kernel(gating_output, correction_bias)
