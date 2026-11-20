"""MoE unpermute op (cutlass path): scatter-add padded expert outputs back to token order.

Ported from GPU TileOPs ``tileops/ops/moe/routed_expert/unpermute.py`` to the
NPU backend.

Adaptation from GPU (TileOPs) to NPU:

- O1: the device check is routed through :func:`get_device_backend` inside the
  kernel's ``forward`` (``backend.is_device_tensor``), replacing the CUDA
  ``mm2_pad.is_cuda`` assertion; this op delegates input validation to the
  kernel as the GPU op did.
- O2: no hard-coded ``device="cuda"``; output allocation uses the input's device.
- O3: no ``tune`` parameter existed in the GPU op, so nothing to remove; the
  kernel constructor is called with the GPU positional argument list (no
  ``tune``).
- O4: ``from ...op_base import Op`` becomes the NPU package import
  ``from tileops.ops.op_base import Op``.
- O5: the GPU op did not use ``compile_boundary`` (nothing to remove).
- O6: the op-specific flow (constructor signature, ``default_kernel_map``,
  dispatch, ``forward`` pass-through) is preserved unchanged.

``eval_roofline()`` implements the GPU manifest-declared roofline inline
(``flops = 2*T*K*H``, ``bytes = (T*K*H + T*H) * elem_bytes + T*K*4 + T*K*4``)
-- roofline arithmetic is device-agnostic.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.moe_unpermute import MoeUnpermuteKernel
from tileops.ops.op_base import Op

__all__ = ["MoeUnpermuteFwdOp"]


class MoeUnpermuteFwdOp(Op):
    """Scatter padded expert outputs back to original token order with weighted reduction.

    Args:
        total_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        hidden_size: Hidden dimension H.
        dtype: Data type of mm2_pad and output (bf16 or fp16).
        padded_batch_sum: Size of the padded mm2_pad buffer (first dim of mm2_pad).
            Must be >= T*K. When used with MoePermuteOp, pass the padded_batch_sum
            value returned by the kernel (T*K + E*block_m upper bound).
            Defaults to total_tokens * top_k for standalone testing only -- do NOT
            use the default when mm2_pad comes from MoePermuteOp, as the padded
            buffer will be larger and the kernel will index out of bounds.
        kernel_map: Optional kernel override dict.
        routed_scaling_factor: Scalar applied to the reduced output, folded into
            the unpermute kernel. Defaults to 1.0 (no scaling).

    Example:
        >>> op = MoeUnpermuteFwdOp(total_tokens=4, top_k=2, hidden_size=128, padded_batch_sum=512)
        >>> output = op(mm2_pad, fwd_idx, topk_weights)
    """

    def __init__(
        self,
        total_tokens: int,
        top_k: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        padded_batch_sum: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.padded_batch_sum = padded_batch_sum if padded_batch_sum is not None else total_tokens * top_k
        self.routed_scaling_factor = routed_scaling_factor

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["unpermute_kernel"](
            total_tokens, top_k, hidden_size, self.padded_batch_sum,
            scaling=routed_scaling_factor, dtype=dtype,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"unpermute_kernel": MoeUnpermuteKernel}

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op (manifest formula, verbatim)."""
        T = self.total_tokens
        K = self.top_k
        H = self.hidden_size
        elem_bytes = self.dtype.itemsize
        flops = 2 * T * K * H
        nbytes = (
            (T * K * H + T * H) * elem_bytes
            + T * K * 4
            + T * K * 4
        )
        return int(flops), int(nbytes)

    def forward(
        self,
        mm2_pad: torch.Tensor,
        fwd_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run moe_unpermute.

        Args:
            mm2_pad: [padded_batch_sum, H] bf16/fp16 down-proj output (padded layout).
            fwd_idx: [T*K] int32 forward mapping: flat_idx -> padded slot.
            topk_weights: [T, K] float32 routing weights.
            out: optional [T, H] output buffer to write into; allocated internally
                if omitted.

        Returns:
            output: [T, H] bf16/fp16 (``out`` if provided).
        """
        return self.kernel(mm2_pad, fwd_idx, topk_weights, out=out)
