"""MoE no-pad 3WG grouped GEMM with fused activation op (NPU).

Ported from GPU TileOPs
``tileops/ops/moe/routed_expert/moe_grouped_gemm_nopad_fused_act.py``
(``MoeGroupedGemmNopad3WGFusedActFwdOp``) to the NPU backend.

Computes ``A[numel, K] @ B[E, 2*ffn, K]^T -> act(gate) * up -> C[numel, ffn]``
where the gate/up projections share the same ``A`` rows and are fused with the
activation in the GEMM epilogue.

Adaptation from GPU (TileOPs) to NPU:

- O3: ``tune`` parameter removed; the kernel constructor takes
  ``(numel, num_experts, ffn, k, dtype, activation, config=None,
  device_index=None)`` and ``init_config(config)`` takes no ``tune`` argument.
- O6: the op-specific flow (constructor signature, kernel dispatch, forward
  pass-through) is preserved unchanged -- the GPU op had no ``is_cuda`` check
  or hard-coded ``device="cuda"``, so O1/O2 do not apply.
- ``eval_roofline()``: the GPU op has no standalone manifest entry, so the
  roofline is inferred from the fused gate/up GEMM work (device-agnostic):
  ``flops = 4 * numel * ffn * k`` (gate + up, multiply + add each);
  ``bytes = (numel * k + num_experts * 2 * ffn * k + numel * ffn) *
  elem_bytes + 2 * num_experts * 4`` (a read + b read + c write + int32
  metadata).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.moe_grouped_gemm_nopad_3wg_fused_act import (
    MoeGroupedGemmPersistent3WGFusedActKernel,
)
from tileops.ops.op_base import Op

__all__ = ["MoeGroupedGemmNopad3WGFusedActFwdOp"]


class MoeGroupedGemmNopad3WGFusedActFwdOp(Op):
    """Fused gate/up grouped GEMM with activation (no-pad, warp-specialized 3WG).

    Wraps :class:`MoeGroupedGemmPersistent3WGFusedActKernel`: the gate (first
    ``ffn`` rows of each expert weight) and up (next ``ffn`` rows) projections
    are computed and fused with the activation in the GEMM epilogue, so the
    ``[numel, 2*ffn]`` gate_up intermediate never reaches global memory.

    Args:
        numel: T * top_k tight row count.
        num_experts: Number of local experts E.
        ffn: FFN width (output column count); B has 2*ffn rows (gate||up).
        k: Hidden size K.
        dtype: Activation/weight dtype (bf16 or fp16).
        activation: "silu_and_mul" or "gelu_and_mul".
        kernel_map: Optional kernel override.

    Example:
        >>> op = MoeGroupedGemmNopad3WGFusedActFwdOp(
        ...     numel=16384, num_experts=256, ffn=768, k=2048,
        ...     dtype=torch.bfloat16, activation="silu_and_mul")
        >>> C = op(A, B, true_sizes, true_offsets)  # [numel, ffn]
    """

    def __init__(
        self,
        numel: int,
        num_experts: int,
        ffn: int,
        k: int,
        dtype: torch.dtype = torch.bfloat16,
        activation: str = "silu_and_mul",
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.numel = numel
        self.num_experts = num_experts
        self.ffn = ffn
        self.k = k
        self.dtype = dtype
        self.activation = activation

        self.dispatch_kernel(kernel_map)
        # O3: ``tune`` removed; the kernel constructor uses heuristic config.
        self.kernel = self.kernel_map["moe_grouped_gemm_fused_act_kernel"](
            numel, num_experts, ffn, k, dtype=dtype, activation=activation
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "moe_grouped_gemm_fused_act_kernel": MoeGroupedGemmPersistent3WGFusedActKernel,
        }

    def forward(
        self,
        a: torch.Tensor,             # [numel, K]
        b: torch.Tensor,             # [num_experts, 2*ffn, K]
        true_sizes: torch.Tensor,    # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run the fused gate_up GEMM + activation.

        Args:
            a: [numel, K] tight permuted activations.
            b: [num_experts, 2*ffn, K] gate||up expert weights.
            true_sizes: [E] int32 token count per expert.
            true_offsets: [E] int32 tight start offset per expert in a.

        Returns:
            C: [numel, ffn] activated gate_up output.
        """
        return self.kernel(a, b, true_sizes, true_offsets)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inferred from the fused gate/up GEMM (device-agnostic):
        ``flops = 4 * numel * ffn * k`` (gate + up, multiply + add each);
        ``bytes = (numel * k + num_experts * 2 * ffn * k + numel * ffn) *
        elem_bytes + 2 * num_experts * 4``.
        """
        numel = int(self.numel)
        num_experts = int(self.num_experts)
        ffn = int(self.ffn)
        k = int(self.k)
        elem_bytes = self.dtype.itemsize
        flops = 4 * numel * ffn * k
        nbytes = (
            (numel * k + num_experts * 2 * ffn * k + numel * ffn) * elem_bytes
            + 2 * num_experts * 4
        )
        return int(flops), int(nbytes)
