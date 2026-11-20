"""MoE grouped GEMM op (no-pad variant): NT GEMM with precomputed tile scheduling.

Ported from GPU TileOPs ``tileops/ops/moe/routed_expert/moe_grouped_gemm_nopad.py``
to the NPU backend.

Adaptation from GPU (TileOPs) to NPU:

- O3: ``tune`` parameter removed; the kernel constructor takes
  ``(numel, num_experts, n, k, dtype, config=None, device_index=None)`` and
  ``init_config(config)`` takes no ``tune`` argument.
- O6: the op-specific flow (constructor signature, kernel dispatch, forward
  pass-through) is preserved unchanged -- the GPU op had no ``is_cuda`` check
  or hard-coded ``device="cuda"``, so O1/O2 do not apply.
- ``eval_roofline()``: the GPU manifest-declared roofline
  (``flops = 2 * numel * n * k``, ``bytes = (numel * k + num_experts * n * k
  + numel * n) * elem_bytes``) is implemented inline (roofline arithmetic is
  device-agnostic).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel
from tileops.ops.op_base import Op

__all__ = ["MoeGroupedGemmNopadFwdOp"]


class MoeGroupedGemmNopadFwdOp(Op):
    """NT grouped GEMM for MoE without block_m-aligned padding.

    Uses a tile scheduler to map each CTA to its (expert, row_offset) in O(1),
    eliminating the O(E) per-CTA expert scan in standard grouped GEMM.

    Accepts tight A[T*K, K] inputs (no padding between experts) from
    MoePermuteNoPadOp, producing tight C[T*K, N] outputs.

    Args:
        numel: T * top_k, total (token, expert) pairs = tight row count.
        num_experts: Total number of experts E.
        n: Output feature dimension N (e.g. 2*ffn_size or hidden_size).
        k: Input feature dimension K (hidden_size or ffn_size).
        dtype: Activation and weight dtype (bf16 or fp16).
        kernel_map: Optional kernel override dict.

    Example:
        >>> op = MoeGroupedGemmNopadFwdOp(numel=16384, num_experts=256, n=4096, k=2048,
        ...                       dtype=torch.bfloat16)
        >>> C = op(A, B, true_sizes, true_offsets)  # [numel, N]
    """

    def __init__(
        self,
        numel: int,
        num_experts: int,
        n: int,
        k: int,
        dtype: torch.dtype = torch.bfloat16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        # O3: ``tune`` removed; the kernel constructor uses heuristic config.
        self.kernel = self.kernel_map["moe_grouped_gemm_kernel"](
            numel, num_experts, n, k, dtype=dtype
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"moe_grouped_gemm_kernel": MoeGroupedGemmNopadKernel}

    def forward(
        self,
        a: torch.Tensor,             # [numel, K]
        b: torch.Tensor,             # [num_experts, N, K]
        true_sizes: torch.Tensor,    # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run tile-scheduled NT GEMM.

        Args:
            a: [numel, K] tight permuted activations.
            b: [num_experts, N, K] expert weights (NT: B^T applied).
            true_sizes: [E] int32 true token count per expert.
            true_offsets: [E] int32 tight start offset per expert in a.

        Returns:
            C: [numel, N] GEMM output.
        """
        return self.kernel(a, b, true_sizes, true_offsets)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Mirrors the GPU manifest roofline entry (device-agnostic):
        ``flops = 2 * numel * n * k`` (multiply + add per element per expert
        slot); ``bytes = (numel * k + num_experts * n * k + numel * n) *
        elem_bytes`` (a read + b read + c write).
        """
        numel = int(self.numel)
        num_experts = int(self.num_experts)
        n = int(self.n)
        k = int(self.k)
        elem_bytes = self.dtype.itemsize
        flops = 2 * numel * n * k
        nbytes = (numel * k + num_experts * n * k + numel * n) * elem_bytes
        return int(flops), int(nbytes)
