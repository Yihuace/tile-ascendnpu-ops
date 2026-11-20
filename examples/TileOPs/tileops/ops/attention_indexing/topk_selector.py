"""TopkSelectorOp -- exact top-k indices along the last dimension (NPU-adapted).

Adaptation from GPU (TileOPs) to NPU:

- O1: ``input.is_cuda`` -> ``backend.is_device_tensor(input)``.
- O3: the ``tune`` parameter is removed; the kernel constructor takes
      ``(M, N, k, rounds, tile, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O6: op-specific flow (2-D input -> kernel -> int32 indices) preserved.

The algorithm itself was redesigned, not ported: the GPU kernel is a CUDA-thread
radix-select that needs ``threads``/shared-memory atomics/CUDA barriers, none of
which exist on npuir.  See the kernel wrapper docstring for the NPU design and for
the measurements that force it.

Dtype scope: float32 input (the kernel's counting path runs in the fp32 domain);
the returned indices are int32, matching the manifest.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.attention_indexing.topk_selector.topk_selector import (
    ROUNDS,
    TopkSelectorKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["TopkSelectorOp"]


class TopkSelectorOp(Op):
    """Exact top-k: given ``x`` of shape ``(M, N)`` return ``(M, k)`` indices.

    Args:
        M: Number of rows.
        N: Width of each row (the dimension selected along).
        k: How many indices to return per row.
        rounds: Bisection rounds (host side).
        tile: Stream each row in pieces of this width; ``None`` = whole row.
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "topk_selector"
    kernel_cls = TopkSelectorKernel

    def __init__(
        self,
        M: int,
        N: int,
        k: int,
        rounds: int = ROUNDS,
        tile: Optional[int] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.M = M
        self.N = N
        self.k = k
        self.rounds = rounds
        self.tile = tile
        self.output_dtype = torch.int32
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            M, N, k, rounds=rounds, tile=tile
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    def _validate_input(self, input: torch.Tensor) -> None:
        backend = get_device_backend()
        if not backend.is_device_tensor(input):
            raise ValueError(
                f"input must be a {backend.name} tensor, got device {input.device}"
            )
        if input.dim() != 2:
            raise ValueError(f"expected a 2-D (M, N) input, got shape {tuple(input.shape)}")
        if tuple(input.shape) != (self.M, self.N):
            raise ValueError(
                f"Expected input shape {(self.M, self.N)}, got {tuple(input.shape)}"
            )

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.kernel(input.contiguous())

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._validate_input(input)
        return self._eager_forward(input)

    @property
    def total_memory(self) -> float:
        """Read x (fp32) + write the k indices (int32) per row."""
        return self.M * self.N * 4 + self.M * self.k * 4

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)``.

        Counting dominates here: the host bisection issues ``rounds`` counting
        passes, each comparing every element once.
        """
        flops = self.rounds * self.M * self.N
        return flops, int(self.total_memory)
