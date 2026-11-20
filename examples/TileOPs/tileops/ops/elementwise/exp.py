"""ExpFwdOp -- elementwise exponential (NPU-adapted).

Adaptation from GPU (TileOPs) to NPU: O1/O3/O4/O5/O6 as per the elementwise
op adaptation set.

Dtype scope: float-only (the GPU manifest declares only float dtypes here).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.exp import ExpFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["ExpFwdOp"]


class ExpFwdOp(Op):
    """Element-wise exponential: y = exp(x).

    Single-input, single-output unary op: flatten -> kernel -> reshape.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        inplace: When True, copy the result back into ``input``.
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "exp"
    kernel_cls = ExpFwdKernel
    # Manifest: flops = "N".
    FLOPS_PER_ELEM = 1

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        inplace: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.N_total = N_total
        self.dtype = dtype
        self.inplace = inplace
        self.output_dtype = dtype  # same_as(input)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](N_total, dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    def _validate_input(self, input: torch.Tensor) -> None:
        backend = get_device_backend()
        if not backend.is_device_tensor(input):
            raise ValueError(f"input must be a {backend.name} tensor, got device {input.device}")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if input.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        flat = input.contiguous().reshape(-1)
        return self.kernel(flat).reshape(orig_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._validate_input(input)
        if self.inplace:
            result = self._eager_forward(input)
            input.copy_(result.reshape(input.shape))
            return input
        return self._eager_forward(input)

    @property
    def total_memory(self) -> float:
        """Read x + write y."""
        return self.N_total * (self.dtype.itemsize + self.output_dtype.itemsize)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)``; mirrors the manifest roofline."""
        return self.FLOPS_PER_ELEM * self.N_total, int(self.total_memory)
