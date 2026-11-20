"""ClampScalarFwdOp -- scalar-bound clamp (NPU-adapted).

Adaptation from GPU (TileOPs) to NPU:

- O1: ``input.is_cuda`` -> ``backend.is_device_tensor(input)``.
- O3: ``tune`` parameter removed; kernel constructor takes
      ``(N_total, dtype, min_val, max_val, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: ``from .compile_boundary import register_instance`` removed.
- O6: Op-specific flow (flatten -> kernel -> reshape, roofline) preserved.

The scalar bounds are construction-time parameters (``min`` / ``max``), not
tensors, so the Op validates only the single input tensor.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.clamp_scalar import ClampScalarFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["ClampScalarFwdOp"]


class ClampScalarFwdOp(Op):
    """Element-wise scalar-bound clamp: ``y = clamp(x, min, max)``.

    Single-input, single-output elementwise op.  The input is flattened to a
    1-D vector of ``N_total`` elements, the kernel is dispatched, and the output
    is reshaped back to the original shape.  A bound that is ``None`` is not
    applied; at least one of ``min`` / ``max`` must be present.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        min: Lower bound (Number or None).
        max: Upper bound (Number or None); at least one bound is given.
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "clamp_scalar"
    kernel_cls = ClampScalarFwdKernel
    # Manifest: flops = "1 * N".  clamp = compare-and-select; the clamp / min-max
    # family collapses to 1 FLOP per output element regardless of the number of
    # bounds.
    FLOPS_PER_ELEM = 1

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        min: Optional[float] = None,
        max: Optional[float] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if min is None and max is None:
            raise ValueError("present(min) or present(max)")
        self.N_total = N_total
        self.dtype = dtype
        self.min = min
        self.max = max
        self.output_dtype = dtype  # same_as(input)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](N_total, dtype, min, max)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    def _validate_input(self, input: torch.Tensor) -> None:
        """Validate input tensor.

        NPU adaptation (O1): device check uses ``backend.is_device_tensor``
        instead of ``input.is_cuda``.
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(input):
            raise ValueError(f"input must be a {backend.name} tensor, got device {input.device}")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if input.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Direct kernel call: flatten -> kernel -> reshape."""
        orig_shape = input.shape
        flat = input.contiguous().reshape(-1)
        return self.kernel(flat).reshape(orig_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._validate_input(input)
        return self._eager_forward(input)

    @property
    def total_memory(self) -> float:
        """Read x + write y."""
        return self.N_total * (self.dtype.itemsize + self.output_dtype.itemsize)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this unary elementwise op instance.

        Mirrors the manifest roofline: ``flops = FLOPS_PER_ELEM * N`` and
        ``bytes = 2 * N * elem_bytes``.
        """
        return self.FLOPS_PER_ELEM * self.N_total, int(self.total_memory)
