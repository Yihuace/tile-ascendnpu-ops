"""Expm1FwdOp -- elementwise exp(x) - 1 (NPU-adapted).

Ported from the GPU ``Expm1FwdOp(UnaryOp)`` in
``tileops/ops/elementwise/math_unary.py``.  It is a single-input,
single-output unary elementwise math op: the input is flattened to a 1-D
vector of ``N_total`` elements, the kernel is dispatched, and the output is
reshaped back to the original shape.  Output dtype is ``same_as(input)``.

Adaptation from GPU (TileOPs) to NPU:

- O1: ``input.is_cuda`` -> ``backend.is_device_tensor(input)``.
- O3: ``tune`` parameter removed; kernel constructor takes
      ``(N_total, dtype, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: the ``torch.compile`` custom-op registry (``_OP_REGISTRY`` /
      ``_wrapped`` / ``compile_boundary.register_instance``) is removed —
      NPU ops are dispatched through the ``forward``/``_eager_forward``
      path directly.
- O6: Op-specific flow (flatten -> kernel -> reshape, roofline) preserved.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.expm1 import Expm1FwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["Expm1FwdOp"]


class Expm1FwdOp(Op):
    """Element-wise exp(x) - 1.

    Single-input, single-output unary math op.  The input is flattened to a
    1-D vector of ``N_total`` elements, the kernel is dispatched, and the
    output is reshaped back to the original shape.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "expm1"
    kernel_cls = Expm1FwdKernel
    # Manifest roofline: flops = "2 * N" (1 exp + 1 sub per element).
    FLOPS_PER_ELEM = 2

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.N_total = N_total
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(input)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](N_total, dtype)

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
