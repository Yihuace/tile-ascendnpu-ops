"""IsinfFwdOp -- elementwise float infinity predicate (bool output) (NPU-adapted).

Adaptation from GPU (TileOPs) to NPU, per docs/gpu_to_npu_adaptation.md
section 3:

- O1: ``input.is_cuda`` -> ``backend.is_device_tensor(input)``.
- O3: ``tune`` parameter removed; the kernel constructor takes
      ``(N_total, dtype, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: ``from .compile_boundary import register_instance`` removed.
- O6: Op-specific flow (flatten -> kernel dispatch -> reshape, roofline)
      preserved; the element count and dtype arrive with construction (the NPU
      benchmark pipeline reconstructs Ops from their constructor arguments).

The GPU ``IsinfFwdOp`` extends a unary elementwise base and restricts itself to
the float element types of ``torch.isinf`` (its ``isinf_exact`` fallback for the
integer / bool inputs is not migrated).  The input is flattened to 1-D, the
predicate kernel is dispatched, and the bool output is reshaped back to the
input shape.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.isinf import IsinfFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["IsinfFwdOp"]


class IsinfFwdOp(Op):
    """Element-wise infinity predicate: ``y = isinf(x)`` (bool output).

    Single-input, single-output unary op.  The input is flattened to a 1-D
    vector of ``N_total`` elements, the kernel is dispatched, and the bool
    output is reshaped back to the original shape.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Input data type (float16, bfloat16, or float32).
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "isinf"
    kernel_cls = IsinfFwdKernel

    # Manifest: flops = "prod(S)"; one elementwise predicate per element.
    FLOPS_PER_ELEM = 1

    # Manifest float subset (the GPU union also lists int / bool).
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"IsinfFwdOp does not support dtype {dtype}. Supported: [{names}]"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.output_dtype = torch.bool  # predicate output
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
        """Read x (float) + write y (bool, one byte per element)."""
        return self.N_total * (self.dtype.itemsize + self.output_dtype.itemsize)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this unary elementwise op instance.

        Mirrors the manifest roofline: ``flops = 1 * N`` and
        ``bytes = N * (input_elem_bytes + 1)`` (a bool result is one byte).
        """
        return self.FLOPS_PER_ELEM * self.N_total, int(self.total_memory)
