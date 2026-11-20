"""GeluTanhAndMulFwdOp -- fused gated tanh-GELU (NPU-adapted).

Adaptation from GPU (TileOPs) to NPU, per docs/gpu_to_npu_adaptation.md §2/§3:

- O1: device check uses ``backend.is_device_tensor(x)`` instead of
      ``x.is_cuda``.
- O2/O4: ``from tileops.device import get_device_backend`` (device name comes
      from the backend, not a hard-coded ``"cuda"``).
- O3: ``tune`` parameter removed; the kernel constructor takes
      ``(M, N, dtype, config=None)``.
- O5: ``from .compile_boundary import register_instance`` removed.
- O6: the GPU ``FusedGatedOp`` flow is preserved: one ``(M, 2*N)`` input is
      handed to the gated kernel (gate = ``x[:, :N]``, value = ``x[:, N:]``),
      which returns the ``(M, N)`` output.

The GPU ``GeluTanhAndMulFwdOp`` extends ``FusedGatedOp`` and reads shapes
lazily from the first call.  The NPU port receives ``M`` / ``N`` in its
constructor (the NPU benchmark pipeline reconstructs Ops from their constructor
arguments -- see ``tileops.benchmark.msprof``), which also lets the msprof
runner rebuild the op in a subprocess.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.gelu_tanh_and_mul import GeluTanhAndMulFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["GeluTanhAndMulFwdOp"]

# Per output element: gelu_tanh(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) *
# (g + 0.044715 * g^3))).  cube(2 muls) + mul-coeff(1) + add(1) + mul-sqrt(1)
# + tanh(1) + add(1) + mul-half(1) + mul-by-gate(1) = 9, plus the final
# mul-by-value = 1 -> 10 FLOPs.  Manifest formula: ``10 * M * N``.
_FLOPS_PER_OUTPUT = 10


class GeluTanhAndMulFwdOp(Op):
    """Fused gated tanh-GELU: ``y = gelu_tanh(gate) * value``.

    The input ``x`` has shape ``(M, 2 * N)``: ``gate = x[:, :N]`` and
    ``value = x[:, N:]``.  The output has shape ``(M, N)`` and matches
    ``torch.nn.functional.gelu(gate, approximate="tanh") * value`` (tanh-based
    GELU approximation).  Only a float dtype (float16 / bfloat16 / float32) is
    admitted, per the manifest.

    Args:
        M: Number of rows of ``x`` (and of the output).
        N: Output width; ``x`` has ``2 * N`` columns.
        dtype: Torch dtype (float16, bfloat16, or float32).
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "gelu_tanh_and_mul"

    # Manifest declares ``float16 | bfloat16 | float32``.
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"GeluTanhAndMulFwdOp does not support dtype {dtype}. Supported: [{names}]"
            )
        # Keep constructor-name attributes so the msprof subprocess can
        # reconstruct this Op through ``_extract_op_init_args`` (which matches
        # ``__init__`` parameter names to instance attributes).
        self.M = int(M)
        self.N = int(N)
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(input)
        self.x_shape = (self.M, 2 * self.N)
        self.out_shape = (self.M, self.N)
        self.N_total = self.M * self.N
        self.dispatch_kernel(kernel_map)
        # O3: ``tune`` removed; the kernel constructor takes (M, N, dtype).
        self.kernel = self.kernel_map[self._op_name](self.M, self.N, dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: GeluTanhAndMulFwdKernel}

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate the fused gated input tensor.

        NPU adaptation (O1): the device check uses ``backend.is_device_tensor``
        instead of ``x.is_cuda``; the shape check uses the device backend name
        in the error message (O2).
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(x):
            raise ValueError(f"x must be a {backend.name} tensor, got device {x.device}")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if tuple(x.shape) != self.x_shape:
            raise ValueError(f"Expected x.shape {self.x_shape}, got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fused gated tanh-GELU on ``x`` of shape ``(M, 2 * N)``."""
        x = x.contiguous()
        self._validate_input(x)
        return self.kernel(x)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Mirrors the GPU manifest roofline: ``flops = 10 * M * N`` (one output
        element each).  Memory: read ``x`` (``2 * M * N`` elements) + write
        ``y`` (``M * N`` elements) -> ``bytes = 3 * M * N * elem_bytes``.
        """
        elem_bytes = self.dtype.itemsize
        flops = _FLOPS_PER_OUTPUT * self.N_total
        mem_bytes = 3 * self.N_total * elem_bytes
        return flops, mem_bytes
