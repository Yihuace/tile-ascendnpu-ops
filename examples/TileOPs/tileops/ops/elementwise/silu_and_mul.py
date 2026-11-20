"""SiluAndMulFwdOp -- fused SiLU-and-multiply (NPU-adapted).

Implements ``y = silu(gate) * value`` where the single input ``x`` of shape
``(M, 2*N)`` packs ``gate = x[:, :N]`` and ``value = x[:, N:]``; the output
has shape ``(M, N)``.  This is the activation front-end of a SwiGLU FFN / MoE
expert (``F.silu(x[..., :N]) * x[..., N:]``).

Adaptation from GPU (TileOPs) to NPU, per docs/gpu_to_npu_adaptation.md §2.5:

- O1: ``x.is_cuda`` device check -> ``backend.is_device_tensor(x)`` via
  :func:`get_device_backend`.
- O3: ``tune`` parameter removed; the kernel constructor takes
  ``(M, N, dtype, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: ``from .compile_boundary import register_instance`` removed, along with
  the GPU ``_OP_REGISTRY`` / ``_wrapped`` custom_op dispatch boundary.
- O6: the op-specific flow (validation, gate/value split, dispatch, roofline)
  is preserved unchanged -- only device-specific calls are adapted.

``FusedGatedOp`` is inlined here (the NPU project does not ship the GPU
``tileops/ops/elementwise/_base.py`` module); ``SiluAndMulFwdOp`` is the only
concrete leaf in this scaffold.  The sibling ``GeluAndMulFwdOp`` /
``GeluTanhAndMulFwdOp`` leaves are not migrated.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.silu_and_mul import SiluAndMulFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["FusedGatedOp", "SiluAndMulFwdOp"]


class FusedGatedOp(Op):
    """Template base class for fused gated elementwise ops (NPU-adapted).

    Input: ``x`` of shape ``(M, 2*N)``; ``gate = x[:, :N]``,
    ``value = x[:, N:]``.  Output: ``y = activation(gate) * value`` of shape
    ``(M, N)``.  Subclasses set ``kernel_cls`` and ``_op_name``.

    Args:
        M: Optional number of rows.  Inferred from ``x`` when omitted.
        N: Optional half column dim (output width).  Inferred from ``x`` when
            omitted.
        dtype: Optional torch dtype.  Inferred from ``x`` when omitted.
        kernel_map: Optional kernel dispatch override.
    """

    kernel_cls: type
    _op_name: str
    FLOPS_PER_ELEM: int = 6

    def __init__(
        self,
        M: Optional[int] = None,
        N: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if (M is None) != (N is None):
            raise ValueError("M and N must be provided together")
        if dtype is not None:
            self._validate_dtype(dtype)
        self._explicit_shape = M is not None and N is not None
        self._explicit_dtype = dtype is not None
        self.M = M
        self.N = N
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(x)
        self.dispatch_kernel(kernel_map)
        self.kernel = None
        self._kernel_key = None
        if M is not None and N is not None and dtype is not None:
            self._ensure_kernel(M, N, dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x (M*2N) + write y (M*N), both at the input element size."""
        if self.M is None or self.N is None or self.dtype is None:
            raise RuntimeError(
                "Fused gated dimensions are available after first forward"
            )
        elem_bytes = self.dtype.itemsize
        return self.M * 2 * self.N * elem_bytes + self.M * self.N * elem_bytes

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this fused gated op instance.

        Mirrors the manifest roofline: ``flops = FLOPS_PER_ELEM * M * N`` and
        ``bytes = 3 * M * N * elem_bytes`` (2 reads + 1 write).
        """
        if self.M is None or self.N is None or self.dtype is None:
            raise RuntimeError(
                "Fused gated roofline is available after first forward"
            )
        flops = self.FLOPS_PER_ELEM * self.M * self.N
        return flops, int(self.total_memory)

    def _validate_dtype(self, dtype: torch.dtype) -> None:
        supported = self.kernel_cls.SUPPORTED_DTYPES
        if supported is not None and dtype not in supported:
            names = ", ".join(str(dt) for dt in supported)
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )

    def _ensure_kernel(self, M: int, N: int, dtype: torch.dtype) -> None:
        self._validate_dtype(dtype)
        key = (M, N, dtype)
        if self._kernel_key == key and self.kernel is not None:
            return
        self.M = M
        self.N = N
        self.dtype = dtype
        self.output_dtype = dtype
        # O3: ``tune`` removed; the kernel constructor takes (M, N, dtype).
        self.kernel = self.kernel_map[self._op_name](M, N, dtype)
        self._kernel_key = key

    def _validate_runtime_input(self, x: torch.Tensor) -> tuple[int, int]:
        """Validate the packed ``(M, 2*N)`` input.

        O1: device check uses ``backend.is_device_tensor`` instead of
        ``x.is_cuda``.
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(x):
            raise ValueError(f"Input must be a {backend.name} tensor")
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D, got {x.ndim}D")
        if x.shape[1] % 2 != 0:
            raise ValueError(f"Expected x.shape[1] to be even, got {x.shape[1]}")
        M = x.shape[0]
        N = x.shape[1] // 2
        if self._explicit_shape and (M, N) != (self.M, self.N):
            raise ValueError(
                f"Expected shape ({self.M}, {2 * self.N}), got {tuple(x.shape)}"
            )
        if self._explicit_dtype and x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        return M, N

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Direct kernel call: validate -> ensure kernel -> dispatch."""
        M, N = self._validate_runtime_input(x)
        self._ensure_kernel(M, N, x.dtype)
        x = x.contiguous()
        return self.kernel(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # O5/E12: no ``_wrapped`` custom_op dispatch; eager path directly.
        return self._eager_forward(x)


class SiluAndMulFwdOp(FusedGatedOp):
    """SiLU-and-Mul: ``y = silu(gate) * value``, ``x = [gate, value]`` (M, 2N)."""

    _op_name = "silu_and_mul"
    kernel_cls = SiluAndMulFwdKernel
    # Manifest: flops = "6 * M * N" (sigmoid 4 + silu mul 1 + value mul 1).
    FLOPS_PER_ELEM = 6
