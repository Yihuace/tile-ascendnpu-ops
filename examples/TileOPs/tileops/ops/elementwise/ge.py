"""GeFwdOp -- element-wise greater-equal with broadcast, bool output (NPU-adapted).

Conforms to ``torch.ge``: ``y = (input >= other)``.  The output is always
``torch.bool``; the two operands must share the op dtype and broadcast
together (``output.shape == broadcast_shapes(input.shape, other.shape)``).

Adaptation from GPU (TileOPs) to NPU, per docs/gpu_to_npu_adaptation.md §3.4:

- O1: ``input.is_cuda and other.is_cuda`` device check ->
  ``backend.is_device_tensor(...)`` for both operands.
- O3 (E13): ``tune`` parameter removed; the kernel constructor takes
  ``(a_shape, b_shape, dtype, config=None)``.
- O4: ``from tileops.device import get_device_backend`` (was
  ``from tileops.utils import get_sm_version``).
- O6: op-specific flow preserved unchanged -- PyTorch broadcast output
  shape, flatten to 1-D, kernel dispatch, reshape back to the broadcast
  shape.  The GPU ``_BoolOutputBinaryOp`` bool fast path is preserved:
  ``torch.bool`` operands are viewed as ``uint8`` before dispatch
  (TileLang/Torch bool storage requires the uint8 round-trip), and the
  bool-storage kernel output is viewed back as ``torch.bool``.
- ``eval_roofline()`` mirrors the GPU roofline func
  ``tileops.perf.formulas.ge_fwd_roofline`` (``_binary_broadcast_roofline``
  with ``flops_per_elem=1, bool_output=True``):
  ``flops = 1 * N_total``;
  ``bytes = (a_numel + b_numel) * elem_bytes + N_total * 1``
  (2 reads at the input element size + 1 write of a 1-byte bool output).

The GPU ``_BoolOutputBinaryOp`` dispatched the ``torch.bool`` fast path
through a distinct ``ge_bool_storage`` kernel key; that structure is kept
here so the bool fast path stays explicit and mirrors the GPU op layout.
"""

from __future__ import annotations

from math import prod
from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.ge import (
    GeBoolStorageFwdKernel,
    GeFwdKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["GeFwdOp"]

# Manifest dtype contract for the comparison family (GPU ``_BINARY_FULL_DTYPES``).
_GE_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


class GeFwdOp(Op):
    """Element-wise greater-equal with broadcast: ``y = (input >= other)``.

    Conforms to ``torch.ge``. Both operands share the op dtype; the output
    is always ``torch.bool`` and its shape is the PyTorch broadcast of the
    two operand shapes
    (``output.shape == broadcast_shapes(input.shape, other.shape)``).

    Args:
        a_shape: Shape of input ``input``.
        b_shape: Shape of input ``other``.
        dtype: Torch dtype (one of the manifest comparison dtypes).
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "ge"
    _SUPPORTED_DTYPES = _GE_DTYPES
    kernel_cls = GeFwdKernel
    bool_storage_kernel_cls = GeBoolStorageFwdKernel

    def __init__(
        self,
        *,
        a_shape: tuple,
        b_shape: tuple,
        dtype: torch.dtype,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )
        self.dtype = dtype
        self.a_shape = tuple(a_shape)
        self.b_shape = tuple(b_shape)
        # O6: PyTorch broadcast output shape (0-dim normalized to (1,)).
        out_shape = torch.broadcast_shapes(
            self.a_shape or (1,), self.b_shape or (1,),
        )
        self.out_shape = tuple(out_shape)
        self._out_shape_list = list(self.out_shape)  # cached for dispatch
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.a_numel = prod(self.a_shape) if self.a_shape else 1
        self.b_numel = prod(self.b_shape) if self.b_shape else 1
        # GPU parity: bool tensors take the uint8-backed storage fast path.
        self._bool_storage = (
            self.dtype == torch.bool and self.bool_storage_kernel_cls is not None
        )
        self.dispatch_kernel(kernel_map)
        self.kernel = self._build_kernel_instance()

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernel_map = {self._op_name: self.kernel_cls}
        if self.bool_storage_kernel_cls is not None:
            kernel_map[f"{self._op_name}_bool_storage"] = self.bool_storage_kernel_cls
        return kernel_map

    def _build_kernel_instance(self) -> Kernel:
        if self._bool_storage:
            return self.kernel_map[f"{self._op_name}_bool_storage"](
                self.a_shape, self.b_shape, torch.uint8,
            )
        return self.kernel_map[self._op_name](
            self.a_shape, self.b_shape, self.dtype,
        )

    def _eager_forward(
        self,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        if self._bool_storage:
            result = self.kernel(
                input.contiguous().view(-1).view(torch.uint8),
                other.contiguous().view(-1).view(torch.uint8),
            )
            return result.view(torch.bool).reshape(self.out_shape)
        result = self.kernel(
            input.contiguous().view(-1), other.contiguous().view(-1),
        ).reshape(self.out_shape)
        # GPU ``_BoolOutputBinaryOp`` parity: the kernel emits bool, but fall
        # back to a cast if an implementation returns a wider storage dtype.
        if result.dtype is not torch.bool:
            return result.to(torch.bool)
        return result

    def forward(
        self,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        # O1: device check via the device backend.
        backend = get_device_backend()
        if not (
            backend.is_device_tensor(input) and backend.is_device_tensor(other)
        ):
            raise ValueError(f"Inputs must be {backend.name} tensors")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if other.dtype != self.dtype:
            raise ValueError(f"Expected other.dtype {self.dtype}, got {other.dtype}")
        if input.numel() != self.a_numel:
            raise ValueError(
                f"Expected input to have {self.a_numel} elements, got {input.numel()}"
            )
        if other.numel() != self.b_numel:
            raise ValueError(
                f"Expected other to have {self.b_numel} elements, got {other.numel()}"
            )
        return self._eager_forward(input, other)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Mirrors ``tileops.perf.formulas.ge_fwd_roofline``
        (``_binary_broadcast_roofline(flops_per_elem=1, bool_output=True)``):
        ``flops = 1 * N_total`` (one comparison per output element);
        ``bytes = (a_numel + b_numel) * elem_bytes + N_total * 1``
        (read both operands at the input element size, write the broadcast
        bool output at 1 byte per element).
        """
        n_total = int(self.N_total)
        elem_bytes = self.dtype.itemsize
        return n_total, (self.a_numel + self.b_numel) * elem_bytes + n_total
