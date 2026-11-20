"""ClampMaxFwdOp -- Tensor-bound upper clamp with broadcast (NPU-adapted).

Conforms to ``torch.clamp_max(input, max: Tensor)``:
``output = min(input, max)`` with PyTorch broadcast semantics
(``output.shape == broadcast_shapes(input.shape, max.shape)``).  This is the
single upper-bound Tensor variant of the clamp family; it is runtime
equivalent to ``torch.clamp(input, min=None, max=max)``.

Adaptation from GPU (TileOPs) to NPU, per docs/gpu_to_npu_adaptation.md §3.4:

- O1: ``input.is_cuda and max.is_cuda`` device check ->
  ``backend.is_device_tensor(...)`` for both operands.
- O3 (E13): ``tune`` parameter removed; the kernel constructor takes
  ``(N_total, dtype, has_min, has_max, config=None)``.
- O4: ``from tileops.device import get_device_backend`` (was
  ``from tileops.utils import get_sm_version``).
- O5 (E12): the GPU ``_OP_REGISTRY`` / ``_wrapped`` custom_op /
  ``compile_boundary.register_instance`` dispatch is removed -- NPU has no
  ``torch.compile`` boundary; ``forward`` calls ``_eager_forward`` directly.
- O6 (E15): op-specific flow preserved unchanged -- broadcast output shape,
  expand + flatten to 1-D, kernel dispatch, reshape back to the broadcast
  shape.  The GPU base class ``_ClampTensorBase._expand_flat`` is inlined
  here (the NPU project ships no ``_ClampTensorBase``).

The GPU op inherited shared infra from ``_ClampTensorBase``; on NPU the
broadcast/flatten helper is a private static method on this class, so the op
is self-contained.
"""

from __future__ import annotations

from math import prod
from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.clamp_max import ClampTensorFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["ClampMaxFwdOp"]

# Manifest dtype contract for the Tensor-bound clamp family (GPU
# ``ParametricUnaryKernel.SUPPORTED_DTYPES`` = ``_FLOAT_DTYPES``).
_CLAMP_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


class ClampMaxFwdOp(Op):
    """Single-bound Tensor upper clamp (``torch.clamp_max``).

    Args:
        input: Shape of the input tensor.
        max: Shape of the upper-bound tensor.
        dtype: Torch dtype (float16, bfloat16, or float32).
        kernel_map: Optional kernel dispatch override.

    Raises:
        ValueError: If *dtype* is outside the supported set.
    """

    _op_name = "clamp_max"
    _SUPPORTED_DTYPES = _CLAMP_DTYPES
    kernel_cls = ClampTensorFwdKernel

    def __init__(
        self,
        input: tuple,
        max: tuple,
        dtype: torch.dtype,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )
        self.input_shape = tuple(input)
        self.max_shape = tuple(max)
        # Aliases matching the ``__init__`` parameter names.  Tooling that
        # reconstructs an op from its constructor args reads
        # ``getattr(op, <param_name>)`` (see
        # ``tileops/benchmark/msprof.py::_extract_op_init_args``); without
        # these, ``input``/``max`` are missing and the msprof profiling
        # script is generated with an incomplete constructor call.
        self.input = self.input_shape
        self.max = self.max_shape
        self.dtype = dtype
        # O6: PyTorch broadcast output shape (0-dim normalized to (1,)).
        self.out_shape = tuple(
            torch.broadcast_shapes(self.input_shape, self.max_shape)
        )
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["clamp_tensor"](
            self.N_total,
            dtype,
            has_min=False,
            has_max=True,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"clamp_tensor": self.kernel_cls}

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Broadcast *t* to *target_shape* and return a contiguous 1-D view.

        Inlined from the GPU ``_ClampTensorBase._expand_flat``.
        """
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def _eager_forward(
        self,
        input: torch.Tensor,
        max: torch.Tensor,
    ) -> torch.Tensor:
        # Broadcast input/max to out_shape and dispatch the TileLang
        # max-only Tensor-bound clamp kernel (has_max=True, has_min=False).
        out_shape = self.out_shape if self.out_shape else (1,)
        x_flat = self._expand_flat(input, out_shape)
        hi_flat = self._expand_flat(max, out_shape)
        result = self.kernel(x_flat, None, hi_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self,
        input: torch.Tensor,
        max: torch.Tensor,
    ) -> torch.Tensor:
        # O1: device check via the device backend.
        backend = get_device_backend()
        if not (
            backend.is_device_tensor(input) and backend.is_device_tensor(max)
        ):
            raise ValueError(f"Inputs must be {backend.name} tensors")
        for name, t, expected in [
            ("input", input, self.input_shape),
            ("max", max, self.max_shape),
        ]:
            if t.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"Expected {name}.shape {expected}, got {tuple(t.shape)}"
                )
        return self._eager_forward(input, max)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Mirrors the GPU roofline func
        ``tileops.perf.formulas.clamp_max_fwd_roofline``:
        ``flops = N_total`` (one ``min(input, max)`` per output element);
        ``bytes = 3 * N_total * elem_bytes`` (read input + read max + write
        output, all post-broadcast, at the input element size).
        """
        n_total = int(self.N_total)
        elem_bytes = self.dtype.itemsize
        return n_total, 3 * n_total * elem_bytes
