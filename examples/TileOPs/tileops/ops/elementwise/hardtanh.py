"""HardtanhFwdOp -- elementwise Hardtanh activation (NPU-adapted).

Ported from the GPU ``HardtanhFwdOp(_ParametricActivationOp)`` in
``tileops/ops/elementwise/activations.py``.  It is a single-input,
single-output unary activation:

    y = clamp(x, min_val, max_val)

The input is flattened to a 1-D vector of ``N_total`` elements, the kernel is
dispatched, and the output is reshaped back to the original shape.  Output
dtype is ``same_as(input)``.

Adaptation from GPU (TileOPs) to NPU:

- O1: ``input.is_cuda`` -> ``backend.is_device_tensor(input)``.
- O3: ``tune`` parameter removed; kernel constructor takes
      ``(N_total, dtype, min_val, max_val, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: the ``torch.compile`` custom-op registry (``_OP_REGISTRY`` /
      ``_wrapped`` / ``compile_boundary.register_instance``) is removed —
      NPU ops are dispatched through the ``forward`` / ``_eager_forward``
      path directly.  The ``inplace`` companion custom-op is likewise
      replaced by an in-place ``copy_`` in ``forward`` (matching the NPU
      ``MishFwdOp`` / ``EluFwdOp`` / ``LeakyReluFwdOp``).
- O6: Op-specific flow (scalar-parameter validation, flatten -> kernel ->
      reshape, ``inplace`` handling, roofline) preserved unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.hardtanh import HardtanhFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["HardtanhFwdOp"]


def _validate_scalar_param(
    param_name: str, value, dtype: torch.dtype, op_name: str = "hardtanh"
) -> None:
    """Reject a bound scalar that is not representable in ``dtype``.

    Ported from the GPU ``_validate_scalar_param_repr`` float branch
    (``tileops/ops/elementwise/_base.py``).  ``hardtanh`` is float-only, so
    only the float path applies: the bound must be finite and within the
    dtype's ``finfo`` range; ``NaN`` is accepted, ``+/-Inf`` and out-of-range
    values are rejected.  The NPU project does not ship the shared ``_base``
    helper, so the relevant branch is inlined here.
    """
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int``; PyTorch reduces it to {0, 1}.
        return
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{op_name} expected scalar {param_name} to be int/float, got {type(value)}"
        )
    finfo = torch.finfo(dtype)
    value_f64 = float(value)
    if math.isnan(value_f64):
        return
    if math.isinf(value_f64):
        raise ValueError(
            f"{op_name} received {param_name}={value!r}, but {param_name} must be "
            f"finite and representable in dtype {dtype}"
        )
    if not (finfo.min <= value_f64 <= finfo.max):
        raise ValueError(
            f"{op_name} received {param_name}={value!r}, which is not representable in "
            f"dtype {dtype} (valid finite range: [{finfo.min}, {finfo.max}])"
        )


class HardtanhFwdOp(Op):
    """Element-wise Hardtanh: y = clamp(x, min_val, max_val).

    Single-input, single-output unary activation.  The input is flattened to a
    1-D vector of ``N_total`` elements, the kernel is dispatched, and the
    output is reshaped back to the original shape.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        min_val: Lower clamp bound (default -1.0).
        max_val: Upper clamp bound (default 1.0).
        inplace: When True, copy the result back into ``input`` and return
            ``input`` (preserving tensor identity).  The kernel still computes
            into a fresh buffer; only the user-visible tensor is mutated,
            mirroring ``torch.nn.functional.hardtanh``.
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "hardtanh"
    kernel_cls = HardtanhFwdKernel
    # Manifest: flops = "N".  Per roofline.md §1.3, a two-sided clamp
    # collapses to 1 compare-and-select per output element.
    FLOPS_PER_ELEM = 1

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        _validate_scalar_param("min_val", min_val, dtype, self._op_name)
        _validate_scalar_param("max_val", max_val, dtype, self._op_name)
        self.N_total = N_total
        self.dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        self.output_dtype = dtype  # same_as(input)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            N_total, dtype, min_val=min_val, max_val=max_val
        )

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
        """Return ``(flops, bytes)`` for this unary elementwise op instance.

        Mirrors the manifest roofline:
        ``flops = FLOPS_PER_ELEM * N`` and ``bytes = 2 * N * elem_bytes``.
        """
        return self.FLOPS_PER_ELEM * self.N_total, int(self.total_memory)
