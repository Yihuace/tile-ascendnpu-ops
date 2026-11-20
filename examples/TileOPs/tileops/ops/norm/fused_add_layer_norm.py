"""Fused residual-add + LayerNorm operator (NPU-adapted).

Adaptation from GPU (TileOPs) ``tileops/ops/norm/fused_add_layer_norm.py``:

- O1: device checks use ``backend.is_device_tensor(...)`` for all four inputs
  instead of the GPU ``kernel_for`` / signature-check path.
- O2: no hard-coded device (the device is resolved through the backend).
- O3: ``tune`` / ``target`` removed; the kernel constructor takes
  ``(M, N, eps, dtype, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: ``from .compile_boundary import register_instance`` removed.
- O6: the op-specific flow is preserved -- ``contiguous().reshape(-1, N)`` for
  the row inputs, ``reshape(N)`` for the affine tensors, kernel dispatch cached
  on ``(M, N, dtype, device_index)``, both outputs reshaped back to the original
  shape, and the device-agnostic roofline formula.

The op computes, per row ``s``::

    s = x + residual                              # residual_out
    y = (s - E[s]) / sqrt(Var[s] + eps) * weight + bias

and returns dual outputs ``(y, residual_out)`` so downstream residual
connections can reuse the pre-norm sum without recomputation.
"""

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import FusedAddLayerNormKernel

from ..op_base import Op

__all__ = ["FusedAddLayerNormFwdOp"]


class FusedAddLayerNormFwdOp(Op):
    """Fused residual addition and Layer Normalization operator.

    Computes the residual sum followed by layer normalization in a single fused
    kernel and returns ``(y, residual_out)``, where ``residual_out = x + residual``.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims.

    Args:
        M: Optional committed row count (product of all leading dims). Preferred
            API infers it from ``x.shape[:-1]``.
        N: Optional committed hidden dimension. Preferred API infers it from
            ``x.shape[-1]``.
        dtype: Data type (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
    """

    _op_name = "fused_add_layer_norm"

    def __init__(
        self,
        M: Optional[int] = None,
        N: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-5,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.M = M
        self.N = N
        self.dtype = dtype
        self._committed_M = M
        self._committed_N = N
        self._committed_dtype = dtype
        self.eps = eps
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple[int, int, torch.dtype, int | None], Kernel] = {}
        self._last_roofline_mn: Optional[tuple[int, int]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: FusedAddLayerNormKernel}

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the last ``forward`` call.

        Mirrors the manifest roofline: ``flops = 8 * M * N`` (residual add +
        affine LayerNorm) and ``bytes = (4 * M * N + 2 * N) * elem_bytes``
        (read x / residual, write y / residual_out, read weight / bias).
        """
        if self._last_roofline_mn is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind input shape and dtype"
            )
        M, N = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        return 8 * M * N, (4 * M * N + 2 * N) * elem_bytes

    def _get_kernel(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (M, N, dtype, device_index)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map[self._op_name](
                M,
                N,
                self.eps,
                dtype,
            )
        return self._kernel_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply fused residual addition and layer normalization.

        Args:
            x: Input tensor of shape ``(*leading, N)`` on the target device.
            residual: Residual tensor of the same shape as *x*.
            weight: Affine scale of shape ``(N,)``.
            bias: Affine shift of shape ``(N,)``.

        Returns:
            ``(y, residual_out)``, where *residual_out* is ``x + residual``, both
            of the same shape as *x*.

        Raises:
            ValueError: Tensors are not on the target device, dtypes mismatch, or
                shapes are incompatible with the configured dimensions.
        """
        # O1: device checks via the device backend.
        backend = get_device_backend()
        for name, t in (("x", x), ("residual", residual), ("weight", weight), ("bias", bias)):
            if not backend.is_device_tensor(t):
                raise ValueError(f"{name} must be a {backend.name} tensor")

        expected_dtype = self._committed_dtype
        if expected_dtype is not None and x.dtype != expected_dtype:
            raise ValueError(f"Expected x.dtype {expected_dtype}, got {x.dtype}")
        if expected_dtype is None:
            expected_dtype = x.dtype
        for name, t in (("residual", residual), ("weight", weight), ("bias", bias)):
            if t.dtype != expected_dtype:
                raise ValueError(f"Expected {name}.dtype {expected_dtype}, got {t.dtype}")

        if residual.shape != x.shape:
            raise ValueError(f"Expected residual shape {x.shape}, got {residual.shape}")
        N = x.shape[-1]
        if tuple(weight.shape) != (N,):
            raise ValueError(f"Expected weight shape {(N,)}, got {tuple(weight.shape)}")
        if tuple(bias.shape) != (N,):
            raise ValueError(f"Expected bias shape {(N,)}, got {tuple(bias.shape)}")
        if self._committed_N is not None and self._committed_N != N:
            raise ValueError(f"Expected hidden dim {self._committed_N}, got {N}")

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, N)
        residual = residual.contiguous().reshape(-1, N)
        weight = weight.contiguous().reshape(N)
        bias = bias.contiguous().reshape(N)
        M_actual = x.shape[0]
        if self._committed_M is not None and M_actual != self._committed_M:
            raise ValueError(
                f"Expected M={self._committed_M} (product of leading dims), got {M_actual}"
            )
        self.M = M_actual
        self.N = N
        dtype = expected_dtype
        assert dtype is not None
        self.dtype = dtype

        kernel = self._get_kernel(M_actual, N, dtype, x.device.index)
        y, residual_out = kernel(x, residual, weight, bias)
        self._last_roofline_mn = (M_actual, N)

        return y.reshape(orig_shape), residual_out.reshape(orig_shape)
