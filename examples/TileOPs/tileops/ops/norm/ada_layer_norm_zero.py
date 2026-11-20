"""Adaptive LayerNorm-Zero (AdaLN-Zero) operator (NPU-adapted).

Ported from GPU (TileOPs) ``tileops/ops/norm/ada_layer_norm_zero.py``.

Adaptation from GPU to NPU:

- O1: ``x.is_cuda`` / ``scale.is_cuda`` / ``shift.is_cuda`` / ``gate.is_cuda``
  device checks -> ``backend.is_device_tensor(...)`` for all four inputs.
- O2: no hard-coded ``device="cuda"`` (device resolved via the backend).
- O3: the ``tune`` parameter is removed (autotune has no NPU counterpart); the
  kernel constructor takes ``(M, N, eps, dtype, has_gate=True, config=None)``.
- O6: op-specific flow preserved unchanged -- validation, 4-input
  ``contiguous().reshape(-1, N)``, kernel cache keyed on
  ``(M, N, dtype, device_index)``, ``y.reshape(orig_shape)``, and the roofline
  formula (device-agnostic).

Note the kernel itself is **shared** with ``AdaLayerNormFwdOp``: the NPU
``AdaLayerNormKernel`` already implements both variants behind ``has_gate``
(``False`` -> AdaLN, ``True`` -> AdaLN-Zero), so this module only adds the
gated op surface -- no new kernel is needed.
"""

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm.ada_layer_norm_zero import AdaLayerNormKernel

from ..op_base import Op

__all__ = ["AdaLayerNormZeroFwdOp"]


class AdaLayerNormZeroFwdOp(Op):
    """Adaptive Layer Normalization-Zero (AdaLN-Zero) operator.

    Applies layer normalization with per-token adaptive scale, shift, and
    gating:

    .. math::

        y = g \\cdot \\left( s \\cdot \\frac{x - \\mathrm{E}[x]}
            {\\sqrt{\\mathrm{Var}[x] + \\epsilon}} + d \\right)

    where *s* (scale), *d* (shift), and *g* (gate) are per-token tensors of
    shape ``(M, N)``, pre-computed by the caller from a conditioning signal.
    Linear projection from the conditioning input to scale/shift/gate is the
    caller's responsibility.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims.

    Args:
        M: Optional committed row count for strict compatibility. Preferred
            API infers it from ``x.shape[:-1]``.
        N: Optional committed hidden dimension. Preferred API infers it from
            ``x.shape[-1]``.
        dtype: Data type (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
    """

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
        return {"ada_layer_norm": AdaLayerNormKernel}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_mn is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind input shape and dtype"
            )
        M, N = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        # Manifest: flops = 6 * M * N, bytes = 5 * M * N * elem_bytes
        # (x + scale + shift + gate reads, output write).
        return 6 * M * N, 5 * M * N * elem_bytes

    def _get_kernel(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (M, N, dtype, device_index)
        if key not in self._kernel_cache:
            # O3/K8: no `tune` argument on NPU.
            self._kernel_cache[key] = self.kernel_map["ada_layer_norm"](
                M,
                N,
                self.eps,
                dtype,
                has_gate=True,
            )
        return self._kernel_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive layer normalization with zero-init gating.

        Args:
            x: Input tensor of shape ``(*leading, N)`` on the target device.
            scale: Per-token scale tensor of shape ``(*leading, N)``.
            shift: Per-token shift tensor of shape ``(*leading, N)``.
            gate: Per-token gate tensor of shape ``(*leading, N)``.

        Returns:
            Normalized, modulated, and gated tensor of the same shape as *x*.

        Raises:
            ValueError: If tensors are not on the target device, dtypes
                mismatch, or shapes are incompatible with the configured
                dimensions.
        """
        # O1: device checks via the device backend (was ``.is_cuda``).
        backend = get_device_backend()
        for name, t in (("x", x), ("scale", scale), ("shift", shift), ("gate", gate)):
            if not backend.is_device_tensor(t):
                raise ValueError(f"{name} must be a {backend.name} tensor")
        expected_dtype = self._committed_dtype
        if expected_dtype is not None and x.dtype != expected_dtype:
            raise ValueError(f"Expected x.dtype {expected_dtype}, got {x.dtype}")
        if expected_dtype is None:
            expected_dtype = x.dtype
        for name, t in (("scale", scale), ("shift", shift), ("gate", gate)):
            if t.dtype != expected_dtype:
                raise ValueError(f"Expected {name}.dtype {expected_dtype}, got {t.dtype}")
        for name, t in (("scale", scale), ("shift", shift), ("gate", gate)):
            if t.shape != x.shape:
                raise ValueError(f"Expected {name} shape {x.shape}, got {t.shape}")
        N = x.shape[-1]
        if self._committed_N is not None and self._committed_N != N:
            raise ValueError(f"Expected hidden dim {self._committed_N}, got {N}")

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, N)
        scale = scale.contiguous().reshape(-1, N)
        shift = shift.contiguous().reshape(-1, N)
        gate = gate.contiguous().reshape(-1, N)
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
        y = kernel(x, scale, shift, gate)
        self._last_roofline_mn = (M_actual, N)

        return y.reshape(orig_shape)
