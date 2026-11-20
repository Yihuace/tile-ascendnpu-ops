"""Fused Add RMSNorm operator (NPU-adapted).

Ported from GPU (TileOPs) ``tileops/ops/norm/fused_add_rms_norm.py``.

    s            = x + residual
    rstd         = 1 / sqrt(mean(s^2) + eps)
    y            = s * rstd * weight
    residual_out = s

Adaptation from GPU to NPU:

- O1: device checks go through ``backend.is_device_tensor`` for all three
  inputs.
- O3: the ``tune`` parameter is removed; the kernel constructor takes
  ``(M, N, eps, dtype, config=None, device_index=None)``.
- O6: op-specific flow preserved -- 2-D flatten of ``x``/``residual``, the
  ``(N,)`` weight passed through unchanged, per-``(M, N)`` kernel caching, and
  the manifest roofline (device-agnostic).

The op has **two outputs** (``y`` and ``residual_out``), matching the GPU
``out_idx=[3, 4]`` contract; ``forward`` returns them as a tuple.
"""

from typing import Dict, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm.fused_add_rms_norm import FusedAddRMSNormKernel

from ..op_base import Op

__all__ = ["FusedAddRMSNormFwdOp"]


class FusedAddRMSNormFwdOp(Op):
    """Fused Add RMSNorm: ``(y, residual_out) = fused_add_rms_norm(x, residual, weight)``.

    Supported dtypes: ``torch.float16``, ``torch.bfloat16``, ``torch.float32``.

    Note:
        ``x`` and ``residual`` may have arbitrary leading dimensions (3-D+);
        they are flattened to ``(M, N)`` and both outputs are reshaped back.
        ``weight`` is the 1-D ``(N,)`` scale.

    Args:
        M: Optional committed row count. Preferred API infers it from the input.
        N: Optional committed hidden dimension. Preferred API infers it from
            ``x.shape[-1]``.
        dtype: Data type.
        eps: Epsilon for numerical stability (manifest default 1e-6).
        kernel_map: Optional kernel override dictionary.
    """

    def __init__(
        self,
        M: Optional[int] = None,
        N: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.M = M
        self.N = N
        self.dtype = dtype
        self._committed_M = M
        self._committed_N = N
        self._committed_dtype = dtype
        self.eps = float(eps)
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple[int, int, torch.dtype, int | None], Kernel] = {}
        self._last_roofline_mn: Optional[tuple[int, int]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fused_add_rms_norm": FusedAddRMSNormKernel}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_mn is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind input shape and dtype"
            )
        M, N = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        # Manifest: flops = 5 * M * N, bytes = (4 * M * N + N) * elem_bytes
        # (read x + residual + weight, write y + residual_out).
        return 5 * M * N, (4 * M * N + N) * elem_bytes

    def _get_kernel(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (M, N, dtype, device_index)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["fused_add_rms_norm"](
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fused add + RMSNorm.

        Args:
            x: Input tensor of shape ``(*leading, N)`` on the target device.
            residual: Same shape/dtype as ``x``.
            weight: 1-D ``(N,)`` scale, same dtype as ``x``.

        Returns:
            ``(y, residual_out)``, both of shape ``x.shape``.

        Raises:
            ValueError: If tensors are not on the target device, have
                mismatched dtypes, or shapes are incompatible.
        """
        backend = get_device_backend()
        for name, t in (("x", x), ("residual", residual), ("weight", weight)):
            if not backend.is_device_tensor(t):
                raise ValueError(f"{name} must be a {backend.name} tensor")
        expected_dtype = self._committed_dtype
        if expected_dtype is not None and x.dtype != expected_dtype:
            raise ValueError(f"Expected x.dtype {expected_dtype}, got {x.dtype}")
        if expected_dtype is None:
            expected_dtype = x.dtype
        if residual.dtype != expected_dtype:
            raise ValueError(f"Expected residual.dtype {expected_dtype}, got {residual.dtype}")
        if weight.dtype != expected_dtype:
            raise ValueError(f"Expected weight.dtype {expected_dtype}, got {weight.dtype}")
        if residual.shape != x.shape:
            raise ValueError(f"Expected residual shape {x.shape}, got {residual.shape}")
        N = x.shape[-1]
        if self._committed_N is not None and self._committed_N != N:
            raise ValueError(f"Expected hidden dim {self._committed_N}, got {N}")
        if weight.shape != (N,):
            raise ValueError(f"Expected weight shape ({N},), got {weight.shape}")

        orig_shape = x.shape
        x2d = x.contiguous().reshape(-1, N)
        r2d = residual.contiguous().reshape(-1, N)
        M_actual = x2d.shape[0]
        if self._committed_M is not None and M_actual != self._committed_M:
            raise ValueError(
                f"Expected M={self._committed_M} (product of leading dims), got {M_actual}"
            )
        self.M = M_actual
        self.N = N
        self.dtype = expected_dtype
        kernel = self._get_kernel(M_actual, N, expected_dtype, x2d.device.index)
        y, residual_out = kernel(x2d, r2d, weight.contiguous())
        self._last_roofline_mn = (M_actual, N)

        return y.reshape(orig_shape), residual_out.reshape(orig_shape)
