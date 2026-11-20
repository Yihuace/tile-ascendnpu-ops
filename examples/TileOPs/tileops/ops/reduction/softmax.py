"""LogSumExp operator (spec-conformant interface).

Adaptation from GPU (TileOPs) to NPU:

- ``_validate``: ``x.is_cuda`` → ``backend.is_device_tensor(x)`` (O1).
- Kernel dispatch: ``LogSumExpKernel`` is a TileLang-based kernel
  (NPUIR target).  The GPU TileLang kernel functions are extracted and
  imported; the NPU component rewrites them for ``target="npuir"``.
- ``eval_roofline``: identical arithmetic (roofline is device-agnostic).
- ``tune`` parameter removed (O3); kernel constructor takes
  ``(M, N, op_kind, dtype, config=None, device_index=None)``.
- ``from .compile_boundary import register_instance`` removed (O5).
"""

from __future__ import annotations

from math import prod
from typing import Dict, List, Optional, Tuple, Union

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.logsumexp.logsumexp import LogSumExpKernel
from tileops.ops._multidim import (
    EmptyDimPolicy,
    flatten_for_multidim,
    normalize_dim,
    restore_multidim_shape,
)
from tileops.ops.op_base import Op

__all__ = ["LogSumExpFwdOp"]


class LogSumExpFwdOp(Op):
    """LogSumExp operator: y = logsumexp(x, dim, keepdim).

    Output shape is input shape without the reduction dimension
    (or with size-1 if keepdim=True).

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).
        keepdim: Retain reduced dimension (default False).
        kernel_map: Optional override for kernel dispatch.
    """

    _op_kind = "logsumexp"
    _kernel_key = "logsumexp_fwd"
    _kernel_class = LogSumExpKernel
    _supports_multidim = True
    _empty_dim_policy: EmptyDimPolicy = "reject"
    _static_axes: frozenset = frozenset()

    def __init__(
        self,
        dtype: torch.dtype,
        dim: Union[int, List[int]] = -1,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.dtype = dtype
        self._committed_dtype = dtype
        self.dim = dim
        self.N: Optional[int] = None
        self.keepdim = keepdim
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}
        self.kernel: object | None = None
        self._last_roofline_spec: tuple[int, int, torch.dtype] | None = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_class}

    def _validate(self, x: torch.Tensor) -> None:
        """Validate input tensor.

        NPU adaptation: device check uses ``backend.is_device_tensor(x)``
        instead of ``x.is_cuda``.
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(x):
            raise ValueError(f"x must be a {backend.name} tensor, got device {x.device}")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"x.dtype must be float16, bfloat16, or float32, got {x.dtype}")
        if self._committed_dtype is not None and x.dtype != self._committed_dtype:
            raise ValueError(f"Expected x.dtype {self._committed_dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")
        self.dtype = x.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the logsumexp op.

        Accepts arbitrary-dim input along the configured dim.
        Supports ``dim=list[int]`` for multi-dim reduction.
        """
        self._validate(x)
        orig_shape = x.shape

        effective_dim: Union[int, List[int], Tuple[int, ...], None] = self.dim

        if isinstance(effective_dim, (list, tuple)) or effective_dim is None:
            dims = normalize_dim(
                effective_dim,
                x.ndim,
                empty_dim_policy=self._empty_dim_policy,
            )
            self._static_axes = frozenset((0, d) for d in dims)
            x, orig_shape, _kept = flatten_for_multidim(x, dims)
            N = x.shape[-1]
            M = prod(x.shape[:-1])
            dtype = x.dtype
            self._last_roofline_spec = (M, N, dtype)
            x = x.reshape(M, N)
            kernel = self._get_or_create_kernel(
                M,
                N,
                dtype=dtype,
                device_index=x.device.index if hasattr(x.device, "index") else None,
            )
            self.kernel = kernel
            y = kernel(x)
            return restore_multidim_shape(y, orig_shape, dims, self.keepdim)

        assert isinstance(effective_dim, int)
        if effective_dim < -x.ndim or effective_dim >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-x.ndim}, {x.ndim - 1}], but got {effective_dim})"
            )
        dim = effective_dim % x.ndim

        N = x.shape[dim]
        if self.N is not None and N != self.N:
            raise ValueError(
                f"{type(self).__name__}: committed N={self.N} does not match "
                f"x.shape[{effective_dim}]={N}"
            )
        self._static_axes = frozenset({(0, dim)})
        M = prod(s for i, s in enumerate(x.shape) if i != dim)
        dtype = x.dtype
        self._last_roofline_spec = (M, N, dtype)

        needs_transpose = dim != x.ndim - 1
        if needs_transpose:
            x = x.movedim(dim, -1)

        x = x.contiguous().reshape(M, N)

        kernel = self._get_or_create_kernel(
            M,
            N,
            dtype=dtype,
            device_index=x.device.index if hasattr(x.device, "index") else None,
        )
        self.kernel = kernel

        y = kernel(x)

        return self._reshape_output(y, orig_shape, dim, needs_transpose)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        M, N, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return 4 * M * N, (M * N + M) * elem_bytes

    def _get_or_create_kernel(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        device_index: int | None = None,
    ) -> object:
        key = (M, N, dtype, device_index)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map[self._kernel_key]
            self._kernel_cache[key] = kernel_cls(
                M,
                N,
                self._op_kind,
                dtype,
                device_index=device_index,
            )
        return self._kernel_cache[key]

    def _reshape_output(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
        dim: int,
        needs_transpose: bool,
    ) -> torch.Tensor:
        if y.ndim == 2:
            if needs_transpose:
                transposed_shape = list(orig_shape)
                transposed_shape.append(transposed_shape.pop(dim))
                y = y.reshape(transposed_shape)
                y = y.movedim(-1, dim)
            else:
                y = y.reshape(orig_shape)
        else:
            if self.keepdim:
                kept_shape = list(orig_shape)
                kept_shape[dim] = 1
                y = y.reshape(kept_shape)
            else:
                reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
                y = y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)
        return y
