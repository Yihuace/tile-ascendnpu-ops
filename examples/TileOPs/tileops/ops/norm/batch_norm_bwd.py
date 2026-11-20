"""Batch Normalization backward operator (NPU-adapted).

Computes the gradients of the training batch-norm forward with respect to its
input, scale and shift, from the per-channel ``mean`` and ``rstd`` that forward
computed:

    fwd_op = BatchNormFwdOp(training=False, momentum=0.1, eps=1e-5)
    y = fwd_op(x, running_mean, running_var, weight, bias)

    bwd_op = BatchNormBwdOp()
    grad_x, grad_weight, grad_bias = bwd_op(grad_out, x, weight, mean, rstd)

Input tensors accept any shape ``(N, C, *spatial)``; the kernels take its
contiguous ``(N, C, S)`` view, with S the product of the spatial axes.

Adaptation from GPU (TileOPs) ``tileops/ops/norm/batch_norm.py``:

- O1: device checks use ``backend.is_device_tensor(...)`` for every input.
- O2: no hard-coded device (the device is resolved through the backend).
- O3: ``tune`` / ``target`` removed; the kernel constructors take no ``tune``.
- O4: ``from tileops.device import get_device_backend``.
- O5: ``from .compile_boundary import register_instance`` removed.
- O6: the op-specific flow is preserved -- ``contiguous().view(N, C, S)`` for
  ``grad_out`` / ``x``, kernel selection cached on the call shape, and the
  ``grad_x`` output reshaped back to the original input shape.
"""

from typing import Dict, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import (
    BatchNormBwdKernel,
    BatchNormBwdSplitKernel,
    BatchNormBwdWideKernel,
)
from tileops.kernels.norm.batch_norm_bwd.batch_norm_bwd import select_bwd_kernel

from ..op_base import Op

__all__ = ["BatchNormBwdOp"]


class BatchNormBwdOp(Op):
    """Batch Normalization backward operator.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Args:
        kernel_map: Optional override for kernel dispatch.
    """

    _SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

    def __init__(self, kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel: Optional[Kernel] = None
        # Forward-time shape facts (bound by ``forward``); consumed by
        # ``eval_roofline``.
        self.n: Optional[int] = None
        self.c: Optional[int] = None
        self.s: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "bwd_wide": BatchNormBwdWideKernel,
            "bwd_split": BatchNormBwdSplitKernel,
            "bwd_kernel": BatchNormBwdKernel,
        }

    def _get_kernel(
        self, key: str, n: int, c: int, s: int, dtype: torch.dtype
    ) -> Kernel:
        cache_key = (key, n, c, s, dtype)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]
        cls = self.kernel_map[key]
        kernel = cls(n, c, s, dtype)
        self._kernel_cache[cache_key] = kernel
        return kernel

    def eval_roofline(self) -> Tuple[int, int]:
        """Return ``(flops, bytes)`` for the last ``forward`` call.

        Mirrors the manifest roofline: ``flops = 9 * N * C * S`` (x_hat, the
        grad_bias / grad_weight reductions, and grad_x from them) and
        ``bytes = (3 * N * C * S + 5 * C) * elem_bytes`` (read grad_out / x and
        weight / mean / rstd, write grad_x and grad_weight / grad_bias).
        """
        if self.dtype is None or self.n is None or self.c is None or self.s is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind input shape and dtype"
            )
        n, c, s = int(self.n), int(self.c), int(self.s)
        elem_bytes = self.dtype.itemsize
        return 9 * n * c * s, (3 * n * c * s + 5 * c) * elem_bytes

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run batch normalization backward pass.

        Args:
            grad_out: Upstream gradient of shape ``(N, C, *spatial)``.
            x: Original input tensor of the same shape.
            weight: Affine scale (gamma) of shape ``(C,)``, ``torch.float32``.
            mean: Per-channel batch mean from the forward pass, ``(C,)``.
            rstd: Per-channel reciprocal std from the forward pass, ``(C,)``.

        Returns:
            Tuple of ``(grad_x, grad_weight, grad_bias)`` where ``grad_x`` has
            the same shape as ``x``, and ``grad_weight`` / ``grad_bias`` have
            shape ``(C,)``.

        Raises:
            ValueError: Tensors are not on the target device, dtypes mismatch, or
                shapes are incompatible.
        """
        # O1: device checks via the device backend.
        backend = get_device_backend()
        for name, t in (
            ("grad_out", grad_out),
            ("x", x),
            ("weight", weight),
            ("mean", mean),
            ("rstd", rstd),
        ):
            if not backend.is_device_tensor(t):
                raise ValueError(f"{name} must be a {backend.name} tensor")

        if grad_out.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"{type(self).__name__} does not support dtype {grad_out.dtype}. "
                f"Supported: [{names}]"
            )
        if x.dtype != grad_out.dtype:
            raise ValueError(f"Expected x.dtype {grad_out.dtype}, got {x.dtype}")
        if x.shape != grad_out.shape:
            raise ValueError(f"Expected x shape {grad_out.shape}, got {x.shape}")

        channels = grad_out.shape[1]
        for name, t in (("weight", weight), ("mean", mean), ("rstd", rstd)):
            if tuple(t.shape) != (channels,):
                raise ValueError(f"Expected {name} shape {(channels,)}, got {tuple(t.shape)}")

        batch = grad_out.shape[0]
        spatial = 1
        for dim in grad_out.shape[2:]:
            spatial *= int(dim)
        grad_out_ncs = grad_out.contiguous().view(batch, channels, spatial)
        x_ncs = x.contiguous().view(batch, channels, spatial)
        weight = weight.contiguous()
        mean = mean.contiguous()
        rstd = rstd.contiguous()

        key = select_bwd_kernel(batch, channels, spatial, x.dtype)
        kernel = self._get_kernel(key, batch, channels, spatial, x.dtype)

        self.n = batch
        self.c = channels
        self.s = spatial
        self.dtype = x.dtype
        self.kernel = kernel

        grad_x, grad_weight, grad_bias = kernel(
            grad_out_ncs, x_ncs, weight, mean, rstd
        )
        return grad_x.view(x.shape), grad_weight, grad_bias
