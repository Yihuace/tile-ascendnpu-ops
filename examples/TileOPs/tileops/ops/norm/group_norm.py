"""GroupNorm forward operator (NPU-adapted).

Wraps ``GroupNormKernel`` in the standard TileOPs Op interface.

User-facing API mirrors ``torch.nn.functional.group_norm``:

    op = GroupNormFwdOp(num_groups=groups)
    y = op(x, weight, bias)

Input tensors accept shape ``(N, C, *spatial)``; the op reshapes to
``(N*num_groups, D)`` internally where ``D = (C/num_groups) * spatial_size``.

This module ports the **affine primary** only.  ``GroupNormNoAffineFwdOp``
(the affine-free variant, ``variant_of: GroupNormFwdOp``) is out of scope and
is not migrated.

Affine layering (preserved from GPU -- do not collapse):

  - ``GroupNormKernel``'s row-broadcast affine takes a ``(D_padded,)`` tensor.
  - ``GroupNormFwdOp.forward`` takes **per-channel** ``(C,)`` weight/bias and
    does *not* forward them to the kernel.  It normalizes without an affine
    and applies the per-channel scale/shift after, outside the kernel:
    ``y * weight.reshape(1, C, 1, ...) + bias.reshape(1, C, 1, ...)``.

Adaptation from GPU (TileOPs) ``tileops/ops/norm/group_norm.py``:

- O1: ``x.is_cuda`` / ``weight.is_cuda`` / ``bias.is_cuda`` device checks
  -> ``backend.is_device_tensor(...)`` for all three inputs.
- O2: no hard-coded ``device="cuda"`` (device resolved via the backend).
- O3: ``tune`` parameter removed from ``__init__`` and ``_get_kernel``;
  the kernel constructor takes ``(M, D, eps, dtype, config=None)``.
- O4: ``from tileops.utils import get_sm_version`` -> ``get_device_backend``.
- O5: ``from .compile_boundary import register_instance`` removed.
- O6: op-specific flow preserved unchanged -- validation, group reshape,
  kernel cache keyed on ``(M, D, dtype, device_index, eps)``, output reshape,
  post-kernel per-channel affine, and the roofline formula
  (device-agnostic).
"""

import math
from typing import Dict, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import GroupNormKernel

from ..op_base import Op

__all__ = ["GroupNormFwdOp"]


class GroupNormFwdOp(Op):
    """Group Normalization forward operator (affine primary).

    Computes group normalization over ``(C/num_groups, *spatial)`` slices:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
            \\cdot w + b

    where the mean and variance are computed per group over
    ``(C/num_groups, *spatial)`` elements.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary spatial dimensions (1-D, 2-D, 3-D+).
        Handles non-contiguous inputs via explicit ``contiguous()`` call.
        Hidden dimension is padded to 256-element alignment internally.

    Args:
        num_groups: Number of groups (manifest ``params.num_groups``).
            Must divide *C* evenly.
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
    """

    def __init__(
        self,
        num_groups: int,
        eps: float = 1e-5,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.N: Optional[int] = None
        self.C: Optional[int] = None
        self.spatial: Optional[Tuple[int, ...]] = None
        self.num_groups = num_groups
        self.dtype: Optional[torch.dtype] = None
        self.eps = eps
        self.spatial_size: Optional[int] = None
        self.D: Optional[int] = None
        self.M: Optional[int] = None
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel: Optional[Kernel] = None
        self._last_roofline_spec: Optional[tuple[int, int, int, torch.dtype]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"group_norm": GroupNormKernel}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError("GroupNormFwdOp.eval_roofline() requires a prior forward() call")
        N, C, spatial_size, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return (
            5 * N * C * spatial_size,
            (2 * N * C * spatial_size + 2 * C) * elem_bytes,
        )

    def _resolve_spec(
        self, x: torch.Tensor
    ) -> Tuple[int, int, Tuple[int, ...], int, int, int, int, torch.dtype]:
        # O1: device check via the device backend (was ``x.is_cuda``).
        backend = get_device_backend()
        if not backend.is_device_tensor(x):
            raise ValueError(f"x must be a {backend.name} tensor")
        if x.ndim < 2:
            raise ValueError("x must have shape (N, C, *spatial)")
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"x.dtype must be float32, float16, or bfloat16, got {x.dtype}")
        N, C, *spatial_list = x.shape
        spatial = tuple(spatial_list)
        if C % self.num_groups != 0:
            raise ValueError(f"C={C} must be divisible by num_groups={self.num_groups}")
        spatial_size = math.prod(spatial)
        cpg = C // self.num_groups
        D = cpg * spatial_size
        M = N * self.num_groups
        return N, C, spatial, spatial_size, cpg, D, M, x.dtype

    def _bind_spec(
        self,
        N: int,
        C: int,
        spatial: Tuple[int, ...],
        spatial_size: int,
        D: int,
        M: int,
        dtype: torch.dtype,
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.spatial_size = spatial_size
        self.D = D
        self.M = M
        self.dtype = dtype
        self._last_roofline_spec = (N, C, spatial_size, dtype)

    def _get_kernel(
        self,
        M: int,
        D: int,
        dtype: torch.dtype,
        device_index: Optional[int],
    ) -> Kernel:
        # O3: ``tune`` removed from the cache key and the kernel constructor.
        key = (M, D, dtype, device_index, self.eps)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["group_norm"](
                M,
                D,
                self.eps,
                dtype,
            )
        kernel = self._kernel_cache[key]
        self.kernel = kernel
        return kernel

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Apply group normalization.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on the target device.
            weight: Affine scale of shape ``(C,)`` on the target device.
                Required; the affine-free path is ``GroupNormNoAffineFwdOp``
                (not migrated).
            bias: Affine shift of shape ``(C,)`` on the target device.
                Required; the affine-free path is ``GroupNormNoAffineFwdOp``
                (not migrated).

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If any tensor is not on the target device, dtypes
                mismatch, or shapes are incompatible with the configured
                dimensions.
        """
        (
            N,
            C,
            spatial,
            spatial_size,
            cpg,
            D,
            M,
            dtype,
        ) = self._resolve_spec(x)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(
                "weight is required; use GroupNormNoAffineFwdOp for the affine-free path"
            )
        if not isinstance(bias, torch.Tensor):
            raise ValueError(
                "bias is required; use GroupNormNoAffineFwdOp for the affine-free path"
            )
        # O1: device checks via the device backend (was ``.is_cuda``).
        backend = get_device_backend()
        if not backend.is_device_tensor(weight):
            raise ValueError(f"weight must be a {backend.name} tensor")
        if weight.device != x.device:
            raise ValueError(f"Expected weight on {x.device}, got {weight.device}")
        if weight.dtype != dtype:
            raise ValueError(f"Expected weight.dtype {dtype}, got {weight.dtype}")
        if weight.ndim != 1 or weight.shape[0] != C:
            raise ValueError(f"Expected weight shape ({C},), got {weight.shape}")
        if not backend.is_device_tensor(bias):
            raise ValueError(f"bias must be a {backend.name} tensor")
        if bias.device != x.device:
            raise ValueError(f"Expected bias on {x.device}, got {bias.device}")
        if bias.dtype != dtype:
            raise ValueError(f"Expected bias.dtype {dtype}, got {bias.dtype}")
        if bias.ndim != 1 or bias.shape[0] != C:
            raise ValueError(f"Expected bias shape ({C},), got {bias.shape}")

        self._bind_spec(N, C, spatial, spatial_size, D, M, dtype)
        kernel = self._get_kernel(M, D, dtype, x.device.index)
        orig_shape = x.shape
        x = x.contiguous()
        x_reshaped = x.reshape(
            N,
            self.num_groups,
            cpg,
            *spatial,
        )
        x_2d = x_reshaped.reshape(M, D)

        # The kernel broadcasts its affine row-wise; GroupNorm's per-channel
        # affine doesn't fit that layout, so normalize without an affine and
        # apply the per-channel scale/shift after.
        y_2d = kernel(x_2d)

        y = y_2d.reshape(N, self.num_groups, cpg, *spatial)
        y = y.reshape(orig_shape)

        affine_shape = (1, C) + (1,) * len(spatial)
        # S5 attempt-2 (user-authorized, single-site): compute the per-channel
        # affine in fp32 and round back to the input dtype exactly once.  The
        # naive in-dtype form ``y * weight + bias`` double-rounds for fp16/bf16
        # and marginally violates the fp16 gate; the golden computes the whole
        # group_norm (normalize + affine) in fp32 opmath and rounds once.
        y = (
            y.float() * weight.float().reshape(affine_shape) + bias.float().reshape(affine_shape)
        ).to(dtype)
        return y
