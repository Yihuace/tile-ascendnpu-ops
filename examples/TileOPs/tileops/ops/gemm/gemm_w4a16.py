"""W4A16 GEMM operator ``output = activation @ dequant(packed_weight).T`` (NPU-adapted).

Ported from GPU TileOPs ``tileops/ops/gemm.py`` -- the ``GemmW4A16Op`` class
only (the shared GPU file also defines ``GemmOp`` / ``GemmFp8Op``).

Adaptation from GPU (TileOPs) to NPU:

- ``_validate_dtypes``: the GPU version was manifest-codegen-installed on the
  ``Op`` base; the NPU ``Op`` base has no codegen, so ``GemmW4A16Op``
  implements it directly and validates device placement via
  ``backend.is_device_tensor`` (O1) on top of the dtype contract.
- ``tune`` parameter removed (O3); the kernel constructor takes
  ``(m, n, k, dtype, config=None, device_index=None, group_size=...)``.
- ``eval_roofline``: the GPU
  ``tileops.perf.formulas.gemm_w4a16_fwd_roofline`` re-implemented inline (the
  NPU project has no ``tileops.perf`` module; roofline arithmetic is
  device-agnostic).
- The ``autotune`` method is removed (C26).
- The forward flow (input-inferred m/n/k, lazy per-config kernel cache,
  dtype/shape validation) is preserved unchanged (O6).
"""

from __future__ import annotations

from typing import Dict, Hashable, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.gemm import GemmW4A16Kernel
from tileops.kernels.gemm.gemm_w4a16 import GROUP_SIZE
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["GemmW4A16Op"]


class GemmW4A16Op(Op):
    """Dense W4A16 NT GEMM with group-wise affine weight dequantization.

    Public layout is ``activation=[M, K]`` and ``packed_weight=[N, K / 2]``.
    Two unsigned INT4 values are packed per byte: the low nibble stores even K
    and the high nibble stores odd K. ``weight_scale`` and ``weight_zero`` are
    group128 metadata with shape ``[N, K / 128]``. The kernel dequantizes the
    current W4 tile into A16 shared memory and computes ``activation @ W.T``.

    Args:
        group_size: Affine quantization group size (only 128 supported).
        kernel_map: Optional kernel override dict.
    """

    def __init__(
        self,
        group_size: int = GROUP_SIZE,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        if group_size != GROUP_SIZE:
            raise ValueError(
                f"GemmW4A16Op currently supports group_size={GROUP_SIZE}, got {group_size}"
            )
        self.group_size = group_size
        self.dispatch_kernel(kernel_map)
        # (m, n, k, dtype, group_size) -> Kernel instance; built lazily.
        self._kernel_cache: Dict[Hashable, Kernel] = {}
        # Fast path: skip re-validation when the input signature is unchanged.
        self._active_sig: Optional[tuple] = None
        self._active: Optional[Kernel] = None
        # Roofline / dtype bindings, populated on the first forward().
        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gemm_w4a16_kernel": GemmW4A16Kernel}

    def _validate_dtypes(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> None:
        """Validate device placement (O1) and the dtype contract of the inputs."""
        backend = get_device_backend()
        named = (
            ("activation", activation),
            ("packed_weight", packed_weight),
            ("weight_scale", weight_scale),
            ("weight_zero", weight_zero),
        )
        for name, tensor in named:
            if not backend.is_device_tensor(tensor):
                raise ValueError(
                    f"{name} must be a {backend.name} tensor, got device {tensor.device}"
                )
        if activation.dtype != torch.float16:
            raise ValueError(
                f"GemmW4A16Op currently supports float16 activation, got {activation.dtype}"
            )
        if packed_weight.dtype != torch.uint8:
            raise ValueError(f"GemmW4A16Op expects uint8 packed_weight, got {packed_weight.dtype}")
        if weight_scale.dtype != torch.float32:
            raise ValueError(f"GemmW4A16Op expects float32 weight_scale, got {weight_scale.dtype}")
        if weight_zero.dtype != torch.uint8:
            raise ValueError(f"GemmW4A16Op expects uint8 weight_zero, got {weight_zero.dtype}")

    def _infer_mnk(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
    ) -> Tuple[int, int, int]:
        if activation.ndim != 2 or packed_weight.ndim != 2:
            raise ValueError(
                "GemmW4A16Op expects rank-2 activation and packed_weight, got "
                f"{activation.ndim} and {packed_weight.ndim}"
            )
        m, k = activation.shape
        n, packed_k = packed_weight.shape
        if k % 2 != 0:
            raise ValueError(f"GemmW4A16Op expects even K for W4 packing, got {k}")
        if packed_k != k // 2:
            raise ValueError(
                "GemmW4A16Op packed_weight shape mismatch: expected second dim "
                f"{k // 2}, got {packed_k}"
            )
        if k % self.group_size != 0:
            raise ValueError(
                f"GemmW4A16Op expects K divisible by group_size={self.group_size}, got {k}"
            )
        return m, n, k

    def _infer_output_shapes(
        self,
        activation_shape: Tuple[int, ...],
        packed_weight_shape: Tuple[int, ...],
        weight_scale_shape: Tuple[int, ...],
        weight_zero_shape: Tuple[int, ...],
    ) -> dict[str, Tuple[int, int]]:
        return {"output": (activation_shape[0], packed_weight_shape[0])}

    def _validate_shapes(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> tuple[int, int, int]:
        m, n, k = self._infer_mnk(activation, packed_weight)
        groups = k // self.group_size
        metadata_shape = (n, groups)
        if tuple(weight_scale.shape) != metadata_shape:
            raise ValueError(
                f"GemmW4A16Op weight_scale must have shape {metadata_shape}, "
                f"got {tuple(weight_scale.shape)}"
            )
        if tuple(weight_zero.shape) != metadata_shape:
            raise ValueError(
                f"GemmW4A16Op weight_zero must have shape {metadata_shape}, "
                f"got {tuple(weight_zero.shape)}"
            )
        return m, n, k

    def _get_kernel(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
    ) -> Kernel:
        key = (m, n, k, dtype, self.group_size)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            # O3: no ``tune`` passed to the kernel constructor.
            kernel = self.kernel_map["gemm_w4a16_kernel"](
                m, n, k, dtype, group_size=self.group_size
            )
            self._kernel_cache[key] = kernel
        return kernel

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        sig = (
            activation.shape,
            packed_weight.shape,
            weight_scale.shape,
            weight_zero.shape,
            activation.dtype,
            packed_weight.dtype,
            weight_scale.dtype,
            weight_zero.dtype,
            self.group_size,
        )
        if sig != self._active_sig:
            self._validate_dtypes(activation, packed_weight, weight_scale, weight_zero)
            m, n, k = self._validate_shapes(activation, packed_weight, weight_scale, weight_zero)
            self.m, self.n, self.k = m, n, k
            self.dtype = activation.dtype
            self.activation_shape = tuple(activation.shape)
            self.packed_weight_shape = tuple(packed_weight.shape)
            self.weight_scale_shape = tuple(weight_scale.shape)
            self.weight_zero_shape = tuple(weight_zero.shape)
            kernel = self._get_kernel(m, n, k, activation.dtype)
            self.kernel = kernel
            self._active = kernel
            self._active_sig = sig

        return self._active(activation, packed_weight, weight_scale, weight_zero)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inlined from the GPU ``tileops.perf.formulas.gemm_w4a16_fwd_roofline``
        (device-agnostic).  ``GemmW4A16Op`` is input-inferred, so m/n/k and the
        dtype are bound on the op during ``forward()``.
        """
        if self.m is None or self.n is None or self.k is None or self.dtype is None:
            raise RuntimeError(
                "GemmW4A16Op.eval_roofline() is valid only after the first forward(); "
                "m/n/k and dtype are inferred from the inputs."
            )
        m, n, k = self.m, self.n, self.k
        elem_bytes = self.dtype.itemsize
        groups = k // self.group_size
        flops = 2 * m * n * k
        activation_bytes = m * k * elem_bytes
        packed_weight_bytes = n * k // 2
        metadata_bytes = n * groups * (4 + 1)
        output_bytes = m * n * elem_bytes
        return int(flops), int(
            activation_bytes + packed_weight_bytes + metadata_bytes + output_bytes
        )
