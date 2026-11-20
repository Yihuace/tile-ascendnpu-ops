"""Batched GEMM op (BmmFwdOp) -- NPU-adapted.

Strict 3D-3D batched matrix multiplication matching ``torch.bmm``: every
batch item is an independent GEMM, no broadcasting.

Adaptation from GPU (TileOPs) to NPU:

- ``_validate_dtypes`` implemented inline: ``backend.is_device_tensor`` device
  check (O1) plus the manifest dtype contract (``float16 | bfloat16`` with
  ``b.dtype == a.dtype``).  The GPU version inherited this from the
  TileOPs manifest codegen, which the NPU ``Op`` base does not carry.
- ``tune`` parameter removed (O3); the kernel constructor takes
  ``(batch, m, n, k, dtype, config=None, device_index=None)``.
- ``autotune`` override removed (C26) -- NPU kernels use heuristic config
  selection only.
- ``eval_roofline``: the GPU
  ``tileops.perf.formulas.bmm_fwd_roofline`` re-implemented inline (the NPU
  project has no tileops.perf module; roofline arithmetic is device-agnostic).
- Forward flow (validate -> infer dims -> lazy kernel dispatch) preserved
  unchanged (O6).
"""

from __future__ import annotations

from typing import Dict, Hashable, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.bmm import BmmKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["BmmFwdOp"]

_ALLOWED_DTYPES = (torch.float16, torch.bfloat16)


class BmmFwdOp(Op):
    """Batched dense GEMM: ``d[i] = a[i] @ b[i]``.

    Input shapes are strict 3D -- ``a: [B, M, K]``, ``b: [B, K, N]``,
    ``d: [B, M, N]``.  Batch and contraction dims are checked at
    ``forward()`` time; the kernel is compiled on first use for each
    ``(batch, m, n, k, dtype)`` combo and cached.

    Args:
        kernel_map: Optional kernel override dict.

    Example:
        >>> op = BmmFwdOp()
        >>> d = op(a, b)                          # a=[B,M,K], b=[B,K,N] -> d=[B,M,N]
        >>> flops, nbytes = op.eval_roofline()    # valid after the forward
    """

    def __init__(
        self,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.dispatch_kernel(kernel_map)
        # (batch, m, n, k, dtype) -> Kernel instance; built lazily on first use.
        self._kernel_cache: Dict[Hashable, Kernel] = {}
        # Fast path: skip re-inference when the input signature is unchanged.
        self._active_sig: Optional[tuple] = None
        self._active_kernel: Optional[Kernel] = None
        # Roofline / dtype bindings, populated on the first forward().
        self.batch: Optional[int] = None
        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None
        self.a_shape: Optional[Tuple[int, ...]] = None
        self.b_shape: Optional[Tuple[int, ...]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"bmm_kernel": BmmKernel}

    def _validate_dtypes(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """Device + dtype checks (O1 + manifest dtype contract)."""
        backend = get_device_backend()
        if not backend.is_device_tensor(a) or not backend.is_device_tensor(b):
            raise ValueError(
                f"BmmFwdOp expects {backend.name} tensors, got "
                f"a.device={a.device}, b.device={b.device}"
            )
        if a.dtype not in _ALLOWED_DTYPES:
            raise ValueError(
                f"BmmFwdOp only supports torch.float16 / torch.bfloat16, got {a.dtype}"
            )
        if b.dtype != a.dtype:
            raise ValueError(
                f"BmmFwdOp expects a and b to share a dtype, got "
                f"a.dtype={a.dtype}, b.dtype={b.dtype}"
            )

    def _infer_bmnk(
        self, a: torch.Tensor, b: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """Derive logical ``(batch, m, n, k)`` from ``[B,M,K]`` and ``[B,K,N]``.

        Raises:
            ValueError: If ranks are wrong, batch dims mismatch, or the
                k dim disagrees between ``a`` and ``b``.
        """
        if a.dim() != 3 or b.dim() != 3:
            raise ValueError(
                f"BmmFwdOp expects strict 3D inputs a=[B,M,K] and b=[B,K,N] "
                f"(got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}); "
            )
        batch_a, m, k_a = a.shape
        batch_b, k_b, n = b.shape
        if batch_a != batch_b:
            raise ValueError(
                f"BmmFwdOp batch dim mismatch: a.shape[0]={batch_a} vs "
                f"b.shape[0]={batch_b}"
            )
        if k_a != k_b:
            raise ValueError(
                f"BmmFwdOp contraction dim mismatch: a contributes K={k_a}, "
                f"b contributes K={k_b} (a.shape={tuple(a.shape)}, "
                f"b.shape={tuple(b.shape)})."
            )
        if k_a % 16 != 0:
            raise ValueError(
                f"BmmFwdOp requires contraction dim K to be a multiple of 16 "
                f"(WGMMA alignment; see manifest shape_rules), got K={k_a}")
        return batch_a, m, n, k_a

    def _cache_key(self, *input_shapes: Tuple[int, ...]) -> Hashable:
        """Project onto the dims the kernel actually specializes on."""
        if len(input_shapes) == 2:
            a_shape, b_shape = input_shapes
            if len(a_shape) == 3 and len(b_shape) == 3:
                batch, m, k = a_shape
                _, _, n = b_shape
                return (batch, m, n, k, None if self.dtype is None else str(self.dtype))
        return (self.batch, self.m, self.n, self.k,
                None if self.dtype is None else str(self.dtype))

    def _get_kernel(
        self, batch: int, m: int, n: int, k: int, dtype: torch.dtype,
    ) -> Kernel:
        """Return the cached BmmKernel for the given dims, building lazily."""
        key = (batch, m, n, k, dtype)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self.kernel_map["bmm_kernel"](batch, m, n, k, dtype)
            self._kernel_cache[key] = kernel
        return kernel

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Fast path: same input signature as the last call -> reuse the already
        # built/JIT'd kernel directly.
        sig = (a.shape, b.shape, a.dtype, b.dtype)
        if sig != self._active_sig:
            self._validate_dtypes(a, b)
            batch, m, n, k = self._infer_bmnk(a, b)
            # Bind dims/dtype for the manifest func-mode roofline.
            self.batch, self.m, self.n, self.k = batch, m, n, k
            self.dtype = a.dtype
            self.a_shape = tuple(a.shape)
            self.b_shape = tuple(b.shape)
            kernel = self._get_kernel(batch, m, n, k, a.dtype)
            # Expose the active kernel so introspection can find it.
            self.kernel = kernel
            self._active_kernel = kernel
            self._active_sig = sig

        return self._active_kernel(a, b)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inlined from the GPU ``tileops.perf.formulas.bmm_fwd_roofline``
        (device-agnostic).  Valid only after the first ``forward()`` has bound
        the dims and dtype.
        """
        if self.m is None or self.dtype is None:
            raise RuntimeError(
                "BmmFwdOp.eval_roofline() is valid only after the first "
                "forward(); batch/m/n/k and dtype are inferred from the inputs."
            )
        batch, m, n, k = int(self.batch), int(self.m), int(self.n), int(self.k)
        elem_bytes = self.dtype.itemsize
        flops = 2 * batch * m * n * k
        nbytes = batch * (m * k + n * k + m * n) * elem_bytes
        return int(flops), int(nbytes)
