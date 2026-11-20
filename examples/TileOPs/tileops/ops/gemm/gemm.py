"""Dense GEMM operator ``d = op(a) @ op(b)`` (NPU-adapted).

Ported from GPU TileOPs ``tileops/ops/gemm.py`` -- the ``GemmOp`` class only
(the shared GPU file also defines ``GemmFp8Op`` / ``GemmW4A16Op``, which are
not migrated).

Adaptation from GPU (TileOPs) to NPU:

- ``_validate_dtypes``: the GPU version is manifest-codegen-installed on the
  ``Op`` base; the NPU ``Op`` base has no codegen, so ``GemmOp`` implements it
  directly and validates the device via ``backend.is_device_tensor`` (O1) and
  the ``float16 | bfloat16`` input contract.
- ``tune`` parameter removed (O3); the kernel constructors take
  ``(m, n, k, dtype, config=None, device_index=None, trans_a=..., trans_b=...)``
  / ``(n, k, dtype, config=None, device_index=None)``.
- ``default_kernel_map``: the GPU advertised ``gemv_kernel`` only where
  ``get_sm_version()`` matched ``GemvKernel.supported_archs`` (SM90).  With
  K5 (``supported_archs = None``) both kernels are supported everywhere, so
  both are advertised (O4).
- ``eval_roofline``: the GPU
  ``tileops.perf.formulas.gemm_fwd_roofline`` re-implemented inline (the NPU
  project has no ``tileops.perf`` module; roofline arithmetic is
  device-agnostic).
- The ``autotune`` method is removed (O5/C26).
- The forward flow (input-inferred m/n/k, lazy per-config kernel cache,
  GEMV fast path) is preserved unchanged (O6).
"""

from __future__ import annotations

from typing import Dict, Hashable, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.gemm import GemmKernel, GemvKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["GemmOp"]


class GemmOp(Op):
    """Dense GEMM, input-inferred and aligned to DeepGEMM's call-time JIT.

    The logical dims ``m, n, k`` and the dtype are derived from the ``forward``
    inputs; nothing is committed at construction.  The dtype-specialized kernel
    is built (and cached) on first use for each ``(m, n, k, dtype)``.

    Layouts via ``(trans_a, trans_b)`` (== DeepGEMM ``nt``/``nn``/``tn``/``tt``):
      - ``(False, True)``  NT (default): ``A @ B.T``
      - ``(False, False)`` NN:           ``A @ B``
      - ``(True,  False)`` TN:           ``A.T @ B``
      - ``(True,  True)``  TT:           ``A.T @ B.T``

    Args:
        trans_a: Whether ``a`` is stored transposed (``[K, M]``).
        trans_b: Whether ``b`` is stored transposed (``[N, K]``). Default ``True``.
        kernel_map: Optional kernel override dict.

    Example:
        >>> op = GemmOp()                       # NT by default
        >>> d = op(a, b)                         # a=[M,K], b=[N,K] -> d=[M,N]
        >>> flops, nbytes = op.eval_roofline()   # valid after the forward
    """

    def __init__(
        self,
        trans_a: bool = False,
        trans_b: bool = True,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.dispatch_kernel(kernel_map)
        # (mode, m, n, k, dtype) -> Kernel instance; built lazily on first use.
        self._kernel_cache: Dict[Hashable, Kernel] = {}
        # Fast path: skip re-inference when the input signature is unchanged.
        self._active_sig: Optional[tuple] = None
        self._active: Optional[tuple] = None
        # Roofline / dtype bindings, populated on the first forward().
        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # GPU gated GemvKernel on ``get_sm_version()`` (SM90-only); with K5 the
        # GEMV kernel supports every architecture, so it is always advertised.
        return {
            "gemm_kernel": GemmKernel,
            "gemv_kernel": GemvKernel,
        }

    def _validate_dtypes(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """Validate device placement and dtype contract of the inputs.

        The GPU ``GemmOp`` inherited a manifest-codegen ``_validate_dtypes``;
        the NPU ``Op`` base has no codegen, so this is implemented directly and
        adapted to the device backend (O1).
        """
        backend = get_device_backend()
        for name, tensor in (("a", a), ("b", b)):
            if not backend.is_device_tensor(tensor):
                raise ValueError(
                    f"{name} must be a {backend.name} tensor, got device {tensor.device}"
                )
        supported = (torch.float16, torch.bfloat16)
        if a.dtype not in supported or b.dtype != a.dtype:
            raise ValueError(
                "GemmOp expects float16/bfloat16 a and b with equal dtypes, "
                f"got a={a.dtype}, b={b.dtype}"
            )

    def _infer_mnk(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[int, int, int]:
        """Derive logical ``(m, n, k)`` from input shapes per the trans flags."""
        k_a, m = (a.shape[0], a.shape[1]) if self.trans_a else (a.shape[1], a.shape[0])
        n, k_b = (b.shape[0], b.shape[1]) if self.trans_b else (b.shape[1], b.shape[0])
        if k_a != k_b:
            raise ValueError(
                f"GEMM contraction dim mismatch: a contributes K={k_a}, b contributes K={k_b} "
                f"(a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}, "
                f"trans_a={self.trans_a}, trans_b={self.trans_b})"
            )
        return m, n, k_a

    def _get_kernel(self, m: int, n: int, k: int, dtype: torch.dtype) -> Tuple[str, Kernel]:
        """Return ``(mode, kernel)`` for the given dims, building/caching lazily.

        ``mode`` is ``"lhs_row"``/``"rhs_col"`` for the GEMV fast path, else
        ``"gemm"`` -- the dense ``GemmKernel``, covering all four
        ``(trans_a, trans_b)`` layouts.
        """
        gemv_lhs_row = m == 1 and not self.trans_a and self.trans_b
        gemv_rhs_col = n == 1 and not self.trans_a and not self.trans_b
        gemv_cls = self.kernel_map.get("gemv_kernel")
        if (gemv_lhs_row or gemv_rhs_col) and gemv_cls is not None:
            mode = "lhs_row" if gemv_lhs_row else "rhs_col"
            key = (mode, m, n, k, dtype)
            kernel = self._kernel_cache.get(key)
            if kernel is None:
                # lhs_row: a is [1, K], reduce over K -> use (n, k); rhs_col uses (m, k).
                kernel = gemv_cls(n if mode == "lhs_row" else m, k, dtype)
                self._kernel_cache[key] = kernel
            return mode, kernel

        key = ("gemm", m, n, k, dtype)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self.kernel_map["gemm_kernel"](
                m, n, k, dtype, trans_a=self.trans_a, trans_b=self.trans_b
            )
            self._kernel_cache[key] = kernel
        return "gemm", kernel

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Fast path: same input signature as the last call -> reuse the already
        # built/JIT'd kernel directly, skipping dtype validation, shape
        # inference, and the cache lookup (this is the steady state in
        # benchmarking / serving, where per-call Python overhead matters).
        sig = (a.shape, b.shape, a.dtype)
        if sig != self._active_sig:
            self._validate_dtypes(a, b)
            m, n, k = self._infer_mnk(a, b)
            # Bind dims/dtype for the manifest func-mode roofline (read post-forward).
            self.m, self.n, self.k = m, n, k
            self.dtype = a.dtype
            self.a_shape = tuple(a.shape)
            self.b_shape = tuple(b.shape)
            mode, kernel = self._get_kernel(m, n, k, a.dtype)
            # Expose the active kernel so introspection can find it.
            self.kernel = kernel
            self._active = (mode, kernel, n, m)
            self._active_sig = sig

        mode, kernel, n, m = self._active
        if mode == "lhs_row":
            return kernel(a.reshape(-1), b).reshape(1, n)
        if mode == "rhs_col":
            return kernel(b.reshape(-1), a).reshape(m, 1)
        return kernel(a, b)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inlined from the GPU ``tileops.perf.formulas.gemm_fwd_roofline``
        (device-agnostic).  ``GemmOp`` is input-inferred, so the logical dims
        and dtype are bound on the op during ``forward()``; they are
        transpose-independent.
        """
        if self.m is None or self.n is None or self.k is None or self.dtype is None:
            raise RuntimeError(
                "GemmOp.eval_roofline() is valid only after the first forward(); "
                "m/n/k and dtype are inferred from the inputs."
            )
        m, n, k = self.m, self.n, self.k
        elem_bytes = self.dtype.itemsize
        flops = 2 * m * n * k
        nbytes = (m * k + n * k + m * n) * elem_bytes
        return int(flops), int(nbytes)
