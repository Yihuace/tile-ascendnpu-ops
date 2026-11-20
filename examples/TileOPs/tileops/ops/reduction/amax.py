"""Amax reduction operator (NPU).

Ported from GPU TileOPs ``tileops/ops/reduction/reduce.py`` — the
``AmaxFwdOp`` migration unit only (the sibling simple reduce ops sum/mean/
amin/prod are their own migration units; all of them share ``ReduceKernel``
via ``op_kind``).

Adaptation from GPU (TileOPs) to NPU:

- ``_kernel_handles_padding = True`` preserved: the ReduceKernel handles
  alignment padding internally via masked loads, so the Op layer dispatches
  the raw ``(M, N)`` tensor and never applies host-side ``F.pad`` (O6).
- ``_empty_dim_policy = "full"`` preserved (matches the manifest's
  full-reduction semantics for an empty ``dim``).
- ``tune`` parameter removed (O3); the kernel constructor takes
  ``(M, N, op_kind, dtype, config=None, device_index=None)``.
- Device checks live in ``_ReduceOpBase`` via ``backend.is_device_tensor``
  (O1).
- Op-specific flow (validate -> transpose -> reshape -> dispatch -> reshape)
  preserved unchanged (O6).
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.amax import ReduceKernel
from tileops.ops._multidim import EmptyDimPolicy

from .reduce import _ReduceOpBase

__all__ = ["AmaxFwdOp"]


class AmaxFwdOp(_ReduceOpBase):
    """Amax (element-wise maximum) reduction along the configured ``dim``.

    Construction: ``AmaxFwdOp(dtype=..., dim=None, keepdim=False)``.  ``dim``
    accepts ``int``, ``list[int]``, ``tuple[int, ...]``, or ``None`` (full
    reduction, the default).  M and N are derived from the input tensor at
    forward time, and kernels are cached by ``(M, N)`` to avoid rebuilds.

    Args:
        dtype: Input data type (float32, float16, or bfloat16).
        dim: Reduction dimension(s). ``None`` (the default) reduces over all
            dimensions.
        keepdim: Whether to retain the reduced dimension(s) as size 1.
        kernel_map: Optional custom kernel map.
    """

    _op_kind = "amax"
    _kernel_key = "reduce"
    _kernel_cls = ReduceKernel
    # Kernel-side boundary handling: the extracted ``_simple_reduce_kernel``
    # substitutes the amax identity (-inf) for out-of-range columns, so the
    # Op layer passes the unpadded (M, N) tensor through.
    _kernel_handles_padding = True
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        dtype: torch.dtype,
        dim=None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        super().__init__(
            dtype=dtype,
            dim=dim,
            keepdim=keepdim,
            kernel_map=kernel_map,
        )
