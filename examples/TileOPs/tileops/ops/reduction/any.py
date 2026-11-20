"""Any reduction operator (``any``).

Ported from the GPU ``tileops/ops/reduction/logical_reduce.py``
(``AnyFwdOp``). ``AllFwdOp`` / ``CountNonzeroFwdOp`` share the same GPU file
but are separate migration units; only ``AnyFwdOp`` is ported here.

Adaptation from GPU (TileOPs) to NPU:

- ``tune`` parameter removed (O3); kernel construction does not pass ``tune``.
- Device checks live in ``_ReduceOpBase`` via ``backend.is_device_tensor`` (O1).
- ``to_logical_float32`` / ``_UNSUPPORTED_STORAGE_DTYPES`` are imported from
  the NPU kernel package (the GPU module they came from).
- Op-specific flow (validate -> flatten -> reshape -> dispatch -> reshape) and
  the ``eval_roofline()`` formula are preserved unchanged (O6).

``AnyFwdOp`` reduces along ``dim`` (logical OR) and returns ``bool`` -- unlike
the sibling ``CountNonzeroFwdOp``, whose output dtype is ``int64``. It exposes a
``keepdim`` parameter (default ``False``), matching ``torch.any`` semantics.

``_kernel_handles_padding = True``: the kernel handles 256-element alignment
padding internally via masked loads (pad value 0.0, neutral for OR), so the raw
``(M, N)`` tensor is passed through without host-side ``F.pad``.

Empty-dim contract (``_empty_dim_policy = "noop"``): ``dim=[]`` / ``dim=()`` is a
no-op -- forward returns ``x.bool()`` with the input shape, matching
``torch.any``'s empty-tuple handling (which differs from the full-reduction
contract of ``count_nonzero``).

Supports any numeric dtype including torch.bool, int32, int64, and complex
types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
complex64, complex128) are pre-converted to float32 in ``forward()``.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.any import LogicalReduceKernel
from tileops.kernels.reduction.any.any import (
    _UNSUPPORTED_STORAGE_DTYPES,
    to_logical_float32,
)
from tileops.ops._multidim import EmptyDimPolicy

from .reduce import _ReduceOpBase

__all__ = ["AnyFwdOp"]


class AnyFwdOp(_ReduceOpBase):
    """Any reduction along ``dim``, returning bool.

    Construction: ``AnyFwdOp(dtype=..., dim=None, keepdim=False)``.  M and N
    are derived from the input tensor at forward time, and kernels are
    cached by ``(M, N)`` to avoid rebuilds.

    Padded positions use 0 (False), which is neutral for OR/any.

    Supports any numeric dtype including torch.bool, int32, int64, and complex
    types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
    complex64, complex128) are pre-converted to float32 in forward().

    Empty-dim contract: ``dim=[]`` / ``dim=()`` is a no-op -- forward returns
    ``x.bool()`` with the input shape, matching ``torch.any`` semantics.

    Args:
        dtype: Input data type (float16, bfloat16, float32, int32, int64,
               bool, complex64, complex128).
        dim: Reduction dimension (default ``None``, i.e. full reduction).
            Accepts ``int``, ``list[int]``, or ``tuple[int, ...]`` for
            multi-dim reduction.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
    """

    _op_kind = "any"
    _kernel_key = "logical_reduce"
    _kernel_cls = LogicalReduceKernel
    _kernel_handles_padding = True
    _empty_dim_policy: EmptyDimPolicy = "noop"

    def __init__(
        self,
        dtype: torch.dtype,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        """Construct AnyFwdOp.

        Args:
            dtype: Input data type.
            dim: Reduction dimension (default ``None``, i.e. full reduction).
                Accepts ``int``, ``list[int]``, ``tuple[int, ...]``, or
                ``None``.
            keepdim: Whether to retain reduced dims as size 1.
            kernel_map: Optional override for kernel dispatch.
        """
        super().__init__(
            dtype=dtype,
            dim=dim,
            keepdim=keepdim,
            kernel_map=kernel_map,
        )

    def _pad_value(self) -> float:
        """Pad with 0 (False), neutral for OR/any."""
        return 0.0

    def _noop_output_dtype(self) -> torch.dtype:
        """Any returns bool per manifest contract."""
        return torch.bool

    def _pre_kernel(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Convert unsupported storage dtypes to float32."""
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_float32(x)
        return x, None
