"""Count-nonzero reduction operator (``count_nonzero``).

Ported from the GPU ``tileops/ops/reduction/logical_reduce.py``
(``CountNonzeroFwdOp``). ``AllFwdOp`` / ``AnyFwdOp`` share the same GPU file
but are separate migration units; only ``CountNonzeroFwdOp`` is ported here.

Adaptation from GPU (TileOPs) to NPU:

- ``tune`` parameter removed (O3); kernel construction does not pass ``tune``.
- Device checks live in ``_ReduceOpBase`` via ``backend.is_device_tensor`` (O1).
- ``to_logical_float32`` / ``_UNSUPPORTED_STORAGE_DTYPES`` are imported from
  the NPU kernel package (the GPU module they came from).
- Op-specific flow (validate -> flatten -> reshape -> dispatch -> reshape) and
  the ``eval_roofline()`` formula are preserved unchanged (O6).

``CountNonzeroFwdOp`` counts the non-zero elements along ``dim`` and returns
``int64`` -- unlike the sibling ``AllFwdOp`` / ``AnyFwdOp`` kinds, whose output
dtype is ``bool``. There is no ``keepdim`` parameter: the reduced dims are
always removed, matching ``torch.count_nonzero`` semantics.

``_kernel_handles_padding = True``: the kernel handles 256-element alignment
padding internally via masked loads, so the raw ``(M, N)`` tensor is passed
through without host-side ``F.pad``.

Empty-dim contract (``_empty_dim_policy = "full"``): ``dim=[]`` / ``dim=()`` is
treated as a full reduction, matching ``torch.count_nonzero``'s empty-tuple
handling (which differs from the no-op contract of ``all`` / ``any``).

Supports any numeric dtype including torch.bool, int32, int64, and complex
types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
complex64, complex128) are pre-converted to float32 in ``forward()``.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.count_nonzero import LogicalReduceKernel
from tileops.kernels.reduction.count_nonzero.count_nonzero import (
    _UNSUPPORTED_STORAGE_DTYPES,
    to_logical_float32,
)
from tileops.ops._multidim import EmptyDimPolicy

from .reduce import _ReduceOpBase

__all__ = ["CountNonzeroFwdOp"]


class CountNonzeroFwdOp(_ReduceOpBase):
    """Count-nonzero reduction along ``dim``, returning int64.

    Construction: ``CountNonzeroFwdOp(dtype=..., dim=None)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Padded positions use 0, which is neutral for sum/count.

    Note: No ``keepdim`` parameter -- the reduction dimension is always
    removed, matching ``torch.count_nonzero`` semantics.

    Supports any numeric dtype including torch.bool, int32, int64, and complex
    types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
    complex64, complex128) are pre-converted to float32 in forward().

    Args:
        dtype: Input data type (float16, bfloat16, float32, int32, int64,
               bool, complex64, complex128).
        dim: Reduction dimension (default ``None``, i.e. full reduction).
            Accepts ``int``, ``list[int]``, or ``tuple[int, ...]`` for
            multi-dim reduction.
        kernel_map: Optional custom kernel map.
    """

    _op_kind = "count_nonzero"
    _kernel_key = "logical_reduce"
    _kernel_cls = LogicalReduceKernel
    _empty_dim_policy: EmptyDimPolicy = "full"
    _kernel_handles_padding = True

    def __init__(
        self,
        dtype: torch.dtype,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        # count_nonzero never keeps dim (matches torch.count_nonzero)
        super().__init__(
            dtype=dtype,
            dim=dim,
            keepdim=False,
            kernel_map=kernel_map,
        )

    def _pad_value(self) -> float:
        """Pad with 0, neutral for sum/count."""
        return 0.0

    def _noop_output_dtype(self) -> torch.dtype:
        """count_nonzero returns int64 per manifest contract.

        Although count_nonzero's ``_empty_dim_policy`` is ``"full"`` (so the
        empty-dim no-op short-circuit never fires), the shared scalar
        forward in the base class consults this hook to cast the
        ``x != 0`` predicate to the declared output dtype.
        """
        return torch.int64

    def _pre_kernel(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Convert unsupported storage dtypes to float32."""
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_float32(x)
        return x, None
