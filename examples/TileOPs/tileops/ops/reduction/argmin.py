"""Arg-reduction operator (argmin).

Adaptation from GPU (TileOPs) to NPU:

- ``_validate_dim`` / ``_pad_value`` / reshape flow preserved unchanged (O6).
- ``tune`` parameter removed (O3); kernel constructor takes
  ``(M, N, op_kind, dtype, config=None, device_index=None)``.
- Device checks live in ``_ReduceOpBase`` via ``backend.is_device_tensor``
  (O1).
- Only ``ArgminFwdOp`` is ported here (ArgmaxFwdOp is its own migration
  unit; both share ``ArgreduceKernel`` via ``op_kind``).
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.argmin import ArgreduceKernel

from .reduce import _ReduceOpBase

__all__ = ["ArgminFwdOp"]


class ArgminFwdOp(_ReduceOpBase):
    """Argmin reduction along an arbitrary dim, returning int64 indices.

    Construction: ``ArgminFwdOp(dtype=..., dim=None, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dtype: Input data type.
        dim: Reduction dimension. ``None`` (the default) matches
            ``torch.argmin(x)`` semantics: the input is treated as a
            contiguous flattened 1D buffer and the returned index is into
            that flattened tensor.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
    """

    _op_kind = "argmin"
    _kernel_key = "argreduce"
    _kernel_cls = ArgreduceKernel

    def __init__(
        self,
        dtype: torch.dtype,
        dim: Optional[int] = None,
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

    def _validate_dim(self) -> None:
        """Argmin accepts a scalar ``int`` dim or ``None`` (full-tensor reduction).

        ``dim=None`` matches ``torch.argmin(x)`` semantics: the input is
        treated as a contiguous flattened 1D buffer and the returned index
        is into that flattened tensor.
        """
        if self.dim is None or isinstance(self.dim, int):
            return
        raise ValueError(
            f"ArgminFwdOp only supports scalar dim (int) or None, "
            f"got {type(self.dim).__name__}: {self.dim!r}"
        )

    def _pad_value(self) -> float:
        """Pad with +inf so padded positions never win argmin."""
        return float("inf")

    def _degenerate_reduce_value(self):
        """``N == 1`` closed form: the only index is 0 (see ``_ReduceOpBase``).

        The kernel path cannot express an empty reduction extent (it emits
        an empty-broadcast ``vbrc`` and fails to compile), so a length-1
        reduction axis is resolved on the host as index 0.
        """
        return 0
