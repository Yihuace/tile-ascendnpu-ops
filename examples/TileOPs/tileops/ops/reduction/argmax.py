"""Arg-reduction operator (argmax).

Adaptation from GPU (TileOPs) to NPU:

- ``_validate_dim`` / ``_pad_value`` / reshape flow preserved unchanged (O6).
- ``tune`` parameter removed (O3); kernel constructor takes
  ``(M, N, op_kind, dtype, config=None, device_index=None)``.
- Device checks live in ``_ReduceOpBase`` via ``backend.is_device_tensor``
  (O1).
- Only ``ArgmaxFwdOp`` is ported here (ArgminFwdOp is its own migration
  unit; both share ``ArgreduceKernel`` via ``op_kind``).
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.argmax import ArgreduceKernel

from .reduce import _ReduceOpBase

__all__ = ["ArgmaxFwdOp"]


class ArgmaxFwdOp(_ReduceOpBase):
    """Argmax reduction along an arbitrary dim, returning int64 indices.

    Construction: ``ArgmaxFwdOp(dtype=..., dim=None, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dtype: Input data type.
        dim: Reduction dimension. ``None`` (the default) matches
            ``torch.argmax(x)`` semantics: the input is treated as a
            contiguous flattened 1D buffer and the returned index is into
            that flattened tensor.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
    """

    _op_kind = "argmax"
    _kernel_key = "argreduce"
    _kernel_cls = ArgreduceKernel
    # NPU kernel declares the raw (M, N) shape (kernel DESIGN.md 0.6 R3):
    # disable the host-side alignment pad so the unpadded tensor reaches the
    # kernel (required for the narrow-N path on dim=0-style reductions).
    _kernel_handles_padding = True

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
        """Argmax accepts a scalar ``int`` dim or ``None`` (full-tensor reduction).

        ``dim=None`` matches ``torch.argmax(x)`` semantics: the input is
        treated as a contiguous flattened 1D buffer and the returned index
        is into that flattened tensor.
        """
        if self.dim is None or isinstance(self.dim, int):
            return
        raise ValueError(
            f"ArgmaxFwdOp only supports scalar dim (int) or None, "
            f"got {type(self.dim).__name__}: {self.dim!r}"
        )

    def _pad_value(self) -> float:
        """Pad with -inf so padded positions never win argmax."""
        return float("-inf")
