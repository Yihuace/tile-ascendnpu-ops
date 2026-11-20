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
from tileops.kernels.reduction.argmax import ArgreduceKernel, leading_layout_ok

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

    def _leading_layout_ok(self, x, dim, M, N) -> bool:
        """Skip the dim-0 transpose when the kernel can read the (N, M) view.

        The input must be contiguous (the reduced axis is then the outer axis
        of the buffer) and the reduced extent small enough that per-plane work
        stays bounded; otherwise the historical transpose path is kept.
        """
        return bool(x.is_contiguous() and leading_layout_ok(M, N))

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

    def _degenerate_reduce_value(self):
        """``N == 1`` closed form: the only index is 0 (see ``_ReduceOpBase``).

        The kernel path cannot express an empty reduction extent (it emits
        an empty-broadcast ``vbrc`` and fails to compile), so a length-1
        reduction axis is resolved on the host as index 0.
        """
        return 0
