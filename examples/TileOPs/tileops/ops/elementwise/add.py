"""AddFwdOp -- scalar-alpha binary addition with broadcast (NPU-adapted).

Adaptation from GPU (TileOPs) to NPU, per docs/gpu_to_npu_adaptation.md §3:

- O1: device checks use ``backend.is_device_tensor(...)`` instead of
      ``input.is_cuda``.
- O3: ``tune`` parameter removed; the kernel constructor takes
      ``(N_total, dtype, alpha, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: ``from .compile_boundary import register_instance`` removed.
- O6: Op-specific flow (broadcast -> flatten -> kernel dispatch -> reshape,
      roofline) preserved from ``torch.add`` semantics.

The GPU ``AddFwdOp`` extends ``_AlphaScaledBinaryOp(BinaryOp)``; the NPU port
keeps the same alpha-scaled binary contract but receives the two operand shapes
in its constructor (the NPU benchmark pipeline reconstructs Ops from their
constructor arguments -- see ``tileops.benchmark.msprof``).  Broadcasting is
resolved here: both operands are expanded to ``broadcast_shapes(input, other)``
and flattened before kernel dispatch (the ``register_copy`` case).
"""

from __future__ import annotations

from math import prod
from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.add import AddFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["AddFwdOp"]


class AddFwdOp(Op):
    """Element-wise addition with broadcast: ``y = input + alpha * other``.

    Conforms to ``torch.add(input, other, *, alpha=1)``.  ``alpha`` is baked
    into the kernel (one specialization per distinct ``alpha``).  The Op layer
    broadcasts ``input`` and ``other`` together, flattens them, dispatches the
    flat ``AddFwdKernel`` on ``N_total = product(broadcast_shape)`` elements,
    and reshapes the result back.

    Args:
        input: Shape of the first operand.
        other: Shape of the second operand (broadcasts against ``input``).
        dtype: Torch dtype for both operands (float16, bfloat16, or float32).
        alpha: Multiplier on ``other`` (default 1).
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "add"

    # Manifest declares ``float16 | bfloat16 | float32`` (the GPU op's full
    # union also lists ints/bool; this NPU port follows the GPU/test float
    # subset).
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        *,
        input: tuple,
        other: tuple,
        dtype: torch.dtype,
        alpha: "int | float" = 1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(f"AddFwdOp does not support dtype {dtype}. Supported: [{names}]")
        # Keep constructor-name aliases so the msprof subprocess can reconstruct
        # this Op through ``_extract_op_init_args`` (which matches ``__init__``
        # parameter names to instance attributes).
        self.input = tuple(input)
        self.other = tuple(other)
        self.input_shape = self.input
        self.other_shape = self.other
        self.dtype = dtype
        self.alpha = alpha
        self.output_dtype = dtype  # same_as(input)
        self.out_shape = tuple(
            torch.broadcast_shapes(
                self.input_shape,
                self.other_shape,
            )
        )
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        # O3: ``tune`` removed; the kernel constructor takes
        # (N_total, dtype, alpha, config=None).
        self.kernel = self.kernel_map[self._op_name](
            self.N_total,
            dtype,
            self.alpha,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"add": AddFwdKernel}

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Expand ``t`` to ``target_shape`` and return a contiguous flat view."""
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def _eager_forward(
        self,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        """Direct kernel call: broadcast -> flatten -> kernel -> reshape."""
        out_shape = self.out_shape if self.out_shape else (1,)
        a_flat = self._expand_flat(input, out_shape)
        b_flat = self._expand_flat(other, out_shape)
        result = self.kernel(a_flat, b_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        # O1: device checks via the device backend.
        backend = get_device_backend()
        if not (
            backend.is_device_tensor(input) and backend.is_device_tensor(other)
        ):
            raise ValueError(f"Inputs must be {backend.name} tensors")
        for name, t, expected in [
            ("input", input, self.input_shape),
            ("other", other, self.other_shape),
        ]:
            if t.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(f"Expected {name}.shape {expected}, got {tuple(t.shape)}")
        return self._eager_forward(input, other)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Mirrors the GPU manifest roofline:
        ``flops = (1 if alpha == 1 else 2) * prod(broadcast(A, B))``.
        Memory: 2 reads (both operands at the post-broadcast ``N_total``) +
        1 write -> ``bytes = 3 * N_total * elem_bytes``.
        """
        n_total = int(self.N_total)
        elem_bytes = self.dtype.itemsize
        flops = (1 if self.alpha == 1 else 2) * n_total
        return flops, 3 * n_total * elem_bytes
