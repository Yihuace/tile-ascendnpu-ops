"""ClampFwdOp -- Tensor-bound clamp (NPU-adapted).

Ported from the GPU ``ClampFwdOp(_ClampTensorBase)`` in
``tileops/ops/elementwise/clamp.py``.  Implements the Tensor-bound form of
``torch.clamp``:

    y = clamp(x, lo, hi)          # both Tensor bounds
    y = clamp(x, lo, None)        # Tensor lower bound only
    y = clamp(x, None, hi)        # Tensor upper bound only

All Tensor operands broadcast together to the output shape; the Op layer
expands/flattens them to contiguous 1-D vectors of ``N_total`` elements,
dispatches the TileLang Tensor-bound clamp kernel, and reshapes the result
back.  Output dtype is ``same_as(input)``.

Adaptation from GPU (TileOPs) to NPU:

- O1: ``t.is_cuda`` -> ``backend.is_device_tensor(t)``.
- O3: ``tune`` parameter removed; the kernel constructor takes
      ``(N_total, dtype, has_min=..., has_max=...)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: the ``torch.compile`` custom-op registry (``_OP_REGISTRY`` /
      ``_wrapped`` / ``compile_boundary.register_instance``) is removed —
      NPU ops dispatch through the ``forward`` / ``_eager_forward`` path
      directly (matching the NPU ``LerpTensorFwdOp`` / ``EluFwdOp``).
- O6: Op-specific flow (broadcast, ``_expand_flat``, kernel dispatch,
      output reshape, roofline) preserved unchanged.
"""

from __future__ import annotations

from math import prod
from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.clamp import ClampTensorFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["ClampFwdOp"]


class ClampFwdOp(Op):
    """Clamp with Tensor lower and/or upper bounds (broadcasting).

    Conforms to ``torch.clamp(input, min, max)`` where ``min`` and ``max``
    are each either a Tensor or ``None``.  At least one of the two bounds
    must be a Tensor.  All Tensor operands broadcast together.  The primary
    spec entry in ``tileops/manifest/`` covers the both-Tensor form; the
    mixed Tensor/``None`` cases are accepted here so callers can mirror
    PyTorch's ``torch.clamp`` API directly.

    Args:
        input: Shape of the input tensor.
        min: Shape of the lower-bound tensor, or ``None`` for no lower bound.
        max: Shape of the upper-bound tensor, or ``None`` for no upper bound.
        dtype: Torch dtype for all operands.
        kernel_map: Optional kernel dispatch override.

    Raises:
        ValueError: If both ``min`` and ``max`` are ``None``.
    """

    _op_name = "clamp"

    def __init__(
        self,
        input: tuple,
        min: Optional[tuple] = None,
        max: Optional[tuple] = None,
        dtype: torch.dtype = torch.float32,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if min is None and max is None:
            raise ValueError(
                "ClampFwdOp requires at least one of `min` or `max` to be a "
                "Tensor shape; both None is not a valid clamp."
            )
        # Constructor-name aliases so the msprof subprocess can reconstruct
        # this Op through ``_extract_op_init_args`` (that helper matches
        # ``__init__`` parameter names to instance attributes).
        self.input = tuple(input)
        self.min = None if min is None else tuple(min)
        self.max = None if max is None else tuple(max)
        self.input_shape = self.input
        self.min_shape = self.min
        self.max_shape = self.max
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(input)
        broadcast_args = [self.input_shape]
        if self.min_shape is not None:
            broadcast_args.append(self.min_shape)
        if self.max_shape is not None:
            broadcast_args.append(self.max_shape)
        self.out_shape = tuple(torch.broadcast_shapes(*broadcast_args))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        # O3: ``tune`` removed; the kernel constructor takes
        # (N_total, dtype, has_min=..., has_max=...).
        self.kernel = self.kernel_map["clamp_tensor"](
            self.N_total,
            dtype,
            has_min=self.min_shape is not None,
            has_max=self.max_shape is not None,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"clamp_tensor": ClampTensorFwdKernel}

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Expand ``t`` to ``target_shape`` and return a contiguous flat view."""
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def _eager_forward(
        self,
        input: torch.Tensor,
        min: Optional[torch.Tensor] = None,
        max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Broadcast all operands to ``out_shape`` and dispatch the TileLang
        # Tensor-bound clamp kernel.  The kernel branches on ``has_min`` /
        # ``has_max`` at build time, so this single Op class also covers the
        # mixed Tensor/None cases.
        out_shape = self.out_shape if self.out_shape else (1,)
        x_flat = self._expand_flat(input, out_shape)
        lo_flat = None if min is None else self._expand_flat(min, out_shape)
        hi_flat = None if max is None else self._expand_flat(max, out_shape)
        result = self.kernel(x_flat, lo_flat, hi_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self,
        input: torch.Tensor,
        min: Optional[torch.Tensor] = None,
        max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Validate that the runtime None / Tensor pattern matches what
        # __init__ was configured for — the broadcast shape and the presence
        # of each bound is baked in at construction.
        if (min is None) != (self.min_shape is None):
            raise ValueError(
                f"min was {'None' if self.min_shape is None else 'a Tensor shape'} at "
                f"__init__ but {'None' if min is None else 'a Tensor'} at forward()"
            )
        if (max is None) != (self.max_shape is None):
            raise ValueError(
                f"max was {'None' if self.max_shape is None else 'a Tensor shape'} at "
                f"__init__ but {'None' if max is None else 'a Tensor'} at forward()"
            )
        backend = get_device_backend()
        tensors = [("input", input, self.input_shape)]
        if min is not None:
            tensors.append(("min", min, self.min_shape))
        if max is not None:
            tensors.append(("max", max, self.max_shape))
        # O1: device check via the device backend instead of ``t.is_cuda``.
        for _, t, _ in tensors:
            if not backend.is_device_tensor(t):
                raise ValueError(f"Inputs must be {backend.name} tensors")
        for name, t, expected in tensors:
            if t.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(f"Expected {name}.shape {expected}, got {tuple(t.shape)}")
        # O5: no ``_wrapped`` custom_op dispatch; call the eager path directly.
        return self._eager_forward(input, min, max)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Mirrors the GPU roofline func
        ``tileops.perf.formulas.clamp_fwd_roofline``: per output element one
        fused compare-and-select (``flops = N_total``); bytes read input,
        min, max and write out -> ``4 * N_total * elem_bytes``.
        """
        n_total = int(self.N_total)
        elem_bytes = self.dtype.itemsize
        return n_total, 4 * n_total * elem_bytes
