"""LerpFwdOp -- scalar-weight lerp with broadcast (NPU-adapted).

Adaptation from GPU (TileOPs) to NPU, per docs/gpu_to_npu_adaptation.md §3:

- O1: device checks use ``backend.is_device_tensor(...)`` instead of
      ``input.is_cuda``.
- O3: ``tune`` parameter removed; the kernel constructor takes
      ``(N_total, dtype, weight, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: ``from .compile_boundary import register_instance`` removed.
- O6: Op-specific flow (broadcast -> flatten -> kernel dispatch -> reshape,
      roofline) preserved from ``torch.lerp`` scalar-weight semantics.

The GPU ``LerpScalarFwdOp`` extends ``BinaryOp`` and takes ``weight`` only
(shapes/dtype come from the first call); the NPU port receives the two operand
shapes in its constructor (the NPU benchmark pipeline reconstructs Ops from
their constructor arguments -- see ``tileops.benchmark.msprof``).  The scalar
``weight`` is a construction-time constant baked into the kernel.  Broadcasting
is resolved here: both operands are expanded to ``broadcast_shapes(input, end)``
and flattened before kernel dispatch (the ``register_copy`` case).
"""

from __future__ import annotations

from math import prod
from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.lerp_scalar import LerpScalarFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["LerpFwdOp"]


class LerpFwdOp(Op):
    """Scalar-weight lerp with broadcast: ``out = input + weight * (end - input)``.

    Conforms to the scalar-weight overload of ``torch.lerp`` --
    ``torch.lerp(input, end, weight: float)``.  Unlike the runtime-weight
    PyTorch call, ``weight`` here is a **construction-time constant** baked
    into the compiled kernel (compile-time folding), so a new Op instance is
    needed for each distinct weight value.  The Op layer broadcasts ``input``
    and ``end`` together, flattens them, dispatches the flat
    ``LerpScalarFwdKernel`` on ``N_total = product(broadcast_shape)`` elements,
    and reshapes the result back.  The tensor-weight overload is handled
    separately by ``LerpTensorFwdOp`` (``variant_of`` this op).

    Args:
        input: Shape of the start tensor.
        end: Shape of the end tensor (broadcasts against ``input``).
        dtype: Torch dtype for both operands (float16, bfloat16, or float32).
        weight: Scalar interpolation weight (default 0.5); baked in.
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "lerp_scalar"

    # Manifest declares ``float16 | bfloat16 | float32``.
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        *,
        input: tuple,
        end: tuple,
        dtype: torch.dtype,
        weight: float = 0.5,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(f"LerpFwdOp does not support dtype {dtype}. Supported: [{names}]")
        # Keep constructor-name aliases so the msprof subprocess can reconstruct
        # this Op through ``_extract_op_init_args`` (which matches ``__init__``
        # parameter names to instance attributes).
        self.input = tuple(input)
        self.end = tuple(end)
        self.input_shape = self.input
        self.end_shape = self.end
        self.dtype = dtype
        self.weight = weight
        self.output_dtype = dtype  # same_as(input)
        self.out_shape = tuple(
            torch.broadcast_shapes(
                self.input_shape,
                self.end_shape,
            )
        )
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        # O3: ``tune`` removed; the kernel constructor takes
        # (N_total, dtype, weight, config=None).
        self.kernel = self.kernel_map[self._op_name](
            self.N_total,
            dtype,
            self.weight,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"lerp_scalar": LerpScalarFwdKernel}

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Expand ``t`` to ``target_shape`` and return a contiguous flat view."""
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def _eager_forward(
        self,
        input: torch.Tensor,
        end: torch.Tensor,
    ) -> torch.Tensor:
        """Direct kernel call: broadcast -> flatten -> kernel -> reshape."""
        out_shape = self.out_shape if self.out_shape else (1,)
        a_flat = self._expand_flat(input, out_shape)
        b_flat = self._expand_flat(end, out_shape)
        result = self.kernel(a_flat, b_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self,
        input: torch.Tensor,
        end: torch.Tensor,
    ) -> torch.Tensor:
        # O1: device checks via the device backend.
        backend = get_device_backend()
        if not (
            backend.is_device_tensor(input) and backend.is_device_tensor(end)
        ):
            raise ValueError(f"Inputs must be {backend.name} tensors")
        for name, t, expected in [
            ("input", input, self.input_shape),
            ("end", end, self.end_shape),
        ]:
            if t.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(f"Expected {name}.shape {expected}, got {tuple(t.shape)}")
        return self._eager_forward(input, end)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Mirrors the GPU manifest roofline for the scalar-weight lerp:
        ``flops = 3 * prod(broadcast(A, B))`` (sub + scalar mul + add);
        ``bytes = 4 * N_total * elem_bytes`` (2 reads + 1 write, with the
        broadcast read of the smaller operand).
        """
        n_total = int(self.N_total)
        elem_bytes = self.dtype.itemsize
        return 3 * n_total, 4 * n_total * elem_bytes
