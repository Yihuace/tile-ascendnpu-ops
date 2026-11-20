"""Dropout op with PyTorch-compatible semantics (NPU-adapted).

Ported from the GPU ``DropoutOp(Op)`` in
``{gpu_repo_root}/tileops/ops/dropout.py``.  Wraps ``DropoutKernel`` with shape
handling and training/eval mode support.  Implements inverted dropout:
``output = x * mask / (1 - p)`` during training, identity pass-through during
eval (``training=False``).

Edge cases:
    - p=0: identity (no dropout)
    - p=1: all zeros
    - training=False: identity pass-through

The input is flattened to a 1-D vector of ``N_total`` elements, the kernel is
dispatched, and the output is reshaped back to the original shape.  Kernels are
cached per ``(numel, dtype, device_index)`` because the shape is bound at
``forward()`` time (not at construction), mirroring the GPU op flow.

Adaptation from GPU (TileOPs) to NPU:

- O1: ``input.is_cuda`` -> ``backend.is_device_tensor(input)``.
- O3: ``tune`` parameter removed; the kernel constructor takes
      ``(N_total, dtype, p=..., seed=..., device_index=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: the ``torch.compile`` custom-op registry (``_OP_REGISTRY`` /
      ``_wrapped`` / ``compile_boundary.register_instance``) is removed -- NPU
      ops dispatch through the ``forward`` / ``_eager_forward`` path directly.
- O6: Op-specific flow (short-circuit paths, flatten -> kernel -> reshape,
      per-shape kernel cache, roofline) preserved.
"""

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.regularization.dropout import DropoutKernel
from tileops.ops.op_base import Op

__all__ = ["DropoutOp"]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class DropoutOp(Op):
    """Dropout operation with deterministic replay.

    Compatible with PyTorch dropout semantics:
        - Training mode: ``output = x * mask / (1 - p)``, ``mask ~ Bernoulli(1 - p)``
        - Eval mode (``training=False``): ``output = x`` (identity)
        - p=0: identity
        - p=1: all zeros

    Same seed produces identical masks for deterministic replay.

    Args:
        p: Drop probability in [0, 1].
        seed: Integer seed for the RNG.
        training: If ``False``, dropout is disabled (identity pass-through).
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "dropout"
    kernel_cls = DropoutKernel

    def __init__(
        self,
        p: float = 0.5,
        seed: int = 0,
        training: bool = True,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.N_total = None
        self.dtype = None
        self.p = p
        self.seed = seed
        self.training = training

        # Skip kernel build when dropout has no effect (identity or all-zeros).
        self._skip = not training or p == 0.0
        self._all_zero = training and p == 1.0

        # Always populate kernel_map for Op base class consistency.
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple[int, torch.dtype, int | None], Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x + write y."""
        if self.N_total is None or self.dtype is None:
            raise RuntimeError(
                "DropoutOp.total_memory requires a prior forward() call to bind "
                "input shape and dtype"
            )
        return self.N_total * self.dtype.itemsize * 2

    def _get_kernel(self, x: torch.Tensor) -> Kernel:
        key = (x.numel(), x.dtype, x.device.index)
        if key not in self._kernel_cache:
            # O3: no `tune` argument on the NPU kernel constructor.
            self._kernel_cache[key] = self.kernel_map[self._op_name](
                x.numel(), x.dtype, p=self.p, seed=self.seed,
                device_index=x.device.index,
            )
        return self._kernel_cache[key]

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.N_total = x.numel()
        self.dtype = x.dtype
        if self._skip:
            return x.clone()
        if self._all_zero:
            return torch.zeros_like(x)
        orig_shape = x.shape
        x_flat = x.contiguous().reshape(-1)
        self.kernel = self._get_kernel(x_flat)
        y_flat = self.kernel(x_flat)
        return y_flat.reshape(orig_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        backend = get_device_backend()
        # O1: device check via the device backend instead of ``input.is_cuda``.
        if not backend.is_device_tensor(input):
            raise ValueError(
                f"input must be a {backend.name} tensor, got device {input.device}"
            )
        if input.dtype not in _FLOAT_DTYPES:
            raise ValueError(
                f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}"
            )
        # O5: no ``_wrapped`` custom_op dispatch; call the eager path directly.
        return self._eager_forward(input)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inlined from the GPU ``tileops.perf.formulas.dropout_roofline`` (the
        NPU project has no ``tileops.perf`` module; the arithmetic is
        device-agnostic):

            training=True, 0 < p < 1:  flops = N, bytes = 2 * N * elem_bytes
            p == 0 or training=False:  flops = 0, bytes = 2 * N * elem_bytes
            p == 1:                    flops = 0, bytes = N * elem_bytes

        ``N`` and ``dtype`` are bound by the last ``forward()`` call.
        """
        if self.N_total is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind input shape and dtype"
            )
        n_total = int(self.N_total)
        elem_bytes = int(self.dtype.itemsize)
        if not self.training or self.p == 0.0:
            return 0, int(2 * n_total * elem_bytes)
        if self.p == 1.0:
            return 0, int(n_total * elem_bytes)
        return int(n_total), int(2 * n_total * elem_bytes)
