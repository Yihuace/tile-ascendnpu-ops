"""FP8 per-group quantization operator (NPU-adapted).

Adaptation from GPU (TileOPs) ``tileops/ops/fp8_quant.py``:

- O1: ``input_tensor.is_cuda`` device check -> ``backend.is_device_tensor(...)``
  via ``get_device_backend()``.
- O2: no hard-coded ``device="cuda"`` (the device is resolved by the backend).
- O3: ``tune`` parameter removed from ``__init__`` and ``_get_kernel``; the
  kernel constructor takes ``(batch, seq_len_kv, kv_group, index_dim, dtype,
  config=None)``.
- O6: op-specific flow preserved unchanged -- the 4-D validation, pass-through
  reshape (unary per-group quantization, no flatten), kernel cache keyed on
  ``(batch, seq_len_kv, kv_group, index_dim, dtype, device_index)``, and the
  dual ``(scale, output)`` tuple return.
- C24: the GPU manifest ships a ``roofline.func`` codegen hook
  (``tileops.perf.formulas.fp8_quant_roofline``); the NPU Op implements the
  same arithmetic directly in ``eval_roofline()`` (no codegen on NPU).
"""

from typing import Dict, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization import FP8QuantKernel

from ..op_base import Op

__all__ = ["FP8QuantOp"]


class FP8QuantOp(Op):
    """Per-group fp8 quantization of a ``[B, S_kv, G, D]`` index tensor.

    Quantizes the last (index) dimension of every ``(b, s, g)`` group: the
    per-group scale is ``amax / 448`` (amax clamped to ``>= 1e-4``), the values
    are clamped to ``[-448, 448]`` and cast to ``float8_e4m3fn``.

    Returns:
        A ``(scale_tensor, output_tensor)`` tuple with shapes
        ``[B, S_kv, G]`` (``torch.float32``) and ``[B, S_kv, G, D]``
        (``torch.float8_e4m3fn``).

    Args:
        kernel_map: Optional kernel override dictionary.
    """

    def __init__(self, kernel_map: Optional[Dict[str, Kernel]] = None):
        self.batch = None
        self.seq_len_kv = None
        self.kv_group = None
        self.index_dim = None
        self.in_dtype = None
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fp8_quant_kernel": FP8QuantKernel}

    def _get_kernel(
        self,
        batch: int,
        seq_len_kv: int,
        kv_group: int,
        index_dim: int,
        in_dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        # O3: ``tune`` removed from the cache key and the constructor call.
        key = (batch, seq_len_kv, kv_group, index_dim, in_dtype, device_index)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["fp8_quant_kernel"](
                batch, seq_len_kv, kv_group, index_dim, in_dtype
            )
        return self._kernel_cache[key]

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the last ``forward`` call.

        Device-agnostic; identical arithmetic to the GPU
        ``tileops.perf.formulas.fp8_quant_roofline`` hook (and the manifest
        ``roofline`` entry):

            groups = batch * seq_len_kv * kv_group
            elems  = groups * index_dim
            flops  = 6 * elems + groups
            bytes  = elems * in_elem + elems * 1 + groups * 4
        """
        if self.batch is None or self.in_dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind input shape and dtype"
            )
        groups = self.batch * self.seq_len_kv * self.kv_group
        elems = groups * self.index_dim
        in_elem = self.in_dtype.itemsize
        flops = 6 * elems + groups
        nbytes = elems * in_elem + elems * 1 + groups * 4
        return int(flops), int(nbytes)

    def forward(self, input_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # O1: device check via the device backend (was ``.is_cuda``).
        backend = get_device_backend()
        if not backend.is_device_tensor(input_tensor):
            raise ValueError(f"FP8QuantOp expects a {backend.name} input tensor")
        if input_tensor.ndim != 4:
            raise ValueError("FP8QuantOp expects input_tensor shape [B, S, G, D]")
        if input_tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"FP8QuantOp does not support dtype {input_tensor.dtype}")

        batch, seq_len_kv, kv_group, index_dim = input_tensor.shape
        if min(batch, seq_len_kv, kv_group, index_dim) <= 0:
            raise ValueError("FP8QuantOp input dimensions must be positive")

        self.batch = batch
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.index_dim = index_dim
        self.in_dtype = input_tensor.dtype
        self.kernel = self._get_kernel(
            batch, seq_len_kv, kv_group, index_dim, input_tensor.dtype, input_tensor.device.index
        )
        return self.kernel(input_tensor)
