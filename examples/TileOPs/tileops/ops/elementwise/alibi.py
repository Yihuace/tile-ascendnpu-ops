"""AlibiFwdOp -- ALiBi position-encoding generative op (NPU-adapted).

Ported from the GPU ``AlibiFwdOp(Op)`` in ``tileops/ops/elementwise/alibi.py``.

ALiBi (Attention with Linear Biases) produces the bias tensor
    bias[h, i, j] = -slope_h * |i - j|,  slope_h = 2^(-8 * (h + 1) / num_heads)
of shape ``(num_heads, seq_len, seq_len)``.  The op has **zero tensor
inputs**: the full output is synthesized from the ``__init__`` parameters
(``seq_len`` / ``num_heads`` / ``dtype``).

Adaptation from GPU (TileOPs) to NPU:

- O3: ``tune`` parameter removed (the GPU op already takes no ``tune``;
  the kernel constructor is heuristic-config-only).
- O6: Op-specific flow (output-shape inference, ``total_memory``,
  ``eval_roofline``, no-arg ``forward`` -> reshape) preserved unchanged.
  There is no tensor input, so the device-check adaptations O1/O2 do not
  apply.

Note:
    Eager-only.  ``AlibiFwdOp`` is not registered as a
    ``torch.library.custom_op`` — with no tensor input there is no
    compile-time shape guard to install, matching the GPU source (E10).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.kernels.elementwise.alibi import AlibiFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["AlibiFwdOp"]


class AlibiFwdOp(Op):
    """ALiBi position encoding: bias[h, i, j] = -slope_h * |i - j|.

    Generates the full ``(num_heads, seq_len, seq_len)`` bias tensor.

    Args:
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        dtype: Torch dtype (float16, bfloat16, or float32).
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses.  Falls back to ``default_kernel_map``.
    """

    _op_name = "alibi"

    def __init__(
        self,
        seq_len: int,
        num_heads: int,
        dtype: torch.dtype,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](seq_len, num_heads, dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: AlibiFwdKernel}

    def _infer_output_shapes(self) -> dict[str, tuple[int, ...]]:
        return {"output": (self.num_heads, self.seq_len, self.seq_len)}

    def _validate_dtypes(self) -> None:
        return None

    @property
    def total_memory(self) -> int:
        """Write-only output: ``num_heads * seq_len * seq_len * elem_bytes``."""
        return self.num_heads * self.seq_len * self.seq_len * self.dtype.itemsize

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this instance.

        Mirrors the manifest roofline for ``AlibiFwdOp``:
        ``flops = 3 * H * S * S`` (sub + abs + mul per element) and
        ``bytes = H * S * S * elem_bytes`` (write-only output).
        """
        n_elem = self.num_heads * self.seq_len * self.seq_len
        return 3 * n_elem, self.total_memory

    def forward(self) -> torch.Tensor:
        return self.kernel().reshape(self.num_heads, self.seq_len, self.seq_len)
