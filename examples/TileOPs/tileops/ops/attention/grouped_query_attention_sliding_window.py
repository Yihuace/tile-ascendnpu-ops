"""GroupedQueryAttentionSlidingWindowFwdOp (fixed-length GQA sliding-window forward).

Adaptation from GPU (TileOPs) to NPU:

- O1: ``t.device.type != 'cuda'`` device check -> ``backend.is_device_tensor(t)``
  via :func:`get_device_backend`.
- O2: device name in the error message comes from ``backend.name``.
- O3: ``tune`` parameter removed from ``__init__``; the kernel constructor
  takes no ``tune`` argument.
- O6: op-specific flow preserved unchanged -- shape/window validation ->
  per-input device/dtype check -> ``contiguous`` -> kernel dispatch ->
  ``forward`` (the kernel returns ``(output, lse)`` and the op returns the
  attention output only).  ``total_flops`` / ``total_memory`` keep the GPU
  arithmetic; ``eval_roofline`` is added to expose them via the NPU ``Op``
  interface.
- K5: the GPU ``default_kernel_map`` selects the Hopper warp-specialized
  ``GQASlidingWindowFwdWgmmaPipelinedKernel`` when ``is_hopper()`` else the
  architecture-generic ``GQASlidingWindowFwdKernel``; NPU collapses to the
  architecture-generic kernel.
"""

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.attention.grouped_query_attention_sliding_window import (
    GQASlidingWindowFwdKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["GroupedQueryAttentionSlidingWindowFwdOp"]


class GroupedQueryAttentionSlidingWindowFwdOp(Op):
    """Fixed-length GQA forward with sliding window attention.

    Token at ``q_pos`` attends to ``k_pos`` when ALL applicable conditions hold:

      - ``k_pos <= q_pos``                          (``is_causal=True``)
      - ``k_pos >= q_pos - window_size_left``       (``window_size_left >= 0``)
      - ``k_pos <= q_pos + window_size_right``      (``window_size_right >= 0``)

    Use ``window_size_left=-1`` / ``window_size_right=-1`` for no restriction.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        heads_kv: Number of KV heads (must divide heads evenly).
        seq_len: Sequence length (same for Q, K, V).
        dim: Head dimension.
        is_causal: Whether to apply causal masking.
        window_size_left: Left window size (-1 = unlimited).
        window_size_right: Right window size (-1 = unlimited).
        dtype: Tensor data type.
        kernel_map: Optional override for hardware-specific kernel dispatch.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool = True,
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if window_size_left != -1 and window_size_left < 0:
            raise ValueError(
                f"window_size_left must be -1 (unlimited) or >= 0, got {window_size_left}"
            )
        if window_size_right != -1 and window_size_right < 0:
            raise ValueError(
                f"window_size_right must be -1 (unlimited) or >= 0, got {window_size_right}"
            )
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        # O3: the kernel constructor takes no ``tune`` argument.
        self.kernel = self.kernel_map["gqa_sliding_window_fwd"](
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            seq_len=seq_len,
            dim=dim,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            dtype=dtype,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # K5: the GPU selects the Hopper warp-specialized variant via
        # ``is_hopper()``; NPU uses the architecture-generic kernel.
        return {"gqa_sliding_window_fwd": GQASlidingWindowFwdKernel}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Run fixed-length GQA sliding window forward.

        Args:
            q: Query tensor, shape [batch, seq_len, heads, dim].
            k: Key tensor, shape [batch, seq_len, heads_kv, dim].
            v: Value tensor, shape [batch, seq_len, heads_kv, dim].

        Returns:
            Output tensor, shape [batch, seq_len, heads, dim].
        """
        backend = get_device_backend()
        for t, name in [(q, "q"), (k, "k"), (v, "v")]:
            # O1/O2: device check via the device backend instead of
            # ``t.device.type != 'cuda'``.
            if not backend.is_device_tensor(t):
                raise ValueError(
                    f"{name} must be a {backend.name} tensor, got device {t.device}"
                )
            if t.dtype != self.dtype:
                raise ValueError(
                    f"{name} dtype {t.dtype} does not match op dtype {self.dtype}"
                )
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        if q.shape != (self.batch, self.seq_len, self.heads, self.dim):
            raise ValueError(
                f"q shape {q.shape} does not match expected "
                f"({self.batch}, {self.seq_len}, {self.heads}, {self.dim})"
            )
        if k.shape != (self.batch, self.seq_len, self.heads_kv, self.dim):
            raise ValueError(
                f"k shape {k.shape} does not match expected "
                f"({self.batch}, {self.seq_len}, {self.heads_kv}, {self.dim})"
            )
        if v.shape != (self.batch, self.seq_len, self.heads_kv, self.dim):
            raise ValueError(
                f"v shape {v.shape} does not match expected "
                f"({self.batch}, {self.seq_len}, {self.heads_kv}, {self.dim})"
            )

        output, _ = self.kernel(q, k, v)
        return output

    @property
    def total_flops(self) -> int:
        """Approximate FLOPs for QK^T and PV GEMMs (device-agnostic)."""
        S = self.seq_len
        wl = self.window_size_left
        wr = self.window_size_right
        total_attended = 0
        for q in range(S):
            hi = q if self.is_causal else (min(S - 1, q + wr) if wr >= 0 else S - 1)
            lo = max(0, q - wl) if wl >= 0 else 0
            total_attended += hi - lo + 1
        return 4 * self.batch * self.heads * total_attended * self.dim

    @property
    def total_memory(self) -> int:
        """Approximate bytes accessed: read Q/K/V, write O (device-agnostic)."""
        elem = torch.tensor([], dtype=self.dtype).element_size()
        return 2 * self.batch * self.seq_len * (self.heads + self.heads_kv) * self.dim * elem

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the configured workload shape.

        Ported verbatim (device-agnostic arithmetic) from the GPU repo's
        ``tileops.perf.formulas.gqa_sliding_window_fwd_roofline``:
        ``4 * B * H * total_attended * D`` FLOPs (``total_attended`` counts,
        per query position, the key positions inside the causal / sliding
        window) and ``2 * B * S * (H + H_kv) * D * elem_bytes`` bytes.
        """
        return int(self.total_flops), int(self.total_memory)
