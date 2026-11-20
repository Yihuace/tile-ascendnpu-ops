"""GroupedQueryAttentionSlidingWindowVarlenFwdOp (variable-length GQA sliding-window forward).

Adaptation from GPU (TileOPs) to NPU:

- O1: ``t.device.type != 'cuda'`` device check -> ``backend.is_device_tensor(t)``
  via :func:`get_device_backend`.
- O2: device name in the error message comes from ``backend.name``.
- O3: ``tune`` parameter removed from ``__init__``; the kernel constructor
  takes no ``tune`` argument.
- O4: ``from tileops.utils import get_sm_version`` replaced by
  ``from tileops.device import get_device_backend``.
- O5: ``from .compile_boundary import register_instance`` removed (not used on
  NPU).
- O6: op-specific flow preserved unchanged -- shape/window/per-sample
  validation -> per-input device/dtype check -> ``contiguous`` -> kernel
  dispatch.  ``forward`` returns the attention output only (the kernel also
  produces an ``lse`` that the op drops).  ``total_flops`` / ``total_memory``
  keep the GPU roofline arithmetic (device-agnostic); ``eval_roofline``
  exposes them via the NPU ``Op`` interface.
- K5: the GPU ``default_kernel_map`` selects the Hopper warp-specialized
  ``GQASlidingWindowVarlenFwdWgmmaPipelinedKernel`` when ``is_hopper()`` else
  the architecture-generic ``GQASlidingWindowVarlenFwdKernel``; NPU collapses
  to the architecture-generic kernel.
"""

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.attention.grouped_query_attention_sliding_window_varlen import (
    GQASlidingWindowVarlenFwdKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["GroupedQueryAttentionSlidingWindowVarlenFwdOp"]


def _distribute_total(total: int, batch: int, max_len: int) -> list[int]:
    """Deterministically split *total* into *batch* lengths, each ``<= max_len``.

    Ported verbatim from the GPU ``tileops.perf.formulas._distribute_total``:
    used as the roofline fallback when per-sample ``q_lens`` / ``k_lens`` are
    not supplied.
    """
    lengths = [0] * batch
    remaining = total
    for idx in range(batch):
        slots_left = batch - idx - 1
        value = min(max_len, remaining - slots_left)
        lengths[idx] = value
        remaining -= value
    return lengths


class GroupedQueryAttentionSlidingWindowVarlenFwdOp(Op):
    """Variable-length GQA forward with sliding window attention.

    Inputs are packed (no padding); per-sample boundaries are given via
    cu_seqlens arrays.  seqlen_q and seqlen_k may differ per sample:

      offset = seqlen_k - seqlen_q  (per sample, FA3 bottom-right convention)

    A token at local q_pos attends to local k_pos when ALL conditions hold:
      k_pos <= q_pos + offset                      (is_causal=True)
      k_pos >= q_pos + offset - window_size_left   (window_size_left >= 0)
      k_pos <= q_pos + offset + window_size_right  (window_size_right >= 0)

    Args:
        batch: Number of sequences in the batch.
        heads: Number of query heads.
        heads_kv: Number of KV heads (must divide heads evenly).
        dim: Head dimension.
        is_causal: Whether to apply causal masking.
        window_size_left: Left window size (-1 = unlimited).
        window_size_right: Right window size (-1 = unlimited).
        dtype: Tensor data type.
        accum_dtype: Accumulator data type for intermediate computations.
        kernel_map: Optional override for hardware-specific kernel dispatch.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool = True,
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
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
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype
        self.accum_dtype = accum_dtype

        # total_q / total_k / per-sample lengths are not known at init; callers
        # (e.g. the benchmark) set them per workload before invoking
        # ``eval_roofline``.  The GPU op leaves these unset and reads them from
        # the instance at roofline time -- preserved here.
        self.total_q: Optional[int] = None
        self.total_k: Optional[int] = None
        self.q_lens: Optional[list[int]] = None
        self.k_lens: Optional[list[int]] = None
        self.max_seqlen_q: Optional[int] = None
        self.max_seqlen_k: Optional[int] = None

        self.dispatch_kernel(kernel_map)
        # O3: the kernel constructor takes no ``tune`` argument.
        self.kernel = self.kernel_map["gqa_sliding_window_varlen_fwd"](
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            dtype=dtype,
            accum_dtype=accum_dtype,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # K5: the GPU selects the Hopper warp-specialized variant via
        # ``is_hopper()``; NPU uses the architecture-generic kernel.
        return {"gqa_sliding_window_varlen_fwd": GQASlidingWindowVarlenFwdKernel}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        """Run variable-length GQA sliding window forward.

        Args:
            q: Query tensor, shape [total_q, heads, dim].
            k: Key tensor, shape [total_k, heads_kv, dim].
            v: Value tensor, shape [total_k, heads_kv, dim].
            cu_seqlens_q: Cumulative Q lengths, shape [batch+1], dtype int32.
            cu_seqlens_k: Cumulative K lengths, shape [batch+1], dtype int32.
            max_seqlen_q: Maximum Q sequence length across the batch.

        Returns:
            Output tensor, shape [total_q, heads, dim].
        """
        backend = get_device_backend()
        for t, name in [(q, "q"), (k, "k"), (v, "v")]:
            # O1/O2: device check via the device backend instead of
            # ``t.device.type != 'cuda'``.
            if not backend.is_device_tensor(t):
                raise ValueError(
                    f"{name} must be a {backend.name} device, got {t.device}"
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

        if q.ndim != 3 or q.shape[1] != self.heads or q.shape[2] != self.dim:
            raise ValueError(
                f"q shape {q.shape} incompatible with heads={self.heads}, dim={self.dim}"
            )
        if k.ndim != 3 or k.shape[1] != self.heads_kv or k.shape[2] != self.dim:
            raise ValueError(
                f"k shape {k.shape} incompatible with heads_kv={self.heads_kv}"
                f", dim={self.dim}"
            )
        if v.ndim != 3 or v.shape[1] != self.heads_kv or v.shape[2] != self.dim:
            raise ValueError(
                f"v shape {v.shape} incompatible with heads_kv={self.heads_kv}"
                f", dim={self.dim}"
            )
        if cu_seqlens_q.shape[0] != self.batch + 1:
            raise ValueError(
                f"cu_seqlens_q.shape[0] ({cu_seqlens_q.shape[0]}) must equal "
                f"batch+1 ({self.batch + 1})"
            )
        if cu_seqlens_k.shape[0] != self.batch + 1:
            raise ValueError(
                f"cu_seqlens_k.shape[0] ({cu_seqlens_k.shape[0]}) must equal "
                f"batch+1 ({self.batch + 1})"
            )
        for cu, name in [(cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")]:
            if not backend.is_device_tensor(cu):
                raise ValueError(
                    f"{name} must be a {backend.name} device, got {cu.device}"
                )
            if cu.dtype != torch.int32:
                raise ValueError(
                    f"{name} must have dtype int32, got {cu.dtype}"
                )
            if not cu.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if cu_seqlens_q[0].item() != 0:
            raise ValueError(f"cu_seqlens_q[0] must be 0, got {cu_seqlens_q[0].item()}")
        if cu_seqlens_k[0].item() != 0:
            raise ValueError(f"cu_seqlens_k[0] must be 0, got {cu_seqlens_k[0].item()}")
        if not torch.all(cu_seqlens_q[1:] >= cu_seqlens_q[:-1]):
            raise ValueError("cu_seqlens_q must be non-decreasing")
        if not torch.all(cu_seqlens_k[1:] >= cu_seqlens_k[:-1]):
            raise ValueError("cu_seqlens_k must be non-decreasing")
        if cu_seqlens_q[-1].item() > q.shape[0]:
            raise ValueError(
                f"cu_seqlens_q[-1] ({cu_seqlens_q[-1].item()}) exceeds "
                f"q.shape[0] ({q.shape[0]})"
            )
        if cu_seqlens_k[-1].item() > k.shape[0]:
            raise ValueError(
                f"cu_seqlens_k[-1] ({cu_seqlens_k[-1].item()}) exceeds "
                f"k.shape[0] ({k.shape[0]})"
            )
        actual_max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        if max_seqlen_q < actual_max_q:
            raise ValueError(
                f"max_seqlen_q ({max_seqlen_q}) must be >= actual max Q "
                f"sequence length ({actual_max_q})"
            )

        output, _ = self.kernel(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q
        )
        return output

    def _effective_q_lens(self) -> list[int]:
        """Per-sample Q lengths (explicit ``q_lens`` or a deterministic split)."""
        if self.q_lens is not None:
            return [int(x) for x in self.q_lens]
        if self.total_q is None:
            raise ValueError(
                "GroupedQueryAttentionSlidingWindowVarlenFwdOp: total_q (or q_lens) "
                "must be set per workload before eval_roofline()."
            )
        max_len = self.max_seqlen_q or (self.total_q // self.batch if self.batch else self.total_q)
        return _distribute_total(int(self.total_q), self.batch, int(max_len))

    def _effective_k_lens(self) -> list[int]:
        """Per-sample K lengths (explicit ``k_lens`` or a deterministic split)."""
        if self.k_lens is not None:
            return [int(x) for x in self.k_lens]
        if self.total_k is None:
            raise ValueError(
                "GroupedQueryAttentionSlidingWindowVarlenFwdOp: total_k (or k_lens) "
                "must be set per workload before eval_roofline()."
            )
        max_len = self.max_seqlen_k or (self.total_k // self.batch if self.batch else self.total_k)
        return _distribute_total(int(self.total_k), self.batch, int(max_len))

    @property
    def total_flops(self) -> int:
        """FLOPs for the configured workload (device-agnostic).

        Ported from the GPU ``tileops.perf.formulas.gqa_sliding_window_varlen_fwd_roofline``:
        ``4 * heads * total_attended * dim`` where ``total_attended`` counts, per
        query position and sample, the key positions inside the causal / sliding
        window (bottom-right aligned by ``offset = seqlen_k - seqlen_q``).
        """
        q_lens = self._effective_q_lens()
        k_lens = self._effective_k_lens()
        wl = self.window_size_left
        wr = self.window_size_right
        total_attended = 0
        for sq, sk in zip(q_lens, k_lens, strict=True):
            offset = sk - sq
            for q_pos in range(sq):
                hi = (
                    min(q_pos + offset, sk - 1)
                    if self.is_causal
                    else (min(q_pos + offset + wr, sk - 1) if wr >= 0 else sk - 1)
                )
                lo = max(0, q_pos + offset - wl) if wl >= 0 else 0
                total_attended += max(0, hi - lo + 1)
        return 4 * self.heads * total_attended * self.dim

    @property
    def total_memory(self) -> int:
        """Bytes accessed: read Q/K/V, write O (device-agnostic).

        Ported from the GPU ``gqa_sliding_window_varlen_fwd_roofline``:
        ``(total_q*H*D + 2*total_k*H_kv*D + total_q*H*D) * elem_bytes``.
        """
        q_lens = self._effective_q_lens()
        k_lens = self._effective_k_lens()
        total_q = sum(q_lens)
        total_k = sum(k_lens)
        elem = torch.tensor([], dtype=self.dtype).element_size()
        return (
            total_q * self.heads * self.dim
            + 2 * total_k * self.heads_kv * self.dim
            + total_q * self.heads * self.dim
        ) * elem

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the configured workload shape."""
        return int(self.total_flops), int(self.total_memory)
