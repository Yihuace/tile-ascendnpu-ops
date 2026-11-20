"""Workload definitions for the position-encoding op family (NPU).

Ported from the GPU ``RopeWorkload`` in
``{gpu_repo_root}/workloads/rope.py``.  The GPU file hosts a single workload
class shared by all five RoPE variants (``variant`` is carried but only used by
the tests); this NPU port carries one workload class per ported op, so the
``variant`` field is dropped.  ``RopeNeoxOpWorkload`` and
``RopeLlama31OpWorkload`` and ``RopeLongRopeOpWorkload`` are carried here.

The workload draws one ``randn`` input tensor ``x`` -- the ``cos`` / ``sin``
frequency tables are computed inside the Op, not supplied here.

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"`` in the original TileOPs code.
"""

from typing import Optional

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase

__all__ = ["RopeLlama31OpWorkload", "RopeLongRopeOpWorkload", "RopeNeoxOpWorkload"]


class RopeNeoxOpWorkload(WorkloadBase):
    """Workload for the GPT-NeoX style RoPE op.

    Args:
        layout: ``"1d"`` for ``(seq_len, head_dim)`` or ``"2d"`` for
            ``(batch, seq_len, num_heads, head_dim)``.
        batch: Batch size (used only by the 2d layout).
        seq_len: Sequence length.
        num_heads: Number of heads (used only by the 2d layout).
        head_dim: Head dimension (must be even).
        dtype: Data type (float16, bfloat16, or float32).
        base: Frequency base (carried so the test can reproduce the reference
            frequency tables).
    """

    def __init__(
        self,
        layout: str,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        base: float = 10000.0,
    ):
        self.layout = layout
        self.batch = batch
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.base = base

    def gen_inputs(self) -> tuple[torch.Tensor]:
        """Generate only ``x``; cos/sin are computed by the op internally."""
        # W1: device resolved via the device backend (was device="cuda").
        device = get_device_backend().name
        if self.layout == "1d":
            x = torch.randn(self.seq_len, self.head_dim, device=device, dtype=self.dtype)
        else:
            x = torch.randn(
                self.batch, self.seq_len, self.num_heads, self.head_dim,
                device=device, dtype=self.dtype,
            )
        return (x,)


class RopeLlama31OpWorkload(WorkloadBase):
    """Workload for the Llama 3.1 RoPE op.

    Llama 3.1 shares the neox rotation with :class:`RopeNeoxOpWorkload`; the
    only difference is the piecewise-scaled frequency computation, which is
    handled by the Op layer.  The extra Llama 3.1 parameters are carried so the
    tests can independently reproduce the same reference frequency tables.

    Args:
        layout: ``"1d"`` for ``(seq_len, head_dim)`` or ``"2d"`` for
            ``(batch, seq_len, num_heads, head_dim)``.
        batch: Batch size (used only by the 2d layout).
        seq_len: Sequence length.
        num_heads: Number of heads (used only by the 2d layout).
        head_dim: Head dimension (must be even).
        dtype: Data type (float16, bfloat16, or float32).
        base: Frequency base (carried so the test can reproduce the reference
            frequency tables).
        scale_factor: Scaling factor for low frequencies.
        low_freq_factor: Threshold for low-frequency wavelengths.
        high_freq_factor: Threshold for high-frequency wavelengths.
        original_max_position: Original maximum position length.
    """

    def __init__(
        self,
        layout: str,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        base: float = 10000.0,
        scale_factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position: int = 8192,
    ):
        self.layout = layout
        self.batch = batch
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.base = base
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position = original_max_position

    def gen_inputs(self) -> tuple[torch.Tensor]:
        """Generate only ``x``; cos/sin are computed by the op internally."""
        # W1: device resolved via the device backend (was device="cuda").
        device = get_device_backend().name
        if self.layout == "1d":
            x = torch.randn(self.seq_len, self.head_dim, device=device, dtype=self.dtype)
        else:
            x = torch.randn(
                self.batch, self.seq_len, self.num_heads, self.head_dim,
                device=device, dtype=self.dtype,
            )
        return (x,)


class RopeLongRopeOpWorkload(WorkloadBase):
    """Workload for the LongRoPE op.

    LongRoPE shares the neox rotation with :class:`RopeNeoxOpWorkload`; the only
    difference is the per-dimension rescaled frequency computation, which is
    handled by the Op layer.  The extra LongRoPE parameters are carried so the
    tests can independently reproduce the same reference frequency tables.

    Args:
        layout: ``"1d"`` for ``(seq_len, head_dim)`` or ``"2d"`` for
            ``(batch, seq_len, num_heads, head_dim)``.
        batch: Batch size (used only by the 2d layout).
        seq_len: Sequence length.
        num_heads: Number of heads (used only by the 2d layout).
        head_dim: Head dimension (must be even).
        dtype: Data type (float16, bfloat16, or float32).
        base: Frequency base (carried so the test can reproduce the reference
            frequency tables).
        rescale_factors: Per-dimension rescale factors (``ext_factors``) of
            shape ``(head_dim // 2,)``; ``None`` means no rescaling.
        max_position_embeddings: Extended max position length.
        original_max_position_embeddings: Original max position length.
    """

    def __init__(
        self,
        layout: str,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        base: float = 10000.0,
        rescale_factors: Optional[torch.Tensor] = None,
        max_position_embeddings: int = 4096,
        original_max_position_embeddings: int = 4096,
    ):
        self.layout = layout
        self.batch = batch
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.base = base
        self.rescale_factors = rescale_factors
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings

    def gen_inputs(self) -> tuple[torch.Tensor]:
        """Generate only ``x``; cos/sin are computed by the op internally."""
        # W1: device resolved via the device backend (was device="cuda").
        device = get_device_backend().name
        if self.layout == "1d":
            x = torch.randn(self.seq_len, self.head_dim, device=device, dtype=self.dtype)
        else:
            x = torch.randn(
                self.batch, self.seq_len, self.num_heads, self.head_dim,
                device=device, dtype=self.dtype,
            )
        return (x,)
