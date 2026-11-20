"""Correctness tests for RopeLlama31Op (NPU).

Adapted from TileOPs ``tests/ops/test_rope.py`` -- the Llama 3.1
(``RopeLlama31Op``) subset only (the shared GPU file also covers the GPT-NeoX
and LongRoPE variants; only this migrated op's cases are ported).

Llama 3.1 shares the neox rotation with ``RopeNeoxOp`` and differs only in the
piecewise-scaled frequency tables: high-frequency dims keep their frequency,
low-frequency dims are divided by ``scale_factor``, and intermediate dims blend
the two smoothly.

The op computes cos/sin internally; the test generates only ``x`` and computes
the reference rotation using independently generated Llama 3.1 frequency
tables (on the same device as ``x``).

Adaptations from GPU (TileOPs) to NPU:

- T1: tensor creation uses ``get_device_backend().name`` instead of the
  hard-coded ``"cuda"``.
- T2: the GPU ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)``
  guard is dropped (the NPU device backend resolves the target).
- T3/T4: the GPU ``tune`` parametrization and autotune try/except are
  omitted -- the NPU kernel uses heuristic config selection only.
"""

import math

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.position_encoding.rope_llama31 import RopeLlama31Op
from tileops.testing.test_base import TestBase
from tileops.workloads.position_encoding import RopeLlama31OpWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


# Reference implementation (pure PyTorch)


def _compute_llama31_freqs(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    scale_factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    original_max_position: int = 8192,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Llama 3.1 piecewise-scaled frequency computation.

    Reference: Meta Llama 3.1 model implementation.

    Key formula:

    - high-frequency dims (``wavelen < high_freq_wavelen``) keep the frequency
    - low-frequency dims (``wavelen > low_freq_wavelen``) divide by ``scale_factor``
    - intermediate dims blend the two smoothly

    Returns:
        ``(cos, sin)`` each of shape ``(seq_len, head_dim // 2)``.
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))

    low_freq_wavelen = original_max_position / low_freq_factor
    high_freq_wavelen = original_max_position / high_freq_factor

    scaled_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq.item()
        if wavelen < high_freq_wavelen:
            scaled_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            scaled_freqs.append(freq / scale_factor)
        else:
            smooth = (original_max_position / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor)
            scaled_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

    freqs = torch.stack(scaled_freqs)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_vals = torch.cos(angles).to(dtype)
    sin_vals = torch.sin(angles).to(dtype)
    return cos_vals, sin_vals


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    """Neox-style rotation: split at the midpoint and negate the first half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def ref_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference neox RoPE: full-dim cos/sin broadcast with half rotation."""
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    if x.ndim == 2:
        return (x.float() * cos_full.float()
                + _rotate_half_neox(x).float() * sin_full.float()).to(x.dtype)
    elif x.ndim == 4:
        cos_full = cos_full.unsqueeze(0).unsqueeze(2)
        sin_full = sin_full.unsqueeze(0).unsqueeze(2)
        return (x.float() * cos_full.float()
                + _rotate_half_neox(x).float() * sin_full.float()).to(x.dtype)
    else:
        raise ValueError(f"Unsupported ndim={x.ndim}")


class RopeLlama31Test(RopeLlama31OpWorkload, TestBase):
    """Generic test fixture for the Llama 3.1 RoPE op.

    The op computes cos/sin internally; the test generates only ``x`` and
    computes the reference rotation using independently generated Llama 3.1
    piecewise-scaled frequency tables (on the same device as ``x``).  Llama 3.1
    shares the neox rotation, so the reference rotation is
    :func:`ref_rope_neox`.
    """

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Independently compute Llama 3.1 cos/sin for the reference."""
        return _compute_llama31_freqs(
            self.head_dim, self.seq_len, base=self.base,
            scale_factor=self.scale_factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            original_max_position=self.original_max_position,
            dtype=self.dtype, device=_device(),
        )

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        """Pure-PyTorch reference: independently computes cos/sin and rotates."""
        cos, sin = self._compute_cos_sin()
        return ref_rope_neox(x, cos, sin)


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:
        return 1.6e-2, 1.6e-2


# Fixtures


class RopeBasicFixture(FixtureBase):
    """Basic RoPE fixture: shapes x dtypes."""
    PARAMS = [
        ("batch, seq_len, num_heads, head_dim, dtype", [
            pytest.param(2, 128, 8, 64, torch.float32,
                         marks=[pytest.mark.smoke, pytest.mark.packaging]),
            pytest.param(2, 128, 8, 64, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 128, 8, 64, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1, 256, 4, 128, torch.float16, marks=pytest.mark.full),
        ]),
    ]


class RopeEdgeFixture(FixtureBase):
    """Edge case fixture: seq_len=1, small head_dim."""
    PARAMS = [
        ("batch, seq_len, num_heads, head_dim, dtype", [
            pytest.param(1, 1, 1, 16, torch.float32, marks=pytest.mark.smoke),
            pytest.param(1, 1, 1, 16, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 512, 8, 64, torch.float16, marks=pytest.mark.full),
        ]),
    ]


# Llama 3.1 tests


@RopeBasicFixture
def test_rope_llama31_1d(batch: int, seq_len: int, num_heads: int,
                         head_dim: int, dtype: torch.dtype) -> None:
    extra = {"scale_factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
             "original_max_position": 8192}
    test = RopeLlama31Test("1d", batch, seq_len, num_heads, head_dim, dtype, **extra)
    op = RopeLlama31Op(layout="1d", **extra)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeBasicFixture
def test_rope_llama31_2d(batch: int, seq_len: int, num_heads: int,
                         head_dim: int, dtype: torch.dtype) -> None:
    extra = {"scale_factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
             "original_max_position": 8192}
    test = RopeLlama31Test("2d", batch, seq_len, num_heads, head_dim, dtype, **extra)
    op = RopeLlama31Op(layout="2d", **extra)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeEdgeFixture
def test_rope_llama31_edge(batch: int, seq_len: int, num_heads: int,
                           head_dim: int, dtype: torch.dtype) -> None:
    """Edge cases: seq_len=1 and longer sequences."""
    test = RopeLlama31Test("2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeLlama31Op(layout="2d")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
