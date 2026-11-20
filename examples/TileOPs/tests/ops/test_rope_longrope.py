"""Correctness tests for RopeLongRopeOp (NPU).

Adapted from TileOPs ``tests/ops/test_rope.py`` -- the LongRoPE
(``RopeLongRopeOp``) subset only (the shared GPU file also covers the
GPT-NeoX and Llama 3.1 variants; only this migrated op's cases are ported).

LongRoPE shares the neox rotation with ``RopeNeoxOp`` and differs only in the
frequency tables: per-dimension ``rescale_factors`` (``ext_factors``) multiply
the inverse-frequency divisor, and a scale-dependent amplitude factor
(``scaling_factor``) multiplies the ``cos`` / ``sin`` output.

The op computes cos/sin internally; the test generates only ``x`` and computes
the reference rotation using independently generated LongRoPE frequency tables
(on the same device as ``x``).

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
from tileops.ops.position_encoding.rope_longrope import RopeLongRopeOp
from tileops.testing.test_base import TestBase
from tileops.workloads.position_encoding import RopeLongRopeOpWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


# Reference implementation (pure PyTorch)


def _compute_longrope_freqs(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    rescale_factors: torch.Tensor | None = None,
    max_position_embeddings: int = 4096,
    original_max_position_embeddings: int = 4096,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical LongRoPE frequency computation with amplitude scaling.

    Reference: TVM ``rope_freq_longrope`` in ``position_embedding.py``.

    Key formula:

    - ``divisor = ext_factors[k] * base^(2k/d)`` (ext_factors multiply divisor)
    - ``scaling_factor = sqrt(1 + log(scale) / log(orig_max_pos))`` if scale > 1
    - ``cos`` / ``sin`` are multiplied by ``scaling_factor`` (amplitude factor)

    Returns:
        ``(cos, sin)`` each of shape ``(seq_len, head_dim // 2)``.
    """
    half = head_dim // 2
    dim_indices = torch.arange(0, half, device=device, dtype=torch.float32)
    divisor = base ** (dim_indices / half)

    if rescale_factors is not None:
        rf = rescale_factors.to(device=device, dtype=torch.float32)
        divisor = rf * divisor

    freqs = 1.0 / divisor

    scale = max_position_embeddings / original_max_position_embeddings
    if scale > 1.0:
        scaling_factor = math.sqrt(
            1.0 + math.log(scale) / math.log(original_max_position_embeddings)
        )
    else:
        scaling_factor = 1.0

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_vals = (torch.cos(angles) * scaling_factor).to(dtype)
    sin_vals = (torch.sin(angles) * scaling_factor).to(dtype)
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


class RopeLongRopeTest(RopeLongRopeOpWorkload, TestBase):
    """Generic test fixture for the LongRoPE op.

    The op computes cos/sin internally; the test generates only ``x`` and
    computes the reference rotation using independently generated LongRoPE
    frequency tables (on the same device as ``x``).  LongRoPE shares the neox
    rotation, so the reference rotation is :func:`ref_rope_neox`.
    """

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Independently compute LongRoPE cos/sin for the reference."""
        return _compute_longrope_freqs(
            self.head_dim, self.seq_len, base=self.base,
            rescale_factors=self.rescale_factors,
            max_position_embeddings=self.max_position_embeddings,
            original_max_position_embeddings=self.original_max_position_embeddings,
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


# LongRoPE tests


@RopeBasicFixture
def test_rope_longrope_1d(batch: int, seq_len: int, num_heads: int,
                          head_dim: int, dtype: torch.dtype) -> None:
    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device=_device())
    max_pos = 16384
    orig_max_pos = 4096
    test = RopeLongRopeTest(
        "1d", batch, seq_len, num_heads, head_dim, dtype,
        rescale_factors=rescale,
        max_position_embeddings=max_pos,
        original_max_position_embeddings=orig_max_pos,
    )
    op = RopeLongRopeOp(
        layout="1d", rescale_factors=rescale,
        max_position_embeddings=max_pos,
        original_max_position_embeddings=orig_max_pos,
    )
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeBasicFixture
def test_rope_longrope_2d(batch: int, seq_len: int, num_heads: int,
                          head_dim: int, dtype: torch.dtype) -> None:
    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device=_device())
    max_pos = 16384
    orig_max_pos = 4096
    test = RopeLongRopeTest(
        "2d", batch, seq_len, num_heads, head_dim, dtype,
        rescale_factors=rescale,
        max_position_embeddings=max_pos,
        original_max_position_embeddings=orig_max_pos,
    )
    op = RopeLongRopeOp(
        layout="2d", rescale_factors=rescale,
        max_position_embeddings=max_pos,
        original_max_position_embeddings=orig_max_pos,
    )
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeEdgeFixture
def test_rope_longrope_edge(batch: int, seq_len: int, num_heads: int,
                            head_dim: int, dtype: torch.dtype) -> None:
    """Edge cases: seq_len=1 and longer sequences."""
    test = RopeLongRopeTest("2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeLongRopeOp(layout="2d")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
