"""Correctness tests for RopeNeoxOp (NPU).

Adapted from TileOPs ``tests/ops/test_rope.py`` -- the GPT-NeoX (``neox``)
subset only (the shared GPU file also covers LongRoPE and Llama 3.1; only
this migrated op's cases are ported).

The op computes cos/sin internally; the test generates only ``x`` and computes
the reference rotation using independently generated frequency tables (on the
same device as ``x``).

Adaptations from GPU (TileOPs) to NPU:

- T1: tensor creation uses ``get_device_backend().name`` instead of the
  hard-coded ``"cuda"``.
- T2: the GPU ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)``
  guard is dropped (the NPU device backend resolves the target).
- T3/T4: the GPU ``tune`` parametrization and autotune try/except are
  omitted — the NPU kernel uses heuristic config selection only.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.position_encoding.rope_neox import RopeNeoxOp
from tileops.testing.test_base import TestBase
from tileops.workloads.position_encoding import RopeNeoxOpWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


# Reference implementation (pure PyTorch)


def _compute_freqs_cis_base(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute standard RoPE cos/sin tables.

    Returns:
        ``(cos, sin)`` each of shape ``(seq_len, head_dim // 2)``.
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
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


class RopeTest(RopeNeoxOpWorkload, TestBase):
    """Generic test fixture for the GPT-NeoX RoPE op.

    The op computes cos/sin internally; the test generates only ``x`` and
    computes the reference rotation using independently generated frequency
    tables (on the same device as ``x``).
    """

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Independently compute cos/sin for the reference implementation."""
        return _compute_freqs_cis_base(
            self.head_dim, self.seq_len, base=self.base,
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


# Neox RoPE tests


@RopeBasicFixture
def test_rope_neox_1d(batch: int, seq_len: int, num_heads: int,
                      head_dim: int, dtype: torch.dtype) -> None:
    test = RopeTest("1d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeNeoxOp(layout="1d")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeBasicFixture
def test_rope_neox_2d(batch: int, seq_len: int, num_heads: int,
                      head_dim: int, dtype: torch.dtype) -> None:
    test = RopeTest("2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeNeoxOp(layout="2d")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeEdgeFixture
def test_rope_neox_edge(batch: int, seq_len: int, num_heads: int,
                        head_dim: int, dtype: torch.dtype) -> None:
    """Edge cases: seq_len=1 and longer sequences."""
    test = RopeTest("2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeNeoxOp(layout="2d")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
