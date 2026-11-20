"""Correctness tests for cumulative ops -- CumprodFwdOp subset (NPU).

Adapted from TileOPs ``tests/ops/test_cumulative.py`` -- the CumprodFwdOp
subset only (the shared GPU file also covers CumsumFwdOp, which is not
migrated).  Each op computes an inclusive prefix product along dim and
supports 1D-4D input.  Output has the same shape as input.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses ``get_device_backend().name`` instead of the
  hard-coded ``device="cuda"``.
- T2: the GPU cumprod tests have no ``torch.cuda`` availability guard --
  nothing to remove.
- T3: ``tune`` parametrization dimension removed.
- T4: no autotune try/except to remove.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.scan.cumprod import CumprodFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.scan import CumprodFwdWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


def _cumprod_tol(dtype: torch.dtype) -> dict:
    """Tolerances for cumprod tests (more numerically sensitive)."""
    if dtype == torch.float32:
        return {"atol": 1e-3, "rtol": 1e-3}
    return {"atol": 5e-2, "rtol": 5e-2}


# Fixtures


class CumprodBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(
                    128, 512, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
                ),
                pytest.param(128, 512, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(128, 512, torch.bfloat16, marks=[pytest.mark.smoke]),
                pytest.param(256, 4096, torch.float16, marks=[pytest.mark.full]),
                pytest.param(256, 4096, torch.bfloat16, marks=[pytest.mark.full]),
                # Non-aligned N (non-pow2)
                pytest.param(128, 300, torch.float16, marks=[pytest.mark.full]),
                pytest.param(128, 300, torch.bfloat16, marks=[pytest.mark.full]),
                # Tail-M: M not divisible by block_m
                pytest.param(129, 512, torch.float16, marks=[pytest.mark.full]),
            ],
        ),
    ]


class CumprodNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(128, 512, torch.bfloat16, marks=[pytest.mark.smoke]),
            ],
        ),
    ]


class Cumprod3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(2, 64, 512, torch.bfloat16, marks=[pytest.mark.smoke]),
            ],
        ),
    ]


class Cumprod4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=[pytest.mark.smoke]),
            ],
        ),
    ]


class Cumprod1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float32, marks=[pytest.mark.smoke]),
                pytest.param(512, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(512, torch.bfloat16, marks=[pytest.mark.smoke]),
            ],
        ),
    ]


class CumprodDimAxis1Fixture(FixtureBase):
    PARAMS = [
        (
            "batch, hidden, seq, dtype",
            [
                pytest.param(2, 512, 256, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(2, 512, 256, torch.bfloat16, marks=[pytest.mark.smoke]),
            ],
        ),
    ]


# TestBase helper


class CumprodFwdTest(CumprodFwdWorkload, TestBase):
    """Parameterized test helper for CumprodFwdOp."""

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().cumprod(dim=-1).to(x.dtype)


# CumprodFwdOp tests


@CumprodBasicFixture
def test_cumprod_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = CumprodFwdTest((m, n), dtype, use_small_range=True)
    op = CumprodFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_cumprod_tol(dtype))


@CumprodNonContigFixture
def test_cumprod_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = torch.rand(m, n * 2, dtype=dtype, device=_device()) * 0.01 + 0.99
    x = x_full[:, :n]
    op = CumprodFwdOp(dtype=dtype)
    ref = x.contiguous().float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@Cumprod3DFixture
def test_cumprod_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.rand(batch, seq, hidden, dtype=dtype, device=_device()) * 0.01 + 0.99
    op = CumprodFwdOp(dtype=dtype)
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D cumprod max err: {(y - ref).abs().max()}"


@Cumprod4DFixture
def test_cumprod_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.rand(b0, b1, b2, n, dtype=dtype, device=_device()) * 0.01 + 0.99
    op = CumprodFwdOp(dtype=dtype)
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"4D cumprod max err: {(y - ref).abs().max()}"


@Cumprod1DFixture
def test_cumprod_1d(n: int, dtype: torch.dtype) -> None:
    x = torch.rand(n, dtype=dtype, device=_device()) * 0.01 + 0.99
    op = CumprodFwdOp(dtype=dtype)
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"1D cumprod max err: {(y - ref).abs().max()}"


@CumprodDimAxis1Fixture
def test_cumprod_dim_axis1(
    batch: int, hidden: int, seq: int, dtype: torch.dtype
) -> None:
    """Cumprod along dim=1 (3D) — exercises movedim choreography in ``_run``."""
    # Values close to 1 to avoid over/underflow in cumprod over hidden dim.
    x = torch.rand(batch, hidden, seq, dtype=dtype, device=_device()) * 0.01 + 0.99
    op = CumprodFwdOp(dtype=dtype, dim=1)
    ref = x.float().cumprod(dim=1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), \
        f"cumprod dim=1 max err: {(y - ref).abs().max()}"


@pytest.mark.smoke
def test_cumprod_dynamic_shape_kernel_cache() -> None:
    op = CumprodFwdOp()
    x1 = torch.rand(4, 8, dtype=torch.float16, device=_device())
    x2 = torch.rand(5, 8, dtype=torch.float16, device=_device())

    op(x1)
    assert len(op._kernel_cache) == 1
    op(x1)
    assert len(op._kernel_cache) == 1
    op(x2)
    assert len(op._kernel_cache) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
