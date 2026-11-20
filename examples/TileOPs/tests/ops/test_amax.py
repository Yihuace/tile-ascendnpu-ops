"""Correctness tests for AmaxFwdOp (NPU).

Per-op test file: the shared GPU ``test_reduce.py`` was split so that only this
migration unit's cases run when ``AmaxFwdOp`` is integrated (the shared file
also carried AminFwdOp cases, which belong to a separate migration unit).
Uses the device backend instead of the hard-coded ``"cuda"`` (T1).

Covers: AmaxFwdOp (element-wise maximum reduction).
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.testing.test_base import TestBase
from tileops.workloads.reduction import AmaxWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


# Fixtures


class ReduceBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(
                    128,
                    512,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                ),
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-aligned N
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class SpecReduceFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype",
            [
                pytest.param((128, 512), -1, False, torch.float16, marks=pytest.mark.smoke),
                pytest.param((4, 32, 512), -1, False, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param((128, 512), -1, True, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), 0, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), 1, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), -1, True, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


# TestBase helper -- inherits gen_inputs() from the workload class


class ReduceTest(AmaxWorkload, TestBase):
    """Parameterized test helper for AmaxFwdOp (dim=-1, 2-D input)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        super().__init__((m, n), dtype)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().amax(dim=-1).to(x.dtype)


# Helper to get tolerances


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 1e-2, "rtol": 1e-2}


# AmaxFwdOp tests


@ReduceBasicFixture
def test_amax_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.amax import AmaxFwdOp

    test = ReduceTest(m, n, dtype)
    op = AmaxFwdOp(dtype=dtype, dim=-1)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


@SpecReduceFixture
def test_amax_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: AmaxFwdOp with dim + keepdim."""
    from tileops.ops.reduction.amax import AmaxFwdOp

    x = torch.randn(shape, dtype=dtype, device=_device())
    op = AmaxFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = torch.amax(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"amax spec max err: {(y - ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
