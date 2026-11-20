"""Correctness tests for AminFwdOp (NPU).

Per-op test file: the shared GPU ``test_reduce.py`` was split so that only this
migration unit's cases run when ``AminFwdOp`` is integrated (the shared file
also carried AmaxFwdOp cases, which belong to a separate migration unit).
Uses the device backend instead of the hard-coded ``"cuda"`` (T1).

Covers: AminFwdOp (element-wise minimum reduction).
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.testing.test_base import TestBase
from tileops.workloads.reduction import AminWorkload
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


class ReduceTest(AminWorkload, TestBase):
    """Parameterized test helper for AminFwdOp (dim=-1, 2-D input)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        super().__init__((m, n), dtype)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().amin(dim=-1).to(x.dtype)


# Helper to get tolerances


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 1e-2, "rtol": 1e-2}


# AminFwdOp tests


@ReduceBasicFixture
def test_amin_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.amin import AminFwdOp

    test = ReduceTest(m, n, dtype)
    op = AminFwdOp(dtype=dtype, dim=-1)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


@SpecReduceFixture
def test_amin_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: AminFwdOp with dim + keepdim."""
    from tileops.ops.reduction.amin import AminFwdOp

    x = torch.randn(shape, dtype=dtype, device=_device())
    op = AminFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = torch.amin(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"amin spec max err: {(y - ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
