"""Correctness tests for MishFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_activation.py`` — Mish subset.
Uses the device backend instead of hard-coded ``"cuda"``.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.kernels.elementwise.mish import MishFwdKernel
from tileops.ops.elementwise.mish import MishFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.workload_base import FixtureBase, RandnWorkload


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


def _device() -> str:
    return get_device_backend().name


class MishTest(RandnWorkload, TestBase):
    """Test fixture for MishFwdOp — flat 1-D randn input."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__((n_total,), dtype)
        self.n_total = n_total

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(x.float()).to(x.dtype)


class MishFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(
                    1_048_576, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
                ),
                pytest.param(1_048_576, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(1_048_576, torch.bfloat16, marks=[pytest.mark.smoke]),
                # Full: larger follow-up coverage
                pytest.param(4_000_000, torch.float16, marks=[pytest.mark.full]),
                pytest.param(4_000_000, torch.bfloat16, marks=[pytest.mark.full]),
            ],
        ),
    ]


@MishFixture
def test_mish_op(n_total: int, dtype: torch.dtype) -> None:
    test = MishTest(n_total, dtype)
    op = MishFwdOp(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_mish_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        MishFwdKernel(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_mish_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = MishFwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 4 * N, f"flops {flops} != 4 * N = {4 * N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
