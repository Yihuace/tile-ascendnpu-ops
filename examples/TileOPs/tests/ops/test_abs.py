"""Correctness tests for AbsFwdOp (NPU).

Ported from the GPU TileOPs test for AbsFwdOp (``tests/ops/test_unary_math.py``
per the GPU manifest ``source.test``).  Uses the device backend instead of
hard-coded ``"cuda"`` (T1/T2).

The reference is exact (sign-bit clear), so tolerances only need to absorb the
fp16/bf16 round-trip through the kernel's fp32 intermediate.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.abs import AbsFwdKernel
from tileops.ops.elementwise.abs import AbsFwdOp
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


class AbsTest(RandnWorkload, TestBase):
    """Test fixture for AbsFwdOp — flat 1-D randn input.

    ``randn`` spans both signs, so the sign-flip branch of ``|x|`` is
    exercised by every element (a non-negative generator would make the op a
    no-op and mask sign bugs).
    """

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__((n_total,), dtype)
        self.n_total = n_total

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x.float()).to(x.dtype)


class AbsFixture(FixtureBase):
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
                # Full: tail path (non-divisible by block_size=2048)
                pytest.param(4_000_001, torch.float16, marks=[pytest.mark.full]),
            ],
        ),
    ]


@AbsFixture
def test_abs_op(n_total: int, dtype: torch.dtype) -> None:
    test = AbsTest(n_total, dtype)
    op = AbsFwdOp(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_abs_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        AbsFwdKernel(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_abs_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = AbsFwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


@pytest.mark.smoke
def test_abs_negative_input() -> None:
    """All-negative input: every element must take the sign-flip branch."""
    N = 4096
    dtype = torch.float16
    x = (-torch.rand(N, dtype=torch.float32) - 0.5).to(dtype).to(_device())
    op = AbsFwdOp(N_total=N, dtype=dtype)
    y = op(x)
    torch.testing.assert_close(
        y.float().cpu(), x.float().cpu().abs(), atol=1e-3, rtol=1e-3
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
