"""Correctness tests for ExpFwdOp (NPU).

Ported from the GPU TileOPs test for ExpFwdOp (``tests/ops/test_unary_math.py``
per the GPU manifest ``source.test``).  Device backend instead of hard-coded
``"cuda"`` (T1/T2).

Input scale is kept moderate (randn * 3) so exp stays comfortably inside fp16
range (exp(9) ~ 8.1e3, vs fp16 max 6.55e4).
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.exp import ExpFwdKernel
from tileops.ops.elementwise.exp import ExpFwdOp
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


class ExpTest(RandnWorkload, TestBase):
    """Test fixture for ExpFwdOp — flat 1-D randn input."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__((n_total,), dtype)
        self.n_total = n_total

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x.float()).to(x.dtype)


class ExpFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(
                    1_048_576, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
                ),
                pytest.param(1_048_576, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(1_048_576, torch.bfloat16, marks=[pytest.mark.smoke]),
                pytest.param(4_000_000, torch.float16, marks=[pytest.mark.full]),
                pytest.param(4_000_001, torch.float16, marks=[pytest.mark.full]),
            ],
        ),
    ]


@ExpFixture
def test_exp_op(n_total: int, dtype: torch.dtype) -> None:
    test = ExpTest(n_total, dtype)
    op = ExpFwdOp(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_exp_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        ExpFwdKernel(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_exp_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = ExpFwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


@pytest.mark.smoke
def test_exp_known_values() -> None:
    """exp(0)=1, exp(1)=e, exp(-inf)=0 — anchor values against the reference."""
    N = 4096
    dtype = torch.float32
    vals = torch.tensor([0.0, 1.0, -1.0, float("-inf")])
    x = torch.cat([vals, torch.zeros(N - vals.numel())]).to(dtype).to(_device())
    op = ExpFwdOp(N_total=N, dtype=dtype)
    y = op(x)
    torch.testing.assert_close(y.cpu(), torch.exp(x.cpu()), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
