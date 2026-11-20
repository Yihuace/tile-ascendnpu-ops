"""Correctness tests for CeilFwdOp (NPU).

Ported from the GPU TileOPs test for CeilFwdOp (``tests/ops/test_unary_math.py``
per the GPU manifest ``source.test``).  Device backend instead of hard-coded
``"cuda"`` (T1/T2).

Note the kernel implements ceil via an int32 ``vcast`` round-trip, so inputs are
kept inside the representable magnitude range (randn-scaled).
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.ceil import CeilFwdKernel
from tileops.ops.elementwise.ceil import CeilFwdOp
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


class CeilTest(RandnWorkload, TestBase):
    """Test fixture for CeilFwdOp — flat 1-D randn input (both signs)."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__((n_total,), dtype)
        self.n_total = n_total

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ceil(x.float()).to(x.dtype)


class CeilFixture(FixtureBase):
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


@CeilFixture
def test_ceil_op(n_total: int, dtype: torch.dtype) -> None:
    test = CeilTest(n_total, dtype)
    op = CeilFwdOp(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_ceil_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        CeilFwdKernel(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_ceil_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = CeilFwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


@pytest.mark.smoke
def test_ceil_signed_boundary() -> None:
    """Values either side of integers: ceil must round up, including negatives."""
    N = 4096
    dtype = torch.float32
    vals = torch.tensor([0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.5, -2.5])
    x = torch.cat([vals, torch.full((N - vals.numel(),), 0.25)]).to(dtype).to(_device())
    op = CeilFwdOp(N_total=N, dtype=dtype)
    y = op(x)
    torch.testing.assert_close(y.cpu(), torch.ceil(x.cpu()), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
