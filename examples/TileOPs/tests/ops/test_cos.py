"""Correctness tests for CosFwdOp (NPU).

Ported from the GPU TileOPs test for CosFwdOp (``tests/ops/test_unary_math.py``
per the GPU manifest ``source.test``).  Device backend instead of hard-coded
``"cuda"`` (T1/T2).
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.cos import CosFwdKernel
from tileops.ops.elementwise.cos import CosFwdOp
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


# `vcos` on this build (910B2C / CANN 9.0.0 / npuir wheel v0.1.2) is only
# accurate on about [-pi/4, pi/4]: a dense sweep measured up to 48.7 absolute
# error at |x| ~ 6.45 when the instruction is called directly.  The kernel
# compensates with explicit four-quadrant range reduction (see the module
# docstring of tileops/kernels/elementwise/cos/cos_kernel/cos.py), which brings
# the worst measured error back to ~3e-6.  The tests therefore use the ordinary
# dtype tolerances, plus a wide-range regression (below) that fails loudly if
# the reduction is ever removed or broken.
_WIDE_RANGE = 12.0
_WIDE_RANGE_ATOL = 1e-4


class CosTest(RandnWorkload, TestBase):
    """Test fixture for CosFwdOp — flat 1-D randn input.

    cos is even but not monotone and not sign-symmetric on an arbitrary range,
    so the raw output is compared (no abs() shortcut in the reference or check).
    """

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__((n_total,), dtype)
        self.n_total = n_total

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x.float()).to(x.dtype)


class CosFixture(FixtureBase):
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


@CosFixture
def test_cos_op(n_total: int, dtype: torch.dtype) -> None:
    test = CosTest(n_total, dtype)
    op = CosFwdOp(N_total=n_total, dtype=dtype)
    # Ordinary dtype tolerances: the kernel's range reduction makes the result
    # independent of the instruction's narrow validity interval.
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_cos_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        CosFwdKernel(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_cos_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = CosFwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


@pytest.mark.smoke
def test_cos_wide_range_regression() -> None:
    """Range regression guard for the four-quadrant reduction.

    The raw ``vcos`` instruction diverges outside ~[-pi/4, pi/4] (measured
    48.7 absolute error at |x| ~ 6.45).  With the kernel's reduction in place
    the whole range must be accurate.  If someone removes the reduction, this
    test fails immediately — it is the guard that used to be an xfail pinning
    the defect.
    """
    N = 4096
    dtype = torch.float32
    x = torch.linspace(-_WIDE_RANGE, _WIDE_RANGE, N, dtype=torch.float32).to(_device())
    op = CosFwdOp(N_total=N, dtype=dtype)
    y = op(x)
    torch.testing.assert_close(
        y.cpu(), torch.cos(x.cpu()), atol=_WIDE_RANGE_ATOL, rtol=_WIDE_RANGE_ATOL
    )


@pytest.mark.smoke
def test_cos_quadrant_boundaries() -> None:
    """Exact multiples of pi/2 exercise every quadrant of the reduction.

    cos(pi/2) = 0 (q=1, sin branch), cos(pi) = -1 (q=2, negated cos branch),
    cos(3pi/2) = 0 (q=3, sin branch) — each quadrant is hit at least once.
    """
    N = 4096
    dtype = torch.float32
    pi = 3.14159265358979
    vals = torch.tensor(
        [0.0, pi / 2, pi, 3 * pi / 2, 2 * pi, -pi / 2, -pi, -3 * pi / 2, -2 * pi]
    )
    x = torch.cat([vals, torch.zeros(N - vals.numel())]).to(dtype).to(_device())
    op = CosFwdOp(N_total=N, dtype=dtype)
    y = op(x)
    torch.testing.assert_close(
        y.cpu(), torch.cos(x.cpu()), atol=_WIDE_RANGE_ATOL, rtol=_WIDE_RANGE_ATOL
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
