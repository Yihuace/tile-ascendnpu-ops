"""Correctness tests for IsinfFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_unary_math.py`` (the ``IsinfFwdOp``
float-predicate subset) with the GPU workload's special-value draw
(``workloads/elementwise.py::_draw_special``: a quarter each of NaN, +Inf, -Inf
and finite values).

``torch.isinf`` returns a bool tensor whose every entry must match PyTorch
exactly, so the comparison uses exact equality (``atol = rtol = 0``) -- see
``_get_tolerances``.  The NaN quarter is the discriminant for the ordered ``==``
``isinf`` body: ``isinf(NaN)`` must answer false.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) instead of
  hard-coded ``device="cuda"`` (the workload generator carries the same
  adaptation, W1).
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards removed
  (NPU availability is a runtime concern).
- T3/T4: the GPU predicate tests carry no ``tune`` parametrization / autotune
  try-except -- nothing to remove.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.isinf import IsinfFwdKernel
from tileops.ops.elementwise.isinf import IsinfFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.elementwise import IsinfWorkload
from tileops.workloads.workload_base import FixtureBase


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """``torch.isinf`` is exact for every admitted float dtype.

    The output is bool, so the comparison is exact equality:
    ``atol = rtol = 0`` (max_abs_err must be 0).
    """
    return 0.0, 0.0


def _device() -> str:
    return get_device_backend().name


class IsinfTest(IsinfWorkload, TestBase):
    """Test fixture for IsinfFwdOp -- float input, exact bool output."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__((n_total,), dtype)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # Golden isinf computed on the host CPU: ``torch.isinf`` is the standard
        # PyTorch predicate and is independent of the kernel's infinity-detection
        # implementation.  Returned to the input device so the comparison stays
        # same-device.
        return torch.isinf(x.detach().cpu().float()).to(x.device)


class IsinfFixture(FixtureBase):
    """Parametrize over the supported float dtypes."""

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


@IsinfFixture
def test_isinf_op(n_total: int, dtype: torch.dtype) -> None:
    test = IsinfTest(n_total, dtype)
    op = IsinfFwdOp(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_isinf_output_dtype_and_values() -> None:
    """Output is bool; +/-Inf positions are True, NaN and finite positions False."""
    n = 4096
    op = IsinfFwdOp(N_total=n, dtype=torch.float32)
    x = torch.randn(n, device=_device(), dtype=torch.float32)
    quarter = n // 4
    x[:quarter] = float("nan")
    x[quarter : 2 * quarter] = float("inf")
    x[2 * quarter : 3 * quarter] = float("-inf")
    y = op(x)
    assert y.dtype == torch.bool
    assert tuple(y.shape) == (n,)
    # NaN must be false under the ordered == body.
    assert not bool(y[:quarter].any())
    assert bool(y[quarter : 3 * quarter].all())
    assert not bool(y[3 * quarter :].any())


@pytest.mark.smoke
def test_isinf_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        IsinfFwdKernel(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_isinf_op_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="does not support dtype"):
        IsinfFwdOp(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_isinf_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = IsinfFwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    # Read N float elements + write N bool (one byte) elements.
    assert mem_bytes == N * (elem_bytes + 1), (
        f"bytes {mem_bytes} != N * (elem_bytes + 1) = {N * (elem_bytes + 1)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
