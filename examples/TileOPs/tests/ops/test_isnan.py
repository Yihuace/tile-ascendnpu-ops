"""Correctness tests for IsnanFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_unary_math.py`` (the ``IsnanFwdOp``
float-predicate subset) with the GPU workload's special-value draw
(``workloads/elementwise.py::_draw_special``: a quarter each of NaN, +Inf, -Inf
and finite values).

``torch.isnan`` returns a bool tensor whose every entry must match PyTorch
exactly, so the comparison uses exact equality (``atol = rtol = 0``) -- see
``_get_tolerances``.

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
from tileops.kernels.elementwise.isnan import IsnanFwdKernel
from tileops.ops.elementwise.isnan import IsnanFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.elementwise import IsnanWorkload
from tileops.workloads.workload_base import FixtureBase


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """``torch.isnan`` is exact for every admitted float dtype.

    The output is bool, so the comparison is exact equality:
    ``atol = rtol = 0`` (max_abs_err must be 0).
    """
    return 0.0, 0.0


def _device() -> str:
    return get_device_backend().name


class IsnanTest(IsnanWorkload, TestBase):
    """Test fixture for IsnanFwdOp -- float input, exact bool output."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__((n_total,), dtype)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # Golden isnan computed on the host CPU: ``torch.isnan`` is the standard
        # PyTorch predicate and is independent of the kernel's NaN-detection
        # implementation.  Returned to the input device so the comparison stays
        # same-device.
        return torch.isnan(x.detach().cpu().float()).to(x.device)


class IsnanFixture(FixtureBase):
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


@IsnanFixture
def test_isnan_op(n_total: int, dtype: torch.dtype) -> None:
    test = IsnanTest(n_total, dtype)
    op = IsnanFwdOp(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_isnan_output_dtype_and_values() -> None:
    """Output is bool; NaN positions are True, finite positions are False."""
    n = 4096
    op = IsnanFwdOp(N_total=n, dtype=torch.float32)
    x = torch.randn(n, device=_device(), dtype=torch.float32)
    half = n // 2
    x[:half] = float("nan")
    y = op(x)
    assert y.dtype == torch.bool
    assert tuple(y.shape) == (n,)
    assert bool(y[:half].all())
    assert not bool(y[half:].any())


@pytest.mark.smoke
def test_isnan_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        IsnanFwdKernel(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_isnan_op_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="does not support dtype"):
        IsnanFwdOp(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_isnan_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = IsnanFwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    # Read N float elements + write N bool (one byte) elements.
    assert mem_bytes == N * (elem_bytes + 1), (
        f"bytes {mem_bytes} != N * (elem_bytes + 1) = {N * (elem_bytes + 1)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
