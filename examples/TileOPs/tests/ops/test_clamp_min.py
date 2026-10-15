"""Correctness tests for ClampMinFwdOp (NPU).

ClampMinFwdOp is the scalar-min specialization of ClampScalarFwdOp
(``torch.clamp_min(input, min) == torch.clamp(input, min, max=None)``), so this
file mirrors ``test_clamp_scalar.py`` restricted to the min-only form.  The
independent golden is ``torch.clamp_min``.

NPU adaptations (T1-T4, same as ``test_clamp_scalar.py``):

- T1: manual tensor creation uses the device backend (``_device()``) instead of
  hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards removed
  (NPU availability is a runtime concern).
- T3/T4: no ``tune`` parametrization / autotune try-except to remove.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.clamp_scalar import ClampScalarFwdKernel
from tileops.ops.elementwise.clamp_min import ClampMinFwdOp
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


class ClampMinTest(RandnWorkload, TestBase):
    """Test fixture for ClampMinFwdOp -- flat 1-D randn input with a lower bound."""

    def __init__(self, n_total: int, dtype: torch.dtype, min_val: float):
        super().__init__((n_total,), dtype)
        self.n_total = n_total
        self.min_val = min_val

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp_min(x.float(), self.min_val).to(x.dtype)


class ClampMinFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype, min_val",
            [
                pytest.param(
                    1_048_576,
                    torch.float32,
                    -0.5,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                ),
                pytest.param(1_048_576, torch.float16, -0.5, marks=[pytest.mark.smoke]),
                pytest.param(1_048_576, torch.bfloat16, -0.5, marks=[pytest.mark.smoke]),
                # Full: larger follow-up coverage
                pytest.param(4_000_000, torch.float16, -0.5, marks=[pytest.mark.full]),
                pytest.param(4_000_000, torch.bfloat16, -0.5, marks=[pytest.mark.full]),
            ],
        ),
    ]


@ClampMinFixture
def test_clamp_min_op(n_total: int, dtype: torch.dtype, min_val: float) -> None:
    test = ClampMinTest(n_total, dtype, min_val)
    op = ClampMinFwdOp(N_total=n_total, dtype=dtype, min=min_val)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("min_val", [-0.5, 0.0, 0.25])
def test_clamp_min_bounds(min_val: float) -> None:
    """``torch.clamp_min`` with several lower bounds, all three dtypes."""
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        test = ClampMinTest(1024, dtype, min_val)
        op = ClampMinFwdOp(N_total=1024, dtype=dtype, min=min_val)
        test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_clamp_min_nan_semantics(dtype: torch.dtype) -> None:
    """NaN input propagates NaN; +-inf and finite inputs match torch.clamp_min."""
    n_total = 4096
    op = ClampMinFwdOp(N_total=n_total, dtype=dtype, min=-0.5)
    x = torch.randn(n_total, dtype=dtype, device=_device())
    x[0::8] = float("nan")
    x[1::8] = float("inf")
    x[2::8] = float("-inf")

    out = op(x).cpu()
    ref = torch.clamp_min(x.float(), -0.5).to(dtype).cpu()
    atol, rtol = _get_tolerances(dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol, equal_nan=True)
    assert torch.equal(torch.isnan(out), torch.isnan(ref)), "NaN mask mismatch"


@pytest.mark.smoke
def test_clamp_min_none_rejected() -> None:
    """ClampMinFwdOp must reject min=None (``present(min)`` shape rule)."""
    with pytest.raises(ValueError, match=r"requires min"):
        ClampMinFwdOp(N_total=16, dtype=torch.float32, min=None)


@pytest.mark.smoke
def test_clamp_min_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        ClampScalarFwdKernel(N_total=16, dtype=torch.int32, min_val=-0.5)


@pytest.mark.smoke
def test_clamp_min_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = ClampMinFwdOp(N_total=N, dtype=dtype, min=-0.5)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
