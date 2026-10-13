"""Correctness tests for ClampScalarFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_where_clamp_masked_fill_conformance.py``
-- the ``ClampScalar`` subset only (the shared GPU file also covers where,
Tensor-bound clamp and masked_fill, which are not migrated).  Uses the device
backend instead of hard-coded ``"cuda"``.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) instead of
  hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards removed
  (NPU availability is a runtime concern).
- T3/T4: the GPU ClampScalar tests carry no ``tune`` parametrization / autotune
  try-except -- nothing to remove.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.clamp_scalar import ClampScalarFwdKernel
from tileops.ops.elementwise.clamp_scalar import ClampScalarFwdOp
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


class ClampScalarTest(RandnWorkload, TestBase):
    """Test fixture for ClampScalarFwdOp -- flat 1-D randn input with scalar bounds."""

    def __init__(
        self,
        n_total: int,
        dtype: torch.dtype,
        min_val: float | None = None,
        max_val: float | None = None,
    ):
        super().__init__((n_total,), dtype)
        self.n_total = n_total
        self.min_val = min_val
        self.max_val = max_val

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x.float(), self.min_val, self.max_val).to(x.dtype)


class ClampScalarFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype, min_val, max_val",
            [
                pytest.param(
                    1_048_576,
                    torch.float32,
                    -0.5,
                    0.5,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                ),
                pytest.param(1_048_576, torch.float16, -0.5, 0.5, marks=[pytest.mark.smoke]),
                pytest.param(1_048_576, torch.bfloat16, -0.5, 0.5, marks=[pytest.mark.smoke]),
                # Full: larger follow-up coverage
                pytest.param(4_000_000, torch.float16, -0.5, 0.5, marks=[pytest.mark.full]),
                pytest.param(4_000_000, torch.bfloat16, -0.5, 0.5, marks=[pytest.mark.full]),
            ],
        ),
    ]


@ClampScalarFixture
def test_clamp_scalar_op(
    n_total: int, dtype: torch.dtype, min_val: float | None, max_val: float | None
) -> None:
    test = ClampScalarTest(n_total, dtype, min_val, max_val)
    op = ClampScalarFwdOp(N_total=n_total, dtype=dtype, min=min_val, max=max_val)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "min_val, max_val",
    [(-0.5, 0.5), (None, 0.5), (-0.5, None)],
)
def test_clamp_scalar_param_names(min_val: float | None, max_val: float | None) -> None:
    """One Op serves the clamp / clamp_min / clamp_max scalar forms."""
    n_total = 1024
    dtype = torch.float32
    test = ClampScalarTest(n_total, dtype, min_val, max_val)
    op = ClampScalarFwdOp(N_total=n_total, dtype=dtype, min=min_val, max=max_val)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_clamp_scalar_both_none_rejected() -> None:
    """ClampScalarFwdOp must reject min=None and max=None.

    Mirrors torch.clamp(input, None, None), which raises
    ``RuntimeError("At least one of min or max must not be None")``.
    """
    with pytest.raises(ValueError, match=r"present\(min\) or present\(max\)"):
        ClampScalarFwdOp(N_total=16, dtype=torch.float32, min=None, max=None)


@pytest.mark.smoke
def test_clamp_scalar_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        ClampScalarFwdKernel(N_total=16, dtype=torch.int32, min_val=-0.5)


@pytest.mark.smoke
def test_clamp_scalar_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = ClampScalarFwdOp(N_total=N, dtype=dtype, min=-0.5, max=0.5)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
