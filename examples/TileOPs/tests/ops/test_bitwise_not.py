"""Correctness tests for BitwiseNotFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_bitwise.py`` -- the ``bitwise_not`` subset
only (the shared GPU file also covers bitwise_and / or / xor and their broadcast
variants, which are not migrated).

``torch.bitwise_not`` is an exact op over bool / integer element types; the
output is the all-ones complement of the input and must match PyTorch bit for
bit.  The comparison therefore uses exact equality (``atol = rtol = 0``), not a
floating-point tolerance -- see ``_get_tolerances``.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) instead of
  hard-coded ``device="cuda"`` (the workload generator carries the same
  adaptation, W1).
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards removed
  (NPU availability is a runtime concern).
- T3/T4: the GPU ``bitwise_not`` tests carry no ``tune`` parametrization /
  autotune try-except -- nothing to remove.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.bitwise_not import BitwiseNotFwdKernel
from tileops.ops.elementwise.bitwise_not import BitwiseNotFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.elementwise import BitwiseNotWorkload
from tileops.workloads.workload_base import FixtureBase


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Bitwise NOT is exact for every admitted (bool / integer) dtype.

    All manifest dtypes are integral, so the comparison is exact equality:
    ``atol = rtol = 0`` (max_abs_err must be 0).
    """
    return 0.0, 0.0


def _device() -> str:
    return get_device_backend().name


class BitwiseNotTest(BitwiseNotWorkload, TestBase):
    """Test fixture for BitwiseNotFwdOp -- bool / integer input, exact output."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__((n_total,), dtype)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_not(x)


class BitwiseNotFixture(FixtureBase):
    """Parametrize over the torch-supported bitwise_not dtypes."""

    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(
                    1_048_576, torch.int32, marks=[pytest.mark.smoke, pytest.mark.packaging]
                ),
                pytest.param(1_048_576, torch.int64, marks=[pytest.mark.smoke]),
                pytest.param(1_048_576, torch.bool, marks=[pytest.mark.smoke]),
                pytest.param(1_048_576, torch.uint8, marks=[pytest.mark.smoke]),
                pytest.param(1_048_576, torch.int8, marks=[pytest.mark.smoke]),
                pytest.param(1_048_576, torch.int16, marks=[pytest.mark.smoke]),
                # Full: larger follow-up coverage (int32 / int64)
                pytest.param(4_000_000, torch.int32, marks=[pytest.mark.full]),
                pytest.param(4_000_000, torch.int64, marks=[pytest.mark.full]),
            ],
        ),
    ]


@BitwiseNotFixture
def test_bitwise_not_op(n_total: int, dtype: torch.dtype) -> None:
    test = BitwiseNotTest(n_total, dtype)
    op = BitwiseNotFwdOp(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_bitwise_not_rejects_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseNotFwdKernel(N_total=16, dtype=torch.float32)


@pytest.mark.smoke
def test_bitwise_not_op_rejects_float_dtype() -> None:
    with pytest.raises(ValueError, match="does not support dtype"):
        BitwiseNotFwdOp(N_total=16, dtype=torch.float32)


@pytest.mark.smoke
def test_bitwise_not_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.int32
    op = BitwiseNotFwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
