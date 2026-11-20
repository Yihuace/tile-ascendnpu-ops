"""Correctness tests for BitwiseNotFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_bitwise.py`` — the ``bitwise_not``
subset only (the GPU file also covers the binary bitwise_and / bitwise_or /
bitwise_xor ops, which are not migrated here).

Bitwise NOT operates on bool / integer inputs; the test sweeps every
torch-supported dtype and additionally checks that float dtypes are
rejected by the kernel.  Uses the device backend instead of hard-coded
``"cuda"`` (T1) and drops the GPU ``@pytest.mark.skipif(not
torch.cuda.is_available(), ...)`` guard (T2).
"""

import pytest
import torch

from tileops.kernels.elementwise.bitwise_not import BitwiseNotFwdKernel
from tileops.ops.elementwise.bitwise_not import BitwiseNotFwdOp
from tileops.testing.test_base import TestBase, exact_compare
from tileops.workloads.elementwise import BitwiseNotWorkload
from tileops.workloads.workload_base import FixtureBase


class BitwiseNotTest(BitwiseNotWorkload, TestBase):
    """Test fixture for bitwise_not — flat 1-D input of ``n_total`` elements."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__((n_total,), dtype)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_not(x)


class BitwiseFixture(FixtureBase):
    """Parametrize over torch-supported bitwise_not dtypes."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(
                1_048_576, torch.bool,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
            ),
            pytest.param(1_048_576, torch.uint8, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int8, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int32, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int64, marks=pytest.mark.smoke),
            # Full: larger follow-up coverage
            pytest.param(4_000_000, torch.int32, marks=pytest.mark.full),
            pytest.param(4_000_000, torch.int64, marks=pytest.mark.full),
        ]),
    ]


@BitwiseFixture
def test_bitwise_not(n_total: int, dtype: torch.dtype) -> None:
    test = BitwiseNotTest(n_total, dtype)
    op = BitwiseNotFwdOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=exact_compare)


@pytest.mark.parametrize("dtype", [
    pytest.param(torch.float16, marks=pytest.mark.smoke),
    pytest.param(torch.bfloat16, marks=pytest.mark.smoke),
    pytest.param(torch.float32, marks=pytest.mark.smoke),
])
def test_bitwise_not_rejects_float_dtype(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseNotFwdKernel(N_total=16, dtype=dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
