"""Correctness tests for GemmOp (NPU).

Per-op test file: the shared GPU ``test_gemm.py`` was split so that only this
migration unit's cases run when ``GemmOp`` is integrated (the shared file also
carried ``GemmW4A16Op`` cases, which belong to a separate migration unit, now
in ``test_gemm_w4a16.py``).  The shared GPU file also covers ``GemmFp8Op``,
which is not migrated.

Adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (the workload's
  ``gen_inputs``); there is no hard-coded ``device="cuda"`` left.
- T2: no ``torch.cuda`` availability guard was present for the Gemm cases --
  nothing to remove.
- T3: the ``tune`` fixture param and the ``tune=`` op kwarg are removed.
"""

import pytest
import torch

from tileops.ops.gemm.gemm import GemmOp
from tileops.testing.test_base import TestBase
from tileops.workloads.gemm import GemmWorkload
from tileops.workloads.workload_base import FixtureBase


class GemmTest(GemmWorkload, TestBase):
    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)


class GemmFixture(FixtureBase):
    # T3: the GPU "m, n, k, dtype, trans_a, trans_b, tune" param list drops
    # the trailing ``tune`` flag.
    PARAMS = [
        ("m, n, k, dtype, trans_a, trans_b", [
            pytest.param(
                1024, 1024, 1024, torch.float16, False, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-fp16-square",
            ),
            pytest.param(
                1024, 1024, 1024, torch.bfloat16, False, False,
                marks=pytest.mark.smoke,
                id="smoke-bf16-square",
            ),
            pytest.param(
                1, 1024, 1024, torch.float16, False, True,
                marks=pytest.mark.full,
                id="full-fp16-trans-b-small-m",
            ),
            pytest.param(
                128, 2112, 4096, torch.float16, False, True,
                marks=pytest.mark.full,
                id="full-fp16-nt-dense",
            ),
            pytest.param(
                1, 7168, 16384, torch.float16, False, True,
                marks=pytest.mark.full,
                id="full-fp16-wide",
            ),
            pytest.param(
                1024, 1, 1024, torch.float16, False, False,
                marks=pytest.mark.full,
                id="full-fp16-thin-n",
            ),
            pytest.param(
                1, 1024, 1024, torch.bfloat16, False, True,
                marks=pytest.mark.full,
                id="full-bf16-trans-b-small-m",
            ),
            pytest.param(
                1, 7168, 16384, torch.bfloat16, False, True,
                marks=pytest.mark.full,
                id="full-bf16-wide",
            ),
            pytest.param(
                1024, 1, 1024, torch.bfloat16, False, False,
                marks=pytest.mark.full,
                id="full-bf16-thin-n",
            ),
        ]),
    ]


class GemvBoundaryFixture(FixtureBase):
    """GEMV cases with non-aligned n/k to exercise partial-tile paths."""

    PARAMS = [
        ("n, k, dtype", [
            # lhs_row: m=1, trans_b=True -- non-aligned n
            pytest.param(3000, 1024, torch.float16, marks=pytest.mark.smoke),
            pytest.param(3000, 1024, torch.bfloat16, marks=pytest.mark.smoke),
            # lhs_row: non-aligned k
            pytest.param(1024, 3000, torch.float16, marks=pytest.mark.full),
            # rhs_col: n=1 -- non-aligned m (mapped to gemv n param)
            pytest.param(3001, 1024, torch.float16, marks=pytest.mark.full),
        ]),
    ]


@GemmFixture
def test_gemm(
    m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool
) -> None:
    test = GemmTest(m, n, k, dtype, trans_a, trans_b)
    op = GemmOp(trans_a=trans_a, trans_b=trans_b)
    if dtype == torch.float16:
        tolerances = {"atol": 1e-3, "rtol": 1e-3}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@GemvBoundaryFixture
def test_gemv_boundary_lhs_row(n: int, k: int, dtype: torch.dtype) -> None:
    """GEMV lhs_row path (m=1, trans_b=True) with non-aligned n or k."""
    test = GemmTest(1, n, k, dtype, trans_a=False, trans_b=True)
    op = GemmOp(trans_a=False, trans_b=True)
    tolerances = {"atol": 1e-2, "rtol": 1e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@GemvBoundaryFixture
def test_gemv_boundary_rhs_col(n: int, k: int, dtype: torch.dtype) -> None:
    """GEMV rhs_col path (n=1, no transpose) with non-aligned m or k."""
    m = n  # reuse fixture's n as the non-aligned m dimension
    test = GemmTest(m, 1, k, dtype, trans_a=False, trans_b=False)
    op = GemmOp(trans_a=False, trans_b=False)
    tolerances = {"atol": 1e-2, "rtol": 1e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
