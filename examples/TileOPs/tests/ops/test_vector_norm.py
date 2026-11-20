"""Correctness tests for L1NormFwdOp / L2NormFwdOp / InfNormFwdOp (NPU).

Ported from the GPU TileOPs test for the vector-norm family
(``tests/ops/test_vector_norm.py`` per the GPU manifest ``source.test``).
Uses the device backend instead of hard-coded ``"cuda"`` (T1/T2).

All three ops share one kernel family and differ only by ``op_kind``, so they
are exercised by the same param sets here; the reference is
``torch.linalg.vector_norm`` with the matching ``ord``.

Shapes are chosen to fit the single-tile UB budget (see the kernel module
docstring); the largest N here is 8192, well inside it.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.reduction.vector_norm import InfNormFwdOp, L1NormFwdOp, L2NormFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.reduction import InfNormWorkload, L1NormWorkload, L2NormWorkload
from tileops.workloads.workload_base import FixtureBase


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


def _device() -> str:
    return get_device_backend().name


def _ref(x: torch.Tensor, ord_value, dim) -> torch.Tensor:
    return torch.linalg.vector_norm(x.float(), ord=ord_value, dim=dim).to(x.dtype)


class L1NormTest(L1NormWorkload, TestBase):
    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return _ref(x, 1, self.dim)


class L2NormTest(L2NormWorkload, TestBase):
    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return _ref(x, 2, self.dim)


class InfNormTest(InfNormWorkload, TestBase):
    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return _ref(x, float("inf"), self.dim)


class NormFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype",
            [
                pytest.param(
                    (32, 512), -1, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
                ),
                pytest.param((32, 512), -1, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param((32, 512), -1, torch.bfloat16, marks=[pytest.mark.smoke]),
                # Full: non-divisible rows/cols exercise both tail paths
                pytest.param((33, 513), -1, torch.float32, marks=[pytest.mark.full]),
                pytest.param((33, 513), -1, torch.float16, marks=[pytest.mark.full]),
                # Full: realistic wide rows
                pytest.param((2048, 4096), -1, torch.float16, marks=[pytest.mark.full]),
                pytest.param((64, 8192), -1, torch.bfloat16, marks=[pytest.mark.full]),
                # Full: 3-D multi-dim reduce
                pytest.param((4, 128, 512), [0, 2], torch.float16, marks=[pytest.mark.full]),
            ],
        ),
    ]


@NormFixture
def test_l1_norm_op(shape: tuple, dim, dtype: torch.dtype) -> None:
    test = L1NormTest(shape, dtype, dim=dim)
    op = L1NormFwdOp(dtype=dtype, dim=dim)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@NormFixture
def test_l2_norm_op(shape: tuple, dim, dtype: torch.dtype) -> None:
    test = L2NormTest(shape, dtype, dim=dim)
    op = L2NormFwdOp(dtype=dtype, dim=dim)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@NormFixture
def test_inf_norm_op(shape: tuple, dim, dtype: torch.dtype) -> None:
    test = InfNormTest(shape, dtype, dim=dim)
    op = InfNormFwdOp(dtype=dtype, dim=dim)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_norm_rejects_wrong_ord() -> None:
    """Each op only accepts its manifest-fixed ord."""
    with pytest.raises(ValueError, match="only supports ord"):
        L1NormFwdOp(dtype=torch.float16, ord=2)
    with pytest.raises(ValueError, match="only supports ord"):
        L2NormFwdOp(dtype=torch.float16, ord=1)
    with pytest.raises(ValueError, match="only supports ord"):
        InfNormFwdOp(dtype=torch.float16, ord=2)


@pytest.mark.smoke
def test_l1_norm_eval_roofline() -> None:
    M, N = 64, 256
    dtype = torch.float16
    op = L1NormFwdOp(dtype=dtype, dim=-1)
    x = torch.randn(M, N, dtype=dtype, device=_device())
    op(x)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 2 * M * N, f"flops {flops} != 2 * M * N = {2 * M * N}"
    assert mem_bytes == (M * N + M) * elem_bytes, (
        f"bytes {mem_bytes} != (M*N + M) * elem_bytes = {(M * N + M) * elem_bytes}"
    )


@pytest.mark.smoke
def test_inf_norm_propagates_nan() -> None:
    """ord=inf must yield NaN for rows containing NaN (torch semantics).

    ``T.reduce_max`` does not propagate NaN, so the Op layer detects NaN rows
    before the kernel and patches the output afterwards.  This test guards that
    mechanism.
    """
    M, N = 8, 512
    dtype = torch.float32
    x = torch.randn(M, N, dtype=dtype)
    x[2, 17] = float("nan")
    x[5, 0] = float("nan")
    x_dev = x.to(_device())
    op = InfNormFwdOp(dtype=dtype, dim=-1)
    y = op(x_dev)
    ref = torch.linalg.vector_norm(x, ord=float("inf"), dim=-1).to(dtype)
    assert torch.isnan(y[2]) and torch.isnan(y[5]), "NaN rows must produce NaN"
    finite = [i for i in range(M) if i not in (2, 5)]
    torch.testing.assert_close(y[finite].cpu(), ref[finite], atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_l1_norm_all_negative_row() -> None:
    """All-negative rows: |x| must be applied (l1 == sum of magnitudes)."""
    M, N = 8, 512
    dtype = torch.float32
    x = -torch.rand(M, N, dtype=dtype) - 0.5
    op = L1NormFwdOp(dtype=dtype, dim=-1)
    y = op(x.to(_device()))
    ref = torch.linalg.vector_norm(x, ord=1, dim=-1).to(dtype)
    torch.testing.assert_close(y.cpu(), ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
