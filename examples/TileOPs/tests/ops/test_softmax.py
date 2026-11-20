"""Correctness tests for LogSumExpFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_softmax.py`` — LogSumExp subset only.
Uses the device backend instead of hard-coded ``"cuda"``.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.reduction.softmax import LogSumExpFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.reduction import LogSumExpWorkload
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


class LogSumExpFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype",
            [
                pytest.param(
                    (32, 256), -1, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
                ),
                pytest.param((32, 256), -1, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param((32, 256), -1, torch.bfloat16, marks=[pytest.mark.smoke]),
                pytest.param((32, 300), -1, torch.float32, marks=[pytest.mark.full]),
                pytest.param((32, 300), -1, torch.float16, marks=[pytest.mark.full]),
                pytest.param((32, 300), -1, torch.bfloat16, marks=[pytest.mark.full]),
                pytest.param((33, 256), -1, torch.float32, marks=[pytest.mark.full]),
                pytest.param((33, 256), -1, torch.float16, marks=[pytest.mark.full]),
                pytest.param((2, 16, 256), -1, torch.float32, marks=[pytest.mark.full]),
                pytest.param((2, 16, 256), -1, torch.float16, marks=[pytest.mark.full]),
                pytest.param((4, 32768), -1, torch.float16, marks=[pytest.mark.full]),
                pytest.param((256, 32), 0, torch.float32, marks=[pytest.mark.full]),
                pytest.param((256, 32), 0, torch.float16, marks=[pytest.mark.full]),
            ],
        ),
    ]


class LogSumExpTest(LogSumExpWorkload, TestBase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x.float(), dim=self.dim).to(x.dtype)

    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim


@LogSumExpFixture
def test_logsumexp_op(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    test = LogSumExpTest(shape, dtype, dim=dim)
    op = LogSumExpFwdOp(dtype=dtype, dim=dim)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class LogSumExpKeepdimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype",
            [
                pytest.param((32, 256), -1, torch.float32, marks=[pytest.mark.smoke]),
                pytest.param((32, 256), -1, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param((256, 32), 0, torch.float32, marks=[pytest.mark.full]),
                pytest.param((256, 32), 0, torch.float16, marks=[pytest.mark.full]),
            ],
        ),
    ]


@LogSumExpKeepdimFixture
def test_logsumexp_keepdim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    x = torch.randn(*shape, dtype=dtype, device=_device())
    op = LogSumExpFwdOp(dtype=dtype, dim=dim, keepdim=True)

    y_ref = torch.logsumexp(x.float(), dim=dim, keepdim=True).to(dtype)
    y = op(x)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"keepdim logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


class LogSumExp1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(256, torch.float32, marks=[pytest.mark.smoke]),
                pytest.param(256, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(300, torch.float32, marks=[pytest.mark.full]),
            ],
        ),
    ]


@LogSumExp1DFixture
def test_logsumexp_1d(n: int, dtype: torch.dtype) -> None:
    x = torch.randn(n, dtype=dtype, device=_device())
    op = LogSumExpFwdOp(dtype=dtype, dim=-1)

    y_ref = torch.logsumexp(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"1D logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


class LogSumExpMultiDimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype",
            [
                pytest.param((4, 128, 4096), [0, 2], torch.float16, marks=[pytest.mark.smoke]),
                pytest.param((4, 128, 4096), [0, 2], torch.float32, marks=[pytest.mark.full]),
            ],
        ),
    ]


@LogSumExpMultiDimFixture
def test_logsumexp_multidim(shape: tuple, dim: list, dtype: torch.dtype) -> None:
    x = torch.randn(*shape, dtype=dtype, device=_device())
    op = LogSumExpFwdOp(dtype=dtype, dim=dim)

    y_ref = torch.logsumexp(x.float(), dim=dim).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"multidim logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


@pytest.mark.smoke
def test_logsumexp_eval_roofline() -> None:
    M, N = 64, 256
    dtype = torch.float16
    op = LogSumExpFwdOp(dim=-1, dtype=dtype)
    x = torch.randn(M, N, dtype=dtype, device=_device())
    op(x)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 4 * M * N, f"flops {flops} != 4 * M * N = {4 * M * N}"
    assert mem_bytes == (M * N + M) * elem_bytes, (
        f"bytes {mem_bytes} != (M*N + M) * elem_bytes = {(M * N + M) * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
