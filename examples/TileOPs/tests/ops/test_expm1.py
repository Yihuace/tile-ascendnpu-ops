"""Correctness tests for Expm1FwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_unary_math.py`` — the ``expm1`` subset
only (the shared GPU file covers every implemented unary math op; only this
migrated op's cases are ported).  Covers L1 correctness across supported
float dtypes and the L4 edge case for the numerically sensitive tails of
``expm1``.

Adaptations from GPU (TileOPs) to NPU:

- T1: manual tensor creation uses ``get_device_backend().name`` instead of
      the hard-coded ``"cuda"``.
- T2: the GPU ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)``
      guard is dropped (the NPU device backend resolves the target).
- T3/T4: the GPU ``tune`` parametrization and autotune try/except are
      omitted — the NPU kernel uses heuristic config selection only.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.expm1 import Expm1FwdKernel
from tileops.ops.elementwise.expm1 import Expm1FwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


class MathFixture(FixtureBase):
    """Parametrize over supported float dtypes for the unary math op."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(
                1_048_576, torch.float32,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
            ),
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
            # Full: larger follow-up coverage
            pytest.param(4_000_000, torch.float16, marks=pytest.mark.full),
            pytest.param(4_000_000, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


class MathEdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


class UnaryMathTest(TestBase):
    """Generic test fixture for a single-input, single-output unary op."""

    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None, ref_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._gen_fn = gen_fn or _randn
        self._ref_fn = ref_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (self._gen_fn(self.n_total, self.dtype),)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return self._ref_fn(x)


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-3, 1e-3
    if dtype == torch.bfloat16:
        return 1.6e-2, 1.6e-2
    return 1e-5, 1e-5


def _randn(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(n, device=_device(), dtype=dtype)


def _repeat_values(values: list[float], n: int, dtype: torch.dtype) -> torch.Tensor:
    base = torch.tensor(values, device=_device(), dtype=dtype)
    repeats = (n + len(values) - 1) // len(values)
    return base.repeat(repeats)[:n]


def _make_math_test(n_total, dtype, gen_fn, ref_fn, op_cls):
    """Build test, instantiate op, and run check."""
    test = UnaryMathTest(n_total, dtype, gen_fn=gen_fn, ref_fn=ref_fn)
    op = op_cls(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# L1 test


@MathFixture
def test_expm1(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.expm1, Expm1FwdOp)


@pytest.mark.smoke
def test_expm1_rejects_non_float_dtype() -> None:
    # Every non-float dtype must be rejected by the float-only kernel.
    for bad_dtype in (torch.int32, torch.int64, torch.bool):
        with pytest.raises(ValueError, match="only supports dtypes"):
            Expm1FwdKernel(N_total=16, dtype=bad_dtype)


# L4 edge test


@MathEdgeFixture
def test_expm1_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([0.0, 88.8, -88.8, 1e-7], n, d),
        torch.expm1,
        Expm1FwdOp,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
