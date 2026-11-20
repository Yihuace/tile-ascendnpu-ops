"""Correctness tests for HardtanhFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_activation.py`` — the ``hardtanh``
subset only (the shared GPU file also covers Elu and LeakyReLU; only this
migrated op's cases are ported).

The GPU source has a single correctness case (``test_hardtanh``, default
``min_val=-1.0`` / ``max_val=1.0``) plus a float-only dtype-rejection case;
there is no ``min_val`` / ``max_val`` parametrization in the GPU source and
none is added here.  Every golden below uses the same default bounds
(-1.0, 1.0) that the op is constructed with, so the reference and the
operator parameters stay consistent.

Adaptations from GPU (TileOPs) to NPU:

- T1: the test fixture inherits ``RandnWorkload``, whose ``gen_inputs``
      resolves ``device`` from :func:`get_device_backend` instead of the
      hard-coded ``"cuda"``.
- T2: the GPU ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)``
      guard is dropped (the NPU device backend resolves the target).
- T3/T4: the GPU ``tune`` parametrization and autotune try/except are
      omitted — the NPU kernel uses heuristic config selection only.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.kernels.elementwise.hardtanh import HardtanhFwdKernel
from tileops.ops.elementwise.hardtanh import HardtanhFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.workload_base import FixtureBase, RandnWorkload


def _device() -> str:
    return get_device_backend().name


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


class HardtanhTest(RandnWorkload, TestBase):
    """Test fixture for HardtanhFwdOp -- flat 1-D randn input, bounds (-1, 1)."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__((n_total,), dtype)
        self.n_total = n_total

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardtanh(x.float(), -1.0, 1.0).to(x.dtype)


class ActivationFixture(FixtureBase):
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


class ActivationEdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


def _repeat_values(values: list[float], n: int, dtype: torch.dtype) -> torch.Tensor:
    base = torch.tensor(values, device=_device(), dtype=dtype)
    repeats = (n + len(values) - 1) // len(values)
    return base.repeat(repeats)[:n]


@ActivationFixture
def test_hardtanh(n_total: int, dtype: torch.dtype) -> None:
    test = HardtanhTest(n_total, dtype)
    op = HardtanhFwdOp(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_hardtanh_rejects_non_float_dtype() -> None:
    # Every non-float dtype must be rejected by the float-only kernel.
    for bad_dtype in (torch.int32, torch.int64, torch.bool):
        with pytest.raises(ValueError, match="only supports dtypes"):
            HardtanhFwdKernel(N_total=16, dtype=bad_dtype)


@pytest.mark.smoke
def test_hardtanh_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = HardtanhFwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


@ActivationEdgeFixture
def test_hardtanh_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: below/at/above both bounds and the linear interior (default -1, 1)."""
    test = HardtanhTest(n_total, dtype)
    x = _repeat_values([-88.8, -1.0, -0.5, 0.0, 0.5, 1.0, 88.8], n_total, dtype)
    op = HardtanhFwdOp(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, x, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
