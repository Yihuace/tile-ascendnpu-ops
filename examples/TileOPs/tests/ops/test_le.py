"""Correctness tests for LeFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_comparison.py`` -- Le subset only
(the shared GPU file also covers eq / ne / gt / lt / ge, which are separate
ops).  Covers flat same-shape float inputs, the broadcast patterns, the
``uint8``-backed bool fast path, the integer-dtype union, the known-equal
edge case, float/complex dtype rejection, and the roofline formula.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``)
  instead of hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards
  removed (NPU availability is a runtime concern).
- T3/T4: no ``tune`` parametrization / autotune try-except in the GPU
  Le tests -- nothing to remove.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.elementwise.le import LeFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.elementwise import RandnPairWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


def _bool_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact comparison for boolean outputs."""
    assert output.dtype == torch.bool, f"Expected bool dtype, got {output.dtype}"
    assert torch.equal(output, output_ref), (
        f"Bool mismatch: {(output != output_ref).sum().item()} elements differ"
    )


class LeTest(RandnPairWorkload, TestBase):
    """Reusable test body for the less-equal op (two flat randn inputs)."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__(n_total, dtype)

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.le(a, b)


class LeFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(
                    4_096,
                    torch.float32,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                ),
                pytest.param(4_096, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(4_096, torch.bfloat16, marks=[pytest.mark.smoke]),
                # Full: larger same-shape coverage
                pytest.param(16_384, torch.float32, marks=[pytest.mark.full]),
            ],
        ),
    ]


@LeFixture
def test_le_op(n_total: int, dtype: torch.dtype) -> None:
    test = LeTest(n_total, dtype)
    shape = (n_total,)
    op = LeFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


_BROADCAST_PATTERNS = [
    ((2, 64, 128), (1, 1, 128)),   # bias-add
    ((2, 64, 128), (2, 64, 1)),    # row broadcast
    ((64, 128), (1, 1)),           # scalar broadcast
]


class LeBroadcastFixture(FixtureBase):
    PARAMS = [
        (
            "a_shape, b_shape",
            [
                pytest.param(
                    a_s,
                    b_s,
                    marks=pytest.mark.smoke if i == 0 else pytest.mark.full,
                )
                for i, (a_s, b_s) in enumerate(_BROADCAST_PATTERNS)
            ],
        ),
    ]


@LeBroadcastFixture
def test_le_broadcast(a_shape: tuple, b_shape: tuple) -> None:
    device = _device()
    a = torch.randn(a_shape, dtype=torch.float16, device=device)
    b = torch.randn(b_shape, dtype=torch.float16, device=device)
    op = LeFwdOp(a_shape=a_shape, b_shape=b_shape, dtype=torch.float16)
    ref = torch.le(a, b)
    with torch.no_grad():
        out = op(a, b)
    _bool_compare(out, ref)
    assert tuple(out.shape) == tuple(ref.shape)


class LeEdgeCaseFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@LeEdgeCaseFixture
def test_le_edge_case(n_total: int, dtype: torch.dtype) -> None:
    """L4: le with known-equal elements at specific positions."""
    device = _device()
    a = torch.randn(n_total, dtype=dtype, device=device)
    b = a.clone()
    # Make some elements differ
    b[::2] = torch.randn(n_total // 2, dtype=dtype, device=device)
    shape = (n_total,)
    op = LeFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.le(a, b)
    with torch.no_grad():
        out = op(a, b)
    assert out.dtype == torch.bool
    assert torch.equal(out, ref)


_INT_DTYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def _gen_int_inputs(
    n: int, dtype: torch.dtype, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype == torch.uint8:
        lo, hi = 0, 16
    elif dtype == torch.int8:
        lo, hi = -16, 16
    else:
        lo, hi = -64, 64
    a = torch.randint(lo, hi, (n,), dtype=dtype, device=device)
    b = torch.randint(lo, hi, (n,), dtype=dtype, device=device)
    # Inject some equal positions so le exercises the True branch.
    b[: n // 4] = a[: n // 4]
    return a, b


class LeIntDtypeFixture(FixtureBase):
    PARAMS = [
        (
            "dtype",
            [pytest.param(dt, marks=pytest.mark.smoke) for dt in _INT_DTYPES],
        ),
    ]


@LeIntDtypeFixture
def test_le_integer_dtype(dtype: torch.dtype) -> None:
    """LeFwdOp matches torch.le on every manifest-declared int dtype."""
    n = 4_096
    shape = (n,)
    a, b = _gen_int_inputs(n, dtype, _device())
    op = LeFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.le(a, b)
    with torch.no_grad():
        out = op(a, b)
    _bool_compare(out, ref)


@pytest.mark.smoke
def test_le_bool_dtype() -> None:
    """LeFwdOp matches torch.le on torch.bool inputs (uint8 storage path)."""
    device = _device()
    n = 4_096
    shape = (n,)
    a = torch.randint(0, 2, (n,), device=device).to(torch.bool)
    b = torch.randint(0, 2, (n,), device=device).to(torch.bool)
    op = LeFwdOp(a_shape=shape, b_shape=shape, dtype=torch.bool)
    ref = torch.le(a, b)
    with torch.no_grad():
        out = op(a, b)
    assert out.dtype == torch.bool
    _bool_compare(out, ref)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.complex64, marks=pytest.mark.smoke),
        pytest.param(torch.float8_e4m3fn, marks=pytest.mark.smoke),
        pytest.param(torch.float8_e5m2, marks=pytest.mark.smoke),
    ],
)
def test_le_rejects_unsupported_dtype(dtype: torch.dtype) -> None:
    """Comparison ops reject dtypes outside the supported set (e.g. complex/fp8)."""
    shape = (16,)
    with pytest.raises(ValueError, match="does not support dtype"):
        LeFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)


@pytest.mark.smoke
def test_le_eval_roofline() -> None:
    """eval_roofline() matches the GPU formula (le_fwd_roofline).

    flops = 1 * N_total; bytes = (a_numel + b_numel) * elem_bytes + N_total
    (bool output is 1 byte per element).
    """
    a_shape, b_shape = (64, 128), (1, 128)
    dtype = torch.float32
    op = LeFwdOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    n_total = 64 * 128
    a_numel, b_numel = 64 * 128, 1 * 128
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == n_total, f"flops {flops} != N = {n_total}"
    assert mem_bytes == (a_numel + b_numel) * elem_bytes + n_total, (
        f"bytes {mem_bytes} != (a + b) * elem_bytes + N = "
        f"{(a_numel + b_numel) * elem_bytes + n_total}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
