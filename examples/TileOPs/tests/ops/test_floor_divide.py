"""Correctness tests for FloorDivideFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_binary_arith.py`` -- FloorDivide subset
only (the shared GPU file also covers add / sub / mul / div / remainder / pow /
lerp / maximum / minimum, which are separate ops).  Covers flat same-shape
float inputs, the broadcast patterns, an L4 edge case, the float-only dtype
contract, runtime dtype mismatch, and the roofline formula.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``)
  instead of hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards
  removed (NPU availability is a runtime concern).
- T3/T4: no ``tune`` parametrization / autotune try-except in the GPU
  FloorDivide tests -- nothing to remove.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.elementwise.floor_divide import FloorDivideFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.elementwise import PositivePairWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


class FloorDivideTest(PositivePairWorkload, TestBase):
    """Floor divide reference matches the kernel: fp32 division+floor, cast back."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__(n_total, dtype)

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # fp32 division+floor, cast back to native dtype.
        return torch.floor(a.float() / b.float()).to(a.dtype)


class FloorDivideFixture(FixtureBase):
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


@FloorDivideFixture
def test_floor_divide_op(n_total: int, dtype: torch.dtype) -> None:
    test = FloorDivideTest(n_total, dtype)
    shape = (n_total,)
    op = FloorDivideFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    # Floor divide in reduced precision can differ by 1; use atol=1.0.
    atol = 1.0 if dtype != torch.float32 else 1e-5
    test.check(op, *test.gen_inputs(), atol=atol, rtol=0.0)


_BROADCAST_PATTERNS = [
    ((2, 64, 128), (1, 1, 128)),   # bias-add
    ((2, 64, 128), (2, 64, 1)),    # row broadcast
    ((64, 128), (1, 1)),           # scalar broadcast
]


class FloorDivideBroadcastFixture(FixtureBase):
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


@FloorDivideBroadcastFixture
def test_floor_divide_broadcast(a_shape: tuple, b_shape: tuple) -> None:
    device = _device()
    a = torch.rand(a_shape, dtype=torch.float16, device=device) + 0.1
    b = torch.rand(b_shape, dtype=torch.float16, device=device) + 0.1
    op = FloorDivideFwdOp(a_shape=a_shape, b_shape=b_shape, dtype=torch.float16)
    ref = torch.floor(a.float() / b.float()).to(a.dtype)
    with torch.no_grad():
        out = op(a, b)
    # fp16 floor rounding tolerance.
    torch.testing.assert_close(out, ref, atol=1.0, rtol=0.0)
    assert tuple(out.shape) == tuple(ref.shape)


class FloorDivideEdgeCaseFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FloorDivideEdgeCaseFixture
def test_floor_divide_edge_case(n_total: int, dtype: torch.dtype) -> None:
    """L4: floor_divide on positive inputs (fp32, quotient boundaries)."""
    device = _device()
    a = torch.rand(n_total, dtype=dtype, device=device) + 0.1
    b = torch.rand(n_total, dtype=dtype, device=device) + 0.1
    shape = (n_total,)
    op = FloorDivideFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.floor(a.float() / b.float()).to(dtype)
    with torch.no_grad():
        out = op(a, b)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.int64, marks=pytest.mark.smoke),
        pytest.param(torch.int32, marks=pytest.mark.smoke),
        pytest.param(torch.int16, marks=pytest.mark.smoke),
        pytest.param(torch.int8, marks=pytest.mark.smoke),
        pytest.param(torch.uint8, marks=pytest.mark.smoke),
        pytest.param(torch.bool, marks=pytest.mark.smoke),
    ],
)
def test_floor_divide_rejects_integer_dtype(dtype: torch.dtype) -> None:
    """FloorDivideFwdOp is float-only; integer/bool dtypes are rejected."""
    shape = (16,)
    with pytest.raises(ValueError, match="does not support dtype"):
        FloorDivideFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)


@pytest.mark.smoke
def test_floor_divide_rejects_runtime_dtype_mismatch() -> None:
    """Runtime inputs must fail fast instead of reaching backend lowering."""
    device = _device()
    op = FloorDivideFwdOp(a_shape=(16,), b_shape=(16,), dtype=torch.float16)
    a = torch.randn(16, dtype=torch.float32, device=device)
    b = torch.randn(16, dtype=torch.float16, device=device)
    with pytest.raises(ValueError, match="Expected input.dtype"):
        op(a, b)


@pytest.mark.smoke
def test_floor_divide_eval_roofline() -> None:
    """eval_roofline() matches the GPU formula (floor_divide_fwd_roofline).

    flops = 2 * N_total; bytes = (a_numel + b_numel + N_total) * elem_bytes.
    """
    a_shape, b_shape = (64, 128), (1, 128)
    dtype = torch.float32
    op = FloorDivideFwdOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    n_total = 64 * 128
    a_numel, b_numel = 64 * 128, 1 * 128
    elem_bytes = dtype.itemsize
    flops, mem_bytes = op.eval_roofline()
    assert flops == 2 * n_total, f"flops {flops} != 2 * N = {2 * n_total}"
    assert mem_bytes == (a_numel + b_numel + n_total) * elem_bytes, (
        f"bytes {mem_bytes} != (a + b + N) * elem_bytes = "
        f"{(a_numel + b_numel + n_total) * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
