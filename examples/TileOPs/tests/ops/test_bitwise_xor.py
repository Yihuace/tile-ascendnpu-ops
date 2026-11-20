"""Correctness tests for BitwiseXorFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_bitwise.py`` — BitwiseXor subset
only (the shared GPU file also covers bitwise_and / bitwise_or /
bitwise_not, which are migrated under separate test files).  Covers flat
same-shape inputs, the broadcast patterns, the ``uint8``-backed bool fast
path, float dtype rejection, and the roofline formula.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``)
  instead of hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards
  removed (NPU availability is a runtime concern).
- T3/T4: no ``tune`` parametrization / autotune try-except in the GPU
  BitwiseXor tests — nothing to remove.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.elementwise.bitwise_xor import BitwiseXorFwdOp
from tileops.testing.test_base import TestBase, exact_compare
from tileops.workloads.elementwise import BitwiseWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


class BitwiseXorTest(BitwiseWorkload, TestBase):
    """Reusable test body for bitwise binary ops (flat ``dtype`` inputs)."""

    def __init__(self, n_total: int, dtype: torch.dtype = torch.int32):
        super().__init__(n_total, dtype)

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_xor(a, b)


class BitwiseXorFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(
                    4_096,
                    torch.int32,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                ),
                pytest.param(4_096, torch.int64, marks=[pytest.mark.smoke]),
                pytest.param(4_096, torch.uint8, marks=[pytest.mark.smoke]),
                # Full: larger follow-up coverage
                pytest.param(16_384, torch.int32, marks=[pytest.mark.full]),
                pytest.param(16_384, torch.int16, marks=[pytest.mark.full]),
            ],
        ),
    ]


@BitwiseXorFixture
def test_bitwise_xor_op(n_total: int, dtype: torch.dtype) -> None:
    test = BitwiseXorTest(n_total, dtype)
    shape = (n_total,)
    op = BitwiseXorFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=exact_compare)


_BROADCAST_PATTERNS = [
    ((2, 64, 128), (1, 1, 128)),  # bias-add
    ((2, 64, 128), (2, 64, 1)),  # row broadcast
    ((64, 128), (1, 1)),  # scalar broadcast
]


class BitwiseXorBroadcastFixture(FixtureBase):
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


@BitwiseXorBroadcastFixture
def test_bitwise_xor_broadcast(a_shape: tuple, b_shape: tuple) -> None:
    device = _device()
    a = torch.randint(-1000, 1000, a_shape, dtype=torch.int32, device=device)
    b = torch.randint(-1000, 1000, b_shape, dtype=torch.int32, device=device)
    op = BitwiseXorFwdOp(a_shape=a_shape, b_shape=b_shape, dtype=torch.int32)
    ref = torch.bitwise_xor(a, b)
    with torch.no_grad():
        out = op(a, b)
    exact_compare(out, ref)
    assert tuple(out.shape) == tuple(ref.shape)


class BoolBitwiseXorFixture(FixtureBase):
    PARAMS = [
        (
            "a_shape, b_shape",
            [
                pytest.param(
                    a_s,
                    b_s,
                    marks=pytest.mark.smoke if a_s == b_s else pytest.mark.full,
                )
                for a_s, b_s in [
                    ((2048, 4096), (2048, 4096)),
                    ((2, 512, 768), (1, 1, 768)),
                ]
            ],
        ),
    ]


@BoolBitwiseXorFixture
def test_bitwise_xor_bool_fast_path(a_shape: tuple, b_shape: tuple) -> None:
    device = _device()
    a = torch.randint(0, 2, a_shape, device=device).bool()
    b = torch.randint(0, 2, b_shape, device=device).bool()
    op = BitwiseXorFwdOp(a_shape=a_shape, b_shape=b_shape, dtype=torch.bool)
    ref = torch.bitwise_xor(a, b)
    with torch.no_grad():
        out = op(a, b)
    assert out.dtype == torch.bool
    exact_compare(out, ref)


# Stage 4 round 8 (v4): int64 goes through the host-side int64 -> two int32
# word packing on the channel / scalar broadcast forms and falls back to the
# int64 kernel everywhere else, so every broadcast *form* (full, row, channel,
# scalar) plus the odd-``L`` fallback must be covered for int64 explicitly.
_INT64_BROADCAST_PATTERNS = [
    ((2048, 2048), (2048, 2048), "full"),
    ((512, 1024), (1, 1024), "row"),
    ((16, 64, 32), (1, 64, 1), "channel"),
    ((256, 512), (256, 1), "col"),
    ((256, 512), (1, 1), "scalar"),
    ((64, 96), (), "scalar-0d"),
    ((64, 12), (64, 12), "even-l12"),
    ((64, 15), (64, 15), "odd-l15-fallback"),
    ((32, 15), (32, 1), "odd-col-l15-fallback"),
]


class BitwiseXorInt64BroadcastFixture(FixtureBase):
    PARAMS = [
        (
            "a_shape, b_shape",
            [
                pytest.param(
                    a_s,
                    b_s,
                    id=label,
                    marks=pytest.mark.smoke if label == "channel" else pytest.mark.full,
                )
                for a_s, b_s, label in _INT64_BROADCAST_PATTERNS
            ],
        ),
    ]


@BitwiseXorInt64BroadcastFixture
def test_bitwise_xor_int64_broadcast(a_shape: tuple, b_shape: tuple) -> None:
    device = _device()
    torch.manual_seed(0)
    a = torch.randint(-(2**62), 2**62, a_shape, dtype=torch.int64, device=device)
    b = torch.randint(-(2**62), 2**62, b_shape, dtype=torch.int64, device=device)
    # pin the two extremes so the low/high word split sees 0x8000.. / 0x7fff..
    if a.numel() > 2:
        a.view(-1)[0] = -(2**63)
        a.view(-1)[1] = 2**63 - 1
    op = BitwiseXorFwdOp(a_shape=a_shape, b_shape=b_shape, dtype=torch.int64)
    ref = torch.bitwise_xor(a, b)
    with torch.no_grad():
        out = op(a, b)
    exact_compare(out, ref)
    assert tuple(out.shape) == tuple(ref.shape)
    assert out.dtype == torch.int64


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, marks=pytest.mark.smoke),
        pytest.param(torch.bfloat16, marks=pytest.mark.smoke),
        pytest.param(torch.float32, marks=pytest.mark.smoke),
    ],
)
def test_bitwise_xor_rejects_float_dtype(dtype: torch.dtype) -> None:
    """Binary bitwise ops only support integer/bool dtypes; floats are rejected."""
    shape = (16,)
    with pytest.raises(ValueError, match="does not support dtype"):
        BitwiseXorFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)


@pytest.mark.smoke
def test_bitwise_xor_eval_roofline() -> None:
    """eval_roofline() matches the GPU formula (bitwise_xor_fwd_roofline).

    flops = 1 * N_total; bytes = (a_numel + b_numel + N_total) * elem_bytes.
    """
    a_shape, b_shape = (64, 128), (1, 128)
    dtype = torch.int32
    op = BitwiseXorFwdOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    n_total = 64 * 128
    a_numel, b_numel = 64 * 128, 1 * 128
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == n_total, f"flops {flops} != N = {n_total}"
    assert mem_bytes == (a_numel + b_numel + n_total) * elem_bytes, (
        f"bytes {mem_bytes} != (a + b + N) * elem_bytes = "
        f"{(a_numel + b_numel + n_total) * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
