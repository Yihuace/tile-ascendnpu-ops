"""Correctness tests for AddFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_binary_arith.py`` -- the Add subset only
(same-shape add, broadcast add, dtype/alpha rejection, roofline).  The shared
GPU file also covers sub / mul / div / ... / maximum / minimum, which are not
migrated.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) instead of
  hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards removed
  (NPU availability is a runtime concern).
- T3/T4: the GPU Add tests carry no ``tune`` parametrization / autotune
  try-except -- nothing to remove.
"""

from math import prod

import pytest
import torch

from tileops.benchmark.msprof import _extract_op_init_args
from tileops.device import get_device_backend
from tileops.kernels.elementwise.add import AddFwdKernel
from tileops.ops.elementwise.add import AddFwdOp
from tileops.kernels.elementwise.add.add import BCAST_ENABLED
from tileops.testing.test_base import TestBase
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


class AddTest(TestBase):
    """Test fixture for AddFwdOp -- two randn operands with broadcast."""

    def __init__(
        self,
        input_shape: tuple,
        other_shape: tuple,
        dtype: torch.dtype,
        alpha: float = 1,
    ):
        self.input_shape = tuple(input_shape)
        self.other_shape = tuple(other_shape)
        self.shape = self.input_shape
        self.dtype = dtype
        self.alpha = alpha
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.other_shape))
        self.n_total = prod(self.out_shape) if self.out_shape else 1

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = _device()
        a = torch.randn(self.input_shape, device=device, dtype=self.dtype)
        b = torch.randn(self.other_shape, device=device, dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.add(a.float(), b.float(), alpha=self.alpha).to(a.dtype)


class AddSameShapeFixture(FixtureBase):
    PARAMS = [
        (
            "input_shape, other_shape, dtype, alpha",
            [
                pytest.param(
                    (4096,),
                    (4096,),
                    torch.float32,
                    1,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                ),
                pytest.param((4096,), (4096,), torch.float16, 1, marks=[pytest.mark.smoke]),
                pytest.param((4096,), (4096,), torch.bfloat16, 1, marks=[pytest.mark.smoke]),
                # Full: larger same-shape coverage
                pytest.param((16_384,), (16_384,), torch.float16, 1, marks=[pytest.mark.full]),
            ],
        ),
    ]


@AddSameShapeFixture
def test_add_same_shape(input_shape, other_shape, dtype, alpha) -> None:
    test = AddTest(input_shape, other_shape, dtype, alpha)
    op = AddFwdOp(input=input_shape, other=other_shape, dtype=dtype, alpha=alpha)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class AddBroadcastFixture(FixtureBase):
    PARAMS = [
        (
            "input_shape, other_shape, dtype, alpha",
            [
                # bias-add: (B,S,D) + (1,1,D)
                pytest.param(
                    (2, 512, 768), (1, 1, 768), torch.float16, 1, marks=[pytest.mark.smoke]
                ),
                # row broadcast: (B,S,D) + (B,S,1)
                pytest.param((2, 512, 768), (2, 512, 1), torch.float16, 1, marks=[pytest.mark.full]),
                # scalar broadcast: (M,N) + (1,1)
                pytest.param((1024, 1024), (1, 1), torch.float16, 1, marks=[pytest.mark.full]),
                # 2-byte scalar broadcast -> period path (cnn-feat layout)
                pytest.param(
                    (4, 32, 8, 8), (32, 1, 1), torch.bfloat16, 1, marks=[pytest.mark.full]
                ),
                # interleaved: (A,1,C) + (1,B,1)
                pytest.param((4, 1, 8), (1, 8, 1), torch.float16, 1, marks=[pytest.mark.full]),
            ],
        ),
    ]


@AddBroadcastFixture
def test_add_broadcast(input_shape, other_shape, dtype, alpha) -> None:
    test = AddTest(input_shape, other_shape, dtype, alpha)
    op = AddFwdOp(input=input_shape, other=other_shape, dtype=dtype, alpha=alpha)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_add_alpha(alpha: float, dtype: torch.dtype) -> None:
    """Non-default ``alpha`` runs through the same kernel (baked at construction)."""
    n = 4096
    test = AddTest((n,), (n,), dtype, alpha)
    op = AddFwdOp(input=(n,), other=(n,), dtype=dtype, alpha=alpha)
    test.check(op, *test.gen_inputs())


class AddNonFloatRejectFixture(FixtureBase):
    PARAMS = [
        (
            "dtype",
            [
                pytest.param(torch.int32, marks=pytest.mark.smoke),
                pytest.param(torch.float8_e4m3fn, marks=pytest.mark.smoke),
                pytest.param(torch.bool, marks=pytest.mark.smoke),
            ],
        ),
    ]


@AddNonFloatRejectFixture
def test_add_rejects_non_float_dtype(dtype: torch.dtype) -> None:
    """AddFwdOp is restricted to the float subset (manifest GPU union aside)."""
    with pytest.raises(ValueError, match="does not support dtype"):
        AddFwdOp(input=(16,), other=(16,), dtype=dtype)


@pytest.mark.smoke
def test_add_kernel_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        AddFwdKernel(N_total=16, dtype=torch.int32, alpha=1)


@pytest.mark.smoke
def test_add_rejects_dtype_mismatch() -> None:
    """forward() must reject operands whose dtype disagrees with __init__."""
    shape = (16,)
    device = _device()
    op = AddFwdOp(input=shape, other=shape, dtype=torch.float32)
    a = torch.randn(shape, device=device, dtype=torch.float32)
    b = torch.randn(shape, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="other.dtype"):
        op(a, b)


@pytest.mark.smoke
def test_add_broadcast_output_shape() -> None:
    a_shape, b_shape = (3, 1), (1, 4)
    dtype = torch.float32
    device = _device()
    op = AddFwdOp(input=a_shape, other=b_shape, dtype=dtype)
    a = torch.randn(a_shape, device=device, dtype=dtype)
    b = torch.randn(b_shape, device=device, dtype=dtype)
    out = op(a, b)
    assert tuple(out.shape) == (3, 4)


@pytest.mark.smoke
def test_add_eval_roofline() -> None:
    input_shape, other_shape = (64, 128), (1, 128)
    dtype = torch.float16
    op = AddFwdOp(input=input_shape, other=other_shape, dtype=dtype)
    n_total = 64 * 128
    b_len = 128
    elem_bytes = dtype.itemsize
    flops, mem_bytes = op.eval_roofline()
    assert flops == n_total, f"flops {flops} != N = {n_total}"
    # Row-broadcast reuse path (opt-in): 2 reads at N + one broadcast period;
    # otherwise the flat path materializes ``other`` -> 3*N.
    if BCAST_ENABLED:
        assert mem_bytes == (2 * n_total + b_len) * elem_bytes, (
            f"bytes {mem_bytes} != (2 * N + B) * elem_bytes = "
            f"{(2 * n_total + b_len) * elem_bytes}"
        )
    else:
        assert mem_bytes == 3 * n_total * elem_bytes, (
            f"bytes {mem_bytes} != 3 * N * elem_bytes = {3 * n_total * elem_bytes}"
        )

    # 2-byte scalar broadcast -> period path: one period (B*L) is charged.
    p_input, p_other = (4, 32, 8, 8), (32, 1, 1)
    p_op = AddFwdOp(input=p_input, other=p_other, dtype=dtype)
    assert p_op.bcast_plan["mode"] == "period", p_op.bcast_plan
    p_n = 4 * 32 * 8 * 8
    p_period = 32 * (8 * 8)
    _, p_bytes = p_op.eval_roofline()
    assert p_bytes == (2 * p_n + p_period) * dtype.itemsize, (
        f"bytes {p_bytes} != (2*N + period) * elem_bytes = "
        f"{(2 * p_n + p_period) * dtype.itemsize}"
    )

    # Fallback (no reuse plan): ``other`` is materialized -> 3*N traffic.
    fb_input, fb_other = (4, 1, 8), (1, 8, 1)
    fb_op = AddFwdOp(input=fb_input, other=fb_other, dtype=dtype)
    fb_n = 4 * 8 * 8
    assert fb_op.bcast_plan is None
    _, fb_bytes = fb_op.eval_roofline()
    assert fb_bytes == 3 * fb_n * elem_bytes, (
        f"bytes {fb_bytes} != 3 * N * elem_bytes = {3 * fb_n * elem_bytes}"
    )

    op_alpha = AddFwdOp(input=input_shape, other=other_shape, dtype=dtype, alpha=2.0)
    flops_alpha, _ = op_alpha.eval_roofline()
    assert flops_alpha == 2 * n_total, f"flops {flops_alpha} != 2 * N = {2 * n_total}"

    # The msprof runner reconstructs Op instances in a subprocess from
    # constructor-name attributes.  Keep this contract covered explicitly.
    assert _extract_op_init_args(op) == {
        "input": input_shape,
        "other": other_shape,
        "dtype": dtype,
        "alpha": 1,
    }


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
