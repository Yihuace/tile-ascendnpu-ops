"""Correctness tests for ClampFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_special_elementwise.py`` — the
Tensor-bound ``clamp`` subset only (the shared GPU file also covers the
generative ALiBi op; only this migrated op's cases are ported).

The GPU ``ClampTensorFwdKernel`` broadcasts input / min / max together and
dispatches to a has_min / has_max / both kernel variant, so this file covers
three-way broadcast, 0-dim Tensor bounds, min-only / max-only routing, the
``min=None and max=None`` rejection, forward-time None/Tensor agreement, NaN
propagation and the roofline assertion.

Adaptations from GPU (TileOPs) to NPU:

- T1: tensor creation uses ``get_device_backend().name`` instead of the
  hard-coded ``"cuda"``.
- T2: the GPU ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)``
  guard is dropped (the NPU device backend resolves the target).
- T3/T4: the GPU ``tune`` parametrization and autotune try/except are
  omitted — the NPU kernel uses heuristic config selection only.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.clamp import ClampTensorFwdKernel
from tileops.ops.elementwise.clamp import ClampFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.workload_base import FixtureBase, RandnWorkload


def _device() -> str:
    return get_device_backend().name


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1.6e-2, 1.6e-2  # bfloat16


class ClampTensorTest(RandnWorkload, TestBase):
    """Broadcast test fixture for ClampFwdOp (Tensor min / max)."""

    def __init__(self, input_shape: tuple, min_shape: tuple, max_shape: tuple, dtype):
        super().__init__(tuple(input_shape), dtype)
        self.input_shape = tuple(input_shape)
        self.min_shape = tuple(min_shape)
        self.max_shape = tuple(max_shape)

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = _device()
        x = torch.randn(self.input_shape, device=device, dtype=self.dtype)
        mn = torch.randn(self.min_shape, device=device, dtype=self.dtype) - 0.5
        mx = torch.randn(self.max_shape, device=device, dtype=self.dtype) + 0.5
        return x, mn, mx

    def ref_program(self, x: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x.float(), mn.float(), mx.float()).to(x.dtype)


class ClampTensorFixture(FixtureBase):
    PARAMS = [
        ("input_shape, min_shape, max_shape, dtype", [
            pytest.param(
                (1024, 1024), (1024, 1024), (1024, 1024), torch.float32,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
            ),
            pytest.param(
                (1024, 1024), (1024, 1024), (1024, 1024), torch.float16,
                marks=[pytest.mark.smoke],
            ),
            pytest.param(
                (1024, 1024), (1024, 1024), (1024, 1024), torch.bfloat16,
                marks=[pytest.mark.smoke],
            ),
            # Full: 3-way broadcast
            pytest.param((4, 8), (1, 8), (4, 1), torch.float32, marks=[pytest.mark.full]),
            # Full: 0-dim Tensor bounds
            pytest.param((4, 8), (), (), torch.float32, marks=[pytest.mark.full]),
        ]),
    ]


@ClampTensorFixture
def test_clamp_tensor(
    input_shape: tuple, min_shape: tuple, max_shape: tuple, dtype: torch.dtype
) -> None:
    test = ClampTensorTest(input_shape, min_shape, max_shape, dtype)
    op = ClampFwdOp(input=input_shape, min=min_shape, max=max_shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_clamp_min_only_tensor_parity() -> None:
    """ClampFwdOp(min=Tensor, max=None) routes to the has_min-only kernel."""
    x = torch.randn((4, 8), device=_device(), dtype=torch.float32)
    mn = torch.randn((4, 8), device=_device(), dtype=torch.float32) - 0.5
    ref = torch.clamp(x, mn, None)
    op = ClampFwdOp(input=(4, 8), min=(4, 8), max=None, dtype=torch.float32)
    torch.testing.assert_close(op(x, mn, None), ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_clamp_max_only_tensor_parity() -> None:
    """ClampFwdOp(min=None, max=Tensor) routes to the has_max-only kernel."""
    x = torch.randn((4, 8), device=_device(), dtype=torch.float32)
    mx = torch.randn((4, 8), device=_device(), dtype=torch.float32) + 0.5
    ref = torch.clamp(x, None, mx)
    op = ClampFwdOp(input=(4, 8), min=None, max=(4, 8), dtype=torch.float32)
    torch.testing.assert_close(op(x, None, mx), ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_clamp_both_none_rejected() -> None:
    """ClampFwdOp must reject min=None and max=None (no-op clamp is invalid)."""
    with pytest.raises(ValueError, match="at least one of"):
        ClampFwdOp(input=(4,), min=None, max=None, dtype=torch.float32)


@pytest.mark.smoke
def test_clamp_runtime_tensor_none_must_match_init() -> None:
    """Forward-time None / Tensor presence must agree with __init__ config."""
    x = torch.randn(4, device=_device(), dtype=torch.float32)
    mn = torch.zeros(4, device=_device(), dtype=torch.float32)

    # Configured for min-only at __init__, then passed a Tensor for max:
    op = ClampFwdOp(input=(4,), min=(4,), max=None, dtype=torch.float32)
    with pytest.raises(ValueError, match="max"):
        op(x, mn, mn)

    # Configured for max-only at __init__, then passed a Tensor for min:
    op2 = ClampFwdOp(input=(4,), min=None, max=(4,), dtype=torch.float32)
    with pytest.raises(ValueError, match="min"):
        op2(x, mn, mn)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_tensor_nan_propagation(dtype: torch.dtype) -> None:
    """ClampFwdOp must match torch.clamp NaN semantics (Tensor min + max)."""
    device = _device()
    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device=device, dtype=dtype)
    mn = torch.tensor([-1.0, -1.0, float("nan"), -1.0], device=device, dtype=dtype)
    mx = torch.tensor([1.0, 1.0, 1.0, float("nan")], device=device, dtype=dtype)

    ref = torch.clamp(x, mn, mx)
    op = ClampFwdOp(input=(4,), min=(4,), max=(4,), dtype=dtype)
    out = op(x, mn, mx)
    torch.testing.assert_close(out, ref, equal_nan=True, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_clamp_tensor_rejects_non_float_dtype() -> None:
    for bad_dtype in (torch.int32, torch.int64, torch.bool):
        with pytest.raises(ValueError, match="only supports dtypes"):
            ClampTensorFwdKernel(N_total=16, dtype=bad_dtype, has_min=True, has_max=True)


@pytest.mark.smoke
def test_clamp_tensor_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = ClampFwdOp(input=(N,), min=(N,), max=(N,), dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    assert mem_bytes == 4 * N * elem_bytes, (
        f"bytes {mem_bytes} != 4 * N * elem_bytes = {4 * N * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
