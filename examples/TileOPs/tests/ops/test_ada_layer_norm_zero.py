"""Correctness tests for AdaLayerNormZeroFwdOp (NPU).

Ported from the GPU TileOPs ``tests/ops/test_ada_layer_norm_zero.py``.
Uses the device backend instead of hard-coded ``"cuda"`` (T1/T2).

The gated variant shares the AdaLayerNorm kernel (``has_gate=True``), so the
reference is ``geometry = gate * (scale * LayerNorm(x) + shift)`` with the same
``F.layer_norm`` convention used by ``test_ada_layer_norm.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.norm import AdaLayerNormZeroWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


def _ref(x, scale, shift, gate, n, eps):
    normed = F.layer_norm(x.float(), (n,), weight=None, bias=None, eps=eps)
    return (gate.float() * (scale.float() * normed + shift.float())).to(x.dtype)


class AdaLayerNormZeroTest(AdaLayerNormZeroWorkload, TestBase):
    def ref_program(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        # AdaLN-Zero: y = gate * (scale * LayerNorm(x) + shift)
        return _ref(x, scale, shift, gate, self.n, self.eps)


class AdaLayerNormZeroFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(
                    1024, 4096, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
                ),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, marks=pytest.mark.full),
                # DiT-XL/2 shape from the manifest workloads
                pytest.param(1024, 1152, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@AdaLayerNormZeroFixture
def test_ada_layer_norm_zero_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormZeroTest(m, n, dtype)
    op = AdaLayerNormZeroFwdOp(dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_ada_layer_norm_zero_eval_roofline() -> None:
    M, N = 64, 256
    dtype = torch.float16
    op = AdaLayerNormZeroFwdOp(dtype=dtype)
    x = torch.randn(M, N, dtype=dtype, device=_device())
    scale = torch.randn(M, N, dtype=dtype, device=_device())
    shift = torch.randn(M, N, dtype=dtype, device=_device())
    gate = torch.randn(M, N, dtype=dtype, device=_device())
    op(x, scale, shift, gate)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 6 * M * N, f"flops {flops} != 6 * M * N = {6 * M * N}"
    assert mem_bytes == 5 * M * N * elem_bytes, (
        f"bytes {mem_bytes} != 5 * M * N * elem_bytes = {5 * M * N * elem_bytes}"
    )


class AdaLayerNormZero3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@AdaLayerNormZero3DFixture
def test_ada_layer_norm_zero_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """3-D input (batch, seq, hidden) must flatten/unflatten transparently."""
    device = _device()
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=device)
    scale = torch.randn(batch, seq, hidden, dtype=dtype, device=device)
    shift = torch.randn(batch, seq, hidden, dtype=dtype, device=device)
    gate = torch.randn(batch, seq, hidden, dtype=dtype, device=device)

    op = AdaLayerNormZeroFwdOp(dtype=dtype)
    y = op(x, scale, shift, gate)
    y_ref = _ref(x, scale, shift, gate, hidden, 1e-5)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    atol, rtol = _get_tolerances(dtype)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_ada_layer_norm_zero_gate_is_used() -> None:
    """Regression: the gate must actually multiply.

    A zero gate must zero the output regardless of x/scale/shift -- this fails
    if the gated branch is ever wired to the non-gated wrapper by mistake.
    """
    m, n = 8, 512
    dtype = torch.float32
    device = _device()
    x = torch.randn(m, n, dtype=dtype, device=device)
    scale = torch.randn(m, n, dtype=dtype, device=device)
    shift = torch.randn(m, n, dtype=dtype, device=device)
    gate = torch.zeros(m, n, dtype=dtype, device=device)

    op = AdaLayerNormZeroFwdOp(dtype=dtype)
    y = op(x, scale, shift, gate)
    torch.testing.assert_close(
        y, torch.zeros_like(y), atol=1e-6, rtol=1e-6, msg="gate == 0 must zero the output"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
