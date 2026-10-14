"""Correctness tests for FusedAddLayerNormFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_layer_norm.py`` -- the
``FusedAddLayerNorm*`` subset only (the shared GPU file also covers
``LayerNormFwdOp`` and its large-offset / misaligned cases, which are not part
of this op).

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) instead of
  hard-coded ``device="cuda"``.
- T2: no ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards.
- T3: the ``tune`` parameterization dimension is removed (K8);
  ``request_tune()`` is not called.
- T4: no autotune try-except to remove.

The reference ``ref_program`` is an independent pure-PyTorch implementation of
the fused residual-add + LayerNorm contract (upcast to fp32, round the residual
sum back to the input dtype, then normalize), which is device-independent and
can be verified on CPU.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.kernels.norm.fused_add_layer_norm import FusedAddLayerNormKernel
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.norm import FusedAddLayerNormWorkload
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


class FusedAddLayerNormTest(FusedAddLayerNormWorkload, TestBase):
    """Test fixture for FusedAddLayerNormFwdOp.

    Inherits ``gen_inputs`` (four operands) from the workload and implements the
    independent ``ref_program``.
    """

    def ref_program(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # residual_out = x + residual, rounded back to the input dtype.
        added = (x.float() + residual.float()).to(x.dtype)
        # y = LayerNorm(added) with the affine pair.
        y = F.layer_norm(
            added.float(),
            (self.n,),
            weight.float(),
            bias.float(),
            self.eps,
        ).to(x.dtype)
        return y, added


class FusedAddLayerNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                # Smoke: small Llama-style block (all three dtypes)
                pytest.param(
                    64, 4096, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
                ),
                pytest.param(64, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(64, 4096, torch.bfloat16, marks=pytest.mark.smoke),
                # Standard aligned shapes
                pytest.param(2048, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(2048, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-power-of-two hidden dim (5120) -- fp32
                pytest.param(2048, 5120, torch.float32, marks=pytest.mark.full),
                # Tail-M: M not divisible by the row block
                pytest.param(1025, 4096, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@FusedAddLayerNormFixture
def test_fused_add_layer_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = FusedAddLayerNormTest(m, n, dtype)
    op = FusedAddLayerNormFwdOp(dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class FusedAddLayerNorm3DFixture(FixtureBase):
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


@FusedAddLayerNorm3DFixture
def test_fused_add_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3-D input ``(batch, seq, hidden)``."""
    device = _device()
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=device)
    residual = torch.randn(batch, seq, hidden, dtype=dtype, device=device)
    weight = torch.randn(hidden, dtype=dtype, device=device)
    bias = torch.randn(hidden, dtype=dtype, device=device)

    test = FusedAddLayerNormTest(batch * seq, hidden, dtype)
    op = FusedAddLayerNormFwdOp(dtype=dtype)

    y_ref, added_ref = test.ref_program(x, residual, weight, bias)
    y, residual_out = op(x, residual, weight, bias)

    atol, rtol = _get_tolerances(dtype)
    assert y.shape == x.shape and residual_out.shape == x.shape
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(residual_out, added_ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_fused_add_layer_norm_root_shape() -> None:
    """The op also accepts the normalized 2-D ``(M, N)`` form directly."""
    m, n, dtype = 16, 1152, torch.float16
    test = FusedAddLayerNormTest(m, n, dtype)
    op = FusedAddLayerNormFwdOp(dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_fused_add_layer_norm_eval_roofline() -> None:
    m, n, dtype = 64, 4096, torch.float16
    test = FusedAddLayerNormTest(m, n, dtype)
    op = FusedAddLayerNormFwdOp(dtype=dtype)
    op(*test.gen_inputs())
    elem_bytes = dtype.itemsize
    flops, mem_bytes = op.eval_roofline()
    assert flops == 8 * m * n, f"flops {flops} != 8*M*N = {8 * m * n}"
    assert mem_bytes == (4 * m * n + 2 * n) * elem_bytes, (
        f"bytes {mem_bytes} != (4*M*N + 2*N)*elem_bytes = {(4 * m * n + 2 * n) * elem_bytes}"
    )


@pytest.mark.smoke
def test_fused_add_layer_norm_rejects_bad_shape() -> None:
    shape = (16, 4096)
    device = _device()
    op = FusedAddLayerNormFwdOp(dtype=torch.float16)
    x = torch.randn(shape, device=device, dtype=torch.float16)
    residual = torch.randn(shape, device=device, dtype=torch.float16)
    weight = torch.randn(shape[-1], device=device, dtype=torch.float16)
    bias = torch.randn(shape[-1], device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="residual shape"):
        op(x, residual[:, : shape[-1] // 2], weight, bias)


@pytest.mark.smoke
def test_fused_add_layer_norm_kernel_returns_two_outputs() -> None:
    """The kernel declares exactly two outputs (``out_idx`` count)."""
    m, n, dtype = 8, 4096, torch.float16
    test = FusedAddLayerNormTest(m, n, dtype)
    inputs = test.gen_inputs()
    kernel = FusedAddLayerNormKernel(m, n, test.eps, dtype)
    outputs = kernel(*inputs)
    assert isinstance(outputs, list) and len(outputs) == 2
    assert all(out.shape == (m, n) for out in outputs)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
