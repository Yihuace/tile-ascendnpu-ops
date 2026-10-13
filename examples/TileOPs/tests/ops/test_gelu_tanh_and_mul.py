"""Correctness tests for GeluTanhAndMulFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_fused_gated.py`` -- the GeluTanhAndMul
subset only (the shared GPU file also covers silu_and_mul and gelu_and_mul,
which are not part of this port).  Covers smoke correctness, multi-dtype
coverage, dtype rejection, shape/dtype mismatch rejection, and the roofline
formula.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) instead of
  hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards removed
  (NPU availability is a runtime concern).
- T3/T4: the GPU fused-gated tests carry no ``tune`` parametrization / autotune
  try-except -- nothing to remove.
"""

from math import prod, sqrt

import pytest
import torch

from tileops.benchmark.msprof import _extract_op_init_args
from tileops.device import get_device_backend
from tileops.kernels.elementwise.gelu_tanh_and_mul import GeluTanhAndMulFwdKernel
from tileops.ops.elementwise.gelu_tanh_and_mul import GeluTanhAndMulFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    # Mirrors the GPU fused-gated verification: fp16 uses a looser exact bound,
    # the rest follow the default reference-tolerance table.
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 1.6e-2, 1.6e-2  # bfloat16


class GeluTanhAndMulTest(TestBase):
    """Test fixture for GeluTanhAndMulFwdOp -- one (M, 2 * N) randn tensor."""

    def __init__(self, M: int, N: int, dtype: torch.dtype):
        self.M = int(M)
        self.N = int(N)
        self.m = self.M
        self.n = self.N
        self.dtype = dtype
        self.x_shape = (self.M, 2 * self.N)
        self.shape = self.x_shape
        self.n_total = self.M * self.N

    def gen_inputs(self) -> tuple[torch.Tensor]:
        device = _device()
        x = torch.randn(self.M, 2 * self.N, device=device, dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # tanh-approximate GELU, evaluated in fp32 and cast back, matching the
        # GPU ``_gated(lambda g: F.gelu(g, approximate="tanh"))`` reference.
        #
        # Computed on CPU with an EXPLICIT tanh formula: this torch_npu build's
        # device ``F.gelu(approximate="tanh")`` is unreliable, so a device-side
        # ``F.gelu`` is not a faithful reference here (cf. the gelu_and_mul
        # port, which hit the same issue with the erf form).
        #
        #   gelu_tanh(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3)))
        half = x.shape[-1] // 2
        xc = x.detach().float().cpu()
        gate = xc[..., :half]
        value = xc[..., half:]
        inner = sqrt(2.0 / 3.141592653589793) * (gate + 0.044715 * gate**3)
        gelu = 0.5 * gate * (1.0 + torch.tanh(inner))
        return (gelu * value).to(x.dtype).to(x.device)


class GeluTanhAndMulFixture(FixtureBase):
    PARAMS = [
        (
            "M, N, dtype",
            [
                pytest.param(
                    16,
                    128,
                    torch.float32,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                ),
                pytest.param(16, 128, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(16, 128, torch.bfloat16, marks=[pytest.mark.smoke]),
                # Full: SwiGLU FFN intermediate (Llama-3.1-8B, hidden_dim=14336)
                pytest.param(2048, 14336, torch.float16, marks=[pytest.mark.full]),
                pytest.param(2048, 14336, torch.bfloat16, marks=[pytest.mark.full]),
                # Full: decode-time row (M=1)
                pytest.param(1, 14336, torch.bfloat16, marks=[pytest.mark.full]),
            ],
        ),
    ]


@GeluTanhAndMulFixture
def test_gelu_tanh_and_mul_op(M: int, N: int, dtype: torch.dtype) -> None:
    test = GeluTanhAndMulTest(M, N, dtype)
    op = GeluTanhAndMulFwdOp(M=M, N=N, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_gelu_tanh_and_mul_rejects_non_float_dtype() -> None:
    """GeluTanhAndMulFwdOp is restricted to the float subset of the manifest."""
    with pytest.raises(ValueError, match="does not support dtype"):
        GeluTanhAndMulFwdOp(M=16, N=16, dtype=torch.int32)


@pytest.mark.smoke
def test_gelu_tanh_and_mul_kernel_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        GeluTanhAndMulFwdKernel(M=16, N=16, dtype=torch.int32)


@pytest.mark.smoke
def test_gelu_tanh_and_mul_rejects_dtype_mismatch() -> None:
    """forward() must reject an input whose dtype disagrees with __init__."""
    M, N = 4, 8
    device = _device()
    op = GeluTanhAndMulFwdOp(M=M, N=N, dtype=torch.float32)
    x = torch.randn(M, 2 * N, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="x.dtype"):
        op(x)


@pytest.mark.smoke
def test_gelu_tanh_and_mul_rejects_shape_mismatch() -> None:
    """forward() must reject an input whose shape disagrees with __init__."""
    M, N = 4, 8
    device = _device()
    op = GeluTanhAndMulFwdOp(M=M, N=N, dtype=torch.float32)
    x = torch.randn(M, 2 * N + 2, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="x.shape"):
        op(x)


@pytest.mark.smoke
def test_gelu_tanh_and_mul_eval_roofline() -> None:
    """eval_roofline() matches the GPU manifest formula.

    flops = 10 * M * N; bytes = 3 * M * N * elem_bytes (read 2*M*N + write M*N).
    """
    M, N = 64, 128
    dtype = torch.float16
    op = GeluTanhAndMulFwdOp(M=M, N=N, dtype=dtype)
    n_total = M * N
    elem_bytes = dtype.itemsize
    flops, mem_bytes = op.eval_roofline()
    assert flops == 10 * n_total, f"flops {flops} != 10 * M * N = {10 * n_total}"
    assert mem_bytes == 3 * n_total * elem_bytes, (
        f"bytes {mem_bytes} != 3 * M * N * elem_bytes = {3 * n_total * elem_bytes}"
    )
    assert op.N_total == n_total
    assert prod(op.out_shape) == n_total

    # The msprof runner reconstructs Op instances in a subprocess from
    # constructor-name attributes.  Keep this contract covered explicitly.
    assert _extract_op_init_args(op) == {"M": M, "N": N, "dtype": dtype}


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
