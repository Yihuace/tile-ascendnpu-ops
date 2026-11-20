"""Correctness tests for GeluAndMulFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_fused_gated.py`` -- the GeluAndMul subset
only (the shared GPU file also covers silu_and_mul and gelu_tanh_and_mul, which
are not migrated).  Covers smoke correctness, multi-dtype coverage, dtype
rejection, shape/dtype mismatch rejection, and the roofline formula.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) instead of
  hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards removed
  (NPU availability is a runtime concern).
- T3/T4: the GPU fused-gated tests carry no ``tune`` parametrization / autotune
  try-except -- nothing to remove.
"""

from math import prod

import pytest
import torch

from tileops.benchmark.msprof import _extract_op_init_args
from tileops.device import get_device_backend
from tileops.kernels.elementwise.gelu_and_mul import GeluAndMulFwdKernel
from tileops.ops.elementwise.gelu_and_mul import GeluAndMulFwdOp
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


class GeluAndMulTest(TestBase):
    """Test fixture for GeluAndMulFwdOp -- one (M, 2 * N) randn tensor."""

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
        # Exact GELU (erf form), evaluated in fp32 and cast back, matching the
        # GPU ``_gated(F.gelu)`` reference.
        #
        # Computed on CPU with an EXPLICIT erf: this torch_npu build returns the
        # *tanh approximation* for ``F.gelu(approximate="none")`` on device
        # (deviates ~1.2e-3 vs exact, exceeding the fp32 test tolerance 1e-5),
        # so a device-side ``F.gelu`` is not a faithful erf-GELU reference here.
        half = x.shape[-1] // 2
        xc = x.detach().float().cpu()
        gate = xc[..., :half]
        value = xc[..., half:]
        gelu = 0.5 * gate * (1.0 + torch.erf(gate * (0.5**0.5)))
        return (gelu * value).to(x.dtype).to(x.device)


class GeluAndMulFixture(FixtureBase):
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


@GeluAndMulFixture
def test_gelu_and_mul_op(M: int, N: int, dtype: torch.dtype) -> None:
    test = GeluAndMulTest(M, N, dtype)
    op = GeluAndMulFwdOp(M=M, N=N, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gelu_and_mul_special_values(dtype: torch.dtype) -> None:
    """Frozen NaN/Inf semantics, exercised through the production wrapper.

    ``gelu(-inf) = -inf * 0 = NaN``; ``gelu(+inf) = +inf``; NaN propagates from
    either operand.  This is a **regression guard** for the Stage-4 optimization
    that dropped the explicit NaN recovery (``vcmp`` + ``vselect``) from the erf
    chain, on the assumption that the clamped polynomial + ``vclamp`` already
    propagate NaN.  That assumption is a compiler/IR-level behaviour and is
    therefore pinned to the toolchain -- see ``docs/cann9_adaptation.md`` section
    12.5 (T6) and ``perf_opt/opt_log.md`` section 1.  Without a case like this,
    a toolchain change that stops propagating NaN would leave the whole suite
    green while the results are silently wrong.

    Mirrors the kernel-level ``[Boundary]`` probe (frozen boundary semantics),
    but goes through ``GeluAndMulFwdOp`` so the shipped call path is covered.
    """
    M, N = 8, 4096
    x = torch.randn(M, 2 * N, dtype=torch.float32)
    x[:, N] = 1.0                # default value operand for the gate probes
    x[0, 0] = float("inf")       # gate +inf, value +1 -> +inf
    x[1, 0] = float("-inf")      # gate -inf -> -inf * 0 = NaN
    x[2, 0] = float("nan")       # gate NaN -> NaN
    x[3, 0] = 1.0                # gate 1, value +inf -> +inf
    x[3, N] = float("inf")
    x[4, 0] = 1.0                # gate 1, value -inf -> -inf
    x[4, N] = float("-inf")
    x[5, 0] = 1.0                # gate 1, value NaN -> NaN
    x[5, N] = float("nan")
    x[6, 0] = 0.0                # gate 0 -> signed zero
    x[7, 0] = 1e4                # large gate -> erf saturation to 1

    device = _device()
    xt = x.to(dtype).to(device)
    op = GeluAndMulFwdOp(M=M, N=N, dtype=dtype)
    with torch.no_grad():
        y = op(xt)
    assert y.dtype == dtype, f"output dtype {y.dtype} != {dtype}"

    y_cpu = y.float().cpu()
    golden = GeluAndMulTest(M, N, dtype).ref_program(xt).float().cpu()
    atol, rtol = _get_tolerances(dtype)
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol, equal_nan=True)

    # Explicit frozen-semantics probes.
    assert bool(torch.isnan(y_cpu[1, 0])), "gelu(-inf) must be NaN"
    assert bool(torch.isinf(y_cpu[0, 0]) and y_cpu[0, 0] > 0), "gelu(+inf) must be +inf"
    assert bool(torch.isnan(y_cpu[2, 0])), "gelu(NaN) must be NaN"
    assert bool(torch.isinf(y_cpu[4, 0]) and y_cpu[4, 0] < 0), "value -inf must propagate"
    assert bool(torch.isnan(y_cpu[5, 0])), "value NaN must propagate to NaN"


@pytest.mark.smoke
def test_gelu_and_mul_rejects_non_float_dtype() -> None:
    """GeluAndMulFwdOp is restricted to the float subset of the manifest."""
    with pytest.raises(ValueError, match="does not support dtype"):
        GeluAndMulFwdOp(M=16, N=16, dtype=torch.int32)


@pytest.mark.smoke
def test_gelu_and_mul_kernel_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        GeluAndMulFwdKernel(M=16, N=16, dtype=torch.int32)


@pytest.mark.smoke
def test_gelu_and_mul_rejects_dtype_mismatch() -> None:
    """forward() must reject an input whose dtype disagrees with __init__."""
    M, N = 4, 8
    device = _device()
    op = GeluAndMulFwdOp(M=M, N=N, dtype=torch.float32)
    x = torch.randn(M, 2 * N, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="x.dtype"):
        op(x)


@pytest.mark.smoke
def test_gelu_and_mul_rejects_shape_mismatch() -> None:
    """forward() must reject an input whose shape disagrees with __init__."""
    M, N = 4, 8
    device = _device()
    op = GeluAndMulFwdOp(M=M, N=N, dtype=torch.float32)
    x = torch.randn(M, 2 * N + 2, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="x.shape"):
        op(x)


@pytest.mark.smoke
def test_gelu_and_mul_eval_roofline() -> None:
    """eval_roofline() matches the GPU manifest formula.

    flops = 6 * M * N; bytes = 3 * M * N * elem_bytes (read 2*M*N + write M*N).
    """
    M, N = 64, 128
    dtype = torch.float16
    op = GeluAndMulFwdOp(M=M, N=N, dtype=dtype)
    n_total = M * N
    elem_bytes = dtype.itemsize
    flops, mem_bytes = op.eval_roofline()
    assert flops == 6 * n_total, f"flops {flops} != 6 * M * N = {6 * n_total}"
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
