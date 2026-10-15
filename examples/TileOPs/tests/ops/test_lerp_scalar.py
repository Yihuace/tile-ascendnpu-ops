"""Correctness tests for LerpFwdOp (NPU).

LerpFwdOp is the scalar-weight overload of ``torch.lerp``:
``out = input + weight * (end - input)`` with ``weight`` a construction-time
float constant (default 0.5).  This file mirrors ``test_lerp_tensor.py``
restricted to the scalar-weight form: the independent golden is
``torch.lerp(input, end, weight)``.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``)
  instead of hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)``
  guards removed (NPU availability is a runtime concern).
- T3/T4: no ``tune`` parametrization / autotune try-except to remove.
"""

import pytest
import torch

from tileops.benchmark.msprof import _extract_op_init_args
from tileops.device import get_device_backend
from tileops.ops.elementwise.lerp_scalar import LerpFwdOp
from tileops.testing import record_check_result

_LERP_SCALAR_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _device() -> str:
    return get_device_backend().name


def _lerp_tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float16:
        return {"atol": 1e-3, "rtol": 1e-3}
    if dtype == torch.bfloat16:
        return {"atol": 1e-2, "rtol": 1e-2}
    return {"atol": 1e-6, "rtol": 1e-6}


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]),
        pytest.param(torch.float16, marks=[pytest.mark.smoke]),
        pytest.param(torch.bfloat16, marks=[pytest.mark.smoke]),
    ],
)
def test_lerp_scalar_default_weight_same_shape(dtype: torch.dtype) -> None:
    """LerpFwdOp matches torch.lerp on same-shape inputs with the default weight."""
    shape = (4, 8)
    device = _device()
    weight = 0.5
    a = torch.randn(shape, device=device, dtype=dtype)
    b = torch.randn(shape, device=device, dtype=dtype)
    op = LerpFwdOp(input=shape, end=shape, dtype=dtype)
    out = op(a, b)
    ref = torch.lerp(a, b, weight)
    record_check_result(op, out, ref)
    torch.testing.assert_close(out, ref, **_lerp_tol(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _LERP_SCALAR_DTYPES)
@pytest.mark.parametrize("weight", [0.0, 0.3, 0.5, 0.7, 1.0, -1.5, 2.0])
def test_lerp_scalar_custom_weights(weight: float, dtype: torch.dtype) -> None:
    """Construction-time weights are baked into the kernel (one per value)."""
    shape = (64, 128)
    device = _device()
    a = torch.randn(shape, device=device, dtype=dtype)
    b = torch.randn(shape, device=device, dtype=dtype)
    op = LerpFwdOp(input=shape, end=shape, dtype=dtype, weight=weight)
    out = op(a, b)
    ref = torch.lerp(a, b, weight)
    record_check_result(op, out, ref)
    torch.testing.assert_close(out, ref, **_lerp_tol(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ((3, 1), (1, 4)),
        ((2, 512, 768), (1, 1, 768)),
        ((2, 512, 768), (2, 512, 1)),
        ((16, 256, 56, 56), (256, 1, 1)),
        ((1024, 4096), (1, 4096)),
    ],
)
def test_lerp_scalar_broadcast(a_shape: tuple, b_shape: tuple) -> None:
    """LerpFwdOp supports the manifest broadcast rule (output = broadcast(A, B))."""
    dtype = torch.float32
    device = _device()
    a = torch.randn(a_shape, device=device, dtype=dtype)
    b = torch.randn(b_shape, device=device, dtype=dtype)
    op = LerpFwdOp(input=a_shape, end=b_shape, dtype=dtype, weight=0.3)
    out = op(a, b)
    ref = torch.lerp(a, b, 0.3)
    record_check_result(op, out, ref)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
    assert tuple(out.shape) == tuple(torch.broadcast_shapes(a_shape, b_shape))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2, torch.int32],
)
def test_lerp_scalar_rejects_unsupported_dtype(bad_dtype: torch.dtype) -> None:
    """LerpFwdOp must reject non-float / fp8 dtypes (manifest declares no fp8)."""
    shape = (4, 8)
    with pytest.raises((ValueError, TypeError)):
        LerpFwdOp(input=shape, end=shape, dtype=bad_dtype)


@pytest.mark.smoke
def test_lerp_scalar_dtype_mismatch_rejected() -> None:
    """forward() must reject inputs whose dtype disagrees with __init__."""
    shape = (4, 8)
    device = _device()
    op = LerpFwdOp(input=shape, end=shape, dtype=torch.float32)
    a = torch.randn(shape, device=device, dtype=torch.float32)
    b_bad = torch.randn(shape, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="end.dtype"):
        op(a, b_bad)


@pytest.mark.smoke
def test_lerp_scalar_kernel_rejects_non_float_dtype() -> None:
    from tileops.kernels.elementwise.lerp_scalar import LerpScalarFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        LerpScalarFwdKernel(N_total=16, dtype=torch.int32, weight=0.5)


@pytest.mark.smoke
def test_lerp_scalar_nan_semantics() -> None:
    """NaN operands propagate NaN; the result matches torch.lerp elementwise."""
    shape = (4096,)
    dtype = torch.float32
    device = _device()
    a = torch.randn(shape, device=device, dtype=dtype)
    b = torch.randn(shape, device=device, dtype=dtype)
    a[0::8] = float("nan")
    b[1::8] = float("nan")
    op = LerpFwdOp(input=shape, end=shape, dtype=dtype, weight=0.5)
    out = op(a, b).cpu()
    ref = torch.lerp(a, b, 0.5).cpu()
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6, equal_nan=True)
    assert torch.equal(torch.isnan(out), torch.isnan(ref)), "NaN mask mismatch"


@pytest.mark.smoke
def test_lerp_scalar_eval_roofline() -> None:
    """eval_roofline() matches the manifest formula.

    flops = 3 * N_total (sub + scalar mul + add per element);
    bytes = 4 * N_total * elem_bytes (2 reads + 1 write, broadcast read).
    """
    input_shape, end_shape = (64, 128), (1, 128)
    dtype = torch.float16
    op = LerpFwdOp(input=input_shape, end=end_shape, dtype=dtype, weight=0.3)
    n_total = 64 * 128
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 3 * n_total, f"flops {flops} != 3 * N = {3 * n_total}"
    assert mem_bytes == 4 * n_total * elem_bytes, (
        f"bytes {mem_bytes} != 4 * N * elem_bytes = {4 * n_total * elem_bytes}"
    )

    # The msprof runner reconstructs Op instances in a subprocess from
    # constructor-name attributes.  Keep this contract covered explicitly.
    assert _extract_op_init_args(op) == {
        "input": input_shape,
        "end": end_shape,
        "dtype": dtype,
        "weight": 0.3,
    }


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
