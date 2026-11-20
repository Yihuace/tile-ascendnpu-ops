"""Correctness tests for FusedAddRMSNormFwdOp (NPU).

Ported from the GPU TileOPs test for the fused-add-RMSNorm family
(per the GPU manifest ``source.test``).  Uses the device backend instead of
hard-coded ``"cuda"`` (T1/T2).

The op has **two outputs** (``y`` and ``residual_out``), so the tests assert
against both explicitly rather than going through ``TestBase.check`` (which
expects a single tensor result).

The reference mirrors ``examples/norm/fused_add_rms_norm/fused_add_rmsnorm.py``
``_reference``: ``s = x + residual``; ``rstd = rsqrt(mean(s^2) + eps)``;
``y = s * rstd * weight``; ``residual_out = s``.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.norm.fused_add_rms_norm import FusedAddRMSNormFwdOp
from tileops.workloads.norm import FusedAddRMSNormWorkload
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


def _ref(x, residual, weight, eps=1e-6):
    """CPU reference returning ``(y, residual_out)``."""
    s_f32 = x.float() + residual.float()
    rstd = torch.rsqrt(s_f32.square().mean(dim=-1, keepdim=True) + eps)
    return (s_f32 * rstd * weight.float()).to(x.dtype), s_f32.to(x.dtype)


def _check(m, n, dtype, eps=1e-6):
    device = _device()
    x = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    op = FusedAddRMSNormFwdOp(dtype=dtype, eps=eps)
    y, r_out = op(x, residual, weight)
    ref_y, ref_r = _ref(x.cpu(), residual.cpu(), weight.cpu(), eps)
    atol, rtol = _get_tolerances(dtype)
    assert y.shape == (m, n), f"y shape {y.shape} != ({m}, {n})"
    assert r_out.shape == (m, n), f"residual_out shape {r_out.shape} != ({m}, {n})"
    torch.testing.assert_close(y.cpu(), ref_y, atol=atol, rtol=rtol)
    torch.testing.assert_close(r_out.cpu(), ref_r, atol=atol, rtol=rtol)


class FusedAddRMSNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(
                    64, 512, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
                ),
                pytest.param(64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(64, 512, torch.bfloat16, marks=pytest.mark.smoke),
                # Full: non-divisible N exercises the alignment tail
                pytest.param(8, 300, torch.float32, marks=pytest.mark.full),
                pytest.param(8, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(8, 300, torch.bfloat16, marks=pytest.mark.full),
                # Full: realistic prefill / decode shapes
                pytest.param(2048, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(1, 4096, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@FusedAddRMSNormFixture
def test_fused_add_rms_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    _check(m, n, dtype)


@pytest.mark.smoke
def test_fused_add_rms_norm_eval_roofline() -> None:
    M, N = 64, 256
    dtype = torch.float16
    device = _device()
    op = FusedAddRMSNormFwdOp(dtype=dtype)
    x = torch.randn(M, N, dtype=dtype, device=device)
    residual = torch.randn(M, N, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    op(x, residual, weight)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 5 * M * N, f"flops {flops} != 5 * M * N = {5 * M * N}"
    assert mem_bytes == (4 * M * N + N) * elem_bytes, (
        f"bytes {mem_bytes} != (4*M*N + N) * elem_bytes = "
        f"{(4 * M * N + N) * elem_bytes}"
    )


@pytest.mark.smoke
def test_fused_add_rms_norm_residual_out_is_the_sum() -> None:
    """``residual_out`` must be exactly ``x + residual`` (the fused-add half).

    This is the piece callers chain into the next layer; if the kernel wrote
    the normalized value here instead, the residual stream would be corrupted
    silently.
    """
    m, n = 8, 512
    dtype = torch.float32
    device = _device()
    x = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    op = FusedAddRMSNormFwdOp(dtype=dtype)
    _, r_out = op(x, residual, weight)
    torch.testing.assert_close(
        r_out.cpu(), (x + residual).cpu(), atol=1e-6, rtol=1e-6,
        msg="residual_out must equal x + residual",
    )


@pytest.mark.smoke
def test_fused_add_rms_norm_unit_weight_matches_manual_rms() -> None:
    """With ``weight == 1`` the output must be the plain RMS-normalized sum."""
    m, n = 8, 512
    dtype = torch.float32
    device = _device()
    x = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.ones(n, dtype=dtype, device=device)

    op = FusedAddRMSNormFwdOp(dtype=dtype)
    y, _ = op(x, residual, weight)
    s = (x + residual).float()
    rstd = torch.rsqrt(s.square().mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(
        y.cpu(), (s * rstd).to(dtype).cpu(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.smoke
def test_fused_add_rms_norm_3d() -> None:
    """3-D input (batch, seq, hidden) must flatten/unflatten transparently."""
    b, s, n = 2, 512, 1024
    dtype = torch.float16
    device = _device()
    x = torch.randn(b, s, n, dtype=dtype, device=device)
    residual = torch.randn(b, s, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    op = FusedAddRMSNormFwdOp(dtype=dtype)
    y, r_out = op(x, residual, weight)
    ref_y, ref_r = _ref(x.cpu(), residual.cpu(), weight.cpu())
    assert y.shape == (b, s, n) and r_out.shape == (b, s, n)
    atol, rtol = _get_tolerances(dtype)
    torch.testing.assert_close(y.cpu(), ref_y, atol=atol, rtol=rtol)
    torch.testing.assert_close(r_out.cpu(), ref_r, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
