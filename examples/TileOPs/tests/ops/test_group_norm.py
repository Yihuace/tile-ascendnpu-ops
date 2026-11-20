"""Correctness tests for GroupNormFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_group_norm.py`` -- the **affine
primary** only.  The no-affine variant coverage in the shared GPU file
(``GroupNormNoAffineFixture`` / ``test_group_norm_no_affine_*``, which
exercises ``GroupNormNoAffineFwdOp``) is not migrated and is dropped here
together with the no-affine imports.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``)
  instead of hard-coded ``device="cuda"``.
- T2: the GPU file has no ``@pytest.mark.skipif(not torch.cuda.is_available(),
  ...)`` guards; the multi-device tests guard on the NPU device count via
  ``get_device_backend().device_count()``.
- T3: the ``tune`` parametrization column is removed (K8) -- the GPU fixture
  carried an always-``False`` ``tune`` column and the op no longer accepts it.
- T4: no autotune try/except in the GPU file -- nothing to remove.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.ops.norm.group_norm import GroupNormFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.norm import GroupNormWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


class GroupNormTest(GroupNormWorkload, TestBase):
    def ref_program(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return F.group_norm(
            x.float(),
            self.g,
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)


class GroupNormFixture(FixtureBase):
    PARAMS = [
        (
            "n, c, spatial, g, dtype",
            [
                # Small CI-friendly shapes -- fp32 (packaging) / fp16 / bf16
                pytest.param(
                    2,
                    32,
                    (8, 8),
                    8,
                    torch.float32,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                ),
                pytest.param(2, 32, (8, 8), 8, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 32, (8, 8), 8, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4, 16, (4, 4), 4, torch.float32, marks=pytest.mark.full),
                pytest.param(4, 16, (4, 4), 4, torch.float16, marks=pytest.mark.full),
                pytest.param(4, 16, (4, 4), 4, torch.bfloat16, marks=pytest.mark.full),
                # Different group counts
                pytest.param(2, 32, (4, 4), 1, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 32, (4, 4), 32, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 32, (4, 4), 16, torch.float16, marks=pytest.mark.full),
                # 1D spatial
                pytest.param(2, 32, (16,), 8, torch.float16, marks=pytest.mark.full),
                # 3D spatial
                pytest.param(2, 16, (4, 4, 4), 4, torch.float16, marks=pytest.mark.full),
                # Non-power-of-two channels per group
                pytest.param(2, 30, (4, 4), 5, torch.float16, marks=pytest.mark.full),
                # Non-aligned spatial: exercises partial-tile path
                pytest.param(2, 32, (7, 7), 8, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 32, (7, 7), 8, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@GroupNormFixture
def test_group_norm_op(n: int, c: int, spatial: tuple, g: int, dtype: torch.dtype) -> None:
    test = GroupNormTest(n, c, spatial, g, dtype)
    op = GroupNormFwdOp(num_groups=g)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class GroupNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "n, c, spatial, g, dtype",
            [
                pytest.param(2, 32, (8, 8), 8, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 32, (8, 8), 8, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@GroupNormNonContigFixture
def test_group_norm_non_contiguous(
    n: int, c: int, spatial: tuple, g: int, dtype: torch.dtype
) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    # T1: device via the device backend (was ``device="cuda"``).
    device = _device()
    shape = (n, c * 2, *spatial)
    x_full = torch.randn(shape, dtype=dtype, device=device)
    x = x_full[:, :c]  # non-contiguous slice
    weight = torch.randn(c, dtype=dtype, device=device)
    bias = torch.randn(c, dtype=dtype, device=device)

    op = GroupNormFwdOp(num_groups=g)

    y_ref = F.group_norm(
        x.contiguous().float(),
        g,
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"
    )


@pytest.mark.smoke
def test_group_norm_rejects_none_weight_or_bias() -> None:
    """Affine op rejects ``weight=None`` / ``bias=None``; affine-free path lives on NoAffine."""
    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOp(num_groups=g)
    x = torch.randn((n, c, *spatial), dtype=dtype, device=_device())
    weight = torch.randn((c,), dtype=dtype, device=_device())
    bias = torch.randn((c,), dtype=dtype, device=_device())

    with pytest.raises((ValueError, TypeError)):
        op(x, None, bias)
    with pytest.raises((ValueError, TypeError)):
        op(x, weight, None)
    with pytest.raises((ValueError, TypeError)):
        op(x, None, None)


@pytest.mark.smoke
def test_group_norm_forward_required_signature() -> None:
    """`forward` declares weight and bias as required (no Optional, no default)."""
    import inspect

    sig = inspect.signature(GroupNormFwdOp.forward)
    weight_param = sig.parameters["weight"]
    bias_param = sig.parameters["bias"]
    assert weight_param.default is inspect.Parameter.empty
    assert bias_param.default is inspect.Parameter.empty


@pytest.mark.smoke
def test_group_norm_lazily_specializes_per_device() -> None:
    """A single op can lazily build specializations for different devices."""
    # T2: NPU device-count guard (was ``torch.cuda.device_count()``).
    backend = get_device_backend()
    if backend.device_count() < 2:
        pytest.skip("multi-device test requires >= 2 devices")

    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOp(num_groups=g)
    x_other = torch.randn(
        (n, c, *spatial),
        dtype=dtype,
        device=torch.device(_device(), 1),
    )
    weight_other = torch.randn(
        (c,),
        dtype=dtype,
        device=torch.device(_device(), 1),
    )
    bias_other = torch.randn(
        (c,),
        dtype=dtype,
        device=torch.device(_device(), 1),
    )
    y = op(x_other, weight_other, bias_other)
    assert y.device == x_other.device
    assert len(op._kernel_cache) == 1


@pytest.mark.smoke
def test_group_norm_lazy_cache_reuse_and_respecialization() -> None:
    """One op instance reuses identical specs and caches changed specs."""
    device = _device()
    op = GroupNormFwdOp(num_groups=4)

    def run_case(n: int, c: int, spatial: tuple[int, ...], dtype: torch.dtype) -> None:
        x = torch.randn((n, c, *spatial), dtype=dtype, device=device)
        weight = torch.randn((c,), dtype=dtype, device=device)
        bias = torch.randn((c,), dtype=dtype, device=device)

        y = op(x, weight, bias)
        y_ref = F.group_norm(
            x.float(),
            4,
            weight=weight.float(),
            bias=bias.float(),
            eps=1e-5,
        ).to(dtype)
        atol, rtol = _get_tolerances(dtype)
        assert torch.allclose(y, y_ref, atol=atol, rtol=rtol)

    run_case(2, 16, (4, 4), torch.float16)
    assert len(op._kernel_cache) == 1
    assert op.eval_roofline() == (
        5 * 2 * 16 * 16,
        (2 * 2 * 16 * 16 + 2 * 16) * torch.float16.itemsize,
    )

    run_case(2, 16, (4, 4), torch.float16)
    assert len(op._kernel_cache) == 1

    run_case(3, 24, (2, 8), torch.bfloat16)
    assert len(op._kernel_cache) == 2
    assert op.eval_roofline() == (
        5 * 3 * 24 * 16,
        (2 * 3 * 24 * 16 + 2 * 24) * torch.bfloat16.itemsize,
    )


@pytest.mark.smoke
def test_group_norm_rejects_affine_device_mismatch() -> None:
    """Forward must raise ValueError when weight/bias live on a different device than x.

    Without an explicit check the kernel call would either dispatch on
    cross-device tensors (slow / wrong) or surface as an opaque device
    error; surface a clean ValueError instead.
    """
    # T2: NPU device-count guard (was ``torch.cuda.device_count()``).
    backend = get_device_backend()
    if backend.device_count() < 2:
        pytest.skip("affine-device-mismatch test requires >= 2 devices")

    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOp(num_groups=g)
    x = torch.randn((n, c, *spatial), dtype=dtype, device=torch.device(_device(), 0))
    weight_other = torch.randn((c,), dtype=dtype, device=torch.device(_device(), 1))
    bias_other = torch.randn((c,), dtype=dtype, device=torch.device(_device(), 1))

    bias_same = torch.randn((c,), dtype=dtype, device=torch.device(_device(), 0))

    weight_same = torch.randn(
        (c,),
        dtype=dtype,
        device=torch.device(_device(), 0),
    )
    with pytest.raises(ValueError, match="weight on"):
        op(x, weight_other, bias_same)
    with pytest.raises(ValueError, match="bias on"):
        op(x, weight_same, bias_other)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
