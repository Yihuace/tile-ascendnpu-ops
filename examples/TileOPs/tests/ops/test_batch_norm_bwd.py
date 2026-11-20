"""Correctness tests for BatchNormBwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_batch_norm.py`` -- the ``BatchNormBwd``
subset only (the shared GPU file also covers ``BatchNormFwdOp`` and the
CUDA-only compile-boundary / validation tests, which are not part of this op).

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) /
  ``BatchNormBwdWorkload.gen_inputs`` instead of hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.cuda_only`` / ``torch.cuda`` availability guards removed.
- T3: the ``tune`` parametrization dimension is removed.
- T4: no autotune try/except to remove.

The reference ``ref_program`` is the independent pure-PyTorch
``batch_norm_backward`` (upcast to fp32, rounded back), device-independent and
verifiable on CPU. ``grad_x`` is compared elementwise; ``grad_weight`` /
``grad_bias`` reduce ``N * S`` values, so they use the wider accumulated bound.
"""

import math
from functools import partial

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.norm.batch_norm_bwd import BatchNormBwdOp
from tileops.testing.test_base import TestBase, allclose_compare
from tileops.workloads.norm import BatchNormBwdWorkload, batch_norm_bwd_tolerance
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


class BatchNormBwdTest(BatchNormBwdWorkload, TestBase):
    """Test fixture for BatchNormBwdOp (inherits ``gen_inputs`` + ``ref_program``)."""


def _comparators(dtype: torch.dtype) -> list:
    tol = batch_norm_bwd_tolerance(dtype)
    return [
        partial(allclose_compare, atol=tol["atol"], rtol=tol["rtol"]),
        partial(allclose_compare, atol=tol["sum_atol"], rtol=tol["rtol"]),
        partial(allclose_compare, atol=tol["sum_atol"], rtol=tol["rtol"]),
    ]


class BatchNormBwdFixture(FixtureBase):
    """(n, c, spatial, dtype)"""

    PARAMS = [
        (
            "n, c, spatial, dtype",
            [
                # Smoke: smallest robust shape, one per dtype (fp32 also packaging).
                pytest.param(
                    32, 64, (), torch.float32,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                ),
                pytest.param(32, 64, (), torch.float16, marks=pytest.mark.smoke),
                pytest.param(32, 64, (), torch.bfloat16, marks=pytest.mark.smoke),
                # Full: ResNet-style spatial shapes.
                pytest.param(8, 64, (32, 32), torch.float16, marks=pytest.mark.full),
                pytest.param(4, 128, (32, 32), torch.bfloat16, marks=pytest.mark.full),
                # Non-aligned spatial: H*W=900, exercises the partial-tile path.
                pytest.param(8, 64, (30, 30), torch.float16, marks=pytest.mark.full),
                pytest.param(4, 256, (28, 28), torch.float32, marks=pytest.mark.full),
                # A register-held channel taking several steps per block.
                pytest.param(4, 64, (64, 64), torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@BatchNormBwdFixture
def test_batch_norm_bwd(n: int, c: int, spatial: tuple, dtype: torch.dtype) -> None:
    s = math.prod(spatial) if spatial else 1
    test = BatchNormBwdTest(n, c, s, dtype)
    grad_out, x, weight, mean, rstd = test.gen_inputs()
    if spatial:
        grad_out = grad_out.view(n, c, *spatial)
        x = x.view(n, c, *spatial)
    op = BatchNormBwdOp()
    test.check(
        op, grad_out, x, weight, mean, rstd, compare=_comparators(dtype)
    )


@pytest.mark.smoke
def test_batch_norm_bwd_matches_torch() -> None:
    """A channel length no tile divides still matches the analytical gradient."""
    n, c, h, w = 3, 5, 30, 31
    x = torch.randn(n, c, h, w, dtype=torch.float16, device=_device())
    grad_out = torch.randn_like(x)
    weight = torch.randn(c, dtype=torch.float32, device=_device())
    var, mean = torch.var_mean(x.float(), dim=[0, 2, 3], correction=0)
    rstd = torch.rsqrt(var + 1e-5)

    test = BatchNormBwdTest(n, c, h * w, x.dtype)
    inputs = (grad_out, x, weight, mean, rstd)
    test.check(BatchNormBwdOp(), *inputs, compare=_comparators(x.dtype))


@pytest.mark.smoke
def test_batch_norm_bwd_eval_roofline() -> None:
    n, c, s, dtype = 8, 64, 1024, torch.float16
    test = BatchNormBwdTest(n, c, s, dtype)
    grad_out, x, weight, mean, rstd = test.gen_inputs()
    op = BatchNormBwdOp()
    op(grad_out, x, weight, mean, rstd)
    flops, nbytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 9 * n * c * s
    assert nbytes == (3 * n * c * s + 5 * c) * elem_bytes


@pytest.mark.smoke
def test_batch_norm_bwd_rejects_bad_dtype() -> None:
    op = BatchNormBwdOp()
    n, c, s = 4, 8, 16
    grad_out = torch.randn(n, c, s, dtype=torch.float64, device=_device())
    x = torch.randn(n, c, s, dtype=torch.float64, device=_device())
    weight = torch.randn(c, dtype=torch.float32, device=_device())
    mean = torch.randn(c, dtype=torch.float32, device=_device())
    rstd = torch.randn(c, dtype=torch.float32, device=_device())
    with pytest.raises(ValueError, match="does not support dtype"):
        op(grad_out, x, weight, mean, rstd)


@pytest.mark.smoke
def test_batch_norm_bwd_rejects_bad_shape() -> None:
    op = BatchNormBwdOp()
    n, c, s = 4, 8, 16
    grad_out = torch.randn(n, c, s, dtype=torch.float16, device=_device())
    x = torch.randn(n, c, s * 2, dtype=torch.float16, device=_device())
    weight = torch.randn(c, dtype=torch.float32, device=_device())
    mean = torch.randn(c, dtype=torch.float32, device=_device())
    rstd = torch.randn(c, dtype=torch.float32, device=_device())
    with pytest.raises(ValueError, match="Expected x shape"):
        op(grad_out, x, weight, mean, rstd)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
