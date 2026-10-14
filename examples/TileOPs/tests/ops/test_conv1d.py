"""Correctness tests for Conv1dFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_convolution.py`` -- the Conv1d subset
only (the shared GPU file also covers Conv2d / Conv3d, which are not migrated).

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) /
  ``Conv1dWorkload.gen_inputs`` instead of hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.cuda_only`` / ``torch.cuda`` availability guards removed
  (NPU availability is a runtime concern).
- T3: the ``tune`` parametrization dimension is removed.
- T4: no autotune try/except -- the GPU Conv1d tests have none to remove.

GPU-only tests that are not ported: the ``torch.compile``/dynamo compile
boundary tests (``tests.compile_contract``), the CUDAGraph inference-weight
test, and ``test_conv_past_the_grid_z_limit_is_refused_before_building`` (the
CUDA grid-z limit does not exist on NPU).
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.ops.convolution import Conv1dFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.convolution import (
    Conv1dFixture,
    Conv1dWorkload,
    conv1d_tolerance,
)


def _device() -> str:
    return get_device_backend().name


class Conv1dTest(Conv1dWorkload, TestBase):
    pass


@Conv1dFixture
def test_conv1d(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_size: int,
    stride: int,
    padding: int | str,
    dilation: int,
    groups: int,
    dtype: torch.dtype,
    use_bias: bool,
) -> None:
    test = Conv1dTest(
        n, c_in, l_in, c_out, kernel_size, stride, padding, dilation, groups, dtype, use_bias
    )
    op = Conv1dFwdOp(stride=stride, padding=padding, dilation=dilation, groups=groups)
    tol = conv1d_tolerance(dtype, padding=padding, kernel_shape=(kernel_size,))
    test.check(op, *test.gen_inputs(), atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
def test_conv1d_no_bias_matches_torch() -> None:
    op = Conv1dFwdOp(stride=2, padding=2)
    x = torch.randn(1, 32, 256, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, device=_device(), dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv1d(x.float(), weight.float(), bias=None, stride=2, padding=2).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, kernel_shape=(5,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
def test_conv1d_bias_matches_torch() -> None:
    op = Conv1dFwdOp(stride=2, padding=2)
    x = torch.randn(1, 32, 256, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.zeros(64, device=_device(), dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv1d(
        x.float(), weight.float(), bias=bias.float(), stride=2, padding=2
    ).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, kernel_shape=(5,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dilation, use_bias",
    [
        pytest.param(2, False, marks=pytest.mark.smoke, id="no-bias"),
        pytest.param(2, True, marks=pytest.mark.full, id="bias"),
    ],
)
def test_conv1d_dilation_matches_torch(dilation: int, use_bias: bool) -> None:
    n, c_in, l_in, c_out, kernel_size = 1, 32, 128, 64, 3
    stride, padding = 1, 2
    op = Conv1dFwdOp(stride=stride, padding=padding, dilation=dilation)
    x = torch.randn(n, c_in, l_in, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(c_out, c_in, kernel_size, device=_device(), dtype=torch.float16).contiguous()
    bias = (
        torch.randn(c_out, device=_device(), dtype=torch.float16).contiguous() if use_bias else None
    )
    out = op(x, weight, bias) if use_bias else op(x, weight)
    ref = F.conv1d(
        x.float(),
        weight.float(),
        bias=bias.float() if use_bias else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, kernel_shape=(kernel_size,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
@pytest.mark.parametrize(
    "use_bias",
    [pytest.param(False, id="no-bias"), pytest.param(True, id="bias")],
)
def test_conv1d_same_padding_even_kernel_matches_torch(use_bias: bool) -> None:
    n, c_in, l_in, c_out, kernel_size = 1, 16, 129, 32, 2
    op = Conv1dFwdOp(padding="same")
    x = torch.randn(n, c_in, l_in, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(c_out, c_in, kernel_size, device=_device(), dtype=torch.float16).contiguous()
    bias = (
        torch.randn(c_out, device=_device(), dtype=torch.float16).contiguous() if use_bias else None
    )
    out = op(x, weight, bias) if use_bias else op(x, weight)
    ref = F.conv1d(
        x.float(), weight.float(), bias=bias.float() if use_bias else None, padding="same"
    ).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, padding="same", kernel_shape=(kernel_size,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, expected_key",
    [
        pytest.param(3, 1, 1, 1, "conv1d_unit_stride", id="unit-stride"),
        pytest.param(3, 2, 1, 1, "conv1d", id="generic"),
        pytest.param(1, 1, 0, 1, "conv1d_pointwise", id="pointwise"),
    ],
)
def test_conv1d_dispatches_kernel(
    kernel_size: int, stride: int, padding: int, dilation: int, expected_key: str
) -> None:
    op = Conv1dFwdOp(stride=stride, padding=padding, dilation=dilation)
    x = torch.randn(1, 32, 256, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, kernel_size, device=_device(), dtype=torch.float16).contiguous()
    assert op._select_kernel_key(32, 32, 64, kernel_size, stride, padding, padding, dilation) == (
        expected_key
    )
    out = op(x, weight)
    ref = F.conv1d(
        x.float(), weight.float(), bias=None, stride=stride, padding=padding, dilation=dilation
    ).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, kernel_shape=(kernel_size,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
def test_conv1d_depthwise_matches_torch() -> None:
    """One input channel per group dispatches the depthwise-direct kernel."""
    groups = 32
    op = Conv1dFwdOp(padding=1, groups=groups)
    x = torch.randn(1, groups, 128, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(groups, 1, 3, device=_device(), dtype=torch.float16).contiguous()
    assert op._select_kernel_key(groups, 1, 1, 3, 1, 1, 1, 1) == "depthwise_conv1d"
    out = op(x, weight)
    ref = F.conv1d(x.float(), weight.float(), bias=None, padding=1, groups=groups).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, kernel_shape=(3,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
@pytest.mark.parametrize(
    "use_bias",
    [pytest.param(False, id="no-bias"), pytest.param(True, id="bias")],
)
def test_conv1d_grouped_matches_torch(use_bias: bool) -> None:
    groups = 2
    op = Conv1dFwdOp(padding=1, groups=groups)
    x = torch.randn(1, 32, 128, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 16, 3, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.zeros(64, device=_device(), dtype=torch.float16).contiguous() if use_bias else None
    assert op._select_kernel_key(32, 16, 32, 3, 1, 1, 1, 1) == "group_conv1d"
    out = op(x, weight, bias) if use_bias else op(x, weight)
    ref = F.conv1d(
        x.float(), weight.float(), bias=bias.float() if use_bias else None, padding=1, groups=groups
    ).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, kernel_shape=(3,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
def test_conv1d_rejects_unsupported_dtype() -> None:
    op = Conv1dFwdOp()
    x = torch.randn(1, 8, 32, device=_device(), dtype=torch.float32).contiguous()
    weight = torch.randn(16, 8, 3, device=_device(), dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match="does not support dtype"):
        op(x, weight)


@pytest.mark.smoke
def test_conv1d_eval_roofline() -> None:
    """eval_roofline() matches the inlined GPU ``conv_roofline``."""
    n, c_in, l_in, c_out, kernel_size = 1, 32, 128, 64, 3
    op = Conv1dFwdOp(padding=0)
    x = torch.randn(n, c_in, l_in, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(c_out, c_in, kernel_size, device=_device(), dtype=torch.float16).contiguous()
    op(x, weight)
    flops, nbytes = op.eval_roofline()
    out_l = l_in - kernel_size + 1
    # No padding: every (output, tap) pair lands in range, so taps == out_l * kernel.
    taps = out_l * kernel_size
    assert flops == 2 * n * c_out * c_in * taps
    # Every input position is read and the bias is absent: all tensors move once.
    total_elems = n * c_in * l_in + c_out * c_in * kernel_size + n * c_out * out_l
    assert nbytes == total_elems * x.element_size()


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
