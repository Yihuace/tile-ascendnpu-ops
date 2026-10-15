"""Correctness tests for Conv1dBiasFwdOp (NPU).

``Conv1dBiasFwdOp`` is the bias-required variant of ``Conv1dFwdOp``
(``torch.nn.functional.conv1d`` with a mandatory per-output-channel bias), so
this file mirrors ``test_conv1d.py`` restricted to the always-bias contract.
The independent golden is ``torch.nn.functional.conv1d(..., bias=bias)``.

NPU adaptations (T1-T4, same as ``test_conv1d.py``):

- T1: manual tensor creation uses the device backend (``_device()``).
- T2: ``@pytest.mark.cuda_only`` / ``torch.cuda`` availability guards removed.
- T3: the ``tune`` parametrization dimension is removed.
- T4: no autotune try/except.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.ops.convolution import Conv1dBiasFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.convolution import (
    Conv1dFixture,
    Conv1dWorkload,
    conv1d_tolerance,
)


def _device() -> str:
    return get_device_backend().name


class Conv1dBiasTest(Conv1dWorkload, TestBase):
    """Conv1d workload that always produces a non-trivial (randn) bias."""

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, weight, _ = super().gen_inputs()
        bias = torch.randn(self.c_out, device=_device(), dtype=self.dtype).contiguous()
        return x, weight, bias


@Conv1dFixture
def test_conv1d_bias(
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
    # The shared Conv1d fixture carries a ``use_bias`` flag; this op always
    # takes a bias, so every case is forced to ``use_bias=True`` (the fixture's
    # flag is accepted only for parametrization parity).
    test = Conv1dBiasTest(
        n, c_in, l_in, c_out, kernel_size, stride, padding, dilation, groups, dtype, True
    )
    op = Conv1dBiasFwdOp(stride=stride, padding=padding, dilation=dilation, groups=groups)
    tol = conv1d_tolerance(dtype, padding=padding, kernel_shape=(kernel_size,))
    test.check(op, *test.gen_inputs(), atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
def test_conv1d_bias_matches_torch() -> None:
    op = Conv1dBiasFwdOp(stride=2, padding=2)
    x = torch.randn(1, 32, 256, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(64, device=_device(), dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv1d(
        x.float(), weight.float(), bias=bias.float(), stride=2, padding=2
    ).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, kernel_shape=(5,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
def test_conv1d_bias_required() -> None:
    """The bias argument is mandatory: omitted or ``None`` is rejected."""
    op = Conv1dBiasFwdOp()
    x = torch.randn(1, 8, 32, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, device=_device(), dtype=torch.float16).contiguous()
    with pytest.raises(TypeError):
        op(x, weight)
    with pytest.raises(ValueError, match="requires a bias"):
        op(x, weight, None)


@pytest.mark.smoke
def test_conv1d_bias_rejects_bad_shape() -> None:
    op = Conv1dBiasFwdOp()
    x = torch.randn(1, 8, 32, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(15, device=_device(), dtype=torch.float16).contiguous()
    with pytest.raises(ValueError, match=r"bias must have shape \[C_out\]"):
        op(x, weight, bias)


@pytest.mark.smoke
def test_conv1d_bias_rejects_bad_dtype() -> None:
    op = Conv1dBiasFwdOp()
    x = torch.randn(1, 8, 32, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(16, device=_device(), dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match="bias.dtype"):
        op(x, weight, bias)


@pytest.mark.smoke
def test_conv1d_bias_rejects_cpu_bias() -> None:
    op = Conv1dBiasFwdOp()
    x = torch.randn(1, 8, 32, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(16, dtype=torch.float16).contiguous()
    with pytest.raises(ValueError, match="bias must be a"):
        op(x, weight, bias)


@pytest.mark.smoke
def test_conv1d_bias_dilation_matches_torch() -> None:
    n, c_in, l_in, c_out, kernel_size = 1, 32, 128, 64, 3
    stride, padding, dilation = 1, 2, 2
    op = Conv1dBiasFwdOp(stride=stride, padding=padding, dilation=dilation)
    x = torch.randn(n, c_in, l_in, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(c_out, c_in, kernel_size, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(c_out, device=_device(), dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv1d(
        x.float(),
        weight.float(),
        bias=bias.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, kernel_shape=(kernel_size,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
def test_conv1d_bias_same_padding_even_kernel_matches_torch() -> None:
    n, c_in, l_in, c_out, kernel_size = 1, 16, 129, 32, 2
    op = Conv1dBiasFwdOp(padding="same")
    x = torch.randn(n, c_in, l_in, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(c_out, c_in, kernel_size, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(c_out, device=_device(), dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv1d(x.float(), weight.float(), bias=bias.float(), padding="same").to(x.dtype)
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
def test_conv1d_bias_dispatches_kernel(
    kernel_size: int, stride: int, padding: int, dilation: int, expected_key: str
) -> None:
    op = Conv1dBiasFwdOp(stride=stride, padding=padding, dilation=dilation)
    x = torch.randn(1, 32, 256, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, kernel_size, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(64, device=_device(), dtype=torch.float16).contiguous()
    assert op._select_kernel_key(32, 32, 64, kernel_size, stride, padding, padding, dilation) == (
        expected_key
    )
    out = op(x, weight, bias)
    ref = F.conv1d(
        x.float(), weight.float(), bias=bias.float(), stride=stride, padding=padding, dilation=dilation
    ).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, kernel_shape=(kernel_size,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])
    assert op.has_bias is True


@pytest.mark.smoke
def test_conv1d_bias_depthwise_matches_torch() -> None:
    """One input channel per group dispatches the depthwise-direct kernel."""
    groups = 32
    op = Conv1dBiasFwdOp(padding=1, groups=groups)
    x = torch.randn(1, groups, 128, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(groups, 1, 3, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(groups, device=_device(), dtype=torch.float16).contiguous()
    assert op._select_kernel_key(groups, 1, 1, 3, 1, 1, 1, 1) == "depthwise_conv1d"
    out = op(x, weight, bias)
    ref = F.conv1d(
        x.float(), weight.float(), bias=bias.float(), padding=1, groups=groups
    ).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, kernel_shape=(3,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
def test_conv1d_bias_grouped_matches_torch() -> None:
    groups = 2
    op = Conv1dBiasFwdOp(padding=1, groups=groups)
    x = torch.randn(1, 32, 128, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 16, 3, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(64, device=_device(), dtype=torch.float16).contiguous()
    assert op._select_kernel_key(32, 16, 32, 3, 1, 1, 1, 1) == "group_conv1d"
    out = op(x, weight, bias)
    ref = F.conv1d(
        x.float(), weight.float(), bias=bias.float(), padding=1, groups=groups
    ).to(x.dtype)
    tol = conv1d_tolerance(out.dtype, kernel_shape=(3,))
    torch.testing.assert_close(out, ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.smoke
def test_conv1d_bias_rejects_unsupported_dtype() -> None:
    op = Conv1dBiasFwdOp()
    x = torch.randn(1, 8, 32, device=_device(), dtype=torch.float32).contiguous()
    weight = torch.randn(16, 8, 3, device=_device(), dtype=torch.float32).contiguous()
    bias = torch.randn(16, device=_device(), dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match="does not support dtype"):
        op(x, weight, bias)


@pytest.mark.smoke
def test_conv1d_bias_eval_roofline() -> None:
    """eval_roofline() matches Conv1dFwdOp's inlined GPU ``conv_roofline`` + bias."""
    n, c_in, l_in, c_out, kernel_size = 1, 32, 128, 64, 3
    op = Conv1dBiasFwdOp(padding=0)
    x = torch.randn(n, c_in, l_in, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(c_out, c_in, kernel_size, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(c_out, device=_device(), dtype=torch.float16).contiguous()
    op(x, weight, bias)
    flops, nbytes = op.eval_roofline()
    out_l = l_in - kernel_size + 1
    taps = out_l * kernel_size
    # A multiply-add per in-range tap, plus the per-output bias add.
    assert flops == 2 * n * c_out * c_in * taps + n * c_out * out_l
    # No padding: every input position is read; the bias tensor is also moved.
    total_elems = n * c_in * l_in + c_out * c_in * kernel_size + n * c_out * out_l + c_out
    assert nbytes == total_elems * x.element_size()


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
