"""Correctness tests for Conv3dBiasFwdOp (NPU).

``Conv3dBiasFwdOp`` is the bias-required 3-D convolution
(``torch.nn.functional.conv3d`` with a mandatory per-output-channel bias).  The
independent golden is the pure-torch CPU reference ``F.conv3d(..., bias=bias)``
(via ``Conv3dWorkload.ref_program``), which is independent of the NPU
Cube/persistent/polyphase algorithm.

Coverage: stride 1 and stride 2 (polyphase W-axis), padding, f16/bfloat16,
the three required manifest workloads (r3d-stem-bias / video-down-bias /
unet-encoder-bias), a smoke shape and non-divisible dims.

NPU adaptations (T1-T4, same as ``test_conv1d.py``):

- T1: manual tensor creation uses the device backend (``_device()``).
- T2: ``@pytest.mark.cuda_only`` / ``torch.cuda`` availability guards removed.
- T3: the ``tune`` parametrization dimension is removed.
- T4: no autotune try/except.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.convolution import Conv3dBiasFwdOp
from tileops.workloads.convolution import Conv3dWorkload, conv3d_tolerance


def _device() -> str:
    return get_device_backend().name


_F16 = torch.float16
_BF16 = torch.bfloat16

# (n, c_in, d, h, w, c_out, k, stride, padding, dilation, dtype, id)
_CASES = [
    (1, 8, 8, 16, 16, 16, 3, 1, 1, 1, _F16, "smoke-k3-s1-fp16"),
    (1, 8, 8, 16, 16, 16, 3, 1, 1, 1, _BF16, "smoke-k3-s1-bf16"),
    # required workload 1: R(2+1)D stem, stride 1
    (1, 3, 16, 112, 112, 64, 3, 1, 1, 1, _F16, "r3d-stem-bias"),
    # required workload 2: video downsampling, stride 2 (W-axis polyphase)
    (1, 64, 8, 56, 56, 128, 3, 2, 1, 1, _F16, "video-down-bias"),
    # required workload 3: 3-D U-Net encoder block, bf16
    (1, 32, 32, 64, 64, 64, 3, 1, 1, 1, _BF16, "unet-encoder-bias"),
    # non-divisible dims / odd channels
    (1, 17, 7, 23, 25, 33, 3, 1, 1, 1, _F16, "nondiv-fp16"),
    # stride 2 mixed axes + f16
    (1, 16, 10, 20, 30, 24, 3, 2, 1, 1, _F16, "stride2-fp16"),
    # padding 0 (valid) + even kernel + f16
    (1, 20, 5, 8, 8, 40, 2, 1, 0, 1, _F16, "k2-pad0-fp16"),
    # batch > 1
    (2, 5, 6, 9, 13, 20, 3, 1, 1, 1, _F16, "batch2-fp16"),
    # dilation > 1 (non-polyphase W-axis)
    (1, 16, 8, 12, 12, 32, 3, 1, 2, 2, _F16, "dilation2-fp16"),
]


def _mark(idx: int, dtype: torch.dtype, case_id: str):
    if idx == 0 and dtype is _F16:
        return [pytest.mark.smoke, pytest.mark.packaging]
    if "bias" in case_id or "smoke" in case_id:
        return pytest.mark.smoke
    return pytest.mark.full


@pytest.mark.parametrize(
    "n, c_in, d_in, h_in, w_in, c_out, kernel_size, stride, padding, dilation, dtype, case_id",
    [
        pytest.param(*c, id=c[-1], marks=_mark(i, c[10], c[-1]))
        for i, c in enumerate(_CASES)
    ],
)
def test_conv3d_bias(
    n, c_in, d_in, h_in, w_in, c_out, kernel_size, stride, padding, dilation, dtype, case_id,
) -> None:
    test = Conv3dWorkload(
        n, c_in, d_in, h_in, w_in, c_out, kernel_size, stride, padding, dilation, 1, dtype, True
    )
    x, weight, bias = test.gen_inputs()
    assert bias is not None
    op = Conv3dBiasFwdOp(stride=stride, padding=padding, dilation=dilation, groups=1)
    out = op(x, weight, bias)
    ref = test.ref_program(x, weight, bias)
    tol = conv3d_tolerance(dtype)
    max_abs = float((out.cpu().float() - ref.cpu().float()).abs().max().item())
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=tol["atol"], rtol=tol["rtol"])
    print(f"[{case_id}] max_abs_err = {max_abs:.3e}")


@pytest.mark.smoke
def test_conv3d_bias_required() -> None:
    """The bias argument is mandatory: omitted or ``None`` is rejected."""
    op = Conv3dBiasFwdOp()
    x = torch.randn(1, 8, 4, 8, 8, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device=_device(), dtype=torch.float16).contiguous()
    with pytest.raises(TypeError):
        op(x, weight)
    with pytest.raises(ValueError, match="requires a bias"):
        op(x, weight, None)


@pytest.mark.smoke
def test_conv3d_bias_rejects_bad_shape() -> None:
    op = Conv3dBiasFwdOp()
    x = torch.randn(1, 8, 4, 8, 8, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(15, device=_device(), dtype=torch.float16).contiguous()
    with pytest.raises(ValueError, match=r"bias must have shape \[C_out\]"):
        op(x, weight, bias)


@pytest.mark.smoke
def test_conv3d_bias_rejects_bad_dtype() -> None:
    op = Conv3dBiasFwdOp()
    x = torch.randn(1, 8, 4, 8, 8, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(16, device=_device(), dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match="bias.dtype"):
        op(x, weight, bias)


@pytest.mark.smoke
def test_conv3d_bias_rejects_cpu_bias() -> None:
    op = Conv3dBiasFwdOp()
    x = torch.randn(1, 8, 4, 8, 8, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(16, dtype=torch.float16).contiguous()
    with pytest.raises(ValueError, match="bias must be a"):
        op(x, weight, bias)


@pytest.mark.smoke
def test_conv3d_bias_rejects_bad_input_ndim() -> None:
    op = Conv3dBiasFwdOp()
    x = torch.randn(1, 8, 8, 8, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(16, device=_device(), dtype=torch.float16).contiguous()
    with pytest.raises(ValueError, match=r"input must have shape \[N, C_in, D, H, W\]"):
        op(x, weight, bias)


@pytest.mark.smoke
def test_conv3d_bias_rejects_unsupported_dtype() -> None:
    op = Conv3dBiasFwdOp()
    x = torch.randn(1, 8, 4, 8, 8, device=_device(), dtype=torch.float32).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device=_device(), dtype=torch.float32).contiguous()
    bias = torch.randn(16, device=_device(), dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match="does not support dtype"):
        op(x, weight, bias)


@pytest.mark.smoke
def test_conv3d_bias_eval_roofline() -> None:
    """eval_roofline() matches the inlined GPU ``conv_roofline`` (3-D) + bias."""
    n, c_in, d_in, h_in, w_in, c_out, k = 1, 8, 4, 8, 8, 16, 3
    op = Conv3dBiasFwdOp(padding=0)
    x = torch.randn(n, c_in, d_in, h_in, w_in, device=_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(c_out, c_in, k, k, k, device=_device(), dtype=torch.float16).contiguous()
    bias = torch.randn(c_out, device=_device(), dtype=torch.float16).contiguous()
    op(x, weight, bias)
    flops, nbytes = op.eval_roofline()
    out_d, out_h, out_w = d_in - k + 1, h_in - k + 1, w_in - k + 1
    taps = out_d * out_h * out_w * (k * k * k)
    # A multiply-add per in-range tap, plus the per-output bias add.
    assert flops == 2 * n * c_out * c_in * taps + n * c_out * out_d * out_h * out_w
    # No padding: every input position is read; the bias tensor is also moved.
    total_elems = (
        n * c_in * d_in * h_in * w_in
        + c_out * c_in * k * k * k
        + n * c_out * out_d * out_h * out_w
        + c_out
    )
    assert nbytes == total_elems * x.element_size()


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
