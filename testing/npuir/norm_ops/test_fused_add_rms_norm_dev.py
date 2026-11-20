import pytest
import torch
import torch_npu  # noqa: F401

from examples.norm.fused_add_rms_norm import fused_add_rmsnorm_full
from examples.norm.fused_add_rms_norm.kernels.common import (
    DEFAULT_ASCENDC_UB_BYTES,
    max_normal_rows,
    normal_ub_bytes,
)
from examples.norm.fused_add_rms_norm.selector import select_config


pytestmark = [
    pytest.mark.op("fused_add_rms_norm"),
    pytest.mark.mode("Developer"),
]

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

CASES = [
    pytest.param((7, 128), (128,), "float16", 1.0, id="fp16-aligned-row-tail"),
    pytest.param((2, 5, 1025), (1025,), "float16", 0.5, id="fp16-unaligned-nd"),
    pytest.param((64, 1025), (1025,), "float16", 0.3, id="fp16-merge-fp32-add"),
    pytest.param((64, 2048), (2048,), "float16", 0.3, id="fp16-multi-fp32-add"),
    pytest.param((64, 2001), (2001,), "float16", 0.5, id="fp16-normal-auto-fp32-add"),
    pytest.param((257, 2049), (2049,), "float16", 1.0, id="fp16-normal-fp32-add"),
    pytest.param((257, 5969), (5969,), "float16", 0.3, id="fp16-row-one-fp32-add"),
    pytest.param((65, 16384), (16384,), "float16", 0.3, id="fp16-split-d-fp32-add"),
    pytest.param(
        (65, 12289),
        (12289,),
        "float16",
        0.3,
        id="fp16-row-group-fp32-add",
    ),
    pytest.param((31, 4096), (4096,), "bfloat16", 1.0, id="bf16-aligned-small-m"),
    pytest.param((33, 4097), (4097,), "bfloat16", 0.5, id="bf16-unaligned-scale-half"),
    pytest.param((3, 10241), (10241,), "float32", 1.0, id="fp32-unaligned-split-d"),
    pytest.param((2, 3, 2048), (3, 2048), "float32", 0.5, id="fp32-nd-gamma"),
]


def _reference(x, residual, weight, eps, scale):
    reduce_dims = tuple(range(x.ndim - weight.ndim, x.ndim))
    residual_out = x.float() * scale + residual.float()
    rstd = torch.rsqrt(residual_out.square().mean(dim=reduce_dims, keepdim=True) + eps)
    out = (residual_out * rstd * weight.float()).to(x.dtype)
    return out, residual_out.to(x.dtype), rstd


@pytest.mark.parametrize(
    "dtype,n,expected_low,expected_high",
    [
        ("bfloat16", 16384, ("split_d", 31, 256), ("split_d", 4, 2048)),
        ("float32", 10241, ("split_d", 31, 256), ("split_d", 4, 1024)),
    ],
)
def test_split_d_balanced_boundary(monkeypatch, dtype, n, expected_low, expected_high):
    monkeypatch.setenv("TILELANG_ASCEND_AIV_CORES", "48")
    assert select_config(31 * 48, n, dtype) == expected_low
    assert select_config(32 * 48, n, dtype) == expected_high


@pytest.mark.parametrize(
    "n,expected_rows",
    [
        (2001, 6),
        (2271, 6),
        (2272, 5),
        (2287, 5),
        (2288, 5),
        (2703, 5),
        (2704, 4),
        (3311, 4),
        (3312, 3),
        (4255, 3),
        (4256, 2),
        (5967, 2),
        (5968, 1),
    ],
)
def test_fp16_normal_ub_model_includes_fp32_add_workspace(n, expected_rows):
    rows = max_normal_rows(n, "float16")
    assert rows == expected_rows
    assert normal_ub_bytes(rows, n, "float16") <= DEFAULT_ASCENDC_UB_BYTES
    assert normal_ub_bytes(rows + 1, n, "float16") > DEFAULT_ASCENDC_UB_BYTES


@pytest.mark.parametrize("shape,weight_shape,dtype,scale", CASES)
def test_fused_add_rms_norm_dev(shape, weight_shape, dtype, scale):
    eps = 1e-6
    torch.manual_seed(0)
    torch_dtype = DTYPES[dtype]
    x_cpu = torch.randn(shape, dtype=torch_dtype)
    residual_cpu = torch.randn(shape, dtype=torch_dtype)
    weight_cpu = torch.randn(weight_shape, dtype=torch_dtype)
    expected = _reference(x_cpu, residual_cpu, weight_cpu, eps, scale)

    actual = fused_add_rmsnorm_full(
        x_cpu.npu(),
        residual_cpu.npu(),
        weight_cpu.npu(),
        eps=eps,
        scale=scale,
    )

    tolerance = 2e-2 if dtype == "bfloat16" else 1e-2
    for index, result in enumerate(actual):
        reference = expected[index]
        torch.testing.assert_close(
            result.cpu().float(),
            reference.float(),
            atol=tolerance,
            rtol=tolerance,
        )
