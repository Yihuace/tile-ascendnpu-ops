import math

import pytest
import torch
import torch_npu  # noqa: F401

from examples.posembedding.rope_llama31 import RopeLlama31Op


pytestmark = [
    pytest.mark.op("rope_llama31"),
    pytest.mark.mode("Developer"),
]

TOLERANCES = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-3, 1e-3),
    torch.float32: (1e-4, 1e-4),
}


def reference_tables(seq_len, dim, dtype, device):
    indices = torch.arange(0, dim, 2, dtype=torch.float32)
    original = 1.0 / (10000.0 ** (indices / dim))
    wavelength = 2.0 * math.pi / original
    smooth = (8192 / wavelength - 1.0) / 3.0
    smooth = smooth.clamp(0.0, 1.0)
    scaled = (1.0 - smooth) * (original / 8.0) + smooth * original
    frequencies = torch.where(wavelength < 2048.0, original, scaled)
    frequencies = torch.where(wavelength > 8192.0, original / 8.0, frequencies)
    angles = torch.arange(seq_len, dtype=torch.float32)[:, None] * frequencies[None, :]
    return angles.cos().to(device=device, dtype=dtype), angles.sin().to(
        device=device, dtype=dtype
    )


def reference(x, cos, sin):
    output_dtype = x.dtype
    x = x.float()
    cos = cos.float()
    sin = sin.float()
    half = x.shape[-1] // 2
    if x.ndim == 4:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    return torch.cat((x_lo * cos - x_hi * sin, x_hi * cos + x_lo * sin), dim=-1).to(
        output_dtype
    )


CASES = [
    pytest.param("1d", torch.float16, 64, id="1d-fp16-aligned"),
    pytest.param("1d", torch.float16, 80, id="1d-fp16-padded"),
    pytest.param("1d", torch.bfloat16, 64, id="1d-bf16-aligned"),
    pytest.param("1d", torch.bfloat16, 80, id="1d-bf16-padded"),
    pytest.param("1d", torch.float32, 64, id="1d-fp32-aligned"),
    pytest.param("1d", torch.float32, 72, id="1d-fp32-padded"),
    pytest.param("2d", torch.float16, 64, id="bsnd-fp16-aligned"),
    pytest.param("2d", torch.float16, 80, id="bsnd-fp16-padded"),
    pytest.param("2d", torch.bfloat16, 64, id="bsnd-bf16-aligned"),
    pytest.param("2d", torch.bfloat16, 80, id="bsnd-bf16-padded"),
    pytest.param("2d", torch.float32, 64, id="bsnd-fp32-aligned"),
    pytest.param("2d", torch.float32, 72, id="bsnd-fp32-padded"),
]


@pytest.mark.parametrize("layout,dtype,dim", CASES)
def test_rope_llama31_dev(layout, dtype, dim):
    batch, seq_len, heads = (1, 16, 1) if layout == "1d" else (2, 16, 8)
    shape = (seq_len, dim) if layout == "1d" else (batch, seq_len, heads, dim)
    torch.manual_seed(seq_len + heads + dim)
    x = torch.randn(shape, dtype=dtype, device="npu")
    cos, sin = reference_tables(seq_len, dim, dtype, x.device)
    expected = reference(x, cos, sin)
    actual = RopeLlama31Op(seq_len, dim, dtype, layout, batch, heads)(x)
    torch.npu.synchronize()
    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
