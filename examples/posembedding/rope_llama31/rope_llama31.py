import functools
import math
from typing import Dict, Tuple

import torch

if __package__:
    from .kernels import (
        build_blocked_fp32,
        build_blocked_low_precision,
        build_table_fp32,
        build_table_low_precision,
    )
    from .selector import TilePlan, select_tile_plan
else:
    from kernels import (
        build_blocked_fp32,
        build_blocked_low_precision,
        build_table_fp32,
        build_table_low_precision,
    )
    from selector import TilePlan, select_tile_plan


_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


@functools.lru_cache()
def _build_plan_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    dtype: str,
    plan: TilePlan,
):
    if plan.route == "table":
        if dtype == "float32":
            return build_table_fp32(
                batch,
                seq_len,
                heads,
                dim,
                plan.block_tokens,
                plan.bn_groups,
            )
        return build_table_low_precision(
            batch,
            seq_len,
            heads,
            dim,
            dtype,
            plan.block_tokens,
            plan.bn_groups,
        )
    partition_by_seq = plan.route == "arbitrary"
    if dtype == "float32":
        return build_blocked_fp32(
            batch,
            seq_len,
            heads,
            dim,
            plan.block_tokens,
            plan.block_heads,
            partition_by_seq,
            plan.merge_copyout,
        )
    return build_blocked_low_precision(
        batch,
        seq_len,
        heads,
        dim,
        dtype,
        plan.block_tokens,
        plan.block_heads,
        partition_by_seq,
        plan.merge_copyout,
    )


def _make_tables(
    seq_len: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    base: float,
    scale_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(0, dim, 2, dtype=torch.float32)
    original = 1.0 / (base ** (indices / dim))
    wavelength = 2.0 * math.pi / original
    low_wavelength = original_max_position / low_freq_factor
    high_wavelength = original_max_position / high_freq_factor
    smooth = (original_max_position / wavelength - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smooth = smooth.clamp(0.0, 1.0)
    scaled = (1.0 - smooth) * (original / scale_factor) + smooth * original
    frequencies = torch.where(wavelength < high_wavelength, original, scaled)
    frequencies = torch.where(
        wavelength > low_wavelength, original / scale_factor, frequencies
    )
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = positions[:, None] * frequencies[None, :]
    return angles.cos().to(device=device, dtype=dtype), angles.sin().to(
        device=device, dtype=dtype
    )


class RopeLlama31Op(torch.nn.Module):
    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        layout: str,
        batch: int = 1,
        num_heads: int = 1,
        base: float = 10000.0,
        scale_factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position: int = 8192,
    ):
        super().__init__()
        if layout not in ("1d", "2d"):
            raise ValueError(f"layout must be '1d' or '2d', got {layout!r}")
        if seq_len <= 0 or head_dim <= 0 or head_dim % 2:
            raise ValueError(
                f"positive S and positive even D are required, got S={seq_len}, D={head_dim}"
            )
        if batch <= 0 or num_heads <= 0:
            raise ValueError(
                f"positive batch and num_heads are required, got B={batch}, N={num_heads}"
            )
        if dtype not in _DTYPE_NAMES:
            raise TypeError(f"unsupported dtype: {dtype}")
        if base <= 0 or scale_factor <= 0 or low_freq_factor <= 0:
            raise ValueError("base, scale_factor and low_freq_factor must be positive")
        if high_freq_factor <= low_freq_factor:
            raise ValueError("high_freq_factor must be greater than low_freq_factor")
        if original_max_position <= 0:
            raise ValueError("original_max_position must be positive")

        self.seq_len = int(seq_len)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.layout = layout
        self.batch = int(batch)
        self.num_heads = int(num_heads)
        self.base = float(base)
        self.scale_factor = float(scale_factor)
        self.low_freq_factor = float(low_freq_factor)
        self.high_freq_factor = float(high_freq_factor)
        self.original_max_position = int(original_max_position)
        dtype_name = _DTYPE_NAMES[dtype]
        batch_size = 1 if layout == "1d" else self.batch
        self._heads = 1 if layout == "1d" else self.num_heads
        self._tokens = batch_size * self.seq_len
        plan = select_tile_plan(
            batch_size,
            self.seq_len,
            self._heads,
            self.head_dim,
            dtype_name,
            self.layout,
        )
        self._kernel = _build_plan_kernel(
            batch_size,
            self.seq_len,
            self._heads,
            self.head_dim,
            dtype_name,
            plan,
        )
        self._table_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if self.layout == "1d":
            return (self.seq_len, self.head_dim)
        return (self.batch, self.seq_len, self.num_heads, self.head_dim)

    def _tables(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        if key not in self._table_cache:
            self._table_cache[key] = _make_tables(
                self.seq_len,
                self.head_dim,
                self.dtype,
                device,
                self.base,
                self.scale_factor,
                self.low_freq_factor,
                self.high_freq_factor,
                self.original_max_position,
            )
        return self._table_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape) != self.input_shape:
            raise ValueError(
                f"expected x.shape={self.input_shape}, got {tuple(x.shape)}"
            )
        if x.dtype != self.dtype:
            raise TypeError(f"expected x.dtype={self.dtype}, got {x.dtype}")
        if x.device.type != "npu":
            raise ValueError(f"only NPU tensors are supported, got {x.device}")
        x_kernel = x.contiguous().reshape(self._tokens, self._heads, self.head_dim)
        cos, sin = self._tables(x.device)
        return self._kernel(x_kernel, cos, sin).reshape(self.input_shape)


__all__ = ["RopeLlama31Op"]


_EXAMPLE_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}


def _reference(x, cos, sin):
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


def _run_example(
    seq_len=128,
    head_dim=128,
    dtype="float16",
    layout="2d",
    batch=1,
    num_heads=8,
    seed=0,
):
    import torch_npu  # noqa: F401

    torch.manual_seed(seed)
    torch_dtype = _EXAMPLE_DTYPES[dtype]
    shape = (
        (seq_len, head_dim) if layout == "1d" else (batch, seq_len, num_heads, head_dim)
    )
    x_cpu = torch.randn(shape, dtype=torch_dtype)
    cos, sin = _make_tables(
        seq_len,
        head_dim,
        torch_dtype,
        torch.device("cpu"),
        10000.0,
        8.0,
        1.0,
        4.0,
        8192,
    )
    expected = _reference(x_cpu, cos, sin)
    op = RopeLlama31Op(
        seq_len,
        head_dim,
        torch_dtype,
        layout,
        batch,
        num_heads,
    )
    actual = op(x_cpu.npu()).cpu()
    tolerance = 1e-4 if dtype == "float32" else 1e-3
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=tolerance, rtol=tolerance
    )
    kernel_batch = 1 if layout == "1d" else batch
    kernel_heads = 1 if layout == "1d" else num_heads
    plan = select_tile_plan(
        kernel_batch,
        seq_len,
        kernel_heads,
        head_dim,
        dtype,
        layout,
    )
    config = (
        f"route={plan.route}, block_tokens={plan.block_tokens}, "
        f"block_heads={plan.block_heads}"
    )
    if plan.bn_groups:
        config += f", bn_groups={plan.bn_groups}"
    print(f"Pass: {config}")


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Llama 3.1 RoPE example")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=_EXAMPLE_DTYPES, default="float16")
    parser.add_argument("--layout", choices=("1d", "2d"), default="2d")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    _run_example(
        args.seq_len,
        args.head_dim,
        args.dtype,
        args.layout,
        args.batch,
        args.num_heads,
        args.seed,
    )


if __name__ == "__main__":
    import os

    os.environ.setdefault("TILELANG_ASCEND_MODE", "Dev")
    _main()
