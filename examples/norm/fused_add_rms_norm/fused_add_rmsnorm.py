import functools
import math
from numbers import Real
from typing import Tuple

import torch

if __package__:
    from . import kernels as _kernels
    from .selector import select_config
    from .shape_utils import flatten_for_tilelang, rstd_shape
else:
    import kernels as _kernels
    from selector import select_config
    from shape_utils import flatten_for_tilelang, rstd_shape


_KERNEL_FACTORIES = {
    "single_n": _kernels.build_single_n,
    "normal": _kernels.build_normal,
    "normal_auto_multibuffer": _kernels.build_normal_multibuffer,
    "normal_row_one_auto_multibuffer": _kernels.build_row_one_multibuffer,
    "merge_n": _kernels.build_merge_n,
    "multi_n": _kernels.build_multi_n,
    "split_d": _kernels.build_split_d,
    "split_d_row_group": _kernels.build_split_d_row_group,
}


def _validate_scalar(name: str, value: float, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{name} must be a finite Python real number, got {type(value).__name__}"
        )
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return value


@functools.lru_cache()
def _build_kernel(
    M: int,
    N: int,
    eps: float,
    scale: float,
    dtype: str,
):
    resolved_impl, block_m, block_n = select_config(M, N, dtype)
    factory = _KERNEL_FACTORIES[resolved_impl]
    return (
        factory(M, N, eps, dtype, scale)(block_m, block_n),
        resolved_impl,
        block_m,
        block_n,
    )


def _run_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, int]:
    eps = _validate_scalar("eps", eps, nonnegative=True)
    scale = _validate_scalar("scale", scale)
    x_2d, residual_2d, weight_1d, M, N = flatten_for_tilelang(x, residual, weight)
    if x.device.type not in {"npu", "privateuseone"}:
        raise ValueError(
            f"fused_add_rmsnorm requires NPU tensors, got device={x.device}"
        )
    kernel, resolved_impl, block_m, block_n = _build_kernel(
        M,
        N,
        eps,
        scale,
        str(x.dtype).replace("torch.", ""),
    )
    out_2d, rstd_1d, residual_out_2d = kernel(x_2d, residual_2d, weight_1d)
    rstd_out = rstd_1d.reshape(rstd_shape(tuple(x.shape), tuple(weight.shape)))
    return (
        out_2d.reshape_as(x),
        residual_out_2d.reshape_as(x),
        rstd_out,
        resolved_impl,
        block_m,
        block_n,
    )


def fused_add_rmsnorm_full(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out, residual_out, rstd, _, _, _ = _run_tilelang(
        x=x,
        residual=residual,
        weight=weight,
        eps=eps,
        scale=scale,
    )
    return out, residual_out, rstd


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, residual_out, _ = fused_add_rmsnorm_full(
        x=x,
        residual=residual,
        weight=weight,
        eps=eps,
        scale=scale,
    )
    return out, residual_out


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _reference(x, residual, weight, eps, scale):
    value = x.float() * scale + residual.float()
    rstd = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * rstd * weight.float()).to(x.dtype), value.to(x.dtype), rstd


def _run_example(M=128, N=4096, dtype="float16", eps=1e-6, scale=0.5, seed=0):
    import torch_npu  # noqa: F401

    torch.manual_seed(seed)
    torch_dtype = _DTYPES[dtype]
    x_cpu = torch.randn((M, N), dtype=torch_dtype)
    residual_cpu = torch.randn((M, N), dtype=torch_dtype)
    weight_cpu = torch.randn((N,), dtype=torch_dtype)
    expected = _reference(x_cpu, residual_cpu, weight_cpu, eps, scale)
    actual = _run_tilelang(
        x_cpu.npu(), residual_cpu.npu(), weight_cpu.npu(), eps=eps, scale=scale
    )
    out, residual_out, rstd, impl, block_m, block_n = actual
    tolerance = 2e-2 if dtype == "bfloat16" else 1e-2
    for index, result in enumerate((out, residual_out, rstd)):
        reference = expected[index]
        torch.testing.assert_close(
            result.cpu().float(), reference.float(), atol=tolerance, rtol=tolerance
        )
    print(f"Pass: impl={impl}, block_m={block_m}, block_n={block_n}")


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Fused Add RMSNorm example")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--dtype", choices=_DTYPES, default="float16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    _run_example(args.m, args.n, args.dtype, args.eps, args.scale, args.seed)


if __name__ == "__main__":
    import os

    os.environ.setdefault("TILELANG_ASCEND_MODE", "Dev")
    _main()
