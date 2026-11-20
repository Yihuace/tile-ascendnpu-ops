from math import prod
from typing import Sequence, Tuple

import torch


def validate_rmsnorm_shapes(
    shape: Sequence[int], gamma_shape: Sequence[int]
) -> Tuple[int, int]:
    shape = tuple(int(dim) for dim in shape)
    gamma_shape = tuple(int(dim) for dim in gamma_shape)
    if not shape or not gamma_shape or any(dim <= 0 for dim in shape + gamma_shape):
        raise ValueError(
            f"Invalid AddRmsNorm shape: shape={shape}, gamma_shape={gamma_shape}"
        )
    if tuple(shape[-len(gamma_shape) :]) != gamma_shape:
        raise ValueError(
            "gamma_shape must match the normalized suffix of input shape: "
            f"shape={shape}, gamma_shape={gamma_shape}"
        )
    M = max(int(prod(shape[: -len(gamma_shape)])), 1)
    N = int(prod(gamma_shape))
    return M, N


def rstd_shape(shape: Sequence[int], gamma_shape: Sequence[int]) -> Tuple[int, ...]:
    validate_rmsnorm_shapes(shape, gamma_shape)
    return tuple(shape[: len(shape) - len(gamma_shape)]) + (1,) * len(gamma_shape)


def flatten_for_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (x, residual, weight)):
        raise TypeError("x, residual, and weight must all be torch.Tensor instances")
    if tuple(x.shape) != tuple(residual.shape):
        raise ValueError(
            f"x and residual shape mismatch: {tuple(x.shape)} vs {tuple(residual.shape)}"
        )
    if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(
            f"Unsupported dtype {x.dtype}; expected float16, bfloat16, or float32"
        )
    if residual.dtype != x.dtype or weight.dtype != x.dtype:
        raise TypeError(
            "x, residual, and weight must have the same dtype: "
            f"x={x.dtype}, residual={residual.dtype}, weight={weight.dtype}"
        )
    if residual.device != x.device or weight.device != x.device:
        raise ValueError(
            "x, residual, and weight must be on the same device: "
            f"x={x.device}, residual={residual.device}, weight={weight.device}"
        )
    M, N = validate_rmsnorm_shapes(tuple(x.shape), tuple(weight.shape))
    return (
        x.contiguous().reshape(M, N),
        residual.contiguous().reshape(M, N),
        weight.contiguous().reshape(N),
        M,
        N,
    )
