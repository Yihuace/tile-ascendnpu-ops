"""Multi-dim reduction utilities for the Op layer (device-agnostic).

Copied verbatim from ``tileops/ops/reduction/_multidim.py`` — no GPU/NPU
adaptation needed; this module operates purely on PyTensor shapes.
"""

from __future__ import annotations

from typing import Literal, Union

import torch

__all__ = [
    "flatten_for_multidim",
    "normalize_dim",
    "restore_multidim_shape",
]

EmptyDimPolicy = Literal["reject", "full", "noop"]


def normalize_dim(
    dim: Union[int, list[int], None],
    ndim: int,
    *,
    empty_dim_policy: EmptyDimPolicy = "reject",
) -> list[int]:
    if dim is None:
        return list(range(ndim))

    dims = [dim] if isinstance(dim, int) else list(dim)

    if len(dims) == 0:
        if empty_dim_policy == "full":
            return list(range(ndim))
        if empty_dim_policy == "noop":
            return []
        raise ValueError(
            "dim=[] is not supported by this op; pass "
            'empty_dim_policy="full" to opt in to full-reduction '
            'or empty_dim_policy="noop" to opt in to the identity contract.'
        )

    normalized = []
    for d in dims:
        if d < -ndim or d >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {d})"
            )
        normalized.append(d % ndim)

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate dims in reduction: {dims}")

    return sorted(normalized)


def flatten_for_multidim(
    x: torch.Tensor,
    dims: list[int],
) -> tuple[torch.Tensor, torch.Size, list[int]]:
    orig_shape = x.shape
    ndim = x.ndim
    kept_dims = [i for i in range(ndim) if i not in dims]
    perm = kept_dims + dims
    x = x.permute(perm).contiguous()
    kept_shape = [orig_shape[i] for i in kept_dims]
    reduced_size = 1
    for d in dims:
        reduced_size *= orig_shape[d]
    new_shape = kept_shape + [reduced_size] if kept_shape else [1, reduced_size]
    x = x.reshape(new_shape)
    return x, orig_shape, kept_dims


def restore_multidim_shape(
    y: torch.Tensor,
    orig_shape: torch.Size,
    dims: list[int],
    keepdim: bool,
) -> torch.Tensor:
    ndim = len(orig_shape)
    kept_dims = [i for i in range(ndim) if i not in dims]
    if keepdim:
        out_shape = list(orig_shape)
        for d in dims:
            out_shape[d] = 1
        return y.reshape(out_shape)
    else:
        out_shape = [orig_shape[i] for i in kept_dims]
        if not out_shape:
            return y.squeeze()
        return y.reshape(out_shape)
