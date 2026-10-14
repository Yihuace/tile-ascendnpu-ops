"""Workload definitions for the norm op family.

Ported from TileOPs ``workloads/normalization.py`` --
``AdaLayerNormWorkload`` and ``FusedAddLayerNormWorkload`` only (the other norm
workloads are not migrated).
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase


class AdaLayerNormWorkload(WorkloadBase):
    """Workload definition for AdaLayerNormFwdOp.

    Generates the three same-shape operands (``x`` / ``scale`` / ``shift``)
    via ``randn``.  Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        x = torch.randn(self.m, self.n, dtype=self.dtype, device=device)
        scale = torch.randn(self.m, self.n, dtype=self.dtype, device=device)
        shift = torch.randn(self.m, self.n, dtype=self.dtype, device=device)
        return x, scale, shift


class FusedAddLayerNormWorkload(WorkloadBase):
    """Workload definition for FusedAddLayerNormFwdOp.

    Generates the four operands of the fused residual-add + LayerNorm op:
    ``x`` / ``residual`` of shape ``(m, n)`` and the affine ``weight`` / ``bias``
    of shape ``(n,)``.  Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.

    Ported from TileOPs ``workloads/norm.py:FusedAddLayerNormWorkload``.
    """

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        x = torch.randn(self.m, self.n, dtype=self.dtype, device=device)
        residual = torch.randn(self.m, self.n, dtype=self.dtype, device=device)
        weight = torch.randn(self.n, dtype=self.dtype, device=device)
        bias = torch.randn(self.n, dtype=self.dtype, device=device)
        return x, residual, weight, bias


def batch_norm_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The gradients of the training forward, from the saved statistics.

    Independent pure-PyTorch reference (fp32 intermediate, rounded back to the
    input dtype). Ported from the GPU ``workloads/norm.py:batch_norm_backward``.
    """
    shape = (1, -1, *[1] * (x.ndim - 2))
    axes = [0, *range(2, x.ndim)]
    x_hat = (x.float() - mean.view(shape)) * rstd.view(shape)
    grad_bias = grad_out.float().sum(axes)
    grad_weight = (grad_out.float() * x_hat).sum(axes)
    length = x.numel() // x.shape[1]
    grad_x = (
        (weight * rstd).view(shape)
        / length
        * (length * grad_out.float() - grad_bias.view(shape) - x_hat * grad_weight.view(shape))
    )
    return grad_x.to(x.dtype), grad_weight, grad_bias


def batch_norm_bwd_tolerance(dtype: torch.dtype) -> dict:
    """Elementwise (input-gradient) and accumulated (channel-gradient) bounds.

    Ported from the GPU ``workloads/norm.py:batch_norm_backward_verification``:
    the channel gradients reduce ``N * S`` values, so cancellation needs an
    absolute bound.
    """
    if dtype == torch.float32:
        atol = rtol = 1e-5
        sum_atol = 1e-4
    elif dtype == torch.float16:
        atol = rtol = 1e-2
        sum_atol = 1e-2
    else:  # bfloat16
        atol = rtol = 2e-2
        sum_atol = 2e-2
    return {"atol": atol, "rtol": rtol, "sum_atol": sum_atol}


class BatchNormBwdWorkload(WorkloadBase):
    """Workload definition for BatchNormBwdOp.

    Generates the five operands of the batch-norm backward op: ``grad_out`` and
    ``x`` of shape ``(n, c, s)`` (the flattened ``(N, C, *spatial)`` view), the
    affine ``weight`` of shape ``(c,)`` in fp32, and the per-channel ``mean`` /
    ``rstd`` the training forward saved -- recomputed here as the batch
    statistics of ``x`` so the reference is self-consistent.

    Ported from the GPU ``workloads/norm.py:BatchNormBwdCall`` (which derives
    ``mean`` / ``rstd`` from ``x``). Adaptation point W1: ``device`` is resolved
    from :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, n: int, c: int, s: int, dtype: torch.dtype):
        self.n = n
        self.c = c
        self.s = s
        self.dtype = dtype

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        grad_out = torch.randn(self.n, self.c, self.s, dtype=self.dtype, device=device)
        x = torch.randn(self.n, self.c, self.s, dtype=self.dtype, device=device)
        weight = torch.randn(self.c, dtype=torch.float32, device=device)
        var, mean = torch.var_mean(x.float(), dim=[0, 2], correction=0)
        rstd = torch.rsqrt(var + 1e-5)
        return grad_out, x, weight, mean, rstd

    def ref_program(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return batch_norm_backward(grad_out, x, weight, mean, rstd)
