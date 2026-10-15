"""Workload definitions for the convolution op family (NPU).

Ported from the GPU TileOPs ``workloads/convolution.py`` -- the
``Conv1dWorkload`` subset only (the shared GPU file also defines the Conv2d /
Conv3d workloads, which are not migrated).

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"`` in the original TileOPs code.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.workloads.workload_base import FixtureBase, WorkloadBase


class Conv1dFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "n, c_in, l_in, c_out, kernel_size, stride, padding, dilation, "
                "groups, dtype, use_bias",
                [
                    pytest.param(
                        2,
                        64,
                        512,
                        128,
                        3,
                        1,
                        1,
                        1,
                        1,
                        torch.float16,
                        False,
                        marks=[pytest.mark.smoke, pytest.mark.packaging],
                        id="smoke-tcn-k3-s1-fp16",
                    ),
                    pytest.param(
                        2,
                        64,
                        512,
                        128,
                        3,
                        1,
                        1,
                        1,
                        1,
                        torch.bfloat16,
                        False,
                        marks=pytest.mark.smoke,
                        id="smoke-tcn-k3-s1-bf16",
                    ),
                    pytest.param(
                        1,
                        17,
                        5,
                        7,
                        3,
                        1,
                        5,
                        2,
                        1,
                        torch.float16,
                        False,
                        marks=pytest.mark.smoke,
                        id="smoke-padding-wider-than-row-fp16",
                    ),
                    pytest.param(
                        4,
                        256,
                        32000,
                        512,
                        1,
                        1,
                        0,
                        1,
                        1,
                        torch.float16,
                        False,
                        marks=pytest.mark.full,
                        id="full-convtasnet-pointwise-k1-s1-fp16",
                    ),
                    pytest.param(
                        4,
                        128,
                        4096,
                        256,
                        3,
                        1,
                        1,
                        1,
                        1,
                        torch.float16,
                        False,
                        marks=pytest.mark.full,
                        id="full-seanet-residual-k3-s1-fp16",
                    ),
                    pytest.param(
                        4,
                        64,
                        16000,
                        128,
                        5,
                        2,
                        2,
                        1,
                        1,
                        torch.float16,
                        False,
                        marks=pytest.mark.full,
                        id="full-audio-downsample-k5-s2-fp16",
                    ),
                    pytest.param(
                        1,
                        32,
                        256,
                        64,
                        7,
                        1,
                        3,
                        1,
                        1,
                        torch.float16,
                        False,
                        marks=pytest.mark.full,
                        id="full-small-seanet-stem-k7-s1-fp16",
                    ),
                    pytest.param(
                        2,
                        128,
                        4096,
                        256,
                        3,
                        2,
                        1,
                        1,
                        1,
                        torch.bfloat16,
                        False,
                        marks=pytest.mark.full,
                        id="full-sequence-downsample-k3-s2-bf16",
                    ),
                    pytest.param(
                        1,
                        32,
                        512,
                        64,
                        3,
                        1,
                        2,
                        2,
                        1,
                        torch.float16,
                        False,
                        marks=pytest.mark.full,
                        id="full-dilation-k3-d2-fp16",
                    ),
                    pytest.param(
                        1,
                        32,
                        128,
                        64,
                        3,
                        1,
                        "valid",
                        1,
                        1,
                        torch.float16,
                        False,
                        marks=pytest.mark.full,
                        id="full-padding-valid-fp16",
                    ),
                    pytest.param(
                        1,
                        32,
                        128,
                        64,
                        3,
                        1,
                        "same",
                        1,
                        1,
                        torch.float16,
                        False,
                        marks=pytest.mark.full,
                        id="full-padding-same-fp16",
                    ),
                    pytest.param(
                        1,
                        32,
                        128,
                        64,
                        3,
                        1,
                        1,
                        1,
                        2,
                        torch.float16,
                        False,
                        marks=pytest.mark.full,
                        id="full-groups2-k3-fp16",
                    ),
                    pytest.param(
                        1,
                        48,
                        128,
                        72,
                        3,
                        1,
                        1,
                        1,
                        3,
                        torch.float16,
                        False,
                        marks=pytest.mark.full,
                        id="full-groups3-coutg24-fp16",
                    ),
                    pytest.param(
                        1,
                        64,
                        128,
                        64,
                        31,
                        1,
                        15,
                        1,
                        64,
                        torch.float16,
                        True,
                        marks=pytest.mark.full,
                        id="full-conformer-depthwise-k31-fp16",
                    ),
                ],
            ),
        ]


class Conv1dWorkload(WorkloadBase):
    """1-D convolution workload: NCL input, OIL weight, optional per-channel bias."""

    def __init__(
        self,
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
        use_bias: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype
        self.use_bias = use_bias

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        device = get_device_backend().name
        x = torch.randn(
            self.n, self.c_in, self.l_in, device=device, dtype=self.dtype
        ).contiguous()
        weight = torch.randn(
            self.c_out,
            self.c_in // self.groups,
            self.kernel_size,
            device=device,
            dtype=self.dtype,
        ).contiguous()
        bias = (
            torch.zeros(self.c_out, device=device, dtype=self.dtype).contiguous()
            if self.use_bias
            else None
        )
        return x, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out = F.conv1d(
            x.float(),
            weight.float(),
            bias=bias.float() if bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out.to(x.dtype).contiguous()


class Conv3dWorkload(WorkloadBase):
    """3-D convolution workload: NCDHW input, O(kDkHkW) weight, optional bias."""

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        c_out: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int],
        padding: int | tuple[int, int, int] | str,
        dilation: int | tuple[int, int, int],
        groups: int,
        dtype: torch.dtype,
        use_bias: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype
        self.use_bias = use_bias

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        device = get_device_backend().name
        kd, kh, kw = (
            (self.kernel_size,) * 3
            if isinstance(self.kernel_size, int)
            else tuple(self.kernel_size)
        )
        x = torch.randn(
            self.n, self.c_in, self.d_in, self.h_in, self.w_in,
            device=device, dtype=self.dtype,
        ).contiguous()
        weight = torch.randn(
            self.c_out,
            self.c_in // self.groups,
            kd,
            kh,
            kw,
            device=device,
            dtype=self.dtype,
        ).contiguous()
        bias = (
            torch.zeros(self.c_out, device=device, dtype=self.dtype).contiguous()
            if self.use_bias
            else None
        )
        return x, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x_cpu = x.detach().to("cpu")
        out = F.conv3d(
            x_cpu.float(),
            weight.detach().to("cpu").float(),
            bias=bias.detach().to("cpu").float() if bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out.to(x_cpu.dtype).contiguous()


def conv3d_tolerance(
    dtype: torch.dtype,
    *,
    padding: int | tuple | str = 0,
    kernel_shape: tuple = (),
) -> dict:
    """Numerical tolerance for a Conv3d comparison against the fp32 reference.

    A half-precision kernel accumulating in fp32 drifts with the reduction
    length (``C_in * kD * kH * kW``), so bf16 needs the looser pair.
    """
    if dtype == torch.float16:
        return {"atol": 1.5e-2, "rtol": 1e-2}
    if dtype == torch.bfloat16:
        return {"atol": 3e-2, "rtol": 2e-2}
    return {"atol": 1e-4, "rtol": 1e-4}


def conv1d_tolerance(
    dtype: torch.dtype,
    *,
    padding: int | str = 0,
    kernel_shape: tuple = (),
) -> dict:
    """Numerical tolerance for a Conv1d comparison against the fp32 reference.

    A half-precision kernel accumulating in fp32 drifts with the reduction
    length; an even filter under ``'same'`` padding needs an asymmetric pad and
    the reference and kernel may pick different reduction orders.
    """
    if dtype == torch.float16:
        tolerance = {"atol": 1e-2, "rtol": 1e-2}
        if padding == "same" and any(k % 2 == 0 for k in kernel_shape):
            tolerance = {"atol": 2e-3, "rtol": 3e-3}
        return tolerance
    if dtype == torch.bfloat16:
        return {"atol": 2e-2, "rtol": 2e-2}
    return {"atol": 1e-4, "rtol": 1e-4}
