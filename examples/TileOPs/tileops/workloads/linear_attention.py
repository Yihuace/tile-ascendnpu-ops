"""Workload definitions for the linear_attention op family (NPU).

Ported from the GPU TileOPs ``workloads/linear_attention`` for the single op
migrated so far, ``GLADecodeOp`` (baseline ``GLARecurrentFwdOp``).  Device
placement goes through :func:`tileops.device.get_device_backend`.
"""

import torch

from tileops.device import get_device_backend
from tileops.kernels.linear_attention.gla_decode.gla_decode_kernel._gla_decode_kernel import (
    golden_gla_decode,
)
from tileops.workloads.workload_base import WorkloadBase

__all__ = ["GLADecodeWorkload", "gla_decode_tolerance"]


class GLADecodeWorkload(WorkloadBase):
    """Single-token GLA decode workload (layout BHD)."""

    def __init__(
        self,
        B: int,
        H: int,
        DK: int,
        DV: int,
        dtype: torch.dtype,
        scale: float = -1.0,
    ) -> None:
        self.B = B
        self.H = H
        self.DK = DK
        self.DV = DV
        self.dtype = dtype
        self.scale = float(scale) if scale is not None and scale > 0 else DK ** -0.5

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        q = (torch.randn(self.B, self.H, self.DK, device=device, dtype=self.dtype) * 0.5).contiguous()
        k = (torch.randn(self.B, self.H, self.DK, device=device, dtype=self.dtype) * 0.5).contiguous()
        v = (torch.randn(self.B, self.H, self.DV, device=device, dtype=self.dtype) * 0.5).contiguous()
        gk = (-torch.rand(self.B, self.H, self.DK, device=device, dtype=self.dtype)).contiguous()
        state = torch.randn(self.B, self.H, self.DK, self.DV, device=device, dtype=self.dtype).contiguous()
        return q, k, v, gk, state

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return golden_gla_decode(
            q.detach().to("cpu"),
            k.detach().to("cpu"),
            v.detach().to("cpu"),
            gk.detach().to("cpu"),
            state.detach().to("cpu"),
            self.scale,
        )


def gla_decode_tolerance(dtype: torch.dtype) -> dict:
    """Numerical tolerance for a GLA decode comparison against the fp32 golden."""
    if dtype == torch.float16:
        return {"atol": 2e-3, "rtol": 2e-3}
    if dtype == torch.bfloat16:
        return {"atol": 2e-2, "rtol": 2e-2}
    return {"atol": 1e-4, "rtol": 1e-4}
