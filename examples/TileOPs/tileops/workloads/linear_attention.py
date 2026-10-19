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


# ---------------------------------------------------------------------------
# GDN prefill (GatedDeltaNetPrefillFwdOp)
# ---------------------------------------------------------------------------
def gdn_prefill_golden(q, k, v, g, beta, state, scale):
    """Token-sequential delta-rule reference (fp64 accumulation)."""
    B, S, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    group = HV // H
    o = torch.zeros(B, S, HV, V, dtype=torch.float64)
    st = state.clone().double()
    for b in range(B):
        for t in range(S):
            for hv in range(HV):
                h = hv // group
                dec = torch.exp(g[b, t, hv].double())
                kt = k[b, t, h].double()
                vt = v[b, t, hv].double()
                qt = q[b, t, h].double()
                corr = vt - kt @ st[b, hv]
                st[b, hv] = dec * st[b, hv] + beta[b, t, hv].double() * torch.outer(kt, corr)
                o[b, t, hv] = scale * (qt @ st[b, hv])
    return o, st


class GDNWorkload(WorkloadBase):
    """Equal-length GDN prefill workload (B=1, key-major state)."""

    def __init__(self, B, S, H, HV, K, V, dtype, scale=-1.0, has_state=True):
        self.B, self.S, self.H, self.HV = B, S, H, HV
        self.K, self.V = K, V
        self.dtype = dtype
        self.scale = float(scale) if scale is not None and scale > 0 else K ** -0.5
        self.has_state = has_state

    def gen_inputs(self):
        dev = get_device_backend().name
        q = torch.randn(self.B, self.S, self.H, self.K, device=dev, dtype=self.dtype)
        k = torch.nn.functional.normalize(
            torch.randn(self.B, self.S, self.H, self.K, device=dev, dtype=self.dtype), dim=-1
        )
        v = torch.randn(self.B, self.S, self.HV, self.V, device=dev, dtype=self.dtype)
        g = -torch.rand(self.B, self.S, self.HV, device=dev) * 0.2
        beta = torch.rand(self.B, self.S, self.HV, device=dev) * 0.1
        state = (
            torch.randn(self.B, self.HV, self.K, self.V, device=dev) * 0.1
            if self.has_state
            else torch.zeros(self.B, self.HV, self.K, self.V, device=dev)
        )
        return q, k, v, g, beta, state

    def ref_program(self, q, k, v, g, beta, state):
        return gdn_prefill_golden(
            q.detach().to("cpu"), k.detach().to("cpu"), v.detach().to("cpu"),
            g.detach().to("cpu"), beta.detach().to("cpu"), state.detach().to("cpu"),
            self.scale,
        )


def gdn_tolerance(dtype):
    if dtype == torch.float16:
        return {"atol": 5e-3, "rtol": 0}
    if dtype == torch.bfloat16:
        return {"atol": 1.6e-2, "rtol": 0}
    return {"atol": 1e-4, "rtol": 0}


__all__ += ["GDNWorkload", "gdn_prefill_golden", "gdn_tolerance"]
