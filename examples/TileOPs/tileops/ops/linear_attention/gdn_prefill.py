"""GatedDeltaNetPrefillFwdOp (chunked gated delta-rule prefill).

Delivered name ``GatedDeltaNetPrefillFwdOp``; baseline GPU op ``GDNFwdOp``.
Prefill only (equal length, key-major state).  The kernel is currently a
correctness-first NPU reference; the six-kernel TileLang pipeline is being
ported incrementally.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention.gdn_prefill.gdn_prefill import GDNPrefillKernel
from tileops.ops.op_base import Op

__all__ = ["GatedDeltaNetPrefillFwdOp"]


class GatedDeltaNetPrefillFwdOp(Op):
    def __init__(
        self,
        B: int,
        S: int,
        H: int,
        HV: int,
        K: int,
        V: int,
        dtype: torch.dtype = torch.bfloat16,
        scale: Optional[float] = None,
        chunk: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.B, self.S, self.H, self.HV, self.K, self.V = B, S, H, HV, K, V
        self.dtype = dtype
        self.scale = float(scale) if scale is not None else K ** -0.5
        self.chunk = chunk
        self.dispatch_kernel(kernel_map)
        self.kernel: Kernel = self.kernel_map["gdn_prefill"](
            B, S, H, HV, K, V, dtype, self.scale, chunk
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gdn_prefill": GDNPrefillKernel}

    def forward(self, q, k, v, g, beta, state=None):
        return self.kernel(q, k, v, g, beta, state)

    def eval_roofline(self) -> tuple[int, int]:
        flops = self.B * self.S * self.HV * (
            4 * self.K * self.V + 4 * self.K + 3 * self.V
        )
        elem_bytes = self.dtype.itemsize
        nbytes = self.B * self.S * (
            self.H * self.K * 2 + self.HV * self.V + self.HV * 2
        ) + self.B * self.HV * self.K * self.V * 2
        return int(flops), int(nbytes * elem_bytes)
