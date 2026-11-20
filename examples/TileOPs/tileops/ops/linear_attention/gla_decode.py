"""GLADecodeOp (single-token gated linear attention decode).

Migrated from the GPU ``GLARecurrentFwdOp`` (delivered under the task-list name
``GLADecodeOp``).  Layout BHD; one recurrent step per call:

    state_new[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
    o[dv]             = scale * sum_dk q[dk] * state_new[dk, dv]

The ``scale`` parameter is semantic and fixed at op construction (non-positive
means ``DK ** -0.5``); ``state`` is returned functionally as ``new_state``.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention.gla_decode import GLADecodeKernel
from tileops.ops.op_base import Op

__all__ = ["GLADecodeOp"]


class GLADecodeOp(Op):
    """Single-step (decode) gated linear attention, layout BHD.

    Args:
        B, H, DK, DV: batch / heads / key-dim / value-dim.
        scale: readout scale; non-positive means ``DK ** -0.5``.
        dtype: float16 / bfloat16 / float32.
        k_tile: optional DK sub-tile override (else fits UB automatically).
        kernel_map: optional override for kernel dispatch.
    """

    def __init__(
        self,
        B: int,
        H: int,
        DK: int,
        DV: int,
        scale: float = -1.0,
        dtype: torch.dtype = torch.bfloat16,
        k_tile: Optional[int] = None,
        state_dtype: Optional[torch.dtype] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.B = B
        self.H = H
        self.DK = DK
        self.DV = DV
        self.scale = scale
        self.dtype = dtype
        # P0-0h: bf16 128x128 serving shapes keep the recurrent state in fp16.
        # Compute is already fp16 and fp16 has the wider mantissa, so this
        # removes two full-plane casts with no extra rounding.
        if state_dtype is None:
            state_dtype = (
                torch.float16
                if (dtype == torch.bfloat16 and DK == 128 and DV == 128 and k_tile is None)
                else dtype
            )
        self.state_dtype = state_dtype
        self.dispatch_kernel(kernel_map)
        self.kernel: Kernel = self.kernel_map["gla_decode"](
            B, H, DK, DV, dtype, scale, k_tile, state_dtype
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gla_decode": GLADecodeKernel}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one decode step; returns ``(o, new_state)``.

        P0-0h contract: bf16 128x128 serving shapes carry the recurrent state
        in fp16 (see ``state_dtype``); a mismatched tensor is rejected rather
        than silently reinterpreted.
        """
        if state.dtype != self.state_dtype:
            raise ValueError(
                f"Expected state.dtype {self.state_dtype}, got {state.dtype}. "
                "bf16 128x128 GLADecode keeps the recurrent state in fp16."
            )
        return self.kernel(q, k, v, gk, state)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for ``B*H*(5*DK*DV + DK + DV)`` FLOPs."""
        elem_bytes = self.dtype.itemsize
        state_bytes = self.state_dtype.itemsize
        flops = self.B * self.H * (5 * self.DK * self.DV + self.DK + self.DV)
        nbytes = self.B * self.H * (
            (3 * self.DK + 2 * self.DV) * elem_bytes
            + 2 * self.DK * self.DV * state_bytes
        )
        return int(flops), int(nbytes)
