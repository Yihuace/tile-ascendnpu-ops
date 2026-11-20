"""Engram GateConv forward operator — post-projection fusion (NPU-adapted).

Assumes ``k = E @ W_K`` and ``v = E @ W_V`` have been computed externally via
standard GEMM. This op fuses the remaining memory-bound stages:

    RMSNorm gating -> causal DWConv1D -> SiLU + residual

Returns ``Y`` plus saved intermediates for backward (strategy B):
``vhat, alpha, rrms_h, rrms_k, rrms_v``.

Adaptation from GPU (TileOPs) to NPU:

- ``forward``: ``H.is_cuda`` device check -> ``backend.is_device_tensor(H)``
  via :func:`get_device_backend` (O1/O2).
- ``tune`` parameter removed (O3); the kernel constructor is called without it.
- ``eval_roofline``: the GPU
  ``tileops.perf.formulas.engram_gate_conv_fwd_roofline`` re-implemented inline
  (the NPU project has no ``tileops.perf`` module; roofline arithmetic is
  device-agnostic) (C24).
- Forward flow (device/dtype/shape validation -> contiguous -> kernel dispatch)
  preserved unchanged (O6).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sequence_modeling.engram_gate_conv_fwd import EngramGateConvFwdKernel
from tileops.ops.op_base import Op

__all__ = ["EngramGateConvFwdOp"]

CONV_KERNEL_SIZE = 4


class EngramGateConvFwdOp(Op):
    """Engram GateConv forward operator (post-projection fusion).

    Assumes k = E @ W_K and v = E @ W_V have been computed externally
    via standard GEMM. This op fuses the remaining memory-bound stages:

        RMSNorm gating -> causal DWConv1D -> SiLU + residual

    Returns Y plus saved intermediates for backward (strategy B):
        vhat, alpha, rrms_h, rrms_k, rrms_v

    Args:
        M: Batch size.
        seq_len: Sequence length.
        d: Model hidden dimension.
        dtype: Data type (float16 or bfloat16).
        eps: RMSNorm epsilon (default 1e-6).
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        M: int,
        seq_len: int,
        d: int,
        dtype: torch.dtype,
        eps: float = 1e-6,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = dtype
        self.eps = eps
        self.dispatch_kernel(kernel_map)
        # O3: kernel constructor takes no `tune` argument.
        self.kernel = self.kernel_map["engram_gate_conv_fwd"](
            M, seq_len, d, eps, dtype,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"engram_gate_conv_fwd": EngramGateConvFwdKernel}

    def forward(
        self,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            H: (M, seq_len, d) — hidden states from preceding layers.
            k: (M, seq_len, d) — key projection output (E @ W_K).
            v: (M, seq_len, d) — value projection output (E @ W_V).
            rms_w_h: (d,) — RMSNorm weight for H and k.
            rms_w_v: (d,) — RMSNorm weight for gated value (conv input).
            conv_w: (4, d) — depthwise causal conv1d weights.

        Returns:
            List of [Y, vhat, alpha, rrms_h, rrms_k, rrms_v]:
              Y:      (M, seq_len, d) — output to add as residual to H.
              vhat:   (M, seq_len, d) — saved for backward.
              alpha:  (M, seq_len) — scalar gate, saved for backward.
              rrms_h: (M, seq_len) — RMSNorm reciprocal rms of H.
              rrms_k: (M, seq_len) — RMSNorm reciprocal rms of k.
              rrms_v: (M, seq_len) — RMSNorm reciprocal rms of v_hat.
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(H):
            raise ValueError(f"H must be a {backend.name} tensor, got device {H.device}")
        if H.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {H.dtype}")
        if H.shape[-1] != self.d:
            raise ValueError(
                f"Expected hidden dim {self.d}, got {H.shape[-1]}"
            )
        if conv_w.shape[0] != CONV_KERNEL_SIZE:
            raise ValueError(
                f"Expected conv kernel size {CONV_KERNEL_SIZE}, got {conv_w.shape[0]}"
            )

        H = H.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        return self.kernel(H, k, v, rms_w_h, rms_w_v, conv_w)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inlined from the GPU ``tileops.perf.formulas.engram_gate_conv_fwd_roofline``
        (the NPU project has no ``tileops.perf`` module).  ``d`` is used as-is
        (the Op does not pad; padding to ``d_padded`` is internal to the kernel
        and, like the GPU roofline, does not change the model FLOPs).
        """
        M = int(self.M)
        seq_len = int(self.seq_len)
        d = int(self.d)
        elem = self.dtype.itemsize

        flops = M * seq_len * (24 * d) + 20 * M * seq_len
        nbytes = (5 * M * seq_len * d) * elem + 4 * M * seq_len * 4 + 6 * d * elem
        return int(flops), int(nbytes)
