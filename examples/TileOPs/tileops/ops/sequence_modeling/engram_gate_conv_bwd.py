"""Engram GateConv backward operator — post-projection fusion grads (NPU-adapted).

Given ``dY`` and the saved intermediates from forward (``vhat, alpha, rrms_*``),
computes gradients for ``H, k, v, rms_w_h, rms_w_v, conv_w``.

The caller is responsible for the projection backward (GEMM):

    dE   = dk @ W_K^T + dv @ W_V^T
    dW_K = E^T @ dk
    dW_V = E^T @ dv

Adaptation from GPU (TileOPs) to NPU:

- ``forward``: ``dY.is_cuda`` device check -> ``backend.is_device_tensor(dY)``
  via :func:`get_device_backend` (O1/O2).
- ``tune`` parameter removed (O3); the kernel constructor is called without it.
- ``eval_roofline``: the GPU
  ``tileops.perf.formulas.engram_gate_conv_bwd_roofline`` re-implemented inline
  (the NPU project has no ``tileops.perf`` module; roofline arithmetic is
  device-agnostic) (C24).
- Forward flow (device/dtype validation -> contiguous -> kernel dispatch)
  preserved unchanged (O6).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sequence_modeling.engram_gate_conv_bwd import EngramGateConvBwdKernel
from tileops.ops.op_base import Op

__all__ = ["EngramGateConvBwdOp"]


class EngramGateConvBwdOp(Op):
    """Engram GateConv backward operator.

    Given dY and saved intermediates from forward (vhat, alpha, rrms_*),
    computes gradients for H, k, v, rms_w_h, rms_w_v, conv_w.

    The caller is responsible for the projection backward (GEMM):
        dE  = dk @ W_K^T + dv @ W_V^T
        dW_K = E^T @ dk
        dW_V = E^T @ dv

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
        self.kernel = self.kernel_map["engram_gate_conv_bwd"](
            M, seq_len, d, eps, dtype,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"engram_gate_conv_bwd": EngramGateConvBwdKernel}

    def forward(
        self,
        dY: torch.Tensor,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
        vhat: torch.Tensor,
        alpha: torch.Tensor,
        rrms_h: torch.Tensor,
        rrms_k: torch.Tensor,
        rrms_v: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            dY:     (M, T, d) — gradient of output Y.
            H:      (M, T, d) — hidden states (forward input).
            k:      (M, T, d) — key projection (forward input, or recomputed).
            v:      (M, T, d) — value projection (forward input).
            rms_w_h: (d,) — RMSNorm weight for H and k.
            rms_w_v: (d,) — RMSNorm weight for v_hat.
            conv_w:  (4, d) — conv weights.
            vhat:   (M, T, d) — saved from forward.
            alpha:  (M, T) — saved from forward.
            rrms_h: (M, T) — saved from forward.
            rrms_k: (M, T) — saved from forward.
            rrms_v: (M, T) — saved from forward.

        Returns:
            List of [dH, dk, dv, drms_w_h, drms_w_v, dconv_w]:
              dH:       (M, T, d)
              dk:       (M, T, d)
              dv:       (M, T, d)
              drms_w_h: (d,) — fp32
              drms_w_v: (d,) — fp32
              dconv_w:  (4, d) — fp32
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(dY):
            raise ValueError(f"dY must be a {backend.name} tensor, got device {dY.device}")
        if dY.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {dY.dtype}")

        dY = dY.contiguous()
        H = H.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        vhat = vhat.contiguous()

        return self.kernel(
            dY, H, k, v, rms_w_h, rms_w_v, conv_w,
            vhat, alpha, rrms_h, rrms_k, rrms_v,
        )

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inlined from the GPU ``tileops.perf.formulas.engram_gate_conv_bwd_roofline``
        (the NPU project has no ``tileops.perf`` module).  ``d`` is used as-is
        (the Op does not pad; padding to ``d_padded`` is internal to the kernel
        and, like the GPU roofline, does not change the model FLOPs).
        """
        M = int(self.M)
        seq_len = int(self.seq_len)
        d = int(self.d)
        elem = self.dtype.itemsize

        fwd_flops = M * seq_len * (24 * d) + 20 * M * seq_len
        read_bytes = 5 * M * seq_len * d * elem + 6 * d * elem + 4 * M * seq_len * 4
        write_bytes = 3 * M * seq_len * d * elem + 10 * d * 4 + M * seq_len * d * 4
        return int(fwd_flops * 2.5), int(read_bytes + write_bytes)
