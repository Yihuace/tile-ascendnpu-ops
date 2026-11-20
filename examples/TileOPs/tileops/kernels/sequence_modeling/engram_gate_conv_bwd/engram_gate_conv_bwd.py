"""Engram GateConv backward kernel (post-projection fusion grads, NPU-adapted).

Given ``dY`` (gradient of output), the saved intermediates from forward
(``vhat, alpha, rrms_h, rrms_k, rrms_v``) and the forward inputs
(``H, k, v, rms_w_h, rms_w_v, conv_w``), computes:

    dH, dk, dv          — gradients w.r.t. hidden state / projection outputs
    drms_w_h, drms_w_v  — gradients w.r.t. RMSNorm weights
    dconv_w             — gradient w.r.t. conv weights

The caller is responsible for the projection backward (GEMM):

    dE   = dk @ W_K^T + dv @ W_V^T
    dW_K = E^T @ dk
    dW_V = E^T @ dv

Staged GPU design (kept as the reference for the NPU kernel component):

    Pass 1:  SiLU+residual bwd -> store d_silu to buffer
    Pass 1a: Reduce dconv_w from the buffered d_silu values
    Pass 1b: Conv transpose -> RMSNorm(v_hat) bwd -> d(v_hat)
    Pass 2:  Gate bwd -> RMSNorm(H) bwd, RMSNorm(k) bwd -> dH, dk, dv

Adaptation summary (GPU -> NPU):

  **Part A — TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_engram_gate_conv_bwd_kernel``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` (pattern B) and
    imported as-is, together with the ``_align_up`` helper and the
    ``ALIGNMENT`` / ``CONV_KERNEL_SIZE`` constants it closes over
    (``_align_up`` manually appended by the helper-completeness step).  It
    serves as the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 adaptations (decorator, grid/sync semantics,
    ``threads`` removal, padding strategy) are handled by the NPU kernel
    component during re-implementation.

  **Part B — custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7: ``custom_op("npub::engram_gate_conv_bwd")`` (was
        ``"top::engram_gate_conv_bwd"``).
    K8: ``autotune_configs`` / ``tune`` param removed; heuristic config
        selection only (``init_config(config)``).
    K9: ``threads`` removed from the wrapper signature, ``default_config``
        and the ``forward`` dispatch (GPU default was ``threads=128``).

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation whose inner callable takes ``(threads)``.  The dispatch
  below passes the NPU-shaped argument list (no ``threads``); it becomes
  callable once the NPU kernel component rewrites the factory for
  ``target="npuir"`` and drops ``threads`` from the callable signature (K3).
  Running ``forward`` before that rewrite fails — this is the expected
  scaffold state.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/engram/engram_bwd.py.
# ``_align_up`` / ``ALIGNMENT`` / ``CONV_KERNEL_SIZE`` are the
# helpers/constants the factory closes over (``_align_up`` manually appended
# by the extractor's helper-completeness step).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .engram_gate_conv_bwd_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .engram_gate_conv_bwd_kernel import (
    ALIGNMENT,
    CONV_KERNEL_SIZE,
    _align_up,
    _engram_gate_conv_bwd_kernel,
)
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .engram_gate_conv_bwd_kernel.perf_opt._engram_gate_conv_bwd_kernel import ALIGNMENT
# from .engram_gate_conv_bwd_kernel.perf_opt._engram_gate_conv_bwd_kernel import CONV_KERNEL_SIZE
# from .engram_gate_conv_bwd_kernel.perf_opt._engram_gate_conv_bwd_kernel import _align_up
# from .engram_gate_conv_bwd_kernel.perf_opt._engram_gate_conv_bwd_kernel import _engram_gate_conv_bwd_kernel

__all__ = ["EngramGateConvBwdKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::engram_gate_conv_bwd", mutates_args=())
def _engram_gate_conv_bwd_wrapped(
    M: int,
    seq_len: int,
    d: int,
    eps: float,
    dtype_str: str,
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
) -> list[torch.Tensor]:
    # NPU callable convention (K3): the re-implemented factory returns a
    # compiled NPU kernel that is invoked with the tensor arguments in one
    # step (no inner ``threads`` closure as in the GPU scaffold).
    results = _engram_gate_conv_bwd_kernel(
        M, seq_len, d, eps, dtype_str,
    )(dY, H, k, v, rms_w_h, rms_w_v, conv_w,
      vhat, alpha, rrms_h, rrms_k, rrms_v)
    return list(results)


@_engram_gate_conv_bwd_wrapped.register_fake
def _(M, seq_len, d, eps, dtype_str,
      dY, H, k, v, rms_w_h, rms_w_v, conv_w,
      vhat, alpha, rrms_h, rrms_k, rrms_v):
    d_padded = _align_up(d, ALIGNMENT)
    device = dY.device
    dt = dY.dtype
    # Contract (DESIGN section 3.4 / R-13): the NPU factory closure returns the
    # 6 public outputs only (dH, dk, dv, drms_w_h, drms_w_v, dconv_w); the
    # internal staging workspaces (dvhat_buf / ws_*) are allocated inside the
    # closure and are not part of the op contract.  The fake must mirror the
    # real return arity and order so fake/meta dispatch stays consistent.
    return [
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),       # dH
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),       # dk
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),       # dv
        torch.empty((d_padded,), dtype=torch.float32, device=device),       # drms_w_h
        torch.empty((d_padded,), dtype=torch.float32, device=device),       # drms_w_v
        torch.empty((CONV_KERNEL_SIZE, d_padded), dtype=torch.float32, device=device),  # dconv_w
    ]


# ---------------------------------------------------------------------------
# Kernel class (K5/K8/K9 adaptations)
# ---------------------------------------------------------------------------


class EngramGateConvBwdKernel(Kernel):
    """Engram GateConv backward kernel.

    Staged backward:
      Pass 1:  SiLU bwd -> d_silu buffer
      Pass 1a: dconv_w reduction
      Pass 1b: Conv transpose + RMSNorm(vhat) bwd -> d(vhat)
      Pass 2:  Gate bwd -> dH, dk, dv + drms_w_h accumulation

    Inputs:  dY (M,T,d), H (M,T,d), k (M,T,d), v (M,T,d), rms_w_h, rms_w_v,
             conv_w, vhat (M,T,d), alpha (M,T), rrms_h/k/v (M,T)
    Outputs: dH (M,T,d), dk (M,T,d), dv (M,T,d), drms_w_h (d,),
             drms_w_v (d,), dconv_w (4,d)  — the last three fp32.

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures; GPU pinned CUDA
          SM ints).
      K8: autotune removed — heuristic config selection only;
          ``init_config(config)`` takes no ``tune`` argument.
      K9: ``threads`` removed from ``default_config`` and the ``forward``
          dispatch (GPU default was ``threads=128``).

    Args:
        M: batch size.
        seq_len: sequence length.
        d: model hidden dimension.
        eps: RMSNorm epsilon.
        dtype: data type.
        config: Optional kernel configuration dict.
        device_index: Device index (accepted for API parity with other NPU
            kernels; this op queries no device memory budget).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form fused vector kernel; invocations are scoped to
    # Developer mode regardless of the ambient TILELANG_ASCEND_MODE.
    ascend_mode = "Developer"

    def __init__(
        self,
        M: int,
        seq_len: int,
        d: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ):
        super().__init__()
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.eps = eps
        self.dtype = dtype
        self.device_index = device_index
        self.d_padded = _align_up(d, ALIGNMENT)
        self.kernel = _engram_gate_conv_bwd_kernel(
            M, seq_len, d, eps, self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # K9: ``threads`` removed — GPU returned {"threads": 128}.
        return {}

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
    ) -> list[torch.Tensor]:
        """Run the fused gate + conv backward pass.

        Args:
            dY: Gradient of the output, shape ``(M, seq_len, d)``.
            H: Hidden states of shape ``(M, seq_len, d)``.
            k: Key projection of shape ``(M, seq_len, d)``.
            v: Value projection of shape ``(M, seq_len, d)``.
            rms_w_h: RMSNorm weight of shape ``(d,)`` for ``H`` and ``k``.
            rms_w_v: RMSNorm weight of shape ``(d,)`` for the gated value.
            conv_w: Depthwise conv weights of shape ``(4, d)``.
            vhat: Gated value saved by the forward pass, ``(M, seq_len, d)``.
            alpha: Scalar gate saved by the forward pass, ``(M, seq_len)``.
            rrms_h: Reciprocal RMS of ``H``, shape ``(M, seq_len)``.
            rrms_k: Reciprocal RMS of ``k``, shape ``(M, seq_len)``.
            rrms_v: Reciprocal RMS of ``vhat``, shape ``(M, seq_len)``.

        Returns:
            ``[dH, dk, dv, drms_w_h, drms_w_v, dconv_w]``, trimmed to ``d``.
            The staging buffers the prim_func writes are internal and are not
            returned.
        """
        pad = self.d_padded - self.d
        if pad:
            dY = F.pad(dY, (0, pad))
            H = F.pad(H, (0, pad))
            k = F.pad(k, (0, pad))
            v = F.pad(v, (0, pad))
            vhat = F.pad(vhat, (0, pad))
            rms_w_h = F.pad(rms_w_h, (0, pad))
            rms_w_v = F.pad(rms_w_v, (0, pad))
            conv_w = F.pad(conv_w, (0, pad))
        # K9: no `threads` in the dispatch.
        results = _engram_gate_conv_bwd_wrapped(
            self.M, self.seq_len, self.d, self.eps, self.dtype_str,
            dY, H, k, v, rms_w_h, rms_w_v, conv_w,
            vhat, alpha, rrms_h, rrms_k, rrms_v,
        )
        dH, dk, dv, drms_w_h, drms_w_v, dconv_w = results[:6]
        if pad:
            dH = dH[:, :, : self.d]
            dk = dk[:, :, : self.d]
            dv = dv[:, :, : self.d]
            drms_w_h = drms_w_h[: self.d]
            drms_w_v = drms_w_v[: self.d]
            dconv_w = dconv_w[:, : self.d]
        return [dH, dk, dv, drms_w_h, drms_w_v, dconv_w]
