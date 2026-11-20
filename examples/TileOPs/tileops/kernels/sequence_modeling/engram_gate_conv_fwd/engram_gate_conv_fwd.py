"""Engram GateConv forward kernel (post-projection fusion, NPU-adapted).

Assumes projection (k = E @ W_K, v = E @ W_V) is done externally via GEMM.
This kernel fuses the remaining memory-bound operations:

    h_norm = RMSNorm(H, w_h)                                   # (M, T, d)
    k_norm = RMSNorm(k, w_h)                                   # (M, T, d)
    alpha  = sigmoid(dot(h_norm, k_norm) / sqrt(d))             # (M, T)
    v_hat  = alpha * v                                          # (M, T, d)
    v_hat_norm = RMSNorm(v_hat, w_v)                            # (M, T, d)
    conv_out = CausalDWConv1D(v_hat_norm, conv_w, kernel=4)     # (M, T, d)
    Y = SiLU(conv_out) + v_hat                                  # (M, T, d)

Saved for backward (strategy B):
    alpha, rrms_h, rrms_k, rrms_v  — all (M, T) scalars per position

Grid: (T, M) — one thread block per (batch, position).

Adaptation summary (GPU -> NPU):

  **Part A — TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_engram_gate_conv_fwd_kernel``) is
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
    K7: ``custom_op("npub::engram_gate_conv_fwd")`` (was
        ``"top::engram_gate_conv_fwd"``).
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
# (pattern B) from {gpu_repo_root}/tileops/kernels/engram/engram_fwd.py.
# ``_align_up`` / ``ALIGNMENT`` are the helpers/constants the factory closes
# over (``_align_up`` manually appended by the extractor's helper-completeness
# step).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .engram_gate_conv_fwd_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .engram_gate_conv_fwd_kernel import ALIGNMENT, _align_up, _engram_gate_conv_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .engram_gate_conv_fwd_kernel.perf_opt._engram_gate_conv_fwd_kernel import ALIGNMENT
from .engram_gate_conv_fwd_kernel.perf_opt._engram_gate_conv_fwd_kernel import _align_up
from .engram_gate_conv_fwd_kernel.perf_opt._engram_gate_conv_fwd_kernel import _engram_gate_conv_fwd_kernel

__all__ = ["EngramGateConvFwdKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::engram_gate_conv_fwd", mutates_args=())
def _engram_gate_conv_fwd_wrapped(
    M: int,
    seq_len: int,
    d: int,
    eps: float,
    dtype_str: str,
    H: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
) -> list[torch.Tensor]:
    # NPU callable convention (K3): the re-implemented factory returns a
    # compiled NPU kernel that is invoked with the tensor arguments in one
    # step (no inner ``threads`` closure as in the GPU scaffold).
    results = _engram_gate_conv_fwd_kernel(
        M, seq_len, d, eps, dtype_str,
    )(H, k, v, rms_w_h, rms_w_v, conv_w)
    return list(results)


@_engram_gate_conv_fwd_wrapped.register_fake
def _(M, seq_len, d, eps, dtype_str, H, k, v, rms_w_h, rms_w_v, conv_w):
    d_padded = _align_up(d, ALIGNMENT)
    device = H.device
    dt = H.dtype
    return [
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),     # Y
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),     # vhat_buf
        torch.empty((M, seq_len), dtype=torch.float32, device=device),    # alpha
        torch.empty((M, seq_len), dtype=torch.float32, device=device),    # rrms_h
        torch.empty((M, seq_len), dtype=torch.float32, device=device),    # rrms_k
        torch.empty((M, seq_len), dtype=torch.float32, device=device),    # rrms_v
    ]


# ---------------------------------------------------------------------------
# Kernel class (K5/K8/K9 adaptations)
# ---------------------------------------------------------------------------


class EngramGateConvFwdKernel(Kernel):
    """Engram GateConv forward kernel.

    Fuses RMSNorm-based scalar gating and depthwise causal Conv1D
    with SiLU activation. Projection (GEMM) is done externally.

    Inputs:  H (M,T,d), k (M,T,d), v (M,T,d), rms_w_h, rms_w_v, conv_w
    Outputs: Y (M,T,d), vhat (M,T,d), alpha (M,T), rrms_h/k/v (M,T)

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
        self.kernel = _engram_gate_conv_fwd_kernel(
            M, seq_len, d, eps, self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # K9: ``threads`` removed — GPU returned {"threads": 128}.
        return {}

    def forward(
        self,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run the fused gate + conv forward pass.

        Args:
            H: Hidden states of shape ``(M, seq_len, d)``.
            k: Key projection of shape ``(M, seq_len, d)``.
            v: Value projection of shape ``(M, seq_len, d)``.
            rms_w_h: RMSNorm weight of shape ``(d,)`` for ``H`` and ``k``.
            rms_w_v: RMSNorm weight of shape ``(d,)`` for the gated value.
            conv_w: Depthwise conv weights of shape ``(4, d)``.

        Returns:
            ``[Y, vhat, alpha, rrms_h, rrms_k, rrms_v]``. ``Y`` and ``vhat``
            are ``(M, seq_len, d)``; the alignment padding the prim_func
            requires is applied and trimmed here.
        """
        pad = self.d_padded - self.d
        if pad:
            H = F.pad(H, (0, pad))
            k = F.pad(k, (0, pad))
            v = F.pad(v, (0, pad))
            rms_w_h = F.pad(rms_w_h, (0, pad))
            rms_w_v = F.pad(rms_w_v, (0, pad))
            conv_w = F.pad(conv_w, (0, pad))
        # K9: no `threads` in the dispatch.
        results = _engram_gate_conv_fwd_wrapped(
            self.M, self.seq_len, self.d, self.eps, self.dtype_str,
            H, k, v, rms_w_h, rms_w_v, conv_w,
        )
        if pad:
            results[0] = results[0][:, :, : self.d]
            results[1] = results[1][:, :, : self.d]
        return results
