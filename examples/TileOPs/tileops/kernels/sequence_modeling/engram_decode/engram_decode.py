"""Engram fused decode kernel — single-token inference (NPU-adapted).

In decode mode (seq_len=1), all operations are fused into one kernel per batch:

    k = e @ W_K                                          # GEMV: (d_mem,) -> (d,)
    v = e @ W_V                                          # GEMV: (d_mem,) -> (d,)
    h_norm = RMSNorm(h, w_h)
    k_norm = RMSNorm(k, w_h)
    alpha  = sigmoid(dot(h_norm, k_norm) / sqrt(d))
    v_hat  = alpha * v
    v_hat_norm = RMSNorm(v_hat, w_v)
    conv_state = shift_left(conv_state) | v_hat_norm     # update cache
    conv_out = dilated_conv(conv_w, conv_state, current) # depthwise dilated conv
    y = SiLU(conv_out) + v_hat

Grid: (B,) — one block per batch element.

Inputs:
    e_t:        (B, d_mem)              — gathered N-gram embedding for current token
    h_t:        (B, d)                  — hidden state for current token
    conv_state: (B, max_conv_len, d)    — cached v_hat_norm history (left-padded zeros)
    W_K, W_V:   (d_mem, d)              — projection weights
    rms_w_h:    (d,)                    — RMSNorm weight for h and k
    rms_w_v:    (d,)                    — RMSNorm weight for v_hat
    conv_w:     (w, d)                  — depthwise conv weights (w = conv_kernel_size)

Outputs:
    y_t:            (B, d)              — output to add as residual to h_t
    new_conv_state: (B, max_conv_len, d)— updated conv state for next step

Adaptation summary (GPU -> NPU):

  **Part A — TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_engram_decode_kernel``) is extracted
    from the GPU repo via ``extract_tl_kernel.py`` (pattern B) and imported
    as-is, together with the ``_align_up`` helper and the ``ALIGNMENT``
    constant it closes over.  It serves as the reference for the NPU kernel
    component to reimplement for ``target="npuir"``.  K1-K4 adaptations
    (decorator, grid/sync semantics, ``threads`` removal, padding strategy)
    are handled by the NPU kernel component during re-implementation.

  **Part B — custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7: ``custom_op("npub::engram_decode")`` (was ``"top::engram_decode"``).
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
# (pattern B) from {gpu_repo_root}/tileops/kernels/engram/engram_decode.py.
# ``_align_up`` / ``ALIGNMENT`` are the helpers/constants the factory
# closes over (manually appended by the extractor's helper-completeness step).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .engram_decode_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .engram_decode_kernel import ALIGNMENT, _align_up, _engram_decode_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .engram_decode_kernel.perf_opt._engram_decode_kernel import ALIGNMENT
from .engram_decode_kernel.perf_opt._engram_decode_kernel import _align_up
from .engram_decode_kernel.perf_opt._engram_decode_kernel import _engram_decode_kernel

__all__ = ["EngramDecodeKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::engram_decode", mutates_args=())
def _engram_decode_wrapped(
    batch: int,
    d_mem: int,
    d: int,
    max_conv_len: int,
    conv_kernel_size: int,
    dilation: int,
    eps: float,
    dtype_str: str,
    e_t: torch.Tensor,
    h_t: torch.Tensor,
    conv_state: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
) -> list[torch.Tensor]:
    # NPU callable convention (K3): the Stage 3 factory is directly
    # ``@tilelang.jit``-decorated and returns a compiled ``JitKernel_NPU``
    # that is invoked with the tensor arguments in one step (no inner
    # ``threads`` closure as in the GPU scaffold).  Verified empirically:
    # ``type(_engram_decode_kernel(...)) == JitKernel_NPU`` (Stage 5).
    results = _engram_decode_kernel(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, eps, dtype_str,
    )(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)
    return list(results)


@_engram_decode_wrapped.register_fake
def _(
    batch: int,
    d_mem: int,
    d: int,
    max_conv_len: int,
    conv_kernel_size: int,
    dilation: int,
    eps: float,
    dtype_str: str,
    e_t: torch.Tensor,
    h_t: torch.Tensor,
    conv_state: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
):
    d_padded = _align_up(d, ALIGNMENT)
    device = e_t.device
    dt = e_t.dtype
    return [
        torch.empty((batch, d_padded), dtype=dt, device=device),
        torch.empty((batch, max_conv_len, d_padded), dtype=dt, device=device),
    ]


# ---------------------------------------------------------------------------
# Kernel class (K5/K8/K9 adaptations)
# ---------------------------------------------------------------------------


class EngramDecodeKernel(Kernel):
    """Engram fused decode kernel — full single-token pipeline.

    Fuses GEMV projection, RMSNorm gating, dilated causal conv with state
    update, and SiLU activation into a single kernel launch.

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures; GPU pinned CUDA
          SM ints).
      K8: autotune removed — heuristic config selection only;
          ``init_config(config)`` takes no ``tune`` argument.
      K9: ``threads`` removed from ``default_config`` and the ``forward``
          dispatch (GPU default was ``threads=128``).

    Args:
        batch: batch size.
        d_mem: memory embedding dimension.
        d: model hidden dimension.
        max_conv_len: max conv cache capacity (compile-time).
        conv_kernel_size: number of conv taps (w=4 in paper).
        dilation: dilation factor (δ = max N-gram order in paper).
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
        batch: int,
        d_mem: int,
        d: int,
        max_conv_len: int,
        conv_kernel_size: int,
        dilation: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ):
        super().__init__()
        self.batch = batch
        self.d_mem = d_mem
        self.d = d
        self.max_conv_len = max_conv_len
        self.conv_kernel_size = conv_kernel_size
        self.dilation = dilation
        self.eps = eps
        self.dtype = dtype
        self.device_index = device_index
        self.d_padded = _align_up(d, ALIGNMENT)

        min_cache = dilation * (conv_kernel_size - 1)
        if max_conv_len < min_cache:
            raise ValueError(
                f"max_conv_len ({max_conv_len}) must be >= "
                f"dilation * (conv_kernel_size - 1) = {min_cache}"
            )

        self.kernel = _engram_decode_kernel(
            batch, d_mem, d, max_conv_len, conv_kernel_size, dilation,
            eps, self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # K9: ``threads`` removed — GPU returned {"threads": 128}.
        return {}

    def forward(
        self,
        e_t: torch.Tensor,
        h_t: torch.Tensor,
        conv_state: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run one fused decode step.

        Args:
            e_t: Gathered N-gram embedding, shape ``(B, d_mem)``.
            h_t: Hidden state for the current token, shape ``(B, d)``.
            conv_state: Conv history of shape ``(B, L, d)`` with
                ``L <= max_conv_len``; left-padded to the compiled capacity
                here.
            W_K: Key projection weight of shape ``(d_mem, d)``.
            W_V: Value projection weight of shape ``(d_mem, d)``.
            rms_w_h: RMSNorm weight of shape ``(d,)`` for ``h`` and ``k``.
            rms_w_v: RMSNorm weight of shape ``(d,)`` for the gated value.
            conv_w: Depthwise conv weights of shape ``(w, d)``.

        Returns:
            ``[y_t, new_conv_state]`` of shapes ``(B, d)`` and
            ``(B, max_conv_len, d)``. The alignment padding the prim_func
            requires is applied and trimmed here.
        """
        state_pad = self.max_conv_len - conv_state.shape[1]
        if state_pad > 0:
            conv_state = F.pad(conv_state, (0, 0, state_pad, 0))
        pad = self.d_padded - self.d
        if pad:
            h_t = F.pad(h_t, (0, pad))
            conv_state = F.pad(conv_state, (0, pad))
            W_K = F.pad(W_K, (0, pad))
            W_V = F.pad(W_V, (0, pad))
            rms_w_h = F.pad(rms_w_h, (0, pad))
            rms_w_v = F.pad(rms_w_v, (0, pad))
            conv_w = F.pad(conv_w, (0, pad))
        # K9: no `threads` in the dispatch.
        results = _engram_decode_wrapped(
            self.batch, self.d_mem, self.d, self.max_conv_len,
            self.conv_kernel_size, self.dilation, self.eps,
            self.dtype_str,
            e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
        )
        if pad:
            results[0] = results[0][:, : self.d]
            results[1] = results[1][:, :, : self.d]
        return results
