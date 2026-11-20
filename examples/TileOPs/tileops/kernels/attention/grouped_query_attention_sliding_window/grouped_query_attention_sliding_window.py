"""Grouped-query sliding-window attention forward kernel (NPU-adapted wrapper + Kernel class).

GQA forward with sliding-window attention over BSHD-layout q/k/v tensors
(flash-attention style online softmax):

    o = softmax(scale * q @ k^T + sliding-window mask) @ v   (shape (B, S, H, D))
    lse = per-row log-sum-exp                                 (shape (B, H, S))

The GPU op is ``GroupedQueryAttentionSlidingWindowFwdOp`` (tileops/ops/attention/gqa.py),
which dispatches ``GQASlidingWindowFwdKernel`` (fixed-length, architecture-generic)
or ``GQASlidingWindowFwdWgmmaPipelinedKernel`` (Hopper warp-specialized) via
``default_kernel_map``.  The NPU port keeps the architecture-generic kernel class.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factories (``_gqa_sw_fwd_kernel`` and the Hopper
    ``_gqa_sw_fwd_wgmma_pipelined_kernel``) are extracted from the GPU repo
    into ``_grouped_query_attention_sliding_window_fwd_kernels.py`` and
    imported as-is.  They serve as the reference for the NPU kernel component
    to reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync semantics, ``threads`` removal, boundary-mask strategy) are
    handled by the NPU component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 89, 90]``).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``tune`` param -- removed; heuristic config
        selection only.
    K9: ``threads`` removed from the wrapper signature, ``default_config``
        and the ``forward`` call.

  **Scope note (K5)**: the Hopper warp-specialized variant
  (``GQASlidingWindowFwdWgmmaPipelinedKernel``) is an architecture
  specialization; NPU collapses to the architecture-generic
  ``GQASlidingWindowFwdKernel``.  Its extracted factory is kept in the
  extracted file for reference but is not wired here.
"""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: the GPU TileLang kernel factories extracted by extract_tl_kernel.py
# (single output file ``_grouped_query_attention_sliding_window_fwd_kernels.py``,
# shared by all per-kernel migration prompts for this op).  They are the GPU
# (CUDA-target) implementations imported as-is -- the reference the NPU kernel
# component rewrites for ``target="npuir"``.
# Stage 5 integration scope: only the architecture-generic factory
# ``_gqa_sw_fwd_kernel`` is migrated; the Hopper warp-specialized variant
# (``_gqa_sw_fwd_wgmma_pipelined_kernel``) collapses away on NPU (see the
# scope note above) and stays only in the extracted reference file.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .grouped_query_attention_sliding_window_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .grouped_query_attention_sliding_window_kernel import _gqa_sw_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .grouped_query_attention_sliding_window_kernel.perf_opt._gqa_sw_fwd_kernel import _gqa_sw_fwd_kernel

__all__ = [
    "GQASlidingWindowFwdKernel",
    # Extracted TileLang kernel factory re-exported for the integration stage.
    "_gqa_sw_fwd_kernel",
]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::gqa_sw_fwd_wrapped_kernel", mutates_args=())
def _gqa_sw_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # K3/K9: the factory callable is invoked without ``threads`` (the CUDA
    # execution detail is dropped on NPU); the NPU kernel component removes
    # the ``threads`` parameter from the extracted GPU factory when rewriting
    # it for ``target="npuir"``.
    return _gqa_sw_fwd_kernel(
        batch,
        heads,
        heads_kv,
        seq_len,
        dim,
        is_causal,
        window_size_left,
        window_size_right,
        dtype,
    )(block_m, block_n, num_stages)(q, k, v)


@_gqa_sw_fwd_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    *inputs: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class GQASlidingWindowFwdKernel(Kernel):
    """Fixed-length GQA forward with sliding-window attention (BSHD, online softmax).

    Computes online-softmax attention with a running max / rescaled accumulator
    over KV tiles, masked by a causal and/or sliding-window mask, emitting both
    the attention output and the per-row log-sum-exp.  The KV loop range is
    trimmed to the window (``k_start`` / ``k_end``).

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures; was ``[80, 89, 90]``).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        heads_kv: Number of key/value heads (``heads % heads_kv == 0``).
        seq_len: Sequence length (same for Q, K, V).
        dim: Head dimension.
        is_causal: Whether to apply a causal mask.
        window_size_left: Left window size (``-1`` = unlimited).
        window_size_right: Right window size (``-1`` = unlimited).
        dtype: Data type (float16 or bfloat16).
        config: Optional dict with "block_m", "block_n", "num_stages".

    Raises:
        ValueError: If ``heads % heads_kv != 0``.
    """

    ascend_mode = "Expert"

    # K5: [80, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        window_size_left: int,
        window_size_right: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype

        # Build the factory callable (no TileLang JIT compilation yet --
        # compilation happens when the returned func is invoked inside the
        # custom_op wrapper).  The imported function is the GPU implementation;
        # the NPU component rewrites it for ``target="npuir"``.
        self.kernel = _gqa_sw_fwd_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
            self.dim,
            self.is_causal,
            self.window_size_left,
            self.window_size_right,
            self.dtype_str,
        )

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Default tiling config (GPU defaults minus the CUDA ``threads``).

        K9: ``threads`` (CUDA execution detail) dropped from the config dict;
        the tiling knobs ``block_m`` / ``block_n`` / ``num_stages`` are
        preserved from the GPU defaults.
        """
        return {
            "block_m": 64,
            "block_n": 64 if self.dim <= 128 else 32,
            "num_stages": 1,
        }

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the GQA sliding-window forward kernel.

        Accepts BSHD-layout q ``(B, S, H, D)`` and k/v ``(B, S, H_kv, D)``.
        Returns ``(output, lse)``: ``output`` has the q shape and ``lse`` has
        shape ``[batch, heads, seq_len]`` in float32.

        K9: ``threads`` removed from the call.
        """
        return _gqa_sw_fwd_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
            self.dim,
            self.is_causal,
            self.window_size_left,
            self.window_size_right,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            q,
            k,
            v,
        )
