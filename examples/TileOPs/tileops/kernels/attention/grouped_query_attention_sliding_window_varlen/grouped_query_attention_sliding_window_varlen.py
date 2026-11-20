"""Variable-length grouped-query sliding-window attention forward kernel (NPU-adapted wrapper + Kernel class).

GQA forward with sliding-window attention over packed (no padding) q/k/v
tensors (flash-attention style online softmax).  Per-sample boundaries are
given by ``cu_seqlens_q`` / ``cu_seqlens_k``; ``seqlen_q`` and ``seqlen_k``
may differ per sample (``offset = seqlen_k - seqlen_q``, FA3 bottom-right
causal alignment):

    o = softmax(scale * q @ k^T + causal/sliding-window mask) @ v
                                        (shape (total_q, H, D))
    lse = per-row log-sum-exp           (shape (H, total_q))

The GPU op is ``GroupedQueryAttentionSlidingWindowVarlenFwdOp``
(tileops/ops/attention/gqa.py), which dispatches
``GQASlidingWindowVarlenFwdKernel`` (architecture-generic) or
``GQASlidingWindowVarlenFwdWgmmaPipelinedKernel`` (Hopper warp-specialized)
via ``default_kernel_map``.  The NPU port keeps the architecture-generic
kernel class.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factories (``_gqa_sw_fwd_varlen_kernel`` and the
    Hopper ``_gqa_sw_fwd_varlen_wgmma_pipelined_kernel``) are extracted from
    the GPU repo into
    ``_grouped_query_attention_sliding_window_varlen_fwd_kernels.py`` and
    imported as-is.  They serve as the reference for the NPU kernel component
    to reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync semantics, ``threads`` removal, boundary-mask strategy) are
    handled by the NPU component during re-implementation.  The referenced
    helpers (``LOG2E`` / ``make_log2e_scale`` /
    ``make_online_softmax_with_mask_guard`` / ``make_rescale`` /
    ``_make_apply_mask``) are appended to that same file so the package is
    self-contained.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 89, 90]``).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``tune`` param -- removed; heuristic config
        selection only.
    K9: ``threads`` removed from the wrapper signature, ``default_config``
        and the ``forward`` call.

  **Scope note (K5)**: the Hopper warp-specialized variant
  (``GQASlidingWindowVarlenFwdWgmmaPipelinedKernel``) is an architecture
  specialization; NPU collapses to the architecture-generic
  ``GQASlidingWindowVarlenFwdKernel``.  Its extracted factory is kept in the
  extracted file for reference but is not wired here.
"""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: the GPU TileLang kernel factories extracted by extract_tl_kernel.py
# (single output file
# ``_grouped_query_attention_sliding_window_varlen_fwd_kernels.py``, shared by
# all per-kernel migration prompts for this op).  They are the GPU
# (CUDA-target) implementations imported as-is -- the reference the NPU kernel
# component rewrites for ``target="npuir"``.
# Stage 5 integration scope: only the architecture-generic factory
# ``_gqa_sw_fwd_varlen_kernel`` is migrated; the Hopper warp-specialized
# variant (``_gqa_sw_fwd_varlen_wgmma_pipelined_kernel``) collapses away on NPU
# (see the scope note above) and stays only in the extracted reference file.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .grouped_query_attention_sliding_window_varlen_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .grouped_query_attention_sliding_window_varlen_kernel import _gqa_sw_fwd_varlen_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .grouped_query_attention_sliding_window_varlen_kernel.perf_opt._gqa_sw_fwd_varlen_kernel import _gqa_sw_fwd_varlen_kernel

__all__ = [
    "GQASlidingWindowVarlenFwdKernel",
    # Extracted TileLang kernel factory re-exported for the integration stage.
    "_gqa_sw_fwd_varlen_kernel",
]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::gqa_sw_fwd_varlen_wrapped_kernel", mutates_args=())
def _gqa_sw_fwd_varlen_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_k: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: str,
    accum_dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    max_seqlen_q: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # K3/K9: the factory callable is invoked without ``threads`` (the CUDA
    # execution detail is dropped on NPU); the NPU kernel component removes
    # the ``threads`` parameter from the extracted GPU factory when rewriting
    # it for ``target="npuir"``.
    return _gqa_sw_fwd_varlen_kernel(
        batch,
        heads,
        heads_kv,
        total_q,
        total_k,
        dim,
        is_causal,
        window_size_left,
        window_size_right,
        dtype,
        accum_dtype,
    )(block_m, block_n, num_stages)(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q
    )


@_gqa_sw_fwd_varlen_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_k: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: str,
    accum_dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    max_seqlen_q: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty([total_q, heads, dim], dtype=q.dtype, device=q.device)
    fake_lse = fake_o.new_empty([heads, total_q])
    return fake_o, fake_lse


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class _GQASlidingWindowVarlenFwdKernelBase(Kernel):
    """Shared base for variable-length GQA sliding-window forward kernels.

    NPU adaptations:

    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and the wrapped call.

    Args:
        batch: Number of sequences in the batch.
        heads: Number of query heads.
        heads_kv: Number of key/value heads (``heads % heads_kv == 0``).
        dim: Head dimension.
        is_causal: Whether to apply a causal mask.
        window_size_left: Left window size (``-1`` = unlimited).
        window_size_right: Right window size (``-1`` = unlimited).
        dtype: Data type (float16 or bfloat16).
        accum_dtype: Accumulator data type for intermediate computations.
        config: Optional dict with "block_m", "block_n", "num_stages".

    Raises:
        ValueError: If ``heads % heads_kv != 0``.
    """

    # Programming mode of the wrapped integrated kernel. Declared on the base
    # class so the op's Kernel hierarchy explicitly carries the mode (the
    # integrated kernel uses Expert mode). Value matches the derived class.
    ascend_mode = "Expert"

    # msprof ``--kernel-name`` filter: the device kernel is named after the
    # ``@T.prim_func`` in the integrated kernel module.  The benchmark
    # harness's generic derivation cannot resolve it for this op (several
    # candidate prim_funcs are reachable from the op object graph, so it
    # returns None), and an unfiltered msprof capture then reports an
    # auxiliary ACL kernel -- e.g. 3-4 us for an 8192-token workload, which
    # is physically impossible.  Declaring the name pins the filter to the
    # GQA varlen device kernel, mirroring the fixed-length sibling (whose
    # name the derivation does resolve).
    msprof_kernel_name = "_gqa_sw_fwd_varlen_main"

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        # total_q / total_k are not known at init; resolved per forward() call.
        self.init_config(config)

    @property
    def _accum_dtype_str(self) -> str:
        return "float" if self.accum_dtype == torch.float32 else "double"

    def _call_wrapped(
        self,
        wrapped_fn,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_q, total_k = q.shape[0], k.shape[0]
        # K9: ``threads`` is not passed (CUDA execution detail dropped on NPU).
        return wrapped_fn(
            self.batch,
            self.heads,
            self.heads_kv,
            total_q,
            total_k,
            self.dim,
            self.is_causal,
            self.window_size_left,
            self.window_size_right,
            self.dtype_str,
            self._accum_dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            max_seqlen_q,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
        )


class GQASlidingWindowVarlenFwdKernel(_GQASlidingWindowVarlenFwdKernelBase):
    """Variable-length GQA sliding window forward kernel (architecture-generic).

    Computes online-softmax attention with a running max / rescaled accumulator
    over KV tiles, masked by a causal and/or sliding-window mask plus
    per-sample Q/K out-of-bounds guards, emitting both the attention output and
    the per-row log-sum-exp.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures; was ``[80, 89, 90]``).
    - K9: ``threads`` removed from ``default_config``.
    """

    ascend_mode = "Expert"

    # K5: [80, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

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
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the variable-length GQA sliding-window forward kernel.

        Accepts packed q ``(total_q, H, D)`` and k/v ``(total_k, H_kv, D)``.
        Returns ``(output, lse)``: ``output`` has the q shape and ``lse`` has
        shape ``[heads, total_q]`` in float32.
        """
        return self._call_wrapped(
            _gqa_sw_fwd_varlen_wrapped_kernel,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
        )
