"""Paged MHA decode kernel (NPU-adapted wrapper + Kernel class).

Multi-head attention *decode* with a paged (block-table addressed) KV cache and
dynamic per-batch KV lengths.  Layout: ``Q [B, S_q, H, D]`` (BSHD); ``K`` / ``V``
are the *physical* cache ``[N_kv, H, D]``; ``real_seqlen_kv [B]`` (int32) gives
the logical KV length per batch; ``block_table [B, num_pages]`` (int32) maps
logical pages to physical pages.  ``O`` has the q shape ``[B, S_q, H, D]``.

The GPU kernel class (``MHADecodePagedKernel``) builds **two** TileLang
factories and dispatches between them at ``forward`` time:

* ``_mha_decode_no_split_kernel`` -- single-pass paged decode over all valid KV
  blocks (used when the batch's KV length is shorter than ``num_split *
  block_N``).
* ``_mha_decode_split_kernel`` -- split-K style decode: per-split partial
  outputs + per-split log-sum-exp, then a combine pass (used for long KV).

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    Both GPU TileLang factories (``_mha_decode_no_split_kernel`` /
    ``_mha_decode_split_kernel``) are extracted from the GPU repo via
    ``extract_tl_kernel.py`` and imported as-is; they are the reference for the
    NPU kernel component to reimplement for ``target="npuir"``.  K1-K4
    adaptations (decorator/target, grid/sync semantics, ``threads`` removal,
    boundary-mask strategy) are handled by the NPU component during
    re-implementation.  The online-softmax helpers the factories call
    (``make_log2e_scale`` / ``make_online_softmax`` / ``make_rescale`` + ``LOG2E``)
    are appended verbatim into the extracted file so the ``mha_decode_paged/``
    package is self-contained (SKILL.md S3 "Helper function extraction").

  **Part B -- custom_op wrappers + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 89, 90]``).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``); the
        ``mha_decode_paged_no_split_op`` / ``mha_decode_paged_split_op`` names are
        kept so paged and non-paged MHA decode do not collide in the registry.
    K8: ``autotune_configs`` / ``autotune()`` / ``_make_supply_prog`` /
        ``autotune_supply_prog`` / ``tune`` param -- removed; heuristic
        ``default_config`` selection only.
    K9: ``threads`` removed from the wrapper signatures, ``default_config`` and
        the ``forward`` dispatch (CUDA execution detail with no NPU analogue;
        the extracted GPU factories still carry ``threads`` in their callable
        signature -- the NPU component drops it when rewriting for NPUIR).

  **Op-specific flow preserved (O6)**: the two-factory construction, the
    no-split-vs-split threshold dispatch, and the split bookkeeping
    (``split_length`` / ``acc_split_length`` / ``glse`` / ``Output_partial``,
    all allocated on ``Q.device``) are preserved verbatim from the GPU kernel.

Pre-Stage-3 note: the imported factories are still GPU (CUDA target)
implementations whose inner callables take ``(..., threads)``.  The dispatch
below already passes the NPU-shaped argument list (no ``threads``); it becomes
callable once the NPU kernel component rewrites the factories for
``target="npuir"``.  Running ``forward`` before that rewrite fails -- this is
the expected scaffold state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: the two GPU TileLang factories extracted by extract_tl_kernel.py
# (single output file ``_multi_head_attention_decode_paged_with_k_v_cache_fwd_kernels.py``,
# shared by all per-kernel migration prompts for this op).  They are the GPU
# (CUDA target) implementations imported as-is -- the reference the NPU kernel
# component rewrites for ``target="npuir"``.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .mha_decode_paged_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .mha_decode_paged_kernel import _mha_decode_no_split_kernel, _mha_decode_split_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .mha_decode_paged_kernel.perf_opt._mha_decode_no_split_kernel import _mha_decode_no_split_kernel
# from .mha_decode_paged_kernel.perf_opt._mha_decode_split_kernel import _mha_decode_split_kernel

__all__ = ["MHADecodePagedKernel"]


# ---------------------------------------------------------------------------
# custom_op wrappers (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::mha_decode_paged_no_split_op", mutates_args=())
def _mha_decode_paged_no_split_op(
        batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int, page_size: int,
        is_causal: bool, dtype: str, block_M: int, block_N: int, num_stages: int,
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor) -> torch.Tensor:
    # K3/K9: the factory callable is invoked without ``threads`` (the CUDA
    # execution detail is dropped on NPU); the NPU kernel component removes the
    # ``threads`` parameter from the extracted GPU factory when rewriting it for
    # ``target="npuir"``.
    return _mha_decode_no_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size,
                                       is_causal,
                                       dtype)(block_M, block_N,
                                              num_stages)(Q, K, V, real_seqlen_kv, block_table)


@_mha_decode_paged_no_split_op.register_fake
def _(batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int, page_size: int,
      is_causal: bool, dtype: str, block_M: int, block_N: int, num_stages: int,
      Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, real_seqlen_kv: torch.Tensor,
      block_table: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(Q)


@torch.library.custom_op("npub::mha_decode_paged_split_op", mutates_args=())
def _mha_decode_paged_split_op(
        batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int, page_size: int,
        is_causal: bool, dtype: str, block_M: int, block_N: int, num_stages: int,
        num_split: int, full_kv: bool, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        real_seqlen_kv: torch.Tensor, block_table: torch.Tensor, glse: torch.Tensor,
        Output_partial: torch.Tensor,
        acc_split_length: torch.Tensor) -> torch.Tensor:
    return _mha_decode_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal,
                                    dtype, full_kv)(block_M, block_N, num_split,
                                           num_stages)(Q, K, V, real_seqlen_kv, block_table, glse,
                                                       Output_partial, acc_split_length)


@_mha_decode_paged_split_op.register_fake
def _(batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int, page_size: int,
      is_causal: bool, dtype: str, block_M: int, block_N: int, num_stages: int,
      num_split: int, full_kv: bool, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
      real_seqlen_kv: torch.Tensor, block_table: torch.Tensor, glse: torch.Tensor,
      Output_partial: torch.Tensor, acc_split_length: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(Q)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class MHADecodePagedKernel(Kernel):
    """Paged MHA decode kernel with a dynamic (block-table) KV cache.

    Builds both decode variants up front (no-split + split/combine) and
    dispatches per call based on the batch's maximum real KV length.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures; was ``[80, 89, 90]``).
    - K8: autotune removed (``autotune_configs`` / ``autotune()`` /
      ``_make_supply_prog`` / ``autotune_supply_prog`` / ``tune`` param);
      heuristic config selection only.
    - K9: ``threads`` removed from ``default_config`` and the ``forward``
      dispatch.

    Args:
        batch: Batch size.
        heads: Number of heads (== number of KV heads for MHA).
        seqlen_q: Query sequence length.
        seqlen_kv: Physical KV cache length (``N_kv``, page-aligned).
        dim: Head dimension.
        page_size: Number of tokens per paged KV block.
        is_causal: Whether to apply a causal mask.
        dtype: Data type (float16 or bfloat16).
        config: Optional dict with "block_M", "block_N", "num_split",
            "num_stages".
    """

    ascend_mode = "Developer"

    # K5: [80, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        batch: int,
        heads: int,
        seqlen_q: int,
        seqlen_kv: int,
        dim: int,
        page_size: int,
        is_causal: bool,
        dtype: str = "bfloat16",
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.page_size = page_size

        # Build both factory callables (no TileLang JIT compilation yet --
        # compilation happens when the returned func is invoked inside the
        # custom_op wrapper).  The imported functions are the GPU
        # implementations; the NPU component rewrites them for
        # ``target="npuir"``.
        self.no_split_jit = _mha_decode_no_split_kernel(
            self.batch, self.heads, self.seqlen_q, self.seqlen_kv, self.dim, self.page_size,
            self.is_causal, self.dtype_str)
        self.split_jit = _mha_decode_split_kernel(
            self.batch, self.heads, self.seqlen_q, self.seqlen_kv, self.dim, self.page_size,
            self.is_causal, self.dtype_str)

        # The GPU autotune targeted the split kernel; with autotune removed the
        # class default remains the split kernel.
        self.kernel = self.split_jit
        # K8: ``_make_supply_prog`` / ``autotune_supply_prog`` removed with the
        # autotuner.
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Default tiling config (GPU defaults minus the CUDA ``threads``).

        K9: ``threads`` (CUDA execution detail) dropped from the config dict;
        the tiling knobs ``block_M`` / ``block_N`` / ``num_split`` /
        ``num_stages`` are preserved from the GPU defaults.
        """
        return {
            "block_M": 128,
            "block_N": 64 if self.dim <= 128 else 32,
            "num_split": 4,
            "num_stages": 2,
        }

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        """Run paged MHA decode; accepts physical K/V cache + block table.

        Dispatches to the no-split variant for short KV (``real_max <
        num_split * block_N``) and to the split/combine variant otherwise.
        K9: ``threads`` removed from the dispatch calls.
        """
        block_M = self.config["block_M"]
        block_N = self.config["block_N"]
        num_split = self.config["num_split"]
        num_stages = self.config["num_stages"]

        if self.seqlen_q == 1:
            # The vector decode handles both short and full caches. A full-cache
            # specialization removes masks after checking all batch lengths.
            full_kv = bool(real_seqlen_kv.min().item() >= self.seqlen_kv)
            packed_ns = min(num_split, max(1, 32 // (self.batch * self.heads)))
            use_packed = (full_kv and not self.is_causal and self.dtype == torch.float16
                          and self.seqlen_kv % self.page_size == 0
                          and self.seqlen_kv % packed_ns == 0
                          and (self.seqlen_kv // packed_ns) % 128 == 0
                          and (self.seqlen_kv // packed_ns) * self.dim <= 32768
                          and self.dim % 16 == 0)
            self.msprof_kernel_name = "decode_vector_packed" if use_packed else "split_partial"
            glse = torch.empty((self.batch, self.heads, num_split, 1),
                               dtype=torch.float32, device=Q.device)
            partial = torch.empty((self.batch, 1, self.heads, num_split, self.dim),
                                  dtype=torch.float32, device=Q.device)
            acc_lengths = torch.empty((self.batch, num_split), dtype=torch.int32,
                                      device=Q.device)
            return _mha_decode_paged_split_op(
                self.batch, self.heads, self.seqlen_q, self.seqlen_kv, self.dim,
                self.page_size, self.is_causal, self.dtype_str, block_M, block_N,
                num_stages, num_split, full_kv, Q, K, V, real_seqlen_kv,
                block_table, glse, partial, acc_lengths)

        # Dispatch: use no-split for short sequences where splitting is not beneficial
        real_max = real_seqlen_kv.max().item() if real_seqlen_kv.dim() > 0 else real_seqlen_kv.item()
        threshold = num_split * block_N
        if real_max < threshold:
            return _mha_decode_paged_no_split_op(
                self.batch, self.heads, self.seqlen_q, self.seqlen_kv, self.dim, self.page_size,
                self.is_causal, self.dtype_str, block_M, block_N, num_stages, Q, K, V,
                real_seqlen_kv, block_table)

        # Split path: compute cumulative per-split lengths
        chunk_size = real_max // (num_split * block_N) * block_N
        split_length = torch.full((self.batch, num_split), chunk_size, dtype=torch.int32,
                                  device=Q.device)
        split_length[:, -1] = int(real_max - (num_split - 1) * chunk_size)
        acc_split_length = torch.cumsum(split_length, dim=1).to(torch.int32)

        # S5 precision fix: the split/combine workspaces are fp32 (the segment
        # LSE feeds exponent-domain combine weights exp(l_i - LSE), so an io
        # dtype (bf16) workspace overflows the 1e-3 correctness atol).
        glse = torch.zeros((self.batch, self.heads, num_split, self.seqlen_q),
                           dtype=torch.float32,
                           device=Q.device).contiguous()
        Output_partial = torch.zeros(
            (self.batch, self.seqlen_q, self.heads, num_split, self.dim),
            dtype=torch.float32,
            device=Q.device).contiguous()

        return _mha_decode_paged_split_op(
            self.batch, self.heads, self.seqlen_q, self.seqlen_kv, self.dim, self.page_size,
            self.is_causal, self.dtype_str, block_M, block_N, num_stages, num_split, False, Q, K,
            V, real_seqlen_kv, block_table, glse, Output_partial, acc_split_length)
