"""FP8 lightning indexer kernel (NPU-adapted wrapper + Kernel class).

Lightning-indexer logits: per-query scores over an fp8-quantized index cache.
Layout: ``IndexQ`` ``[batch, seq_len, heads, index_dim]`` (viewed by the kernel
as ``[batch, seq_len * heads, index_dim]``); ``IndexK``
``[batch, seq_len_kv, kv_group, index_dim]``; ``IndexKScale``
``[batch, seq_len_kv, kv_group]`` (fp32); ``Weights`` ``[seq_len, heads]``
(fp32); ``CuSeqLenKS`` / ``CuSeqLenKE`` ``[seq_len]`` (int32); output
``Logits`` ``[batch, seq_len, seq_len_kv, kv_group]`` (fp32).

The GPU kernel class builds **two** TileLang factories:

* ``_fp8_lightning_indexer_kernel`` -- the fp8 score/index kernel (CUDA
  tensor-core path; ``dtype = T.float8_e4m3fn``, accum fp32).
* ``clean_logits_`` -- a small post-pass that writes ``-inf`` outside each
  query's ``[cu_seqlen_ks, cu_seqlen_ke)`` window (used when ``clean_logits``).

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    Both GPU TileLang factories (``_fp8_lightning_indexer_kernel`` /
    ``clean_logits_``) are extracted from the GPU repo via
    ``extract_tl_kernel.py`` (resolution pattern B) and imported as-is; they
    are the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 adaptations (decorator/target, grid/sync
    semantics, ``threads`` removal, boundary-mask strategy) are handled by the
    NPU component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[90]``).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``autotune()`` / ``supply_prog`` /
        ``autotune_supply_prog`` / ``tune`` param -- removed; heuristic
        ``default_config`` selection only.
    K9: ``threads`` removed from the wrapper signature, ``default_config``
        and the ``forward`` dispatch (CUDA execution detail with no NPU
        analogue; the extracted GPU factories still carry ``threads`` in
        their callable signature -- the NPU component drops it when rewriting
        for NPUIR).

  **FP8 path preserved (O6)**: the kernel validates
    ``dtype == torch.float8_e4m3fn`` (the fp8 tensor-core path is the only
    supported index dtype); the fp8 quantization itself lives in the Op layer
    (``FP8LightningIndexerOp.per_custom_dims_cast_to_fp8``).

Pre-Stage-3 note: the imported factories are still GPU (CUDA target)
implementations whose inner callables take ``threads``.  The dispatch below
already passes the NPU-shaped argument list (no ``threads``); it becomes
callable once the NPU kernel component rewrites the factories for
``target="npuir"``.  Running ``forward`` before that rewrite fails -- this is
the expected scaffold state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import scoped_ascend_mode

# Part A: the two GPU TileLang factories extracted by extract_tl_kernel.py
# (single output file ``_f_p8_lightning_indexer_kernels.py``, shared by all
# per-kernel migration prompts for this op).  They are the GPU (CUDA target)
# implementations imported as-is -- the reference the NPU kernel component
# rewrites for ``target="npuir"``.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .fp8_lightning_indexer_kernel/perf_opt/{func}.py and pass their L0/L1
# regression; the baseline (Stage 3) source is active otherwise.  ``pytest
# tests/ops/`` and ``pytest benchmarks/ops/`` dispatch through whichever source
# is active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .fp8_lightning_indexer_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .fp8_lightning_indexer_kernel import _fp8_lightning_indexer_kernel, clean_logits_
# _TUNED_BLOCK_N, _TUNED_BLOCK_Q, _TUNED_NUM_STAGES = 64, 4, 2
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .fp8_lightning_indexer_kernel.perf_opt._fp8_lightning_indexer_kernel import (
    _fp8_lightning_indexer_kernel,
    TUNED_BLOCK_N as _TUNED_BLOCK_N,
    TUNED_BLOCK_Q as _TUNED_BLOCK_Q,
    TUNED_NUM_STAGES as _TUNED_NUM_STAGES,
    EXPERT_BLOCK_Q,
    EXPERT_BLOCK_N,
    EXPERT_CORES,
    _fp8_lightning_indexer_expert,
    _fp8_indexer_normalize_qk,
)
from .fp8_lightning_indexer_kernel.perf_opt.clean_logits_ import clean_logits_

__all__ = ["FP8LightningIndexerKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::fp8_lightning_indexer_wrapped_kernel", mutates_args=("Logits",))
def fp8_lightning_indexer_wrapped_kernel(batch: int, seq_len: int, heads: int, index_dim: int,
                                        seq_len_kv: int, kv_group: int, clean_logits: bool,
                                        block_N: int, num_stages: int, block_Q: int,
                                        IndexQ: torch.Tensor, IndexK: torch.Tensor,
                                        IndexKScale: torch.Tensor, Logits: torch.Tensor,
                                        Weights: torch.Tensor, CuSeqLenKS: torch.Tensor,
                                        CuSeqLenKE: torch.Tensor) -> None:
    # K3/K9: the factory callables are invoked without ``threads`` (the CUDA
    # execution detail is dropped on NPU); the NPU kernel component removes the
    # ``threads`` parameter from the extracted GPU factories when rewriting them
    # for ``target="npuir"``.
    _fp8_lightning_indexer_kernel(batch, seq_len, heads, index_dim, seq_len_kv,
                                 kv_group)(block_N, num_stages,
                                           block_Q)(IndexQ.view(batch, seq_len * heads,
                                                                index_dim), IndexK, IndexKScale,
                                                    Logits, Weights, CuSeqLenKS, CuSeqLenKE)
    if clean_logits:
        clean_logits_()(Logits, CuSeqLenKS, CuSeqLenKE)


@fp8_lightning_indexer_wrapped_kernel.register_fake
def _(
        batch: int,
        seq_len: int,
        heads: int,
        index_dim: int,
        seq_len_kv: int,
        kv_group: int,
        clean_logits: bool,
        block_N: int,
        num_stages: int,
        block_Q: int,
        IndexQ: torch.Tensor,
        IndexK: torch.Tensor,
        IndexKScale: torch.Tensor,
        Logits: torch.Tensor,
        Weights: torch.Tensor,
        CuSeqLenKS: torch.Tensor,
        CuSeqLenKE: torch.Tensor) -> None:
    return None



@torch.library.custom_op(
    "npub::fp8_lightning_indexer_fused_kernel", mutates_args=("Logits", "Score", "QNorm", "KNorm", "WNorm", "QWScale", "KScaleNorm"))
def fp8_lightning_indexer_fused_kernel(
        batch: int, seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
        clean_logits: bool, IndexQ: torch.Tensor, IndexK: torch.Tensor,
        IndexKScale: torch.Tensor, Logits: torch.Tensor, Weights: torch.Tensor,
        CuSeqLenKS: torch.Tensor, CuSeqLenKE: torch.Tensor,
        Score: torch.Tensor, QNorm: torch.Tensor, KNorm: torch.Tensor,
        WNorm: torch.Tensor, QWScale: torch.Tensor, KScaleNorm: torch.Tensor) -> None:
    _fp8_indexer_normalize_qk(batch, seq_len, heads, index_dim, seq_len_kv)(
        IndexQ, IndexK, Weights, IndexKScale, QNorm, KNorm, WNorm, QWScale, KScaleNorm)
    _fp8_lightning_indexer_expert(
        batch, seq_len, heads, index_dim, seq_len_kv,
        EXPERT_BLOCK_Q, EXPERT_BLOCK_N, EXPERT_CORES, True, clean_logits)(
            QNorm.view(batch, seq_len * heads, index_dim), KNorm, KScaleNorm,
            Logits, WNorm, CuSeqLenKS.view(seq_len, 1),
            CuSeqLenKE.view(1, seq_len), Score, QWScale, Logits.view(batch, seq_len * seq_len_kv))


@fp8_lightning_indexer_fused_kernel.register_fake
def _(
        batch: int, seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
        clean_logits: bool, IndexQ: torch.Tensor, IndexK: torch.Tensor,
        IndexKScale: torch.Tensor, Logits: torch.Tensor, Weights: torch.Tensor,
        CuSeqLenKS: torch.Tensor, CuSeqLenKE: torch.Tensor,
        Score: torch.Tensor, QNorm: torch.Tensor, KNorm: torch.Tensor,
        WNorm: torch.Tensor, QWScale: torch.Tensor, KScaleNorm: torch.Tensor) -> None:
    return None


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class FP8LightningIndexerKernel(Kernel):
    """FP8 lightning indexer: per-query logits over an fp8-quantized index cache.

    Args:
        batch: Batch size.
        seq_len: Query sequence length.
        heads: Number of index heads.
        index_dim: Index head dimension.
        seq_len_kv: Key/value sequence length.
        kv_group: Number of key/value groups.
        clean_logits: Whether to zero the logits outside each query's
            ``[cu_seqlen_ks, cu_seqlen_ke)`` window.
        dtype: Torch dtype of the quantized index tensors. The tensor-core
            path this kernel emits accepts ``torch.float8_e4m3fn`` only; the
            weights, scales and logits are fixed at ``torch.float32``.
        config: Optional dict with "block_N", "num_stages" and "block_Q".

    Raises:
        ValueError: If *dtype* is not ``torch.float8_e4m3fn``.

    NPU adaptations:
        K5: ``supported_archs = None`` (all architectures; was ``[90]``).
        K8: autotune removed -- heuristic config selection only.
        K9: ``threads`` removed from ``default_config`` and ``forward``.
    """

    # K5: [90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Scaffold placeholder for the NPU programming mode; the NPU kernel
    # component owns the final value and ``integrate_kernel.py`` re-syncs it
    # from the verified Stage 3 product.  The indexer contains a Cube
    # ``T.gemm`` (fp8 score) plus a vector epilogue; the Developer-mode default
    # is used here.
    ascend_mode = "Developer"

    def __init__(self,
                 batch,
                 seq_len,
                 heads,
                 index_dim,
                 seq_len_kv,
                 kv_group,
                 clean_logits=True,
                 dtype: torch.dtype = torch.float8_e4m3fn,
                 config: Optional[dict] = None):
        super().__init__()
        if dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"FP8LightningIndexerKernel indexes float8_e4m3fn tensors, got {dtype}")
        self.dtype = dtype
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.clean_logits = clean_logits
        self.config = config

        # Build the factory callable (does not compile yet -- compilation
        # happens when the returned func is invoked inside the custom_op
        # wrapper).  The imported function is the GPU implementation; the NPU
        # component rewrites it for ``target="npuir"``.
        self.kernel = _fp8_lightning_indexer_kernel(self.batch, self.seq_len, self.heads,
                                                   self.index_dim, self.seq_len_kv, self.kv_group,
                                                   self.clean_logits)

        self._expert_shape = (
            kv_group == 1 and index_dim == 64 and heads in (8, 32)
            and seq_len % EXPERT_BLOCK_Q == 0
            and seq_len_kv % EXPERT_BLOCK_N == 0)
        self._score = None
        self._norm_buffers = None
        self.init_config(config)
        self._use_fused = (
            self._expert_shape
            and self.config["block_Q"] == EXPERT_BLOCK_Q
            and self.config["block_N"] == EXPERT_BLOCK_N
            and self.config["num_stages"] == 1)
        self.ascend_mode = "Expert" if self._use_fused else "Developer"
        self.msprof_kernel_name = (
            "fp8_indexer_distinctmeta_inline_mix_aic" if self._use_fused
            else "_fp8_lightning_indexer_main_mix_aic")

    @property
    def default_config(self) -> dict:
        """Default tiling config (GPU defaults minus the CUDA ``threads``).

        K9: ``threads`` (CUDA execution detail) dropped; the GPU default was
        ``{"block_N": 64, "num_stages": 2, "threads": 128, "block_Q": 4}``.

        NPU UB adaptation (K10): the grouped path (``kv_group > 1``) of the
        migrated kernel additionally stages the whole Q tile
        ``qall_ub = [block_Q * heads, index_dim]`` in UB (see
        ``_fp8_lightning_indexer_kernel``).  At ``heads=32, index_dim=64`` the
        bf16 tile for ``block_Q=4`` (8192 elements) overflows the 192 KB UB
        (``bishengir-compile``: requires 2326528 bits while 1572864 are
        available); ``block_Q=2`` (4096 elements) is UB-safe.  The single-group
        path stages ``q_ub`` only (its ``qall_ub`` collapses to one row ``[1,
        index_dim]``) and keeps the GPU-default ``block_Q=4``.  For the grouped
        path the tile is capped so ``block_Q * heads * index_dim <= 4096``
        elements (the verified UB-safe boundary), bounded to ``[1, 4]``.
        """
        if self._expert_shape:
            return {
                "block_N": EXPERT_BLOCK_N,
                "num_stages": 1,
                "block_Q": EXPERT_BLOCK_Q,
            }
        if self.kv_group > 1:
            block_Q = max(1, min(4, 4096 // (self.heads * self.index_dim)))
        else:
            # Stage 4 tuned default, paired with the switch block above.  The
            # single-group path stages ``q_ub`` only (``qall_ub`` collapses to
            # one row), so the tuned ``block_Q`` is UB-safe and halves the
            # kv-tile iteration count (`perf_records.jsonl` r1a winner).
            block_Q = _TUNED_BLOCK_Q
        return {
            "block_N": _TUNED_BLOCK_N,
            "num_stages": _TUNED_NUM_STAGES,
            "block_Q": block_Q,
        }

    def forward(
            self,
            IndexQ: torch.Tensor,  # type: ignore
            IndexK: torch.Tensor,  # type: ignore
            IndexKScale: torch.Tensor,  # type: ignore
            Weights: torch.Tensor,  # type: ignore
            CuSeqLenKS: torch.Tensor,  # type: ignore
            CuSeqLenKE: torch.Tensor,  # type: ignore
    ) -> torch.Tensor:
        """Run the fp8 lightning indexer; returns the fp32 logits.

        K9: ``threads`` removed from the dispatch (the custom_op wrapper no
        longer takes it).
        """
        Logits = torch.empty([self.batch, self.seq_len, self.seq_len_kv, self.kv_group],
                             device=IndexQ.device,
                             dtype=torch.float32)
        if (self._use_fused and IndexQ.dtype == torch.bfloat16
                and IndexK.dtype == torch.bfloat16):
            self.msprof_kernel_name = "fp8_indexer_distinctmeta_inline_mix_aic"
            if self._score is None or self._score.device != IndexQ.device:
                self._score = torch.empty(
                    (EXPERT_CORES, 2, ((self.seq_len_kv + EXPERT_BLOCK_N - 1) // EXPERT_BLOCK_N) * EXPERT_BLOCK_Q * self.heads, EXPERT_BLOCK_N),
                    device=IndexQ.device, dtype=torch.float16)
                self._norm_buffers = (
                    torch.empty_like(IndexQ, dtype=torch.float16), torch.empty_like(IndexK, dtype=torch.float16),
                    torch.empty_like(Weights),
                    torch.empty((self.batch, self.seq_len, 1), device=IndexQ.device, dtype=torch.float32),
                    torch.empty((self.batch, self.seq_len_kv, 1), device=IndexQ.device, dtype=torch.float32))
            fp8_lightning_indexer_fused_kernel(
                self.batch, self.seq_len, self.heads, self.index_dim,
                self.seq_len_kv, self.clean_logits, IndexQ, IndexK, IndexKScale,
                Logits, Weights, CuSeqLenKS, CuSeqLenKE, self._score, *self._norm_buffers)
        else:
            self.msprof_kernel_name = "_fp8_lightning_indexer_main_mix_aic"
            with scoped_ascend_mode("Developer"):
                fp8_lightning_indexer_wrapped_kernel(
                    self.batch, self.seq_len, self.heads, self.index_dim,
                    self.seq_len_kv, self.kv_group, self.clean_logits,
                    self.config["block_N"], self.config["num_stages"],
                    self.config["block_Q"], IndexQ, IndexK, IndexKScale, Logits,
                    Weights, CuSeqLenKS, CuSeqLenKE)
        return Logits
