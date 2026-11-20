"""Grouped-query attention backward kernels (NPU-adapted wrapper + Kernel classes).

GQA backward (flash-attention style, BSHD layout) is a three-kernel pipeline
ported from the GPU TileOPs ``tileops/kernels/attention/gqa_bwd.py``:

  1. ``FlashAttnBwdPreprocessKernel`` -- row-wise ``delta = rowsum(o * do)``
     (shape ``[B, H, S]``, fp32).
  2. ``GQABwdKernel`` -- the main backward kernel.  Recomputes the softmax
     probabilities ``p = exp2(qk * scale - lse)`` from ``q``/``k``/``lse``,
     then accumulates ``dk``/``dv`` and atomically accumulates ``dq``.
     GQA head grouping: key/value head ``h // (heads // heads_kv)``.
  3. ``FlashAttnBwdPostprocessKernel`` -- casts the fp32 ``dq`` accumulator
     back to the storage dtype.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The three GPU TileLang kernels (``_flashattn_bwd_preprocess_kernel``,
    ``_gqa_bwd_kernel``, ``_flashattn_bwd_postprocess_kernel``) are extracted
    from the GPU repo via ``extract_tl_kernel.py`` into the single file
    ``_grouped_query_attention_bwd_kernels.py`` and imported as-is.  K1-K4 /
    K9 / K11 (decorator, grid/sync semantics, ``threads`` removal, boundary
    handling) are handled by the NPU kernel component during
    re-implementation for ``target="npuir"``.

  **Part B -- Kernel classes** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 89, 90]``).
    K8: ``autotune_configs`` (``tile_stage_thread_configs``) / ``tune``
        param -- removed; heuristic config selection only.
    K9: ``threads`` removed from ``default_config`` and the ``forward``
        dispatch.

  **Hopper-only variant (K5)**: the GPU op conditionally selects
  ``GQABwdWgmmaPipelinedKernel`` (``supported_archs = [90]``, WGMMA) on
  Hopper and skips the postprocess kernel there.  NPU has no WGMMA, so the
  manifest's ``source.kernel_map`` and this package keep only the
  architecture-generic ``GQABwdKernel`` path (``supported_archs = None``).
  Of the GPU file's MHA variants (``MHABwdKernel`` /
  ``MHABwdWgmmaPipelinedKernel``) none are ported -- they belong to the
  unmigrated MHA-backward op.

The imported functions are the GPU (CUDA target) implementations; they will
not run on NPU until the NPU kernel component reimplements them for
``target="npuir"``.
"""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: the GPU TileLang kernels extracted by extract_tl_kernel.py into the
# single file ``_grouped_query_attention_bwd_kernels.py`` (shared by all
# per-kernel migration prompts for this op).  They are the GPU (CUDA-target)
# implementations imported as-is -- the reference the NPU kernel component
# rewrites for ``target="npuir"``.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .grouped_query_attention_bwd_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .grouped_query_attention_bwd_kernel import (
#     _gqa_bwd_kernel,
# )
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .grouped_query_attention_bwd_kernel.perf_opt._gqa_bwd_kernel import _gqa_bwd_kernel

# The auxiliary Kernel classes (preprocess / postprocess) were moved verbatim
# to ``_aux_bwd_kernels.py`` so this wrapper declares exactly ONE top-level
# ``Kernel`` subclass (``GQABwdKernel``), as required by the Stage 5 gate
# (``gate_lint`` rule S5-ASCEND-MODE).  Re-imported here so this module's public
# names -- and the package ``__init__`` / op adapter that import them from here
# -- stay unchanged.
from ._aux_bwd_kernels import (
    FlashAttnBwdPostprocessKernel,
    FlashAttnBwdPreprocessKernel,
)

__all__ = [
    "FlashAttnBwdPostprocessKernel",
    "FlashAttnBwdPreprocessKernel",
    "GQABwdKernel",
]


class GQABwdKernel(Kernel):
    """Grouped-query attention backward kernel (BSHD layout).

    One program per ``(head, query-block, batch)``; iterates over the KV tiles
    and accumulates ``dk``/``dv`` (and, via ``T.atomic_add``, ``dq``).  GQA
    head grouping maps query head ``bx`` to key/value head
    ``bx // (heads // heads_kv)``; ``heads % heads_kv == 0`` is required.

    NPU adaptations:

    - K5: ``supported_archs = None`` (was ``[80, 89, 90]``).
    - K8: autotune (``tile_stage_thread_configs``) removed; heuristic config
      selection only.
    - K9: ``threads`` removed from ``default_config`` and the dispatch
      (``threads`` is a CUDA execution detail, not an NPU parameter).

    Args:
        batch: Batch size.
        heads: Number of query heads.
        heads_kv: Number of key/value heads (``heads % heads_kv == 0``).
        seq_len: Sequence length (query and kv; ``s_q != s_kv`` is not
            supported yet).
        dim: Head dimension.
        is_causal: Whether to apply a causal mask.
        dtype: Data type (float16 or bfloat16).
        config: Optional dict with "block_m", "block_n", "num_stages".

    Raises:
        ValueError: If ``heads % heads_kv != 0``.
    """

    ascend_mode = "Developer"

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
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _gqa_bwd_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
            self.dim,
            self.is_causal,
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
            "block_n": 64 if self.dim <= 64 else 32,
            "num_stages": 1,
        }

    def forward(self, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Dispatch to the NPU backward kernel.

        Called by the Op with ``(q, k, v, do, lse, delta, dq, dk, dv)``;
        ``dq``/``dk``/``dv`` are mutated in place.

        S5 integration fix: the source (GPU) wrapper used the two-level
        convention ``self.kernel(**self.config)(*inputs)`` -- on CUDA the
        ``@tilelang.jit`` factory returns a callable that still takes the
        autotune tiling config.  The NPU ``@tilelang.jit(target="npuir")``
        factory returns a ``JitKernel_NPU`` directly (the tiling is derived
        inside the factory from ``seq_len``/``dim``), so the extra
        ``(**self.config)`` call raised
        ``TypeError: JitKernel_NPU.__call__() got an unexpected keyword
        argument 'block_m'``.  The kernel is invoked with the tensors only.
        """
        return self.kernel(*inputs)
