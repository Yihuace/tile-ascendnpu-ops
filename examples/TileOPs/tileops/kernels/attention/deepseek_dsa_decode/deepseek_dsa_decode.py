"""Sparse MLA decode kernel — NPU-adapted wrapper + Kernel class.

DeepSeek sparse attention *decode* with a KV cache and per-(b, s, h_kv)
top-k index list.  Layout: ``Q [B, S, H, D + dim_tail]`` (BSHD, the first
``D`` channels are the "no-PE" q/k and the ``dim_tail`` tail is the shared
RoPE key); ``KV [B, S_kv, H_kv, D + dim_tail]``; ``indices [B, S, H_kv,
topk]`` (int32) selects, per query row, the KV positions to attend to;
output ``O [B, S, H, D]``.

The GPU kernel (``SparseMlaKernel``) is a Hopper **warp-specialized**
sparse-MLA decode: 384 threads split into a 128-thread consumer for the
left q-half + online softmax, a 128-thread consumer for the right q-half,
and a 128-thread async producer that gathers the selected KV rows into
double-buffered shared memory.  It is CUDA-specific (WGMMA, named
barriers, ``cp.async``, ``set_max_nreg``, ``T.get_thread_binding``) and has
no NPU analogue — the NPU kernel component designs and implements the NPU
replacement in Stage 1-3.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang factory ``_sparse_mla_kernel`` (``@functools.lru_cache``
    wrapping a ``@tilelang.jit`` builder whose body defines the
    ``@T.prim_func``) is extracted from the GPU repo via
    ``extract_tl_kernel.py`` and imported as-is; it is the reference for the
    NPU kernel component to reimplement for ``target="npuir"``.  K1-K4
    adaptations (decorator/target, grid/sync semantics, ``threads`` removal,
    boundary-mask strategy) are handled by the NPU component during
    re-implementation.  The ``LOG2E`` constant the factory references is
    appended into the extracted file so the ``deepseek_dsa_decode/`` package
    is self-contained (SKILL.md S3 "Helper function extraction").

  **Part B -- custom_op wrappers + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[90]``; the GPU SM90/Hopper gate
        is dropped — the source op additionally hardcoded an ``is_hopper()``
        check, removed in the Op layer).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``autotune()`` / ``supply_prog`` / ``tune``
        param and the ``itertools`` import — removed; heuristic
        ``default_config`` selection only.
    K9: ``threads`` removed from the wrapper signature, ``default_config``
        and the ``forward`` dispatch (CUDA execution detail with no NPU
        analogue; the extracted GPU factory still carries ``threads`` in its
        callable signature -- the NPU component drops it when rewriting for
        NPUIR).

  **Op-specific flow preserved (O6)**: the single-factory construction, the
    ``kernel_map`` dispatch, and the ``forward`` passthrough of the 3 inputs
    (q, kv, indices) are preserved verbatim from the GPU kernel.

Pre-Stage-3 note: the imported factory is still a GPU (CUDA target)
implementation whose inner callable takes ``(block_i, threads)``.  The
dispatch below already passes the NPU-shaped argument list (no ``threads``);
it becomes callable once the NPU kernel component rewrites the factory for
``target="npuir"``.  Running ``forward`` before that rewrite fails — this is
the expected scaffold state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: the GPU TileLang factory extracted by extract_tl_kernel.py (single
# output file ``_deep_seek_sparse_attention_decode_with_k_v_cache_fwd_kernels.py``,
# shared by the per-kernel migration prompt for this op).  It is the GPU
# (CUDA target) implementation imported as-is -- the reference the NPU kernel
# component rewrites for ``target="npuir"``.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .deepseek_dsa_decode_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .deepseek_dsa_decode_kernel import _sparse_mla_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .deepseek_dsa_decode_kernel.perf_opt._sparse_mla_kernel import _sparse_mla_kernel

__all__ = ["SparseMlaKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::sparse_mla_fwd_wrapped_kernel", mutates_args=())
def _sparse_mla_wrapped_kernel(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    kv_stride: int,
    q_start_index_s: int,
    kv_group: int,
    sm_scale: Optional[float],
    is_causal: bool,
    cp0: bool,
    dtype: str,
    block_heads: int,
    block_N: int,
    num_stages: int,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Wrapper for sparse multi-head attention kernel execution.

    K3/K9: the factory callable is invoked without ``threads`` (the CUDA
    execution detail is dropped on NPU).  The Stage 3 NPU factory
    (``@tilelang.jit(...)``) compiles with the NPU tiling knobs
    (``block_heads`` / ``block_N`` / ``num_stages``) and returns the
    ``JitKernel_NPU`` directly, so it is invoked once and the returned kernel
    is called with ``(q, kv, indices)`` (the GPU ``(block_i, threads)``
    two-level call convention is dropped).
    """
    return _sparse_mla_kernel(
        batch,
        seq_len,
        seq_len_kv,
        heads,
        dim,
        tail_dim,
        topk,
        kv_stride,
        q_start_index_s,
        kv_group,
        sm_scale,
        is_causal,
        cp0,
        dtype,
        block_heads,
        block_N,
        num_stages,
    )(q, kv, indices)


@_sparse_mla_wrapped_kernel.register_fake
def _(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    kv_stride: int,
    q_start_index_s: int,
    kv_group: int,
    sm_scale: Optional[float],
    is_causal: bool,
    cp0: bool,
    dtype: str,
    block_heads: int,
    block_N: int,
    num_stages: int,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    # Output shape is ``(B, S, H, D)`` (the ``dim_tail`` PE channels are not
    # part of the output).
    return torch.empty([batch, seq_len, heads, dim], device=q.device, dtype=q.dtype)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class SparseMlaKernel(Kernel):
    """Sparse MLA decode kernel (KV cache + per-row top-k indices).

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures; the GPU class was
      ``[90]`` -- Hopper only).
    - K8: autotune removed (``autotune_configs`` / ``autotune()`` /
      ``supply_prog`` / ``tune``); heuristic config selection only.
    - K9: ``threads`` removed from ``default_config`` and the ``forward``
      dispatch.

    Args:
        batch: Batch size.
        seq_len: Query sequence length.
        seq_len_kv: Key-value sequence length.
        heads: Number of query heads.
        dim: Head dimension (the non-tail part, ``D``).
        tail_dim: Shared RoPE tail dimension (``dim_tail``).
        dtype: Data type (float16 or bfloat16).
        topk: Number of selected KV positions per query row.
        kv_stride: KV stride (decode step spacing).
        q_start_index_s: Query start position in the global sequence.
        kv_group: Number of key-value groups (GQA); default 1.
        sm_scale: Softmax scale; default ``(1 / (dim + tail_dim)) ** 0.5``.
        is_causal: Whether the attention is causal (must be True).
        cp0: Whether this compute unit owns the first chunk (``cp_rank == 0``).
        config: Optional dict with "block_i".
    """

    ascend_mode = "Developer"

    # K5: [90] (CUDA SM90 / Hopper) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        heads: int,
        dim: int,
        tail_dim: int,
        dtype: torch.dtype,
        topk: int,
        kv_stride: int,
        q_start_index_s: int,
        kv_group: int = 1,
        sm_scale: float = None,
        is_causal: bool = True,
        cp0: bool = True,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.heads = heads
        self.dim = dim
        self.tail_dim = tail_dim
        self.dtype = dtype
        self.topk = topk
        self.kv_stride = kv_stride
        self.kv_group = kv_group
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.q_start_index_s = q_start_index_s
        self.cp0 = cp0

        # Build the factory callable (no TileLang JIT compilation yet --
        # compilation happens when the returned func is invoked inside the
        # custom_op wrapper).  The imported function is the GPU implementation;
        # the NPU component rewrites it for ``target="npuir"``.
        self.kernel = _sparse_mla_kernel(
            self.batch,
            self.seq_len,
            self.seq_len_kv,
            self.heads,
            self.dim,
            self.tail_dim,
            self.topk,
            self.kv_stride,
            self.q_start_index_s,
            self.kv_group,
            self.sm_scale,
            self.is_causal,
            self.cp0,
            self.dtype_str,
        )
        self.msprof_kernel_name = getattr(self.kernel, "msprof_kernel_name", "")

        # K8: ``supply_prog`` / ``autotune`` removed with the autotuner.
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Default tiling config (NPU factory tiling knobs).

        K9: ``threads`` (CUDA execution detail) dropped.  The GPU ``block_i``
        knob has no NPU analogue and is replaced by the Stage 3 NPU factory's
        compile-time tiling parameters ``block_heads`` / ``block_N`` /
        ``num_stages``.  The defaults below are the Stage 3 verified L0/L1
        configuration; the factory contract requires ``block_heads | heads``
        and ``block_N | topk``.
        """
        return {"block_heads": 32, "block_N": 64, "num_stages": 2}

    def forward(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Run sparse MLA decode; accepts q, kv, and the top-k indices.

        K9: ``threads`` removed from the dispatch call.

        Args:
            q: Query tensor ``(B, S, H, D + dim_tail)``.
            kv: Key-value tensor ``(B, S_kv, H_kv, D + dim_tail)``.
            indices: int32 top-k indices ``(B, S, H_kv, topk)``.

        Returns:
            The output tensor ``(B, S, H, D)``.
        """
        return _sparse_mla_wrapped_kernel(
            self.batch,
            self.seq_len,
            self.seq_len_kv,
            self.heads,
            self.dim,
            self.tail_dim,
            self.topk,
            self.kv_stride,
            self.q_start_index_s,
            self.kv_group,
            self.sm_scale,
            self.is_causal,
            self.cp0,
            self.dtype_str,
            self.config["block_heads"],
            self.config["block_N"],
            self.config["num_stages"],
            q,
            kv,
            indices,
        )
