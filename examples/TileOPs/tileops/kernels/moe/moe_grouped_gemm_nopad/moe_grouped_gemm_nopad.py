"""MoE no-padding grouped GEMM kernel package (NPU-adapted).

Ported from GPU TileOPs ``tileops/kernels/moe/moe_grouped_gemm_nopad.py`` --
the ``MoeGroupedGemmNopadKernel`` class (NT grouped GEMM with a GPU tile
scheduler and a zero-CPU-sync dead-CTA early-exit guard).

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factories (``_tile_scheduler_kernel`` and
    ``_moe_grouped_gemm_kernel``) are extracted from the GPU repo via
    ``extract_tl_kernel.py`` and imported as-is.  They serve as the
    reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 adaptations (decorator, grid/sync semantics,
    ``threads`` removal, padding strategy) are handled by the NPU kernel
    component during re-implementation.

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8: ``autotune_configs`` / ``tune`` param removed; heuristic config
        selection only (``init_config(config)``).
    K9: ``threads`` removed from ``default_config`` and from the
        ``forward`` dispatch of both the scheduler and the GEMM factory
        (the GPU scheduler callable took ``threads``; ``_SCHED_THREADS`` is
        the removal target on the NPU side).  ``num_stages`` is retained
        (``T.Pipelined`` depth; not a CUDA thread parameter).

  Pre-Stage-3 note: the imported factories are still the GPU (CUDA target)
  implementations.  The GEMM factory's inner callable takes
  ``(block_m, block_n, block_k, num_stages, threads, group_size_m)`` and the
  scheduler factory's takes ``(threads)``; the dispatch below passes the
  NPU-shaped argument lists (no ``threads``).  They become callable once the
  NPU kernel component rewrites the factories for ``target="npuir"`` and
  drops ``threads`` from the callable signatures (K3).  Running ``forward``
  before that rewrite fails -- this is the expected scaffold state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import device_ascend_mode

# Part A: GPU TileLang kernel factories extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/moe/moe_grouped_gemm_nopad.py.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .moe_grouped_gemm_nopad_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .moe_grouped_gemm_nopad_kernel import _moe_grouped_gemm_kernel, _tile_scheduler_kernel
# S5 glue: reuse the kernel's own bounded fresh-compile retry (the documented
# reliable net for the flaky bishengir-compile SIGSEGV; see the kernel DESIGN /
# ``_run_gemm``).
from .moe_grouped_gemm_nopad_kernel._moe_grouped_gemm_kernel import _run_gemm
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# Round 12: the tuned drop-in is the single gate-contract artifact
# `perf_opt/MoeGroupedGemmNopadFwdOp.py` (tuned GEMM body + scheduler re-export);
# the sibling scheduler module keeps its own name.
# Adopting it needs BOTH lines below AND `_DEFAULT_CONFIG = dict(TUNED_DEFAULT_CONFIG)`
# (the wrapper passes block_m/block_n/block_k/num_stages explicitly, so flipping
# the import alone would keep the Stage 3 geometry).
# from .moe_grouped_gemm_nopad_kernel.perf_opt.MoeGroupedGemmNopadFwdOp import _moe_grouped_gemm_kernel, TUNED_DEFAULT_CONFIG
# from .moe_grouped_gemm_nopad_kernel.perf_opt._tile_scheduler_kernel import _tile_scheduler_kernel
# from .moe_grouped_gemm_nopad_kernel.perf_opt.MoeGroupedGemmNopadFwdOp import _run_gemm

__all__ = ["MoeGroupedGemmNopadKernel"]

# K9: "threads" removed from the GPU default config ``threads=128``.
_DEFAULT_CONFIG = {
    "block_m": 64,
    "block_n": 256,
    "block_k": 64,
    "num_stages": 3,
    "group_size_m": 1,
}


class MoeGroupedGemmNopadKernel(Kernel):
    """NT grouped GEMM for MoE without block_m-aligned padding.

    Two-phase execution: GPU tile scheduler -> GEMM with dead-CTA early-exit.
    Grid = (max_tiles, ceildiv(N, block_n)); valid CTAs process one M-tile
    each; dead CTAs (bx >= total_tiles[0]) exit after one GPU-memory read.
    total_tiles is never copied to CPU -- zero CPU sync.

    Args:
        numel: T * top_k, tight row count.
        num_experts: Total number of experts E.
        N: Output feature dimension (e.g. 2*ffn_size for gate+up).
        K: Input feature dimension (hidden_size).
        dtype: Activation and weight dtype.
        config: Optional config dict with block_m/n/k, num_stages,
            group_size_m.
        device_index: Optional device index for multi-device dispatch.

    Example:
        >>> kernel = MoeGroupedGemmNopadKernel(numel=16384, num_experts=256, N=4096, K=2048,
        ...                               dtype=torch.bfloat16)
        >>> C = kernel(A, B, true_sizes, true_offsets)  # [numel, N]
    """

    # Expert is required on 910B (Developer persistent+gemm segfaults,
    # CG-2026-0010) but broken on A5/CANN 9.2 (Expert codegen emits a MIX
    # module with an empty AIV sub-block -> bishengir "Unknown PIPE!").
    # ``device_ascend_mode`` downgrades Expert -> Developer on A5 only;
    # __init__ resolves it per instance so the pin matches the device.
    ascend_mode = "Expert"

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # forward() launches three device kernels in one call: the tile
    # scheduler's ``_scan_main`` + ``_expand_main`` and the real grouped GEMM
    # (prim_func ``main`` -> device kernels ``main_mix_aic`` /
    # ``main_mix_aiv``).  msprof without a kernel-name filter captures all
    # three and the report tool cannot resolve a single op, so the Roofline
    # Ratio is lost.  Pin the filter to the real GEMM device kernel (see
    # tileops/benchmark/msprof.py kernel-name resolution chain).
    msprof_kernel_name = "main_mix_aic"

    def __init__(
        self,
        numel: int,
        num_experts: int,
        N: int,
        K: int,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ):
        super().__init__()
        self.numel = numel
        self.num_experts = num_experts
        self.N = N
        self.K = K
        self.dtype = dtype
        self.device_index = device_index
        # A5: Expert codegen is broken for this kernel, use Developer there.
        self.ascend_mode = device_ascend_mode(type(self).ascend_mode)
        # K8: ``tune`` removed -- heuristic config selection only.
        self.init_config(config)

    def _max_tiles(self, block_m: int) -> int:
        return self.numel // block_m + self.num_experts

    def _max_tiles_per_expert(self, block_m: int) -> int:
        return (self.numel + block_m - 1) // block_m

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    def forward(
        self,
        A: torch.Tensor,             # [numel, K]
        B: torch.Tensor,             # [num_experts, N, K]
        true_sizes: torch.Tensor,    # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run tile scheduler + GEMM with GPU-side dead-CTA early-exit.

        Phase 1: TileLang scheduler builds tile_expert_ids, tile_row_offsets,
                 and total_tiles[1] -- all on device, no .item() needed.
        Phase 2: GEMM with grid = (max_tiles, N_tiles).
                 Each CTA reads total_tiles[0] from device; dead CTAs (bx >=
                 total_tiles[0]) exit immediately.

        Args:
            A: [numel, K] tight permuted hidden states.
            B: [num_experts, N, K] expert weight matrix (NT: B^T applied).
            true_sizes: [E] int32 true token count per expert.
            true_offsets: [E] int32 tight start offset per expert in A.

        Returns:
            C: [numel, N] GEMM output.
        """
        block_m = self.config["block_m"]
        max_tiles = self._max_tiles(block_m)

        # Phase 1: build tile schedule -- total_tiles stays on device (no .item()).
        # S5 glue: the integrated NPU scheduler factory returns the ``run``
        # callable directly (``run(true_sizes)`` -> the 3 GM tables).  The
        # Stage-0 scaffold appended a stray ``()`` because it expected a
        # zero-arg factory, which raises on the NPU ``run`` (missing arg).
        sched_fn = _tile_scheduler_kernel(self.num_experts, max_tiles, block_m)
        tile_expert_ids, tile_row_offsets, total_tiles_t = sched_fn(true_sizes)

        # Phase 2: GEMM with dead-CTA early-exit -- 1D grid with GROUP_SIZE_M
        # reordering.  S5 glue: the integrated NPU factory already binds the
        # block geometry in its own signature and returns the compiled kernel
        # directly; the Stage-0 scaffold re-called it with the GPU block tuple
        # (``(block_m, block_n, block_k, num_stages, group_size_m)``), which
        # raises on the NPU JITKernel.  The NPU kernel is no-pad: every tile's
        # M/N extent is clamped by ``true_sizes``/``true_offsets`` at runtime,
        # so the GPU zero-row A padding (a TMA-path workaround) is dropped.
        # S5 glue: bounded fresh-compile retry (``_run_gemm``) -- the kernel's
        # own harness treats the flaky bishengir-compile SIGSEGV as recoverable.
        # The factory call sits inside the lambda so a retry recompiles.
        return _run_gemm(
            lambda: _moe_grouped_gemm_kernel(
                self.numel, max_tiles, self.num_experts, self.N, self.K, self.dtype_str
            ),
            (A, B, tile_expert_ids, tile_row_offsets,
             true_offsets, true_sizes, total_tiles_t),
        )
