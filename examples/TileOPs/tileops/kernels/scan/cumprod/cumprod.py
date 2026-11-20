"""Cumulative-product scan kernel package (cumprod migration unit), TileLang NPU.

Ported from GPU TileOPs ``tileops/kernels/reduction/cumulative.py`` — the
``CumulativeKernel`` class (shared by cumsum/cumprod via ``op_kind``) plus its
``torch.library.custom_op`` wrappers.  ``CumprodFwdOp`` is the migration unit
wired up here; the class structure is preserved in full so the sibling cumsum
op can reuse the same kernel package later (it shares ``CumulativeKernel`` via
``op_kind``).

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel functions (``_cumulative_kernel`` plus the
    three-pass parallel-scan helpers ``_parallel_scan_local_kernel``,
    ``_parallel_scan_carry_kernel``, ``_parallel_scan_propagate_kernel``) are
    extracted from the GPU repo via ``extract_tl_kernel.py`` and imported
    as-is.  They serve as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync, ``threads`` removal, padding strategy) are handled by the NPU
    component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K6: shared-memory budget via ``SHARED_MEMORY_BUDGET_BYTES`` imported from
        ``tileops.kernels.reduction._primitives`` (backend-agnostic).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``autotune()`` / ``tune`` param -- all removed;
        heuristic config selection only.
    K9: ``threads`` removed from ``default_config`` and ``forward``.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    SHARED_MEMORY_BUDGET_BYTES,
    align_up,
)

# ---------------------------------------------------------------------------
# Part A -- TileLang kernel functions (extracted + imported)
# ---------------------------------------------------------------------------
# ``_cumulative_kernel`` is the migrated NPU kernel for ``op_kind="prod"`` and
# is wired in through the baseline/perf_opt selection block below.  The three
# ``_parallel_scan_*`` helpers belong to the ``sum`` (cumsum) parallel path
# only -- ``use_parallel`` is always False for ``op_kind="prod"`` (source
# comment "cumprod has no parallel implementation"; DESIGN.md section 0.3 /
# 0.7) -- so they are outside this migration unit and stay imported verbatim
# from the Stage 0 extracted module.  The cumsum sibling unit owns replacing
# them when it migrates.
from ._cumprod_fwd_kernels import (
    _parallel_scan_carry_kernel,
    _parallel_scan_local_kernel,
    _parallel_scan_propagate_kernel,
)

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .cumprod_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .cumprod_kernel import _cumulative_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# v3 (2026-10-22): one module with a HIERARCHICAL carry ladder -- within-chunk
# (bs x bs) -> within-segment (<=64 x 64) -> cross-segment (max(16, ns) square).
# The segment granularity is derived from N inside the factory, and ns == 1
# collapses to the v2 three-phase structure, so the narrow-row regime and the
# wide rows ([64, 32768], which v2 could not run -- its ncp x ncp carry is 1 MB
# at N = 32768, far past the 192 KB UB) share one kernel family.
# v1 (`perf_opt/_cumulative_kernel.py`) is kept on disk, unmodified, as the
# scalar-scan fallback / rollback artifact.
from .cumprod_kernel.perf_opt.CumprodFwdOp import _cumulative_kernel

# Same module, same entry point: `cumprod_kernels.py` keeps the two names the
# shape-gated call site below used, so the import switch stays a one-liner pair.
from .cumprod_kernel.perf_opt.CumprodFwdOp import (
    _cumulative_kernel as _cumulative_kernel_v2,
)

__all__ = ["CumulativeKernel"]

# Tile size along the N dimension for the prefix scan.
# Must be a multiple of DEFAULT_ALIGNMENT for T.copy shared memory alignment.
_DEFAULT_BLOCK_N: int = 128

# Shared memory padding to eliminate bank conflicts (same reasoning as the
# extracted kernel module).  Used only by the Kernel class heuristics below.
_SMEM_PAD: int = 8


# ---------------------------------------------------------------------------
# custom_op wrappers (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::cumulative_parallel_fwd", mutates_args=())
def _cumulative_parallel_fwd_wrapped(
    M: int,
    N: int,
    dtype_str: str,
    block_m: int,
    block_n: int,
    x: torch.Tensor,
) -> torch.Tensor:
    n_tiles = align_up(N, DEFAULT_ALIGNMENT) // block_n
    local_fn = _parallel_scan_local_kernel(M, N, "sum", dtype_str)(block_m, block_n)
    y_local, tile_sums = local_fn(x)
    carry_fn = _parallel_scan_carry_kernel(M, n_tiles)()
    tile_carries_exclusive = carry_fn(tile_sums)
    propagate_fn = _parallel_scan_propagate_kernel(M, N, dtype_str)(block_m, block_n)
    return propagate_fn(y_local, tile_carries_exclusive)


@_cumulative_parallel_fwd_wrapped.register_fake
def _cumulative_parallel_fwd_wrapped_fake(
    M: int, N: int, dtype_str: str, block_m: int, block_n: int, x: torch.Tensor
):
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    return torch.empty((M, N_padded), dtype=x.dtype, device=x.device)


@torch.library.custom_op("npub::cumulative_fwd", mutates_args=())
def _cumulative_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    block_n: int,
    x: torch.Tensor,
) -> torch.Tensor:
    # op_kind="prod" runs the v3 hierarchical-carry kernel for every N
    # (<= 4096 -> v2's three-phase structure, > 4096 -> the carry ladder).
    if op_kind == "prod" and N <= 4096:
        return _cumulative_kernel_v2(M, N, op_kind, dtype_str)()(x)
    return _cumulative_kernel(M, N, op_kind, dtype_str)(block_m, block_n)(x)


@_cumulative_fwd_wrapped.register_fake
def _cumulative_fwd_wrapped_fake(
    M: int, N: int, op_kind: str, dtype_str: str, block_m: int, block_n: int, x: torch.Tensor
):
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    return torch.empty((M, N_padded), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class CumulativeKernel(Kernel):
    """Inclusive prefix scan kernel (cumsum / cumprod).

    Supports all architectures (NPU adaptation K5: ``supported_archs = None``).
    Uses 256-element alignment for shared memory copies.  Uses a tiled
    sequential scan loop along the last dimension: the N dimension is divided
    into tiles of ``block_n`` elements, reducing shared memory usage and
    improving occupancy.

    Boundary handling for non-aligned N is performed inside the kernel via
    masked loads with identity-element fills (0 for sum, 1 for prod), so no
    host-side ``F.pad`` is needed.  The ``forward()`` method accepts raw
    ``(M, N)`` tensors and returns ``(M, N_padded)`` output.

    NPU adaptation (K8): autotune has been removed; the kernel uses heuristic
    config selection only.  ``init_config(config)`` takes no ``tune`` argument.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: One of "sum", "prod".
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional kernel configuration dict.
        device_index: Device index (unused -- the shared-memory budget is a
            backend-agnostic constant from ``_primitives``; kept for API
            consistency with the NPU kernel constructor contract).
    """

    ascend_mode = "Developer"

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ):
        super().__init__()
        if op_kind not in ("sum", "prod"):
            raise ValueError(f"Unsupported op_kind '{op_kind}'. Expected one of 'sum', 'prod'.")
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)

        # Parallel scan only pays off for small-M, large-N; cumprod has no
        # parallel implementation.
        self.use_parallel = (M < 128 and N > 8192 and op_kind == "sum")

        if self.use_parallel:
            self.kernel = None
        else:
            # op_kind="prod": the v3 factory derives the carry ladder from N.
            if op_kind == "prod" and N <= 4096:
                self.kernel = _cumulative_kernel_v2(M, N, op_kind, self.dtype_str)
            else:
                self.kernel = _cumulative_kernel(M, N, op_kind, self.dtype_str)

        # K8: init_config(config) -- no tune argument.
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Select the default config for the selected backend and shape.

        K9: the returned dict no longer contains ``threads`` (a CUDA
        execution-model parameter with no NPU equivalent).
        """
        if self.use_parallel:
            block_n = 256 if self.N > 16384 else 128
            smem_per_row = (block_n + _SMEM_PAD) * 4  # fp32 intermediate
            max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row
            block_m = max(1, min(16, self.M, max_block_m))
            return {"block_m": block_m, "block_n": block_n}
        else:
            block_n = _DEFAULT_BLOCK_N
            elem_size = torch.tensor([], dtype=self.dtype).element_size()
            smem_per_row = 2 * (block_n + _SMEM_PAD) * elem_size
            max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row

            if self.M < 128:
                block_m = max(1, min(self.M, min(2, max_block_m)))
            else:
                block_m = 1
                for bm in [1, 2, 4, 8, 16]:
                    if bm <= max_block_m:
                        block_m = bm

            return {"block_m": block_m, "block_n": block_n}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the cumulative scan kernel (sequential or parallel).

        Args:
            x: Input tensor of shape (M, N).  Alignment padding is handled
                internally via masked loads.

        Returns:
            Output tensor of shape (M, N).  The prim_func writes an
            alignment-padded row; the surplus columns are trimmed here.

        K9: ``threads`` removed from the dispatch calls.
        """
        if self.use_parallel:
            y = _cumulative_parallel_fwd_wrapped(
                self.M,
                self.N,
                self.dtype_str,
                self.config["block_m"],
                self.config["block_n"],
                x,
            )
        else:
            y = _cumulative_fwd_wrapped(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_str,
                self.config["block_m"],
                self.config["block_n"],
                x,
            )
        return y[:, : self.N] if y.shape[1] > self.N else y
