"""Reduce kernel package (amin migration unit) using TileLang (NPU-adapted).

Ported from GPU TileOPs ``tileops/kernels/reduction/reduce.py`` — the
``ReduceKernel`` class (shared by sum/mean/amin/amax/prod/std/var/var_mean)
plus its ``torch.library.custom_op`` wrappers.  ``AminFwdOp`` is the
migration unit wired up here; the class structure is preserved in full so
the sibling reduce ops can reuse the same kernel package later (they share
``ReduceKernel`` via ``op_kind``).

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel functions (``_simple_reduce_kernel``,
    ``_simple_reduce_kernel_tiled``, ``_welford_reduce_kernel``,
    ``_welford_reduce_kernel_tiled`` and the prod helpers) are extracted
    from the GPU repo via ``extract_tl_kernel.py`` and imported as-is.
    They serve as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync, ``threads`` removal, padding strategy) are handled by the
    NPU component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K6: shared-memory budget via ``device_smem_budget`` from
        ``_primitives`` (backend-agnostic).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``autotune()`` / ``_tile_n_candidates`` /
        ``tune`` param -- all removed; heuristic config selection only.
    K9: ``threads`` removed from ``default_config`` and ``forward``.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    MAX_SINGLE_TILE_COLS,
    SHARED_MEMORY_BUDGET_BYTES,
    align_up,
    compute_tile_n,
    device_smem_budget,
)

# ---------------------------------------------------------------------------
# Part A -- TileLang kernel functions (extracted + imported)
# ---------------------------------------------------------------------------
# Full GPU implementations (signature + docstring + @T.prim_func body), kept
# verbatim as the reference for the NPU kernel component.  The ``prod``
# builders are pulled in by ``_simple_reduce_kernel`` (the ``op_kind ==
# "prod"`` dispatcher branch) and are defined in the extracted module, so they
# only need to be reachable there -- not imported here.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Policy: the perf_opt source is active once tuned drop-in kernels (same factory
# signatures) land under ``amin_kernel/perf_opt/`` and pass their L0/L1
# regression; the baseline (Stage 3) source is active otherwise.
# ``pytest tests/ops/`` and ``pytest benchmarks/ops/`` dispatch through whichever
# source is active here.
#
# Stage 4 TUNING_COMPLETED (2026-10-19 Round 5/8 winner; 2026-10-26 Round 9 bf16
# restructure produced no winner -> winner unchanged) activates the perf_opt
# block.  The resident factory is imported from the operator-name contract file
# ``perf_opt/AminFwdOp.py``; the streaming factory keeps its own module name
# ``perf_opt/_simple_reduce_kernel_tiled.py`` (AminFwdOp.py re-exports the same
# function object).
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .amin_kernel import _simple_reduce_kernel, _simple_reduce_kernel_tiled
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .amin_kernel.perf_opt.AminFwdOp import _simple_reduce_kernel
from .amin_kernel.perf_opt._simple_reduce_kernel_tiled import _simple_reduce_kernel_tiled

# The Welford (std/var) builders belong to op kinds that are NOT part of this
# migration unit (only the two ``simple`` builders are migrated).  Import them
# optionally so an integration package that ships only the migrated builders
# still imports cleanly; the std/var code paths are unreachable for amin.
try:  # pragma: no cover - std/var op kinds are out of this migration's scope
    from ._amin_fwd_kernels import (
        _welford_reduce_kernel,
        _welford_reduce_kernel_tiled,
    )
except ImportError:  # pragma: no cover
    _welford_reduce_kernel = None
    _welford_reduce_kernel_tiled = None

__all__ = ["ReduceKernel"]

# Welford-based op kinds (simple kinds use the ``_simple_reduce_kernel`` path).
_WELFORD_KINDS = {"std", "var", "var_mean"}


# ---------------------------------------------------------------------------
# custom_op wrappers (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::reduce_simple_fwd", mutates_args=())
def _reduce_simple_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _simple_reduce_kernel(M, N, op_kind, dtype_str)(block_m)(x)


@_reduce_simple_wrapped.register_fake
def _reduce_simple_wrapped_fake(M, N, op_kind, dtype_str, block_m, x):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


@torch.library.custom_op("npub::reduce_simple_tiled_fwd", mutates_args=())
def _reduce_simple_tiled_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    tile_n: int,
    block_m: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _simple_reduce_kernel_tiled(M, N, op_kind, dtype_str, tile_n)(block_m)(x)


@_reduce_simple_tiled_wrapped.register_fake
def _reduce_simple_tiled_wrapped_fake(M, N, op_kind, dtype_str, tile_n, block_m, x):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


@torch.library.custom_op("npub::reduce_welford_fwd", mutates_args=())
def _reduce_welford_wrapped(
    M: int,
    N: int,
    op_kind: str,
    correction: int,
    dtype_str: str,
    block_m: int,
    x: torch.Tensor,
) -> list[torch.Tensor]:
    kernel = _welford_reduce_kernel(M, N, op_kind, correction, dtype_str)
    result = kernel(block_m)(x)
    if op_kind == "var_mean":
        # Returns (var, mean)
        return [result[0], result[1]]
    else:
        return [result]


@_reduce_welford_wrapped.register_fake
def _reduce_welford_wrapped_fake(M, N, op_kind, correction, dtype_str, block_m, x):
    if op_kind == "var_mean":
        return [
            torch.empty((M,), dtype=x.dtype, device=x.device),
            torch.empty((M,), dtype=x.dtype, device=x.device),
        ]
    return [torch.empty((M,), dtype=x.dtype, device=x.device)]


@torch.library.custom_op("npub::reduce_welford_tiled_fwd", mutates_args=())
def _reduce_welford_tiled_wrapped(
    M: int,
    N: int,
    op_kind: str,
    correction: int,
    dtype_str: str,
    tile_n: int,
    block_m: int,
    x: torch.Tensor,
) -> list[torch.Tensor]:
    kernel = _welford_reduce_kernel_tiled(M, N, op_kind, correction, dtype_str, tile_n)
    result = kernel(block_m)(x)
    if op_kind == "var_mean":
        return [result[0], result[1]]
    else:
        return [result]


@_reduce_welford_tiled_wrapped.register_fake
def _reduce_welford_tiled_wrapped_fake(
    M, N, op_kind, correction, dtype_str, tile_n, block_m, x
):
    if op_kind == "var_mean":
        return [
            torch.empty((M,), dtype=x.dtype, device=x.device),
            torch.empty((M,), dtype=x.dtype, device=x.device),
        ]
    return [torch.empty((M,), dtype=x.dtype, device=x.device)]


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class ReduceKernel(Kernel):
    """Unified reduce kernel supporting sum/mean/amin/amax/prod/std/var/var_mean.

    Supports all architectures (NPU adaptation K5: ``supported_archs =
    None``).  Uses 256-element alignment for shared-memory copies.
    Dispatches to the simple or Welford kernel based on ``op_kind``.

    When ``N_padded`` exceeds ``MAX_SINGLE_TILE_COLS``, tiled kernel variants
    are used that iterate over N in chunks of ``tile_n`` columns, avoiding the
    TileLang vectorizer limit at the 32768-column boundary.

    Boundary handling for non-aligned N is performed inside the kernel via
    masked loads with identity-element fills, so no host-side ``F.pad`` is
    needed.  The ``forward()`` method accepts raw ``(M, N)`` tensors.

    NPU adaptation (K8): autotune has been removed; the kernel uses
    heuristic config selection only.  ``init_config(config)`` takes no
    ``tune`` argument.

    Args:
        M: Number of rows (product of all dims except the reduced one).
        N: Reduction width (last dim of the reshaped 2-D input).
        op_kind: One of sum/mean/amin/amax/prod/std/var/var_mean.
        dtype: Input data type (float32, float16, or bfloat16).
        correction: Bessel's correction (std/var/var_mean only).
        config: Optional kernel configuration dict.
        device_index: Device index (unused -- the shared-memory budget is a
            backend-agnostic constant from ``_primitives``; kept for API
            consistency with the NPU kernel constructor contract).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        correction: int = 1,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ):
        super().__init__()
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.correction = correction
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._is_welford = op_kind in _WELFORD_KINDS
        self._elem_bytes = torch.tensor([], dtype=dtype).element_size()
        self._smem_budget = device_smem_budget()

        # Determine whether tiling is needed
        self._needs_tiling = self.N_padded > MAX_SINGLE_TILE_COLS

        if not self._needs_tiling:
            if self._is_welford:
                self.kernel = _welford_reduce_kernel(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.correction,
                    self.dtype_str,
                )
            else:
                self.kernel = _simple_reduce_kernel(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.dtype_str,
                )
        # For tiled path, kernel is built lazily using tile_n from config.
        # Tiled kernels are dispatched via the wrapped functions (not a single
        # self.kernel), so there is no autotune entry point for them.
        # K8: init_config(config) -- no tune argument.
        self.init_config(config)

        # After init_config, ensure tile_n is consistent with the chosen
        # block_m.  A caller-provided config may have block_m without tile_n.
        if self._needs_tiling:
            bm = self.config.get("block_m", 1)
            if "tile_n" not in self.config or self.config["tile_n"] == 0:
                self.config["tile_n"] = self._tile_n_for_block_m(bm)

    def _tile_n_for_block_m(self, block_m: int) -> int:
        """Return tile_n for a given block_m (0 means no tiling needed).

        Welford kernels allocate 2 shared buffers per pass, so
        ``num_buffers=2`` is used to ensure both fit in the budget.
        """
        budget = self._smem_budget
        if self.N_padded <= MAX_SINGLE_TILE_COLS:
            single = compute_tile_n(
                block_m, self._elem_bytes, self.N_padded, budget=budget,
            )
            if single == self.N_padded:
                return 0

        num_buffers = 2 if self._is_welford else 1
        col_budget = MAX_SINGLE_TILE_COLS * num_buffers * block_m * self._elem_bytes
        effective_budget = min(budget, col_budget)
        return compute_tile_n(
            block_m, self._elem_bytes, self.N_padded,
            num_buffers=num_buffers,
            budget=effective_budget,
        )

    @property
    def default_config(self) -> dict:
        """Select ``block_m`` (and ``tile_n`` for the tiled path).

        K9: the returned dict no longer contains ``threads`` (a CUDA
        execution-model parameter with no NPU equivalent).
        """
        if not self._needs_tiling:
            smem_per_row = self.N_padded * self._elem_bytes
            max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row
            block_m = 1
            for bm in [1, 2, 4, 8]:
                if bm <= max_block_m:
                    block_m = bm
            return {"block_m": block_m}

        # Tiled path: pick block_m that minimizes tile count
        best_bm = 1
        best_tile_n = self._tile_n_for_block_m(1)

        for bm in [2, 4, 8]:
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            if tn == 0:
                # No tiling needed at this block_m -- always prefer
                best_bm = bm
                best_tile_n = tn
            elif best_tile_n == 0:
                # Current best doesn't need tiling; keep it
                pass
            else:
                # Both need tiling -- pick fewer tiles
                best_num = (self.N_padded + best_tile_n - 1) // best_tile_n
                curr_num = (self.N_padded + tn - 1) // tn
                if curr_num < best_num:
                    best_bm = bm
                    best_tile_n = tn

        return {"block_m": best_bm, "tile_n": best_tile_n}

    def forward(self, x: torch.Tensor) -> object:
        """Run the reduce kernel on a raw ``(M, N)`` tensor.

        Args:
            x: Input tensor of shape ``(M, N)``.

        Returns:
            A single ``(M,)`` tensor (simple/Welford ops) or a list for the
            ``var_mean`` op.

        K9: ``threads`` removed from the dispatch calls.
        """
        if self._is_welford:
            if self._needs_tiling:
                results = _reduce_welford_tiled_wrapped(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.correction,
                    self.dtype_str,
                    self.config["tile_n"],
                    self.config["block_m"],
                    x,
                )
            else:
                results = _reduce_welford_wrapped(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.correction,
                    self.dtype_str,
                    self.config["block_m"],
                    x,
                )
            if self.op_kind == "var_mean":
                return results[0], results[1]
            return results[0]
        else:
            if self._needs_tiling:
                return _reduce_simple_tiled_wrapped(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.dtype_str,
                    self.config["tile_n"],
                    self.config["block_m"],
                    x,
                )
            else:
                return _reduce_simple_wrapped(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.dtype_str,
                    self.config["block_m"],
                    x,
                )
