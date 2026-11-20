"""Logical reduce kernel (``all``) using TileLang (NPU-adapted).

Ported from the GPU ``tileops/kernels/reduction/logical_reduce.py``. The GPU
``LogicalReduceKernel`` is shared by the any / all / count_nonzero kinds (the
``op_kind`` argument selects the branch) and dispatches between a baseline
factory (``_logical_reduce_kernel``) and an N-tiled factory
(``_logical_reduce_kernel_tiled``). This migration unit is a per-op package:
it carries its own copy of the kernel class wired to the ``all`` kind and only
the **baseline** factory (long-row tiling is out of scope).

Operates on raw 2D ``(M, N)`` tensors; the kernel handles 256-element
alignment padding internally via masked loads, so the Op layer passes the raw
``(M, N)`` tensor straight through (``_kernel_handles_padding = True``).

Output dtype: ``bool`` for ``all`` (``torch.all`` semantics; the reduction is
``reduce_min`` over the 0/1-cast values, i.e. logical AND). Padding uses 1.0,
the neutral element for AND.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU baseline TileLang kernel function (``_logical_reduce_kernel``)
    is extracted from the GPU repo via ``extract_tl_kernel.py`` and imported
    as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync, ``threads`` removal, padding strategy) are handled by the NPU
    component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K6: ``device_smem_budget`` imported from ``_primitives``
        (backend-agnostic).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``); the name is
        namespaced per migration unit (``all_logical_reduce_fwd``) so it does
        not collide with the sibling count_nonzero package, which owns
        ``npub::logical_reduce_fwd``.
    K8: ``autotune_configs`` / ``_tile_n_candidates`` / ``tune`` param --
        all removed (heuristic config only).
    K9: ``threads`` removed from the factory callable, ``default_config``,
        and the ``forward`` dispatch.
    K11: not applicable (this is a reduction op -- no ``npt``).

  The N-tiled factory (``_logical_reduce_kernel_tiled``) and the
  ``_needs_tiling`` dispatch are intentionally **not** ported: only the
  baseline factory is part of this migration unit.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    align_up,
    device_smem_budget,
)

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .all_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .all_kernel import _logical_reduce_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .all_kernel.perf_opt._logical_reduce_kernel import _logical_reduce_kernel

__all__ = ["LogicalReduceKernel", "to_logical_float32"]

_LOGICAL_REDUCE_KINDS = {"any", "all", "count_nonzero"}

# TileLang does not support bool, integer, or complex dtypes as a storage
# dtype for T.alloc_shared / T.copy. When the caller's dtype is one of these,
# we use float32 internally instead. The Op layer is responsible for
# pre-converting the tensor to float32 before calling the kernel.
_FLOAT32_STORAGE_DTYPE = torch.float32
_UNSUPPORTED_STORAGE_DTYPES = frozenset(
    {
        torch.bool,
        torch.complex64,
        torch.complex128,
        torch.int32,
        torch.int64,
    }
)


def to_logical_float32(x: torch.Tensor) -> torch.Tensor:
    """Convert an unsupported-storage-dtype tensor to float32 for kernel dispatch.

    - bool:        True -> 1.0, False -> 0.0
    - int32/int64: direct cast to float32
    - complex:     nonzero (either real or imaginary part != 0) -> 1.0, else 0.0
    """
    if x.dtype in (torch.bool, torch.int32, torch.int64):
        return x.to(torch.float32)
    # complex: element is "truthy" if real != 0 OR imag != 0
    return ((x.real != 0) | (x.imag != 0)).to(torch.float32)


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::all_logical_reduce_fwd", mutates_args=())
def _logical_reduce_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    x: torch.Tensor,
) -> torch.Tensor:
    # K9: the NPU factory callable accepts ``block_m`` only (no ``threads``).
    out_raw = _logical_reduce_kernel(M, N, op_kind, dtype_str)(block_m)(x)
    if op_kind == "count_nonzero":
        return out_raw.to(torch.int64)
    return out_raw.bool()


@_logical_reduce_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, x):
    if op_kind == "count_nonzero":
        return torch.empty((M,), dtype=torch.int64, device=x.device)
    return torch.empty((M,), dtype=torch.bool, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class LogicalReduceKernel(Kernel):
    """All forward kernel (baseline, non-tiled factory).

    Supports all architectures (NPU adaptation K5: ``supported_archs = None``).
    Casts the input to bool (0/1) and reduces via ``reduce_min`` per row, so the
    output is ``True`` iff every element of the row is non-zero (logical AND;
    ``torch.all`` semantics).  The kernel handles 256-element alignment padding
    internally via masked loads, using 1.0 as the AND-neutral pad value.

    Output dtype is ``bool`` for the ``all`` kind.

    Note: TileLang does not support bool, integer, or complex dtypes as a
    storage dtype for shared memory. When ``dtype`` is one of these, the kernel
    is compiled for float32 internally and the Op layer is responsible for
    pre-converting the input tensor to float32 before calling ``forward()``.

    NPU adaptation (K8): autotune has been removed; the kernel uses heuristic
    config selection only. ``init_config(config)`` takes no ``tune`` argument.
    The N-tiled fallback of the GPU kernel is not ported (baseline factory
    only).

    Args:
        M: Number of rows (product of all dims except the reduced dims).
        N: Reduction width (product of the reduced dims).
        op_kind: One of "any", "all", "count_nonzero" (this package drives
            "all"; the kind is preserved verbatim from the shared GPU kernel).
        dtype: Input data type (float32, float16, bfloat16, bool, int32,
               int64, complex64, or complex128).
        config: Optional kernel configuration dict.
        device_index: Device index for the shared-memory budget query.
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
        if op_kind not in _LOGICAL_REDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_LOGICAL_REDUCE_KINDS)}."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        # TileLang cannot handle bool, integer, or complex dtypes as
        # shared-memory storage dtypes; remap all unsupported dtypes -> float32.
        self._kernel_dtype = (
            _FLOAT32_STORAGE_DTYPE if dtype in _UNSUPPORTED_STORAGE_DTYPES else dtype
        )
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = torch.tensor([], dtype=self._kernel_dtype).element_size()
        self._smem_budget = device_smem_budget(device_index)
        self.kernel = _logical_reduce_kernel(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_to_str(self._kernel_dtype),
        )
        # K8: init_config(config) -- no tune argument.
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Select ``block_m`` based on the shared-memory budget.

        Ported from the GPU ``LogicalReduceKernel.default_config`` non-tiled
        branch.  K9: the returned dict contains ``block_m`` only (no
        ``threads``); the NPU factory callable consumes ``block_m`` verbatim.
        """
        smem_per_row = self.N_padded * self._elem_bytes
        max_block_m = self._smem_budget // smem_per_row
        block_m = 1
        for bm in [1, 2, 4, 8]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the all kernel.

        Args:
            x: Input tensor of shape ``(M, N)``. Must have dtype matching
               ``_kernel_dtype`` (float32 when the original dtype is bool,
               int32, int64, complex64, or complex128).

        Returns:
            Output tensor of shape ``(M,)`` with dtype ``bool``.

        K9: ``threads`` removed from the dispatch.
        """
        return _logical_reduce_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_to_str(self._kernel_dtype),
            self.config["block_m"],
            x,
        )
