"""Argreduce kernel (argmax) using TileLang (NPU-adapted).

Implements a two-step kernel: first finds the extreme value via parallel reduce,
then scans for the first index matching that value.
Operates on 2D (M, N) tensors with the raw reduction width -- the Op layer
dispatches unpadded inputs (``ArgmaxFwdOp._kernel_handles_padding = True``),
so the kernel sees the original N (narrow-N paths included).

Output is always int64 (index values).

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel function (``_argreduce_kernel``) is extracted
    from the GPU repo via ``extract_tl_kernel.py`` and imported as-is.
    It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync, ``threads`` removal, padding strategy) are handled by the
    NPU component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K6: shared-memory budget constant imported from ``_primitives``
        (backend-agnostic).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``tune`` param -- all removed.
    K9: ``threads`` removed from ``default_config`` and ``forward`` call.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up

# ---------------------------------------------------------------------------
# Part A -- TileLang kernel functions (extracted + imported)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .argmax_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .argmax_kernel import _argreduce_kernel
# from .argmax_kernel._argreduce_kernel import _select_config
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .argmax_kernel.perf_opt._argreduce_kernel import _argreduce_kernel, _select_config

__all__ = ["ArgreduceKernel"]

_ARGREDUCE_KINDS = {"argmax", "argmin"}


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::argreduce_fwd", mutates_args=())
def _argreduce_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _argreduce_kernel(M, N, op_kind, dtype_str)(block_m)(x)


@_argreduce_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, x):
    return torch.empty((M,), dtype=torch.int64, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class ArgreduceKernel(Kernel):
    """Argmax / argmin forward kernel.

    Supports all architectures (NPU adaptation K5: ``supported_archs = None``).
    Uses 256-element alignment for shared memory copies. Implements a
    two-step approach: parallel reduce to find the extreme value, then
    serial scan to find the first matching index.

    Output dtype is always int64.

    NPU adaptation (K8): autotune has been removed; the kernel uses
    heuristic config selection only.  ``init_config(config)`` takes no
    ``tune`` argument.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: One of "argmax", "argmin".
        dtype: Input data type (float32, float16, or bfloat16).
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
        if op_kind not in _ARGREDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_ARGREDUCE_KINDS)}."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        # Raw-N contract: ``ArgmaxFwdOp._kernel_handles_padding = True``
        # disables the Op-layer alignment pad, so forward() receives the
        # unpadded (M, N) tensor and the jit factory must be declared with
        # the raw width N. Declaring a padded width while receiving the raw
        # tensor (or vice versa) misbinds row strides and corrupts every row
        # past the first (see integration_log.md attempt 1). The perf_opt
        # kernel dispatches narrow N (e.g. N=4 on dim=0 reductions) to a
        # deinterleave path that is unreachable through the padded contract.
        self.kernel = _argreduce_kernel(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
        )
        # msprof anchor: the active kernel source declares which compiled
        # kernel this dispatch launches (nsplit -> argreduce_partial; all
        # single-kernel paths -> main). Tier-2/3 resolution in
        # tileops.benchmark.msprof uses it so ``msprof op`` filters out
        # Op-layer prep ops (movedim/contiguous transposes) and sibling
        # kernels of multi-launch dispatches. The baseline (Stage 3) source
        # declares nothing and falls back to "main".
        self.msprof_kernel_name = getattr(self.kernel, "msprof_kernel_name", None) or "main"
        # K8: init_config(config) -- no tune argument.
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Select ``block_m`` via the NPU kernel's UB-budget ladder.

        Stage 5 glue (integration_log.md attempt 1): the NPU kernel is the
        single source of truth for tiling.  ``_select_config`` is the
        probe-calibrated ladder (kernel DESIGN.md section 5.2) whose manual
        UB budget covers the kernel's resident buffers under auto-multi-
        buffering, applies the narrow-N budget halving, and clamps
        ``block_m <= M`` (over-allocating rows corrupts the vectorizer's
        small-N row handling).  The previous GPU-era shared-memory heuristic
        ignored all of these constraints and fed out-of-budget ``block_m``
        values to the kernel (UB overflow on (256, 4096) fp16/bf16).

        The ladder runs on the raw row width ``N`` -- with
        ``_kernel_handles_padding = True`` the Op layer dispatches the
        unpadded tensor, so N is the width the kernel actually sees (the
        narrow-N budget halving only engages on the raw width).  The GPU
        reference below is retained as history: ``threads = 128`` was a GPU
        TileLang layout-constraint heuristic, not a config key (K9 removed
        it).

        K9: the returned dict contains ``block_m`` only (no ``threads``).
        """
        if self.N_padded == 0:
            raise ValueError(
                "Reduction dimension is empty (N=0). "
                "argmax/argmin over an empty dimension is undefined."
            )
        block_m = _select_config(self.M, self.N, self.dtype_str)["block_m"]
        return {"block_m": block_m}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the argmax/argmin kernel.

        Args:
            x: Input tensor of shape (M, N).  The Op layer has already
                transposed/reshaped the original tensor; no alignment
                padding is applied (raw-N contract).

        Returns:
            Output tensor of shape (M,) with dtype int64.

        The kernel is dispatched with the raw ``N`` so the declared (M, N)
        signature matches the unpadded input exactly (a width mismatch
        misbinds strides; see ``__init__`` note).

        K9: ``threads`` removed from the call.
        """
        return _argreduce_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self.config["block_m"],
            x,
        )
