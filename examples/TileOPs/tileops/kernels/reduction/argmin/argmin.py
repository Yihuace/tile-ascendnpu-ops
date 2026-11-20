"""Argreduce kernel (argmin) using TileLang (NPU-adapted).

Implements a two-step kernel: first finds the extreme value via parallel reduce,
then scans for the first index matching that value.
Operates on 2D (M, N_padded) tensors; the Op layer handles reshape.
256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.

``ArgminFwdOp`` and ``ArgmaxFwdOp`` share the GPU ``ArgreduceKernel``
(``op_kind`` selects the branch); on NPU each op is its own migration unit,
so this package carries its own copy of the kernel class wired to the
``argmin`` kind.

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
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    align_up,
)

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .argmin_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .argmin_kernel import _argreduce_kernel
# from .argmin_kernel._argreduce_kernel import _select_config
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .argmin_kernel.perf_opt._argreduce_kernel import _argreduce_kernel, _select_config

# Tiling single source of truth: the NPU kernel module owns the ``block_m`` /
# ``path`` selection (``_select_config``), which is the exact configuration the
# Stage 3 L0/L1 suite validated.  The GPU-era 48 KiB SMEM heuristic below
# over-allocates ``block_m`` on NPU (ub overflow at N_padded=4096, and a hard
# budget error at N_padded=102400), so glue delegates to the kernel selector
# (CASE-argmax-argreduce-integration defect 2 / DESIGN.md 0.2 item 4).
# (baseline source path retained in the switch block above)
# from .argmin_kernel._argreduce_kernel import _select_config

__all__ = ["ArgreduceKernel"]

_ARGREDUCE_KINDS = {"argmax", "argmin"}


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


# NOTE: must not reuse argmax's ``npub::argreduce_fwd``: two modules
# registering the same op name silently hand the op to whichever module
# is imported last (argmin wins over argmax in tileops.kernels.reduction).
@torch.library.custom_op("npub::argreduce_min_fwd", mutates_args=())
def _argreduce_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    x: torch.Tensor,
) -> torch.Tensor:
    # K9: the NPU factory callable accepts ``block_m`` only (no ``threads``).
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
    two-step approach: parallel reduce to find the extreme value, then a
    serial scan to find the first matching index.

    Output dtype is always int64.

    NPU adaptation (K8): autotune has been removed; the kernel uses
    heuristic config selection only.  ``init_config(config)`` takes no
    ``tune`` argument.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: One of "argmax", "argmin" (this package drives "argmin";
            the kind is preserved verbatim from the shared GPU kernel).
        dtype: Input data type (float32, float16, or bfloat16).
        config: Optional kernel configuration dict.
        device_index: Device index (O3 constructor contract; the
            shared-memory budget is a backend-agnostic constant from
            ``_primitives``, so the value is kept for API consistency).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE.
    ascend_mode = "Developer"

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
        self.kernel = _argreduce_kernel(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
        )
        # msprof 锚点：把 kernel factory 声明的实际 kernel 名传给 benchmark，
        # 避免 profiler 全量捕获 Op 层 prep 算子（MemSet/PadV3）导致 profile 无效。
        self.msprof_kernel_name = getattr(self.kernel, "msprof_kernel_name", None) or "main"
        # K8: init_config(config) -- no tune argument.
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Select ``block_m`` from the NPU kernel's own tiling selector.

        The NPU kernel module owns config selection (``_select_config``): it
        is the single source of truth for the resident/tiled ladder and is the
        exact configuration validated by the Stage 3 L0/L1 suite.  Delegating
        here replaces the GPU-era 48 KiB SMEM heuristic, which over-allocates
        ``block_m`` on NPU (``ub overflow`` at ``N_padded=4096`` for fp16/bf16
        and a hard per-row budget error at ``N_padded=102400``); those NPU
        fits depend on the UB budget model, not on the CUDA copy-layout
        machinery the old heuristic mirrored (CASE-argmax-argreduce-integration
        defect 2 / DESIGN.md section 0.2 item 4).

        K9: the returned dict contains ``block_m`` only (no ``threads``).  For
        the tiled path ``_select_config`` returns ``block_m == 1`` (forced by
        the multitile trap), which the tiled factory consumes verbatim.
        """
        return {"block_m": _select_config(self.M, self.N, self.dtype_str)["block_m"]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the argmax/argmin kernel.

        Args:
            x: Input tensor of shape (M, N_padded).

        Returns:
            Output tensor of shape (M,) with dtype int64.

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
