"""Adaptive max pooling 2D forward kernel for CHW/NCHW inputs (NPU-adapted).

``out[n, c, oh, ow] = max(x[n, c, ih_start:ih_end, iw_start:iw_end])`` with the
PyTorch adaptive-bin partition ``[floor(o*in/out), ceil((o+1)*in/out))``.  A NaN
anywhere in the window wins (the GPU reference folds it in with ``|``).

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_adaptive_max_pool2d_kernel``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` (pattern B) and
    imported as-is, together with the ``adaptive_bin`` /
    ``max_adaptive_bin_extent`` helpers it calls (appended manually to keep
    this package self-contained).  It serves as the reference for the NPU
    kernel component to reimplement for ``target="npuir"``.  K1-K4 adaptations
    (decorator, grid/sync semantics, ``threads`` removal, padding strategy) are
    handled by the NPU kernel component during re-implementation.
    Only the plain (single-output) variant is imported; the GPU file's
    ``_adaptive_max_pool2d_with_indices_kernel`` is not migrated.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``tune`` parameter removed; heuristic config
        selection only, ``init_config(config)``.
    K9: ``threads`` removed from the custom_op wrapper signature, the
        factory-callable invocation, ``default_config`` and the ``forward``
        dispatch (the meaningful knob is ``block_m``, the per-block output
        element count).

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation whose inner callable takes ``(block_m, threads)``.  The
  dispatch below passes the NPU-shaped argument list (``block_m`` only); it
  binds 1:1 once the NPU kernel component rewrites the factory for
  ``target="npuir"`` and drops ``threads`` from the callable signature (K3).
  Running ``forward`` before that rewrite fails -- this is the expected
  scaffold state.
"""

from typing import Any, Callable, ClassVar, Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/pool/adaptive_max_pool2d.py.
# The helpers ``adaptive_bin`` / ``max_adaptive_bin_extent`` are imported from
# the same extracted file so the ``adaptive_max_pool2d/`` package is
# self-contained (the extraction script anchors them there manually).
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .adaptive_max_pool2d_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .adaptive_max_pool2d_kernel import (
#     _adaptive_max_pool2d_kernel,
#     adaptive_bin,
#     max_adaptive_bin_extent,
# )
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .adaptive_max_pool2d_kernel.perf_opt._adaptive_max_pool2d_kernel import (
    _adaptive_max_pool2d_kernel,
    adaptive_bin,
    max_adaptive_bin_extent,
)

# ``adaptive_bin`` / ``max_adaptive_bin_extent`` are re-exported (not used by
# this module's own code) so the extracted kernel package stays self-contained:
# the NPUIR rewrite of the factory keeps them alongside the kernel function.
__all__ = [
    "AdaptiveMaxPool2dKernel",
    "adaptive_bin",
    "max_adaptive_bin_extent",
]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::adaptive_max_pool2d_wrapped_kernel", mutates_args=())
def _adaptive_max_pool2d_wrapped_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str,
    block_m: int,
    x: torch.Tensor,
) -> torch.Tensor:
    # GPU callable signature (pre-Stage-3): (block_m, threads).  The
    # NPU-shaped call below omits `threads` (K3/K9); it binds 1:1 once the NPU
    # kernel component rewrites the factory for target="npuir".
    return _adaptive_max_pool2d_kernel(
        n,
        c_in,
        h_in,
        w_in,
        out_h,
        out_w,
        dtype,
    )(block_m)(x)


@_adaptive_max_pool2d_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str,
    block_m: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (dtype, block_m)
    return torch.empty((n, c_in, out_h, out_w), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class AdaptivePool2dKernelBase(Kernel):
    """Shared scaffold for the adaptive 2D pool kernels (NPU-adapted).

    Ported from GPU ``tileops/kernels/pool/common.py:AdaptivePool2dKernelBase``.
    A concrete variant binds two callables and declares nothing else:

    - ``_build`` the cached builder that traces the prim_func.
    - ``_dispatch`` the registered custom op the forward call goes through.

    The split is kept so the max/indices variants port as two-line subclasses
    alongside the average variant already migrated here.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures; was the CUDA SM list
      ``[80, 86, 89, 90]``).
    - K8: autotune removed (no ``autotune_configs``, no ``tune`` parameter);
      heuristic config selection only, ``init_config(config)``.
    - K9: ``threads`` removed from ``default_config`` and the ``forward``
      dispatch.

    Args:
        n: Batch size.
        c_in: Number of channels.
        h_in: Input height.
        w_in: Input width.
        out_h: Output height.
        out_w: Output width.
        dtype: Data type (float16 or bfloat16).
        config: Optional kernel config override.
    """

    # Stage 5 gate (S5-ASCEND-MODE): the class that directly inherits ``Kernel``
    # must carry an explicit ``ascend_mode`` literal matching the integrated
    # kernel module's ``ASCEND_MODE`` (Developer).  The concrete variant below
    # keeps its own literal as well; both resolve to Developer.
    ascend_mode = "Developer"

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: ClassVar[Optional[list]] = None
    _build: ClassVar[Callable[..., Any]]
    _dispatch: ClassVar[Callable[..., Any]]

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        out_h: int,
        out_w: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(f"{type(self).__name__} supports float16 and bfloat16, got {dtype}")
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.out_h = out_h
        self.out_w = out_w
        self.dtype = dtype
        self.kernel = type(self)._build(n, c_in, h_in, w_in, out_h, out_w, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Default config (K9: ``threads`` key removed)."""
        return {"block_m": 256}

    def forward(self, x: torch.Tensor):
        """K9: ``threads`` removed from the dispatch."""
        return type(self)._dispatch(
            self.n,
            self.c_in,
            self.h_in,
            self.w_in,
            self.out_h,
            self.out_w,
            self.dtype_str,
            self.config["block_m"],
            x,
        )


class AdaptiveMaxPool2dKernel(AdaptivePool2dKernelBase):
    """Adaptive max pooling forward kernel for NCHW inputs (single output)."""

    ascend_mode = "Developer"

    _build = staticmethod(_adaptive_max_pool2d_kernel)
    _dispatch = staticmethod(_adaptive_max_pool2d_wrapped_kernel)
