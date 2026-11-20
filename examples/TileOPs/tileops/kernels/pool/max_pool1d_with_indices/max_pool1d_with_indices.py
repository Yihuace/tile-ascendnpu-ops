"""Max pooling 1D forward kernel with indices for NCL inputs (NPU-adapted).

``out[n, c, ow] = max(x[n, c, ow*sW - pW + kw*dW])`` over the ``kernel_w``
window, PyTorch ``max_pool1d(..., return_indices=True)`` semantics: positions
outside the padded input are skipped (equivalent to ``-inf`` padding for a max
reduction), a NaN anywhere in the window wins (value = NaN, index = the *last*
NaN visited), and ties keep the first max.  ``indices[n, c, ow]`` is the input
position ``iw`` (int64) of the winning element along the length axis.  Both
outputs share the ``(n, c_in, out_l)`` shape.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_max_pool1d_with_indices_kernel``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` (pattern B) and
    imported as-is, together with the ``pool_output_dim`` helper it calls
    (appended manually to keep this package self-contained).  It serves as the
    reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 adaptations (decorator, grid/sync semantics,
    ``threads`` removal, padding strategy) are handled by the NPU kernel
    component during re-implementation.  Only the with-indices variant is
    imported; the GPU file's plain (single-output) ``_max_pool1d_kernel`` is
    already migrated in ``tileops/kernels/pool/max_pool1d/`` and is
    intentionally not extracted here.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``tune`` parameter removed; heuristic config
        selection only, ``init_config(config)``.
    K9: ``threads`` removed from the custom_op wrapper signature, the
        factory-callable invocation, ``default_config`` and the ``forward``
        dispatch (the meaningful knob is ``block_m``, the per-block output
        element count).  The dual-output factory keeps ``out_idx=[1, 2]``.

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation whose inner callable takes ``(block_m, threads)``.  The
  dispatch below passes the NPU-shaped argument list (``block_m`` only); it
  binds 1:1 once the NPU kernel component rewrites the factory for
  ``target="npuir"`` and drops ``threads`` from the callable signature (K3).
  Running ``forward`` before that rewrite fails -- this is the expected
  scaffold state.
"""

from typing import Any, Callable, ClassVar, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/pool/max_pool1d.py.
# The ``pool_output_dim`` helper is imported from the same extracted file so the
# ``max_pool1d_with_indices/`` package is self-contained (the extraction script
# anchors it there manually) and the output-extent formula has a single source.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .max_pool1d_with_indices_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .max_pool1d_with_indices_kernel import _max_pool1d_with_indices_kernel, pool_output_dim
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .max_pool1d_with_indices_kernel.perf_opt._max_pool1d_with_indices_kernel import (
    _max_pool1d_with_indices_kernel,
    pool_output_dim,
)

# ``pool_output_dim`` is re-exported (not used by this module's own code except
# through the imported set) so the extracted kernel package stays self-contained:
# the NPUIR rewrite of the factory keeps it alongside the kernel function.
__all__ = [
    "MaxPool1dWithIndicesKernel",
    "pool_output_dim",
]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::max_pool1d_with_indices_wrapped_kernel", mutates_args=())
def _max_pool1d_with_indices_wrapped_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_m: int,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # GPU callable signature (pre-Stage-3): (block_m, threads).  The
    # NPU-shaped call below omits `threads` (K3/K9); it binds 1:1 once the NPU
    # kernel component rewrites the factory for target="npuir".  The dual-output
    # factory keeps out_idx=[1, 2] (values + int64 indices).
    return _max_pool1d_with_indices_kernel(
        n,
        c_in,
        l_in,
        kernel_w,
        stride_w,
        pad_w,
        dilation_w,
        ceil_mode,
        dtype,
    )(block_m)(x)


@_max_pool1d_with_indices_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_m: int,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _ = (dtype, block_m)
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    return (
        torch.empty((n, c_in, out_l), dtype=x.dtype, device=x.device),
        torch.empty((n, c_in, out_l), dtype=torch.int64, device=x.device),
    )


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class _MaxPool1dKernelBase(Kernel):
    """Shared construction and dispatch for the 1d max-pool kernels (NPU-adapted).

    Ported from GPU ``tileops/kernels/pool/max_pool1d.py:_MaxPool1dKernelBase``.
    The concrete variant binds two callables and declares nothing else:

    - ``_build`` the cached builder that traces the prim_func.
    - ``_dispatch`` the registered custom op the forward call goes through.

    The split is kept so the with-indices variant (this op) sits as a two-line
    subclass alongside the value-only variant in
    ``tileops/kernels/pool/max_pool1d/``.

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
        l_in: Input length.
        kernel_w: Window width (the 1d max-pool kernels name their pooling axis
            ``w``).
        stride_w: Stride along the length axis.
        pad_w: Zero padding along the length axis.
        dilation_w: Dilation along the length axis.
        ceil_mode: Whether to use the ceil rule for the output extent.
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel config override.
    """

    ascend_mode = "Developer"

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: ClassVar[Optional[list]] = None
    _build: ClassVar[Callable[..., Any]]
    _dispatch: ClassVar[Callable[..., Any]]

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_w: int,
        stride_w: int,
        pad_w: int,
        dilation_w: int,
        ceil_mode: bool,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"{type(self).__name__} supports float16, bfloat16, and float32, got {dtype}"
            )
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_w = kernel_w
        self.stride_w = stride_w
        self.pad_w = pad_w
        self.dilation_w = dilation_w
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
        self.kernel = type(self)._build(
            n,
            c_in,
            l_in,
            kernel_w,
            stride_w,
            pad_w,
            dilation_w,
            ceil_mode,
            self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Default config (K9: ``threads`` key removed)."""
        return {
            "block_m": 256,
        }

    def forward(self, x: torch.Tensor) -> Any:
        """K9: ``threads`` removed from the dispatch."""
        return type(self)._dispatch(
            self.n,
            self.c_in,
            self.l_in,
            self.kernel_w,
            self.stride_w,
            self.pad_w,
            self.dilation_w,
            self.ceil_mode,
            self.dtype_str,
            self.config["block_m"],
            x,
        )


class MaxPool1dWithIndicesKernel(_MaxPool1dKernelBase):
    """Max pooling forward-with-indices kernel for NCL inputs."""

    ascend_mode = "Developer"

    _build = staticmethod(_max_pool1d_with_indices_kernel)
    _dispatch = staticmethod(_max_pool1d_with_indices_wrapped_kernel)
