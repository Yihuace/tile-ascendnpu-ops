"""Max pooling 2D forward kernel for NCHW inputs (NPU-adapted, single output).

``out[n, c, oh, ow] = max(x[n, c, oh*sH - pH + kh*dH, ow*sW - pW + kw*dW])`` over
the ``kernel_h x kernel_w`` window, PyTorch ``max_pool2d`` semantics: positions
outside the padded input are skipped, a NaN anywhere in the window wins (the
GPU reference folds it in with ``|``), and ties keep the first max.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_max_pool2d_kernel``) is extracted from
    the GPU repo via ``extract_tl_kernel.py`` (pattern B) and imported as-is,
    together with the ``pool_output_dim`` helper it calls (appended manually to
    keep this package self-contained).  It serves as the reference for the NPU
    kernel component to reimplement for ``target="npuir"``.  K1-K4 adaptations
    (decorator, grid/sync semantics, ``threads`` removal, padding strategy) are
    handled by the NPU kernel component during re-implementation.
    Only the plain (single-output) variant is imported; the GPU file's
    ``_max_pool2d_with_indices_kernel`` belongs to the not-yet-migrated
    ``MaxPool2dIndicesFwdOp`` and is intentionally not extracted.

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
# (pattern B) from {gpu_repo_root}/tileops/kernels/pool/max_pool2d.py.
# The ``pool_output_dim`` helper is imported from the same extracted file so
# the ``max_pool2d/`` package is self-contained (the extraction script anchors
# it there manually) and the output-extent formula has a single source.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .max_pool2d_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .max_pool2d_kernel import _max_pool2d_kernel, pool_output_dim
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .max_pool2d_kernel.perf_opt._max_pool2d_kernel import _max_pool2d_kernel, pool_output_dim

# ``pool_output_dim`` is re-exported (not used by this module's own code except
# through the imported set) so the extracted kernel package stays self-contained:
# the NPUIR rewrite of the factory keeps it alongside the kernel function.
__all__ = [
    "MaxPool2dKernel",
    "pool_output_dim",
]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::max_pool2d_wrapped_kernel", mutates_args=())
def _max_pool2d_wrapped_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_m: int,
    x: torch.Tensor,
) -> torch.Tensor:
    # GPU callable signature (pre-Stage-3): (block_m, threads).  The
    # NPU-shaped call below omits `threads` (K3/K9); it binds 1:1 once the NPU
    # kernel component rewrites the factory for target="npuir".
    return _max_pool2d_kernel(
        n,
        c_in,
        h_in,
        w_in,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
        dtype,
    )(block_m)(x)


@_max_pool2d_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_m: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (dtype, block_m)
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    return torch.empty((n, c_in, out_h, out_w), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class _MaxPool2dKernelBase(Kernel):
    """Shared construction and dispatch for the 2d max-pool kernel (NPU-adapted).

    Ported from GPU ``tileops/kernels/pool/max_pool2d.py:_MaxPool2dKernelBase``.
    The concrete variant binds two callables and declares nothing else:

    - ``_build`` the cached builder that traces the prim_func.
    - ``_dispatch`` the registered custom op the forward call goes through.

    The split is kept so the with-indices variant ports as a two-line subclass
    alongside this value-only variant.

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
        kernel_h: Window height.
        kernel_w: Window width.
        stride_h: Stride along height.
        stride_w: Stride along width.
        pad_h: Zero padding along height.
        pad_w: Zero padding along width.
        dilation_h: Dilation along height.
        dilation_w: Dilation along width.
        ceil_mode: Whether to use the ceil rule for the output extent.
        dtype: Data type (float16, bfloat16, or float32).
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
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
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
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
        self.out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
        self.kernel = type(self)._build(
            n,
            c_in,
            h_in,
            w_in,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
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
            self.h_in,
            self.w_in,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.dilation_h,
            self.dilation_w,
            self.ceil_mode,
            self.dtype_str,
            self.config["block_m"],
            x,
        )


class MaxPool2dKernel(_MaxPool2dKernelBase):
    """Max pooling forward kernel for NCHW inputs (single output)."""

    ascend_mode = "Developer"

    _build = staticmethod(_max_pool2d_kernel)
    _dispatch = staticmethod(_max_pool2d_wrapped_kernel)
