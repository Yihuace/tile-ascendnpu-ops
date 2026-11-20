"""Clamp forward kernel (Tensor-bound) using TileLang (NPU-adapted).

Implements the Tensor-bound form of ``torch.clamp``:

    y = clamp(x, lo, hi)          # both Tensor bounds
    y = clamp(x, lo, None)        # Tensor lower bound only
    y = clamp(x, None, hi)        # Tensor upper bound only

All operands are broadcast to the output shape and flattened to contiguous
1-D vectors of ``N_total`` elements by the Op layer.  Output dtype is
``same_as(input)``.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang factory (``_make_clamp_tensor_kernel``) is extracted from
    the GPU repo via ``extract_tl_kernel.py`` and imported as-is.  It serves as
    the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 and K11 adaptations (decorator, grid/sync,
    ``threads``/``npt`` collapse, padding strategy) are handled by the NPU
    component during re-implementation:
      - E1 (K3/K11): the factory keeps its op-specific positional arguments
        ``(has_min, has_max)`` but drops ``output_dtype`` / ``is_fp8`` /
        ``threads`` / ``npt`` -> ``_make_clamp_tensor_kernel(N, dtype,
        has_min, has_max)``.
      - E2 (K11): the returned callable takes a single ``block_size``
        (the GPU ``threads * npt`` product) instead of
        ``(threads_arg, npt_arg)``.
      - E3 (K2): ``T.Kernel(..., is_npu=True)`` replaces the CUDA
        ``threads=`` dimension.
      - E4 (K11): ``T.Parallel(block_size)`` replaces
        ``T.Parallel(threads_arg, npt_arg)``.
    The fp32 transit in the extracted non-fp8 branch (``x32 = T.cast(...,
    "float32")`` -> ``T.min(T.max(x32, lo32), hi32)``) plus the explicit
    ``T.isnan`` NaN-propagation guards are semantically required and must be
    preserved by the NPU re-implementation: they reproduce ``torch.clamp``'s
    fp32 comparison and NaN-propagation semantics.

  **Part B -- Kernel class** (fully ported):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
         heuristic config selection only.
    K9:  ``threads`` removed from ``default_config`` and ``forward`` call.
    K11: GPU ``threads * npt`` product collapsed into a single ``block_size``
         config key (elementwise ops only).
    E6:  the GPU ``ParametricUnaryKernel`` base logic (``_builder_fn`` /
         ``_builder_args`` / ``default_config`` / ``forward``) is inlined --
         the NPU project does not ship that base class.
    E10: no ``@torch.library.custom_op`` wrapper (the GPU
         ``ClampTensorFwdKernel`` has none either); ``forward`` calls the
         compiled factory callable directly.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# --- baseline (extracted GPU TileLang implementation) -----------------------
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .clamp_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .clamp_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .clamp_kernel import _make_clamp_tensor_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .clamp_kernel.perf_opt._make_clamp_tensor_kernel import (
    _make_clamp_tensor_kernel,
    TUNED_BLOCK_SIZE,
)

__all__ = ["ClampTensorFwdKernel"]

# ---------------------------------------------------------------------------
# K11: GPU ``ParametricUnaryKernel.default_config`` = {"threads": 512,
# "num_per_thread": npt} with the npt heuristic (npt=4 for fp32, npt=8 for
# fp16/bf16); the GPU ``ClampTensorFwdKernel`` overrides ``_DEFAULT_THREADS``
# to 512.  Collapsed to a single per-block element count:
# block_size = threads * npt = 2048 (fp32) / 4096 (fp16, bf16).
# ---------------------------------------------------------------------------
# Stage 4 (optimize): active source is perf_opt -> use its per-dtype tuned
# defaults.  (Baseline pair was {"float32": 2048, "float16": 4096, "bfloat16": 4096}.)
_DEFAULT_BLOCK_SIZE = TUNED_BLOCK_SIZE


# ---------------------------------------------------------------------------
# Kernel class (K5-K9, K11, E6, E10 adaptations)
# ---------------------------------------------------------------------------


class ClampTensorFwdKernel(Kernel):
    """Tensor-bound clamp kernel: y = clamp(x, lo, hi).

    Computes ``y = clamp(x, lo, hi)`` over flat tensors of length
    ``N_total``.  The Op layer broadcasts ``input`` / ``min`` / ``max`` to
    the output shape and flattens them before dispatch.  ``has_min`` /
    ``has_max`` select between the three forms used by the Tensor clamp,
    clamp_min, and clamp_max ops.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` product collapsed into ``block_size``.
    - E6: the GPU ``ParametricUnaryKernel`` base logic (builder dispatch /
      ``default_config``) is inlined.
    - E10: no custom_op wrapper; ``forward`` calls the compiled factory
      callable directly (GPU parity).

    Args:
        N_total: Total number of (post-broadcast, flattened) elements.
        dtype: Data type (float16, bfloat16, or float32).
        has_min: Whether the lower-bound Tensor is present.
        has_max: Whether the upper-bound Tensor is present.
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 4096}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Float-only dtypes (inherited from GPU ParametricUnaryKernel).
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        has_min: bool,
        has_max: bool,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if not (has_min or has_max):
            raise ValueError(
                "ClampTensorFwdKernel requires has_min or has_max to be True",
            )
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.has_min = bool(has_min)
        self.has_max = bool(has_max)
        self.output_dtype = dtype  # same_as(input)
        # Build the factory callable (does not compile yet — compilation is
        # deferred to forward()).  The imported function is the GPU
        # implementation; the NPU component rewrites it for target="npuir"
        # with the K11 signature ``(N, dtype, has_min, has_max, block_size)``.
        # ``has_min`` / ``has_max`` are the op-specific positional arguments
        # preserved between ``N``/``dtype`` and the common ``block_size``
        # suffix (GPU ``_builder_args`` -> ``(has_min, has_max)``).
        self.kernel = _make_clamp_tensor_kernel(
            self.N_total, self.dtype_str, self.has_min, self.has_max
        )
        self._compiled_fn = None
        self._compiled_block_size: Optional[int] = None
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * npt).

        GPU defaults (ClampTensorFwdKernel): threads=512, npt=4 (fp32) or
        npt=8 (fp16/bf16).  Collapsed: block_size = 2048 (fp32) or
        4096 (fp16/bf16).
        """
        # K11: collapse threads * npt into block_size.
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(
        self,
        x: torch.Tensor,
        lo: Optional[torch.Tensor] = None,
        hi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the Tensor-bound clamp kernel.

        Accepts flat contiguous tensors of ``N_total`` elements.  ``lo`` /
        ``hi`` are present only for the bounds configured at construction
        (``has_min`` / ``has_max``).  K9: ``threads`` removed from the call;
        K11/E2: the compiled callable takes a single ``block_size``.
        """
        block_size = self.config["block_size"]
        if self._compiled_fn is None or self._compiled_block_size != block_size:
            self._compiled_fn = self.kernel(block_size)
            self._compiled_block_size = block_size
        if self.has_min and self.has_max:
            return self._compiled_fn(x, lo, hi)
        if self.has_min:
            return self._compiled_fn(x, lo)
        return self._compiled_fn(x, hi)
