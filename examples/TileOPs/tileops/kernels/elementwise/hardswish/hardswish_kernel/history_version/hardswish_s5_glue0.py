"""Hardswish forward kernel using TileLang (NPU-adapted).

Implements the elementwise HardSwish activation::

    y = x * clamp(x + 3, 0, 6) * (1 / 6)

Scaling by the reciprocal matches ``torch.nn.functional.hardswish``; finite
inputs come out bit-identical to it.  Unlike HardSigmoid, the HardSwish body
uses plain ``min`` / ``max`` (no NaN-propagating helpers).

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel function (``hardswish_fwd_kernel``) is extracted
    from the GPU repo (``make_unary_regcopy`` builder + ``op_func`` inlined) and
    imported as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 and K11 adaptations (decorator,
    grid/sync, ``threads``/``npt`` collapse, padding strategy) are handled by
    the NPU component during re-implementation.

  **Part B -- Kernel class** (fully ported from the GPU ``FloatUnaryKernel`` /
  ``UnaryKernel`` register_copy flow):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
         heuristic config selection only.
    K9:  ``threads`` removed from ``default_config`` and ``forward``.
    K11: ``threads * npt`` collapsed into a single ``block_size`` config key.
    (No ``@torch.library.custom_op`` wrapper -- the GPU elementwise kernel has
    none either, E10-style; ``forward`` dispatches the compiled callable
    directly.)
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Part A -- imported TileLang kernel function (GPU reference implementation).
#
# ``hardswish_fwd_kernel`` is the extracted GPU implementation imported as-is.
# The NPU kernel component rewrites it for ``target="npuir"``; the import name
# and call shape (factory -> compiled callable) stay unchanged.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .hardswish_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .hardswish_kernel import hardswish_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .hardswish_kernel.perf_opt.hardswish_fwd_kernel import hardswish_fwd_kernel

__all__ = ["HardswishFwdKernel"]

# ---------------------------------------------------------------------------
# K11 constant: GPU ``threads * npt`` collapsed into a single ``block_size``.
#
# The GPU ``FloatUnaryKernel`` register_copy strategy used ``threads * npt``
# (256 * 8 = 2048 elements per block, the default of the extracted factory).
# On NPU this two-level parallel collapses into a single ``block_size`` UB tile
# width; the NPU kernel component re-tunes it during re-implementation.
# ---------------------------------------------------------------------------

_DEFAULT_BLOCK_SIZE = 2048


class HardswishFwdKernel(Kernel):
    """Hardswish forward kernel: y = x * clamp(x + 3, 0, 6) * (1 / 6).

    A single-input, single-output elementwise activation.  The kernel flattens
    the input to a 1-D vector of ``N_total`` elements and applies the pointwise
    HardSwish formula.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` product collapsed into ``block_size``.
    - No custom_op wrapper (GPU elementwise parity, E10); ``forward`` calls the
      compiled callable directly.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 2048}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Manifest: float16 | bfloat16 | float32.
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(input)
        # Build the factory callable (lazy -- compilation is deferred to
        # forward()).  The imported function is the NPU implementation
        # (``target="npuir"``), a factory that returns a compiled callable
        # invoked as ``kernel(x, N)``.
        self.kernel = hardswish_fwd_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * npt).

        NPU: a single ``block_size`` for all dtypes, the GPU ``threads * npt``
        product collapsed away (the NPU kernel component re-tunes it).
        """
        # K11: collapse threads * npt into a single NPU block_size.
        return {"block_size": _DEFAULT_BLOCK_SIZE}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the HardSwish kernel.

        Accepts a 1-D contiguous tensor of ``N_total`` elements.
        K9: ``threads`` removed; K11: ``block_size`` passed to the NPU factory
        directly (the factory returns a compiled callable invoked as
        ``kernel(x, N)``; no GPU ``threads/npt`` decomposition).
        """
        block_size = self.config["block_size"]
        return hardswish_fwd_kernel(
            self.N_total, self.dtype_str, block_size=block_size
        )(x, self.N_total)
