"""ClampScalar forward kernel using TileLang (NPU-adapted).

Implements the scalar-bound clamp (``torch.clamp(input, min: Number|None,
max: Number|None)``)::

    y = clamp(x, min, max)     # a bound that is None is not applied

The comparison runs in float32 and the result is cast back, so a half input
keeps the precision of the comparison; NaN in the input or in a bound propagates
NaN, as in ``torch.clamp``.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §2/§3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``clamp_scalar_fwd_kernel`` -- the
    ``make_unary_regcopy`` builder merged with ``ClampFwdKernel``'s dynamic
    ``_make_op_func`` op body) lives in ``_clamp_scalar_fwd_kernels.py`` and is
    imported as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 and K11 adaptations (decorator,
    grid/sync, ``threads``/``npt`` collapse, padding strategy) are handled by
    the NPU component during re-implementation.

  **Part B -- Kernel class** (ported from the GPU ``ScalarParamUnaryKernel``
  register_copy flow):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
         heuristic config selection only.
    K9:  ``threads`` removed from ``default_config`` and ``forward``.
    K11: ``threads * npt`` collapsed into a single ``block_size`` config key.
    (No ``@torch.library.custom_op`` wrapper -- the GPU elementwise kernel has
    none; ``forward`` dispatches the compiled callable directly.)
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .clamp_scalar_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .clamp_scalar_kernel import clamp_scalar_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .clamp_scalar_kernel.perf_opt.clamp_scalar_fwd_kernel import clamp_scalar_fwd_kernel

__all__ = ["ClampScalarFwdKernel"]

# ---------------------------------------------------------------------------
# K11 constant: UB tile width ``block_size``.
#
# Stage 5 integration takes the NPU value from the Stage 1 design
# (clamp_scalar_fwd_kernel_DESIGN.md §5.2): block_size = 8192 (the same
# element-wise best value measured for the hardsigmoid/mish family; 32B
# aligned for both fp16 and fp32; UB budget ~136-168KB < 192KB).  The GPU
# ``threads * npt`` collapse (fp32 1024, fp16/bf16 2048) is superseded by
# this design decision.
# ---------------------------------------------------------------------------

# fp32 uses a smaller tile: the kernel allocates six [block_size] UB buffers
# (x_ub/x32/buf1/buf2/nan_mask/y_ub) and the fp32 path keeps x_ub in fp32 too,
# so at block_size=8192 fp32 needs ~225 KB with Expert-mode multi-buffering
# (>192 KB UB) and fails to compile when the msprof/benchmark path builds it
# under the ambient Expert mode; 4096 fits under both Developer and Expert.
# fp16/bf16 keep 8192 (their x_ub/y_ub are half-width, ~181 KB worst case).
_DEFAULT_BLOCK_SIZE = {
    "float32": 4096,
    "float16": 8192,
    "bfloat16": 8192,
}


class ClampScalarFwdKernel(Kernel):
    """Scalar-bound clamp forward kernel: ``y = clamp(x, min, max)``.

    A single-input, single-output elementwise op.  The kernel flattens the
    input to a 1-D vector of ``N_total`` elements and applies
    ``torch.clamp(x, min, max)`` with whichever bounds were given.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` product collapsed into ``block_size``.
    - No custom_op wrapper (GPU elementwise parity); ``forward`` calls the
      compiled callable directly.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        min_val: Lower bound (Number or None).
        max_val: Upper bound (Number or None); at least one must be given.
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
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if min_val is None and max_val is None:
            raise ValueError("ClampScalarFwdKernel requires present(min) or present(max)")
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        self.output_dtype = dtype  # same_as(input)
        # Build the factory callable (does not compile yet -- compilation is
        # deferred to forward()).  The imported function is the GPU reference
        # implementation; the NPU component rewrites it for target="npuir".
        # Pass the dtype-appropriate block_size here too: the factory call
        # materializes a kernel object, and leaving block_size to the factory
        # default (8192) overflows UB on the fp32 path when built under the
        # ambient Expert mode (multi-buffer); forward() rebuilds with the same
        # configured value.
        self.kernel = clamp_scalar_fwd_kernel(
            self.N_total,
            self.dtype_str,
            min_val=self.min_val,
            max_val=self.max_val,
            block_size=_DEFAULT_BLOCK_SIZE[self.dtype_str],
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (``block_size = 8192``, DESIGN §5.2)."""
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the clamp kernel.

        Accepts a 1-D contiguous tensor of ``N_total`` elements.  K9:
        ``threads`` removed; K11: the factory is rebuilt with the configured
        ``block_size`` (the GPU ``threads``/``npt`` decomposition collapses into
        the single NPU ``block_size``).
        """
        block_size = self.config["block_size"]
        return clamp_scalar_fwd_kernel(
            self.N_total,
            self.dtype_str,
            min_val=self.min_val,
            max_val=self.max_val,
            block_size=block_size,
        )(x, self.N_total)
