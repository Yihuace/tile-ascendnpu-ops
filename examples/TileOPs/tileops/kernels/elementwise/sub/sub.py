"""Sub forward kernel using TileLang (NPU-adapted).

Implements the scalar-alpha binary subtraction of ``torch.sub``::

    y = a - alpha * b

with ``alpha`` baked into the kernel (one specialization per distinct value).
Float operands take an fp32 intermediate for the scalar multiply, so a half
input keeps the precision of the scaled term; integer / bool operands keep
native integer arithmetic.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md sections 2/3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``sub_fwd_kernel`` -- the
    ``make_binary_register_copy`` builder merged with ``SubFwdKernel``'s
    dynamic ``AlphaScaledBinaryKernel._alpha_op_func`` op body) lives in
    ``_sub_fwd_kernels.py`` and is imported as-is.  It serves as the reference
    for the NPU kernel component to reimplement for ``target="npuir"``.  K1-K4
    and K11 adaptations (decorator, grid/sync, ``threads``/``npt`` collapse,
    padding strategy) are handled by the NPU component during re-implementation.

  **Part B -- Kernel class** (ported from the GPU ``AlphaScaledBinaryKernel``
  register_copy flow):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
         heuristic config selection only.
    K9:  ``threads`` removed from ``default_config`` and ``forward``.
    K11: ``threads * npt`` collapsed into a single ``block_size`` config key.
    (No ``@torch.library.custom_op`` wrapper -- the GPU elementwise kernel has
    none; ``forward`` dispatches the compiled callable directly.)

Binary + broadcast note: the Op layer expands both operands to
``broadcast_shapes(input, other)`` and flattens them, so this kernel only ever
sees two same-shape contiguous 1-D tensors of ``N_total`` elements (the GPU
``register_copy`` case).  The broadcast contract is preserved by ``SubFwdOp``.
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
# .sub_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .sub_kernel import sub_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .sub_kernel.perf_opt.sub_fwd_kernel import sub_fwd_kernel

__all__ = ["SubFwdKernel"]

# ---------------------------------------------------------------------------
# K11 constant: UB tile width ``block_size``.
#
# The NPU value comes from the elementwise family design (same element-wise
# best value measured for the mish / hardsigmoid / hardswish / add family):
# block_size = 8192, 32B aligned for both fp16 and fp32 with a UB budget below
# 192KB.  The GPU ``make_binary_register_copy`` default ``threads * npt``
# collapse is superseded by this design decision.
# ---------------------------------------------------------------------------

_DEFAULT_BLOCK_SIZE = {
    "float32": 8192,
    "float16": 8192,
    "bfloat16": 8192,
}


class SubFwdKernel(Kernel):
    """Scalar-alpha binary subtraction forward kernel: ``y = a - alpha * b``.

    A two-input, single-output elementwise op.  The kernel receives two flat,
    contiguous, same-shape 1-D vectors of ``N_total`` elements (the Op layer
    resolves broadcasting) and applies ``y = a - alpha * b`` with ``alpha``
    baked in.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` product collapsed into ``block_size``.
    - No custom_op wrapper (GPU elementwise parity); ``forward`` calls the
      compiled callable directly.

    Args:
        N_total: Total number of elements (flattened, post-broadcast).
        dtype: Data type (float16, bfloat16, or float32).
        alpha: Multiplier on ``other`` (default 1).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 8192}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Manifest: float16 | bfloat16 | float32 (the GPU op's full union also
    # covers ints; this NPU port follows the float subset -- the migration
    # targets).
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        alpha: "int | float" = 1,
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
        self.alpha = alpha
        self.output_dtype = dtype  # same_as(input)
        # Build the factory callable (does not compile yet -- compilation is
        # deferred to forward()).  The imported function is the GPU reference
        # implementation; the NPU component rewrites it for target="npuir".
        self.kernel = sub_fwd_kernel(self.N_total, self.dtype_str, alpha=self.alpha)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (``block_size = 8192``, DESIGN section 5.2)."""
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run the subtraction kernel.

        Accepts two 1-D contiguous tensors of ``N_total`` elements.  K9:
        ``threads`` removed; K11: the factory is rebuilt with the configured
        ``block_size`` (the GPU ``threads``/``npt`` decomposition collapses into
        the single NPU ``block_size``).
        """
        block_size = self.config["block_size"]
        return sub_fwd_kernel(
            self.N_total,
            self.dtype_str,
            alpha=self.alpha,
            block_size=block_size,
        )(a, b, self.N_total)
