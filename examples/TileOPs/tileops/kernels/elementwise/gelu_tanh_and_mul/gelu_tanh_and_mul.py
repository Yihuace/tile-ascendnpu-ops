"""GeluTanhAndMul forward kernel using TileLang (NPU-adapted).

Implements the fused gated tanh-approximate-GELU activation (flashinfer
``gelu_tanh_and_mul``)::

    gate, value = x[..., :N], x[..., N:]        # x has shape (M, 2 * N)
    y = gelu_tanh(gate) * value                  # gelu is the tanh approximation

with ``gelu_tanh(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3)))``.

The Op layer hands the kernel the raw ``(M, 2 * N)`` tensor; the kernel reads
the gate from the first ``N`` columns and the value from the last ``N``.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``gelu_tanh_and_mul_fwd_kernel`` -- the
    ``make_fused_gated_explicit`` builder merged with
    ``GeluTanhAndMulFwdKernel.activation_func`` inlined and fused as
    ``gelu_tanh(gate) * value``) lives in
    ``_gelu_tanh_and_mul_fwd_kernels.py`` and is imported as-is.  It serves as
    the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 and K11 adaptations (decorator, grid/sync,
    ``threads``/``npt`` collapse, padding strategy) are handled by the NPU
    component during re-implementation.

  **Part B -- Kernel class** (ported from the GPU ``FusedGatedKernel`` /
  ``GeluTanhAndMulFwdKernel`` explicit_parallel flow; E5-E10 style):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
         heuristic config selection only.
    K9:  ``threads`` removed from ``default_config`` and ``forward``.
    K11: GPU ``threads * npt`` collapsed into a single ``block_size``; the
         whole collapse is left to the NPU factory's own default, so
         ``default_config`` returns ``{}`` and ``forward`` does not bake a
         ``block_size`` (per the K11 wrapper rule -- baking the CUDA product
         would force a Stage-5 glue fix).
    E10: no ``@torch.library.custom_op`` wrapper (the GPU FusedGated kernel has
         none); ``forward`` dispatches the compiled callable directly.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Part A -- imported TileLang kernel function (GPU reference implementation).
#
# ``gelu_tanh_and_mul_fwd_kernel`` is the extracted GPU implementation imported
# as-is.  The NPU kernel component rewrites it for ``target="npuir"``; the
# import name and call shape (factory -> compiled callable) stay unchanged.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .gelu_tanh_and_mul_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .gelu_tanh_and_mul_kernel import gelu_tanh_and_mul_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .gelu_tanh_and_mul_kernel.perf_opt.GeluTanhAndMulFwdOp import gelu_tanh_and_mul_fwd_kernel

__all__ = ["GeluTanhAndMulFwdKernel"]


class GeluTanhAndMulFwdKernel(Kernel):
    """Fused gated tanh-GELU forward kernel: ``y = gelu_tanh(gate) * value``.

    A single-input, single-output fused gated op.  The input ``x`` has shape
    ``(M, 2 * N)``: the gate is the first ``N`` columns and the value the last
    ``N``.  The output has shape ``(M, N)``.  GELU uses the tanh approximation,
    matching ``torch.nn.functional.gelu(x, approximate="tanh")``.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` collapsed into a single ``block_size`` whose
      value the NPU factory owns (``default_config`` returns ``{}``).
    - E10: no custom_op wrapper; ``forward`` dispatches the compiled callable
      directly.

    Args:
        M: Number of rows of ``x`` (and of the output).
        N: Output width; ``x`` has ``2 * N`` columns.
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict.
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
        M: int,
        N: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.M = int(M)
        self.N = int(N)
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(input)
        # Build the factory callable (lazy -- compilation is deferred to
        # forward()).  The imported function is the GPU reference
        # implementation; the NPU component rewrites it for target="npuir".
        self.kernel = gelu_tanh_and_mul_fwd_kernel(self.M, self.N, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config.

        K11: no ``block_size`` is baked here -- the NPU factory's own default
        (DESIGN §5.2) governs, so the CUDA ``threads * npt`` product is not
        carried across the migration.
        """
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fused gated tanh-GELU kernel.

        Accepts a contiguous 2-D tensor ``x`` of shape ``(M, 2 * N)`` and
        returns ``(M, N)``.  K9/K11: the factory is invoked without
        ``threads`` / ``npt`` / ``block_size``; its own NPU default governs.
        """
        return gelu_tanh_and_mul_fwd_kernel(self.M, self.N, self.dtype_str)(x)
