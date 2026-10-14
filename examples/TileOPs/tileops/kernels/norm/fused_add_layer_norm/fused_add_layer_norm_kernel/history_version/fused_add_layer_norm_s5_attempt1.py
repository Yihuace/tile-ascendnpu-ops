"""Fused Add + LayerNorm forward kernel using TileLang (NPU-adapted).

Computes ``y = LayerNorm(x + residual)`` and additionally returns the pre-norm
residual sum ``residual_out = x + residual``, in one fused row-reduction kernel.
Fusing the residual add into the normalization kernel saves one global memory
round-trip versus a separate add + norm, and returning the sum lets downstream
residual connections reuse it without recomputation.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_fused_add_layer_norm_kernel``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` into
    ``_fused_add_layer_norm_fwd_kernels.py`` and imported as-is, together with
    its ``align_up`` helper and the ``ALIGNMENT`` constant.  It is a
    row-reduction builder (a ``(block_m, N_padded)`` register fragment reduced
    along the row) that declares two outputs (``out_idx=[4, 5]``).  It serves as
    the reference for the NPU kernel component to reimplement for
    ``target="npuir"``; K1-K4 adaptations (decorator, grid/sync semantics,
    ``threads`` removal, padding policy) are handled by that component during
    re-implementation.

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8: ``autotune_configs`` / ``autotune()`` / ``tune`` removed; heuristic
        config selection only, ``init_config(config)``.
    K9/K11: ``default_config`` returns ``{}``; ``forward`` invokes the factory
        without baking a ``block_m`` / ``threads`` -- the NPU factory's own
        default governs (the GPU ``threads`` is a CUDA-execution-model artifact).
    The GPU ``@torch.library.custom_op`` wrapper and the CUDA-only
    ``vector_aligned`` / ``_require_cuda`` helpers are dropped: the NPU port
    follows the direct-call elementwise/row-reduction precedent (no
    ``torch.compile`` dispatch boundary on NPU, adaptation C23).
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
# .fused_add_layer_norm_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .fused_add_layer_norm_kernel import _fused_add_layer_norm_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .fused_add_layer_norm_kernel.perf_opt._fused_add_layer_norm_kernel import _fused_add_layer_norm_kernel

__all__ = ["FusedAddLayerNormKernel"]


class FusedAddLayerNormKernel(Kernel):
    """Fused Add + LayerNorm forward kernel.

    A four-input, two-output row-reduction op.  Each row of ``N`` elements is::

        s = x + residual                  # residual_out (rounded to input dtype)
        y = (s - mean(s)) / sqrt(var(s) + eps) * weight + bias

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only,
      ``init_config(config)`` takes no ``tune`` argument.
    - K9/K11: ``default_config`` returns ``{}`` and ``forward`` leaves the
      execution block size to the NPU factory's own default.
    - No custom_op wrapper (NPU parity with the elementwise/row-reduction
      precedent); ``forward`` calls the factory directly.

    Args:
        M: Number of rows (product of all leading dims of ``x``).
        N: Hidden dimension (last dim of ``x``).
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional kernel configuration override.
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Manifest: float32 | float16 | bfloat16.
    SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

    def __init__(
        self,
        M: int,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.M = M
        self.N = N
        self.eps = eps
        self.dtype = dtype
        # Build the factory callable (compilation is deferred to forward()).
        # The imported function is the GPU reference implementation; the NPU
        # component rewrites it for target="npuir" with the same leading
        # interface ``(M, N, eps, dtype)``.
        self.kernel = _fused_add_layer_norm_kernel(self.M, self.N, self.eps, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config.

        K11 wrapper rule: no execution parameter is baked here -- the NPU
        factory's own default governs, so no per-op Stage-5 glue fix is needed.
        """
        return {}

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run the fused add + LayerNorm kernel.

        Flattens the leading dims of *x* / *residual* into ``M`` rows of ``N``
        and normalizes each row.  Returns ``[y, residual_out]``, both shaped like
        *x* (``residual_out`` being the pre-norm sum ``x + residual``).

        K9/K11: the factory is invoked without ``threads`` / ``block_m``; its own
        NPU default governs.
        """
        original_shape = x.shape
        rows = x.contiguous().reshape(-1, self.N)
        residual = residual.contiguous().reshape(-1, self.N)
        weight = weight.contiguous().reshape(self.N)
        bias = bias.contiguous().reshape(self.N)

        outputs = _fused_add_layer_norm_kernel(
            rows.shape[0], self.N, self.eps, self.dtype_str
        )(rows, residual, weight, bias)
        return [out.reshape(original_shape) for out in outputs]
