"""Hardsigmoid forward kernel using TileLang (NPU-adapted).

Implements the elementwise HardSigmoid activation::

    y = clamp(x + 3, 0, 6) * (1 / 6)

Scaling by the reciprocal matches ``torch.nn.functional.hardsigmoid``; a NaN
input propagates NaN, as in torch.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel function (``hardsigmoid_fwd_kernel``) is extracted
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
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .hardsigmoid_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .hardsigmoid_kernel import hardsigmoid_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .hardsigmoid_kernel.perf_opt.hardsigmoid_fwd_kernel import hardsigmoid_fwd_kernel

__all__ = ["HardsigmoidFwdKernel"]

# ---------------------------------------------------------------------------
# K11 constants: GPU ``threads * npt`` collapsed into a single ``block_size``.
#
# The GPU ``FloatUnaryKernel`` uses the ``register_copy`` strategy with the
# elementwise defaults in ``_policy.py``:
#   threads = _DEFAULT_THREADS = 128
#   npt     = max(MIN_NUM_PER_THREAD=4, BYTES_PER_THREAD=16 // elem_bytes)
#             -> 4 for fp32 (4 bytes), 8 for fp16/bf16 (2 bytes)
# Per K11 ("block_size = the original threads * npt product"):
#   fp32:      128 * 4 = 512
#   fp16/bf16: 128 * 8 = 1024
# ---------------------------------------------------------------------------

_DEFAULT_THREADS = 128
_BLOCK_SIZE_FP32 = 512
_BLOCK_SIZE_NON_FP32 = 1024


class HardsigmoidFwdKernel(Kernel):
    """Hardsigmoid forward kernel: y = clamp(x + 3, 0, 6) * (1 / 6).

    A single-input, single-output elementwise activation.  The kernel flattens
    the input to a 1-D vector of ``N_total`` elements and applies the pointwise
    HardSigmoid formula (NaN-propagating, as in torch).

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
            ``{"block_size": 1024}``).
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
        # Build the factory callable (does not compile yet -- compilation is
        # deferred to forward()).  The imported function is the GPU
        # implementation; the NPU component rewrites it for target="npuir".
        self.kernel = hardsigmoid_fwd_kernel(self.N_total, self.dtype_str)
        self._compiled_fn = None
        self._compiled_block_size: Optional[int] = None
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * npt).

        GPU defaults: threads=128, npt=4 (fp32) or npt=8 (fp16/bf16).
        Collapsed: block_size = 512 (fp32) or 1024 (fp16/bf16).
        """
        # K11: collapse threads * npt into block_size.
        if self.dtype == torch.float32:
            return {"block_size": _BLOCK_SIZE_FP32}
        return {"block_size": _BLOCK_SIZE_NON_FP32}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Hardsigmoid kernel.

        Accepts a 1-D contiguous tensor of ``N_total`` elements.
        K9: ``threads`` removed; K11: ``block_size`` decomposed back into the
        GPU factory's ``(threads, npt)`` (threads is fixed at the GPU
        register_copy default of 128).  The compiled callable is cached per
        block_size.
        """
        block_size = self.config["block_size"]
        threads = _DEFAULT_THREADS
        npt = block_size // threads
        if self._compiled_fn is None or self._compiled_block_size != block_size:
            self._compiled_fn = self.kernel(threads, npt)
            self._compiled_block_size = block_size
        return self._compiled_fn(x)
