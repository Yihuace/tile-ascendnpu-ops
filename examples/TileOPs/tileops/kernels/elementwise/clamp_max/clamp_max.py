"""ClampMaxFwd forward kernel using TileLang (NPU-adapted).

Implements the Tensor-bound upper clamp ``torch.clamp_max``:

    y = min(x, max)          # NaN-propagating, PyTorch semantics

The Op layer broadcasts ``input`` / ``max`` to the output shape and flattens
them, so the kernel sees two contiguous 1-D tensors of length ``N_total`` and
writes one contiguous 1-D output.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3
(the elementwise family pattern; clamp is a 2-input elementwise map):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_make_clamp_tensor_kernel``, which
    branches on ``has_min`` / ``has_max`` and, for non-fp8 dtypes, uses the
    register-copy strategy with fp32 accumulation) is extracted from the GPU
    repo via ``extract_tl_kernel.py`` and imported as-is.  It serves as the
    reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 and K11 adaptations (decorator, grid/sync,
    ``threads``/``npt`` collapse into ``block_size``, padding strategy) are
    handled by the NPU component during re-implementation:
      - K3/K11: the factory signature drops the GPU ``threads`` /
        ``npt`` execution parameters; the returned callable takes a single
        ``block_size`` (the GPU ``threads * npt`` product).  The NPU
        re-implementation also drops the fp8-only ``output_dtype`` /
        ``is_fp8`` parameters (the manifest contract for this op is
        float16 | bfloat16 | float32 only).

  **Part B -- Kernel class** (ported in full from the GPU
    ``ClampTensorFwdKernel(ParametricUnaryKernel)``):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7/K10: no ``@torch.library.custom_op`` wrapper (the GPU kernel class has
         none either); ``forward`` calls the compiled kernel directly.
    K8/K9: ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
         heuristic config selection only, ``init_config(config)`` takes no
         ``tune`` argument.
    K11: GPU config ``{"threads": 512, "num_per_thread": npt}`` collapsed
         into ``{"block_size": threads * npt}``.  The GPU base
         ``ParametricUnaryKernel.default_config`` uses npt=4 for fp32 and
         npt=8 for fp16/bf16, so the collapsed defaults are
         block_size=2048 (fp32) / 4096 (fp16, bf16).

``has_min`` / ``has_max`` are preserved: ``ClampMaxFwdOp`` constructs the
kernel with ``has_min=False, has_max=True`` (the ``has_max``-only branch of
the extracted factory).  The base ``ParametricUnaryKernel`` logic is inlined
(NPU project ships no such base class); the fp8 output path is skipped
because the manifest dtype contract excludes fp8.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A imports: the extracted GPU TileLang kernel function(s).  The
# conductor's ``integrate_kernel.py`` rewrites this import into the
# baseline/perf_opt source-selection block once the NPU kernel component
# lands the ``target="npuir"`` re-implementation in ``clamp_max_kernel/``.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .clamp_max_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .clamp_max_kernel import _make_clamp_tensor_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .clamp_max_kernel.perf_opt._make_clamp_tensor_kernel import _make_clamp_tensor_kernel
from .clamp_max_kernel.perf_opt._make_clamp_tensor_kernel import TUNED_BLOCK_BY_DTYPE

__all__ = ["ClampTensorFwdKernel"]

# ---------------------------------------------------------------------------
# K11 constants: GPU ``threads * npt`` defaults collapsed into ``block_size``.
#
# The GPU ``ClampTensorFwdKernel`` sets ``_DEFAULT_THREADS = 512`` and keeps
# the base npt heuristic (4 for fp32, 8 for fp16/bf16; fp8 is not in this
# op's manifest contract).  Per the K11 rule ("block_size = the original
# threads * npt product"):
#   fp32:      512 * 4 = 2048
#   fp16/bf16: 512 * 8 = 4096
# ---------------------------------------------------------------------------

_DEFAULT_THREADS = 512
_NPT_FP32 = 4
_NPT_NON_FP32 = 8


def _default_block_size(dtype: torch.dtype) -> int:
    """Stage 4 tuned per-dtype ``block_size`` (perf_opt ``TUNED_BLOCK_BY_DTYPE``).

    Toggle together with the source block above: the baseline (Stage 3) form
    is the K11 GPU ``threads * npt`` product (fp32 2048 / fp16,bf16 4096).
    """
    return TUNED_BLOCK_BY_DTYPE[Kernel.dtype_to_str(dtype)]


class ClampTensorFwdKernel(Kernel):
    """Tensor-bound clamp kernel (``y = clamp(x, lo, hi)``).

    Computes ``y = clamp(x, lo, hi)`` over flat tensors of length
    ``N_total``.  The Op layer broadcasts ``input`` / ``max`` to the output
    shape and flattens them before dispatch.  ``has_min`` / ``has_max``
    select between the three forms used by the Tensor clamp, clamp_min and
    clamp_max ops; ``ClampMaxFwdOp`` uses the ``has_max``-only form.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K7/K10: no custom_op wrapper; ``forward`` dispatches directly.
    - K8/K9: autotune removed; heuristic config selection only,
      ``init_config(config)`` takes no ``tune`` argument.
    - K11: GPU ``threads * npt`` product collapsed into a single
      ``block_size`` config key (2048 for fp32, 4096 for fp16/bf16).

    Args:
        N_total: Total number of (post-broadcast, flattened) elements.
        dtype: Data type (float16, bfloat16, or float32).
        has_min: Whether a lower-bound tensor ``lo`` is present.
        has_max: Whether an upper-bound tensor ``hi`` is present.
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 4096}``).

    Raises:
        ValueError: If neither bound is present, or *dtype* is unsupported.
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer mode
    # regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Manifest: float16 | bfloat16 | float32 (fp8 rejected, GPU parity).
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
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.has_min = bool(has_min)
        self.has_max = bool(has_max)
        # same_as(input); the fp8 output path is skipped (manifest excludes fp8).
        self.output_dtype = dtype
        self._fp8_output_dtype = None
        # Build the factory callable (does not compile yet -- compilation is
        # deferred to forward()).  The imported function is the GPU
        # implementation; the NPU component rewrites it for target="npuir"
        # with the K3/K11 signature ``(N, dtype, has_min, has_max)`` and a
        # returned callable taking a single ``block_size``.
        self.kernel = _make_clamp_tensor_kernel(
            self.N_total,
            self.dtype_str,
            self.has_min,
            self.has_max,
        )
        self._compiled_fn = None
        self._compiled_block_size: Optional[int] = None
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: ``block_size = threads * npt``)."""
        return {"block_size": _default_block_size(self.dtype)}

    def forward(
        self,
        x: torch.Tensor,
        lo: Optional[torch.Tensor] = None,
        hi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the clamp kernel on flat contiguous 1-D tensors.

        Accepts the operands selected by ``has_min`` / ``has_max``:
        ``(x, lo)`` for min-only, ``(x, hi)`` for max-only, ``(x, lo, hi)``
        for the two-sided form.  K11: the (NPU-reimplemented) factory
        callable takes a single ``block_size`` argument and is cached per
        block_size.
        """
        block_size = self.config["block_size"]
        if self._compiled_fn is None or self._compiled_block_size != block_size:
            self._compiled_fn = self.kernel(block_size)
            self._compiled_block_size = block_size
        if self.has_min and self.has_max:
            result = self._compiled_fn(x, lo, hi)
        elif self.has_min:
            result = self._compiled_fn(x, lo)
        else:
            result = self._compiled_fn(x, hi)
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result
