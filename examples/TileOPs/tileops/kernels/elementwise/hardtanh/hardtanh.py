"""Hardtanh forward kernel using TileLang (NPU-adapted).

Implements the elementwise Hardtanh (bounded linear) activation:

    y = clamp(x, min_val, max_val)

A single-input, single-output elementwise activation with two compile-time
scalar parameters ``min_val`` / ``max_val``.  The input is flattened to a 1-D
vector of ``N_total`` elements and the pointwise clamp is applied.  Output
dtype is ``same_as(input)``.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang factory (``_make_hardtanh_kernel``) is extracted from the
    GPU repo via ``extract_tl_kernel.py`` and imported as-is.  It serves as the
    reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 and K11 adaptations (decorator, grid/sync,
    ``threads``/``npt`` collapse, padding strategy) are handled by the NPU
    component during re-implementation:
      - E1 (K3/K11): the factory keeps its op-specific ``min_val`` / ``max_val``
        positional arguments but drops ``output_dtype`` / ``is_fp8`` /
        ``threads`` / ``npt`` -> ``_make_hardtanh_kernel(N, dtype, min_val,
        max_val)``.
      - E2 (K11): the returned callable takes a single ``block_size``
        (the GPU ``threads * npt`` product) instead of
        ``(threads_arg, npt_arg)``.
      - E3 (K2): ``T.Kernel(..., is_npu=True)`` replaces the CUDA
        ``threads=`` dimension.
      - E4 (K11): ``T.Parallel(block_size)`` replaces
        ``T.Parallel(threads_arg, npt_arg)``.
    The fp32 transit in the extracted non-fp8 branch
    (``lo = T.cast(min_val, "float32")`` / ``hi = T.cast(max_val, "float32")``
    / ``v32 = T.cast(val, "float32")`` -> ``T.min(T.max(v32, lo), hi)``) is
    semantically required and must be preserved by the NPU re-implementation:
    it reproduces ``torch.nn.functional.hardtanh``'s fp32 clamping (the bounds
    are compared in fp32 before the downcast back to the input dtype).

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7:  ``custom_op("npub::...")`` (the NPU single-input unary convention adds
         the ``npub::`` wrapper).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed.
    K9:  ``threads`` removed from ``default_config`` and ``forward`` call.
    K11: GPU ``threads * npt`` product collapsed into a single ``block_size``
         config key (elementwise ops only).

The GPU ``HardtanhFwdKernel`` inherits ``ParametricUnaryKernel`` (the shared
base for independent parametric elementwise kernels: ``_builder_fn`` /
``_builder_args`` / ``default_config`` / ``forward``).  The NPU project does
not ship that base class, so its logic is inlined here (E6), exactly as done
for the sibling ``EluFwdKernel`` / ``LeakyReluFwdKernel`` /
``LerpTensorFwdKernel``.
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
# .hardtanh_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .hardtanh_kernel import _make_hardtanh_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .hardtanh_kernel.perf_opt._make_hardtanh_kernel import (
    _make_hardtanh_kernel,
    TUNED_BLOCK_SIZE,
)

__all__ = ["HardtanhFwdKernel"]

# ---------------------------------------------------------------------------
# K11: GPU ``default_config`` = {"threads": 256, "num_per_thread": npt} with
# the ParametricUnaryKernel npt heuristic (npt=4 for fp32, npt=8 for
# fp16/bf16).  Collapsed to a single per-block element count:
# block_size = threads * npt = 1024 (fp32) / 2048 (fp16, bf16).
# ---------------------------------------------------------------------------
# Stage 4 (optimize): active source is perf_opt -> use its per-dtype tuned
# defaults.  (Baseline pair was {"float32": 1024, "float16": 2048, "bfloat16": 2048}.)
_DEFAULT_BLOCK_SIZE = TUNED_BLOCK_SIZE


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: npub:: namespace, K9: threads removed, K11: block_size)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::hardtanh_fwd", mutates_args=())
def _hardtanh_fwd_wrapped(
    N: int,
    dtype_str: str,
    block_size: int,
    min_val: float,
    max_val: float,
    x: torch.Tensor,
) -> torch.Tensor:
    # The NPU-reimplemented factory's prim_func is ``main(x, y, shape)`` with
    # ``out_idx=[1]`` (cf. the erf / expm1 / elu / leaky_relu glue layers): the
    # extra dynamic ``shape`` argument guards the tail tile, so it must be
    # passed alongside x.  ``N`` is both the compile-time length and the
    # runtime logical element count; ``min_val`` / ``max_val`` are the clamp
    # bounds.  The extracted GPU factory below is the Part A reference; the NPU
    # component rewrites it with the K11 signature
    # ``(N, dtype, min_val, max_val, block_size)``.
    return _make_hardtanh_kernel(N, dtype_str, min_val, max_val, block_size=block_size)(x, N)


@_hardtanh_fwd_wrapped.register_fake
def _(
    N: int,
    dtype_str: str,
    block_size: int,
    min_val: float,
    max_val: float,
    x: torch.Tensor,
):
    return torch.empty((N,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9, K11 adaptations)
# ---------------------------------------------------------------------------


class HardtanhFwdKernel(Kernel):
    """Hardtanh forward kernel: y = clamp(x, min_val, max_val).

    A single-input, single-output elementwise activation.  The kernel flattens
    the input to a 1-D vector of ``N_total`` elements and applies the pointwise
    two-sided clamp in fp32 (``v32 = T.cast(val, "float32")`` ->
    ``T.min(T.max(v32, lo), hi)``), matching
    ``torch.nn.functional.hardtanh``.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` product collapsed into ``block_size``.
    - E6: the GPU ``ParametricUnaryKernel`` base logic (``_builder_fn`` /
      ``_builder_args`` / ``default_config`` / ``forward``) is inlined.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        min_val: Lower clamp bound (default -1.0), baked into the compiled
            kernel at construction time.
        max_val: Upper clamp bound (default 1.0), baked into the compiled
            kernel at construction time.
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 2048}``).
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
        min_val: float = -1.0,
        max_val: float = 1.0,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        self.output_dtype = dtype  # same_as(input)
        # Build the factory callable (does not compile yet — compilation
        # is deferred to forward() via the custom_op wrapper).  The
        # imported function is the GPU implementation; the NPU component
        # rewrites it for target="npuir" with the K11 signature
        # ``(N, dtype, min_val, max_val, block_size)``.  ``min_val`` /
        # ``max_val`` are the op-specific positional arguments preserved
        # between ``N``/``dtype`` and the common ``block_size`` suffix (GPU
        # ``_builder_args`` -> ``(min_val, max_val)``).
        self.kernel = _make_hardtanh_kernel(
            self.N_total, self.dtype_str, self.min_val, self.max_val
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * npt).

        GPU defaults (ParametricUnaryKernel): threads=256, npt=4 (fp32) or
        npt=8 (fp16/bf16).  Collapsed: block_size = 1024 (fp32) or
        2048 (fp16/bf16).
        """
        # K11: collapse threads * npt into block_size.
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Hardtanh kernel.

        Accepts a 1-D contiguous tensor of ``N_total`` elements.
        K9: ``threads`` removed from the call; K11: ``block_size`` used.
        """
        return _hardtanh_fwd_wrapped(
            self.N_total,
            self.dtype_str,
            self.config["block_size"],
            self.min_val,
            self.max_val,
            x,
        )
