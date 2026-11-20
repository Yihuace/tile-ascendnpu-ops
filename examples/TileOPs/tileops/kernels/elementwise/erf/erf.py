"""Erf forward kernel using TileLang (NPU-adapted).

Implements the elementwise error function:
    y = erf(x)

A single-input, single-output elementwise math op.  The input is flattened
to a 1-D vector of ``N_total`` elements and the pointwise erf formula is
applied.  Output dtype is ``same_as(input)``.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel function (``erf_fwd_kernel``, the
    ``_make_unary_regcopy`` builder with ``ErfFwdKernel.op_func`` inlined)
    is extracted from the GPU repo via ``extract_tl_kernel.py`` and
    imported as-is.  It serves as the reference for the NPU kernel
    component to reimplement for ``target="npuir"``.  K1-K4 and K11
    adaptations (decorator, grid/sync, ``threads``/``npt`` removal,
    padding strategy) are handled by the NPU component during
    re-implementation.
      - The extracted ``op_func`` inlines ``T.erf(T.cast(x, "float32"))``;
        the fp32 up-cast is semantically required (the half-precision
        ``herf`` intrinsic is not a valid CUDA/NPU built-in) and must be
        preserved by the NPU re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7:  ``custom_op("npub::...")`` (was ``"top::..."``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed.
    K9:  ``threads`` removed from ``default_config`` and ``forward`` call.
    K11: GPU ``threads * npt`` product collapsed into a single
         ``block_size`` param (elementwise ops only).

The GPU ``ErfFwdKernel`` inherits the ``UnaryKernel`` template via
``FloatUnaryKernel`` (``DEFAULT_STRATEGY = "register_copy"``), so its
extracted factory corresponds to the ``register_copy`` strategy.
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
# .erf_kernel/perf_opt/{func}.py and pass their L0/L1 regression; the
# baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
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
# .erf_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .erf_kernel import erf_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .erf_kernel.perf_opt.erf_fwd_kernel import erf_fwd_kernel, TUNED_BLOCK_SIZE

__all__ = ["ErfFwdKernel"]

# ---------------------------------------------------------------------------
# K11: GPU ``default_config`` = {"threads": 256, "num_per_thread": npt} with
# the strategy-aware npt heuristic (_strategy_npt -> 4 for fp32, 8 for
# fp16/bf16 under "register_copy").  Collapsed to a single per-block element
# count: block_size = threads * npt = 1024 (fp32) / 2048 (fp16, bf16).
# ---------------------------------------------------------------------------
# Stage 4 tuned: per-dtype block_size pinned from the active perf_opt module.
# (paired baseline value: {"float32": 1024, "float16": 2048, "bfloat16": 2048})
_DEFAULT_BLOCK_SIZE = TUNED_BLOCK_SIZE


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed, K11: block_size)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::erf_fwd", mutates_args=())
def _erf_fwd_wrapped(
    N: int,
    dtype_str: str,
    block_size: int,
    x: torch.Tensor,
) -> torch.Tensor:
    # The integrated NPU kernel's prim_func is ``main(x, y, shape)`` with
    # ``out_idx=[1]`` (DESIGN.md section 5.5 / K12): the extra dynamic
    # ``shape`` argument guards the tail tile, so it must be passed alongside
    # x (cf. the bitwise_not glue layer).  ``N`` is both the compile-time
    # length and the runtime logical element count.
    return erf_fwd_kernel(N, dtype_str, block_size=block_size)(x, N)


@_erf_fwd_wrapped.register_fake
def _(N: int, dtype_str: str, block_size: int, x: torch.Tensor):
    return torch.empty((N,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9, K11 adaptations)
# ---------------------------------------------------------------------------


class ErfFwdKernel(Kernel):
    """Erf forward kernel: y = erf(x).

    A single-input, single-output elementwise math op.  The kernel
    flattens the input to a 1-D vector of ``N_total`` elements and applies
    the pointwise erf formula (``T.erf(T.cast(x, "float32"))``).

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` product collapsed into ``block_size``.

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

    # Float-only dtypes (inherited from GPU ``FloatUnaryKernel``).
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
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
        self.output_dtype = dtype  # same_as(input)
        # Build the factory callable (does not compile yet — compilation
        # is deferred to forward() via the custom_op wrapper).  The
        # imported function is the GPU implementation; the NPU component
        # rewrites it for target="npuir" with the K11 signature
        # ``(N, dtype, block_size)``.
        self.kernel = erf_fwd_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def output_dtype_str(self) -> str:
        return self.dtype_to_str(self.output_dtype)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * npt).

        GPU defaults: threads=256, npt=4 (fp32) or npt=8 (fp16/bf16).
        Collapsed: block_size = 1024 (fp32) or 2048 (fp16/bf16).
        """
        # K11: collapse threads * npt into block_size.
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Erf kernel.

        Accepts a 1-D contiguous tensor of ``N_total`` elements.
        K9: ``threads`` removed from the call; K11: ``block_size`` used.
        """
        return _erf_fwd_wrapped(
            self.N_total,
            self.dtype_str,
            self.config["block_size"],
            x,
        )
