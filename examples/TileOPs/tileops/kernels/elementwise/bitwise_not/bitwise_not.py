"""BitwiseNot forward kernel using TileLang (NPU-adapted).

Implements the elementwise bitwise NOT for bool / integer inputs:
    y = ~x
computed as an XOR with all-ones (``T.bitwise_xor(x, -1)``) because
``T.bitwise_not`` fails on vectorized ``int4`` CUDA types in the GPU
implementation.  For ``uint8`` the mask is ``255``; for ``bool`` the op is
``x == False``.  Output dtype is ``same_as(input)``.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel function (``bitwise_not_fwd_kernel``, the
    ``_make_unary_direct`` builder with ``BitwiseNotFwdKernel.op_func``
    inlined) is extracted from the GPU repo via ``extract_tl_kernel.py``
    and imported as-is.  It serves as the reference for the NPU kernel
    component to reimplement for ``target="npuir"``.  K1-K4 and K11
    adaptations (decorator, grid/sync, ``threads``/``npt`` removal,
    padding strategy) are handled by the NPU component during
    re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7:  ``custom_op("npub::...")`` (was ``"top::..."``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed.
    K9:  ``threads`` removed from ``default_config`` and ``forward`` call.
    K11: GPU ``threads * npt`` product collapsed into a single
         ``block_size`` param (elementwise ops only).

The GPU ``BitwiseNotFwdKernel`` inherits the ``UnaryKernel`` template
(DEFAULT_STRATEGY ``"direct"``, 1 element per thread).  Its extracted
factory therefore corresponds to the ``direct`` strategy only, so the NPU
class keeps no strategy dispatch: the single imported builder is the whole
kernel surface.
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
# .bitwise_not_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .bitwise_not_kernel import bitwise_not_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .bitwise_not_kernel.perf_opt.bitwise_not_fwd_kernel import (
    bitwise_not_fwd_kernel,
    TUNED_BLOCK_SIZE,
)

__all__ = ["BitwiseNotFwdKernel"]

# K11: GPU ``default_config`` = {"threads": 256, "num_per_thread": npt} with
# the strategy-aware npt heuristic (_strategy_npt -> 8 for every bitwise
# dtype, since none is float32 and the strategy is not explicit_parallel).
# Collapsed to a single per-block element count: block_size = 256 * 8 = 2048.
# Stage 4 tuned: per-dtype block_size pinned from the active perf_opt module
# (paired switch -- baseline scalar pair is ``_DEFAULT_BLOCK_SIZE = 2048``).
_DEFAULT_BLOCK_SIZE = TUNED_BLOCK_SIZE


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed, K11: block_size)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::bitwise_not_fwd", mutates_args=())
def _bitwise_not_fwd_wrapped(
    N: int,
    dtype_str: str,
    block_size: int,
    x: torch.Tensor,
) -> torch.Tensor:
    # The integrated NPU kernel's prim_func is ``main(x, y, shape)`` (DESIGN
    # section 3, K12): the extra dynamic ``shape`` argument guards the tail
    # tile, so it must be passed alongside x.
    kernel = bitwise_not_fwd_kernel(N, dtype_str, block_size=block_size)
    if dtype_str == "bool":
        # R-1: bool (i1) is not GM load/store-able on NPU, so the kernel
        # computes on the 1-byte int8 storage view; view in and back out
        # (host-side reinterpretation only -- no content/stride change).
        y = kernel(x.view(torch.int8), N)
        return y.view(torch.bool)
    y = kernel(x, N)
    if dtype_str == "uint8":
        # R-2 / CG-2026-0014: the out_idx dtype map lacks uint8, so the
        # auto-allocated output buffer is float32; the kernel still wrote
        # the correct N bytes at the buffer base -- recover them by
        # reinterpreting the bytes back to uint8.
        return y.view(torch.uint8)[:N]
    return y


@_bitwise_not_fwd_wrapped.register_fake
def _(N: int, dtype_str: str, block_size: int, x: torch.Tensor):
    return torch.empty((N,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9, K11 adaptations)
# ---------------------------------------------------------------------------


class BitwiseNotFwdKernel(Kernel):
    """Bitwise NOT forward kernel: y = ~x for bool / integer inputs.

    A single-input, single-output elementwise op.  The kernel flattens the
    input to a 1-D vector of ``N_total`` elements and applies the pointwise
    bitwise-NOT formula.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` product collapsed into ``block_size``.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (bool, uint8, int8, int16, int32, or int64).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 2048}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Ported from the GPU ``_BITWISE_DTYPES`` tuple on ``UnaryKernel``.
    SUPPORTED_DTYPES = (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    # GPU BitwiseNotFwdKernel uses the 1-element-per-thread "direct" strategy.
    DEFAULT_STRATEGY = "direct"

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
        # Build the factory callable (does not compile yet — compilation is
        # deferred to forward() via the custom_op wrapper).
        self.kernel = bitwise_not_fwd_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def output_dtype_str(self) -> str:
        return self.dtype_to_str(self.output_dtype)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * npt).

        Stage 4: when the active source is perf_opt, ``_DEFAULT_BLOCK_SIZE``
        is a per-dtype mapping (from ``TUNED_BLOCK_SIZE``); the baseline pair
        is a scalar.  Both forms are handled so the switch stays reversible.
        """
        bs = _DEFAULT_BLOCK_SIZE
        return {"block_size": bs[self.dtype_str] if isinstance(bs, dict) else bs}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the bitwise-NOT kernel.

        Accepts a 1-D contiguous tensor of ``N_total`` elements.
        K9: ``threads`` removed from the call; K11: ``block_size`` used.
        """
        return _bitwise_not_fwd_wrapped(
            self.N_total,
            self.dtype_str,
            self.config["block_size"],
            x,
        )
