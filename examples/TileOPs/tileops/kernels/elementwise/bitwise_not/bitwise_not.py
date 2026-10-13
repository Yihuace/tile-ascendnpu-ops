"""Bitwise NOT forward kernel using TileLang (NPU-adapted).

Implements the elementwise bitwise-NOT of ``torch.bitwise_not`` for
bool / integer inputs::

    y = ~x

The GPU ``BitwiseNotFwdKernel`` takes the base ``UnaryKernel`` looping
strategy (``register_copy``; ``direct`` for a bool operand) and a
dtype-branched ``op_func`` (``x == 0`` for bool, ``xor(x, 255)`` for uint8,
``xor(x, -1)`` otherwise) rather than a single expression, so the per-element
math differs by dtype but the result is always the all-ones complement.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md section 3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``bitwise_not_fwd_kernel`` -- the
    elementwise unary body merged with ``BitwiseNotFwdKernel``'s dtype-branched
    ``op_func``) lives in ``_bitwise_not_fwd_kernels.py`` and is imported as-is.
    It serves as the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 and K11 adaptations (decorator, grid/sync,
    ``threads``/``npt`` collapse, padding strategy) are handled by the NPU
    component during re-implementation.

  **Part B -- Kernel class** (ported from the GPU ``UnaryKernel`` flow):
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
# ``bitwise_not_fwd_kernel`` is the extracted GPU implementation imported as-is
# (hand-merged for the dtype-branched op body, which the extractor cannot inline
# statically).  The NPU kernel component rewrites it for ``target="npuir"``; the
# import name and call shape (factory -> compiled callable) stay unchanged.
# ---------------------------------------------------------------------------
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
from .bitwise_not_kernel import bitwise_not_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .bitwise_not_kernel.perf_opt.bitwise_not_fwd_kernel import bitwise_not_fwd_kernel

__all__ = ["BitwiseNotFwdKernel"]

# ---------------------------------------------------------------------------
# K11 constant: UB tile width ``block_size``.
#
# The NPU value follows the elementwise family design (the same element-wise
# best value measured for the mish / hardsigmoid / hardswish / add / sub
# family): block_size = 8192, 32B aligned for both the 1-byte bool/uint8/int8
# and the 4-/8-byte integer dtypes, with a UB budget below 192KB.  The GPU
# ``UnaryKernel`` register_copy ``threads * npt`` collapse is superseded by this
# design decision.
# ---------------------------------------------------------------------------

_DEFAULT_BLOCK_SIZE = {
    "bool": 8192,
    "uint8": 8192,
    "int8": 8192,
    "int16": 8192,
    "int32": 8192,
    "int64": 8192,
}


class BitwiseNotFwdKernel(Kernel):
    """Bitwise NOT forward kernel: ``y = ~x`` (bool / integer inputs).

    A single-input, single-output elementwise op.  The kernel flattens the
    input to a 1-D vector of ``N_total`` elements and applies the elementwise
    bitwise-NOT (all-ones complement).

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
        dtype: Data type (bool, uint8, int8, int16, int32, or int64).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 8192}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Manifest: bool | uint8 | int8 | int16 | int32 | int64 (BITWISE_DTYPES).
    SUPPORTED_DTYPES = (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )

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
        self.kernel = bitwise_not_fwd_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (``block_size = 8192``)."""
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the bitwise-NOT kernel.

        Accepts a 1-D contiguous tensor of ``N_total`` elements.  K9:
        ``threads`` removed; K11: the factory is rebuilt with the configured
        ``block_size`` (the GPU ``threads``/``npt`` decomposition collapses into
        the single NPU ``block_size``).

        Byte-storage glue (DESIGN.md sections 3.3 / 11.4): the NPU kernel
        load/stores bool (i1 cannot load/store GM) and uint8 (no runtime
        output-dtype map) through an ``int8`` byte tensor.  ``Tensor.view`` is a
        metadata-only reinterpretation, so the byte content is unchanged and the
        result stays bit-exact for every admitted dtype.
        """
        block_size = self.config["block_size"]
        x_bytes = (
            x.contiguous().view(torch.int8)
            if self.dtype in (torch.bool, torch.uint8)
            else x
        )
        y = bitwise_not_fwd_kernel(
            self.N_total, self.dtype_str, block_size=block_size
        )(x_bytes, self.N_total)
        if self.dtype == torch.bool:
            return y.view(torch.bool)
        if self.dtype == torch.uint8:
            return y.view(torch.uint8)
        return y
