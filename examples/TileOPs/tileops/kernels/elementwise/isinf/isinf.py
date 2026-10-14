"""Isinf forward kernel using TileLang (NPU-adapted).

Implements the elementwise float predicate ``torch.isinf`` for floating-point
inputs::

    y = isinf(x)    # bool output

The GPU ``IsinfFwdKernel`` is a ``FloatPredicateKernel`` (a ``register_copy``
unary elementwise kernel) whose op body is
``T.isinf(T.cast(x, "float32"))`` and whose declared output dtype is
``torch.bool``.  Because a bool tensor is one byte wide on this backend, the
kernel writes its result through int8 byte storage (``BOOL_STORAGE_DTYPE``); the
host views the byte tensor back to ``bool`` on the way out.  The GPU
``kernel_map`` also declares ``AlwaysFalseFwdKernel`` for the integer / bool
inputs of the ``isinf_exact`` variant -- only the standard float ``isinf`` path
is migrated here.

Predicate semantics (toolchain fact, 2026-10-14): this toolchain's
``T.vcmp "ne"`` lowering is ORDERED (a NaN self-compare answers false), so only
``le`` / ``lt`` / ``eq`` / ``ge`` / ``gt`` semantics are usable.  ``isinf`` is
expressed with the ordered equality against the two infinities --
``(x == +inf) | (x == -inf)`` (the equivalent ``NOT(isfinite) & NOT(isnan)``
form agrees) -- which is correctly false for NaN.  ``T.isinf`` (scalar) and
``vcmp "ne"`` are not used.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md section 3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``isinf_fwd_kernel`` -- the elementwise
    unary body merged with ``IsinfFwdKernel``'s predicate ``op_func`` and the
    bool-as-int8 store) lives in ``_isinf_fwd_kernels.py`` and is imported
    as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 and K11 adaptations (decorator,
    grid/sync, ``threads``/``npt`` collapse, padding strategy) are handled by
    the NPU component during re-implementation.

  **Part B -- Kernel class** (ported from the GPU ``FloatPredicateKernel``
  register_copy flow):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
         heuristic config selection only.
    K9/K11: ``threads`` / ``npt`` removed; ``default_config`` returns ``{}`` and
         ``forward`` does not bake a ``block_size`` -- the NPU factory's own
         default governs (per the K11 wrapper rule).
    (No ``@torch.library.custom_op`` wrapper -- the GPU elementwise kernel has
    none either; ``forward`` dispatches the compiled callable directly.)
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Part A -- imported TileLang kernel function (GPU reference implementation).
#
# ``isinf_fwd_kernel`` is the extracted GPU implementation, hand-merged for the
# predicate op body + bool-as-int8 store (the extractor cannot statically inline
# a registry-resolved op body).  The NPU kernel component rewrites it for
# ``target="npuir"``; the import name and call shape (factory -> compiled
# callable) stay unchanged.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .isinf_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .isinf_kernel import isinf_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .isinf_kernel.perf_opt.isinf_fwd_kernel import isinf_fwd_kernel

__all__ = ["IsinfFwdKernel"]


class IsinfFwdKernel(Kernel):
    """Isinf forward kernel: ``y = isinf(x)`` (bool output).

    A single-input, single-output elementwise float predicate.  The kernel
    flattens the input to a 1-D vector of ``N_total`` elements and reports,
    per element, whether the value is infinite.  The output is ``torch.bool``;
    the compiled kernel writes it through int8 byte storage and ``forward``
    views it back to bool.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9/K11: ``threads`` removed; ``default_config`` returns ``{}`` and
      ``forward`` leaves ``block_size`` to the NPU factory's own default (the
      GPU ``threads * npt`` product is a CUDA-execution-model artifact).
    - No custom_op wrapper (GPU elementwise parity); ``forward`` calls the
      compiled callable directly.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Input data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict.
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Manifest: the NPU port follows the float workloads (the GPU union also
    # lists int / bool, served by the un-migrated ``isinf_exact`` fallback).
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
        self.output_dtype = torch.bool  # predicate output
        # Build the factory callable (lazy -- compilation is deferred to
        # forward()).  The imported function is the GPU reference
        # implementation; the NPU component rewrites it for target="npuir".
        self.kernel = isinf_fwd_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config.

        K11 wrapper rule: no ``block_size`` is baked here -- the NPU factory's
        own default governs, so no per-op Stage-5 glue fix is needed.
        """
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the isinf kernel.

        Accepts a 1-D contiguous float tensor of ``N_total`` elements and
        returns a 1-D bool tensor of the same length.  K9/K11: the factory is
        invoked without ``threads`` / ``npt`` / ``block_size``; its own NPU
        default governs.

        Bool storage glue: the compiled kernel writes the predicate result
        through int8 byte storage (the GPU ``BOOL_STORAGE_DTYPE``).  ``view`` is
        a metadata-only reinterpretation, so the byte content is unchanged and
        the bool result is bit-exact.
        """
        y = isinf_fwd_kernel(self.N_total, self.dtype_str)(x, self.N_total)
        return y.view(torch.bool)
