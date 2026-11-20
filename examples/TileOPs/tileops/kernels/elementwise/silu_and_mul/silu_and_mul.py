"""SiluAndMul forward kernel using TileLang (NPU-adapted).

Implements the fused gated activation:
    y = silu(gate) * value

where the single input ``x`` of shape ``(M, 2*N)`` packs ``gate = x[:, :N]``
and ``value = x[:, N:]``; the output has shape ``(M, N)``.  This op is the
activation front-end of a MoE / SwiGLU FFN (``F.silu(x[..., :N]) * x[..., N:]``).

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_make_fused_gated_explicit``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` and imported
    as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 and K11 adaptations
    (decorator, grid/sync, ``threads``/``npt`` collapse, padding strategy)
    are handled by the NPU component during re-implementation:
      - K11: the GPU ``threads * npt`` product collapses into a single
        ``block_size``; the returned callable takes ``block_size`` instead
        of ``(threads_arg, npt_arg)``.
      - ``T.Kernel(..., is_npu=True)`` replaces the CUDA ``threads=``
        dimension; ``T.Parallel(block_size)`` replaces
        ``T.Parallel(threads_arg, npt_arg)``.

  **Part B -- Kernel class** (fully ported from GPU ``SiluAndMulFwdKernel``
  / its base ``FusedGatedKernel``):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
         heuristic config selection only.
    K9/K11: ``threads`` / ``num_per_thread`` removed; ``default_config``
         returns ``{"block_size"}`` (the GPU ``threads * npt`` product).
    E6 (analog): the GPU ``FusedGatedKernel`` strategy machinery
         (``direct`` / ``explicit_parallel``) is collapsed to the extracted
         base class; only the default ``explicit_parallel`` builder is
         extracted, so the kernel targets that single form.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: the GPU TileLang kernel implementation, extracted as-is.  The
# NPU component rewrites ``_make_fused_gated_explicit`` for
# ``target="npuir"``; ``activation_func`` is the verbatim SiLU math applied
# to the gate half (the compound op is ``activation_func(gate) * value``).
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .silu_and_mul_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .silu_and_mul_kernel import _make_fused_gated_explicit
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .silu_and_mul_kernel.perf_opt._make_fused_gated_explicit import _make_fused_gated_explicit

# ``activation_func`` is the wrapper-side ``op_func`` descriptor (the SiLU math
# applied to the gate half).  The NPU kernel inlines the SiLU vector chain and
# treats ``op_func`` as a shape-only identifier (DESIGN.md section 0.6 R3), so
# this helper is never invoked at runtime; the integrated kernel deliberately
# does not re-export it.  It is retained from the frozen Stage-0 contract module
# to keep the wrapper's ``_effective_op_func`` contract intact.  Imported via the
# module object so the kernel-source selection block above stays limited to
# kernel factories (keeps integrate_kernel.py idempotent).
from . import _silu_and_mul_fwd_kernels as _stage0_contract

activation_func = _stage0_contract.activation_func

__all__ = ["SiluAndMulFwdKernel"]


# ---------------------------------------------------------------------------
# K11 constant: GPU ``threads * npt`` default collapsed into ``block_size``.
#
# GPU ``FusedGatedKernel.default_config`` (default strategy
# ``explicit_parallel``):
#   fp16/bf16 explicit_parallel: threads=128, num_per_thread=8 -> 1024
#   fp32 (any strategy):         threads=256, num_per_thread=4 -> 1024
# Collapsed product: block_size = 1024 for every supported dtype.
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 1024


class SiluAndMulFwdKernel(Kernel):
    """SiLU-and-Mul: y = silu(gate) * value, x = [gate, value] of shape (M, 2N).

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; ``init_config(config)`` takes no ``tune`` arg.
    - K9/K11: ``threads``/``npt`` collapsed into a single ``block_size``.
    - E6 (analog): the GPU ``FusedGatedKernel`` strategy selection is
      collapsed to the extracted default (``explicit_parallel``) builder.

    Args:
        M: Number of rows.
        N: Half the packed column dimension (output width).
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 1024}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Float-only dtypes (inherited from GPU FusedGatedKernel subclasses).
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.M = M
        self.N = N
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(x); fp8 path not in the contract
        # Build the factory callable (does not compile yet -- compilation is
        # deferred to forward()).  The imported function is the GPU
        # implementation; the NPU component rewrites it for target="npuir"
        # with the K11 signature ``(M, N, dtype, op_func)`` and a callable
        # that takes a single ``block_size``.
        self.kernel = _make_fused_gated_explicit(
            self.M,
            self.N,
            self.dtype_str,
            self._effective_op_func(),
        )
        self._compiled_fn = None
        self._compiled_block_size: Optional[int] = None
        self.init_config(config)

    @staticmethod
    def _effective_op_func():
        """Return the compound op ``(gate, value) -> silu(gate) * value``.

        Ported from GPU ``FusedGatedKernel._get_effective_op_func`` (the fp8
        cast-in/cast-out wrapping is dropped -- fp8 is outside this op's
        manifest contract).
        """

        def fused_op(gate, value):
            return activation_func(gate) * value

        return fused_op

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * npt)."""
        return {"block_size": _BLOCK_SIZE}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the silu_and_mul kernel.

        Accepts a 2-D contiguous tensor of shape ``(M, 2*N)``.  K11: the
        compiled callable is built lazily with the configured
        ``block_size`` and cached per block_size.
        """
        block_size = self.config["block_size"]
        if self._compiled_fn is None or self._compiled_block_size != block_size:
            self._compiled_fn = self.kernel(block_size)
            self._compiled_block_size = block_size
        return self._compiled_fn(x)
