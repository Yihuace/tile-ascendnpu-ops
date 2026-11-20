"""Fused Add RMSNorm kernel using TileLang (NPU-adapted).

    s            = x + residual
    rstd         = 1 / sqrt(mean(s^2) + eps)
    y            = s * rstd * weight
    residual_out = s

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + reimplemented):
    The GPU kernel function (``_fused_add_rms_norm_kernel``) is extracted via
    ``.agents/skills/add-npu-op/scripts/extract_tl_kernel.py`` and preserved in
    ``_fused_add_r_m_s_norm_fwd_kernels.py`` as the porting reference.  The
    npuir re-implementation lives in ``_fused_add_rms_norm_kernel/`` and
    follows the in-repo NPU reference
    ``examples/norm/fused_add_rms_norm/kernels/single_n.py``.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was a CUDA SM list).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: autotune removed -- heuristic config selection only.
    K9: ``threads`` removed (the GPU factory's ``(block_m, threads)`` callable
        becomes ``(block_m)``; rows are processed one per core).

  **TWO outputs**: like the GPU source (``out_idx=[3, 4]``), ``forward``
  returns ``(y, residual_out)``.
"""

import functools
from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .fused_add_rms_norm_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .fused_add_rms_norm_kernel.perf_opt._fused_add_rms_norm_kernel import _fused_add_rms_norm_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .fused_add_rms_norm_kernel.perf_opt._fused_add_rms_norm_kernel import _fused_add_rms_norm_kernel

__all__ = ["FusedAddRMSNormKernel"]


@functools.lru_cache(maxsize=64)
def _fused_add_rms_norm_factory(M: int, N: int, eps: float, dtype: str):
    """Cached factory builder (mirrors the GPU ``functools.lru_cache`` use)."""
    return _fused_add_rms_norm_kernel(M, N, eps, dtype)


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::fused_add_rms_norm_fwd", mutates_args=())
def _fused_add_rms_norm_wrapped(
    M: int,
    N: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _fused_add_rms_norm_factory(M, N, eps, dtype_str)(block_m)(
        x, residual, weight
    )


@_fused_add_rms_norm_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, x, residual, weight):
    return (
        torch.empty((M, N), dtype=x.dtype, device=x.device),
        torch.empty((M, N), dtype=x.dtype, device=x.device),
    )


# ---------------------------------------------------------------------------
# Kernel class (K5, K8, K9 adaptations)
# ---------------------------------------------------------------------------


class FusedAddRMSNormKernel(Kernel):
    """Fused Add RMSNorm forward kernel.

    Computes ``s = x + residual``, normalizes ``s`` by its RMS, scales by
    ``weight``, and returns **both** ``y`` and ``residual_out = s``.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from the factory/dispatch (rows are processed one
      per core).

    Args:
        M: Number of rows (product of all dims except the last).
        N: Hidden dimension (last dim).
        eps: Epsilon for numerical stability (default 1e-6, per the manifest).
        dtype: Input data type (float16, bfloat16, float32).
        config: Optional kernel configuration dict (kept for interface
            compatibility; the row kernel takes no tunable block size).
        device_index: Unused; accepted for interface compatibility with the
            other norm kernels.
    """

    # K5: a CUDA SM list on the GPU side -> None on NPU.
    supported_archs: Optional[list] = None

    ascend_mode = "Developer"

    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        M: int,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.M = M
        self.N = N
        self.eps = float(eps)
        self.dtype = dtype
        self.kernel = _fused_add_rms_norm_factory(self.M, self.N, self.eps, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # One row per core; `block_m` survives only for factory-signature
        # compatibility with the GPU source.
        return {"block_m": 1}

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the kernel on ``(M, N)`` x/residual and an ``(N,)`` weight."""
        return _fused_add_rms_norm_wrapped(
            self.M,
            self.N,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            x,
            residual,
            weight,
        )
