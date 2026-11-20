"""W4A16 GEMM kernel package (NPU-adapted).

Ported from GPU TileOPs ``tileops/kernels/gemm_w4a16.py`` -- the
``GemmW4A16Kernel``: a group128 affine weight dequantization + A16 Tensor Core
GEMM (``output = activation @ dequant(packed_weight).T``) used by
``GemmW4A16Op``.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel function** (extracted + imported):
    The GPU TileLang factory (``_gemm_w4a16_kernel``) is extracted from the
    GPU repo via ``extract_tl_kernel.py`` and imported as-is.  It is the
    reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 (decorator, grid/sync semantics, ``threads``
    removal, padding strategy) are handled by the NPU kernel component during
    re-implementation.

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = None`` (all architectures).
    K8: autotune removed -- heuristic config selection only
        (``init_config(config)``, no ``tune``).
    K9: no ``threads`` in ``default_config`` / ``forward`` -- it is a CUDA
        execution detail (the GPU factory keeps ``threads=128`` as its build
        default).

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation and will not run on NPU until the NPU kernel component
  rewrites it for ``target="npuir"``.  This is the expected scaffold state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/gemm_w4a16.py.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .gemm_w4a16_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .gemm_w4a16_kernel import _gemm_w4a16_kernel
# GROUP_SIZE lives beside the factory; the aggregation __init__ only re-exports
# migrated factory *functions* (meta.extracted_functions), so import the
# constant from the per-function module directly.  The wrapper still re-exports
# it for tileops.kernels.gemm.gemm_w4a16 and tileops.ops.gemm.gemm_w4a16.
# from .gemm_w4a16_kernel._gemm_w4a16_kernel import GROUP_SIZE
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .gemm_w4a16_kernel.perf_opt._gemm_w4a16_kernel import GROUP_SIZE
from .gemm_w4a16_kernel.perf_opt._gemm_w4a16_kernel import _gemm_w4a16_kernel

__all__ = ["GemmW4A16Kernel"]


class GemmW4A16Kernel(Kernel):
    """W4A16 GEMM with group128 affine dequantization and A16 Tensor Core GEMM.

    ``activation`` is ``[M, K]`` fp16; ``packed_weight`` is ``[N, K // 2]``
    uint8 (two unsigned INT4 per byte); ``weight_scale`` / ``weight_zero`` are
    ``[N, K // group_size]`` metadata.  ``output`` is ``[M, N]``.

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures).
      K8: autotune removed -- heuristic config selection only.
      K9: no ``threads`` parameter (CUDA execution detail).
    """

    ascend_mode = "Developer"

    # K5: None -> all architectures supported.
    supported_archs: Optional[list] = None

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
        group_size: int = GROUP_SIZE,
    ) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.group_size = group_size
        self.device_index = device_index

        # Keep a handle to the integrated npuir factory for introspection (the
        # msprof kernel-name derivation scans this module for the
        # ``@T.prim_func`` name).  Do NOT invoke it here: the npuir factory is
        # itself @tilelang.jit-decorated and flat-signatured, so calling it
        # compiles eagerly -- compilation must happen inside ``Kernel.__call__``
        # so it is scoped to ``ascend_mode``.
        self.kernel = _gemm_w4a16_kernel

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # K9: no ``threads`` -- it is a CUDA execution detail.  Tiling follows
        # DESIGN 5.2/9.1 (block_M=16 keeps the m=1 decode shape
        # bandwidth-bound; block_K | group_size | K, no K tail).
        # Stage 4 pairing: with the perf_opt kernel source active, block_K is
        # raised to group_size (=128, matching perf_opt TUNED_DEFAULT_CONFIG);
        # toggle this default-parameter block together with the switch above.
        return {
            "block_m": 16,
            "block_n": 64,
            "block_k": 128,
            "num_stages": 2,
        }

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        # The integrated npuir factory is flat-signatured:
        #   _gemm_w4a16_kernel(M, N, K, group_size=..., dtype=..., block_M=...,
        #                      block_N=..., block_K=..., num_stages=...)
        # Map the lower-case Kernel-config keys onto the factory's kwargs.
        compiled = _gemm_w4a16_kernel(
            self.m,
            self.n,
            self.k,
            group_size=self.group_size,
            dtype=self.dtype_str,
            block_M=self.config["block_m"],
            block_N=self.config["block_n"],
            block_K=self.config["block_k"],
            num_stages=self.config["num_stages"],
        )
        return compiled(activation, packed_weight, weight_scale, weight_zero)
