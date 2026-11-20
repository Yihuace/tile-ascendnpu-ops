"""Batched GEMM kernel for BmmFwdOp (NPU-adapted).

Shapes are strict 3D-3D -- ``a: [B, M, K]``, ``b: [B, K, N]``, ``c: [B, M, N]``.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_bmm_kernel``) is extracted from the
    GPU repo via ``extract_tl_kernel.py`` and imported as-is.  It serves as
    the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 adaptations (decorator, grid/sync semantics,
    ``threads`` removal, padding strategy) are handled by the NPU kernel
    component during re-implementation.

    Cube note: the GPU body stages A/B tiles through ``T.copy`` into
    ``T.alloc_shared`` (SMEM) and accumulates with ``T.gemm`` -- on NPU this
    maps to the Developer-mode cube path (``T.copy``-based ND data movement,
    no explicit L1/L0C NZ buffers; see ``tilelang-cube-skill``).

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[90]``, CUDA SM90 only).
    K7: ``custom_op("npub::bmm_wrapped_kernel")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
        heuristic config selection only (``init_config(config)``).
    K9: ``threads`` removed from the wrapper signature, ``default_config``
        and the ``forward`` dispatch.  ``num_stages`` is retained (pipeline
        depth for ``T.Pipelined``; not a CUDA thread parameter).

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation whose inner callable takes
  ``(block_m, block_n, block_k, num_stages, threads)``.  The dispatch below
  passes the NPU-shaped argument list (no ``threads``); it becomes callable
  once the NPU kernel component rewrites the factory for ``target="npuir"``
  and drops ``threads`` from the callable signature (K3/K9).  Running
  ``forward`` before that rewrite fails -- this is the expected scaffold
  state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/bmm.py.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .bmm_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .bmm_kernel import _bmm_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .bmm_kernel.perf_opt._bmm_kernel import _bmm_kernel

__all__ = ["BmmKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::bmm_wrapped_kernel", mutates_args=())
def _bmm_wrapped_kernel(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Torch custom-op wrapper for ``torch.compile`` compatibility."""
    # The integrated npuir factory is flat and already @tilelang.jit-decorated:
    #   _bmm_kernel(batch, m, n, k, block_M, block_N, block_K, num_stages, dtype)
    # returns the compiled ``(a, b) -> c`` callable (``threads`` dropped, K3/K9).
    # The Kernel-config keys are lower-case; map them onto the factory's
    # keyword args (``block_k`` -> ``block_K``).
    return _bmm_kernel(
        batch, m, n, k,
        block_M=block_m, block_N=block_n, block_K=block_k,
        num_stages=num_stages, dtype=dtype,
    )(a, b)


@_bmm_wrapped_kernel.register_fake
def _(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((batch, m, n), dtype=a.dtype, device=a.device)


# ---------------------------------------------------------------------------
# Kernel class (K5/K8/K9 adaptations)
# ---------------------------------------------------------------------------


class BmmKernel(Kernel):
    """Batched dense GEMM kernel.

    Computes ``C[b] = A[b] @ B[b]`` for ``b in [0, batch)`` where
    ``A: [batch, m, k]``, ``B: [batch, k, n]``, ``C: [batch, m, n]``.
    fp16 / bf16 inputs, fp32 accumulation.  Grid maps the batch axis to the
    third grid dimension so all batches run in a single kernel launch.

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures; GPU pinned CUDA
          SM90).
      K8: autotune removed -- heuristic config selection only;
          ``init_config(config)`` takes no ``tune`` argument.
      K9: ``threads`` removed from ``default_config`` and the ``forward``
          dispatch (GPU default was ``threads=128``).
    """

    # Cube op: Developer-mode GEMM path (T.copy into shared, T.gemm accumulate).
    ascend_mode = "Developer"

    # K5: [90] (CUDA SM90) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__()
        if k % 16 != 0:
            raise ValueError(
                f"BmmKernel requires contraction dim k to be a multiple of 16, got k={k}")
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.device_index = device_index
        # Keep a handle to the integrated npuir factory for introspection
        # (the msprof kernel-name derivation scans this module for the
        # ``@T.prim_func`` name).  Do NOT invoke it here: the npuir factory is
        # itself @tilelang.jit-decorated and flat-signatured, so calling it
        # compiles eagerly -- compilation must happen inside ``Kernel.__call__``
        # so it is scoped to ``ascend_mode``.
        self.kernel = _bmm_kernel
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # 64x64x64, stages=2, 1 warpgroup (GPU heuristic winner; K9: no
        # ``threads`` entry).
        block_k = 64 if self.k % 64 == 0 else (32 if self.k % 32 == 0 else 16)
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": block_k,
            "num_stages": 2,
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run the batched GEMM (K9: no ``threads`` in the dispatch)."""
        return _bmm_wrapped_kernel(
            self.batch,
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            a,
            b,
        )
