"""Dense GEMM kernel package (NPU-adapted).

Ported from GPU TileOPs ``tileops/kernels/gemm.py`` -- the ``GemmKernel``
(hand-written warp-specialized dense GEMM, ``C = op(A) @ op(B)``) and the
``GemvKernel`` (m == 1 / n == 1 GEMV fast path) used by ``GemmOp``.  The
FP8 / W4A16 kernel classes that share the GPU file belong to ``GemmFp8Op`` /
``GemmW4A16Op`` and are intentionally not ported.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factories (``_gemm_kernel``, ``_gemv_kernel``) are
    extracted from the GPU repo via ``extract_tl_kernel.py`` and imported
    as-is.  They serve as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync semantics, ``threads`` removal, padding strategy) are handled by
    the NPU kernel component during re-implementation.

  **Part B -- custom_op wrappers + Kernel classes** (fully ported):
    K5: ``supported_archs = None`` (was ``[90]``).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``tune`` param removed; heuristic config
        selection only (``init_config(config)``).
    K9: no ``threads`` in the wrappers / ``default_config`` / ``forward`` --
        the GPU ``_gemm_kernel`` factory fixes ``threads = 256`` internally,
        and the ``_gemv_kernel`` callable's ``reduce_threads`` is a real
        tiling parameter (kept).

  GPU-specific bits not carried over:

    - ``get_sm_version()`` (CUDA SM integer) is gone; ``GemvKernel``'s
      ``default_config`` uses the general-purpose (non-SM90) tiling directly.
    - ``trace.enabled`` / ``trace.run`` in ``GemmKernel.forward`` (GPU-only
      timeline tracer) are dropped; the compiled kernel is called directly.

  Pre-Stage-3 note: the imported factories are still the GPU (CUDA target)
  implementations and will not run on NPU until the NPU kernel component
  rewrites them for ``target="npuir"``.  This is the expected scaffold state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factories extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/gemm.py.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .gemm_kernel/perf_opt/{OpName}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .gemm_kernel import _gemm_kernel, _gemv_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .gemm_kernel.perf_opt.GemmOp import _gemm_kernel

# ``_gemv_kernel`` is NOT part of the Stage-4 tuning scope (the benchmark
# workloads are all dense GEMM), so it keeps the Stage-3 baseline source.
from .gemm_kernel import _gemv_kernel

__all__ = ["GemmKernel", "GemvKernel"]


# ---------------------------------------------------------------------------
# custom_op wrappers (K7: top:: -> npub::; kept for torch.compile compatibility)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::gemm_wrapped_kernel", mutates_args=())
def _gemm_wrapped_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Run the dense GEMM ``C = op(A) @ op(B)`` (torch custom op).

    Kept for ``torch.compile`` compatibility (registered op + ``register_fake``).
    ``GemmKernel.forward`` calls the compiled JIT directly, so this wrapper is
    not on the eager forward path.

    The integrated npuir factory is flat and already ``@tilelang.jit``-decorated::

        _gemm_kernel(m, n, k, trans_a, trans_b, dtype,
                     block_M, block_N, block_K, num_stages) -> compiled (a, b) -> c

    (the GPU-nested ``factory(...)(tiling)(a, b)`` shape is gone; K9 ``threads``
    was dropped).  Map the lower-case Kernel-config keys onto the factory's
    keyword args (``block_k`` -> ``block_K``).
    """
    return _gemm_kernel(
        m, n, k, trans_a, trans_b,
        dtype=dtype, block_M=block_m, block_N=block_n,
        block_K=block_k, num_stages=num_stages,
    )(a, b)


@_gemm_wrapped_kernel.register_fake
def _(m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str,
      block_m: int, block_n: int, block_k: int, num_stages: int,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((m, n), dtype=inputs[0].dtype, device=inputs[0].device)


@torch.library.custom_op("npub::gemv_wrapped_kernel", mutates_args=())
def _gemv_wrapped_kernel(
    n: int,
    k: int,
    dtype: str,
    block_n: int,
    reduce_threads: int,
    num_stages: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    # torch.compile-only path (not used by the eager forward).  The integrated
    # npuir ``_gemv_kernel(R, K, dtype, block_M, block_K, num_stages)`` takes
    # ``(W[R, K], x[K])``; the GPU callable's ``(a, b)`` are ``(x, W)``, so the
    # operands are reversed.  The legacy tiling names are mapped onto the npuir
    # factory kwargs (block_n -> block_M output-row tile; reduce_threads is the
    # legacy K-reduce knob -> block_K K tile).
    return _gemv_kernel(
        n, k, dtype=dtype,
        block_M=block_n, block_K=reduce_threads, num_stages=num_stages,
    )(b, a)


@_gemv_wrapped_kernel.register_fake
def _(n: int, k: int,
      dtype: str, block_n: int, reduce_threads: int, num_stages: int,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((n,), dtype=inputs[0].dtype, device=inputs[0].device)


# ---------------------------------------------------------------------------
# Kernel classes (K5/K8/K9 adaptations)
# ---------------------------------------------------------------------------


class _GemmFamilyKernelBase(Kernel):
    """Shared base for the two GEMM-family kernels (``GemmKernel``/``GemvKernel``).

    The GEMM migration unit owns two kernel functions (``_gemm_kernel`` for the
    dense ``C = op(A) @ op(B)`` path and ``_gemv_kernel`` for the ``m == 1`` /
    ``n == 1`` fast path, both dispatched by ``GemmOp``).  They share the same
    npuir programming mode and architecture support, so the common ``Kernel``
    contract is declared once here: the integration harness requires the
    wrapper module to expose a single direct ``Kernel`` subclass carrying the
    integrated kernels' ``ascend_mode`` (see Stage 5 gate ``S5-ASCEND-MODE``).
    Both concrete kernels are ``Kernel`` subclasses through this base, so
    ``Kernel.__call__`` (mode-scoped compilation) still applies to each.

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures; GPU pinned SM90).
    """

    # npuir programming mode shared by both integrated kernels (Developer).
    ascend_mode = "Developer"

    # K5: [90] (CUDA SM90) -> None (all architectures).
    supported_archs: Optional[list] = None


class GemmKernel(_GemmFamilyKernelBase):
    """Dense GEMM kernel: ``C = op(A) @ op(B)`` for any ``(trans_a, trans_b)``.

    ``a`` is ``[M, K]`` (or ``[K, M]`` when ``trans_a``); ``b`` is
    ``[K, N]`` (or ``[N, K]`` when ``trans_b``); ``C`` is ``[M, N]``.
    fp16 / bf16 inputs, fp32 accumulation.

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures; GPU pinned SM90).
      K8: autotune removed -- heuristic config selection only.
      K9: no ``threads`` parameter (the GPU factory fixed it at 256).
    """

    # Development-mode cube/vector GEMM path.
    ascend_mode = "Developer"

    # K5: [90] (CUDA SM90) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(self,
                 m: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 device_index: int | None = None,
                 trans_a: bool = False,
                 trans_b: bool = False) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.device_index = device_index

        # Keep a handle to the integrated npuir factory for introspection (the
        # msprof kernel-name derivation scans this module for the
        # ``@T.prim_func`` name).  Do NOT invoke it here: the npuir factory is
        # itself @tilelang.jit-decorated and flat-signatured, so calling it
        # compiles eagerly -- compilation must happen inside ``Kernel.__call__``
        # so it is scoped to ``ascend_mode``.
        self.kernel = _gemm_kernel

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # Stage-4 tuned (round 2): block_K was the dominant knob (64 -> 256
        # roughly halved large-shape latency); block_N=256 only takes effect
        # together with the perf_opt `_tile_size` flexible-cap change.
        return {
            "block_m": 128,
            "block_n": 256,
            "block_k": 256,
            "num_stages": 2,
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Call the flat npuir factory directly (cf. GemvKernel);
        # _gemm_wrapped_kernel is kept only for torch.compile compatibility.
        # The GPU ``trace.run`` timeline wrapper is dropped (GPU-only tracer).
        # ``Kernel.__call__`` scopes this to ``ascend_mode`` so the (lazy) JIT
        # compile runs under Developer mode.
        return _gemm_kernel(
            self.m, self.n, self.k, self.trans_a, self.trans_b,
            dtype=self.dtype_str,
            block_M=self.config["block_m"],
            block_N=self.config["block_n"],
            block_K=self.config["block_k"],
            num_stages=self.config["num_stages"],
        )(a, b)


class GemvKernel(_GemmFamilyKernelBase):
    """GEMV fast path: ``C = a @ op(B)`` for ``m == 1`` / ``n == 1``.

    ``a`` is a 1-D activation of length ``K``; ``b`` is ``[n, k]``; ``C`` is
    ``[n]``.  Used by ``GemmOp`` for the decode-shaped (m == 1) and
    thin-output (n == 1) cases.

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures; GPU pinned SM90).
      K8: ``autotune_configs`` removed -- heuristic config selection only.
    """

    def __init__(self,
                 n: int,
                 k: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 device_index: int | None = None) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self.dtype = dtype
        self.device_index = device_index

        # Handle only (see GemmKernel.__init__): do not invoke the flat npuir
        # factory at construction time -- compilation is deferred into
        # ``Kernel.__call__`` so it is scoped to ``ascend_mode``.
        self.kernel = _gemv_kernel

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # npuir ``_gemv_kernel(R, K, dtype, block_M, block_K, num_stages)``
        # tiling defaults from the Stage 3 L0/L1 run_case (the GPU SM-version
        # branch / ``threads``-derived block_n,reduce_threads knob is gone).
        return {
            "block_M": 64,
            "block_K": 256,
            "num_stages": 2,
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # ``a`` is the 1-D activation (length K) and ``b`` the [R, K] matrix;
        # the npuir kernel is ``y = W @ x`` with signature (W[R, K], x[K]), so
        # the operands are reversed relative to the GPU callable's (x, W).
        a = a.flatten().contiguous()
        return _gemv_kernel(
            self.n, self.k,
            dtype=self.dtype_str,
            block_M=self.config["block_M"],
            block_K=self.config["block_K"],
            num_stages=self.config["num_stages"],
        )(b, a)
