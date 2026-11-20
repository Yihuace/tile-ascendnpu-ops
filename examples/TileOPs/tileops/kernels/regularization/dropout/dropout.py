"""Dropout kernel with deterministic random-mask generation (NPU-adapted).

Implements inverted dropout: ``output = x * mask / (1 - p)`` where
``mask ~ Bernoulli(1 - p)``.  The GPU implementation uses TileLang's
``T.rng_init`` / ``T.rng_rand_float`` (backed by cuRAND Philox4_32_10 by
default) for deterministic replay (same seed + same thread layout = same
mask); strategy: explicit_parallel (``N`` elements per thread) with per-thread
RNG state.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang factory (``_make_dropout_kernel``) is extracted from the
    GPU repo via ``extract_tl_kernel.py`` and imported as-is.  It serves as the
    reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 and K11 adaptations (decorator, grid/sync
    semantics, ``threads``/``npt`` collapse, padding strategy) are handled by
    the NPU kernel component during re-implementation:
      - E1 (K3/K11): the factory keeps its op-specific positional arguments
        ``(p, seed)`` but drops ``threads`` / ``num_per_thread`` ->
        ``_make_dropout_kernel(N, dtype, p, seed)``.
      - E2 (K11): the returned callable takes a single ``block_size``
        (the GPU ``threads * num_per_thread`` product) instead of
        ``(threads_arg, npt_arg)``.
      - E3 (K2): ``T.Kernel(..., is_npu=True)`` replaces the CUDA
        ``threads=`` dimension.
      - E4 (K11): ``T.Parallel(block_size)`` replaces
        ``T.Parallel(threads_arg, npt_arg)``.
    OPEN QUESTION for the Stage 1 design review: TileLang RNG
    (``T.rng_init`` / ``T.rng_rand_float``) availability under
    ``target="npuir"``.  If the RNG primitives are unavailable on NPUIR the
    NPU re-implementation must supply an equivalent device-side PRNG (the
    deterministic replay contract in the tests -- same seed -> same mask --
    must be preserved).

  **Part B -- Kernel class** (fully ported):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
         heuristic config selection only (``init_config(config)``).
    K9:  ``threads`` removed from ``default_config`` and the ``forward`` call.
    K11: GPU ``threads * num_per_thread`` product collapsed into a single
         ``block_size`` config key (elementwise ops only).

Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
implementation whose callable takes ``(threads_arg, npt_arg)``.  The dispatch
below passes the NPU-shaped single ``block_size`` argument; it becomes callable
once the NPU kernel component rewrites the factory for ``target="npuir"`` and
collapses the callable signature to ``block_size`` (K3 + K11).  Running
``forward`` before that rewrite fails -- this is the expected scaffold state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/dropout.py.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .dropout_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
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
# .dropout_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .dropout_kernel import _make_dropout_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .dropout_kernel.perf_opt._make_dropout_kernel import _make_dropout_kernel

# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .dropout_kernel.perf_opt._make_dropout_kernel import _make_dropout_kernel

__all__ = ["DropoutKernel"]


# ---------------------------------------------------------------------------
# K11: GPU ``DropoutKernel.default_config`` = {"threads": 256,
# "num_per_thread": npt} with npt=4 for fp32 and npt=8 for fp16/bf16.
# Collapsed to a single per-block element count:
# block_size = threads * num_per_thread = 1024 (fp32) / 2048 (fp16, bf16).
# ---------------------------------------------------------------------------
_DEFAULT_BLOCK_SIZE = {
    "float32": 1024,
    "float16": 2048,
    "bfloat16": 2048,
}


# ---------------------------------------------------------------------------
# Kernel class (K5, K8, K9, K11 adaptations)
# ---------------------------------------------------------------------------


class DropoutKernel(Kernel):
    """Dropout kernel with deterministic mask generation.

    Applies inverted dropout: ``output = x * mask / (1 - p)`` where ``mask`` is
    ``Bernoulli(1 - p)``.  The GPU implementation uses
    ``T.rng_init`` / ``T.rng_rand_float`` (backed by cuRAND Philox4_32_10 by
    default) for deterministic replay; the NPU re-implementation must preserve
    that contract once the kernel factory is rewritten for ``target="npuir"``.

    NPU adaptations:

        K5:  ``supported_archs = None`` (all architectures).
        K8:  autotune removed; heuristic config selection only
             (``init_config(config)`` takes no ``tune`` argument).
        K9:  ``threads`` removed from ``default_config`` and ``forward``.
        K11: GPU ``threads * num_per_thread`` product collapsed into
             ``block_size``.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Data type (float16, bfloat16, or float32).
        p: Drop probability in [0, 1].
        seed: Integer seed for the TileLang RNG.
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 2048}``).
        device_index: Optional device index (accepted for API parity with the
            dynamic-shape kernels; not used by this static-shape kernel).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Scaffold placeholder for the NPU programming mode; the NPU kernel
    # component owns the final value and ``integrate_kernel.py`` re-syncs it
    # from the verified Stage 3 product.  Dropout is a pure vector/elementwise
    # kernel (no Cube ``T.gemm``), so the Developer-mode default is used.
    ascend_mode = "Developer"

    # Float-only dtypes.
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        p: float = 0.5,
        seed: int = 0,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.N_total = N_total
        self.dtype = dtype
        self.p = p
        self.seed = seed
        self.device_index = device_index
        # Build the factory callable (does not compile yet -- compilation is
        # deferred to forward()).  The imported function is the GPU
        # implementation; the NPU component rewrites it for target="npuir"
        # with the K11 signature ``_make_dropout_kernel(N, dtype, p, seed)``
        # and a single ``block_size``-taking callable.
        self.kernel = _make_dropout_kernel(
            self.N_total, self.dtype_str, self.p, self.seed,
        )
        self._compiled_fn = None
        self._compiled_block_size: Optional[int] = None
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * num_per_thread).

        GPU defaults: threads=256, num_per_thread=4 (fp32) or 8 (fp16/bf16).
        Collapsed: block_size = 1024 (fp32) or 2048 (fp16/bf16).
        """
        # K11: collapse threads * num_per_thread into block_size.
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the dropout kernel on a 1-D contiguous tensor of ``N_total``.

        K9: ``threads`` removed from the call; K11/E2: the compiled callable
        takes a single ``block_size``.
        """
        block_size = self.config["block_size"]
        if self._compiled_fn is None or self._compiled_block_size != block_size:
            self._compiled_fn = self.kernel(block_size)
            self._compiled_block_size = block_size
        return self._compiled_fn(x)
