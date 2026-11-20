"""Llama 3.1 Rotary Position Embedding kernel (NPU-adapted).

Adapted from TileOPs ``{gpu_repo_root}/tileops/kernels/rope.py`` --
``RopeLlama31Kernel`` (``ROTATION_STYLE = "neox"``).  The GPU file hosts five
RoPE variants (neox / non_neox / llama31 / yarn / longrope) x two layouts, all
sharing a ``_RopeKernelBase``; this NPU port carries only the Llama 3.1 variant
(``RopeLlama31Op``), so the base logic is inlined into a single kernel class.

.. note::
   ``RopeLlama31Kernel`` differs from ``RopeNeoxKernel`` **only** in the
   frequency-table computation, which lives in the Op layer
   (``_llama31_freqs``, Llama 3.1 piecewise frequency scaling) -- the TileLang
   kernel factories are the *same* (``_make_rope_neox_1d`` /
   ``_make_rope_neox_2d``; there is no ``_make_rope_llama31_*`` in the GPU
   repo).  This class is therefore a structural sibling of ``RopeNeoxKernel``:
   the kernel performs the neox rotation only::

       y = x * cos + rotate_half(x) * sin
       rotate_half(x) = concat(-x[half:], x[:half])     # split at the midpoint

   Two layouts are supported:

   - ``"1d"``: ``x`` is ``(seq_len, head_dim)``; the kernel indexes
     ``x[row, col]`` directly.
   - ``"2d"``: ``x`` is ``(batch, seq_len, num_heads, head_dim)``; the Op
     flattens it to ``(N_total,)`` and the kernel indexes by flat offset.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang factories are extracted from the GPU repo via
    ``extract_tl_kernel.py`` and imported as-is.  They are the reference for the
    NPU kernel component to reimplement for ``target="npuir"``.  K1-K4 and K11
    adaptations (decorator, grid/sync semantics, ``threads``/``npt`` collapse,
    padding strategy) are handled by the NPU kernel component:
      - K3/K11: the factories keep their op-specific positional arguments
        (``seq_len``/``head_dim`` or ``batch``/``seq_len``/``num_heads``/
        ``head_dim`` plus ``dtype``) and drop ``threads`` / ``num_per_thread``.
      - K11: the returned callable takes a single ``block_size`` (the GPU
        ``threads * num_per_thread`` product) instead of ``(threads_arg,
        npt_arg)``.
      - K2: ``T.Kernel(..., is_npu=True)`` replaces the CUDA ``threads=``
        dimension.
      - K11: ``T.Parallel(block_size)`` replaces
        ``T.Parallel(threads_arg, npt_arg)``.

  **Part B -- Kernel class** (fully ported):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  the shared ``_RopeKernelBase`` is inlined; autotune removed -- no
         ``tune`` parameter, ``init_config(config)`` only.
    K9:  ``threads`` removed from ``default_config`` and the ``forward`` call.
    K11: the GPU ``threads * num_per_thread`` product is collapsed into a single
         ``block_size`` config key (elementwise op).
    The GPU op has no ``@torch.library.custom_op`` wrapper (it calls the
    compiled factory directly), so no custom_op wrapper is ported (parity with
    ``RopeNeoxKernel`` / ``RopeLongRopeKernel`` / ``LerpTensorFwdKernel``).

Pre-Stage-3 note: the imported factories are still the GPU (CUDA target)
implementations whose inner callable takes ``(threads_arg, npt_arg)``.  The
dispatch below passes the NPU-shaped single ``block_size`` argument; it becomes
callable once the NPU kernel component rewrites the factories for
``target="npuir"`` and collapses the callable signature to ``block_size``
(K3 + K11).  Running ``forward`` before that rewrite fails -- this is the
expected scaffold state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factories extracted by extract_tl_kernel.py
# (pattern A) from {gpu_repo_root}/tileops/kernels/rope.py.  RopeLlama31Kernel
# shares the neox rotation, so these are the same factories RopeNeoxKernel uses.
# The later conductor integration step rewrites this import into the
# baseline/perf_opt kernel-source selection block once the per-function NPU
# kernels for this op are integrated.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .rope_llama31_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .rope_llama31_kernel import _make_rope_neox_1d, _make_rope_neox_2d
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .rope_llama31_kernel.perf_opt._make_rope_neox_1d import _make_rope_neox_1d
from .rope_llama31_kernel.perf_opt._make_rope_neox_2d import _make_rope_neox_2d

__all__ = ["RopeLlama31Kernel"]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# ---------------------------------------------------------------------------
# K11: GPU ``_RopeKernelBase.default_config`` = {"threads": 256,
# "num_per_thread": npt} with npt = 4 for fp32 and 8 for fp16/bf16.
# Collapsed to a single per-block element count:
# block_size = threads * num_per_thread = 1024 (fp32) / 2048 (fp16, bf16).
# ---------------------------------------------------------------------------
_DEFAULT_BLOCK_SIZE = {
    "float32": 1024,
    "float16": 2048,
    "bfloat16": 2048,
}


class RopeLlama31Kernel(Kernel):
    """Llama 3.1 RoPE kernel.

    Rotation: split the head dimension at the midpoint and use
    ``rotate_half(x) = concat(-x[half:], x[:half])`` (same neox rotation as
    :class:`RopeNeoxKernel`; Llama 3.1 differs only in the Op-layer piecewise
    frequency computation).  Reference: Meta Llama 3.1 model implementation.

    Args:
        seq_len: Sequence length.
        head_dim: Head dimension (must be even).
        dtype: Torch dtype (float16, bfloat16, or float32).
        layout: ``"1d"`` for ``(seq_len, head_dim)`` or ``"2d"`` for
            ``(batch, seq_len, num_heads, head_dim)``.
        batch: Batch size (required for the 2d layout; default 1).
        num_heads: Number of heads (required for the 2d layout; default 1).
        config: Optional kernel configuration dict (e.g. ``{"block_size":
            2048}``).
        device_index: Optional device index (accepted for API parity with the
            dynamic-shape kernels; not used by this static-shape kernel).

    NPU adaptations:
        K5: ``supported_archs = None`` (all architectures).
        K8: autotune removed -- heuristic config selection only
            (``init_config(config)`` takes no ``tune`` argument).
        K9: ``threads`` removed from ``default_config`` and ``forward``.
        K11: GPU ``threads * num_per_thread`` collapsed into ``block_size``.
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Scaffold placeholder for the NPU programming mode; the NPU kernel
    # component owns the final value and ``integrate_kernel.py`` re-syncs it
    # from the verified Stage 3 product.  RoPE is a pure vector/elementwise
    # kernel (no Cube ``T.gemm``), so the Developer-mode default is used.
    ascend_mode = "Developer"

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        layout: str = "1d",
        batch: int = 1,
        num_heads: int = 1,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        if layout not in ("1d", "2d"):
            raise ValueError(f"layout must be '1d' or '2d', got '{layout}'")

        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.layout = layout
        self.batch = batch
        self.num_heads = num_heads
        self.device_index = device_index

        # Build the factory callable (does not compile yet -- compilation is
        # deferred to forward()).  K3/K11: the factories drop ``threads`` /
        # ``num_per_thread``; the NPU kernel component rewrites them for
        # ``target="npuir"`` with a single ``block_size``-taking callable.
        self.kernel = self._build_kernel()
        self._compiled_fn = None
        self._compiled_block_size: Optional[int] = None
        self.init_config(config)

    def _build_kernel(self) -> object:
        dtype_str = self.dtype_to_str(self.dtype)
        if self.layout == "1d":
            return _make_rope_neox_1d(self.seq_len, self.head_dim, dtype_str)
        return _make_rope_neox_2d(
            self.batch, self.seq_len, self.num_heads, self.head_dim, dtype_str
        )

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * num_per_thread).

        GPU defaults: threads=256, num_per_thread=4 (fp32) or 8 (fp16/bf16).
        Collapsed: block_size = 1024 (fp32) or 2048 (fp16/bf16).
        """
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the neox rotation.

        Args:
            x: Input tensor. 1d: ``(seq_len, head_dim)``; 2d:
                ``(batch, seq_len, num_heads, head_dim)``.
            cos: Cosine table of shape ``(seq_len, head_dim // 2)``.
            sin: Sine table of shape ``(seq_len, head_dim // 2)``.

        Returns:
            Rotated tensor of the same shape as ``x``.
        """
        # K9: no ``threads`` in the dispatch; K11/E2: the compiled callable
        # takes a single ``block_size``.
        block_size = self.config["block_size"]
        if self._compiled_fn is None or self._compiled_block_size != block_size:
            self._compiled_fn = self.kernel(block_size)
            self._compiled_block_size = block_size

        if self.layout == "2d":
            x_flat = x.contiguous().reshape(-1)
            result = self._compiled_fn(x_flat, cos, sin)
            return result.reshape(x.shape)
        return self._compiled_fn(x, cos, sin)
