"""ALiBi position-encoding bias kernel using TileLang (NPU-adapted).

Implements the generative ALiBi (Attention with Linear Biases) position
encoding:
    bias[h, i, j] = -slope_h * |i - j|,  slope_h = 2^(-8 * (h + 1) / num_heads)
The full ``(num_heads, seq_len, seq_len)`` tensor is synthesized from the
construction-time integer parameters -- there is **no tensor input**.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md:

  **Part A -- TileLang kernel functions** (extracted + imported):
    ``_make_alibi_kernel`` is extracted from the GPU repo via
    ``extract_tl_kernel.py`` and imported as-is.  It serves as the reference
    for the NPU kernel component to reimplement for ``target="npuir"``.
    K1-K4 and K11 adaptations (decorator, grid/sync, ``threads``/``npt``
    collapse, padding strategy) are handled by the NPU component during
    re-implementation:
      - E1 (K3/K11): the factory keeps its op-specific positional arguments
        ``(seq_len, num_heads, dtype)`` and drops ``threads`` / ``npt``.
      - E2 (K11): the returned callable takes a single ``block_size`` (the
        GPU ``threads * npt`` product) instead of ``(threads_arg, npt_arg)``.
      - E3 (K2): ``T.Kernel(..., is_npu=True)`` replaces the CUDA
        ``threads=`` dimension.
      - E4 (K11): ``T.Parallel(block_size)`` replaces
        ``T.Parallel(threads_arg, npt_arg)``.

  **Part B -- Kernel class** (fully ported):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  autotune removed; heuristic config selection only.
         ``init_config(config)`` takes no ``tune`` argument.
    K9:  ``threads`` removed from ``default_config`` and ``forward``.
    K11: GPU ``threads * npt`` product collapsed into a single ``block_size``.

The op is **eager-only**: unlike the other elementwise kernels, there is no
``@torch.library.custom_op`` wrapper (no tensor input means no compile-time
shape guard is needed), so ``forward`` dispatches straight to the compiled
factory callable, exactly as in the GPU source (adaptation point E10).
"""

from typing import Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .alibi_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .alibi_kernel import _make_alibi_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .alibi_kernel.perf_opt._make_alibi_kernel import _make_alibi_kernel

__all__ = ["AlibiFwdKernel"]

# ---------------------------------------------------------------------------
# Calling convention (Stage 5, DESIGN §0.6 R5): the Stage 3 NPU factory uses
# an *explicit output parameter* (no ``out_idx``) -- ``out_idx + T.npuir_arange``
# crashes at runtime on device (``MTE DDR out of range``, retCode 507057,
# CG-2026-0016).  The GPU ``block_size`` (= threads * npt) collapse therefore
# does not apply; the factory auto-selects the ``(bi, bj)`` spatial tile from
# the UB budget (DESIGN §5.2).  ``bi`` / ``bj`` stay overridable (None = auto)
# for Stage 4 tuning, and the wrapper allocates ``out`` on the active device.
# ---------------------------------------------------------------------------


class AlibiFwdKernel(Kernel):
    """ALiBi forward kernel: bias[h, i, j] = -slope_h * |i - j|.

    A generative op: it takes no tensor input and produces the full
    ``(num_heads, seq_len, seq_len)`` bias tensor from ``seq_len`` /
    ``num_heads`` baked in at construction time.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11/E10: the GPU ``threads * npt`` collapse is dropped; the NPU
      factory auto-selects the ``(bi, bj)`` tile.  There is no
      ``custom_op`` wrapper — ``forward`` allocates ``out`` and calls the
      compiled factory callable directly.

    Args:
        seq_len: Sequence length (positive integer).
        num_heads: Number of attention heads (positive integer).
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict (e.g. ``{"bi": 64,
            "bj": 128}``; ``None`` values keep the factory auto-selection).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Float-only dtypes (GPU ``_FLOAT_DTYPES``); fp8 is not a manifest dtype
    # for AlibiFwdOp.
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        seq_len: int,
        num_heads: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dtype = dtype
        self.output_dtype = dtype  # float-only, output == input dtype
        self.N_total = num_heads * seq_len * seq_len
        # The Stage 3 NPU factory is called with the compile-time integers +
        # dtype and returns the JIT-compiled callable (DESIGN §0.6 R5).  It
        # uses an explicit output parameter, so the callable takes the
        # pre-allocated ``out`` tensor (see ``forward``).
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config.

        The GPU ``block_size`` (threads * npt) collapse does not apply to
        the NPU kernel: the Stage 3 factory auto-selects the ``(bi, bj)``
        spatial tile from the UB budget (DESIGN §5.2).  ``bi`` / ``bj`` stay
        overridable for Stage 4 tuning (``None`` = factory auto-selection).
        """
        return {"bi": None, "bj": None}

    def init_config(self, config: Optional[dict] = None) -> None:
        """Build the compile-time kernel callable after the config is set.

        The Stage 3 NPU factory signature is
        ``_make_alibi_kernel(seq_len, num_heads, dtype, bi=None, bj=None,
        budget=..., core_num=...)``; ``None`` tiles defer to the factory's
        divisor-based default (DESIGN §5.2).  The factory returns the
        compiled callable directly (explicit ``out`` parameter, no
        ``out_idx``).
        """
        super().init_config(config)
        tile = {k: self.config[k] for k in ("bi", "bj") if self.config.get(k) is not None}
        self.kernel = _make_alibi_kernel(
            self.seq_len, self.num_heads, self.dtype_str, **tile
        )
        self._compiled_fn = self.kernel

    def forward(self):
        """Run the ALiBi kernel (no arguments; the output is synthesized).

        Allocate the ``(num_heads, seq_len, seq_len)`` output on the active
        device and pass it to the explicit-output kernel (DESIGN §0.6 R5).
        """
        out = torch.empty(
            self.num_heads,
            self.seq_len,
            self.seq_len,
            dtype=self.dtype,
            device=get_device_backend().name,
        )
        self._compiled_fn(out)
        return out
