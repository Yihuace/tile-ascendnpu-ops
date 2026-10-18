"""BatchNorm backward kernels (NPU-adapted wrapper + Kernel classes).

Computes the gradients of the training batch-norm forward from the per-channel
``mean`` / ``rstd`` the forward pass saved. For a channel ``c`` with ``L = N*S``
elements and ``x_hat = (x - mean) * rstd``::

    grad_bias[c]   = sum_i( grad_out[c, i] )
    grad_weight[c] = sum_i( grad_out[c, i] * x_hat[c, i] )
    grad_x[c, i]   = weight[c] * rstd[c] / L
                     * ( L * grad_out[c, i]
                         - grad_bias[c]
                         - x_hat[c, i] * grad_weight[c] )

The GPU ``BatchNormBwdOp`` selects among three kernel classes:
``BatchNormBwdWideKernel`` (one channel per block, held in registers),
``BatchNormBwdSplitKernel`` (a channel across several blocks: sum, merge, then
map) and ``BatchNormBwdKernel`` (the general tiled path, streamed through
shared memory). This file ports the three Kernel classes and imports their
``@tilelang.jit`` builders from the extracted reference file.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factories (``_batch_norm_bwd_wide_kernel``,
    ``_batch_norm_bwd_split_kernel``, ``_batch_norm_bwd_kernel``) are extracted
    from the GPU repo via ``extract_tl_kernel.py`` into the self-contained
    reference ``_batch_norm_bwd_kernels.py`` and imported as-is. They are the
    GPU (CUDA-target) implementations the NPU kernel component rewrites for
    ``target="npuir"``; K1-K4 adaptations (decorator, grid/sync semantics,
    ``threads`` removal, padding policy) are handled by that component during
    re-implementation.

  **Part B -- Kernel classes** (ported from the GPU classes):
    K5:  ``supported_archs = None`` (was ``[80, 89, 90]``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` -- removed; heuristic
         config selection only (``init_config(config)``).
    K9/K11: ``default_config`` returns ``{}`` and ``forward`` invokes the
         factory with no baked CUDA execution parameter -- the GPU ``threads``
         / ``num_per_thread`` are CUDA-execution-model artifacts, so the NPU
         factory's own default governs (this file must not bake them).
    The GPU ``@torch.library.custom_op`` wrappers, ``_require_cuda`` /
    ``vector_aligned`` GPU helpers and the ``BatchNormCall`` interface
    machinery are dropped: the NPU port follows the direct-call
    elementwise/row-reduction precedent (no ``torch.compile`` dispatch boundary
    on NPU). The ``BatchNormCall`` based holder policy (which path holds a
    channel) is preserved as plain ``_holder`` / ``_block_launch`` /
    ``_split_seed`` helpers driving ``select_bwd_kernel``.
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
# .batch_norm_bwd_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .batch_norm_bwd_kernel import (
#     _batch_norm_bwd_kernel,
#     _batch_norm_bwd_split_kernel,
#     _batch_norm_bwd_wide_kernel,
# )
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .batch_norm_bwd_kernel.perf_opt.BatchNormBwdOp import (
    _batch_norm_bwd_kernel,
    _batch_norm_bwd_split_kernel,
    _batch_norm_bwd_wide_kernel,
)

__all__ = [
    "BatchNormBwdKernel",
    "BatchNormBwdSplitKernel",
    "BatchNormBwdWideKernel",
    "select_bwd_kernel",
]


# ---------------------------------------------------------------------------
# Channel-holder policy (ported from the GPU ``_BatchNormKernel``).
#
# Decides what holds one channel of ``held_tensors`` tensors: the smallest of
# "thread", "block", "split" (across blocks) and "tiled" that fits. The GPU
# heuristic keys off the CUDA ``sm_count``; the NPU port substitutes the
# physical AI-core count (the probe's persistent core-split target).
# ---------------------------------------------------------------------------

# Vector access width in bytes (the GPU helper ``VECTOR_ACCESS_BYTES``). The
# migrated npuir kernels own their vector width internally; the host-side
# holder heuristic keeps the 16-byte step it was ported with.
VECTOR_ACCESS_BYTES = 16

# AI-core count used to decide whether one block per channel already fills the
# device (GPU ``call.sm_count``).
_AI_CORE_COUNT = 40

# The longest channel of one element per batch item that one thread holds.
_THREAD_MAX_L = 32

# Elements one thread of a register-holding block keeps, over every tensor it holds.
_BLOCK_MAX_HELD = 256
_HOLDING_BLOCK_THREADS = 256
_BLOCK_MAX_THREADS = 1024
# The widest block once the grid alone covers every SM.
_BLOCK_MAX_THREADS_FULL_GRID = 512

# At or above this many channels one block per channel already fills the device.
_SPLIT_MAX_C = 1024
_SPLIT_MIN_L = 1 << 16
# Blocks a split grid aims for before tuning.
_SPLIT_TARGET_BLOCKS = 512


def _block_launch(
    n: int, c: int, spatial: int, dtype: torch.dtype
) -> Optional[Tuple[int, int]]:
    """The ``(threads, num_per_thread)`` of a block holding one channel.

    ``None`` where the channel does not fit. A thread holds *held_tensors*
    elements per channel element; its vector never straddles two batch items.
    """
    L = n * spatial
    full_grid = c >= _AI_CORE_COUNT
    widest = _BLOCK_MAX_THREADS_FULL_GRID if full_grid else _BLOCK_MAX_THREADS
    vector = VECTOR_ACCESS_BYTES // dtype.itemsize
    for num_per_thread in (vector >> k for k in range(vector.bit_length())):
        if spatial % num_per_thread:
            continue
        threads = _HOLDING_BLOCK_THREADS
        while threads > 32 and threads * num_per_thread >= L * 2:
            threads //= 2
        steps = -(-L // (threads * num_per_thread))
        while (
            num_per_thread > 1
            and threads < widest
            and steps > 1
            and steps * threads * num_per_thread != L
        ):
            threads *= 2
            steps = -(-L // (threads * num_per_thread))
        if steps * num_per_thread * 2 <= _BLOCK_MAX_HELD:
            return threads, num_per_thread
    return None


def _holder(n: int, c: int, spatial: int, dtype: torch.dtype, held_tensors: int) -> str:
    """What holds one channel of *held_tensors* tensors."""
    if spatial <= 1 and n * spatial <= _THREAD_MAX_L:
        return "thread"
    if _block_launch(n, c, spatial, dtype) is not None:
        return "block"
    if c < _SPLIT_MAX_C and n * spatial >= _SPLIT_MIN_L:
        return "split"
    return "tiled"


def _split_seed(n: int, c: int, spatial: int) -> int:
    """Pieces a split channel is cut into before anything is measured."""
    return max(1, min(n * spatial, -(-_SPLIT_TARGET_BLOCKS // c)))


def select_bwd_kernel(n: int, c: int, spatial: int, dtype: torch.dtype) -> str:
    """Pick the backward kernel class key that serves this call (GPU refusal priority).

    ``thread`` / ``block`` both land on the register-held ``bwd_wide`` kernel
    (the GPU ``BatchNormBwdWideKernel`` accepts either holder); ``split`` lands
    on ``bwd_split``; everything else on the general ``bwd_kernel``.
    """
    holder = _holder(n, c, spatial, dtype, held_tensors=2)
    if holder in ("thread", "block"):
        return "bwd_wide"
    if holder == "split":
        return "bwd_split"
    return "bwd_kernel"


class _BatchNormBwdBase(Kernel):
    """What the backward candidates share: the channel-holder policy and constants."""

    # K5: [80, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Expert-form vector kernels (T.alloc_ub + single T.Scope("Vector"));
    # invocations are scoped to Expert mode regardless of the ambient
    # TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Expert"

    # Manifest: float32 | float16 | bfloat16.
    SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class BatchNormBwdWideKernel(_BatchNormBwdBase):
    """Backward with one channel per block, ``grad_out`` and ``x`` held in its registers.

    Serves a channel whose two tensors one block holds, a thread-held one included.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: grad_out/x/grad_x data type.
        launch: Optional ``(threads, num_per_thread)`` of the block. The GPU
            CUDA-execution knob is NOT baked (K9/K11); the NPU factory's own
            default governs. Kept as a documented op-shape fact only.
        config: Optional kernel configuration override.
    """

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        launch: Optional[Tuple[int, int]] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.N = N
        self.C = C
        self.S = S
        self.L = N * S
        self.dtype = dtype
        self.launch = launch if launch is not None else _block_launch(N, C, S, dtype)
        # Build the factory callable (compilation is deferred to forward()).
        # The imported function is the GPU reference implementation; the NPU
        # component rewrites it for target="npuir" with the same leading
        # interface ``(N, C, S, dtype)``.
        self.kernel = _batch_norm_bwd_wide_kernel(N, C, S, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # K11 wrapper rule: no CUDA execution parameter is baked here -- the NPU
        # factory's own default governs (do not bake the GPU threads/npt).
        return {}

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the backward pass on ``(N, C, S)`` inputs.

        Returns:
            ``(grad_x, grad_weight, grad_bias)``, ``grad_x`` in ``(N, C, S)``.
        """
        grad_out = grad_out.contiguous()
        x = x.contiguous()
        # NPU wide kernel uses out_idx=[7, 5, 6]: the JIT allocates and returns
        # (grad_x, grad_weight, grad_bias); only the five inputs are passed.
        program = self.kernel()
        grad_x, grad_weight, grad_bias = program(grad_out, x, weight, mean, rstd)
        return grad_x, grad_weight, grad_bias


class BatchNormBwdSplitKernel(_BatchNormBwdBase):
    """Backward with a channel across several blocks: sum, merge, then map.

    Serves a channel whose two tensors one block does not hold and that is long
    enough, among few enough channels, to cut across blocks. The split count and
    block width follow from the shape.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: grad_out/x/grad_x data type.
        splits: Optional untuned number of pieces a channel is cut into.
        config: Optional kernel configuration override.
    """

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        splits: Optional[int] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.N = N
        self.C = C
        self.S = S
        self.L = N * S
        self.dtype = dtype
        self.splits = splits if splits is not None else _split_seed(N, C, S)
        # The factory returns the (stats, finalize, apply) triple of builders.
        self.stages = _batch_norm_bwd_split_kernel(N, C, S, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # K11 wrapper rule: no CUDA execution parameter (threads / num_per_thread)
        # is baked here -- the NPU factories' own defaults govern.
        return {}

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the backward pass on ``(N, C, S)`` inputs.

        Returns:
            ``(grad_x, grad_weight, grad_bias)``, ``grad_x`` in ``(N, C, S)``.
        """
        grad_out = grad_out.contiguous()
        x = x.contiguous()
        grad_weight = torch.empty(self.C, device=x.device, dtype=torch.float32)
        grad_bias = torch.empty_like(grad_weight)
        stats, finalize, apply_ = self.stages
        splits = self.config.get("splits", self.splits)
        empty = functools.partial(torch.empty, device=x.device, dtype=torch.float32)
        partial_sum = empty((self.C, splits))
        partial_sum_xhat = empty((self.C, splits))
        stats()(grad_out, x, mean, rstd, partial_sum, partial_sum_xhat)
        scale, shift, centered_coef = empty(self.C), empty(self.C), empty(self.C)
        finalize()(
            partial_sum,
            partial_sum_xhat,
            weight,
            rstd,
            grad_weight,
            grad_bias,
            scale,
            shift,
            centered_coef,
        )
        grad_x = apply_()(grad_out, x, mean, scale, shift, centered_coef)
        return grad_x, grad_weight, grad_bias


class BatchNormBwdKernel(_BatchNormBwdBase):
    """Backward with one channel per block, streamed through shared memory.

    The general implementation: it serves every shape the specialised ones do not.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: grad_out/x/grad_x data type.
        config: Optional kernel configuration override.
    """

    ascend_mode = "Expert"

    general: bool = True

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.N = N
        self.C = C
        self.S = S
        self.L = N * S
        self.dtype = dtype
        # Build the factory callable (compilation is deferred to forward()).
        self.kernel = _batch_norm_bwd_kernel(N, C, S, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # K9/K11: the GPU {"block_l", "threads"} pair is a CUDA-execution-model
        # artifact; nothing is baked here -- the NPU factory's own default governs.
        return {}

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the backward pass on ``(N, C, S)`` inputs.

        Returns:
            ``(grad_x, grad_weight, grad_bias)``, ``grad_x`` in ``(N, C, S)``.
        """
        grad_out = grad_out.contiguous()
        x = x.contiguous()
        grad_weight = torch.empty(self.C, device=x.device, dtype=torch.float32)
        grad_bias = torch.empty_like(grad_weight)
        program = self.kernel()
        grad_x = program(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)
        return grad_x, grad_weight, grad_bias
