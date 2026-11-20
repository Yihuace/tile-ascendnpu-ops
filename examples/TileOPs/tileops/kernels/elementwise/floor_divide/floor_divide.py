"""FloorDivideFwdKernel -- element-wise floor division with broadcast (NPU-adapted).

Implements element-wise floor division with PyTorch broadcast semantics::

    y = floor(input / other)

matching ``torch.floor_divide``.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3
(the elementwise family pattern):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_make_binary_explicit``, inlined
    with ``FloorDivideFwdKernel.op_func`` = fp32 ``floor(a / b)``,
    emitted as ``floor_divide_fwd_kernel``) is extracted from the GPU repo
    via ``extract_tl_kernel.py`` and imported as-is.  It serves as the
    reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 and K11 adaptations (decorator,
    grid/sync, ``threads``/``npt`` collapse into ``block_size``,
    padding strategy) are handled by the NPU component during
    re-implementation:
      - K11: factory signature keeps its positional shape/stride args
        but drops the GPU ``threads``/``num_per_thread`` execution
        parameters; the returned callable takes a single ``block_size``
        (the GPU ``threads * num_per_thread`` product).

  **Part B -- Kernel class** (ported in full from GPU ``BinaryKernel`` +
    ``FloorDivideFwdKernel``):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7/K10: no ``@torch.library.custom_op`` wrapper (the GPU binary
         kernel has none either); ``forward`` calls the compiled kernel
         directly.
    K8/K9: ``autotune_configs`` / ``autotune()`` / ``tune`` param --
         removed; heuristic config selection only,
         ``init_config(config)`` takes no ``tune`` argument.
    K11: GPU config ``{"strategy", "threads", "num_per_thread"}``
         collapsed into ``{"block_size": threads * num_per_thread}``;
         the strategy dispatch (direct / explicit_parallel /
         register_copy) collapses to the single NPU ``block_size``
         kernel (the extracted GPU function corresponds to the GPU
         ``explicit_parallel`` default).

The GPU op is float-only (``SUPPORTED_DTYPES = _FLOAT_DTYPES``) and
computes the division and floor in fp32 to avoid (1) the missing CUDA
``hfloor`` for ``cutlass::half_t`` and (2) fp16 quotient rounding before
the floor.  That fp32-intermediate semantics is preserved in the
extracted kernel body (``T.cast(..., "float32")`` -> ``T.floor`` ->
cast back).
"""

import math
from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Baseline source (the GPU-extracted reference).  ``integrate_kernel.py``
# rewrites this import into the baseline/perf_opt kernel-source selection
# block once the NPU kernel product is integrated.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .floor_divide_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .floor_divide_kernel import coalesce_broadcast_dims, floor_divide_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .floor_divide_kernel.perf_opt.FloorDivideFwdOp import coalesce_broadcast_dims
from .floor_divide_kernel.perf_opt.FloorDivideFwdOp import floor_divide_fwd_kernel
from .floor_divide_kernel.perf_opt.FloorDivideFwdOp import TUNED_BLOCK_SIZE

__all__ = ["FloorDivideFwdKernel"]

# Manifest dtype contract for the float binary family (GPU ``_FLOAT_DTYPES``).
_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

# K11: GPU ``threads(256) * num_per_thread`` collapsed into ``block_size``.
# The GPU ``BinaryKernel.default_config`` uses threads=256 with
# ``_strategy_npt("explicit_parallel", <dtype>)``; for the float dtypes
# supported here that heuristic returns 4 (fp32: 4-byte x 4 = 128-bit
# alignment; fp16/bf16 explicit_parallel: 4), giving 256 * 4 = 1024.
# --- baseline (Stage 3) ---
# _DEFAULT_BLOCK_SIZE = 256 * 4        # K11 GPU threads(256) * npt(4) = 1024
# --- perf_opt (Stage 4 tuned) ---
_DEFAULT_BLOCK_SIZE = TUNED_BLOCK_SIZE  # 12288 (perf_opt tuned, 1-D path)


class FloorDivideFwdKernel(Kernel):
    """Element-wise floor division with broadcast: ``y = floor(a / b)``.

    GPU ``FloorDivideFwdKernel(BinaryKernel)``.  The GPU broadcast lowering
    (``coalesce_broadcast_dims``) is preserved: the kernel receives the
    coalesced output shape plus per-operand strides (0 for broadcast dims)
    and computes offsets inside the TileLang body.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K11: ``default_config`` returns ``{"block_size": 1024}``
      (the GPU ``threads * num_per_thread`` product).
    - K8/K9: autotune removed; ``init_config(config)`` takes no ``tune``.
    - K7/K10: no custom_op wrapper; ``forward`` dispatches directly.

    Args:
        a_shape: Shape of input ``a``.
        b_shape: Shape of input ``b`` (broadcast against ``a_shape``).
        dtype: Torch dtype (one of ``float16`` / ``bfloat16`` / ``float32``).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 1024}``).
    """

    ascend_mode = "Developer"

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    # K11: block_size = GPU ``threads * num_per_thread`` product.
    _DEFAULT_BLOCK_SIZE = _DEFAULT_BLOCK_SIZE

    def __init__(
        self,
        a_shape,
        b_shape,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.a_shape = tuple(a_shape)
        self.b_shape = tuple(b_shape)
        # Broadcast lowering is the kernel's own choice (GPU parity).
        out_shape, coalesced_shape, a_strides, b_strides = coalesce_broadcast_dims(
            self.a_shape, self.b_shape,
        )
        self.out_shape = out_shape
        self.N_total = math.prod(out_shape)
        self.a_numel = math.prod(self.a_shape) if self.a_shape else 1
        self.b_numel = math.prod(self.b_shape) if self.b_shape else 1
        self.dtype = dtype
        self.kernel_dtype_str = Kernel.dtype_to_str(dtype)
        self.coalesced_shape = coalesced_shape
        self.a_strides = a_strides
        self.b_strides = b_strides
        # Build the factory callable (does not compile yet -- compilation
        # is deferred to forward()).  The imported function is the GPU
        # implementation; the NPU component rewrites it for
        # target="npuir" so its returned callable takes a single
        # ``block_size`` (K11).
        self.kernel = floor_divide_fwd_kernel(
            self.N_total,
            self.kernel_dtype_str,
            self.coalesced_shape,
            self.a_strides,
            self.b_strides,
            self.a_numel,
            self.b_numel,
        )
        self._compiled_fn = None
        self._compiled_block_size: Optional[int] = None
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: ``block_size = threads * npt``)."""
        return {"block_size": self._DEFAULT_BLOCK_SIZE}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run the floor-divide kernel on two flat contiguous 1-D tensors.

        K11: the (NPU-reimplemented) factory callable takes a single
        ``block_size`` argument and is cached per block_size.
        """
        block_size = self.config["block_size"]
        if self._compiled_fn is None or self._compiled_block_size != block_size:
            self._compiled_fn = self.kernel(block_size)
            self._compiled_block_size = block_size
        return self._compiled_fn(a, b)
