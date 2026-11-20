"""LeFwd forward kernel using TileLang (NPU-adapted).

Implements element-wise less-equal with PyTorch broadcast semantics:
    y = (input <= other)         # bool output

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3
(the elementwise family pattern):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_make_binary_explicit``, inlined
    with ``LeFwdKernel.op_func = a <= b`` as ``le_fwd_kernel``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` and imported
    as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 and K11 adaptations
    (decorator, grid/sync, ``threads``/``npt`` collapse into
    ``block_size``, padding strategy) are handled by the NPU component
    during re-implementation:
      - K11: factory signature keeps its positional shape/stride args
        but drops the GPU ``threads``/``num_per_thread`` execution
        parameters; the returned callable takes a single ``block_size``
        (the GPU ``threads * num_per_thread`` product).

  **Part B -- Kernel class** (ported in full from GPU ``BinaryKernel`` +
    ``LeFwdKernel`` / ``LeBoolStorageFwdKernel``):
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

  Comparison-op note: the GPU ``LeFwdKernel`` declares
  ``OUTPUT_DTYPE = torch.bool`` and passes ``output_dtype="bool"`` into
  the factory, so the kernel writes a 1-byte ``bool`` tensor.  That
  output-dtype plumbing is preserved here (the NPU kernel component may
  carry the 1-byte output as ``int8`` storage and reinterpret it at the
  Op boundary, mirroring the bitwise migrations' CG adapter).

  The GPU op routes ``torch.bool`` inputs through a dedicated
  ``LeBoolStorageFwdKernel`` (public bool storage viewed as ``uint8``)
  because the CUDA codegen cannot lower vectorised ``boolx<N>`` loads.
  Both GPU kernel classes are ported here; on the NPU side they share the
  single extracted ``le_fwd_kernel`` (``a <= b`` on ``uint8`` storage
  yields the same 1/0 boolean byte pattern as the GPU bool-storage
  ``bitwise_or(bitwise_xor(a, 1), b)`` op_func), keeping one TileLang
  kernel function for the op per the skill's elementwise model.
"""

import math
from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A imports: the extracted GPU TileLang kernel functions.  The
# conductor's ``integrate_kernel.py`` rewrites this import into the
# baseline/perf_opt source-selection block once the NPU kernel component
# lands the ``target="npuir"`` re-implementation in ``le_kernel/``.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at .le_kernel/perf_opt/{func}.py
# and pass their L0/L1 regression; the baseline (Stage 3) source is active
# otherwise.  ``pytest tests/ops/`` and ``pytest benchmarks/ops/`` dispatch
# through whichever source is active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .le_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .le_kernel import coalesce_broadcast_dims, le_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .le_kernel.perf_opt.LeFwdOp import coalesce_broadcast_dims, le_fwd_kernel
from .le_kernel.perf_opt.LeFwdOp import TUNED_BLOCK_BY_DTYPE

__all__ = ["LeBoolStorageFwdKernel", "LeFwdKernel"]

# Manifest dtype contract for the comparison family (GPU ``_BINARY_FULL_DTYPES``
# = ``_BITWISE_DTYPES + (float16, bfloat16, float32)``).
_LE_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

# K11: GPU ``threads(256) * num_per_thread`` collapsed into ``block_size``.
# The GPU ``BinaryKernel.default_config`` uses threads=256 with
# ``_strategy_npt("explicit_parallel", <dtype>)``: 4 for fp32/fp16/bf16 and
# 8 for the integer dtypes; the uint8 bool-storage path uses npt=16.
_THREADS = 256
_NPT_FLOAT = 4       # _strategy_npt("explicit_parallel", fp16/bf16/fp32)
_NPT_INT = 8         # _strategy_npt("explicit_parallel", int*) / ("direct", bool)
_NPT_BOOL_STORAGE = 16


def _default_block_size(dtype: torch.dtype) -> int:
    """Stage 4 tuned per-dtype ``block_size`` (perf_opt ``TUNED_BLOCK_BY_DTYPE``).

    Toggle together with the source block above: the baseline (Stage 3) form
    is the K11 GPU ``threads * num_per_thread`` product (1024 float / 2048 int).
    """
    return TUNED_BLOCK_BY_DTYPE[Kernel.dtype_to_str(dtype)]


class LeFwdKernel(Kernel):
    """Element-wise less-equal with broadcast: ``y = (a <= b)`` (bool output).

    Comparison-dtype path (GPU ``LeFwdKernel(BinaryKernel)`` with
    ``OUTPUT_DTYPE = torch.bool``). The GPU broadcast lowering
    (``coalesce_broadcast_dims``) is preserved: the kernel receives the
    coalesced output shape plus per-operand strides (0 for broadcast dims)
    and computes offsets inside the TileLang body.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K11: ``default_config`` returns ``{"block_size": threads * npt}``
      (1024 for float dtypes, 2048 for the integer dtypes).
    - K8/K9: autotune removed; ``init_config(config)`` takes no ``tune``.
    - K7/K10: no custom_op wrapper; ``forward`` dispatches directly.

    Args:
        a_shape: Shape of input ``a``.
        b_shape: Shape of input ``b`` (broadcast against ``a_shape``).
        dtype: Torch dtype (one of the manifest comparison dtypes).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 2048}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel (CG-2026-0010).
    ascend_mode = "Developer"

    SUPPORTED_DTYPES = _LE_DTYPES
    OUTPUT_DTYPE = torch.bool

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
        self.N_total = math.prod(out_shape) if out_shape else 1
        self.a_numel = math.prod(self.a_shape) if self.a_shape else 1
        self.b_numel = math.prod(self.b_shape) if self.b_shape else 1
        self.dtype = dtype
        # GPU parity: comparison ops emit a bool output (``OUTPUT_DTYPE``);
        # ``_make_binary_explicit`` receives ``output_dtype="bool"``.
        self.output_dtype_str = Kernel.dtype_to_str(self.OUTPUT_DTYPE)
        self.kernel_dtype_str = Kernel.dtype_to_str(dtype)
        self.coalesced_shape = coalesced_shape
        self.a_strides = a_strides
        self.b_strides = b_strides
        # Build the factory callable (does not compile yet — compilation
        # is deferred to forward()).  The imported function is the GPU
        # implementation; the NPU component rewrites it for
        # target="npuir" so its returned callable takes a single
        # ``block_size`` (K11).
        self.kernel = le_fwd_kernel(
            self.N_total,
            self.kernel_dtype_str,
            self.coalesced_shape,
            self.a_strides,
            self.b_strides,
            self.a_numel,
            self.b_numel,
            output_dtype=self.output_dtype_str,
        )
        self._compiled_fn = None
        self._compiled_block_size: Optional[int] = None
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: ``block_size = threads * npt``)."""
        return {"block_size": _default_block_size(self.dtype)}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run the less-equal kernel on two flat contiguous 1-D tensors.

        K11: the (NPU-reimplemented) factory callable takes a single
        ``block_size`` argument and is cached per block_size.
        """
        block_size = self.config["block_size"]
        if self._compiled_fn is None or self._compiled_block_size != block_size:
            self._compiled_fn = self.kernel(block_size)
            self._compiled_block_size = block_size
        return self._compiled_fn(a, b)


class LeBoolStorageFwdKernel(LeFwdKernel):
    """Element-wise less-equal on uint8-backed bool storage.

    GPU ``LeBoolStorageFwdKernel(_Uint8StorageBinaryKernel)``: the public
    ``torch.bool`` tensors are viewed as ``uint8`` by the Op layer
    (TileLang cannot lower vectorised ``boolx<N>`` loads on the CUDA
    target).  On NPU the two kernel variants share the same extracted
    ``le_fwd_kernel``; this class narrows the dtype contract and keeps the
    GPU bool-storage default ``block_size`` (npt=16).
    """

    SUPPORTED_DTYPES = (torch.uint8,)

    @property
    def default_config(self) -> dict:
        return {"block_size": _THREADS * _NPT_BOOL_STORAGE}
