"""1D Complex-to-Complex FFT kernel (NPU-adapted).

Adapted from TileOPs ``tileops/kernels/fft.py`` -- ``FFTC2CKernel``.

The GPU implementation is a two-phase batched 1D C2C FFT: a fused
bit-reversal + SMEM butterfly stage (single launch) followed by radix-8 /
radix-4 / radix-2 LUT butterfly stages, driven by a pre-computed twiddle LUT.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_fft_c2c_kernel``) is extracted from the
    GPU repo via ``extract_tl_kernel.py`` and imported as-is.  It serves as the
    reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 adaptations (decorator, grid/sync semantics,
    ``threads`` removal, padding strategy) are handled by the NPU kernel
    component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[75, 80, 86, 89, 90]``).
    K7: ``custom_op("npub::fft_c2c")`` (was ``"top::fft_c2c_wrapped_kernel"``).
    K8: ``tune`` param removed; heuristic config selection only
        (``init_config(config)``).
    K9: ``threads`` removed from the custom_op signature, ``default_config``
        and the ``forward`` dispatch (GPU paired ``block_size`` with
        ``threads=512`` to size the SMEM coverage).  ``block_size`` is retained
        as the per-block work knob the NPU kernel component consumes.

Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
implementation whose inner callable takes ``(block_size, threads)``.  The
dispatch below passes the NPU-shaped argument list (no ``threads``); it
becomes callable once the NPU kernel component rewrites the factory for
``target="npuir"`` and drops ``threads`` from the callable signature (K3).
Running ``forward`` before that rewrite fails -- this is the expected scaffold
state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/fft.py.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .fft_c2c_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .fft_c2c_kernel import _fft_c2c_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .fft_c2c_kernel.perf_opt._fft_c2c_kernel import _fft_c2c_kernel

__all__ = ["FFTC2CKernel"]


# ---------------------------------------------------------------------------
# NPU dtype glue (K6/M1): the npuir factory is written against the *real*
# component dtype of the complex input, so the op-layer ``torch.complex64``
# must be mapped to the factory's ``"float32"``.  ``complex128`` has no npuir
# path (no complex dtype + no fp64 vector math; CG-2026-0019) -- its
# ``"float64"`` real dtype is rejected by the factory with a clear ValueError.
# ---------------------------------------------------------------------------
_REAL_DTYPE_STR = {
    "complex64": "float32",
    "complex128": "float64",
    "float32": "float32",
    "float64": "float64",
}


def _real_dtype_str(dtype: torch.dtype) -> str:
    """Map a torch component dtype to the npuir factory's real dtype string."""
    name = str(dtype).split(".")[-1]
    return _REAL_DTYPE_STR.get(name, name)


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::fft_c2c", mutates_args=())
def _fft_c2c_wrapped_kernel(
    n: int,
    batch_size: int,
    dtype: str,
    block_size: int,
    x_real: torch.Tensor,
    x_imag: torch.Tensor,
    lut_real: torch.Tensor,
    lut_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # NPU factory convention (K3): the single-level factory returns the
    # compiled prim_func directly, called with the four inputs; the
    # ``out_idx=[4, 5]`` allocation supplies (y_real, y_imag).  ``block_size``
    # is retained as the per-block work knob but the NPU whole-row kernel has
    # no per-stage blocking, so it is not threaded into the call.
    return _fft_c2c_kernel(n, batch_size, dtype)(
        x_real, x_imag, lut_real, lut_imag
    )


@_fft_c2c_wrapped_kernel.register_fake
def _(
    n: int,
    batch_size: int,
    dtype: str,
    block_size: int,
    *inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    real_dtype = inputs[0].dtype
    device = inputs[0].device
    return (
        torch.empty((batch_size, n), dtype=real_dtype, device=device),
        torch.empty((batch_size, n), dtype=real_dtype, device=device),
    )


# ---------------------------------------------------------------------------
# Kernel class (K5/K8/K9 adaptations)
# ---------------------------------------------------------------------------


class FFTC2CKernel(Kernel):
    """1D Complex-to-Complex FFT kernel with pre-computed twiddle LUT and
    shared-memory stage fusion.

    Inputs:
        x_real:    Real part of the input, shape (batch_size, n), float32/float64.
        x_imag:    Imaginary part of the input, shape (batch_size, n), float32/float64.
        lut_real:  Pre-computed twiddle real parts, shape (n - 1,).
        lut_imag:  Pre-computed twiddle imaginary parts, shape (n - 1,).

    Output:
        (y_real, y_imag): FFT output real/imaginary parts, shape (batch_size, n).

    Note:
        ``n`` must be a power of 2.  The LUT tensors are managed and passed by
        ``FFTC2COp``, not by this class.

    NPU adaptations:
        K5: ``supported_archs = None`` (all architectures; GPU pinned CUDA SM ints).
        K8: autotune removed -- heuristic config selection only;
            ``init_config(config)`` takes no ``tune`` argument.
        K9: ``threads`` removed from ``default_config`` and the ``forward``
            dispatch (GPU default was ``threads=512``).
    """

    # Scaffold placeholder for the NPU programming mode; the NPU kernel
    # component owns the final value and ``integrate_kernel.py`` re-syncs it
    # from the verified Stage 3 product.  FFT is a pure vector/butterfly kernel
    # (no Cube ``T.gemm``), so the Developer-mode default is used.
    ascend_mode = "Developer"

    # K5: [75, 80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        n: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.complex64,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.batch_size = batch_size
        self.dtype = dtype
        self.device_index = device_index

        # K6/M1: the npuir factory takes the *real* component dtype, so map
        # complex64 -> "float32" here (dtype_str would yield "complex64").
        self.real_dtype = _real_dtype_str(self.dtype)
        self.kernel = _fft_c2c_kernel(n, batch_size, self.real_dtype)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # K9: `threads` removed (GPU: {"block_size": 1024, "threads": 512}).
        # block_size stays the per-block LUT-stage work knob consumed by the
        # NPU kernel component.
        return {"block_size": 1024}

    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        lut_real: torch.Tensor,
        lut_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the 1D DFT on pre-split real/imaginary parts using a pre-built LUT.

        Args:
            x_real:   Real part, shape (batch_size, n).
            x_imag:   Imaginary part, shape (batch_size, n).
            lut_real: Twiddle real parts, shape (n - 1,).
            lut_imag: Twiddle imaginary parts, shape (n - 1,).

        Returns:
            (y_real, y_imag): FFT output real and imaginary parts, shape (batch_size, n).
        """
        # K9: no `threads` in the dispatch.
        return _fft_c2c_wrapped_kernel(
            self.n,
            self.batch_size,
            self.real_dtype,
            self.config["block_size"],
            x_real,
            x_imag,
            lut_real,
            lut_imag,
        )
