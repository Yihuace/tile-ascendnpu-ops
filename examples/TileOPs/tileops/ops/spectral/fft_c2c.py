"""1D Complex-to-Complex Fast Fourier Transform operator (NPU).

Ported from TileOPs ``tileops/ops/fft.py`` -- ``FFTC2COp``.  Computes the
one-dimensional discrete Fourier transform of a complex input (equivalent to
``torch.fft.fft``); any leading dimensions are flattened into a single batch
dimension, processed in parallel by the kernel, and reshaped back.

NPU adaptations:

- O1: ``input.is_cuda`` device check -> ``backend.is_device_tensor(input)``.
- O3: ``tune`` parameter removed; the kernel constructor takes
  ``(n, batch_size, dtype, config=None, device_index=None)``.
- O6: the forward flow (validate -> split real/imag -> flatten batch -> build
  LUT -> kernel dispatch -> reshape back) is preserved unchanged.
- ``eval_roofline``: the GPU
  ``tileops.perf.formulas.fft_c2c_roofline`` re-implemented inline (the NPU
  project has no ``tileops.perf`` module; roofline arithmetic is
  device-agnostic).
"""

import math
from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.spectral.fft_c2c import FFTC2CKernel
from tileops.ops.op_base import Op

__all__ = ["FFTC2COp"]


class FFTC2COp(Op):
    """1D Complex-to-Complex Fast Fourier Transform operation.

    Computes the one-dimensional discrete Fourier transform of complex input.
    This is equivalent to ``torch.fft.fft``.

    Supports batched input: any leading dimensions are flattened into a single
    batch dimension, processed in parallel by the kernel, and reshaped back.

    Uses a pre-computed twiddle factor LUT and shared-memory butterfly fusion.

    Args:
        kernel_map: Optional custom kernel mapping for testing.
    """

    def __init__(
        self,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.n = None
        self.dtype = None

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple[int, int, torch.dtype, int | None], Kernel] = {}
        self._twiddle_cache: Dict[
            tuple[int, torch.dtype, int | None], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.kernel = None

    def _get_kernel(
        self,
        n: int,
        batch_size: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (n, batch_size, dtype, device_index)
        if key not in self._kernel_cache:
            # O3: no `tune` argument on the NPU kernel constructor.
            self._kernel_cache[key] = self.kernel_map["fft_c2c_kernel"](
                n,
                batch_size,
                dtype,
            )
        return self._kernel_cache[key]

    @staticmethod
    def _build_lut(
        n: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute the full twiddle factor LUT for all butterfly stages.

        For stage ``s``, ``half_m = 2^s`` twiddle factors are stored at offset
        ``(half_m - 1)``::

            LUT[half_m - 1 + k] = exp(-2πi * k / m),  k = 0..half_m-1,  m = 2*half_m

        All angles are computed in float64 for precision, then cast to the
        real component dtype (float32 for complex64, float64 for complex128).
        """
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        log2n = int(math.log2(n))
        lut_size = n - 1

        angles = torch.zeros(lut_size, dtype=torch.float64)
        for s in range(log2n):
            half_m = 1 << s
            m = half_m * 2
            k_vals = torch.arange(half_m, dtype=torch.float64)
            angles[half_m - 1 : 2 * half_m - 1] = -2.0 * math.pi * k_vals / m

        lut_real = torch.cos(angles).to(real_dtype).to(device)
        lut_imag = torch.sin(angles).to(real_dtype).to(device)
        return lut_real, lut_imag

    def _get_lut(
        self, n: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (n, dtype, device.index)
        if key not in self._twiddle_cache:
            self._twiddle_cache[key] = self._build_lut(n, dtype, device)
        return self._twiddle_cache[key]

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fft_c2c_kernel": FFTC2CKernel}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the 1D FFT of a complex input.

        Args:
            input: Input tensor of shape ``(..., n)`` with complex dtype.

        Returns:
            Output tensor of the same shape as ``input`` with the FFT applied
            along the last dimension.
        """
        backend = get_device_backend()
        x = input
        if not backend.is_device_tensor(x):
            raise ValueError(f"input must be a {backend.name} tensor, got device {x.device}")
        if x.dtype not in (torch.complex64, torch.complex128):
            raise ValueError(f"input.dtype must be complex64 or complex128, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("input must be at least 1D")
        n = x.shape[-1]
        if n <= 0 or n & (n - 1) != 0:
            raise ValueError(f"FFT size must be a positive power of 2, got {n}")

        x_real = x.real.contiguous()
        x_imag = x.imag.contiguous()
        original_shape = x.shape

        # Flatten all batch dimensions into a single batch dimension.
        batch_size = x_real[..., 0].numel() if x.ndim > 1 else 1
        x_real = x_real.reshape(batch_size, n)
        x_imag = x_imag.reshape(batch_size, n)

        self.n = n
        self.dtype = x.dtype
        self.twiddle_real, self.twiddle_imag = self._get_lut(n, x.dtype, x.device)
        kernel = self._get_kernel(n, batch_size, x.dtype, x.device.index)
        self.kernel = kernel
        y_real, y_imag = kernel(
            x_real, x_imag, self.twiddle_real, self.twiddle_imag
        )

        # Reshape back to the original shape.
        y_real = y_real.reshape(original_shape)
        y_imag = y_imag.reshape(original_shape)

        return torch.complex(y_real, y_imag)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inlined from the GPU ``tileops.perf.formulas.fft_c2c_roofline``::

            flops = batch * 5 * n * log2(n)
            bytes = batch * 2 * n * elem_bytes   # read + write one complex (..., n)

        ``batch`` is the flattened batch size bound by the last ``forward``
        call (the FFT size ``n`` and ``dtype`` are bound there as well).
        """
        if self.n is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind the dynamic FFT size and dtype"
            )
        n = int(self.n)
        elem_bytes = int(self.dtype.itemsize)
        kernel = getattr(self, "kernel", None)
        batch = int(getattr(kernel, "batch_size", 1) or 1)
        flops = int(batch * 5 * n * math.log2(n))
        nbytes = int(batch * 2 * n * elem_bytes)
        return flops, nbytes
