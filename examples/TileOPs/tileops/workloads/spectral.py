"""Workload definitions for the spectral op family (NPU).

Ported from TileOPs ``workloads/fft.py`` -- the ``FFTWorkload`` subset
(``FFTC2COp``).  The GPU file defines only this single workload class.

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"`` in the original TileOPs code.
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase


class FFTWorkload(WorkloadBase):
    """Workload for the 1D complex-to-complex FFT (``torch.fft.fft``).

    Args:
        n: FFT size along the last dimension (a positive power of 2).
        dtype: ``torch.complex64`` or ``torch.complex128``.
        batch_shape: Leading (batch) dimensions; ``()`` for unbatched input.
    """

    def __init__(self, n: int, dtype: torch.dtype, batch_shape: tuple = ()):
        self.n = n
        self.dtype = dtype
        self.batch_shape = batch_shape

    def gen_inputs(self) -> tuple[torch.Tensor]:
        # W1: device resolved via the device backend (was device="cuda").
        device = get_device_backend().name
        # W2 (NPU): torch_npu has no complex ``randn``/``normal`` --
        # ``aclnnInplaceNormal`` rejects ``DT_COMPLEX64``/``DT_COMPLEX128``.
        # Draw two independent real normal tensors and combine them with
        # ``torch.complex`` (NPU-supported for float32->complex64 and
        # float64->complex128).  This is the same distribution as the original
        # ``randn(dtype=complex)``: real and imaginary parts are independent
        # standard normals.  The real component dtype follows the complex
        # dtype (float32 for complex64, float64 for complex128).
        if self.dtype == torch.complex64:
            real_dtype = torch.float32
        elif self.dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            real_dtype = self.dtype
        real = torch.randn(*self.batch_shape, self.n, device=device, dtype=real_dtype)
        imag = torch.randn(*self.batch_shape, self.n, device=device, dtype=real_dtype)
        return (torch.complex(real, imag),)
