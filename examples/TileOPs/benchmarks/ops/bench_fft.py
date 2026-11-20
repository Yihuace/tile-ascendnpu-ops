"""Performance benchmarks for FFTC2COp (NPU).

Ported from TileOPs ``benchmarks/ops/bench_fft.py``.

Single-input note: ``FFTC2COp`` has one tensor input, so the manifest-driven
``workloads_to_params`` / ``ManifestBenchmark`` path applies directly (unlike
the multi-input ops, which need a custom parametrization helper).

The ``torch.fft.fft`` baseline is a native torch operator (a performance
baseline, not an accuracy-only golden), so it is kept; the GPU tag
``torch-cufft`` is renamed to ``torch-fft`` on NPU.

NPU adaptations (T3): ``tune=True`` removed from the ``FFTC2COp`` constructor;
``torch.cuda.synchronize()`` replaced with the device-backend synchronize.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.device import get_device_backend
from tileops.ops.spectral.fft_c2c import FFTC2COp
from tileops.workloads.spectral import FFTWorkload

_OP_NAME = "FFTC2COp"


class FFTTestBaseline(FFTWorkload):
    """Adds a baseline ``ref_program`` for benchmark profiling."""

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, dim=-1)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_OP_NAME))
def test_fft_bench(shape: tuple, dtype: torch.dtype) -> None:
    # complex128 is not implementable on the NPU backend: npuir has no complex
    # dtype and no fp64 vector elementwise ops (CG-2026-0019), so the kernel
    # raises ValueError.  Mirror the correctness suite (``tests/ops/test_fft.py``,
    # which marks its complex128 cases xfail) so the unsupported case is
    # disclosed rather than aborting the whole benchmark stage.
    if dtype == torch.complex128:
        pytest.xfail(
            "complex128 unsupported on NPU/npuir (CG-2026-0019: no complex "
            "dtype + no fp64 vector elementwise ops)"
        )

    n = shape[-1]
    batch_shape = shape[:-1]
    test = FFTTestBaseline(n, dtype, batch_shape=batch_shape)
    inputs = test.gen_inputs()

    op = FFTC2COp()

    # Warmup: trigger JIT compilation before timed profiling.
    op(*inputs)
    get_device_backend().synchronize()

    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-fft")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
