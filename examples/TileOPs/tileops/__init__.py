"""Standalone NPU benchmark framework.

Extracted from TileOPs (GPU/TileLang-based) and adapted for NPU backends.
No dependency on the original TileOPs repository.

Layer structure (mirrors the original end-to-end flow):

    manifest  ── spec source-of-truth (YAML)
        │
    workloads ── input generation (shape + dtype)
        │
    ops       ── Op ABC: validation, reshape, kernel dispatch, roofline
        │
    kernels   ── Kernel ABC: device-specific implementation
        │
    testing   ── TestBase: correctness vs PyTorch reference
        │
    benchmark ── BenchmarkBase: latency / TFLOPS / bandwidth

All device-specific code is funneled through :mod:`tileops.device`,
the single GPU-to-NPU adaptation surface.
"""

from tileops.device import get_device_backend

__all__ = ["get_device_backend"]
