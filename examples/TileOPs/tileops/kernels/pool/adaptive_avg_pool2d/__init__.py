"""Adaptive average pooling 2D kernel package (NPU-adapted).

Part A holds the TileLang kernel factory extracted verbatim from the GPU
TileOPs repo; Part B of :mod:`adaptive_avg_pool2d` holds the custom_op wrapper
and the Kernel class, ported with the standard NPU adaptations.
"""

from tileops.kernels.pool.adaptive_avg_pool2d.adaptive_avg_pool2d import (
    AdaptiveAvgPool2dKernel,
)

__all__ = ["AdaptiveAvgPool2dKernel"]
