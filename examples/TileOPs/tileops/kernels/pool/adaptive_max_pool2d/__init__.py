"""Adaptive max pooling 2D kernel package (NPU-adapted).

Part A holds the TileLang kernel factory extracted verbatim from the GPU
TileOPs repo; Part B of :mod:`adaptive_max_pool2d` holds the custom_op wrapper
and the Kernel class, ported with the standard NPU adaptations.
"""

from tileops.kernels.pool.adaptive_max_pool2d.adaptive_max_pool2d import (
    AdaptiveMaxPool2dKernel,
)

__all__ = ["AdaptiveMaxPool2dKernel"]
