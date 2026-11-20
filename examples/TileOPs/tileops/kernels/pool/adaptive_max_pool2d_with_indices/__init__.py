"""Adaptive max pooling 2D with indices kernel package (NPU-adapted).

Part A holds the TileLang kernel factory extracted verbatim from the GPU
TileOPs repo; Part B of :mod:`adaptive_max_pool2d_with_indices` holds the
custom_op wrapper and the Kernel class, ported with the standard NPU
adaptations.
"""

from tileops.kernels.pool.adaptive_max_pool2d_with_indices.adaptive_max_pool2d_with_indices import (
    AdaptiveMaxPool2dWithIndicesKernel,
)

__all__ = ["AdaptiveMaxPool2dWithIndicesKernel"]
