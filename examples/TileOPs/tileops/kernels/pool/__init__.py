"""Pool kernel family (NPU-adapted)."""

from tileops.kernels.pool.adaptive_avg_pool2d import AdaptiveAvgPool2dKernel
from tileops.kernels.pool.adaptive_max_pool2d import AdaptiveMaxPool2dKernel
from tileops.kernels.pool.adaptive_max_pool2d_with_indices import AdaptiveMaxPool2dWithIndicesKernel
from tileops.kernels.pool.max_pool1d_with_indices import MaxPool1dWithIndicesKernel
from tileops.kernels.pool.max_pool2d import MaxPool2dKernel

__all__ = [
    "AdaptiveAvgPool2dKernel",
    "AdaptiveMaxPool2dKernel",
    "AdaptiveMaxPool2dWithIndicesKernel",
    "MaxPool1dWithIndicesKernel",
    "MaxPool2dKernel",
]
