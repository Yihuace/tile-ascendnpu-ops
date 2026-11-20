"""Pool op family (NPU-adapted)."""

from tileops.ops.pool.adaptive_avg_pool2d import AdaptiveAvgPool2dFwdOp
from tileops.ops.pool.adaptive_max_pool2d import AdaptiveMaxPool2dFwdOp
from tileops.ops.pool.adaptive_max_pool2d_with_indices import AdaptiveMaxPool2dIndicesFwdOp
from tileops.ops.pool.max_pool1d_with_indices import MaxPool1dIndicesFwdOp
from tileops.ops.pool.max_pool2d import MaxPool2dFwdOp

__all__ = [
    "AdaptiveAvgPool2dFwdOp",
    "AdaptiveMaxPool2dFwdOp",
    "AdaptiveMaxPool2dIndicesFwdOp",
    "MaxPool1dIndicesFwdOp",
    "MaxPool2dFwdOp",
]
