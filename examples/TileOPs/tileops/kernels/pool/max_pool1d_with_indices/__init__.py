"""Max pooling 1D kernel package (NPU-adapted, values + int64 indices).

Part A holds the TileLang kernel factory extracted verbatim from the GPU
TileOPs repo (plus the ``pool_output_dim`` helper it calls); Part B of
:mod:`max_pool1d_with_indices` holds the custom_op wrapper and the Kernel class,
ported with the standard NPU adaptations.
"""

from tileops.kernels.pool.max_pool1d_with_indices.max_pool1d_with_indices import (
    MaxPool1dWithIndicesKernel,
)

__all__ = ["MaxPool1dWithIndicesKernel"]
