"""Max pooling 2D kernel package (NPU-adapted, single output).

Part A holds the TileLang kernel factory extracted verbatim from the GPU
TileOPs repo (plus the ``pool_output_dim`` helper it calls); Part B of
:mod:`max_pool2d` holds the custom_op wrapper and the Kernel class, ported with
the standard NPU adaptations.
"""

from tileops.kernels.pool.max_pool2d.max_pool2d import MaxPool2dKernel

__all__ = ["MaxPool2dKernel"]
