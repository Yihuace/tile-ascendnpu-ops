"""Aggregated NPU kernels migrated by the conductor pipeline.

Re-exports the integrated correction-bias (Cb) kernel module produced for
fused_topk.  Generated in the same shape as ``fused_topk_kernel/__init__.py``;
edit the per-function module instead.
"""

from ._fused_topk_kernel_cb import _fused_topk_kernel_cb

__all__ = [
    "_fused_topk_kernel_cb",
]
