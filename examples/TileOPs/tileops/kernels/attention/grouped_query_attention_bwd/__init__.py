"""NPU kernel package for GroupedQueryAttentionBwdOp.

Re-exports the three backward kernels (preprocess / main / postprocess) from
the op package so the attention family ``__init__`` can import them by
package name.
"""

from tileops.kernels.attention.grouped_query_attention_bwd.grouped_query_attention_bwd import (
    FlashAttnBwdPostprocessKernel,
    FlashAttnBwdPreprocessKernel,
    GQABwdKernel,
)

__all__ = [
    "FlashAttnBwdPostprocessKernel",
    "FlashAttnBwdPreprocessKernel",
    "GQABwdKernel",
]
