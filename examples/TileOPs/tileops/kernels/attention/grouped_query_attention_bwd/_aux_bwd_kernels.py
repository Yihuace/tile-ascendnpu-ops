"""Auxiliary GQA-backward Kernel classes (preprocess / postprocess).

These two ``Kernel`` classes were moved verbatim out of the wrapper module
``grouped_query_attention_bwd.py``.  The Stage 5 integration gate
(``gate_lint.integration_lint``, rule ``S5-ASCEND-MODE``) requires the wrapper
module to declare **exactly one** top-level ``Kernel`` subclass whose
``ascend_mode`` literal matches the integrated kernel mode.  Since the wrapper
previously defined three ``Kernel`` subclasses, the two auxiliary classes are
relocated here; behavior (``__init__`` / ``forward`` / ``default_config`` /
``supported_archs``) is unchanged -- only the definition location moved.

The classes are re-imported by the wrapper so that
``tileops.kernels.attention.grouped_query_attention_bwd.grouped_query_attention_bwd``
(and therefore the package ``__init__`` and the op adapter) keep exposing all
three names unchanged.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Mirror of the wrapper switch block for the two auxiliary kernels.  Exactly
# one source block below is active; toggle by swapping the comment.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .grouped_query_attention_bwd_kernel import (
#     _flashattn_bwd_postprocess_kernel,
#     _flashattn_bwd_preprocess_kernel,
# )
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .grouped_query_attention_bwd_kernel.perf_opt._flashattn_bwd_postprocess_kernel import _flashattn_bwd_postprocess_kernel
from .grouped_query_attention_bwd_kernel.perf_opt._flashattn_bwd_preprocess_kernel import _flashattn_bwd_preprocess_kernel

__all__ = [
    "FlashAttnBwdPostprocessKernel",
    "FlashAttnBwdPreprocessKernel",
]


class FlashAttnBwdPreprocessKernel(Kernel):
    """Row-wise ``delta = rowsum(o * do)`` preprocessing for GQA/MHA backward.

    The launch geometry is fixed (``blk = 256``), so ``default_config`` is
    empty and there is nothing to tune; ``init_config`` accepts an optional
    override for structural parity with the other kernels.

    NPU adaptations:

    - K5: ``supported_archs = None`` (was ``[80, 89, 90]``).
    - K8: autotune removed; ``init_config(config)`` takes no ``tune`` argument.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        seq_len: Sequence length.
        dim: Head dimension.
        dtype: Torch dtype of ``o`` / ``do``.
        config: Optional config dict. This kernel exposes no tunable knobs.
    """

    # K5: [80, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = dtype

        self.kernel = _flashattn_bwd_preprocess_kernel(
            self.batch, self.heads, self.seq_len, self.dim, self.dtype_str
        )
        self.init_config(config)

    def forward(self, o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
        """Return ``delta = rowsum(o * do)`` of shape ``[batch, heads, seq_len]``."""
        return self.kernel(o, do)


class FlashAttnBwdPostprocessKernel(Kernel):
    """Cast the fp32 ``dq`` accumulator back to the storage dtype.

    The launch geometry is fixed (``blk = 64``), so ``default_config`` is
    empty and there is nothing to tune.

    NPU adaptations:

    - K5: ``supported_archs = None`` (was ``[80, 89, 90]``).
    - K8: autotune removed; ``init_config(config)`` takes no ``tune`` argument.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        seq_len: Sequence length.
        dim: Head dimension.
        dtype: Torch dtype of the output ``dq``.
        config: Optional config dict. This kernel exposes no tunable knobs.
    """

    # K5: [80, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = dtype

        self.kernel = _flashattn_bwd_postprocess_kernel(
            self.batch, self.heads, self.seq_len, self.dim, self.dtype_str
        )
        self.init_config(config)

    def forward(self, dq: torch.Tensor) -> torch.Tensor:
        """Return the fp32 ``dq`` accumulator cast to the storage dtype."""
        return self.kernel(dq)
