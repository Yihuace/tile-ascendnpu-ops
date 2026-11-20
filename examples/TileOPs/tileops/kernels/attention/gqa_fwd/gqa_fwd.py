"""NPU dispatch for paged GQA prefill."""

from __future__ import annotations

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

from ._gqa_decomposed_kernel import gqa_prefill_paged_fwd
from ._gqa_decomposed_kernel._gqa_decomposed_kernel import fused_paged_gqa

__all__ = ["GQAPrefillPagedKernel"]


class GQAPrefillPagedKernel(Kernel):
    """Dispatch paged GQA prefill to a fused or decomposed kernel.

    Args:
        page_size: Tokens per cache page.
        is_causal: Apply the causal mask.
        sm_scale: Softmax scale; ``None`` uses ``1/sqrt(D)``.
        config: Optional kernel config override (unused; kept for the base API).
    """

    supported_archs: Optional[list] = None

    # Both the fused Vector kernel and the fallback Cube kernels use Expert mode.
    ascend_mode = "Expert"

    def __init__(
        self,
        page_size: int,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        self.page_size = page_size
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"page_size": self.page_size, "is_causal": self.is_causal}

    def forward(self, q, k_new, v_new, k_pages, v_pages, k_scale, v_scale,
                cu_seqlens_q, cache_seqlens, block_table) -> torch.Tensor:
        # Use the fallback kernel name until the fused path is selected.
        self.msprof_kernel_name = "main"
        total, heads, dim = q.shape
        batch = cache_seqlens.numel()
        kv_heads = k_new.shape[1]
        if (batch and total % batch == 0 and 0 < batch * heads <= 40
                and dim == 64 and self.page_size == 16
                and kv_heads > 0 and heads % kv_heads == 0
                and k_new.shape == v_new.shape == (total, kv_heads, dim)
                and k_pages.shape == v_pages.shape
                and k_pages.shape[1:] == (kv_heads, dim)
                and all(t.device == q.device and t.is_contiguous() and t.dtype == torch.float16
                        for t in (q, k_new, v_new, k_pages, v_pages))
                and all(t is not None and t.device == q.device and t.is_contiguous()
                        and t.dtype == torch.float32 and t.shape == (1,)
                        for t in (k_scale, v_scale))
                and block_table.device == q.device and block_table.is_contiguous()
                and block_table.dtype == torch.int32 and block_table.ndim == 2
                and block_table.shape[0] == batch
                and cu_seqlens_q.ndim == cache_seqlens.ndim == 1):
            qlen = total // batch
            cache_lens = cache_seqlens.tolist()
            cache = cache_lens[0]
            pages = cache // self.page_size
            if ((qlen, cache) in ((64, 128), (128, 256))
                    and pages <= block_table.shape[1]
                    and cu_seqlens_q.tolist() == list(range(0, total + 1, qlen))
                    and all(length == cache for length in cache_lens)):
                table = [row[:pages] for row in block_table.tolist()]
                if all(0 <= pid < k_pages.shape[0] // self.page_size
                       for row in table for pid in row):
                    contiguous = all(
                        all(pid == row[0] + i for i, pid in enumerate(row))
                        for row in table
                    )
                    tiles_per_head = 40 // (batch * heads)
                    rows = (qlen + tiles_per_head - 1) // tiles_per_head
                    scale = dim ** -0.5 if self.sm_scale is None else float(self.sm_scale)
                    fused = fused_paged_gqa(
                        batch, qlen, cache, heads, kv_heads, dim,
                        k_pages.shape[0], block_table.shape[1], self.page_size,
                        self.is_causal, scale, "float16", rows, contiguous,
                    )
                    output = fused(q, k_new, v_new, k_pages, v_pages, k_scale, v_scale, block_table)
                    self.msprof_kernel_name = "paged_gqa"
                    return output
        return gqa_prefill_paged_fwd(
            q, k_new, v_new, k_pages, v_pages, k_scale, v_scale,
            cu_seqlens_q, cache_seqlens, block_table,
            page_size=self.page_size,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
        )
