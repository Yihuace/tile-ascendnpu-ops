"""Decomposed single-engine GQA kernels (npuir)."""

from ._gqa_decomposed_kernel import (
    gqa_prefill_paged_fwd,
    golden_gqa_prefill_paged,
    kernel_qk,
    kernel_softmax,
    kernel_pv,
)

__all__ = [
    "gqa_prefill_paged_fwd",
    "golden_gqa_prefill_paged",
    "kernel_qk",
    "kernel_softmax",
    "kernel_pv",
]
