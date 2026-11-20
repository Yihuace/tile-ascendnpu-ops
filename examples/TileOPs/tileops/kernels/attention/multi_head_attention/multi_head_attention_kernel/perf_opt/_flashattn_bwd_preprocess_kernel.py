"""Baseline pass-through for the wrapper's `perf_opt` switch block (Stage 4).

The backward *preprocess* kernel is out of scope for this stage: only the main
kernel (`_gqa_bwd_kernel`) was tuned.  This module re-exports the baseline
factory unchanged so that a single wrapper switch block can point all three
backward kernels at `perf_opt/` with zero behaviour change here.
"""

from .._flashattn_bwd_preprocess_kernel import (  # noqa: F401
    _flashattn_bwd_preprocess_kernel,
)

__all__ = ["_flashattn_bwd_preprocess_kernel"]
