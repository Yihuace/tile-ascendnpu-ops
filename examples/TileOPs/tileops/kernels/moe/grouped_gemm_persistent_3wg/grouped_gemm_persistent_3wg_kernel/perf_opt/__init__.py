"""Stage 4 (perf_opt) package for ``grouped_gemm_persistent_3wg``.

Re-exports the Stage-4 final module under the gate-contract name
(``perf_opt/{operator_name}.py``).  The Stage-3 baseline lives one level up in
``grouped_gemm_persistent_3wg_kernel/_persistent_grouped_gemm_v2_kernel.py`` and
is never modified from here; the previous current-best
(``perf_opt/_persistent_grouped_gemm_v2_kernel.py``) and the round-7 experiment
copies are kept alongside for revert / provenance.
"""

from .GroupedGemmPersistent3WG import (
    TUNED_DEFAULT_CONFIG,
    _persistent_grouped_gemm_v2_kernel,
    _run_gemm,
    _select_block_n,
    _use_prefetch_pair,
)

__all__ = [
    "TUNED_DEFAULT_CONFIG",
    "_persistent_grouped_gemm_v2_kernel",
    "_run_gemm",
    "_select_block_n",
    "_use_prefetch_pair",
]
