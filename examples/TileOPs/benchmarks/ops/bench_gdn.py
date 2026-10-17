"""Manifest-driven benchmark for GatedDeltaNetPrefillFwdOp (NPU).

Prefill only (equal-length, ``B=1``, key-major state) — the in-scope workloads
are listed in ``DESIGN.md`` §1.  Shapes / dtypes are taken from the standalone
manifest (``tileops/manifest/linear_attention.yaml``, key ``GDNFwdOp``), filtered
to the prefill rows.

STAGE 0 SCAFFOLD: skipped until the kernel lands in Stage 3; the case list below
is the in-scope subset and will be wired to ``benchmarks.api`` then.
"""

import pytest
import torch  # noqa: F401

IN_SCOPE = "Stage 3 pending: GDN prefill kernel bodies (6-kernel pipeline)"
pytestmark = pytest.mark.skip(reason=IN_SCOPE)

# label, B, S, H, HV, K, V, has_initial_state, dtype
IN_SCOPE_CASES = [
    ("prefill-4k", 1, 4096, 16, 16, 128, 128, False, "float16"),
    ("prefill-4k", 1, 4096, 16, 16, 128, 128, False, "bfloat16"),
    ("prefill-12k", 1, 12288, 16, 16, 128, 128, False, "float16"),
    ("prefill-32k", 1, 32768, 16, 16, 128, 128, False, "bfloat16"),
    ("prefill-3000", 1, 3000, 16, 16, 128, 128, False, "bfloat16"),
    ("prefill-continued", 1, 4096, 16, 16, 128, 128, True, "bfloat16"),
    ("prefill-narrow-state", 1, 4096, 16, 16, 64, 64, False, "bfloat16"),
]


@pytest.mark.parametrize(
    "label,B,S,H,HV,K,V,has_state,dtype",
    IN_SCOPE_CASES,
    ids=[f"{c[0]}-{c[8]}" for c in IN_SCOPE_CASES],
)
def test_gdn_prefill_bench(label, B, S, H, HV, K, V, has_state, dtype):
    """msprof-latency benchmark for one in-scope prefill workload."""
    raise AssertionError("Stage 3 pending")
