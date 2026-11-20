"""Benchmark for MoeGroupedGemmNopadFwdOp (tight, no-pad grouped GEMM).

Ported from GPU TileOPs ``benchmarks/ops/bench_moe_grouped_gemm_nopad.py``.

Workload shapes come from the manifest entry's ``workloads`` (via
``load_workloads``); the benchmark reports TileOPs latency alongside the
manifest-derived roofline (``op.eval_roofline()``).

NPU adaptations:

- T3: ``tune`` parametrization dimension removed (no autotune on NPU).
- T4: no autotune try/except to remove.
- C9: ``torch.cuda.synchronize()`` -> ``get_device_backend().synchronize()``.
- The per-expert NT matmul PyTorch loop is an accuracy reference only
  (identical to the correctness ``_ref_grouped_gemm_nopad``), so its
  ``bm.profile(...)`` / ``BenchmarkReport.record(...)`` pair is omitted from
  the NPU benchmark rather than recorded as a performance baseline.

Multi-input note: ``MoeGroupedGemmNopadFwdOp`` has 4 tensor inputs, so
``workloads_to_params`` cannot be used (its single-input workload contract
raises ``KeyError`` for multi-input ops).  A custom parametrization helper
reads ``load_workloads`` instead.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.device import get_device_backend
from tileops.manifest import load_workloads
from tileops.ops.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp
from tileops.workloads.moe import MoeGroupedGemmNopadWorkload

_OP_NAME = "MoeGroupedGemmNopadFwdOp"


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _manifest_params():
    """Convert manifest workloads to pytest params (numel, E, N, K, dtype)."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["numel"], w["num_experts"], w["n"], w["k"], dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


@pytest.mark.parametrize(
    "numel, num_experts, n, k, dtype_str",
    _manifest_params(),
)
def test_moe_grouped_gemm_nopad_bench(
    numel: int, num_experts: int, n: int, k: int, dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    workload = MoeGroupedGemmNopadWorkload(numel, num_experts, n, k, dtype)
    a, b, true_sizes, true_offsets = workload.gen_inputs()

    op = MoeGroupedGemmNopadFwdOp(numel, num_experts, n, k, dtype=dtype)
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # Warmup: trigger JIT compilation before timed profiling.
    op(a, b, true_sizes, true_offsets)
    get_device_backend().synchronize()

    result = bm.profile(op, a, b, true_sizes, true_offsets)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
