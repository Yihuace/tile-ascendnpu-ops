"""Benchmark for MoePermuteNopadFwdOp (tight layout, no padding).

Ported from GPU TileOPs ``benchmarks/ops/bench_moe_permute_nopad.py``.

Workload shapes come from the manifest entry's ``workloads`` (via
``load_workloads``); the benchmark reports the TileOPs latency alongside the
manifest-derived roofline (``op.eval_roofline()``).

NPU adaptations:

- T3: no ``tune`` parametrization dimension (NPU uses heuristic config only).
- T4: no autotune try/except to remove.
- C9: ``torch.cuda.synchronize()`` -> ``get_device_backend().synchronize()``.
- The GPU benchmark's optional vLLM ``moe_permute`` baseline and the
  PyTorch counting-sort gather baseline are accuracy/performance references
  that are not runnable (vLLM) or accuracy-only (torch counting sort) on
  NPU; their ``bm.profile(...)`` / ``BenchmarkReport.record(...)`` pairs are
  omitted rather than left commented out.  The correctness reference lives
  in ``tests/ops/test_moe_permute_nopad.py``.

Multi-input note: ``MoePermuteNopadFwdOp`` has 2 tensor inputs, so
``workloads_to_params`` cannot be used (its single-input workload contract
raises ``KeyError`` for multi-input ops).  A custom parametrization helper
reads ``load_workloads`` instead.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.device import get_device_backend
from tileops.manifest import load_workloads
from tileops.ops.moe import MoePermuteNopadFwdOp
from tileops.workloads.moe import MoePermuteWorkload

_OP_NAME = "MoePermuteNopadFwdOp"


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _manifest_params():
    """Convert manifest workloads to pytest params (multi-input op)."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        total_tokens, hidden_size = w["hidden_states_shape"]
        topk_tokens, top_k = w["topk_ids_shape"]
        assert topk_tokens == total_tokens
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                total_tokens, top_k, w["num_experts"], hidden_size, dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


@pytest.mark.parametrize(
    "total_tokens, top_k, num_experts, hidden_size, dtype_str",
    _manifest_params(),
)
def test_moe_permute_nopad_bench(
    total_tokens: int, top_k: int, num_experts: int, hidden_size: int, dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    workload = MoePermuteWorkload(total_tokens, top_k, num_experts, hidden_size, dtype)
    torch.manual_seed(42)
    hidden_states, topk_ids = workload.gen_inputs()

    op = MoePermuteNopadFwdOp(num_experts=num_experts, dtype=dtype)
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # Warmup: trigger JIT compilation before timed profiling.
    op(hidden_states, topk_ids)
    get_device_backend().synchronize()

    result = bm.profile(op, hidden_states, topk_ids)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
