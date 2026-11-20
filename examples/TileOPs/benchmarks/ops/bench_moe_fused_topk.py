"""Benchmark for FusedTopKOp.

Ported from GPU TileOPs ``benchmarks/ops/bench_moe_fused_topk.py``.

Workload shapes come from the manifest entry's ``workloads`` (via
``workloads_to_params``); the benchmark reports the TileOPs latency alongside
the manifest-derived roofline (``op.eval_roofline()`` via ``ManifestBenchmark``).

NPU adaptations:

- T3: no ``tune`` parametrization dimension (NPU uses heuristic config only).
- T4: no autotune try/except to remove.
- C9: ``torch.cuda.synchronize()`` -> ``get_device_backend().synchronize()``.
- The GPU benchmark's optional vLLM ``fused_topk`` baseline (not importable on
  NPU) and the handwritten PyTorch ``torch.softmax/sigmoid + torch.topk``
  fallback (an accuracy reference, not a native operator) are omitted entirely
  rather than left commented out.  Their correctness reference lives in
  ``tests/ops/test_moe_fused_topk.py``.

Real model configurations:
  Model              E    K  scoring   renorm
  Kimi K2          384   8  sigmoid   True
  Qwen3-235B-A22B  128   8  softmax   False
"""

import pytest

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark, workloads_to_params
from tileops.device import get_device_backend
from tileops.ops.moe import FusedTopKOp
from tileops.workloads.moe import FusedTopKWorkload

_OP_NAME = "FusedTopKOp"


@pytest.mark.parametrize(
    "shape, dtype, extra",
    workloads_to_params(_OP_NAME, include_extra=True),
)
def test_fused_topk_bench(shape, dtype, extra) -> None:
    num_tokens, num_experts = shape
    top_k = extra["top_k"]
    scoring_func = extra["scoring_func"]
    renormalize = extra["renormalize"]

    workload = FusedTopKWorkload(
        num_tokens, num_experts, top_k, scoring_func, renormalize, dtype
    )
    (gating_output,) = workload.gen_inputs()

    op = FusedTopKOp(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        scoring_func=scoring_func,
        renormalize=renormalize,
    )
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # Warmup: trigger JIT compilation before timed profiling.
    op(gating_output)
    get_device_backend().synchronize()

    result = bm.profile(op, gating_output)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
