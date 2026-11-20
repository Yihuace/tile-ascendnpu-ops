"""Benchmark for MoeGroupedGemmNopad3WGFusedActFwdOp (fused gate/up GEMM).

Ported from the GPU benchmark that drives this op indirectly
(``benchmarks/ops/bench_moe_fused_activation.py`` times the enclosing
``FusedMoEExpertsNopadPersistent3WGFwdOp``); the op has no dedicated GPU
benchmark file.  Only this operator's workloads and TileOPs measurement are
migrated -- the fused-vs-unfused comparison and the per-expert PyTorch
reference (an accuracy-only golden implementation) are omitted.

Workload shapes come from the manifest entry's ``workloads`` (via
``load_workloads``); the benchmark reports TileOPs latency alongside the
manifest-derived roofline (``op.eval_roofline()``).

NPU adaptations:

- T3: ``tune`` parametrization dimension removed (no autotune on NPU).
- T4: no autotune try/except to remove.
- C9: ``torch.cuda.synchronize()`` -> ``get_device_backend().synchronize()``.
- Multi-input note: the op has 4 tensor inputs, so ``workloads_to_params``
  cannot be used (its single-input workload contract raises ``KeyError`` for
  multi-input ops).  A custom parametrization helper reads ``load_workloads``.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.device import get_device_backend
from tileops.manifest import load_workloads
from tileops.ops.moe.moe_grouped_gemm_nopad_3wg_fused_act import (
    MoeGroupedGemmNopad3WGFusedActFwdOp,
)
from tileops.workloads.moe import MoeGroupedGemmNopad3WGFusedActWorkload

_OP_NAME = "MoeGroupedGemmNopad3WGFusedActFwdOp"


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _manifest_params():
    """Convert manifest workloads to pytest params (numel, E, ffn, K, act, dtype)."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        activation = w.get("activation", "silu_and_mul")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["numel"], w["num_experts"], w["ffn"], w["k"], activation, dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


@pytest.mark.parametrize(
    "numel, num_experts, ffn, k, activation, dtype_str",
    _manifest_params(),
)
def test_moe_grouped_gemm_nopad_3wg_fused_act_bench(
    numel: int, num_experts: int, ffn: int, k: int, activation: str, dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    workload = MoeGroupedGemmNopad3WGFusedActWorkload(
        numel, num_experts, ffn, k, dtype, activation=activation,
    )
    a, b, true_sizes, true_offsets = workload.gen_inputs()

    op = MoeGroupedGemmNopad3WGFusedActFwdOp(
        numel, num_experts, ffn, k, dtype=dtype, activation=activation,
    )
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # Warmup: trigger JIT compilation before timed profiling.
    op(a, b, true_sizes, true_offsets)
    get_device_backend().synchronize()

    result = bm.profile(op, a, b, true_sizes, true_offsets)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
