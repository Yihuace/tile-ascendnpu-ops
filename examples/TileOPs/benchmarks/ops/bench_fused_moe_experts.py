"""Benchmark for FusedMoEExpertsNopadPersistent3WGFwdOp (NPU).

Ported from the GPU TileOPs benchmark
``benchmarks/ops/bench_fused_moe_experts.py``.  Only this operator's workloads
and TileOPs measurement are migrated -- the GPU-only baselines (vLLM Triton /
CUTLASS fused_experts, unavailable on NPU) and the per-expert PyTorch reference
(an accuracy-only golden implementation) are omitted.

Workload shapes come from the manifest entry's ``workloads`` (via
``load_workloads``); the benchmark reports TileOPs latency alongside the
manifest-derived roofline (``op.eval_roofline()``).

NPU adaptations:

- T3: ``tune`` parametrization dimension removed (no autotune on NPU).
- T4: no autotune try/except to remove.
- C9: ``torch.cuda.synchronize()`` -> ``get_device_backend().synchronize()``.
- Multi-input note: the op has 5 tensor inputs (plus pre-allocated output and
  two workspaces), so ``workloads_to_params`` cannot be used (its single-input
  workload contract raises ``KeyError`` for multi-input ops).  A custom
  parametrization helper reads ``load_workloads``.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.device import get_device_backend
from tileops.manifest import load_workloads
from tileops.ops.moe.fused_moe_experts_nopad_persistent_3wg import (
    FusedMoEExpertsNopadPersistent3WGFwdOp,
)
from tileops.workloads.moe import MoeExpertsWorkload

_OP_NAME = "FusedMoEExpertsNopadPersistent3WGFwdOp"

# msprof --kernel-name filter for this op.  The op is a composition (scan /
# permute / 3WG GEMM [ + fused activation ] / unpermute); msprof can only filter
# to one device entry, so the baseline pins it to the dominant mix AIC GEMM.
# Declared here (and on the Op itself, see msprof_kernel_name) so the caliber is
# traceable: perf_opt/workload_inventory.json's ``benchmark_source`` must carry
# ``kernel=main_mix_aic``.
_MSPROF_KERNEL_NAME = "main_mix_aic"


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _manifest_params():
    """Convert manifest workloads to pytest params."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["num_tokens"], w["num_experts"], w["top_k"],
                w["hidden_size"], w["ffn_size"], dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, hidden_size, ffn_size, dtype_str",
    _manifest_params(),
)
def test_fused_moe_experts_nopad_persistent_3wg_bench(
    num_tokens: int, num_experts: int, top_k: int, hidden_size: int,
    ffn_size: int, dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    workload = MoeExpertsWorkload(
        num_tokens, num_experts, top_k, hidden_size, ffn_size, dtype,
    )
    hidden, w1, w2, topk_weights, topk_ids = workload.gen_inputs()

    op = FusedMoEExpertsNopadPersistent3WGFwdOp(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size, dtype=dtype,
    )

    device = get_device_backend().name
    output = torch.empty(num_tokens, hidden_size, dtype=dtype, device=device)
    ws1 = torch.empty(0, dtype=dtype, device=device)
    ws2 = torch.empty(0, dtype=dtype, device=device)

    # Profile the Op instance directly, passing ``forward``'s positional
    # arguments in order.  ``bench_kernel_msprof`` requires an Op instance or a
    # module-level callable; a local closure is rejected with a TypeError and
    # silently falls back to events mode, which loses the msprof ratio.
    profile_args = (
        output, hidden, w1, w2, topk_weights, topk_ids,
        None,  # expert_map
        ws1, ws2, num_experts,
    )

    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # Warmup: trigger JIT compilation before timed profiling.
    op(*profile_args)
    get_device_backend().synchronize()

    result = bm.profile(op, *profile_args)
    BenchmarkReport.record(op, locals(), result, tag="tileops-nopad-3wg")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
