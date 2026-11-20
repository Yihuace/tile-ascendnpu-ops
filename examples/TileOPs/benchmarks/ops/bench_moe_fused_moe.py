"""Benchmark for FusedMoeFwdOp (routed MoE FFN) -- NPU.

Per-op benchmark file: the shared GPU ``benchmarks/ops/bench_moe_fused_moe.py``
was split so that only ``FusedMoeFwdOp`` workloads run when it is integrated;
the correction-bias ``FusedMoeFwdCbFwdOp`` workloads now live in
``bench_moe_fused_moe_cb.py``.  The GPU-only baselines (vLLM ``fused_experts``,
unavailable on NPU) and the per-expert PyTorch reference (an accuracy-only
golden implementation) are omitted; accuracy references live in the
correctness tests.

Workload shapes come from the op's manifest ``workloads`` (via
``load_workloads``); the benchmark reports TileOPs latency alongside the
manifest-derived roofline (``op.eval_roofline()``).

NPU adaptations:

- T3: ``tune`` parametrization dimension removed (no autotune on NPU).
- T4: no autotune try/except to remove.
- C9: ``torch.cuda.synchronize()`` -> ``get_device_backend().synchronize()``.
- Multi-input note: the op takes more than 1 tensor input (4 for
  ``FusedMoeFwdOp``), so ``workloads_to_params`` cannot be used (its
  single-input workload contract raises ``KeyError`` for multi-input ops).  A
  custom parametrization helper reads ``load_workloads``.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.device import get_device_backend
from tileops.manifest import load_workloads
from tileops.ops.moe import FusedMoeFwdOp
from tileops.workloads.moe import FusedMoeWorkload

_OP_NAME = "FusedMoeFwdOp"

# msprof --kernel-name filter for this op.  The op is a composition orchestrator
# (fused_topk -> prepare -> experts [permute / 3WG GEMM / activation / unpermute]
# -> finalize); msprof can only filter to one device entry, so the baseline pins
# it to the dominant mix AIC GEMM.  Declared here (and on the Op itself, see
# msprof_kernel_name) so the caliber is traceable:
# perf_opt/workload_inventory.json's ``benchmark_source`` must carry
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
                w["hidden_size"], w["ffn_size"],
                w["scoring_func"], bool(w.get("renormalize", False)),
                float(w.get("routed_scaling_factor", 1.0)),
                dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
    " scoring_func, renormalize, routed_scaling_factor, dtype_str",
    _manifest_params(),
)
def test_fused_moe_fwd_bench(
    num_tokens: int, num_experts: int, top_k: int, hidden_size: int,
    ffn_size: int, scoring_func: str, renormalize: bool,
    routed_scaling_factor: float, dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    workload = FusedMoeWorkload(
        num_tokens, num_experts, top_k, hidden_size, ffn_size,
        scoring_func, renormalize, False, routed_scaling_factor, dtype,
    )
    hidden, gating, _correction_bias, w_gate_up, w_down = workload.gen_inputs()

    op = FusedMoeFwdOp(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        scoring_func=scoring_func, renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor, dtype=dtype,
    )

    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # Warmup: trigger JIT compilation before timed profiling.
    op(hidden, gating, w_gate_up, w_down)
    get_device_backend().synchronize()

    result = bm.profile(op, hidden, gating, w_gate_up, w_down)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
