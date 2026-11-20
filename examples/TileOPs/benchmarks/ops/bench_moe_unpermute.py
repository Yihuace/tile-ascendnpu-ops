"""Benchmark for MoeUnpermuteFwdOp.

Ported from GPU TileOPs ``benchmarks/ops/bench_moe_unpermute.py``.

Workload shapes come from the manifest entry's ``workloads`` (via
``load_workloads``); the benchmark reports the TileOPs latency alongside the
manifest-derived roofline (``op.eval_roofline()``).

NPU adaptations:

- T3: no ``tune`` parametrization dimension (NPU uses heuristic config only).
- T4: no autotune try/except to remove.
- C9: ``torch.cuda.synchronize()`` -> ``get_device_backend().synchronize()``.
- The GPU benchmark's optional vLLM ``moe_unpermute`` baseline (not
  importable on NPU) and the handwritten PyTorch gather + weighted-sum
  baseline (an accuracy/performance reference, not a native operator) are
  omitted entirely rather than left commented out.  Their correctness
  reference lives in ``tests/ops/test_moe_unpermute.py``.

Multi-input note: ``MoeUnpermuteFwdOp`` has 3 tensor inputs
(``mm2_pad``, ``fwd_idx``, ``topk_weights``), so ``workloads_to_params``
cannot be used (its single-input workload contract raises ``KeyError`` for
multi-input ops).  A custom parametrization helper reads ``load_workloads``
instead.

Real model configurations:
  Model              H     K
  Kimi K2          7168   8
  DeepSeek-V3      7168   8
  Qwen3-235B-A22B  7168   8
  Qwen3-30B-A3B    3072   8
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.device import get_device_backend
from tileops.manifest import load_workloads
from tileops.ops.moe import MoeUnpermuteFwdOp
from tileops.workloads.moe import MoeUnpermuteWorkload

_OP_NAME = "MoeUnpermuteFwdOp"


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _manifest_params():
    """Convert manifest workloads to pytest params (multi-input op)."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["total_tokens"], w["top_k"], w["hidden_size"], dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


@pytest.mark.parametrize(
    "total_tokens, top_k, hidden_size, dtype_str",
    _manifest_params(),
)
def test_moe_unpermute_bench(
    total_tokens: int, top_k: int, hidden_size: int, dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    workload = MoeUnpermuteWorkload(total_tokens, top_k, hidden_size, dtype)
    mm2_pad, fwd_idx, topk_weights = workload.gen_inputs()

    op = MoeUnpermuteFwdOp(total_tokens, top_k, hidden_size, dtype)
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # Warmup: trigger JIT compilation before timed profiling.
    op(mm2_pad, fwd_idx, topk_weights)
    get_device_backend().synchronize()

    result = bm.profile(op, mm2_pad, fwd_idx, topk_weights)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
