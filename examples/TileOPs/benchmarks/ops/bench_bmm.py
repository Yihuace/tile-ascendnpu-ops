"""Manifest-driven benchmark for BmmFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_bmm.py`` -- the BmmFwdOp subset
only.  The shared GPU file also benchmarks ``BmmFp8Op`` (flashinfer baseline),
which is not migrated.

NPU adaptations (T3-T4): the ``tune`` parameter is removed; the flashinfer /
fp8 baselines are dropped (not runnable / not migrated).  The torch.bmm
baseline is a native operator and is kept as the independent performance
reference.

Multi-input note: ``BmmFwdOp`` has 2 tensor inputs, so ``workloads_to_params``
cannot be used (its single-input workload contract raises ``KeyError`` for
multi-input ops).  A custom parametrization helper reads the manifest
workloads instead (see the add-npu-op skill S6 multi-input pattern).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.bmm.bmm import BmmFwdOp
from tileops.workloads.bmm import BmmWorkload

_OP_NAME = "BmmFwdOp"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _manifest_params() -> list:
    """Convert manifest workloads to pytest params (batch, m, n, k, dtype)."""
    params = []
    for idx, w in enumerate(load_workloads(_OP_NAME)):
        label = w.get("label", "unlabeled")
        marks = pytest.mark.smoke if idx == 0 else pytest.mark.full
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["b"], w["m"], w["n"], w["k"], dtype_str,
                id=f"{label}-{dtype_str}", marks=marks,
            ))
    return params


@pytest.mark.parametrize("batch, m, n, k, dtype_str", _manifest_params())
def test_bmm_bench(batch: int, m: int, n: int, k: int, dtype_str: str) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    workload = BmmWorkload(batch, m, n, k, dtype)
    a, b = workload.gen_inputs()

    op = BmmFwdOp()
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # eval_roofline() is read lazily after profiling, by which point
    # forward() has bound the dims.
    result = bm.profile(op, a, b)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Native baseline (torch.bmm) -- an independent performance reference.
    result_bl = bm.profile(torch.bmm, a, b)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-bmm")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
