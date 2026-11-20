"""Manifest-driven benchmark for BatchNormBwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_batch_norm.py`` --
``test_batch_norm_bwd_bench`` only. The shared GPU file also benchmarks
``BatchNormFwdOp`` (and the flag_gems / torch.compile / cuDNN baselines); only
this op's TileOPs measurement is migrated (the accuracy-only ``ref_program``
baselines stay in the correctness tests). T3: no ``tune`` step.

Multi-input note: ``BatchNormBwdOp`` has 5 tensor inputs, so
``workloads_to_params`` cannot be used (its single-input workload contract
raises ``KeyError`` for multi-input ops). The custom ``_to_params`` helper reads
the manifest ``x_shape`` and ``dtypes`` instead -- the multi-input pattern of
the add-npu-op skill (S6).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.norm.batch_norm_bwd import BatchNormBwdOp
from tileops.workloads.norm import BatchNormBwdWorkload

_BWD_OP = "BatchNormBwdOp"


def _to_params(workloads: list[dict]) -> list:
    params = []
    for idx, w in enumerate(workloads):
        n, c, s = w["x_shape"]
        label = w.get("label", f"{n}x{c}x{s}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            marks = pytest.mark.smoke if idx == 0 else pytest.mark.full
            params.append(
                pytest.param(n, c, s, dtype, id=f"{label}-{dtype_str}", marks=marks)
            )
    return params


@pytest.mark.parametrize("n, c, s, dtype", _to_params(load_workloads(_BWD_OP)))
def test_batch_norm_bwd_bench(n: int, c: int, s: int, dtype: torch.dtype) -> None:
    test = BatchNormBwdWorkload(n, c, s, dtype)
    grad_out, x, weight, mean, rstd = test.gen_inputs()

    op = BatchNormBwdOp()
    bm = ManifestBenchmark(_BWD_OP, op, test)
    result = bm.profile(op, grad_out, x, weight, mean, rstd)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
