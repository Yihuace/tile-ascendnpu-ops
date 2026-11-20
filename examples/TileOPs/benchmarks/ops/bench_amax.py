"""Benchmarks for AmaxFwdOp (NPU).

Per-op benchmark file: the shared GPU ``bench_reduce.py`` was split so that
only this migration unit's workload runs when ``AmaxFwdOp`` is integrated (the
shared file also carried AminFwdOp workloads, which belong to a separate
migration unit).

Measures latency, TFLOPS, and DRAM bandwidth of the TileOPs NPU kernel
against the native ``torch.amax`` baseline.  Workload shapes, dtypes, and
op-call parameters (e.g. ``dim``) are loaded from the ops manifest
(``tileops/manifest/``) -- the benchmark must not hard-code op parameters
that are declared on manifest workload entries.

NPU adaptation (T3/T4): the GPU try/skip around ``bm.profile`` (autotune
"No configurations to tune") is removed -- the benchmark must execute all
manifest workload shapes.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.reduction.amax import AmaxFwdOp
from tileops.workloads.reduction import AmaxWorkload

_AMAX_OP = "AmaxFwdOp"


# Amax benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_AMAX_OP, include_extra=True),
)
def test_amax_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = AmaxWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = AmaxFwdOp(dtype=dtype, **op_params)
    bm = ManifestBenchmark(_AMAX_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.amax(dim=dim, keepdim=keepdim)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
