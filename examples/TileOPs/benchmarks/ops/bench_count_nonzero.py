"""Benchmarks for CountNonzeroFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_logical_reduce.py`` -- the
``CountNonzeroFwdOp`` subset only (``AnyFwdOp`` / ``AllFwdOp`` are their own
migration units; all three share the GPU ``LogicalReduceKernel`` via
``op_kind``).

Measures latency, TFLOPS, and DRAM bandwidth of the TileOPs NPU kernel against
the native ``torch.count_nonzero`` baseline (a meaningful native operator, so
it is kept as a performance baseline).  Workload shapes, dtypes, and op-call
parameters (e.g. ``dim``) are loaded from the ops manifest
(``tileops/manifest/``) -- the benchmark must not hard-code op parameters that
are declared on manifest workload entries.

NPU adaptation (T3/T4): the GPU staged-rollout try/skip around ``bm.profile``
(autotune "No configurations to tune" and large-N shared memory errors) is
removed -- the benchmark must execute all manifest workload shapes.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp
from tileops.workloads.reduction import CountNonzeroWorkload

_COUNT_NONZERO_OP = "CountNonzeroFwdOp"


# CountNonzero benchmarks


@pytest.mark.parametrize(
    "shape, dtype, extra", workloads_to_params(_COUNT_NONZERO_OP, include_extra=True)
)
def test_count_nonzero_bench(shape: tuple, dtype: torch.dtype, extra: dict) -> None:
    workload = CountNonzeroWorkload(shape, dtype)
    inputs = workload.gen_inputs()

    op = CountNonzeroFwdOp(dtype=dtype, **extra)
    bm = ManifestBenchmark(_COUNT_NONZERO_OP, op, workload)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    dim = extra["dim"]

    def baseline_fn(x):
        return torch.count_nonzero(x, dim=dim).to(torch.int64)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
