"""Benchmarks for AllFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_logical_reduce.py`` -- the
``AllFwdOp`` subset only (``AnyFwdOp`` / ``CountNonzeroFwdOp`` are their own
migration units; all three share the GPU ``LogicalReduceKernel`` via
``op_kind``).

Measures latency, TFLOPS, and DRAM bandwidth of the TileOPs NPU kernel against
the native ``torch.all`` baseline (a meaningful native operator, so it is kept
as a performance baseline).  Workload shapes, dtypes, and op-call parameters
(e.g. ``dim``) are loaded from the ops manifest (``tileops/manifest/``) -- the
benchmark must not hard-code op parameters that are declared on manifest
workload entries.

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
from tileops.ops.reduction.all import AllFwdOp
from tileops.workloads.reduction import AllWorkload

_ALL_OP = "AllFwdOp"


# All benchmarks


@pytest.mark.parametrize("shape, dtype, extra", workloads_to_params(_ALL_OP, include_extra=True))
def test_all_bench(shape: tuple, dtype: torch.dtype, extra: dict) -> None:
    workload = AllWorkload(shape, dtype)
    inputs = workload.gen_inputs()

    # The manifest omits ``dim`` for the mask-validation workloads (default
    # ``None`` = full reduction); the benchmark reduces the last dim for those,
    # matching the GPU bench's ``setdefault("dim", -1)``.
    extra.setdefault("dim", -1)
    op = AllFwdOp(dtype=dtype, **extra)
    bm = ManifestBenchmark(_ALL_OP, op, workload)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    dim = extra["dim"]
    keepdim = extra.get("keepdim", False)

    def baseline_fn(x):
        return x.bool().all(dim=dim, keepdim=keepdim)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
