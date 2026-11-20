"""Benchmarks for ArgminFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_argreduce.py`` — ArgminFwdOp
subset only (ArgmaxFwdOp is its own migration unit; both share the GPU
``ArgreduceKernel``).

Measures latency, TFLOPS, and DRAM bandwidth of the TileOPs NPU kernel.
The PyTorch native ``torch.argmin`` baseline is commented out below
(disabled on request; restore by uncommenting the baseline block).
Workload shapes, dtypes, and op-call parameters (e.g. ``dim``) are loaded
from the ops manifest (``tileops/manifest/``) — the benchmark must not
hard-code op parameters that are declared on manifest workload entries.

NPU adaptation (T3/T4): the GPU staged-rollout try/skip around
``bm.profile`` (autotune "No configurations to tune" and large-N shared
memory errors) is removed — the benchmark must execute all manifest
workload shapes.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.reduction.argmin import ArgminFwdOp
from tileops.workloads.reduction import ArgminWorkload

_ARGMIN_OP = "ArgminFwdOp"


# Argmin benchmarks


@pytest.mark.parametrize("shape, dtype, extra", workloads_to_params(_ARGMIN_OP, include_extra=True))
def test_argmin_bench(shape: tuple, dtype: torch.dtype, extra: dict) -> None:
    workload = ArgminWorkload(shape, dtype)
    inputs = workload.gen_inputs()

    op = ArgminFwdOp(dtype=dtype, **extra)
    bm = ManifestBenchmark(_ARGMIN_OP, op, workload)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # torch native baseline disabled on request (also avoids the
    # locals()-leak case_id mismatch that dropped its records):
    # dim = extra["dim"]
    #
    # def baseline_fn(x):
    #     return x.argmin(dim=dim)
    #
    # result_bl = bm.profile(baseline_fn, *inputs)
    # BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
