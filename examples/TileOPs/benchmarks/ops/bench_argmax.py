"""Benchmarks for ArgmaxFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_argreduce.py`` — ArgmaxFwdOp
subset only (ArgminFwdOp is its own migration unit).

Measures latency, TFLOPS, and DRAM bandwidth of the TileOPs NPU kernel.
The PyTorch native ``torch.argmax`` baseline is commented out below
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
from tileops.ops.reduction.argmax import ArgmaxFwdOp
from tileops.workloads.reduction import ArgmaxWorkload

_ARGMAX_OP = "ArgmaxFwdOp"


# Argmax benchmarks


@pytest.mark.parametrize("shape, dtype, extra", workloads_to_params(_ARGMAX_OP, include_extra=True))
def test_argmax_bench(shape: tuple, dtype: torch.dtype, extra: dict) -> None:
    workload = ArgmaxWorkload(shape, dtype)
    inputs = workload.gen_inputs()

    op = ArgmaxFwdOp(dtype=dtype, **extra)
    bm = ManifestBenchmark(_ARGMAX_OP, op, workload)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # torch native baseline disabled on request (also avoids the
    # locals()-leak case_id mismatch that dropped its records):
    # dim = extra["dim"]
    #
    # def baseline_fn(x):
    #     return x.argmax(dim=dim)
    #
    # result_bl = bm.profile(baseline_fn, *inputs)
    # BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
