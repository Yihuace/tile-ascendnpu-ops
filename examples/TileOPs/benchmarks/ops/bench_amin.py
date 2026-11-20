"""Benchmarks for AminFwdOp (NPU).

Per-op benchmark file: the shared GPU ``bench_reduce.py`` was split so that
only this migration unit's workload runs when ``AminFwdOp`` is integrated (the
shared file also carried AmaxFwdOp workloads, which belong to a separate
migration unit).

Measures latency, TFLOPS, and DRAM bandwidth of the TileOPs NPU kernel
against the native ``torch.amin`` baseline.  Workload shapes, dtypes, and
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
from tileops.ops.reduction.amin import AminFwdOp
from tileops.workloads.reduction import AminWorkload

_AMIN_OP = "AminFwdOp"


# Amin benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_AMIN_OP, include_extra=True),
)
def test_amin_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = AminWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = AminFwdOp(dtype=dtype, **op_params)
    bm = ManifestBenchmark(_AMIN_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.amin(dim=dim, keepdim=keepdim)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
