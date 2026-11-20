"""Benchmark for ErfFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_unary_elementwise.py`` — the
``erf`` subset only (the shared GPU file covers every implemented
elementwise unary manifest entry; only this migrated op's benchmark is
ported).

Workload shapes, dtypes, and the roofline formula are loaded from the
standalone manifest (``tileops/manifest/elementwise.yaml``).  The native
``torch.erf`` is retained as an independent performance baseline (it is a
native operator, not an accuracy-only reference).  T3/T4: the GPU ``tune``
parametrization and the "No configurations to tune" try/except are dropped
— the NPU kernel uses heuristic config selection only.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.erf import ErfFwdOp
from tileops.workloads.elementwise import ErfWorkload

_ERF_OP = "ErfFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ERF_OP, include_extra=True),
)
def test_erf_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = ErfWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = ErfFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_ERF_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(torch.erf, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
