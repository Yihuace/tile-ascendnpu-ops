"""Benchmark for Expm1FwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_unary_elementwise.py`` — the
``expm1`` subset only (the shared GPU file covers every implemented
elementwise unary manifest entry; only this migrated op's benchmark is
ported).

Workload shapes, dtypes, and the roofline formula are loaded from the
standalone manifest (``tileops/manifest/elementwise.yaml``).  The native
``torch.expm1`` is retained as an independent performance baseline (it is a
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
from tileops.ops.elementwise.expm1 import Expm1FwdOp
from tileops.workloads.elementwise import Expm1Workload

_EXPM1_OP = "Expm1FwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_EXPM1_OP, include_extra=True),
)
def test_expm1_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = Expm1Workload(shape, dtype)
    inputs = test.gen_inputs()

    op = Expm1FwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_EXPM1_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(torch.expm1, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
