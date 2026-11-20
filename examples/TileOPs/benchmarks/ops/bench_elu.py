"""Benchmark for EluFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise_manifest.py`` — the
``elu`` subset only (the shared GPU file also covers LeakyReLU and Hardtanh;
only this migrated op's benchmark is ported).

Workload shapes, dtypes and the roofline formula are loaded from the
standalone manifest (``tileops/manifest/elementwise.yaml``).  The native
``torch.nn.functional.elu`` is retained as an independent performance
baseline (it is a native operator, not an accuracy-only reference).  T3/T4:
the GPU ``tune`` parametrization and the "No configurations to tune"
try/except are dropped — the NPU kernel uses heuristic config selection only.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.elu import EluFwdOp
from tileops.workloads.elementwise import EluWorkload

_ELU_OP = "EluFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ELU_OP, include_extra=True),
)
def test_elu_manifest_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = EluWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = EluFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_ELU_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(lambda x: F.elu(x, 1.0), *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
