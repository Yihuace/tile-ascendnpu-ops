"""Benchmark for LeakyReluFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise_manifest.py`` — the
``leaky_relu`` subset only (the shared GPU file also covers Elu and
Hardtanh; only this migrated op's benchmark is ported).

Workload shapes, dtypes and the roofline formula are loaded from the
standalone manifest (``tileops/manifest/elementwise.yaml``).  The native
``torch.nn.functional.leaky_relu`` is retained as an independent performance
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
from tileops.ops.elementwise.leaky_relu import LeakyReluFwdOp
from tileops.workloads.elementwise import LeakyReluWorkload

_LEAKY_RELU_OP = "LeakyReluFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_LEAKY_RELU_OP, include_extra=True),
)
def test_leaky_relu_manifest_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = LeakyReluWorkload(shape, dtype)
    inputs = test.gen_inputs()

    # The op is constructed with the default negative_slope (0.01); the
    # native torch baseline uses the same 0.01 so both measure the same math.
    op = LeakyReluFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_LEAKY_RELU_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(lambda x: F.leaky_relu(x, 0.01), *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
