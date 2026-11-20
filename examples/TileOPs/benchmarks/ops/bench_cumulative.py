"""Benchmarks for CumprodFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_cumulative.py`` -- the CumprodFwdOp
subset only (the shared GPU file also benchmarks CumsumFwdOp, which is not
migrated).  Workload shapes and the roofline formula are loaded from the ops
manifest (``tileops/manifest/scan.yaml``).

NPU adaptations (T3/T4): the ``tune`` parametrization dimension and the
autotune try/except are removed -- the benchmark executes every manifest
workload shape.

Baseline note: the GPU reference program
(``CumulativeBenchmarkWorkload.ref_program`` -- an fp32-upcast cumulative
product used for correctness comparison) is an accuracy reference and is not
benchmarked here.  The native ``torch.cumprod`` operator is kept as the
performance baseline.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.scan.cumprod import CumprodFwdOp
from tileops.workloads.scan import CumprodFwdWorkload

_CUMPROD_OP = "CumprodFwdOp"


# Cumprod benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_CUMPROD_OP, include_extra=True),
)
def test_cumprod_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = CumprodFwdWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = CumprodFwdOp(dtype=dtype, **op_params)
    bm = ManifestBenchmark(_CUMPROD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    dim = op_params["dim"]

    def baseline_fn(x):
        return torch.cumprod(x, dim=dim)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
