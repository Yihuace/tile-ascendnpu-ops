"""Benchmarks for ClampScalarFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise.py`` -- the ClampScalar
subset only.  Workloads and roofline formulas are loaded from the standalone
manifest.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.clamp_scalar import ClampScalarFwdOp
from tileops.workloads.elementwise import ClampScalarWorkload

_CLAMP_SCALAR_OP = "ClampScalarFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_CLAMP_SCALAR_OP, include_extra=True),
)
def test_clamp_scalar_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = ClampScalarWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = ClampScalarFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_CLAMP_SCALAR_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
