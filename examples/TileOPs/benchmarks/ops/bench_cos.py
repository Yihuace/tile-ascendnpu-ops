"""Benchmarks for CosFwdOp (NPU).

Ported from the GPU TileOPs benchmark for CosFwdOp
(``benchmarks/ops/bench_unary_elementwise.py`` per the GPU manifest
``source.bench``).  Workloads and roofline formulas come from the standalone
manifest (T3/T4: autotune params removed).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.cos import CosFwdOp
from tileops.workloads.elementwise import CosWorkload

_COS_OP = "CosFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_COS_OP, include_extra=True),
)
def test_cos_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = CosWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = CosFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_COS_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
