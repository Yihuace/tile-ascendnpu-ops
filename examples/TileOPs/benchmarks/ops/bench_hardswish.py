"""Benchmarks for HardswishFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise.py`` — HardSwish subset.
Workloads and roofline formulas are loaded from the standalone manifest.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.hardswish import HardswishFwdOp
from tileops.workloads.elementwise import HardswishWorkload

_HARDSWISH_OP = "HardswishFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_HARDSWISH_OP, include_extra=True),
)
def test_hardswish_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = HardswishWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = HardswishFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_HARDSWISH_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
