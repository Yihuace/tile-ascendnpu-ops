"""Benchmarks for HardsigmoidFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise.py`` — Hardsigmoid subset.
Workloads and roofline formulas are loaded from the standalone manifest.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.hardsigmoid import HardsigmoidFwdOp
from tileops.workloads.elementwise import HardsigmoidWorkload

_HARDSIGMOID_OP = "HardsigmoidFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_HARDSIGMOID_OP, include_extra=True),
)
def test_hardsigmoid_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = HardsigmoidWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = HardsigmoidFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_HARDSIGMOID_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
