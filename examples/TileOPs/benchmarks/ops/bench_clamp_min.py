"""Benchmarks for ClampMinFwdOp (NPU).

Mirrors ``bench_clamp_scalar.py`` restricted to the ClampMin
(``torch.clamp_min``) subset.  Workloads and roofline formulas are loaded from
the standalone manifest.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.clamp_min import ClampMinFwdOp
from tileops.workloads.elementwise import ClampMinWorkload

_CLAMP_MIN_OP = "ClampMinFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_CLAMP_MIN_OP, include_extra=True),
)
def test_clamp_min_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = ClampMinWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = ClampMinFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_CLAMP_MIN_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
