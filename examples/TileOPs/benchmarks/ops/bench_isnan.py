"""Benchmarks for IsnanFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise.py`` -- the ``isnan``
subset only (the shared GPU file also times the comparison / predicate family;
only ``isnan`` is migrated).  Workloads and roofline formulas are loaded from
the standalone manifest.

Single-input op: ``IsnanFwdOp`` has one tensor input, so ``workloads_to_params``
(the single-input workload contract) applies directly.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.isnan import IsnanFwdOp
from tileops.workloads.elementwise import IsnanWorkload

_ISNAN_OP = "IsnanFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ISNAN_OP, include_extra=True),
)
def test_isnan_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = IsnanWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = IsnanFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_ISNAN_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
