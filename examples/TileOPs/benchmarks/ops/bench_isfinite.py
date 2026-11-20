"""Benchmarks for IsfiniteFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise.py`` -- the ``isfinite``
subset only (the shared GPU file also times the comparison / predicate family;
only ``isfinite`` is migrated).  Workloads and roofline formulas are loaded from
the standalone manifest.

Single-input op: ``IsfiniteFwdOp`` has one tensor input, so ``workloads_to_params``
(the single-input workload contract) applies directly.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.isfinite import IsfiniteFwdOp
from tileops.workloads.elementwise import IsfiniteWorkload

_ISFINITE_OP = "IsfiniteFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ISFINITE_OP, include_extra=True),
)
def test_isfinite_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = IsfiniteWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = IsfiniteFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_ISFINITE_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
