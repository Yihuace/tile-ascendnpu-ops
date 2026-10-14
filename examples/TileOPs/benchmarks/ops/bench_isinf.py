"""Benchmarks for IsinfFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise.py`` -- the ``isinf``
subset only (the shared GPU file also times the comparison / predicate family;
only ``isinf`` is migrated).  Workloads and roofline formulas are loaded from the
standalone manifest.

Single-input op: ``IsinfFwdOp`` has one tensor input, so ``workloads_to_params``
(the single-input workload contract) applies directly.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.isinf import IsinfFwdOp
from tileops.workloads.elementwise import IsinfWorkload

_ISINF_OP = "IsinfFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ISINF_OP, include_extra=True),
)
def test_isinf_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = IsinfWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = IsinfFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_ISINF_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
