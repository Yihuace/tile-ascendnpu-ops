"""Benchmarks for BitwiseNotFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise.py`` -- the
``bitwise_not`` subset only (the shared GPU file also times bitwise_and / or /
xor / ...; only ``bitwise_not`` is migrated).  Workloads and roofline formulas
are loaded from the standalone manifest.

Single-input op: ``BitwiseNotFwdOp`` has one tensor input, so
``workloads_to_params`` (the single-input workload contract) applies directly.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.bitwise_not import BitwiseNotFwdOp
from tileops.workloads.elementwise import BitwiseNotWorkload

_BITWISE_NOT_OP = "BitwiseNotFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_BITWISE_NOT_OP, include_extra=True),
)
def test_bitwise_not_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = BitwiseNotWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = BitwiseNotFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_BITWISE_NOT_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
