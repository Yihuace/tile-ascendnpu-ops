"""Benchmarks for L1NormFwdOp / L2NormFwdOp / InfNormFwdOp (NPU).

Ported from the GPU TileOPs benchmark for the vector-norm family
(``benchmarks/ops/bench_vector_norm.py`` per the GPU manifest ``source.bench``).
Workloads and roofline formulas are loaded from the standalone manifest
(T3/T4: autotune params removed).

All three ops share one bench file because they share one kernel family; the
manifest ``source.bench`` for each of them points here.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.reduction.vector_norm import InfNormFwdOp, L1NormFwdOp, L2NormFwdOp
from tileops.workloads.reduction import InfNormWorkload, L1NormWorkload, L2NormWorkload

_L1_OP = "L1NormFwdOp"
_L2_OP = "L2NormFwdOp"
_INF_OP = "InfNormFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_L1_OP, include_extra=True),
)
def test_l1_norm_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = L1NormWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = L1NormFwdOp(dtype=dtype, **op_params)
    bm = ManifestBenchmark(_L1_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_L2_OP, include_extra=True),
)
def test_l2_norm_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = L2NormWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = L2NormFwdOp(dtype=dtype, **op_params)
    bm = ManifestBenchmark(_L2_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_INF_OP, include_extra=True),
)
def test_inf_norm_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = InfNormWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = InfNormFwdOp(dtype=dtype, **op_params)
    bm = ManifestBenchmark(_INF_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
