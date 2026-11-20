"""Benchmarks for FusedAddRMSNormFwdOp (NPU).

Ported from the GPU TileOPs benchmark for the fused-add-RMSNorm family.
Workloads and roofline formulas are loaded from the standalone manifest
(T3/T4: autotune params removed).

NOTE (deviation): the GPU manifest points this op's bench at the shared
``benchmarks/ops/bench_norm.py``; the NPU project has no such file, so this
dedicated file is used instead (recorded in the manifest entry).

Multi-input note: the op takes three tensor inputs (x / residual / weight), so
``workloads_to_params`` cannot be used; the manifest ``x_shape`` is read
directly through the same custom helper pattern as
``bench_ada_layer_norm.py``.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.norm.fused_add_rms_norm import FusedAddRMSNormFwdOp
from tileops.workloads.norm import FusedAddRMSNormWorkload

_FARN_OP_NAME = "FusedAddRMSNormFwdOp"


def _to_params(workloads):
    params = []
    for w in workloads:
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(m, n, dtype, id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("m, n, dtype", _to_params(load_workloads(_FARN_OP_NAME)))
def test_fused_add_rms_norm_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = FusedAddRMSNormWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = FusedAddRMSNormFwdOp(dtype=dtype)
    bm = ManifestBenchmark(_FARN_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
