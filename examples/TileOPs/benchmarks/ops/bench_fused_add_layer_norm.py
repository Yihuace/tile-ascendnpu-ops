"""Manifest-driven benchmark for FusedAddLayerNormFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_norm.py`` --
``test_fused_add_layer_norm_bench`` only.  The shared GPU file also benchmarks
the other norm ops (and the flash-attn / quack / torch-compile baselines); only
this op's TileOPs measurement is migrated (accuracy-only ``ref_program``
baselines stay in the correctness tests).

Multi-input note: ``FusedAddLayerNormFwdOp`` has 4 tensor inputs
(``x`` / ``residual`` / ``weight`` / ``bias``), so ``workloads_to_params``
cannot be used (its single-input workload contract raises ``KeyError`` for
multi-input ops).  The custom ``_to_params`` helper reads the manifest
``x_shape`` and ``dtypes`` instead -- the multi-input pattern of the
add-npu-op skill (S6), matching ``bench_ada_layer_norm.py``.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from tileops.workloads.norm import FusedAddLayerNormWorkload

_FUSED_OP = "FusedAddLayerNormFwdOp"


def _to_params(workloads: list[dict]) -> list:
    params = []
    for w in workloads:
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(m, n, dtype, id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("m, n, dtype", _to_params(load_workloads(_FUSED_OP)))
def test_fused_add_layer_norm_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = FusedAddLayerNormWorkload(m, n, dtype)
    x, residual, weight, bias = test.gen_inputs()

    op = FusedAddLayerNormFwdOp(dtype=dtype)
    bm = ManifestBenchmark(_FUSED_OP, op, test)
    result = bm.profile(op, x, residual, weight, bias)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
