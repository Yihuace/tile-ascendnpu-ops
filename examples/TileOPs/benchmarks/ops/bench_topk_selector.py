"""Benchmarks for TopkSelectorOp (NPU).

Ported from the GPU TileOPs benchmark for TopkSelectorOp.  Workloads and roofline
formulas are loaded from the standalone manifest (T3/T4: autotune params removed).

Note: the delivered NPU architecture is a single block with a serial row loop, so
these numbers are correctness-driven rather than bandwidth-driven -- see the
manifest's workload-scale note.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.attention_indexing.topk_selector import TopkSelectorOp
from tileops.workloads.attention_indexing import TopkSelectorWorkload

_TOPK_OP = "TopkSelectorOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_TOPK_OP, include_extra=True),
)
def test_topk_selector_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    k = op_params.pop("k", 16)
    tile = op_params.pop("tile", None)
    test = TopkSelectorWorkload(shape, dtype, k)
    inputs = test.gen_inputs()

    op = TopkSelectorOp(M=shape[0], N=shape[1], k=k, tile=tile, **op_params)
    bm = ManifestBenchmark(_TOPK_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
