"""Manifest-driven benchmark for GtFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise_manifest.py`` --
Gt subset only (the shared GPU file covers every implemented elementwise
manifest entry; only this op's benchmark is ported).

Multi-input note: ``GtFwdOp`` has 2 tensor inputs, so ``workloads_to_params``
cannot be used (its single-input workload contract raises ``KeyError`` for
multi-input ops).  A custom parametrization helper reads the manifest
workloads instead -- see the multi-input pattern in the add-npu-op skill (S6).

NPU adaptations (T3/T4): no ``tune`` parametrization / autotune try-except
in the GPU Gt benchmark -- nothing to remove.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
)
from tileops.manifest import load_workloads
from tileops.ops.elementwise.gt import GtFwdOp
from tileops.workloads.elementwise import BinaryManifestWorkload

_GT_OP = "GtFwdOp"


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float32 else pytest.mark.full


def _binary_params(workloads: list[dict]) -> list:
    """Parametrize (input_shape, other_shape, dtype) from the manifest.

    Two-input ops cannot use ``workloads_to_params`` (single-input
    contract); this helper mirrors the GPU ``_binary_params`` helper.
    """
    params = []
    for idx, w in enumerate(workloads):
        input_shape = tuple(w["input_shape"])
        other_shape = tuple(w["other_shape"])
        label = w.get("label", "x".join(str(dim) for dim in input_shape))
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(
                    input_shape,
                    other_shape,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=_mark(idx, dtype),
                )
            )
    return params


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    _binary_params(load_workloads(_GT_OP)),
)
def test_gt_manifest_bench(
    input_shape: tuple[int, ...],
    other_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    a, b = test.gen_inputs()
    op = GtFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_GT_OP, op, test)
    result = bm.profile(op, a, b)
    BenchmarkReport.record(op, locals(), result, tag="tileops")
    # Native torch.gt baseline (meaningful, runnable on NPU).
    result_bl = bm.profile(torch.gt, a, b)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
