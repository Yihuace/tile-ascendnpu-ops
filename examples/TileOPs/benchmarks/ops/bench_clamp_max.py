"""Manifest-driven benchmark for ClampMaxFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_independent_elementwise.py`` --
ClampMax subset only (the shared GPU file also benchmarks clamp / clamp_min,
which are separate ops).  Keeps this operator's workloads and performance
measurements.

Multi-input note: ``ClampMaxFwdOp`` has 2 tensor inputs, so
``workloads_to_params`` cannot be used (its single-input workload contract
raises ``KeyError`` for multi-input ops).  A custom parametrization helper
reads the manifest workloads instead -- see the multi-input pattern in the
add-npu-op skill (S6).

NPU adaptations (T3/T4): the GPU ClampMax benchmark has no ``tune``
parametrization / autotune try-except -- nothing to remove.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.elementwise.clamp_max import ClampMaxFwdOp
from tileops.workloads.elementwise import TensorClampManifestWorkload

_CLAMP_MAX_OP = "ClampMaxFwdOp"


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float32 else pytest.mark.full


def _clamp_max_params(workloads: list[dict]) -> list:
    """Parametrize (input_shape, max_shape, dtype) from the manifest.

    Two-input ops cannot use ``workloads_to_params`` (single-input
    contract); this helper mirrors the GPU ``_workloads_to_clamp_params``
    helper for the max-bound-only variant.
    """
    params = []
    for idx, w in enumerate(workloads):
        input_shape = tuple(w["input_shape"])
        max_shape = tuple(w["max_shape"])
        label = w.get("label", "x".join(str(dim) for dim in input_shape))
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(
                    input_shape,
                    max_shape,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=_mark(idx, dtype),
                )
            )
    return params


@pytest.mark.parametrize(
    "input_shape, max_shape, dtype",
    _clamp_max_params(load_workloads(_CLAMP_MAX_OP)),
)
def test_clamp_max_bench(
    input_shape: tuple[int, ...],
    max_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    test = TensorClampManifestWorkload(input_shape, dtype, max_shape=max_shape)
    x, t_max = test.gen_inputs()

    op = ClampMaxFwdOp(input=input_shape, max=max_shape, dtype=dtype)
    bm = ManifestBenchmark(_CLAMP_MAX_OP, op, test)
    result = bm.profile(op, x, t_max)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Native torch.minimum baseline (meaningful, runnable on NPU).
    result_bl = bm.profile(torch.minimum, x, t_max)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
