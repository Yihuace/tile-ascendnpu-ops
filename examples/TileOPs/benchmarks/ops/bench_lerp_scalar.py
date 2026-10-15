"""Manifest-driven benchmark for LerpFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise_manifest.py`` --
scalar-weight lerp subset only (the shared GPU file covers every implemented
elementwise manifest entry; only this op's benchmark is ported).

Multi-input note: ``LerpFwdOp`` has 2 tensor inputs, so ``workloads_to_params``
cannot be used (its single-input workload contract raises ``KeyError`` for
multi-input ops).  A custom parametrization helper reads the manifest workloads
instead -- see the multi-input pattern in ``bench_lerp_tensor.py``.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
)
from tileops.manifest import load_workloads
from tileops.ops.elementwise.lerp_scalar import LerpFwdOp
from tileops.workloads.elementwise import LerpScalarWorkload

_LERP_SCALAR_OP = "LerpFwdOp"
_DEFAULT_WEIGHT = 0.5


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float16 else pytest.mark.full


def _shape_dtype_params(workloads: list[dict]) -> list:
    """Parametrize (input_shape, end_shape, dtype) from manifest workloads."""
    params = []
    for idx, w in enumerate(workloads):
        input_shape = tuple(w["input_shape"])
        end_shape = tuple(w["end_shape"])
        label = w.get("label", "x".join(str(dim) for dim in input_shape))
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(
                    input_shape,
                    end_shape,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=_mark(idx, dtype),
                )
            )
    return params


@pytest.mark.parametrize(
    "input_shape, end_shape, dtype",
    _shape_dtype_params(load_workloads(_LERP_SCALAR_OP)),
)
def test_lerp_scalar_manifest_bench(
    input_shape: tuple[int, ...],
    end_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    test = LerpScalarWorkload(input_shape, end_shape, dtype)
    a, b = test.gen_inputs()
    op = LerpFwdOp(
        input=input_shape,
        end=end_shape,
        dtype=dtype,
        weight=_DEFAULT_WEIGHT,
    )
    bm = ManifestBenchmark(_LERP_SCALAR_OP, op, test)
    result = bm.profile(op, a, b)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
