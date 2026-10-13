"""Manifest-driven benchmark for SubFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise.py`` -- the Sub subset
only (the shared GPU file also times add / mul / div / ...; only sub is
migrated).  Workloads and roofline formulas are loaded from the standalone
manifest.

Multi-input note: ``SubFwdOp`` has 2 tensor inputs, so ``workloads_to_params``
cannot be used (its single-input workload contract raises ``KeyError`` for
multi-input ops).  A custom parametrization helper reads the manifest workloads
(``input_shape`` / ``other_shape``) instead -- see the multi-input pattern in
the add-npu-op skill (S6).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
)
from tileops.manifest import load_workloads
from tileops.ops.elementwise.sub import SubFwdOp
from tileops.workloads.elementwise import SubWorkload

_SUB_OP = "SubFwdOp"


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float16 else pytest.mark.full


def _shape_dtype_params(workloads: list[dict]) -> list:
    """Parametrize (input_shape, other_shape, dtype) from manifest workloads."""
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
    _shape_dtype_params(load_workloads(_SUB_OP)),
)
def test_sub_manifest_bench(
    input_shape: tuple[int, ...], other_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    test = SubWorkload(input_shape, other_shape, dtype)
    a, b = test.gen_inputs()
    op = SubFwdOp(input=input_shape, other=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_SUB_OP, op, test)
    result = bm.profile(op, a, b)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
