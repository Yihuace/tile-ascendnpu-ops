"""Manifest-driven benchmark for AddFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise.py`` -- the Add subset
only (the shared GPU file also times sub / mul / div / ...; only add is
migrated).  Workloads and roofline formulas are loaded from the standalone
manifest.

Multi-input note: ``AddFwdOp`` has 2 tensor inputs, so ``workloads_to_params``
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
from tileops.ops.elementwise.add import AddFwdOp
from tileops.workloads.elementwise import AddWorkload

_ADD_OP = "AddFwdOp"


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
    _shape_dtype_params(load_workloads(_ADD_OP)),
)
def test_add_manifest_bench(
    input_shape: tuple[int, ...], other_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    test = AddWorkload(input_shape, other_shape, dtype)
    a, b = test.gen_inputs()
    op = AddFwdOp(input=input_shape, other=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_ADD_OP, op, test)
    result = bm.profile(op, a, b)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
