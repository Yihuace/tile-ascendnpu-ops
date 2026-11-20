"""Benchmark for ClampFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_independent_elementwise.py`` —
the Tensor-bound clamp subset only (the shared GPU file also covers the
generative ALiBi op; only this migrated op's benchmark is ported).

``N_total`` is post-broadcast, i.e. ``product(out_shape)``.  Workload shapes,
dtypes and the roofline formula are loaded from the standalone manifest
(``tileops/manifest/elementwise.yaml``).  The native ``torch.clamp`` is
retained as an independent performance baseline (it is a native operator,
not an accuracy-only reference).  T3/T4: the GPU ``tune`` parametrization and
the "No configurations to tune" try/except are dropped — the NPU kernel uses
heuristic config selection only.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.elementwise.clamp import ClampFwdOp
from tileops.workloads.elementwise import TensorClampBenchCase

_CLAMP_FWD_OP = "ClampFwdOp"


def _workloads_to_clamp_params(workloads: list) -> list:
    """Convert manifest workload dicts to clamp-bench pytest params."""
    params = []
    for idx, w in enumerate(workloads):
        input_shape = tuple(w["input_shape"])
        min_shape = tuple(w["min_shape"])
        max_shape = tuple(w["max_shape"])
        label = w.get("label", "x".join(str(s) for s in input_shape))
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            # Smoke = first workload + fp16; everything else is full-mode.
            mark = (
                pytest.mark.smoke
                if (idx == 0 and dtype is torch.float16)
                else pytest.mark.full
            )
            params.append(
                pytest.param(
                    input_shape, min_shape, max_shape, dtype,
                    marks=mark, id=f"{label}-{dtype_str}",
                )
            )
    return params


@pytest.mark.parametrize(
    "input_shape, min_shape, max_shape, dtype",
    _workloads_to_clamp_params(load_workloads(_CLAMP_FWD_OP)),
)
def test_clamp_tensor_bench(
    input_shape: tuple,
    min_shape: tuple,
    max_shape: tuple,
    dtype: torch.dtype,
) -> None:
    test = TensorClampBenchCase(input_shape, dtype, min_shape=min_shape, max_shape=max_shape)
    x, t_min, t_max = test.gen_inputs()

    op = ClampFwdOp(input=input_shape, min=min_shape, max=max_shape, dtype=dtype)
    bm = ManifestBenchmark(_CLAMP_FWD_OP, op, test)
    result = bm.profile(op, x, t_min, t_max)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x, t_min, t_max):
        return torch.clamp(x, t_min, t_max)

    result_bl = bm.profile(baseline_fn, x, t_min, t_max)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
