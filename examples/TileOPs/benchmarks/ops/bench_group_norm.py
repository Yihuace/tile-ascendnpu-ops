"""Manifest-driven benchmark for GroupNormFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_group_norm.py`` --
``test_group_norm_bench`` (the affine primary) only.  The shared GPU file also
benchmarks ``GroupNormNoAffineFwdOp`` (``test_group_norm_no_affine_bench``),
which is not migrated; its param builder, imports, and bench function are
dropped.

Multi-input note: ``GroupNormFwdOp`` has 3 tensor inputs (x / weight / bias),
so ``workloads_to_params`` cannot be used (its single-input workload contract
raises ``KeyError`` for multi-input ops).  The GPU ``_build_params`` helper --
which reads the manifest ``x_shape`` and ``num_groups`` -- is the custom
parametrization helper (same pattern as ``bench_lerp_tensor.py``).

T3: the ``tune`` parametrization column is removed (K8) -- the GPU helper
carried an always-``False`` ``tune`` column and the op no longer accepts it.
The ``torch.nn.functional.group_norm`` baseline is kept: it is a native
operator (a meaningful performance baseline), not an accuracy-only reference.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.norm.group_norm import GroupNormFwdOp
from tileops.workloads.norm import GroupNormWorkload

_OP_NAME = "GroupNormFwdOp"


def _build_params(workloads):
    params = []
    for w in workloads:
        shape = w["x_shape"]
        n, c, spatial = shape[0], shape[1], tuple(shape[2:])
        num_groups = w.get("num_groups")
        if num_groups is None:
            raise KeyError("Workload manifest must contain 'num_groups'")
        label = w.get("label", f"{n}x{c}x{'x'.join(map(str, spatial))}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(n, c, spatial, num_groups, dtype, id=f"{label}-{dtype_str}"))
    return params


_AFFINE_PARAMS = _build_params(load_workloads(_OP_NAME))


@pytest.mark.parametrize("n, c, spatial, num_groups, dtype", _AFFINE_PARAMS)
def test_group_norm_bench(
    n: int, c: int, spatial: tuple, num_groups: int, dtype: torch.dtype
) -> None:
    test = GroupNormWorkload(n, c, spatial, num_groups, dtype)
    x, weight, bias = test.gen_inputs()

    op = GroupNormFwdOp(num_groups=num_groups)
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, x, weight, bias)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: torch.nn.functional.group_norm
    def baseline_fn(x, weight, bias):
        return F.group_norm(x, num_groups, weight=weight, bias=bias, eps=1e-5)

    result_bl = bm.profile(baseline_fn, x, weight, bias)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
