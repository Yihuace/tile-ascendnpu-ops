"""Manifest-driven benchmark for GeluTanhAndMulFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_fused_gated.py`` -- the
GeluTanhAndMul subset only (the shared GPU file also times silu_and_mul and
gelu_and_mul, which are not part of this port).  Workloads and the roofline
formula are loaded from the standalone manifest.

Fused-gated note: ``GeluTanhAndMulFwdOp`` takes one ``(M, 2 * N)`` tensor input
but parametrizes on the two manifest dimensions ``M`` / ``N`` (not a single
``*_shape`` key), so ``workloads_to_params`` cannot be used.  A custom
parametrization helper reads the manifest workloads instead -- same pattern as
the multi-input benches.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
)
from tileops.manifest import load_workloads
from tileops.ops.elementwise.gelu_tanh_and_mul import GeluTanhAndMulFwdOp
from tileops.workloads.elementwise import GeluTanhAndMulWorkload

_GELU_TANH_AND_MUL_OP = "GeluTanhAndMulFwdOp"


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float16 else pytest.mark.full


def _mn_dtype_params(workloads: list[dict]) -> list:
    """Parametrize (M, N, dtype) from the manifest workloads."""
    params = []
    for idx, w in enumerate(workloads):
        M = int(w["M"])
        N = int(w["N"])
        label = w.get("label", f"{M}x{N}")
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(
                    M,
                    N,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=_mark(idx, dtype),
                )
            )
    return params


@pytest.mark.parametrize(
    "M, N, dtype",
    _mn_dtype_params(load_workloads(_GELU_TANH_AND_MUL_OP)),
)
def test_gelu_tanh_and_mul_manifest_bench(M: int, N: int, dtype: torch.dtype) -> None:
    test = GeluTanhAndMulWorkload(M, N, dtype)
    (x,) = test.gen_inputs()
    op = GeluTanhAndMulFwdOp(M=M, N=N, dtype=dtype)
    bm = ManifestBenchmark(_GELU_TANH_AND_MUL_OP, op, test)
    result = bm.profile(op, x)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
