"""Benchmark for AlibiFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_independent_elementwise.py`` —
the ``alibi`` subset only (the shared GPU file also covers the Tensor-bound
clamp ops; only this migrated op's benchmark is ported).

``AlibiFwdOp`` has no tensor input: the ``(num_heads, seq_len, seq_len)``
bias is synthesized from the construction-time parameters.  The single-input
``workloads_to_params`` contract therefore does not apply, so this file uses
the bench's own ``_generative_params`` manifest parametrization and the empty
``_GenerativeWorkload`` (ported from the GPU bench file).

T3/T4: the GPU ``tune`` parametrization / autotune try-except are dropped —
the NPU kernel uses heuristic config selection only.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.device import get_device_backend
from tileops.manifest import load_workloads
from tileops.ops.elementwise.alibi import AlibiFwdOp
from tileops.workloads.elementwise import _GenerativeWorkload
from tileops.workloads.workload_base import FixtureBase

_ALIBI_OP = "AlibiFwdOp"


def _device() -> str:
    return get_device_backend().name


def _generative_params(workloads: list, keys: tuple) -> list:
    """Manifest workloads -> params; first workload smoke, rest full."""
    params = []
    for i, w in enumerate(workloads):
        values = [w[k] for k in keys]
        dtype = getattr(torch, w["dtypes"][0])
        mark = pytest.mark.smoke if i == 0 else pytest.mark.full
        params.append(pytest.param(*values, dtype, marks=mark, id=w.get("label", f"w{i}")))
    return params


class AlibiBenchFixture(FixtureBase):
    PARAMS = [
        ("seq_len, num_heads, dtype",
         _generative_params(load_workloads(_ALIBI_OP), ("seq_len", "num_heads"))),
    ]


def _alibi_reference(seq_len: int, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """Full ALiBi bias: (num_heads, seq_len, seq_len), bias[h,i,j] = -slope_h * |i-j|."""
    positions = torch.arange(seq_len, device=_device(), dtype=torch.float32)
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # (S, S)
    slopes = torch.pow(
        2.0,
        -8.0 * torch.arange(1, num_heads + 1, device=_device(), dtype=torch.float32) / num_heads,
    )
    bias = -slopes[:, None, None] * dist[None, :, :]  # (H, S, S)
    return bias.to(dtype)


@AlibiBenchFixture
def test_alibi_bench(seq_len: int, num_heads: int, dtype: torch.dtype) -> None:
    op = AlibiFwdOp(seq_len=seq_len, num_heads=num_heads, dtype=dtype)
    workload = _GenerativeWorkload((num_heads, seq_len, seq_len), dtype)
    bm = ManifestBenchmark(_ALIBI_OP, op, workload)

    result = bm.profile(op)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(lambda: _alibi_reference(seq_len, num_heads, dtype))
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
