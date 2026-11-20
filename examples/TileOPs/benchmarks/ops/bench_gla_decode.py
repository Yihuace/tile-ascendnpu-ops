"""Manifest-driven benchmark for GLADecodeOp (NPU).

Single-token gated linear attention decode (baseline ``GLARecurrentFwdOp``).
Workload shapes / scale / dtypes are loaded from the standalone manifest
(``tileops/manifest/linear_attention.yaml``); FLOP/byte counts come from
``GLADecodeOp.eval_roofline()``.

Multi-input note: ``GLADecodeOp`` has 5 tensor inputs (q, k, v, gk, state) and
op params fixed at construction, so a custom parametrization helper reads the
manifest workloads instead of ``workloads_to_params``.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.linear_attention.gla_decode import GLADecodeOp
from tileops.workloads.linear_attention import GLADecodeWorkload

_GLA_OP = "GLADecodeOp"


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float16 else pytest.mark.full


def _gla_params(workloads: list[dict]) -> list:
    params = []
    for idx, w in enumerate(workloads):
        B, H, DK, DV = w["B"], w["H"], w["DK"], w["DV"]
        scale = w.get("scale", -1.0)
        label = w.get("label", f"b{B}-h{H}-d{DK}x{DV}")
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(
                    B,
                    H,
                    DK,
                    DV,
                    scale,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=_mark(idx, dtype),
                )
            )
    return params


@pytest.mark.parametrize("B, H, DK, DV, scale, dtype", _gla_params(load_workloads(_GLA_OP)))
def test_gla_decode_manifest_bench(B, H, DK, DV, scale, dtype) -> None:
    test = GLADecodeWorkload(B, H, DK, DV, dtype, scale)
    q, k, v, gk, state = test.gen_inputs()
    op = GLADecodeOp(B, H, DK, DV, scale=scale, dtype=dtype)
    bm = ManifestBenchmark(_GLA_OP, op, test)
    result = bm.profile(op, q, k, v, gk, state)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
