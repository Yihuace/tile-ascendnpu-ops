"""Manifest-driven benchmark for GatedDeltaNetPrefillFwdOp (NPU).

In-scope: B=1, equal-length GDN prefill, chunk=64. Workloads come from
``tileops/manifest/linear_attention.yaml``; roofline comes from
``GatedDeltaNetPrefillFwdOp.eval_roofline()``.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.linear_attention.gdn_prefill import GatedDeltaNetPrefillFwdOp
from tileops.workloads.linear_attention import GDNWorkload

_GDN_OP = "GatedDeltaNetPrefillFwdOp"


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float16 else pytest.mark.full


def _gdn_params(workloads):
    params = []
    for idx, w in enumerate(workloads):
        B, S, H, HV = w["B"], w["S"], w["H"], w["HV"]
        K, V = w["K"], w["V"]
        has_state = bool(w.get("has_state", True))
        scale = w.get("scale", -1.0)
        label = w.get("label", f"b{B}-s{S}-h{H}-k{K}")
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(
                    B, S, H, HV, K, V, scale, has_state, dtype,
                    id=f"{label}-{dtype_name}",
                    marks=_mark(idx, dtype),
                )
            )
    return params


@pytest.mark.parametrize(
    "B,S,H,HV,K,V,scale,has_state,dtype",
    _gdn_params(load_workloads(_GDN_OP)),
)
def test_gdn_manifest_bench(B, S, H, HV, K, V, scale, has_state, dtype) -> None:
    test = GDNWorkload(B, S, H, HV, K, V, dtype, scale=scale, has_state=has_state)
    q, k, v, g, beta, state = test.gen_inputs()
    op = GatedDeltaNetPrefillFwdOp(
        B=B, S=S, H=H, HV=HV, K=K, V=V, dtype=dtype, scale=scale
    )
    bm = ManifestBenchmark(_GDN_OP, op, test)
    result = bm.profile(op, q, k, v, g, beta, state)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
