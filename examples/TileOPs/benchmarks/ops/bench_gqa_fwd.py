"""Benchmarks for GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp (NPU).

This op has TEN tensor inputs, so the ``workloads_to_params`` single-input contract
does not apply -- the manifest workloads are read directly, mirroring
``bench_multi_head_attention.py``.

The delivered NPU op is a DECOMPOSED (non-fused) implementation: three
single-engine kernels with the score matrix round-tripping through GM.  These
numbers are therefore NOT comparable with the GPU fused kernel; they are the honest
cost of the form this toolchain allows.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.attention.gqa_fwd import GQAPrefillPagedWithKVCacheFwdOp
from tileops.workloads.attention import GqaPagedWorkload

_GQA_OP = "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp"

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def _manifest_params(workloads: list[dict]) -> list:
    """Params from the manifest workloads (multi-input op: no shared contract)."""
    params = []
    for wl in workloads:
        label = wl.get("label", "manifest")
        skip = wl.get("bench_skip_reason")
        if skip:
            params.append(pytest.param(
                wl, torch.float16, id=f"{label}-skip",
                marks=pytest.mark.skip(reason=skip)))
            continue
        for dtype_name in wl["dtypes"]:
            params.append(pytest.param(
                wl, _DTYPES[dtype_name], id=f"{label}-{dtype_name}",
                marks=[pytest.mark.smoke] if "smoke" in label
                else [pytest.mark.full]))
    return params


@pytest.mark.parametrize("spec, dtype", _manifest_params(load_workloads(_GQA_OP)))
def test_gqa_fwd_bench(spec: dict, dtype: torch.dtype) -> None:
    page_size = int(spec["page_size"])
    is_causal = bool(spec.get("is_causal", True))

    wl = GqaPagedWorkload(spec, dtype)
    inputs = wl.gen_inputs()
    op = GQAPrefillPagedWithKVCacheFwdOp(page_size=page_size, is_causal=is_causal)

    bm = ManifestBenchmark(_GQA_OP, op, wl)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
