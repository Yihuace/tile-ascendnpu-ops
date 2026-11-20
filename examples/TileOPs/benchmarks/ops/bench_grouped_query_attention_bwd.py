"""Benchmarks for GroupedQueryAttentionBwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/attention/bench_gqa.py`` -- the backward
subset only.  The GPU bench file is shared with the GQA forward / prefill /
varlen benchmarks (not migrated here), so only the backward bench function and
its workload import are ported.

The GPU backward bench profiles the TileOps op and then a baseline (FA3 when
installed, otherwise a torch SDPA backward).  FA3 is CUDA/Hopper-only and not
runnable on NPU; per the established NPU attention-bench convention (see
``bench_multi_head_attention.py`` / ``bench_deepseek_dsa_decode.py``) the torch
SDPA backward is kept as the correctness reference in ``tests/ops`` and only
the TileOps kernel is profiled here.

GroupedQueryAttentionBwdOp is a 6-input op (q, k, v, o, do, lse), so the
single-input ``workloads_to_params`` contract does not apply; the GPU
``manifest_params`` / ``gqa_qkv_args`` helpers are inlined instead -- the same
multi-input pattern as ``bench_multi_head_attention.py``.

Adaptations (T3-T4):

- T3: the ``tune`` bench-run policy knob and its fixture param are removed
  (NPU uses heuristic config selection only).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.attention.grouped_query_attention_bwd import GroupedQueryAttentionBwdOp
from tileops.workloads.attention import GroupedQueryAttentionBwdWorkload

_GQA_BWD_OP = "GroupedQueryAttentionBwdOp"


def _gqa_bwd_args(workload: dict) -> tuple:
    """Extract ``(batch, seq_len, heads, heads_kv, dim, is_causal)`` from a workload."""
    batch, seq_len, heads, dim = workload["q_shape"]
    _, _, heads_kv, _ = workload["kv_shape"]
    return batch, seq_len, heads, heads_kv, dim, workload.get("is_causal", True)


def _manifest_params(workloads: list[dict]) -> list:
    """Params from manifest workloads (GPU ``manifest_params`` inlined).

    T3: the GPU helper's trailing ``tune`` flag is dropped.
    """
    params = []
    for workload in workloads:
        label = workload.get("label", "manifest")
        marks = ()
        if reason := workload.get("bench_skip_reason"):
            marks = (pytest.mark.skip(reason=reason),)
        for dtype_name in workload["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(
                    *_gqa_bwd_args(workload), dtype, id=f"{label}-{dtype_name}", marks=marks
                )
            )
    return params


_GQA_BWD_BENCH_PARAMS = _manifest_params(load_workloads(_GQA_BWD_OP))


@pytest.mark.parametrize(
    "batch, seq_len, heads, heads_kv, dim, causal, dtype", _GQA_BWD_BENCH_PARAMS
)
def test_gqa_bwd_bench(
    batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool, dtype: torch.dtype
) -> None:
    test = GroupedQueryAttentionBwdWorkload(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    bm = ManifestBenchmark(_GQA_BWD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
