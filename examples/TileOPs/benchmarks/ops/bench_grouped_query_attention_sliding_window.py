"""Benchmarks for GroupedQueryAttentionSlidingWindowFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/attention/bench_gqa_sliding_window.py``.
Performance benchmarks profile only the TileOps kernel; PyTorch remains the
correctness reference in ``tests/ops``.  The GPU-only baselines (FA3
``flash_attn_func``, FlashInfer ``BatchPrefillWithRaggedKVCacheWrapper``) are
CUDA kernels that are not runnable on NPU, and the explicit-mask ``torch`` SDPA
helper is a correctness-style reference, so none of them are profiled here.

GroupedQueryAttentionSlidingWindowFwdOp is a 3-input op (q, k, v), so the
single-input ``workloads_to_params`` contract does not apply; the GPU
``manifest_params`` / ``gqa_sliding_window_args`` helpers (from
``benchmarks/ops/attention/manifest_params.py``) are inlined instead -- the same
multi-input pattern as ``bench_multi_head_attention.py``.

Adaptations (T3-T4):

- T3: the ``tune`` bench-run policy knob is removed (NPU uses heuristic config
  selection only).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.attention.grouped_query_attention_sliding_window import (
    GroupedQueryAttentionSlidingWindowFwdOp,
)
from tileops.workloads.attention import GroupedQueryAttentionSlidingWindowFwdWorkload

_OP_NAME = "GroupedQueryAttentionSlidingWindowFwdOp"


def _gqa_sw_args(workload: dict) -> tuple[int, int, int, int, int, bool, int, int]:
    """Extract ``(batch, seq, heads, heads_kv, dim, is_causal, wl, wr)`` from a workload."""
    batch, seq, heads, dim = workload["q_shape"]
    _, _, heads_kv, _ = workload["kv_shape"]
    return (
        batch,
        seq,
        heads,
        heads_kv,
        dim,
        workload.get("is_causal", True),
        workload.get("window_size_left", -1),
        workload.get("window_size_right", -1),
    )


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
                    *_gqa_sw_args(workload), dtype, id=f"{label}-{dtype_name}", marks=marks
                )
            )
    return params


_GQA_SW_FWD_BENCH_PARAMS = _manifest_params(load_workloads(_OP_NAME))


@pytest.mark.parametrize(
    "batch, seq, heads, heads_kv, dim, is_causal, wl, wr, dtype", _GQA_SW_FWD_BENCH_PARAMS
)
def test_gqa_sliding_window_fwd_bench(
    batch: int,
    seq: int,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    wl: int,
    wr: int,
    dtype: torch.dtype,
) -> None:
    test = GroupedQueryAttentionSlidingWindowFwdWorkload(
        batch, seq, heads, heads_kv, dim, is_causal, wl, wr, dtype
    )
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionSlidingWindowFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seq_len=seq,
        dim=dim,
        is_causal=is_causal,
        window_size_left=wl,
        window_size_right=wr,
        dtype=dtype,
    )
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
