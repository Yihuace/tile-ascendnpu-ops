"""Benchmarks for GroupedQueryAttentionSlidingWindowVarlenFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/attention/bench_gqa_sliding_window_varlen.py``.
Performance benchmarks profile only the TileOps kernel; PyTorch remains the
correctness reference in ``tests/ops``.  The GPU-only baselines (FA3
``flash_attn_varlen_func``, FlashInfer ``BatchPrefillWithRaggedKVCacheWrapper``)
are CUDA kernels that are not runnable on NPU, and the explicit-mask ``torch``
SDPA helper is a correctness-style reference, so none of them are profiled here.

GroupedQueryAttentionSlidingWindowVarlenFwdOp is a multi-input op (5 tensor
inputs: q, k, v, cu_seqlens_q, cu_seqlens_k), so the single-input
``workloads_to_params`` contract does not apply; the GPU ``manifest_params`` /
``gqa_sliding_window_varlen_args`` helpers (from
``benchmarks/ops/attention/manifest_params.py``) are inlined instead -- the same
multi-input pattern as ``bench_multi_head_attention.py``.

Mixed dtypes: q/k/v carry the workload ``dtype`` (fp16/bf16) while the
``cu_seqlens_*`` arrays are always ``int32``; the workload's ``gen_inputs`` owns
that split, so the benchmark parametrization only carries the tensor ``dtype``.

Adaptations (T3-T4):

- T3: the ``tune`` bench-run policy knob is removed (NPU uses heuristic config
  selection only).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.attention.grouped_query_attention_sliding_window_varlen import (
    GroupedQueryAttentionSlidingWindowVarlenFwdOp,
)
from tileops.workloads.attention import GroupedQueryAttentionSlidingWindowVarlenFwdWorkload

_OP_NAME = "GroupedQueryAttentionSlidingWindowVarlenFwdOp"


def _gqa_sw_varlen_args(
    workload: dict,
) -> tuple[int, list[int], list[int], int, int, int, bool, int, int]:
    """Extract ``(batch, seqlens_q, seqlens_k, heads, heads_kv, dim, is_causal, wl, wr)``.

    GPU ``gqa_sliding_window_varlen_args`` inlined: when the workload does not
    carry explicit ``q_lens`` / ``k_lens``, split ``total_q`` / ``total_k``
    evenly across the batch.
    """
    batch = workload["batch"]
    q_lens = list(workload.get("q_lens") or [workload["total_q"] // batch] * batch)
    k_lens = list(workload.get("k_lens") or [workload["total_k"] // batch] * batch)
    return (
        batch,
        q_lens,
        k_lens,
        workload["heads"],
        workload["heads_kv"],
        workload["dim"],
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
                    *_gqa_sw_varlen_args(workload), dtype, id=f"{label}-{dtype_name}", marks=marks
                )
            )
    return params


_GQA_SW_VARLEN_FWD_BENCH_PARAMS = _manifest_params(load_workloads(_OP_NAME))


@pytest.mark.parametrize(
    "batch, seqlens_q, seqlens_k, heads, heads_kv, dim, is_causal, wl, wr, dtype",
    _GQA_SW_VARLEN_FWD_BENCH_PARAMS,
)
def test_gqa_sliding_window_varlen_fwd_bench(
    batch: int,
    seqlens_q: list[int],
    seqlens_k: list[int],
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    wl: int,
    wr: int,
    dtype: torch.dtype,
) -> None:
    test = GroupedQueryAttentionSlidingWindowVarlenFwdWorkload(
        batch, seqlens_q, seqlens_k, heads, heads_kv, dim, is_causal, wl, wr, dtype
    )
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        is_causal=is_causal,
        window_size_left=wl,
        window_size_right=wr,
        dtype=dtype,
    )
    # Per-workload roofline inputs (varlen totals are unknown at op construction;
    # resolved per call, mirroring the GPU benchmark).
    op.total_q = sum(seqlens_q)
    op.total_k = sum(seqlens_k)
    op.q_lens = seqlens_q
    op.k_lens = seqlens_k
    op.max_seqlen_q = max(seqlens_q)
    op.max_seqlen_k = max(seqlens_k)

    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
