"""Benchmarks for MultiHeadAttentionDecodePagedWithKVCacheFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/attention/bench_mha_decode_paged.py``.
The GPU bench additionally profiles CUDA-only baselines (FA3
``flash_attn_with_kvcache`` and FlashInfer ``BatchDecodeWithPagedKVCacheWrapper``)
and, when neither is importable, the PyTorch ``ref_program`` accuracy reference.
On NPU those are dropped: FA3 / FlashInfer are CUDA-only and not runnable, and
the PyTorch ``ref_program`` is an accuracy reference (kept in
``tests/ops/test_mha_decode_paged.py``), not a performance baseline.  The
TileOps kernel measurement and ``BenchmarkReport`` recording are preserved.

``MultiHeadAttentionDecodePagedWithKVCacheFwdOp`` is a 5-input op (q, k, v,
real_seqlen_kv, block_table), so the single-input ``workloads_to_params``
contract does not apply; the GPU ``manifest_params`` / ``mha_decode_paged_args``
helpers are inlined instead (same multi-input pattern as
``bench_multi_head_attention.py``).

Adaptations (T3-T4):

- T3: the ``tune`` bench-run policy knob and its fixture param are removed
  (NPU uses heuristic config selection only).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.attention.mha_decode_paged import MultiHeadAttentionDecodePagedWithKVCacheFwdOp
from tileops.workloads.attention import MhaDecodePagedWorkload

_MHA_DECODE_PAGED_OP = "MultiHeadAttentionDecodePagedWithKVCacheFwdOp"


def _mha_decode_paged_args(workload: dict) -> tuple[int, int, int, int, int, int, bool]:
    """Extract ``(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal)``."""
    batch, seqlen_q, heads, dim = workload["q_shape"]
    seqlen_kv, _, _ = workload["kv_shape"]
    return (
        batch,
        heads,
        seqlen_q,
        seqlen_kv,
        dim,
        workload["page_size"],
        workload.get("is_causal", False),
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
                    *_mha_decode_paged_args(workload), dtype, id=f"{label}-{dtype_name}", marks=marks
                )
            )
    return params


_MHA_DECODE_PAGED_BENCH_PARAMS = _manifest_params(load_workloads(_MHA_DECODE_PAGED_OP))


@pytest.mark.parametrize(
    "batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype",
    _MHA_DECODE_PAGED_BENCH_PARAMS,
)
def test_mha_decode_paged_bench(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: torch.dtype,
) -> None:
    test = MhaDecodePagedWorkload(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
    bm = ManifestBenchmark(_MHA_DECODE_PAGED_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
