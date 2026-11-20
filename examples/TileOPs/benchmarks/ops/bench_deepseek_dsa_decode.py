"""Benchmarks for DeepSeekSparseAttentionDecodeWithKVCacheFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/attention/bench_deepseek_dsa_decode.py``.
The GPU bench additionally profiles the handwritten PyTorch ``ref_program``
as a ``torch-ref`` baseline; that is an accuracy reference (kept in
``tests/ops/test_deepseek_dsa_decode.py``), not a performance baseline, so it
is omitted here.  The TileOps kernel measurement and ``BenchmarkReport``
recording are preserved.

``DeepSeekSparseAttentionDecodeWithKVCacheFwdOp`` is a 3-input op (q, kv,
indices), so the single-input ``workloads_to_params`` contract does not
apply; the GPU ``manifest_params`` / ``dsa_decode_args`` helpers are inlined
instead (same multi-input pattern as ``bench_mha_decode_paged.py``).

Adaptations (T3-T4):

- T3: the ``tune`` bench-run policy knob and its fixture param are removed
  (NPU uses heuristic config selection only).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.attention.deepseek_dsa_decode import DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
from tileops.workloads.attention import DsaDecodeWorkload

_DSA_DECODE_OP = "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp"


def _dsa_decode_args(workload: dict) -> tuple:
    """Extract ``(batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk,
    stride_kv, heads_kv, q_start_index_s, sm_scale)`` from a manifest workload.
    """
    batch, seq_len_q, heads, q_dim = workload["q_shape"]
    _, seq_len_kv, heads_kv, _ = workload["kv_shape"]
    dim_tail = workload["dim_tail"]
    dim = q_dim - dim_tail
    return (
        batch,
        heads,
        seq_len_q,
        seq_len_kv,
        dim,
        dim_tail,
        workload["topk"],
        workload["stride_kv"],
        heads_kv,
        workload["q_start_index_s"],
        workload.get("sm_scale"),
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
                    *_dsa_decode_args(workload), dtype, id=f"{label}-{dtype_name}", marks=marks
                )
            )
    return params


_DSA_DECODE_BENCH_PARAMS = _manifest_params(load_workloads(_DSA_DECODE_OP))


@pytest.mark.parametrize(
    "batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, "
    "q_start_index_s, sm_scale, dtype",
    _DSA_DECODE_BENCH_PARAMS,
)
def test_dsa_decode_bench(batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                          dim_tail: int, topk: int, stride_kv: int, heads_kv: int,
                          q_start_index_s: int, sm_scale: float, dtype: torch.dtype) -> None:
    test = DsaDecodeWorkload(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype)
    inputs = test.gen_inputs()

    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype)
    bm = ManifestBenchmark(_DSA_DECODE_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
