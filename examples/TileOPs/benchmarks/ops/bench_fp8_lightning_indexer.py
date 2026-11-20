"""Benchmarks for FP8LightningIndexerOp (NPU).

Ported from TileOPs ``benchmarks/ops/bench_fp8_lightning_indexer.py``.  The GPU
bench additionally profiles a PyTorch ``ref_program`` accuracy reference under
the ``torch-ref`` tag; per the add-npu-op S6 rule that is an accuracy-only
golden implementation (kept in ``tests/ops/test_fp8_lightning_indexer.py``), not
a performance baseline, so it is dropped from the NPU benchmark entirely.  The
TileOps kernel measurement and ``BenchmarkReport`` recording are preserved.

``FP8LightningIndexerOp`` is a multi-input op (index_q, index_k, weights,
cu_seqlen_ks, cu_seqlen_ke [+ optional index_k_scale]), so the single-input
``workloads_to_params`` contract does not apply; the manifest is read directly
with ``load_workloads`` (the GPU ``_indexer_params`` helper inlined, same
multi-input pattern as ``bench_mha_decode_paged.py``).

Adaptations (T3-T4):

- T3: the ``tune`` bench-run policy knob (and its ``_TUNE`` constant) is removed
  (NPU uses heuristic config selection only).
"""

import pytest

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.attention_indexing.fp8_lightning_indexer import FP8LightningIndexerOp
from tileops.workloads.attention_indexing import FP8LightningIndexerWorkload

_FP8_LIGHTNING_INDEXER_OP = "FP8LightningIndexerOp"

_SHAPE_KEYS = (
    "batch", "seq_len", "heads", "index_dim", "seq_len_kv", "kv_group", "clean_logits",
)


def _indexer_params() -> list:
    """Params from manifest workloads, deduped on shape.

    ``FP8LightningIndexerWorkload.gen_inputs`` emits bf16 and quantizes inside
    the op, so workloads differing only in ``dtypes`` are one measurement.
    """
    seen, params = set(), []
    for w in load_workloads(_FP8_LIGHTNING_INDEXER_OP):
        args = tuple(w[k] for k in _SHAPE_KEYS)
        if args in seen:
            continue
        seen.add(args)
        params.append(pytest.param(
            *args, id=w["label"],
            marks=pytest.mark.smoke if not params else pytest.mark.full))
    return params


@pytest.mark.parametrize(
    "batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits",
    _indexer_params(),
)
def test_fp8_lightning_indexer_bench(batch: int, seq_len: int, heads: int, index_dim: int,
                                     seq_len_kv: int, kv_group: int,
                                     clean_logits: bool) -> None:
    test = FP8LightningIndexerWorkload(batch, seq_len, heads, index_dim, seq_len_kv, kv_group,
                                       clean_logits)
    inputs = test.gen_inputs()

    op = FP8LightningIndexerOp(clean_logits=clean_logits)
    bm = ManifestBenchmark(_FP8_LIGHTNING_INDEXER_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
