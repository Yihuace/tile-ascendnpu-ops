"""Manifest-driven benchmark for FP8QuantOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_fp8_quant.py`` -- the
``test_fp8_quant_bench`` TileOPs measurement only.  The GPU file also profiles
``FP8QuantTestBaseline.ref_program`` -- a handwritten PyTorch accuracy reference
for fp8 quantization (amax / 448 / clamp / cast).  Per the NPU benchmark rule
that accuracy-only golden/reference implementations are not benchmarked, its
baseline class and its ``bm.profile`` / ``BenchmarkReport.record`` calls are
dropped here; the reference stays in ``tests/ops/test_fp8_quant.py`` for
correctness.

Parametrization: FP8QuantOp has a single tensor input, but its manifest
signature declares scalar shape fields (``batch`` / ``seq_len_kv`` /
``kv_group`` / ``index_dim``) instead of a single ``input_tensor_shape`` key,
so ``workloads_to_params`` (whose single-input contract requires
``{input_name}_shape``) cannot be used.  The custom ``_to_params`` helper reads
the manifest workload dicts directly -- the same pattern as the GPU file's
``workload_field_params`` and the NPU ``bench_lerp_tensor.py`` /
``bench_ada_layer_norm.py`` multi-input helpers.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.quantization.fp8_quant import FP8QuantOp
from tileops.workloads.quantization import FP8QuantWorkload

_FP8_QUANT_OP = "FP8QuantOp"


def _to_params(workloads):
    params = []
    for i, w in enumerate(workloads):
        dtype = getattr(torch, w["in_dtype"])
        label = w.get("label") or "x".join(
            str(w[k]) for k in ("batch", "seq_len_kv", "kv_group", "index_dim")
        )
        params.append(
            pytest.param(
                w["batch"],
                w["seq_len_kv"],
                w["kv_group"],
                w["index_dim"],
                dtype,
                marks=pytest.mark.smoke if i == 0 else pytest.mark.full,
                id=label,
            )
        )
    return params


@pytest.mark.parametrize(
    "batch, seq_len_kv, kv_group, index_dim, in_dtype",
    _to_params(load_workloads(_FP8_QUANT_OP)),
)
def test_fp8_quant_bench(
    batch: int, seq_len_kv: int, kv_group: int, index_dim: int, in_dtype: torch.dtype
) -> None:
    test = FP8QuantWorkload(batch, seq_len_kv, kv_group, index_dim, in_dtype)
    inputs = test.gen_inputs()

    op = FP8QuantOp()
    bm = ManifestBenchmark(_FP8_QUANT_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
