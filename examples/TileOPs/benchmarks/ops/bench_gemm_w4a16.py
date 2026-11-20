"""Performance benchmarks for GemmW4A16Op (NPU).

Per-op benchmark file: the shared GPU ``bench_gemm.py`` was split so that only
this migration unit's workloads run when ``GemmW4A16Op`` is integrated (the
shared file also carried ``GemmOp`` workloads, which belong to a separate
migration unit, now in ``bench_gemm.py``).

Benchmarks profile the TileOPs kernel against a native, NPU-runnable baseline.
The accuracy reference (``GemmW4A16Test.ref_program``) stays in ``tests/ops``
and is not benchmarked here.

Adaptations (T3-T4):

- T3: the ``tune`` bench knob is removed (NPU uses heuristic config selection).
- The GPU-only Marlin / vLLM W4A16 baseline is dropped -- it is a CUDA
  library baseline with no NPU equivalent; the runnable
  ``torch-dequantized-matmul`` baseline is kept.

Multi-input note: the op has multiple tensor inputs, so ``workloads_to_params``
(the single-input manifest contract) does not apply.  A custom parametrization
helper reads ``load_workloads(...)`` instead, following the multi-input pattern
of ``bench_multi_head_attention.py`` / ``bench_ada_layer_norm.py``.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.gemm.gemm_w4a16 import GemmW4A16Op
from tileops.workloads.gemm import GemmW4A16Workload

_W4A16_OP_NAME = "GemmW4A16Op"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class GemmW4A16BenchmarkWorkload(GemmW4A16Workload):
    def torch_dequantized_matmul(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        del packed_weight, weight_scale, weight_zero
        return torch.matmul(activation, self.dequantized_weight.T)


def _manifest_w4a16_params(workloads: list[dict]) -> list:
    """Convert W4A16 manifest workloads to pytest params (m, n, k, group_size, dtype)."""
    params = []
    for w in workloads:
        label = w.get("label", "manifest")
        group_size = int(w.get("group_size", 128))
        for dtype_name in w["dtypes"]:
            dtype = _DTYPE_MAP[dtype_name]
            params.append(
                pytest.param(
                    w["m"], w["n"], w["k"], group_size, dtype,
                    id=f"{label}-{dtype_name}",
                )
            )
    return params


_W4A16_BENCH_PARAMS = _manifest_w4a16_params(load_workloads(_W4A16_OP_NAME))


@pytest.mark.parametrize(
    "m, n, k, group_size, dtype", _W4A16_BENCH_PARAMS
)
def test_gemm_w4a16_bench(
    m: int,
    n: int,
    k: int,
    group_size: int,
    dtype: torch.dtype,
) -> None:
    workload = GemmW4A16BenchmarkWorkload(m, n, k, dtype, group_size=group_size)
    inputs = workload.gen_inputs()

    op = GemmW4A16Op(group_size=group_size)
    bm = ManifestBenchmark(_W4A16_OP_NAME, op, workload)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # NPU-runnable reference baseline (dequantize then matmul in torch).
    result_bl = bm.profile(workload.torch_dequantized_matmul, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-dequantized-matmul")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
