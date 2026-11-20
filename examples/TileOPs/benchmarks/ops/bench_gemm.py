"""Performance benchmarks for GemmOp (NPU).

Per-op benchmark file: the shared GPU ``bench_gemm.py`` was split so that only
this migration unit's workloads run when ``GemmOp`` is integrated (the shared
file also carried ``GemmW4A16Op`` workloads, which belong to a separate
migration unit, now in ``bench_gemm_w4a16.py``).  The shared GPU file also
benchmarks ``GemmFp8Op``, which is not migrated.

Benchmarks profile the TileOPs kernel against a native, NPU-runnable baseline.
The accuracy reference (``GemmTest.ref_program``) stays in ``tests/ops`` and is
not benchmarked here.

Adaptations (T3-T4):

- T3: the ``tune`` bench knob is removed (NPU uses heuristic config selection).

Multi-input note: the op has multiple tensor inputs, so ``workloads_to_params``
(the single-input manifest contract) does not apply.  A custom parametrization
helper reads ``load_workloads(...)`` instead, following the multi-input pattern
of ``bench_multi_head_attention.py`` / ``bench_ada_layer_norm.py``.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.gemm.gemm import GemmOp
from tileops.workloads.gemm import GemmWorkload

_OP_NAME = "GemmOp"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class GemmBenchmarkWorkload(GemmWorkload):
    def torch_matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)


def _manifest_params(workloads: list[dict]) -> list:
    """Convert manifest workloads to pytest params (m, n, k, trans_a, trans_b, dtype)."""
    params = []
    for w in workloads:
        label = w.get("label", "manifest")
        trans_a = bool(w.get("trans_a", False))
        trans_b = bool(w.get("trans_b", True))
        for dtype_name in w["dtypes"]:
            dtype = _DTYPE_MAP[dtype_name]
            params.append(
                pytest.param(
                    w["m"], w["n"], w["k"], trans_a, trans_b, dtype,
                    id=f"{label}-{dtype_name}",
                )
            )
    return params


_GEMM_BENCH_PARAMS = _manifest_params(load_workloads(_OP_NAME))


@pytest.mark.parametrize(
    "m, n, k, trans_a, trans_b, dtype", _GEMM_BENCH_PARAMS
)
def test_gemm_bench(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: torch.dtype,
) -> None:
    workload = GemmBenchmarkWorkload(m, n, k, dtype, trans_a, trans_b)
    a, b = workload.gen_inputs()

    op = GemmOp(trans_a=trans_a, trans_b=trans_b)
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # The benchmark framework warms up internally; eval_roofline() is read
    # lazily after profiling, by which point forward() has bound the dims.
    result = bm.profile(op, a, b)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(workload.torch_matmul, a, b)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-matmul")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
