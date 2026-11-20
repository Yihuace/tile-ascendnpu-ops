"""Performance benchmarks for DropoutOp (NPU).

Ported from TileOPs ``benchmarks/ops/bench_dropout.py`` (single-operator file).
Profiles TileOPs dropout vs ``torch.nn.functional.dropout`` on DNN-realistic
shapes, driven by the manifest workloads in
``tileops/manifest/regularization.yaml`` and using ``p=0.5`` / ``training=true``
as the representative release-facing configuration.

Single-input note: ``DropoutOp`` has exactly one tensor input, so the
manifest-driven ``workloads_to_params`` / ``ManifestBenchmark`` path applies
directly.  ``include_extra=True`` carries the op params ``p`` / ``training``
from the manifest workloads.

The ``torch.nn.functional.dropout`` baseline is a native operator (a
performance baseline, not an accuracy-only golden), so it is kept.

NPU adaptations (T3/T4): the GPU bench has no ``tune`` parametrization and no
autotune try/except.  The workload's ``ref_program`` draws the NPU device from
the device backend (W1).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.regularization.dropout import DropoutOp
from tileops.workloads.regularization import DropoutOpWorkload

_OP_NAME = "DropoutOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_OP_NAME, include_extra=True),
)
def test_dropout_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = DropoutOpWorkload(shape, dtype, **op_params)
    (x,) = test.gen_inputs()

    op = DropoutOp(seed=42, **op_params)
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, x)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, x)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
