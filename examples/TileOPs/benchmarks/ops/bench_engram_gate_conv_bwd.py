"""Performance benchmarks for EngramGateConvBwdOp (NPU).

Ported from TileOPs ``benchmarks/ops/bench_engram.py`` — the
``EngramGateConvBwdOp`` subset only (the shared GPU file also covers
``EngramGateConvFwdOp`` / ``EngramDecodeOp``, which are separate NPU
migrations).

NPU adaptations:

- T3: the ``tune`` parametrization dimension is removed.
- The PyTorch autograd reference (``ref_engram_gate_conv_bwd``) is accuracy-only
  (it is the same golden used by the correctness tests), so it is *not*
  benchmarked here — only the TileOPs kernel is profiled and recorded.
- Multi-input note: ``EngramGateConvBwdOp`` has 12 tensor inputs, so
  ``workloads_to_params`` cannot be used (its single-input workload contract
  raises ``KeyError`` for multi-input ops).  A custom parametrization helper
  (``_engram_gate_conv_bwd_params``) reads the manifest workloads directly.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and byte
counts come from the op's ``eval_roofline()`` via :class:`ManifestBenchmark`.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.sequence_modeling.engram_gate_conv_bwd import EngramGateConvBwdOp
from tileops.workloads.sequence_modeling import EngramGateConvBwdWorkload

_ENGRAM_GATE_CONV_BWD_OP = "EngramGateConvBwdOp"


def _engram_gate_conv_bwd_params() -> list:
    """Turn the manifest workloads for EngramGateConvBwdOp into pytest params.

    First workload is marked ``smoke``, the rest ``full`` (mirrors the GPU
    ``workload_field_params`` helper, which the NPU project does not ship).
    """
    params = []
    for i, w in enumerate(load_workloads(_ENGRAM_GATE_CONV_BWD_OP)):
        args = (
            w["M"],
            w["seq_len"],
            w["d"],
            getattr(torch, w["dtype"]),
        )
        marks = pytest.mark.smoke if i == 0 else pytest.mark.full
        params.append(pytest.param(*args, marks=marks, id=w["label"]))
    return params


@pytest.mark.parametrize("M, seq_len, d, dtype", _engram_gate_conv_bwd_params())
def test_engram_gate_conv_bwd_bench(
    M: int,
    seq_len: int,
    d: int,
    dtype: torch.dtype,
) -> None:
    test = EngramGateConvBwdWorkload(M, seq_len, d, dtype)
    inputs = test.gen_inputs()

    # T3: no `tune` argument.
    op = EngramGateConvBwdOp(M, seq_len, d, dtype)
    bm = ManifestBenchmark(_ENGRAM_GATE_CONV_BWD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
