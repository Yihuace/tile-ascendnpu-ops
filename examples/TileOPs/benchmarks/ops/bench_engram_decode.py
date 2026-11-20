"""Performance benchmarks for EngramDecodeOp (NPU).

Ported from TileOPs ``benchmarks/ops/bench_engram.py`` — the
``EngramDecodeOp`` subset only (the shared GPU file also covers
``EngramGateConvFwdOp`` / ``EngramGateConvBwdOp``, which are not migrated).

NPU adaptations:

- T3: the ``tune`` parametrization dimension is removed.
- The PyTorch reference (``engram_decode_step_torch``) is accuracy-only
  (it is the same golden used by the correctness tests), so it is *not*
  benchmarked here — only the TileOPs kernel is profiled and recorded.
- Multi-input note: ``EngramDecodeOp`` has 8 tensor inputs, so
  ``workloads_to_params`` cannot be used (its single-input workload contract
  raises ``KeyError`` for multi-input ops).  A custom parametrization helper
  (``_engram_decode_params``, ported from the GPU bench's manifest-driven
  parametrization) reads the manifest workloads directly.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.sequence_modeling.engram_decode import EngramDecodeOp
from tileops.workloads.sequence_modeling import EngramDecodeWorkload

_ENGRAM_DECODE_OP = "EngramDecodeOp"


def _engram_decode_params() -> list:
    """Turn the manifest workloads for EngramDecodeOp into pytest params.

    First workload is marked ``smoke``, the rest ``full`` (mirrors the GPU
    ``workload_field_params`` helper, which the NPU project does not ship).
    """
    params = []
    for i, w in enumerate(load_workloads(_ENGRAM_DECODE_OP)):
        args = (
            w["batch"],
            w["d_mem"],
            w["d"],
            w["max_conv_len"],
            w["conv_kernel_size"],
            w["dilation"],
            getattr(torch, w["dtype"]),
        )
        marks = pytest.mark.smoke if i == 0 else pytest.mark.full
        params.append(pytest.param(*args, marks=marks, id=w["label"]))
    return params


@pytest.mark.parametrize(
    "batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype",
    _engram_decode_params(),
)
def test_engram_decode_bench(
    batch: int,
    d_mem: int,
    d: int,
    max_conv_len: int,
    conv_kernel_size: int,
    dilation: int,
    dtype: torch.dtype,
) -> None:
    test = EngramDecodeWorkload(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    inputs = test.gen_inputs()

    # T3: no `tune` argument.
    op = EngramDecodeOp(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype,
    )
    bm = ManifestBenchmark(_ENGRAM_DECODE_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
