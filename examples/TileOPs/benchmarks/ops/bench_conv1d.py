"""Manifest-driven benchmark for Conv1dFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_convolution.py`` -- the Conv1d
subset only (the shared GPU file also times Conv2d / Conv3d, which are not
migrated).  Workload shapes, op params and dtypes are loaded from the
standalone manifest (``tileops/manifest/convolution.yaml``); FLOP/byte counts
come from ``Conv1dFwdOp.eval_roofline()``.

NPU adaptations (T3): the ``tune`` / ``op.request_tune()`` step is removed; the
flag_gems Triton and torch.compile baselines are not available on NPU, so the
benchmark records the TileOPs kernel against the PyTorch reference only.

Multi-input note: ``Conv1dFwdOp`` has up to 3 tensor inputs (bias optional), so
``workloads_to_params`` cannot be used (its single-input workload contract
raises ``KeyError`` for multi-input ops).  A custom parametrization helper
reads the manifest workloads instead -- see the multi-input pattern in the
add-npu-op skill (S6).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.convolution import Conv1dFwdOp
from tileops.workloads.convolution import Conv1dWorkload

_CONV1D_OP = "Conv1dFwdOp"


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float16 else pytest.mark.full


def _conv1d_params(workloads: list[dict]) -> list:
    """Parametrize the manifest workloads (multi-input + op params + bias)."""
    params = []
    for idx, w in enumerate(workloads):
        input_shape = tuple(w["input_shape"])
        weight_shape = tuple(w["weight_shape"])
        stride = w.get("stride", 1)
        padding = w.get("padding", 0)
        dilation = w.get("dilation", 1)
        groups = w.get("groups", 1)
        use_bias = bool(w.get("bias", False))
        label = w.get("label", "x".join(str(dim) for dim in input_shape))
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(
                    input_shape,
                    weight_shape,
                    stride,
                    padding,
                    dilation,
                    groups,
                    use_bias,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=_mark(idx, dtype),
                )
            )
    return params


@pytest.mark.parametrize(
    "input_shape, weight_shape, stride, padding, dilation, groups, use_bias, dtype",
    _conv1d_params(load_workloads(_CONV1D_OP)),
)
def test_conv1d_manifest_bench(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: int,
    padding: int | str,
    dilation: int,
    groups: int,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    n, c_in, l_in = input_shape
    c_out, c_in_g, kernel_size = weight_shape
    test = Conv1dWorkload(
        n,
        c_in,
        l_in,
        c_out,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        dtype,
        use_bias,
    )
    x, weight, bias = test.gen_inputs()
    op = Conv1dFwdOp(stride=stride, padding=padding, dilation=dilation, groups=groups)
    bm = ManifestBenchmark(_CONV1D_OP, op, test)
    args = (x, weight) if bias is None else (x, weight, bias)
    result = bm.profile(op, *args)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
