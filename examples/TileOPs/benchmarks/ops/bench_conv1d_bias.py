"""Manifest-driven benchmark for Conv1dBiasFwdOp (NPU).

The bias-required variant of ``Conv1dFwdOp``; this file mirrors
``bench_conv1d.py`` restricted to the always-bias contract.  Workload shapes,
op params and dtypes are loaded from the standalone manifest
(``tileops/manifest/convolution.yaml``); FLOP/byte counts come from
``Conv1dBiasFwdOp.eval_roofline()``.

Multi-input note: ``Conv1dBiasFwdOp`` has 3 tensor inputs (input, weight, bias),
so ``workloads_to_params`` cannot be used (its single-input workload contract
raises ``KeyError`` for multi-input ops).  A custom parametrization helper reads
the manifest workloads instead.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.convolution import Conv1dBiasFwdOp
from tileops.workloads.convolution import Conv1dWorkload

_CONV1D_BIAS_OP = "Conv1dBiasFwdOp"


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float16 else pytest.mark.full


def _conv1d_bias_params(workloads: list[dict]) -> list:
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
    _conv1d_bias_params(load_workloads(_CONV1D_BIAS_OP)),
)
def test_conv1d_bias_manifest_bench(
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
    assert bias is not None, "Conv1dBiasFwdOp workloads must carry a bias"
    op = Conv1dBiasFwdOp(stride=stride, padding=padding, dilation=dilation, groups=groups)
    bm = ManifestBenchmark(_CONV1D_BIAS_OP, op, test)
    result = bm.profile(op, x, weight, bias)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
