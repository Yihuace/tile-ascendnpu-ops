"""Manifest-driven benchmark for Conv3dBiasFwdOp (NPU).

The bias-required 3-D convolution; this file mirrors ``bench_conv1d_bias.py``
restricted to the always-bias contract.  Workload shapes, op params and dtypes
are loaded from the standalone manifest
(``tileops/manifest/convolution.yaml``); FLOP/byte counts come from
``Conv3dBiasFwdOp.eval_roofline()``.

Multi-input note: ``Conv3dBiasFwdOp`` has 3 tensor inputs (input, weight, bias),
so ``workloads_to_params`` cannot be used (its single-input workload contract
raises ``KeyError`` for multi-input ops).  A custom parametrization helper reads
the manifest workloads instead.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.convolution import Conv3dBiasFwdOp
from tileops.workloads.convolution import Conv3dWorkload

_CONV3D_BIAS_OP = "Conv3dBiasFwdOp"


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float16 else pytest.mark.full


def _conv3d_bias_params(workloads: list[dict]) -> list:
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
    _conv3d_bias_params(load_workloads(_CONV3D_BIAS_OP)),
)
def test_conv3d_bias_manifest_bench(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    n, c_in, d_in, h_in, w_in = input_shape
    c_out, c_in_g, kd, kh, kw = weight_shape
    assert kd == kh == kw, "bench helper assumes a cubic 3-D kernel"
    test = Conv3dWorkload(
        n, c_in, d_in, h_in, w_in, c_out, kd, stride, padding, dilation, groups, dtype, use_bias
    )
    x, weight, bias = test.gen_inputs()
    assert bias is not None, "Conv3dBiasFwdOp workloads must carry a bias"
    op = Conv3dBiasFwdOp(stride=stride, padding=padding, dilation=dilation, groups=groups)
    bm = ManifestBenchmark(_CONV3D_BIAS_OP, op, test)
    result = bm.profile(op, x, weight, bias)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
