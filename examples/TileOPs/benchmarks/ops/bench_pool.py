"""Manifest-driven benchmark for the adaptive 2D pool ops (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_pool.py`` --
``test_adaptive_avg_pool2d_bench`` and ``test_adaptive_max_pool2d_bench``.  The
shared GPU file also benchmarks the avg/max pool 1d/2d/3d ops and the adaptive
max-pool indices variant, which are not migrated; their imports, parametrization
helpers and bench functions are dropped.

Single-input note: both ops have exactly one tensor input, so the
manifest-driven ``workloads_to_params(..., include_extra=True)`` parametrization
is used -- it validates the ``single_input_workload_contract`` and forwards the
``output_size`` workload parameter alongside the input shape.

Benchmark-policy note: the GPU file also profiles ``test.ref_program``
(``F.adaptive_avg_pool2d`` / ``F.adaptive_max_pool2d``) under the ``torch-ref``
tag.  Those references exist solely for correctness comparison and stay in the
correctness suite (``tests/ops/test_pool.py``); per the NPU benchmark policy
only the TileOPs op is profiled here.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark, workloads_to_params
from tileops.ops.pool.adaptive_avg_pool2d import AdaptiveAvgPool2dFwdOp
from tileops.ops.pool.adaptive_max_pool2d import AdaptiveMaxPool2dFwdOp
from tileops.ops.pool.adaptive_max_pool2d_with_indices import AdaptiveMaxPool2dIndicesFwdOp
from tileops.workloads.pool import AdaptivePool2dWorkload
from tileops.ops.pool.max_pool1d_with_indices import MaxPool1dIndicesFwdOp
from tileops.ops.pool.max_pool2d import MaxPool2dFwdOp
from tileops.workloads.pool import MaxPool1dWorkload
from tileops.workloads.pool import MaxPool2dWorkload

_MAX_POOL2D_OP_NAME = "MaxPool2dFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_MAX_POOL2D_OP_NAME, include_extra=True),
)
def test_max_pool2d_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    n, c_in, h_in, w_in = shape
    kernel_size = tuple(op_params["kernel_size"])
    stride = op_params.get("stride")
    if stride is not None:
        stride = tuple(stride)
    padding = tuple(op_params.get("padding", (0, 0)))
    dilation = tuple(op_params.get("dilation", (1, 1)))
    ceil_mode = op_params.get("ceil_mode", False)

    test = MaxPool2dWorkload(
        n, c_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype
    )
    inputs = test.gen_inputs()

    op = MaxPool2dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    bm = ManifestBenchmark(_MAX_POOL2D_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


def _max_pool1d_as_tuple(value: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalize a scalar-or-tuple 1D pooling param to a length-1 tuple.

    Unlike the 2D manifest entries (which store ``kernel_size`` / ``stride`` /
    ``padding`` / ``dilation`` as ``[k, k]`` lists), the ``MaxPool1dFwdOp``
    workloads store them as scalar ints (``kernel_size: 3``), so a bare
    ``tuple(value)`` would raise ``TypeError: 'int' object is not iterable``.
    A scalar is wrapped to ``(value,)`` -- identical semantics to the manifest
    scalar, and the ``MaxPool1dFwdOp`` / ``MaxPool1dWorkload`` layers both take
    a length-1 tuple (``(k,)``).
    """
    return (value,) if isinstance(value, int) else tuple(value)


_MAX_POOL1D_INDICES_OP_NAME = "MaxPool1dIndicesFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_MAX_POOL1D_INDICES_OP_NAME, include_extra=True),
)
def test_max_pool1d_indices_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    n, c_in, l_in = shape
    kernel_size = _max_pool1d_as_tuple(op_params["kernel_size"])
    stride = op_params.get("stride")
    if stride is not None:
        stride = _max_pool1d_as_tuple(stride)
    padding = _max_pool1d_as_tuple(op_params.get("padding", (0,)))
    dilation = _max_pool1d_as_tuple(op_params.get("dilation", (1,)))
    ceil_mode = op_params.get("ceil_mode", False)

    test = MaxPool1dWorkload(
        n, c_in, l_in, kernel_size, stride, padding, dilation, ceil_mode, dtype
    )
    inputs = test.gen_inputs()

    op = MaxPool1dIndicesFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    bm = ManifestBenchmark(_MAX_POOL1D_INDICES_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


_ADAPTIVE_MAX_POOL2D_INDICES_OP_NAME = "AdaptiveMaxPool2dIndicesFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ADAPTIVE_MAX_POOL2D_INDICES_OP_NAME, include_extra=True),
)
def test_adaptive_max_pool2d_indices_bench(
    shape: tuple, dtype: torch.dtype, op_params: dict
) -> None:
    n, c_in, h_in, w_in = shape
    output_size = op_params["output_size"]

    test = AdaptivePool2dWorkload(n, c_in, h_in, w_in, output_size, dtype)
    inputs = test.gen_inputs()

    op = AdaptiveMaxPool2dIndicesFwdOp(output_size=output_size)
    bm = ManifestBenchmark(_ADAPTIVE_MAX_POOL2D_INDICES_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


_ADAPTIVE_MAX_POOL2D_OP_NAME = "AdaptiveMaxPool2dFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ADAPTIVE_MAX_POOL2D_OP_NAME, include_extra=True),
)
def test_adaptive_max_pool2d_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    n, c_in, h_in, w_in = shape
    output_size = op_params["output_size"]

    test = AdaptivePool2dWorkload(n, c_in, h_in, w_in, output_size, dtype)
    inputs = test.gen_inputs()

    op = AdaptiveMaxPool2dFwdOp(output_size=output_size)
    bm = ManifestBenchmark(_ADAPTIVE_MAX_POOL2D_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


_ADAPTIVE_AVG_POOL2D_OP_NAME = "AdaptiveAvgPool2dFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ADAPTIVE_AVG_POOL2D_OP_NAME, include_extra=True),
)
def test_adaptive_avg_pool2d_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    n, c_in, h_in, w_in = shape
    output_size = op_params["output_size"]

    test = AdaptivePool2dWorkload(n, c_in, h_in, w_in, output_size, dtype)
    inputs = test.gen_inputs()

    op = AdaptiveAvgPool2dFwdOp(output_size=output_size)
    bm = ManifestBenchmark(_ADAPTIVE_AVG_POOL2D_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")
