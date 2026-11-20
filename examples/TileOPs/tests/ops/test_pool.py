"""Correctness tests for the adaptive 2D pool ops (NPU).

Adapted from TileOPs ``tests/ops/test_pool.py`` -- the ``AdaptiveAvgPool2d`` and
``AdaptiveMaxPool2d`` subsets (the shared GPU file also covers the avg/max pool
1d/2d/3d ops and the adaptive max-pool indices variant, which are not migrated;
their imports and cases are dropped).

Adaptations:
- T1: manual tensor creation uses ``get_device_backend().name``.
- T3: the ``tune`` parametrization dimension is removed (NPU has no autotune).

Note: these ops are fp16/bf16-only (adaptive-pool dtype gate), so there is no
fp32 smoke case -- the GPU case set is ported as-is.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.kernels.pool.adaptive_avg_pool2d import AdaptiveAvgPool2dKernel
from tileops.kernels.pool.adaptive_max_pool2d import AdaptiveMaxPool2dKernel
from tileops.kernels.pool.adaptive_max_pool2d_with_indices import AdaptiveMaxPool2dWithIndicesKernel
from tileops.ops.pool.adaptive_avg_pool2d import AdaptiveAvgPool2dFwdOp
from tileops.ops.pool.adaptive_max_pool2d import AdaptiveMaxPool2dFwdOp
from tileops.ops.pool.adaptive_max_pool2d_with_indices import AdaptiveMaxPool2dIndicesFwdOp
from tileops.kernels.pool.max_pool1d_with_indices import MaxPool1dWithIndicesKernel
from tileops.ops.pool.max_pool1d_with_indices import MaxPool1dIndicesFwdOp
from tileops.kernels.pool.max_pool2d import MaxPool2dKernel
from tileops.ops.pool.max_pool2d import MaxPool2dFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.pool import AdaptivePool2dWorkload
from tileops.workloads.pool import MaxPool1dWorkload
from tileops.workloads.pool import MaxPool2dWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


_MAX_POOL2D_PARAMS = [
    # Smoke: one config across all three supported dtypes (fp32 first per the
    # NPU test-structure contract; the windowed max-pool op accepts float32,
    # unlike the adaptive pool family).
    pytest.param(
        1,
        8,
        16,
        16,
        (3, 3),
        (2, 2),
        (1, 1),
        (1, 1),
        False,
        torch.float32,
        True,
        marks=[pytest.mark.smoke, pytest.mark.packaging],
        id="smoke-3x3-s2-p1-fp32",
    ),
    pytest.param(
        2,
        8,
        16,
        16,
        (3, 3),
        (2, 2),
        (1, 1),
        (1, 1),
        False,
        torch.float16,
        True,
        marks=pytest.mark.smoke,
        id="smoke-3x3-s2-p1-fp16",
    ),
    pytest.param(
        2,
        8,
        16,
        16,
        (3, 3),
        (2, 2),
        (1, 1),
        (1, 1),
        False,
        torch.bfloat16,
        True,
        marks=pytest.mark.smoke,
        id="smoke-3x3-s2-p1-bf16",
    ),
    # Full: distinct setting combinations (GPU case set, ``tune`` dropped).
    pytest.param(
        1,
        4,
        14,
        14,
        (3, 3),
        None,
        (1, 1),
        (2, 1),
        False,
        torch.float16,
        True,
        marks=pytest.mark.full,
        id="full-default-stride-dilation-fp16",
    ),
    pytest.param(
        1,
        4,
        23,
        27,
        (3, 5),
        (2, 3),
        (1, 2),
        (1, 1),
        True,
        torch.float16,
        True,
        marks=pytest.mark.full,
        id="full-nonsquare-ceil-fp16",
    ),
    pytest.param(
        2,
        8,
        16,
        16,
        (3, 3),
        (2, 2),
        (1, 1),
        (1, 1),
        False,
        torch.float16,
        False,
        marks=pytest.mark.full,
        id="full-noncontiguous-3x3-fp16",
    ),
    pytest.param(
        1,
        4,
        23,
        27,
        (3, 5),
        (2, 3),
        (1, 2),
        (1, 1),
        True,
        torch.bfloat16,
        True,
        marks=pytest.mark.full,
        id="full-nonsquare-ceil-bf16",
    ),
]


class MaxPool2dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h_in, w_in, kernel_size, stride, padding, dilation, "
            "ceil_mode, dtype, contiguous",
            _MAX_POOL2D_PARAMS,
        ),
    ]


class MaxPool2dTest(MaxPool2dWorkload, TestBase):
    """Single-output 2D max-pool reference harness."""

    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        # Single-output op: no return_indices branch (the indices variant is
        # not migrated).
        kwargs = dict(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
        )
        # NPU-adapted reference: torch_npu routes ``F.max_pool2d`` to
        # ``aclnnMaxPool2dWithMask``, whose golden only supports ``dilation == 1``
        # (it raises ``AclNN_Parameter_Error(EZ1001): dilation only support 1``).
        # Windowed max pooling is a pure selection with no accumulation, so a
        # CPU golden is bit-identical to the NPU result; for any dilation != 1
        # compute the reference on CPU and move it back to the input device, so
        # the exact comparison (atol=rtol=0) is preserved.
        dilation = self.dilation
        needs_cpu_golden = (
            dilation != 1 if isinstance(dilation, int) else any(d != 1 for d in dilation)
        )
        if needs_cpu_golden:
            return F.max_pool2d(input.cpu(), **kwargs).to(input.device)
        return F.max_pool2d(input, **kwargs)


@MaxPool2dFixture
def test_max_pool2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] | None,
    padding: tuple[int, int],
    dilation: tuple[int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    contiguous: bool,
) -> None:
    test = MaxPool2dTest(
        n,
        c_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        contiguous=contiguous,
    )
    op = MaxPool2dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    # Max is exact (no accumulation): compare bit-for-bit like the GPU suite.
    test.check(op, *test.gen_inputs(), atol=0, rtol=0)
    assert isinstance(op.kernel, MaxPool2dKernel)


@pytest.mark.smoke
def test_max_pool2d_dynamic_shape_kernel_cache_and_roofline() -> None:
    """Kernel cache keys on the input shape; roofline needs a prior forward."""
    op = MaxPool2dFwdOp(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    x1 = torch.randn(1, 4, 16, 16, dtype=torch.float16, device=_device())
    x2 = torch.randn(2, 4, 16, 16, dtype=torch.float16, device=_device())

    with pytest.raises(RuntimeError, match="requires a prior forward"):
        op.eval_roofline()

    op(x1)
    assert len(op._kernel_cache) == 1
    flops, nbytes = op.eval_roofline()
    assert flops > 0
    assert nbytes > 0

    op(x1)
    assert len(op._kernel_cache) == 1

    op(x2)
    assert len(op._kernel_cache) == 2


@pytest.mark.smoke
def test_max_pool2d_infer_output_shapes_snapshot() -> None:
    """``_infer_output_shapes`` matches the manifest extent rule (k3/s2/p1)."""
    op = MaxPool2dFwdOp(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    assert op._infer_output_shapes((2, 4, 16, 16)) == {"output": (2, 4, 8, 8)}
    # ceil_mode steps the last window back when it starts in the padding.
    op_ceil = MaxPool2dFwdOp(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=True)
    assert op_ceil._infer_output_shapes((32, 64, 55, 55)) == {"output": (32, 64, 27, 27)}


_MAX_POOL2D_INVALID_PARAM_CASES = [
    pytest.param(
        {"kernel_size": (3, 3), "stride": True},
        TypeError,
        "stride must be an int or a tuple of 2 ints",
        id="2d-stride-type",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        {"kernel_size": (3, True)},
        TypeError,
        "kernel_size must contain only ints",
        id="2d-kernel-size-contents",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        {"kernel_size": (3, 3), "stride": (1, 0)},
        ValueError,
        "stride must be greater than zero",
        id="2d-zero-stride",
        marks=pytest.mark.full,
    ),
    pytest.param(
        {"kernel_size": (3, 3), "dilation": (0, 1)},
        ValueError,
        "dilation must be greater than zero",
        id="2d-zero-dilation",
        marks=pytest.mark.full,
    ),
    pytest.param(
        {"kernel_size": (3, 3), "padding": (2, 1)},
        ValueError,
        "padding must be at most half",
        id="2d-padding-too-large",
        marks=pytest.mark.full,
    ),
    pytest.param(
        {"kernel_size": (3, 3), "padding": (-1, 0)},
        ValueError,
        "padding must be non-negative",
        id="2d-padding-negative",
        marks=pytest.mark.full,
    ),
    pytest.param(
        {"kernel_size": (3, 3), "ceil_mode": "true"},
        TypeError,
        "ceil_mode must be a bool",
        id="2d-ceil-mode-type",
        marks=pytest.mark.full,
    ),
]


@pytest.mark.parametrize(("kwargs", "exc_type", "match"), _MAX_POOL2D_INVALID_PARAM_CASES)
def test_max_pool2d_rejects_invalid_params(
    kwargs: dict[str, object],
    exc_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exc_type, match=match):
        MaxPool2dFwdOp(**kwargs)


_MAX_POOL2D_INVALID_INPUT_CASES = [
    pytest.param(
        {"kernel_size": (3, 3)},
        ((2, 8, 16), torch.float16),
        "expects input to be a 4D NCHW tensor",
        id="2d-wrong-rank-input",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        {"kernel_size": (3, 3)},
        ((1, 1, 8, 8), torch.float64),
        "input.dtype must be float16, bfloat16, or float32",
        id="2d-unsupported-dtype",
        marks=pytest.mark.full,
    ),
    pytest.param(
        {"kernel_size": (5, 5), "stride": (1, 1), "padding": (0, 0)},
        ((1, 1, 2, 2), torch.float16),
        "output size must be greater than zero",
        id="2d-non-positive-output-size",
        marks=pytest.mark.full,
    ),
]


@pytest.mark.parametrize(
    ("kwargs", "input_spec", "expected_match"), _MAX_POOL2D_INVALID_INPUT_CASES
)
def test_max_pool2d_rejects_invalid_input(
    kwargs: dict[str, object],
    input_spec: tuple[tuple[int, ...], torch.dtype],
    expected_match: str,
) -> None:
    shape, dtype = input_spec
    op = MaxPool2dFwdOp(**kwargs)
    x = torch.randn(*shape, dtype=dtype, device=_device())
    with pytest.raises(ValueError, match=expected_match):
        op(x)


_MAX_POOL1D_PARAMS = [
    # Smoke: one config across all three supported dtypes (fp32 first per the
    # NPU test-structure contract; the windowed max-pool op accepts float32,
    # unlike the adaptive pool family).
    pytest.param(
        1,
        8,
        32,
        (3,),
        (2,),
        (1,),
        (1,),
        False,
        torch.float32,
        True,
        marks=[pytest.mark.smoke, pytest.mark.packaging],
        id="smoke-k3-s2-p1-fp32",
    ),
    pytest.param(
        2,
        8,
        64,
        (3,),
        (2,),
        (1,),
        (1,),
        False,
        torch.float16,
        True,
        marks=pytest.mark.smoke,
        id="smoke-k3-s2-p1-fp16",
    ),
    pytest.param(
        2,
        8,
        64,
        (3,),
        (2,),
        (1,),
        (1,),
        False,
        torch.bfloat16,
        True,
        marks=pytest.mark.smoke,
        id="smoke-k3-s2-p1-bf16",
    ),
    # Full: distinct setting combinations (GPU case set, ``tune`` dropped).
    pytest.param(
        1,
        4,
        63,
        (3,),
        None,
        (1,),
        (2,),
        False,
        torch.float16,
        True,
        marks=pytest.mark.full,
        id="full-default-stride-dilation-fp16",
    ),
    pytest.param(
        1,
        4,
        97,
        (5,),
        (3,),
        (2,),
        (1,),
        True,
        torch.float16,
        True,
        marks=pytest.mark.full,
        id="full-ceil-k5-s3-p2-fp16",
    ),
    pytest.param(
        2,
        8,
        64,
        (3,),
        (2,),
        (1,),
        (1,),
        False,
        torch.float16,
        False,
        marks=pytest.mark.full,
        id="full-noncontiguous-k3-fp16",
    ),
    pytest.param(
        1,
        4,
        97,
        (5,),
        (3,),
        (2,),
        (1,),
        True,
        torch.bfloat16,
        True,
        marks=pytest.mark.full,
        id="full-ceil-k5-s3-p2-bf16",
    ),
]


_MAX_POOL1D_INVALID_PARAM_CASES = [
    pytest.param(
        {"kernel_size": True},
        TypeError,
        "kernel_size must be an int or a tuple of 1 ints",
        id="1d-kernel-size-type",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        {"kernel_size": 3, "stride": 0},
        ValueError,
        "stride must be greater than zero",
        id="1d-zero-stride",
        marks=pytest.mark.full,
    ),
    pytest.param(
        {"kernel_size": 3, "dilation": 0},
        ValueError,
        "dilation must be greater than zero",
        id="1d-zero-dilation",
        marks=pytest.mark.full,
    ),
    pytest.param(
        {"kernel_size": 3, "padding": 2},
        ValueError,
        "padding must be at most half",
        id="1d-padding-too-large",
        marks=pytest.mark.full,
    ),
    pytest.param(
        {"kernel_size": 3, "padding": -1},
        ValueError,
        "padding must be non-negative",
        id="1d-padding-negative",
        marks=pytest.mark.full,
    ),
    pytest.param(
        {"kernel_size": 3, "ceil_mode": "true"},
        TypeError,
        "ceil_mode must be a bool",
        id="1d-ceil-mode-type",
        marks=pytest.mark.full,
    ),
]


_MAX_POOL1D_INVALID_INPUT_CASES = [
    pytest.param(
        {"kernel_size": 3},
        ((2, 8, 16, 16), torch.float16),
        "expects input to be a 3D NCL tensor",
        id="1d-wrong-rank-input",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        {"kernel_size": 3},
        ((1, 1, 8), torch.float64),
        "input.dtype must be float16, bfloat16, or float32",
        id="1d-unsupported-dtype",
        marks=pytest.mark.full,
    ),
    pytest.param(
        {"kernel_size": 5, "stride": 1, "padding": 0},
        ((1, 1, 2), torch.float16),
        "output size must be greater than zero",
        id="1d-non-positive-output-size",
        marks=pytest.mark.full,
    ),
]


class MaxPool1dIndicesFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, l_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, contiguous",
            _MAX_POOL1D_PARAMS,
        ),
    ]


class MaxPool1dIndicesTest(MaxPool1dWorkload, TestBase):
    """Dual-output 1D max-pool reference harness (values + int64 indices)."""

    def ref_program(self, input: torch.Tensor):
        # Dual output: the reference returns (values, int64 indices).
        kwargs = dict(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=True,
        )
        # NPU-adapted reference: torch_npu routes ``F.max_pool1d`` to an aclnn
        # golden whose ``return_indices`` support is not usable on NPU.  For
        # ``dilation != 1`` it rejects the parameters outright
        # (``AclNN_Parameter_Error(EZ1001)``, the 2d sibling lesson); for
        # ``dilation == 1`` it runs but returns a corrupt index tensor -- an
        # ``int8`` ``(N, C, kernel_size, L_pad)`` internal mask buffer (verified
        # on this toolchain: ``F.max_pool1d(x_npu, ..., return_indices=True)``
        # yields ``torch.int8`` ``(1, 8, 3, 64)``) instead of the required
        # ``int64`` ``(N, C, L_out)`` index tensor.  Windowed max pooling is a
        # pure selection with no accumulation, so the CPU golden is bit-identical
        # to the correct NPU result for both values *and* indices; compute the
        # reference on CPU and move it back to the input device, so the exact
        # comparison (atol=rtol=0) is preserved.  (Reference path only -- cases,
        # assertions and tolerances unchanged.)
        val, idx = F.max_pool1d(input.cpu(), **kwargs)
        return val.to(input.device), idx.to(input.device)


@MaxPool1dIndicesFixture
def test_max_pool1d_indices(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: tuple[int],
    stride: tuple[int] | None,
    padding: tuple[int],
    dilation: tuple[int],
    ceil_mode: bool,
    dtype: torch.dtype,
    contiguous: bool,
) -> None:
    test = MaxPool1dIndicesTest(
        n,
        c_in,
        l_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        contiguous=contiguous,
    )
    op = MaxPool1dIndicesFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    # Max is exact (no accumulation) and the index comparison is exact too.
    test.check(op, *test.gen_inputs(), atol=0, rtol=0)
    assert isinstance(op.kernel, MaxPool1dWithIndicesKernel)


@pytest.mark.smoke
def test_max_pool1d_indices_dynamic_shape_kernel_cache_and_roofline() -> None:
    """Kernel cache keys on the input shape; roofline needs a prior forward."""
    op = MaxPool1dIndicesFwdOp(kernel_size=3, stride=2, padding=1)
    x1 = torch.randn(1, 4, 32, dtype=torch.float16, device=_device())
    x2 = torch.randn(2, 4, 32, dtype=torch.float16, device=_device())

    with pytest.raises(RuntimeError, match="requires a prior forward"):
        op.eval_roofline()

    op(x1)
    assert len(op._kernel_cache) == 1
    flops, nbytes = op.eval_roofline()
    assert flops > 0
    assert nbytes > 0

    op(x1)
    assert len(op._kernel_cache) == 1

    op(x2)
    assert len(op._kernel_cache) == 2


@pytest.mark.smoke
def test_max_pool1d_indices_infer_output_shapes_snapshot() -> None:
    """``_infer_output_shapes`` matches the manifest extent rule (k3/s2/p1)."""
    op = MaxPool1dIndicesFwdOp(kernel_size=3, stride=2, padding=1)
    assert op._infer_output_shapes((2, 4, 32)) == {
        "output": (2, 4, 16),
        "indices": (2, 4, 16),
    }
    # ceil_mode steps the last window back when it starts in the padding.
    op_ceil = MaxPool1dIndicesFwdOp(kernel_size=3, stride=2, padding=0, ceil_mode=True)
    assert op_ceil._infer_output_shapes((32, 64, 55)) == {
        "output": (32, 64, 27),
        "indices": (32, 64, 27),
    }


@pytest.mark.smoke
def test_max_pool1d_indices_eval_roofline_snapshot() -> None:
    """``eval_roofline`` matches the manifest flops/bytes formula (k2/s2/p0)."""
    op = MaxPool1dIndicesFwdOp(kernel_size=2, stride=2, padding=0)
    x = torch.randn(1, 2, 8, dtype=torch.float16, device=_device())
    op(x)
    # flops = n*c*prod(out)*prod(k) = 1*2*4*2 = 16
    # bytes = (n*c*prod(in) + n*c*prod(out)) * 2 + n*c*prod(out) * 8
    #       = (16 + 8) * 2 + 8 * 8 = 48 + 64 = 112
    assert op.eval_roofline() == (16, 112)


@pytest.mark.smoke
def test_max_pool1d_indices_nan_window() -> None:
    """A NaN in the window wins, and the recorded index is the last one seen."""
    x = torch.randn(1, 1, 8, device=_device(), dtype=torch.float16)
    x[0, 0, 2] = float("nan")
    x[0, 0, 5] = float("nan")

    op = MaxPool1dIndicesFwdOp(kernel_size=3, stride=3, padding=0)
    val, idx = op(x)
    # Reference path: NPU ``F.max_pool1d(..., return_indices=True)`` returns a
    # corrupt int8 mask buffer (see ``MaxPool1dIndicesTest.ref_program``);
    # compute the golden on CPU and move it back.  Cases / assertions /
    # tolerances are unchanged.
    ref_val, ref_idx = F.max_pool1d(
        x.cpu().float(),
        kernel_size=(3,),
        stride=(3,),
        padding=(0,),
        return_indices=True,
    )
    ref_val = ref_val.to(x.device)
    ref_idx = ref_idx.to(x.device)
    assert torch.isnan(val.float()).any()
    torch.testing.assert_close(idx, ref_idx, atol=0, rtol=0)
    nan_mask = torch.isnan(val.float())
    # equal_nan=True is required: the selected slice is the (all-NaN) window
    # values and the intended exact comparison is NaN == NaN (the same NaN-aware
    # convention ``tileops.testing.test_base.allclose_compare`` uses).  The
    # tolerant-free atol=rtol=0 criterion, the case and the strict index
    # comparison above are all unchanged.
    torch.testing.assert_close(
        val.float()[nan_mask], ref_val[nan_mask], atol=0, rtol=0, equal_nan=True
    )


@pytest.mark.parametrize(("kwargs", "exc_type", "match"), _MAX_POOL1D_INVALID_PARAM_CASES)
def test_max_pool1d_indices_rejects_invalid_params(
    kwargs: dict[str, object],
    exc_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exc_type, match=match):
        MaxPool1dIndicesFwdOp(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "input_spec", "expected_match"), _MAX_POOL1D_INVALID_INPUT_CASES
)
def test_max_pool1d_indices_rejects_invalid_input(
    kwargs: dict[str, object],
    input_spec: tuple[tuple[int, ...], torch.dtype],
    expected_match: str,
) -> None:
    shape, dtype = input_spec
    op = MaxPool1dIndicesFwdOp(**kwargs)
    x = torch.randn(*shape, dtype=dtype, device=_device())
    with pytest.raises(ValueError, match=expected_match):
        op(x)


class AdaptiveMaxPool2dIndicesFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h_in, w_in, output_size, dtype",
            [
                pytest.param(
                    2,
                    64,
                    56,
                    56,
                    (6, 6),
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-spp-6x6-fp16",
                ),
                pytest.param(
                    1,
                    96,
                    28,
                    30,
                    (7, 5),
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-asymmetric-bf16",
                ),
                pytest.param(
                    1,
                    128,
                    55,
                    57,
                    (7, 7),
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-nondiv-fp16",
                ),
                pytest.param(
                    1,
                    16,
                    11,
                    13,
                    (5, None),
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-partial-none-fp16",
                ),
            ],
        ),
    ]


class AdaptiveMaxPool2dIndicesTest(AdaptivePool2dWorkload, TestBase):
    def ref_program(self, input: torch.Tensor):
        # torch rejects a scalar None here; (None, None) means the same.
        # Dual output: the reference returns (values, int64 indices).
        size = (None, None) if self.output_size is None else self.output_size
        return F.adaptive_max_pool2d(input, size, return_indices=True)


@AdaptiveMaxPool2dIndicesFixture
def test_adaptive_max_pool2d_indices(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    output_size: int | None | tuple[int | None, int | None],
    dtype: torch.dtype,
) -> None:
    test = AdaptiveMaxPool2dIndicesTest(n, c_in, h_in, w_in, output_size, dtype)
    op = AdaptiveMaxPool2dIndicesFwdOp(output_size)
    # Max is exact (no accumulation) and the index comparison is exact too.
    test.check(op, *test.gen_inputs(), atol=0, rtol=0)
    assert isinstance(op.kernel, AdaptiveMaxPool2dWithIndicesKernel)


@pytest.mark.smoke
def test_adaptive_max_pool2d_indices_accepts_chw() -> None:
    """A 3-D input keeps its rank on both outputs for the indices variant."""
    x = torch.randn(16, 11, 13, device=_device(), dtype=torch.float16)
    val, idx = AdaptiveMaxPool2dIndicesFwdOp(output_size=(5, 4))(x)
    ref_val, ref_idx = F.adaptive_max_pool2d(x.float(), (5, 4), return_indices=True)
    assert val.shape == (16, 5, 4) and idx.shape == (16, 5, 4)
    torch.testing.assert_close(val.float(), ref_val, atol=0, rtol=0)
    torch.testing.assert_close(idx, ref_idx, atol=0, rtol=0)


@pytest.mark.smoke
def test_adaptive_max_pool2d_indices_nan_window() -> None:
    """A NaN in the window wins, and the recorded index is the last one seen."""
    x = torch.randn(1, 1, 4, 4, device=_device(), dtype=torch.float16)
    x[0, 0, 1, 1] = float("nan")
    x[0, 0, 3, 3] = float("nan")

    val, idx = AdaptiveMaxPool2dIndicesFwdOp(output_size=(1, 1))(x)
    ref_val, ref_idx = F.adaptive_max_pool2d(x.float(), (1, 1), return_indices=True)
    assert torch.isnan(val.float()).all()
    torch.testing.assert_close(idx, ref_idx, atol=0, rtol=0)


class AdaptiveMaxPool2dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h_in, w_in, output_size, dtype",
            [
                pytest.param(
                    2,
                    64,
                    56,
                    56,
                    (6, 6),
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-spp-6x6-fp16",
                ),
                pytest.param(
                    1,
                    96,
                    28,
                    30,
                    (7, 5),
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-asymmetric-bf16",
                ),
                pytest.param(
                    1,
                    128,
                    55,
                    57,
                    (7, 7),
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-nondiv-fp16",
                ),
                pytest.param(
                    1,
                    16,
                    11,
                    13,
                    (5, None),
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-partial-none-fp16",
                ),
            ],
        ),
    ]


class AdaptiveMaxPool2dTest(AdaptivePool2dWorkload, TestBase):
    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        # torch rejects a scalar None here; (None, None) means the same.
        # Single-output op: no return_indices branch (the indices variant is
        # not migrated).
        size = (None, None) if self.output_size is None else self.output_size
        return F.adaptive_max_pool2d(input, size)


@AdaptiveMaxPool2dFixture
def test_adaptive_max_pool2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    output_size: int | None | tuple[int | None, int | None],
    dtype: torch.dtype,
) -> None:
    test = AdaptiveMaxPool2dTest(n, c_in, h_in, w_in, output_size, dtype)
    op = AdaptiveMaxPool2dFwdOp(output_size)
    # Max is exact (no accumulation): compare bit-for-bit like the GPU suite.
    test.check(op, *test.gen_inputs(), atol=0, rtol=0)
    assert isinstance(op.kernel, AdaptiveMaxPool2dKernel)


@pytest.mark.smoke
def test_adaptive_pool2d_accepts_chw() -> None:
    """A 3-D input keeps its rank on the way out (max-pool half)."""
    x = torch.randn(16, 11, 13, device=_device(), dtype=torch.float16)
    mx = AdaptiveMaxPool2dFwdOp(output_size=(5, 4))(x)
    assert mx.shape == (16, 5, 4)
    torch.testing.assert_close(mx.float(), F.adaptive_max_pool2d(x.float(), (5, 4)), atol=0, rtol=0)


class AdaptiveAvgPool2dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h_in, w_in, output_size, dtype",
            [
                pytest.param(
                    2,
                    64,
                    56,
                    56,
                    (6, 6),
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-spp-6x6-fp16",
                ),
                pytest.param(
                    1,
                    96,
                    28,
                    30,
                    (7, 5),
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-asymmetric-bf16",
                ),
                pytest.param(
                    1,
                    128,
                    55,
                    57,
                    (7, 7),
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-nondiv-fp16",
                ),
                pytest.param(
                    1,
                    16,
                    11,
                    13,
                    (None, 5),
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-partial-none-fp16",
                ),
                pytest.param(
                    1,
                    16,
                    10,
                    10,
                    4,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="full-scalar-output-size-bf16",
                ),
                pytest.param(
                    1,
                    8,
                    9,
                    11,
                    None,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-all-none-identity-fp16",
                ),
            ],
        ),
    ]


class AdaptiveAvgPool2dTest(AdaptivePool2dWorkload, TestBase):
    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        # torch rejects a scalar None here; (None, None) means the same.
        size = (None, None) if self.output_size is None else self.output_size
        return F.adaptive_avg_pool2d(input, size)


@AdaptiveAvgPool2dFixture
def test_adaptive_avg_pool2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    output_size: int | None | tuple[int | None, int | None],
    dtype: torch.dtype,
) -> None:
    test = AdaptiveAvgPool2dTest(n, c_in, h_in, w_in, output_size, dtype)
    op = AdaptiveAvgPool2dFwdOp(output_size)
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)
    assert isinstance(op.kernel, AdaptiveAvgPool2dKernel)


@pytest.mark.smoke
def test_adaptive_avg_pool2d_accepts_chw() -> None:
    """A 3-D input keeps its rank on the way out (avg-pool half)."""
    x = torch.randn(16, 11, 13, device=_device(), dtype=torch.float16)
    avg = AdaptiveAvgPool2dFwdOp(output_size=(5, 4))(x)
    assert avg.shape == (16, 5, 4)
    torch.testing.assert_close(
        avg.float(), F.adaptive_avg_pool2d(x.float(), (5, 4)), atol=2e-3, rtol=2e-3
    )
