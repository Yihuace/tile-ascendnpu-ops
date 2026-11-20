"""Correctness tests for ArgminFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_argreduce.py`` — ArgminFwdOp subset
only (ArgmaxFwdOp is its own migration unit; both share the GPU
``ArgreduceKernel``, so the ported fixtures are argmin-specific here).
Uses the device backend instead of hard-coded ``"cuda"``.

Covers: ArgminFwdOp.
Each op reduces along a configurable dim and returns int64 indices.
Uses exact match (torch.equal) instead of allclose.
"""

from typing import cast

import pytest
import torch

from tileops.device import get_device_backend
from tileops.testing.test_base import TestBase
from tileops.workloads.reduction import ArgminWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


def _call(op, x: torch.Tensor) -> torch.Tensor:
    """Invoke a single-output argreduce op and narrow the return to ``Tensor``.

    The shared ``OpBase.__call__`` is typed as ``Union[Tensor, tuple]`` to
    accommodate ops with multiple outputs.  Argreduce ops always return a
    single ``Tensor``; this helper keeps the call sites well-typed without
    sprinkling ``cast(...)`` everywhere.
    """
    return cast(torch.Tensor, op(x))


# Fixtures


class ArgreduceBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-aligned N (non-pow2 last dim)
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class ArgreduceNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Argreduce3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Argreduce3DDim0Fixture(FixtureBase):
    """dim=0 reduction on 3D tensors — small outermost dim triggers
    the TileLang layout constraint (N << N_padded)."""

    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(4, 8, 256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(4, 8, 256, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4, 8, 256, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Argreduce4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Argreduce4DDim0Fixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 256, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Argreduce1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class SpecArgreduceFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype",
            [
                pytest.param((128, 512), -1, False, torch.float16, marks=pytest.mark.smoke),
                pytest.param((4, 32, 512), -1, False, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param((128, 512), -1, True, torch.float16, marks=pytest.mark.full),
                pytest.param((512, 4, 32), 0, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), 1, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), -1, True, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


# TestBase helper — inherits gen_inputs() from the workload class


class ArgminTest(ArgminWorkload, TestBase):
    """Parameterized test helper for ArgminFwdOp (dim=-1 2D shapes)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        super().__init__((m, n), dtype)

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        (x,) = inputs
        return x.argmin(dim=-1)


def _exact_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact match comparison using torch.equal."""
    assert output.dtype == torch.int64, f"Expected int64, got {output.dtype}"
    assert output_ref.dtype == torch.int64, f"Expected ref int64, got {output_ref.dtype}"
    assert torch.equal(output, output_ref), (
        f"Indices mismatch.\n"
        f"  output:     {output[:10]}...\n"
        f"  output_ref: {output_ref[:10]}...\n"
        f"  mismatches: {(output != output_ref).sum().item()} / {output.numel()}"
    )


# ArgminFwdOp tests


@ArgreduceBasicFixture
def test_argmin_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminFwdOp

    test = ArgminTest(m, n, dtype)
    op = ArgminFwdOp(dtype=dtype, dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@ArgreduceNonContigFixture
def test_argmin_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device=_device())
    x = x_full[:, :n]
    op = ArgminFwdOp(dtype=dtype, dim=-1)
    ref = x.contiguous().argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"non-contig argmin mismatch: {(y != ref).sum().item()}"


@Argreduce3DFixture
def test_argmin_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=_device())
    op = ArgminFwdOp(dtype=dtype, dim=-1)
    ref = x.argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D argmin mismatch: {(y != ref).sum().item()}"


@Argreduce4DFixture
def test_argmin_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=_device())
    op = ArgminFwdOp(dtype=dtype, dim=-1)
    ref = x.argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D argmin mismatch: {(y != ref).sum().item()}"


@Argreduce1DFixture
def test_argmin_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(n, dtype=dtype, device=_device())
    op = ArgminFwdOp(dtype=dtype, dim=-1)
    ref = x.argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y.view_as(ref), ref), "1D argmin mismatch"


@Argreduce3DDim0Fixture
def test_argmin_3d_dim0(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 on 3D tensors (outermost-dim reduction)."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=_device())
    op = ArgminFwdOp(dtype=dtype, dim=0)
    ref = x.argmin(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 argmin mismatch: {(y != ref).sum().item()}"


@Argreduce3DDim0Fixture
def test_argmin_3d_dim0_keepdim(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 with keepdim=True on 3D tensors."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=_device())
    op = ArgminFwdOp(dtype=dtype, dim=0, keepdim=True)
    ref = x.argmin(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 keepdim argmin mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmin_4d_dim0(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 on 4D tensors (outermost-dim reduction, 3D+ regression)."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=_device())
    op = ArgminFwdOp(dtype=dtype, dim=0)
    ref = x.argmin(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 argmin mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmin_4d_dim0_keepdim(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 with keepdim=True on 4D tensors."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=_device())
    op = ArgminFwdOp(dtype=dtype, dim=0, keepdim=True)
    ref = x.argmin(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 keepdim argmin mismatch: {(y != ref).sum().item()}"


@SpecArgreduceFixture
def test_argmin_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: ArgminFwdOp with dim + keepdim."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(*shape, dtype=dtype, device=_device())
    op = ArgminFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = x.argmin(dim=dim, keepdim=keepdim)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"spec dim={dim} argmin mismatch: {(y != ref).sum().item()}"


# Regression: multidim dim must be rejected for ArgminFwdOp


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls_path, dim",
    [
        ("tileops.ops.reduction.argmin.ArgminFwdOp", [0, 1]),
        ("tileops.ops.reduction.argmin.ArgminFwdOp", (0, 1)),
    ],
)
def test_argmin_rejects_multidim(op_cls_path: str, dim) -> None:
    """ArgminFwdOp only supports scalar dim or None; list/tuple must raise."""
    import importlib

    module_path, cls_name = op_cls_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    op_cls = getattr(mod, cls_name)

    with pytest.raises((TypeError, ValueError)):
        op_cls(dtype=torch.float16, dim=dim)


# dim=None (full-tensor reduction) tests


class ArgreduceDimNoneFixture(FixtureBase):
    """Full-tensor reduction (dim=None).

    Coverage rationale (testing.md §Test case policy):
      - dtype dispatch: one 2D shape across {fp16, bf16, fp32}.
      - ndim shape branches (flatten path): 1D / 3D / 4D in fp16 only;
        ndim and dtype are not crossed since the flatten code path is
        dtype-independent.
      - One non-aligned flat size as full coverage (tail handling).
    """

    PARAMS = [
        (
            "shape, dtype",
            [
                # dtype dispatch on a single 2D shape
                pytest.param((16, 64), torch.float16, marks=pytest.mark.smoke),
                pytest.param((16, 64), torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param((16, 64), torch.float32, marks=pytest.mark.smoke),
                # ndim shape coverage (flatten path is dtype-agnostic)
                pytest.param((512,), torch.float16, marks=pytest.mark.smoke),
                pytest.param((4, 8, 32), torch.float16, marks=pytest.mark.smoke),
                pytest.param((2, 4, 8, 16), torch.float16, marks=pytest.mark.smoke),
                # Non-aligned flat size (tail handling)
                pytest.param((10, 30), torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@ArgreduceDimNoneFixture
def test_argmin_dim_none(shape: tuple, dtype: torch.dtype) -> None:
    """ArgminFwdOp(dim=None) matches torch.argmin(x); covers keepdim={False, True}."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(*shape, dtype=dtype, device=_device())
    ref_flat = torch.argmin(x)

    y = _call(ArgminFwdOp(dtype=dtype, dim=None), x)
    assert y.dtype == torch.int64
    assert y.shape == ref_flat.shape, f"shape mismatch: {y.shape} vs {ref_flat.shape}"
    assert torch.equal(y, ref_flat), f"dim=None argmin mismatch on shape={shape} dtype={dtype}"

    y_keep = _call(ArgminFwdOp(dtype=dtype, dim=None, keepdim=True), x)
    expected_shape = tuple(1 for _ in shape)
    assert y_keep.shape == expected_shape, (
        f"keepdim shape mismatch: {y_keep.shape} vs {expected_shape}"
    )
    assert torch.equal(y_keep.reshape(()), ref_flat), (
        f"dim=None keepdim argmin value mismatch on shape={shape} dtype={dtype}"
    )


# Degenerate reduction extent (N == 1): closed-form index 0
#
# The kernel path has no tiling for an empty reduction extent (every path
# emits an empty-broadcast ``vbrc`` and fails to compile), so
# ``_ReduceOpBase._maybe_degenerate`` resolves it on the host. These cases
# pin the contract that a length-1 reduction axis yields index 0.


class ArgreduceDegenerateNFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype",
            [
                pytest.param((512, 1), -1, False, torch.float16, marks=pytest.mark.smoke),
                pytest.param((512, 1), -1, True, torch.float16, marks=pytest.mark.smoke),
                pytest.param((1, 512), 0, False, torch.float16, marks=pytest.mark.smoke),
                pytest.param((1, 512), 0, True, torch.float16, marks=pytest.mark.smoke),
                pytest.param((512, 1), -1, False, torch.bfloat16, marks=pytest.mark.full),
                pytest.param((1, 512), 0, False, torch.bfloat16, marks=pytest.mark.full),
                pytest.param((512, 1), -1, False, torch.float32, marks=pytest.mark.full),
                pytest.param((4, 1, 256), 1, False, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@ArgreduceDegenerateNFixture
def test_argmin_degenerate_n(shape, dim, keepdim, dtype) -> None:
    """Length-1 reduction axis returns index 0 (torch parity, no kernel)."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(*shape, dtype=dtype, device=_device())
    ref = torch.argmin(x, dim=dim, keepdim=keepdim)
    op = ArgminFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"N=1 ArgminFwdOp mismatch on shape={shape} dim={dim}"


@pytest.mark.smoke
def test_argmin_degenerate_dim_none_scalar() -> None:
    """``dim=None`` over a single element also yields 0 (0-D and keepdim forms)."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    for shape in ((1,), (1, 1)):
        x = torch.randn(*shape, dtype=torch.float16, device=_device())
        ref = torch.argmin(x)
        y = _call(ArgminFwdOp(dtype=torch.float16, dim=None), x)
        assert y.shape == ref.shape and torch.equal(y, ref), f"dim=None {shape}"
        y_keep = _call(ArgminFwdOp(dtype=torch.float16, dim=None, keepdim=True), x)
        assert y_keep.shape == tuple(1 for _ in shape), f"keepdim {shape}"
        assert torch.equal(y_keep.reshape(()), ref), f"keepdim value {shape}"


# Production-scale shapes and first-occurrence ties
#
# The tuned workloads (hidden-state (2048,4096) and lm-head (4,102400)) and
# the tie-break contract (``min{j}``) were only covered by size-reduced
# fixtures; these pin them at the shapes the fused / N-split paths are
# tuned for. Marked ``full``: they compile the production tilings.


class ArgreduceProductionShapeFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype",
            [
                pytest.param((2048, 4096), torch.float16, marks=pytest.mark.full),
                pytest.param((2048, 4096), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((4, 102400), torch.float16, marks=pytest.mark.full),
                pytest.param((4, 102400), torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@ArgreduceProductionShapeFixture
def test_argmin_production_shape(shape, dtype) -> None:
    """dim=-1 argmin matches torch on the tuned workload shapes."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(*shape, dtype=dtype, device=_device())
    ref = torch.argmin(x, dim=-1)
    y = _call(ArgminFwdOp(dtype=dtype, dim=-1), x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"{(y != ref).sum().item()} mismatches on {shape} {dtype}"


class ArgreduceTieFixture(FixtureBase):
    PARAMS = [
        (
            "case, dtype",
            [
                pytest.param("all_equal", torch.float16, marks=pytest.mark.smoke),
                pytest.param("first_and_last", torch.float16, marks=pytest.mark.smoke),
                pytest.param("three_way", torch.float16, marks=pytest.mark.full),
                pytest.param("all_equal", torch.bfloat16, marks=pytest.mark.full),
                pytest.param("first_and_last", torch.bfloat16, marks=pytest.mark.full),
                pytest.param("all_equal_nsplit", torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@ArgreduceTieFixture
def test_argmin_tie_break(case, dtype) -> None:
    """Ties must resolve to the smallest column index (first occurrence)."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    if case == "all_equal":
        x = torch.zeros((512, 4096), dtype=dtype, device=_device())
    elif case == "first_and_last":
        x = torch.zeros((512, 4096), dtype=dtype, device=_device())
        x[:, 0] = 1.0
        x[:, -1] = 1.0
    elif case == "three_way":
        x = torch.zeros((512, 4096), dtype=dtype, device=_device())
        x[:, 2048] = 1.0
        x[:, 3000] = 1.0
        x[:, 4095] = 1.0
    else:  # all_equal_nsplit: across N-split chunk boundaries
        x = torch.zeros((4, 102400), dtype=dtype, device=_device())
    ref = torch.argmin(x, dim=-1)
    y = _call(ArgminFwdOp(dtype=dtype, dim=-1), x)
    assert torch.equal(y, ref), f"tie case {case} {dtype}: {(y != ref).sum().item()} mismatches"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
