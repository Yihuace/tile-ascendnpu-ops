"""Correctness tests for ArgmaxFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_argreduce.py`` — ArgmaxFwdOp subset
only (ArgminFwdOp is its own migration unit). Uses the device backend
instead of hard-coded ``"cuda"``.

Covers: ArgmaxFwdOp.
Each op reduces along a configurable dim and returns int64 indices.
Uses exact match (torch.equal) instead of allclose.
"""

from typing import cast

import pytest
import torch

from tileops.device import get_device_backend
from tileops.testing.test_base import TestBase
from tileops.workloads.reduction import ArgmaxWorkload
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


class ArgmaxTest(ArgmaxWorkload, TestBase):
    """Parameterized test helper for ArgmaxFwdOp (dim=-1 2D shapes)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        super().__init__((m, n), dtype)

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        (x,) = inputs
        return x.argmax(dim=-1)


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


# ArgmaxFwdOp tests


@ArgreduceBasicFixture
def test_argmax_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    test = ArgmaxTest(m, n, dtype)
    op = ArgmaxFwdOp(dtype=dtype, dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@ArgreduceNonContigFixture
def test_argmax_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device=_device())
    x = x_full[:, :n]
    op = ArgmaxFwdOp(dtype=dtype, dim=-1)
    ref = x.contiguous().argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"non-contig argmax mismatch: {(y != ref).sum().item()}"


@Argreduce3DFixture
def test_argmax_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=_device())
    op = ArgmaxFwdOp(dtype=dtype, dim=-1)
    ref = x.argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D argmax mismatch: {(y != ref).sum().item()}"


@Argreduce4DFixture
def test_argmax_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=_device())
    op = ArgmaxFwdOp(dtype=dtype, dim=-1)
    ref = x.argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D argmax mismatch: {(y != ref).sum().item()}"


@Argreduce1DFixture
def test_argmax_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(n, dtype=dtype, device=_device())
    op = ArgmaxFwdOp(dtype=dtype, dim=-1)
    ref = x.argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y.view_as(ref), ref), "1D argmax mismatch"


@Argreduce3DDim0Fixture
def test_argmax_3d_dim0(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 on 3D tensors (outermost-dim reduction)."""
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=_device())
    op = ArgmaxFwdOp(dtype=dtype, dim=0)
    ref = x.argmax(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 argmax mismatch: {(y != ref).sum().item()}"


@Argreduce3DDim0Fixture
def test_argmax_3d_dim0_keepdim(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 with keepdim=True on 3D tensors."""
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=_device())
    op = ArgmaxFwdOp(dtype=dtype, dim=0, keepdim=True)
    ref = x.argmax(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 keepdim argmax mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmax_4d_dim0(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 on 4D tensors (outermost-dim reduction, 3D+ regression)."""
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=_device())
    op = ArgmaxFwdOp(dtype=dtype, dim=0)
    ref = x.argmax(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 argmax mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmax_4d_dim0_keepdim(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 with keepdim=True on 4D tensors."""
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=_device())
    op = ArgmaxFwdOp(dtype=dtype, dim=0, keepdim=True)
    ref = x.argmax(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 keepdim argmax mismatch: {(y != ref).sum().item()}"


@SpecArgreduceFixture
def test_argmax_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: ArgmaxFwdOp with dim + keepdim."""
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(*shape, dtype=dtype, device=_device())
    op = ArgmaxFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = x.argmax(dim=dim, keepdim=keepdim)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"spec dim={dim} argmax mismatch: {(y != ref).sum().item()}"


# Regression: multidim dim must be rejected for ArgmaxFwdOp


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls_path, dim",
    [
        ("tileops.ops.reduction.argmax.ArgmaxFwdOp", [0, 1]),
        ("tileops.ops.reduction.argmax.ArgmaxFwdOp", (0, 1)),
    ],
)
def test_argmax_rejects_multidim(op_cls_path: str, dim) -> None:
    """ArgmaxFwdOp only supports scalar dim or None; list/tuple must raise."""
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
def test_argmax_dim_none(shape: tuple, dtype: torch.dtype) -> None:
    """ArgmaxFwdOp(dim=None) matches torch.argmax(x); covers keepdim={False, True}."""
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(*shape, dtype=dtype, device=_device())
    ref_flat = torch.argmax(x)

    y = _call(ArgmaxFwdOp(dtype=dtype, dim=None), x)
    assert y.dtype == torch.int64
    assert y.shape == ref_flat.shape, f"shape mismatch: {y.shape} vs {ref_flat.shape}"
    assert torch.equal(y, ref_flat), f"dim=None argmax mismatch on shape={shape} dtype={dtype}"

    y_keep = _call(ArgmaxFwdOp(dtype=dtype, dim=None, keepdim=True), x)
    expected_shape = tuple(1 for _ in shape)
    assert y_keep.shape == expected_shape, (
        f"keepdim shape mismatch: {y_keep.shape} vs {expected_shape}"
    )
    assert torch.equal(y_keep.reshape(()), ref_flat), (
        f"dim=None keepdim argmax value mismatch on shape={shape} dtype={dtype}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
