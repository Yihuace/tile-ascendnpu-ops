"""Correctness tests for ClampMaxFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_special_elementwise_conformance.py`` --
ClampMax subset only (the shared GPU file also covers where / clamp /
clamp_min / masked_fill / nan_to_num, which are separate ops).  Covers flat
same-shape float inputs (fp32 / fp16 / bf16), the broadcast patterns
(per-channel, row, scalar, 0-dim), the PyTorch-aligned signature, NaN
propagation, dtype rejection, and the roofline formula.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``)
  instead of hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards
  removed (NPU availability is a runtime concern).
- T3/T4: the GPU ClampMax tests have no ``tune`` parametrization / autotune
  try-except -- nothing to remove.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.elementwise.clamp_max import ClampMaxFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.workload_base import FixtureBase, WorkloadBase


def _device() -> str:
    return get_device_backend().name


class ClampMaxWorkload(WorkloadBase):
    """Two same-broadcast-shape ``randn`` tensors for the clamp_max test."""

    def __init__(self, input_shape: tuple, max_shape: tuple, dtype: torch.dtype):
        self.input_shape = tuple(input_shape)
        self.max_shape = tuple(max_shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = _device()
        x = torch.randn(self.input_shape, dtype=self.dtype, device=device)
        mx = torch.randn(self.max_shape, dtype=self.dtype, device=device)
        return x, mx


class ClampMaxTest(ClampMaxWorkload, TestBase):
    """Reusable test body: ``y = torch.clamp_max(x, max)``."""

    def ref_program(self, x: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
        return torch.clamp_max(x, mx)


class ClampMaxFixture(FixtureBase):
    PARAMS = [
        (
            "input_shape, max_shape, dtype",
            [
                pytest.param(
                    (4_096,),
                    (4_096,),
                    torch.float32,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                ),
                pytest.param(
                    (4_096,), (4_096,), torch.float16, marks=[pytest.mark.smoke]
                ),
                pytest.param(
                    (4_096,), (4_096,), torch.bfloat16, marks=[pytest.mark.smoke]
                ),
                # Full: larger same-shape coverage
                pytest.param(
                    (2_048, 4_096),
                    (2_048, 4_096),
                    torch.float32,
                    marks=[pytest.mark.full],
                ),
            ],
        ),
    ]


@ClampMaxFixture
def test_clamp_max_op(
    input_shape: tuple, max_shape: tuple, dtype: torch.dtype
) -> None:
    test = ClampMaxTest(input_shape, max_shape, dtype)
    op = ClampMaxFwdOp(input=input_shape, max=max_shape, dtype=dtype)
    test.check(op, *test.gen_inputs())


_BROADCAST_PATTERNS = [
    ((2, 64, 128), (1, 1, 128)),   # per-channel broadcast
    ((2, 64, 128), (2, 64, 1)),    # row broadcast
    ((64, 128), (1, 1)),           # scalar broadcast
    ((4, 8), ()),                  # 0-dim upper bound
]


class ClampMaxBroadcastFixture(FixtureBase):
    PARAMS = [
        (
            "input_shape, max_shape",
            [
                pytest.param(
                    in_s,
                    mx_s,
                    marks=pytest.mark.smoke if i == 0 else pytest.mark.full,
                )
                for i, (in_s, mx_s) in enumerate(_BROADCAST_PATTERNS)
            ],
        ),
    ]


@ClampMaxBroadcastFixture
def test_clamp_max_broadcast(input_shape: tuple, max_shape: tuple) -> None:
    device = _device()
    x = torch.randn(input_shape, dtype=torch.float16, device=device)
    mx = torch.randn(max_shape, dtype=torch.float16, device=device)
    op = ClampMaxFwdOp(input=input_shape, max=max_shape, dtype=torch.float16)
    ref = torch.clamp_max(x, mx)
    with torch.no_grad():
        out = op(x, mx)
    assert tuple(out.shape) == tuple(ref.shape)
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_clamp_max_init_signature_pytorch_aligned() -> None:
    import inspect

    init_params = list(inspect.signature(ClampMaxFwdOp.__init__).parameters.keys())
    fwd_params = list(inspect.signature(ClampMaxFwdOp.forward).parameters.keys())
    # __init__: self, input, max, dtype, ...
    assert init_params[1:4] == ["input", "max", "dtype"], init_params
    # forward: self, input, max
    assert fwd_params[1:] == ["input", "max"], fwd_params


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_max_nan_propagation(dtype: torch.dtype) -> None:
    """ClampMaxFwdOp must match torch.clamp_max NaN semantics.

    If ``x`` or ``max`` is NaN at a position, the output there is NaN.
    """
    device = _device()
    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device=device, dtype=dtype)
    mx = torch.tensor([1.0, 1.0, 1.0, float("nan")], device=device, dtype=dtype)

    ref = torch.clamp_max(x, mx)
    op = ClampMaxFwdOp(input=(4,), max=(4,), dtype=dtype)
    with torch.no_grad():
        out = op(x, mx)
    torch.testing.assert_close(out, ref, equal_nan=True, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.int32, marks=pytest.mark.smoke),
        pytest.param(torch.complex64, marks=pytest.mark.smoke),
        pytest.param(torch.float8_e4m3fn, marks=pytest.mark.smoke),
    ],
)
def test_clamp_max_rejects_unsupported_dtype(dtype: torch.dtype) -> None:
    """Clamp ops reject dtypes outside the float16/bfloat16/float32 contract."""
    with pytest.raises(ValueError, match="does not support dtype"):
        ClampMaxFwdOp(input=(16,), max=(16,), dtype=dtype)


@pytest.mark.smoke
def test_clamp_max_eval_roofline() -> None:
    """eval_roofline() matches the GPU formula (clamp_max_fwd_roofline).

    flops = N_total; bytes = 3 * N_total * elem_bytes (read input + read max
    + write output, all post-broadcast).
    """
    input_shape, max_shape = (64, 128), (1, 128)
    dtype = torch.float32
    op = ClampMaxFwdOp(input=input_shape, max=max_shape, dtype=dtype)
    n_total = 64 * 128
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == n_total, f"flops {flops} != N = {n_total}"
    assert mem_bytes == 3 * n_total * elem_bytes, (
        f"bytes {mem_bytes} != 3 * N * elem_bytes = {3 * n_total * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
