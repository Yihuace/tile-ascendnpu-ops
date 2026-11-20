"""Correctness tests for BmmFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_bmm.py`` -- the BmmFwdOp subset only.
The shared GPU file also covers ``BmmFp8Op`` (and its fp8 fixtures), which is
not migrated; those tests are intentionally omitted (importing ``BmmFp8Op``
would fail on NPU).

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) instead
  of hard-coded ``device="cuda"``.
- T2: the GPU BmmFwd tests carry no ``torch.cuda`` availability guard --
  nothing to remove.
- T3: the ``tune`` parametrization dimension is removed.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.bmm.bmm import BmmFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.bmm import BmmWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


class BmmTest(BmmWorkload, TestBase):
    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)


class BmmFixture(FixtureBase):
    PARAMS = [
        ("batch, m, n, k, dtype", [
            pytest.param(
                4, 128, 128, 128, torch.float16,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-fp16-b4-128",
            ),
            pytest.param(
                4, 128, 128, 128, torch.bfloat16,
                marks=pytest.mark.smoke,
                id="smoke-bf16-b4-128",
            ),
            pytest.param(
                8, 512, 512, 512, torch.float16,
                marks=pytest.mark.full,
                id="full-fp16-b8-512",
            ),
            pytest.param(
                8, 512, 512, 512, torch.bfloat16,
                marks=pytest.mark.full,
                id="full-bf16-b8-512",
            ),
            pytest.param(
                16, 256, 256, 256, torch.float16,
                marks=pytest.mark.full,
                id="full-fp16-b16-256",
            ),
            pytest.param(
                1, 1024, 1024, 1024, torch.float16,
                marks=pytest.mark.full,
                id="full-fp16-b1-1k",
            ),
            pytest.param(
                32, 128, 512, 128, torch.float16,
                marks=pytest.mark.full,
                id="full-fp16-b32-mha-qk",
            ),
            pytest.param(
                8, 128, 128, 2048, torch.float16,
                marks=pytest.mark.full,
                id="full-fp16-b8-mha-pv",
            ),
            pytest.param(
                8, 128, 128, 2048, torch.bfloat16,
                marks=pytest.mark.full,
                id="full-bf16-b8-mha-pv",
            ),
            pytest.param(
                32, 256, 256, 1024, torch.bfloat16,
                marks=pytest.mark.full,
                id="full-bf16-b32-moe",
            ),
            pytest.param(
                4, 200, 300, 128, torch.float16,
                marks=pytest.mark.full,
                id="full-fp16-b4-mn-nonaligned",
            ),
        ]),
    ]


@BmmFixture
def test_bmm(batch: int, m: int, n: int, k: int, dtype: torch.dtype) -> None:
    test = BmmTest(batch, m, n, k, dtype)
    op = BmmFwdOp()
    if dtype == torch.float16:
        tolerances = {"atol": 1e-3, "rtol": 1e-3}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@pytest.mark.smoke
def test_bmm_batch_mismatch_raises() -> None:
    op = BmmFwdOp()
    a = torch.randn(4, 16, 16, device=_device(), dtype=torch.float16)
    b = torch.randn(5, 16, 16, device=_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match="batch dim mismatch"):
        op(a, b)


@pytest.mark.smoke
def test_bmm_contraction_mismatch_raises() -> None:
    op = BmmFwdOp()
    a = torch.randn(4, 16, 32, device=_device(), dtype=torch.float16)
    b = torch.randn(4, 16, 16, device=_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match="contraction dim mismatch"):
        op(a, b)


@pytest.mark.smoke
def test_bmm_rank_mismatch_raises() -> None:
    op = BmmFwdOp()
    a = torch.randn(16, 16, device=_device(), dtype=torch.float16)
    b = torch.randn(4, 16, 16, device=_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match="strict 3D"):
        op(a, b)


@pytest.mark.smoke
def test_bmm_dtype_mismatch_raises() -> None:
    op = BmmFwdOp()
    a = torch.randn(4, 16, 16, device=_device(), dtype=torch.float16)
    b = torch.randn(4, 16, 16, device=_device(), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        op(a, b)


@pytest.mark.smoke
def test_bmm_b_dtype_change_after_valid_call_raises() -> None:
    op = BmmFwdOp()
    a = torch.randn(4, 16, 16, device=_device(), dtype=torch.float16)
    b_ok = torch.randn(4, 16, 16, device=_device(), dtype=torch.float16)
    op(a, b_ok)  # populate the fast path
    b_bad = torch.randn(4, 16, 16, device=_device(), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        op(a, b_bad)


@pytest.mark.smoke
def test_bmm_k_not_multiple_of_16_raises() -> None:
    """K must be a multiple of 16 (manifest shape_rules + op precondition)."""
    op = BmmFwdOp()
    a = torch.randn(4, 16, 24, device=_device(), dtype=torch.float16)
    b = torch.randn(4, 24, 16, device=_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match="multiple of 16"):
        op(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
