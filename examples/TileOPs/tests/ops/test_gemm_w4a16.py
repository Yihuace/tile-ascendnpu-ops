"""Correctness tests for GemmW4A16Op (NPU).

Per-op test file: the shared GPU ``test_gemm.py`` was split so that only this
migration unit's cases run when ``GemmW4A16Op`` is integrated (the shared file
also carried ``GemmOp`` cases, which belong to a separate migration unit, now
in ``test_gemm.py``).

Adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (the workload's
  ``gen_inputs``); there is no hard-coded ``device="cuda"`` left.
- T2: no ``torch.cuda`` availability guard was present -- nothing to remove.
- T3: the ``tune`` fixture param and the ``tune=`` op kwarg are removed.
"""

import pytest
import torch

from tileops.ops.gemm.gemm_w4a16 import GemmW4A16Op
from tileops.testing.test_base import TestBase
from tileops.workloads.gemm import GemmW4A16Workload, quantize_weight_int4
from tileops.workloads.workload_base import FixtureBase


class GemmW4A16Test(GemmW4A16Workload, TestBase):
    def ref_program(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        del packed_weight, weight_scale, weight_zero
        return torch.matmul(activation, self.dequantized_weight.T)


class GemmW4A16Fixture(FixtureBase):
    PARAMS = [
        ("m, n, k, dtype", [
            pytest.param(
                64, 64, 128, torch.float16,
                marks=pytest.mark.smoke,
                id="smoke-w4a16-square",
            ),
            pytest.param(
                128, 256, 256, torch.float16,
                marks=pytest.mark.smoke,
                id="smoke-w4a16-rect",
            ),
            pytest.param(
                1, 512, 512, torch.float16,
                marks=pytest.mark.full,
                id="full-w4a16-m1",
            ),
            pytest.param(
                16, 1024, 1024, torch.float16,
                marks=pytest.mark.full,
                id="full-w4a16-m16",
            ),
        ]),
    ]


@GemmW4A16Fixture
def test_gemm_w4a16(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    test = GemmW4A16Test(m, n, k, dtype)
    op = GemmW4A16Op()
    test.check(op, *test.gen_inputs(), atol=7e-2, rtol=5e-2)


@pytest.mark.smoke
def test_quantize_weight_int4_keeps_one_sided_groups_in_range() -> None:
    weight = torch.tensor(
        [
            [0.25, 0.50, 0.75, 1.00],
            [-1.00, -0.75, -0.50, -0.25],
        ],
        dtype=torch.float32,
    )

    _, scale, zero, dequantized = quantize_weight_int4(weight, group_size=4)

    assert torch.equal(zero, torch.tensor([[0], [15]], dtype=torch.uint8))
    assert torch.all(scale > 0)
    torch.testing.assert_close(dequantized[0].max(), weight[0].max())
    torch.testing.assert_close(dequantized[1].min(), weight[1].min())


@pytest.mark.smoke
def test_gemm_w4a16_rejects_invalid_metadata_shapes() -> None:
    test = GemmW4A16Test(64, 64, 128, torch.float16)
    activation, packed_weight, weight_scale, weight_zero = test.gen_inputs()
    op = GemmW4A16Op()

    with pytest.raises(ValueError, match="weight_scale must have shape"):
        op(activation, packed_weight, weight_scale[:, :0], weight_zero)

    with pytest.raises(ValueError, match="packed_weight shape mismatch"):
        op(activation, packed_weight[:, :-1], weight_scale, weight_zero)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
