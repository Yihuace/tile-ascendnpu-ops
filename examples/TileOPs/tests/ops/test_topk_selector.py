"""Correctness tests for TopkSelectorOp (NPU).

Ported from the GPU TileOPs test for TopkSelectorOp.  The comparison is a SET
comparison -- top-k indices are unordered, and the in-repo contract compares them
as sets.  The inputs are DISTINCT along each row on purpose: with ties at the
k-th boundary, which of the tied elements a top-k returns is undefined, so a tie
case would test an undefined choice rather than the kernel.

The scale deviation from the GPU manifest's full workload is documented in
``tileops/manifest/attention_indexing.yaml``.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.attention_indexing.topk_selector import TopkSelectorOp
from tileops.testing.test_base import TestBase
from tileops.workloads.attention_indexing import TopkSelectorWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


def _set_equal(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Set comparison, matching the in-repo ``_set_compare`` contract."""
    if got.shape != want.shape:
        return False
    for i in range(got.shape[0]):
        if set(got[i].tolist()) != set(want[i].tolist()):
            return False
    return True


class TopkSelectorTest(TopkSelectorWorkload, TestBase):
    """Test fixture: a (M, N) row matrix with distinct values per row."""

    def __init__(self, shape: tuple, dtype: torch.dtype, k: int):
        super().__init__(shape, dtype, k)
        self.k_ = k

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.topk(x.detach().float().cpu(), self.k_, dim=-1).indices


class TopkSelectorFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype, k, tile",
            [
                pytest.param((8, 256), torch.float32, 8, None,
                             marks=[pytest.mark.smoke]),
                # k == 1 and k == N: the two edges of the selection range
                pytest.param((4, 512), torch.float32, 1, None,
                             marks=[pytest.mark.smoke]),
                pytest.param((4, 512), torch.float32, 512, None,
                             marks=[pytest.mark.smoke]),
                pytest.param((16, 1024), torch.float32, 16, None,
                             marks=[pytest.mark.full]),
                pytest.param((8, 4096), torch.float32, 64, None,
                             marks=[pytest.mark.full]),
                # streamed path: the row is processed in pieces smaller than N
                pytest.param((8, 8192), torch.float32, 64, 1024,
                             marks=[pytest.mark.full]),
            ],
        ),
    ]


@TopkSelectorFixture
def test_topk_selector_op(shape: tuple, dtype: torch.dtype, k: int,
                          tile) -> None:
    test = TopkSelectorTest(shape, dtype, k)
    op = TopkSelectorOp(M=shape[0], N=shape[1], k=k, tile=tile)
    (x,) = test.gen_inputs()
    got = op(x).cpu()
    want = test.ref_program(x)
    assert _set_equal(got, want), (
        f"index sets differ (M,N,k,tile)={(shape[0], shape[1], k, tile)}; "
        f"got={sorted(got[0].tolist())[:8]} want={sorted(want[0].tolist())[:8]}"
    )


@pytest.mark.smoke
def test_topk_selector_ties_are_a_set() -> None:
    """With MANY ties the SET contract is still satisfiable.

    Every row is all-equal here, so any k of the N indices is admissible; the
    assertion is the weaker but well-defined one: the returned indices are k
    DISTINCT values inside the row.
    """
    M, N, k = 2, 256, 8
    x = torch.ones(M, N, dtype=torch.float32).to(_device())
    op = TopkSelectorOp(M=M, N=N, k=k)
    got = op(x).cpu()
    for i in range(M):
        vals = got[i].tolist()
        assert len(set(vals)) == k, f"row {i}: expected {k} distinct indices, got {vals}"
        assert all(0 <= v < N for v in vals), f"row {i}: index out of range: {vals}"


@pytest.mark.smoke
def test_topk_selector_rejects_bad_shape() -> None:
    op = TopkSelectorOp(M=4, N=64, k=4)
    with pytest.raises(ValueError, match="Expected input shape"):
        op(torch.randn(4, 32, dtype=torch.float32).to(_device()))


@pytest.mark.smoke
def test_topk_selector_eval_roofline() -> None:
    M, N, k, rounds = 8, 256, 8, 40
    op = TopkSelectorOp(M=M, N=N, k=k, rounds=rounds)
    flops, mem_bytes = op.eval_roofline()
    assert flops == rounds * M * N, f"flops {flops} != rounds*M*N"
    assert mem_bytes == 4 * M * N + 4 * M * k, f"bytes {mem_bytes} unexpected"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
