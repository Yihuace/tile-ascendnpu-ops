"""Correctness tests for FP8LightningIndexerOp (NPU).

Ported from TileOPs ``tests/ops/test_fp8_lightning_indexer.py``.  This GPU test
file is op-specific (not shared), so the whole flow is carried over: the
``ref_program`` (einsum score -> relu -> weighted grouped sum -> window mask)
and the finite-mask + correlation comparator are preserved verbatim.

Adaptations (T1-T4):

- T1: ``ref_program``'s ``torch.arange(..., device="cuda")`` and the manual
  tensor creation in the dtype-rejection test resolve the device through
  ``get_device_backend().name``.
- T3: the ``tune`` fixture param and the ``tune=`` op kwarg are removed (NPU
  uses heuristic config selection only).  The GPU test's two params differed
  only by ``tune``; the second is replaced by a distinct shape (larger
  ``seq_len`` and a ``kv_group > 1`` case) so coverage is preserved without a
  duplicate case.
- T4: no autotune try/except exists in the GPU test, so nothing to remove.
"""

from typing import Optional

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.attention_indexing.fp8_lightning_indexer import FP8LightningIndexerOp
from tileops.testing.test_base import TestBase
from tileops.workloads.attention_indexing import FP8LightningIndexerWorkload
from tileops.workloads.workload_base import FixtureBase


class FP8LightningIndexerTest(FP8LightningIndexerWorkload, TestBase):

    @staticmethod
    def _compute_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
        # NPU port note: Ascend910B3 has no fp64, so ``Tensor.double()`` on an
        # NPU tensor is a silent no-op (dtype stays float32).  Reducing a
        # ~3.2M-element correlation in fp32 introduces ~1e-7 relative rounding
        # whose sign is data-dependent, which flips the one-sided
        # ``0 <= 1 - corr`` assertion for a numerically correct op.  Move to
        # CPU first so the reduction runs in the true fp64 precision of the
        # original GPU comparator (tolerance and assertion are unchanged).
        a, b = a.data.cpu().double(), b.data.cpu().double()
        norm_sum = (a * a + b * b).sum()
        return 2 * (a * b).sum() / norm_sum

    @staticmethod
    def _validate_tensor_match(output: torch.Tensor,
                               output_ref: torch.Tensor,
                               tolerance: float = 1e-3) -> None:
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output_ref, tuple):
            output_ref = output_ref[0]

        a_finite = torch.isfinite(output)
        b_finite = torch.isfinite(output_ref)
        assert torch.all(a_finite == b_finite), "Error: isfinite mask mismatch"
        assert torch.isclose(
            output.masked_fill(a_finite, 0),
            output_ref.masked_fill(b_finite, 0),
            rtol=0,
            atol=0,
            equal_nan=True,
        ).all(), "Error: nonfinite value mismatch"
        output = output.masked_fill(~a_finite, 0)
        output_ref = output_ref.masked_fill(~b_finite, 0)
        correlation = FP8LightningIndexerTest._compute_correlation(output, output_ref)
        difference = 1.0 - correlation
        assert 0 <= difference <= tolerance, \
            f"outputs is not close to outputs_ref, difference: {difference}"

    def ref_program(self, q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                    cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor) -> tuple[torch.Tensor]:
        k = kv
        q = q.float()
        k = k.float()
        batch, seq_len, heads, index_dim = q.shape
        seq_len_kv = self.seq_len_kv
        kv_group = self.kv_group
        heads_per_group = heads // kv_group

        k = k.view(batch, seq_len_kv, kv_group, index_dim)
        q = q.view(batch, seq_len, kv_group, heads_per_group, index_dim)

        # T1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        mask_lo = torch.arange(0, seq_len_kv, device=device)[None, :] >= cu_seqlen_ks[:, None]
        mask_hi = torch.arange(0, seq_len_kv, device=device)[None, :] < cu_seqlen_ke[:, None]
        mask = mask_lo & mask_hi

        score = torch.einsum("bsghd,bngd->bghsn", q, k)
        weights = weights.view(seq_len, kv_group, heads_per_group)
        weights = weights.permute(1, 2, 0).unsqueeze(0).unsqueeze(-1)
        score = score.relu() * weights
        logits = score.sum(dim=2)
        logits = logits.permute(0, 2, 3, 1)
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1)
        logits = logits.masked_fill(~mask_expanded, float("-inf"))
        return (logits,)


class FP8LightningIndexerFixture(FixtureBase):
    # T3: the GPU "..., clean_logits, config, tune" param list drops the
    # trailing ``tune`` flag; the first case is smoke.
    PARAMS = [
        ("batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, config", [
            pytest.param(1, 4096, 32, 64, 8192, 1, True, None, marks=pytest.mark.smoke),
            pytest.param(1, 8192, 32, 64, 8192, 1, True, None, marks=pytest.mark.full),
            pytest.param(1, 4096, 32, 64, 8192, 2, True, None, marks=pytest.mark.full),
        ]),
    ]


@FP8LightningIndexerFixture
def test_indexer(batch: int, seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
                 kv_group: int, clean_logits: bool, config: Optional[dict]) -> None:
    test = FP8LightningIndexerTest(batch, seq_len, heads, index_dim, seq_len_kv, kv_group,
                                  clean_logits, config)
    op = FP8LightningIndexerOp(clean_logits=clean_logits, config=config)
    test.check(op, *test.gen_inputs(), compare=FP8LightningIndexerTest._validate_tensor_match)


@pytest.mark.smoke
def test_indexer_rejects_bf16_inputs_with_external_scale() -> None:
    op = FP8LightningIndexerOp()
    device = get_device_backend().name
    batch, seq_len, heads, index_dim, seq_len_kv, kv_group = 1, 8, 4, 16, 16, 1
    index_q = torch.randn(
        batch, seq_len, heads, index_dim, device=device, dtype=torch.bfloat16)
    index_k = torch.randn(
        batch, seq_len_kv, kv_group, index_dim, device=device, dtype=torch.bfloat16)
    weights = torch.randn(seq_len, heads, device=device, dtype=torch.float32)
    cu_seqlen_ks = torch.zeros(seq_len, device=device, dtype=torch.int32)
    cu_seqlen_ke = torch.full((seq_len,), seq_len_kv, device=device, dtype=torch.int32)
    index_k_scale = torch.ones(batch, seq_len_kv, kv_group, device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match="float8_e4m3fn"):
        op(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, index_k_scale)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
