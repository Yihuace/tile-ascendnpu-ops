"""Correctness tests for MultiHeadAttentionFwdOp / MultiHeadAttentionBwdOp (NPU).

Adapted from TileOPs ``tests/ops/attention/test_mha.py`` -- the MhaFwd and
MhaBwd subsets.  The H200-specific
``test_mha_fwd_preserves_gqa_square_dense_fast_path`` test is not ported: the
GPU square dense fast path (``is_h200``) is an arch specialization that
collapses on NPU (K5).

Adaptations (T1-T4):

- T2: the GPU ``ref_program`` forced the CUDA ``FLASH_ATTENTION`` SDP
  backend via ``sdpa_kernel``; on NPU the default SDPA dispatch is used (both
  the forward and backward reference programs).
- T3: the ``tune`` fixture param and the ``tune=`` op kwarg are removed
  (the ``full-fwd-bf16-tuned`` / ``full-bwd-bf16-tuned`` cases become
  ``full-fwd-bf16`` / ``full-bwd-bf16``).
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.ops.attention.multi_head_attention import (
    MultiHeadAttentionBwdOp,
    MultiHeadAttentionFwdOp,
)
from tileops.testing.test_base import TestBase
from tileops.workloads.attention import MhaBwdWorkload, MhaFwdWorkload
from tileops.workloads.workload_base import FixtureBase


class MhaFwdTest(MhaFwdWorkload, TestBase):
    def ref_program(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        # T2: GPU forced SDPBackend.FLASH_ATTENTION (CUDA); NPU uses the
        # default SDPA dispatch.
        output_bhsd = F.scaled_dot_product_attention(
            q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal
        )
        output = output_bhsd.transpose(1, 2).contiguous()
        return output, None  # do not check lse


class MhaFwdFixture(FixtureBase):
    # T3: the GPU "batch, seq_len, heads, dim, causal, dtype, tune" param
    # list drops the trailing ``tune`` flag.
    PARAMS = [
        (
            "batch, seq_len, heads, dim, causal, dtype",
            [
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-fwd-fp16",
                ),
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.bfloat16,
                    marks=[pytest.mark.smoke],
                    id="smoke-fwd-bf16",
                ),
                pytest.param(
                    16,
                    2048,
                    16,
                    128,
                    False,
                    torch.float16,
                    marks=[pytest.mark.full],
                    id="full-fwd-fp16",
                ),
                pytest.param(
                    4,
                    4096,
                    16,
                    128,
                    False,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-fwd-bf16",
                ),
            ],
        ),
    ]


@MhaFwdFixture
def test_mha_fwd(
    batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype
) -> None:
    test = MhaFwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, causal, dtype)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_mha_fwd_dispatches_to_gqa_kernel() -> None:
    op = MultiHeadAttentionFwdOp(1, 8, 128, 64, False, torch.float16)
    assert op.kernel.__class__.__name__.startswith("GQA")


@pytest.mark.smoke
def test_mha_fwd_eval_roofline() -> None:
    batch, seq_len, heads, dim = 2, 512, 8, 64
    elem_bytes = torch.float16.itemsize
    for is_causal in (True, False):
        op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, is_causal, torch.float16)
        flops, mem_bytes = op.eval_roofline()
        full_flops = 4 * batch * heads * seq_len * seq_len * dim
        expected_flops = full_flops // 2 if is_causal else full_flops
        q_elems = batch * seq_len * heads * dim
        assert flops == expected_flops, f"flops {flops} != {expected_flops} (is_causal={is_causal})"
        assert mem_bytes == 2 * (q_elems + q_elems) * elem_bytes, (
            f"bytes {mem_bytes} != 2 * (q + kv) * elem_bytes"
        )


# ===========================================================================
# MultiHeadAttentionBwdOp -- backward tests (ported from the GPU MhaBwd subset)
# ===========================================================================


class _FakeLegacyMhaBwdKernel(Kernel):
    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()

    def forward(self, *args: object, **kwargs: object) -> object:
        return None


class MhaBwdTest(MhaBwdWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        grad_output: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        # T2: GPU forced SDPBackend.FLASH_ATTENTION (CUDA); NPU uses the
        # default SDPA dispatch.
        output_bhsd = F.scaled_dot_product_attention(
            q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal
        )
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(grad_output)
        return q.grad, k.grad, v.grad


class MhaBwdFixture(FixtureBase):
    # T3: the GPU "batch, seq_len, heads, dim, causal, dtype, tune" param
    # list drops the trailing ``tune`` flag.
    PARAMS = [
        (
            "batch, seq_len, heads, dim, causal, dtype",
            [
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.float16,
                    marks=[pytest.mark.smoke],
                    id="smoke-bwd-fp16",
                ),
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.bfloat16,
                    marks=[pytest.mark.smoke],
                    id="smoke-bwd-bf16",
                ),
                pytest.param(
                    16,
                    2048,
                    16,
                    128,
                    False,
                    torch.float16,
                    marks=[pytest.mark.full],
                    id="full-bwd-fp16-large",
                ),
                pytest.param(
                    4,
                    4096,
                    16,
                    128,
                    False,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-bwd-bf16",
                ),
            ],
        ),
    ]


@pytest.mark.smoke
def test_mha_bwd_rejects_legacy_kernel_map_keys() -> None:
    with pytest.raises(ValueError, match="legacy MHA backward kernel_map keys"):
        MultiHeadAttentionBwdOp(
            batch=1,
            heads=8,
            seq_len=128,
            dim=64,
            is_causal=False,
            dtype=torch.float16,
            kernel_map={"mha_bwd_kernel": _FakeLegacyMhaBwdKernel},
        )


@MhaBwdFixture
def test_mha_bwd(
    batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype
) -> None:
    test = MhaBwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, causal, dtype)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_mha_bwd_eval_roofline() -> None:
    batch, seq_len, heads, dim = 2, 512, 8, 64
    elem_bytes = torch.float16.itemsize
    for is_causal in (True, False):
        op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, is_causal, torch.float16)
        flops, mem_bytes = op.eval_roofline()
        full_flops = 10 * batch * heads * seq_len * seq_len * dim
        expected_flops = full_flops // 2 if is_causal else full_flops
        assert flops == expected_flops, f"flops {flops} != {expected_flops} (is_causal={is_causal})"
        expected_bytes = batch * 7 * heads * seq_len * dim * elem_bytes
        assert mem_bytes == expected_bytes, f"bytes {mem_bytes} != 7 * B * H * S * D * elem_bytes"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
