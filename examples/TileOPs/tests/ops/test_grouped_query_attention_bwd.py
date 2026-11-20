"""Correctness tests for GroupedQueryAttentionBwdOp (NPU).

Adapted from TileOPs ``tests/ops/attention/test_gqa.py`` -- the
``GroupedQueryAttentionBwd`` subset only.  The GPU test file is shared with
many other GQA/MHA ops (forward, prefill, varlen, decode, paged, sliding
window -- none of which are migrated here), so only the backward test class /
fixture / test function and the backward workload are ported.

The backward test compares ``dq``/``dk``/``dv`` against a PyTorch SDPA
reference (``enable_gqa=True``) that backprops through the forward pass.

Adaptations (T1-T4):

- T2: the GPU ``ref_program`` forced the CUDA ``FLASH_ATTENTION`` SDP
  backend via ``sdpa_kernel``; on NPU the default SDPA dispatch is used.
- T3: the ``tune`` fixture param and the ``tune=`` op kwarg are removed
  (NPU uses heuristic config selection only).
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.ops.attention.grouped_query_attention_bwd import GroupedQueryAttentionBwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.attention import GroupedQueryAttentionBwdWorkload
from tileops.workloads.workload_base import FixtureBase


class GroupedQueryAttentionBwdTest(GroupedQueryAttentionBwdWorkload, TestBase):
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
            q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True
        )
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(grad_output)
        return q.grad, k.grad, v.grad


class GroupedQueryAttentionBwdFixture(FixtureBase):
    # T3: the GPU "batch, seq_len, heads, heads_kv, dim, causal, dtype, tune"
    # param list drops the trailing ``tune`` flag.
    PARAMS = [
        (
            "batch, seq_len, heads, heads_kv, dim, causal, dtype",
            [
                pytest.param(
                    1,
                    1024,
                    8,
                    4,
                    64,
                    False,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-bwd-fp16",
                ),
                pytest.param(
                    1,
                    1024,
                    8,
                    4,
                    64,
                    False,
                    torch.bfloat16,
                    marks=[pytest.mark.smoke],
                    id="smoke-bwd-bf16",
                ),
                pytest.param(
                    4,
                    2048,
                    64,
                    4,
                    128,
                    False,
                    torch.float16,
                    marks=[pytest.mark.full],
                    id="full-bwd-fp16",
                ),
                pytest.param(
                    4,
                    2048,
                    64,
                    4,
                    128,
                    False,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-bwd-bf16",
                ),
            ],
        ),
    ]


@GroupedQueryAttentionBwdFixture
def test_gqa_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    test = GroupedQueryAttentionBwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_bwd_eval_roofline() -> None:
    batch, seq_len, heads, heads_kv, dim = 2, 512, 8, 4, 64
    elem_bytes = torch.float16.itemsize
    for is_causal in (True, False):
        op = GroupedQueryAttentionBwdOp(
            batch, heads, heads_kv, seq_len, dim, is_causal, torch.float16
        )
        flops, mem_bytes = op.eval_roofline()
        full_flops = 10 * batch * heads * seq_len * seq_len * dim
        expected_flops = full_flops // 2 if is_causal else full_flops
        assert flops == expected_flops, f"flops {flops} != {expected_flops} (is_causal={is_causal})"
        expected_bytes = batch * (3 * heads + 4 * heads_kv) * seq_len * dim * elem_bytes
        assert mem_bytes == expected_bytes, (
            f"bytes {mem_bytes} != B * (3H + 4H_kv) * S * D * elem_bytes"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
