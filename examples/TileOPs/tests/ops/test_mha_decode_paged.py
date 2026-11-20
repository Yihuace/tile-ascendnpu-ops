"""Correctness tests for MultiHeadAttentionDecodePagedWithKVCacheFwdOp (NPU).

Adapted from TileOPs ``tests/ops/attention/test_mha_decode_paged.py``.  This
GPU test file is op-specific (not shared), so the whole flow is ported --
paged KV reassembly ``ref_program`` and the max-diff / cosine-similarity
comparator are preserved verbatim.

Adaptations (T1-T4):

- T1: no manual tensor creation needs a device switch -- the reference
  reassembles K/V onto ``q.device`` (device-agnostic).
- T3: the ``tune`` fixture param and the ``tune=`` op kwarg are removed;
  the first two cases are marked ``smoke`` (fp16 / bf16 coverage) per the
  NPU smoke convention.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from tileops.ops.attention.mha_decode_paged import MultiHeadAttentionDecodePagedWithKVCacheFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.attention import MhaDecodePagedWorkload
from tileops.workloads.workload_base import FixtureBase


class MhaDecodePagedTest(MhaDecodePagedWorkload, TestBase):

    def _maxdiff_cosine_compare(self, output: torch.Tensor, output_ref: torch.Tensor, atol: float = 0.001) -> None:
        """Compare using max-diff and cosine similarity."""
        if isinstance(output, (tuple, list)):
            output = output[0]
        max_diff = (output - output_ref).abs().max().item()
        assert max_diff < atol, (
            f"max diff {max_diff} too large (atol={atol})")
        cos_sim = F.cosine_similarity(
            output.reshape(self.batch, -1), output_ref.reshape(self.batch, -1), dim=-1, eps=1e-8)
        assert cos_sim.min() > 0.99, f"cosine similarity {cos_sim.min().item()} too low"

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        """Reassemble paged K/V to logical layout per batch, then SDPA."""
        batch, seqlen_q, heads, dim = q.shape
        seqlen_kv = k.shape[0]
        out_list = []
        for i_b in range(batch):
            q_b = q[i_b:i_b + 1, :, :, :]
            k_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            v_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            num_pages = math.ceil(real_seqlen_kv[i_b].item() / self.page_size)
            for i_paged in range(num_pages):
                start_pos = block_table[i_b, i_paged].item() * self.page_size
                end_pos = min(start_pos + self.page_size, seqlen_kv)
                page_len = end_pos - start_pos
                k_logical[i_paged * self.page_size:i_paged * self.page_size +
                          page_len, :, :] = k[start_pos:end_pos, :, :]
                v_logical[i_paged * self.page_size:i_paged * self.page_size +
                          page_len, :, :] = v[start_pos:end_pos, :, :]
            k_logical = k_logical[:real_seqlen_kv[i_b].item(), :, :]
            v_logical = v_logical[:real_seqlen_kv[i_b].item(), :, :]
            k_b = k_logical.unsqueeze(0)
            v_b = v_logical.unsqueeze(0)
            q_bhsd = q_b.transpose(1, 2)
            k_bhsd = k_b.transpose(1, 2)
            v_bhsd = v_b.transpose(1, 2)
            out_b = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
            out_b = out_b.transpose(1, 2).contiguous()
            out_list.append(out_b)
        return torch.cat(out_list, dim=0)


class MhaDecodePagedFixture(FixtureBase):
    # T3: the GPU "..., is_causal, dtype, tune" param list drops the trailing
    # ``tune`` flag; the first two cases are marked smoke (fp16 / bf16).
    PARAMS = [
        ("batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype", [
            pytest.param(
                1, 16, 1, 512, 128, 128, False, torch.float16,
                marks=pytest.mark.smoke,
            ),
            pytest.param(
                1, 8, 1, 1024, 64, 256, False, torch.bfloat16,
                marks=pytest.mark.smoke,
            ),
            pytest.param(
                2, 8, 1, 1024, 64, 256, False, torch.float16,
                marks=pytest.mark.full,
            ),
            pytest.param(
                1, 8, 1, 512, 64, 256, False, torch.float16,
                marks=pytest.mark.full,
            ),
        ]),
    ]


@MhaDecodePagedFixture
def test_mha_decode_paged_op(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: torch.dtype,
) -> None:
    test = MhaDecodePagedTest(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        dim=dim,
        page_size=page_size,
        is_causal=is_causal,
        dtype=dtype,
    )
    test.check(op, *test.gen_inputs(), compare=test._maxdiff_cosine_compare)


@pytest.mark.smoke
def test_mha_decode_paged_eval_roofline() -> None:
    batch, heads, seqlen_q, seqlen_kv, dim, page_size = 2, 8, 1, 1024, 64, 256
    elem_bytes = torch.float16.itemsize
    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, False, torch.float16
    )
    flops, mem_bytes = op.eval_roofline()
    expected_flops = 4 * batch * heads * seqlen_q * seqlen_kv * dim
    q_elems = batch * seqlen_q * heads * dim
    kv_elems = seqlen_kv * heads * dim
    metadata_bytes = batch * 4 + batch * max(1, (seqlen_kv + page_size - 1) // page_size) * 4
    assert flops == expected_flops, f"flops {flops} != {expected_flops}"
    assert mem_bytes == (q_elems + 2 * kv_elems + q_elems) * elem_bytes + metadata_bytes, (
        f"bytes {mem_bytes} != (q + 2*kv + o) * elem_bytes + metadata"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
