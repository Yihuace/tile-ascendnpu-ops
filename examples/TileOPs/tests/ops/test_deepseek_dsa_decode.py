"""Correctness tests for DeepSeekSparseAttentionDecodeWithKVCacheFwdOp (NPU).

Adapted from TileOPs ``tests/ops/attention/test_deepseek_dsa_decode.py``.
This GPU test file is op-specific (not shared), so the whole flow is ported
-- the masked sparse-softmax ``ref_program`` is preserved verbatim.

Adaptations (T1-T4):

- T1: the reference constructs its index/mask tensors on ``q.device`` instead
  of the hard-coded ``"cuda"`` (device-agnostic).
- T3: the ``tune`` fixture param and the ``tune=`` op kwarg are removed;
  the first two cases are marked ``smoke`` (fp16 / bf16 coverage) per the NPU
  smoke convention.
"""

import pytest
import torch

from tileops.ops.attention.deepseek_dsa_decode import DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.attention import DsaDecodeWorkload
from tileops.workloads.workload_base import FixtureBase


class DsaDecodeTest(DsaDecodeWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, kv: torch.Tensor,
                    indices: torch.Tensor) -> torch.Tensor:
        q = q.float()
        kv = kv.float()
        indices = indices.transpose(1, 2)
        b, sq, h, dim_q = q.shape
        b, sk, g, _ = kv.shape
        q_start_index_s = self.q_start_index_s
        if self.q_start_index_s is None:
            q_start_index_s = sk * self.stride_kv - sq

        assert kv.shape[-1] == self.dim + self.dim_tail, 'you should assign dim otherwise'
        dim = self.dim
        k = kv
        v = kv[..., :dim]

        b, _, _, dim_v = v.shape
        g_index = g
        h_index = h // g
        # T1: device-agnostic (GPU hard-coded "cuda").
        compressed_causal_mask = torch.arange(
            q_start_index_s, sq + q_start_index_s, dtype=torch.int32,
            device=q.device).view(-1, 1) >= torch.arange(
                self.stride_kv - 1,
                sk * self.stride_kv,
                self.stride_kv,
                dtype=torch.int32,
                device=q.device).view(1, -1)

        mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_causal_mask.view(1, 1, sq, sk)
        mask[:, :, :self.stride_kv - 1, 0] = True
        mask = mask.view(b, g_index, 1, sq, sk)

        q = q.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", q, k)
        sm_scale = dim_q**-0.5 if self.sm_scale is None else self.sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk)
        p = p.view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        # NPU port fix (T5): the GPU reference hard-coded ``.to(torch.float16)``
        # because the GPU fixture only exercised fp16 -- it is wrong once the
        # NPU fixture adds a bf16 case.  The op contract (manifest outputs
        # ``o: same_as(q)``) / the kernel golden both round to the input dtype,
        # so the reference must too.
        return o.to(self.dtype)


class DsaDecodeFixture(FixtureBase):
    # T3: the GPU "..., sm_scale, dtype, tune" param list drops the trailing
    # ``tune`` flag; the first two cases are marked smoke (fp16 / bf16).
    PARAMS = [
        ("batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, "
         "q_start_index_s, sm_scale, dtype", [
             pytest.param(
                 1, 128, 128, 256, 512, 64, 256, 1, 1, 128, None, torch.float16,
                 marks=pytest.mark.smoke,
             ),
             pytest.param(
                 1, 128, 64, 512, 512, 64, 256, 1, 1, 64, None, torch.bfloat16,
                 marks=pytest.mark.smoke,
             ),
             pytest.param(
                 1, 128, 1024, 2048, 512, 64, 2048, 1, 1, 1024, None, torch.float16,
                 marks=pytest.mark.full, id="full-fp16-mainstream",
             ),
             pytest.param(
                 1, 128, 512, 4096, 512, 64, 1024, 1, 1, 512, None, torch.float16,
                 marks=pytest.mark.full, id="full-fp16-longer-kv",
             ),
         ]),
    ]


@DsaDecodeFixture
def test_sparse_mla_decode(batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                           dim_tail: int, topk: int, stride_kv: int, heads_kv: int,
                           q_start_index_s: int, sm_scale: float, dtype: torch.dtype) -> None:
    test = DsaDecodeTest(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype)
    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype)
    # NPU port fix (T5): atol raised 3e-4 -> 1e-2, the project NPU blocking
    # gate standard Stage 3 was validated against.  Rationale/evidence: fp16 /
    # bf16 outputs are O(1), so the last-bit rounding floor alone is ~1 fp16 ulp
    # (4.9e-4-9.8e-4) and ~1 bf16 ulp (7.8e-3); the unavoidable residual
    # (kernel vs reference summation grouping) is <= 1 ulp with dtype-matched
    # reference, and 3e-4 is below that floor.  rtol is unchanged.
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-5)


@pytest.mark.smoke
def test_sparse_mla_decode_eval_roofline() -> None:
    batch, heads, seq_len, seq_len_kv = 1, 128, 128, 256
    dim, dim_tail, topk, heads_kv = 512, 64, 256, 1
    elem_bytes = torch.float16.itemsize
    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(
        batch, heads, seq_len, seq_len_kv, dim, dim_tail, topk, 1, heads_kv, 128,
        dtype=torch.float16)
    flops, mem_bytes = op.eval_roofline()
    expected_flops = 2 * batch * seq_len * heads * topk * (2 * dim + dim_tail)
    q_elems = batch * seq_len * heads * (dim + dim_tail)
    kv_elems = batch * seq_len_kv * heads_kv * (dim + dim_tail)
    o_elems = batch * seq_len * heads * dim
    index_bytes = batch * seq_len * heads_kv * topk * 4
    assert flops == expected_flops, f"flops {flops} != {expected_flops}"
    assert mem_bytes == (q_elems + kv_elems + o_elems) * elem_bytes + index_bytes, (
        f"bytes {mem_bytes} != (q + kv + o) * elem_bytes + index_bytes"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
