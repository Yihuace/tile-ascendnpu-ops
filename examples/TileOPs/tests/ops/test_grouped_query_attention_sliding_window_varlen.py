"""Correctness tests for GroupedQueryAttentionSlidingWindowVarlenFwdOp (NPU).

Adapted from TileOPs ``tests/ops/attention/test_gqa_sliding_window_varlen.py``
-- this op's ``GroupedQueryAttentionSlidingWindowVarlenFwdTest`` / Fixture /
``test_*`` functions and the
``GroupedQueryAttentionSlidingWindowVarlenFwdWorkload`` only.  The fixed-length
sibling (``GroupedQueryAttentionSlidingWindowFwdOp``) has its own test file and
is not ported here.

Adaptations (T1-T4):

- T1: manual tensor creation uses ``_device()`` (``get_device_backend().name``)
  instead of the hard-coded ``"cuda"``.
- T2: no CUDA-only skip guard; the reference is pure PyTorch (device-agnostic).
- T3: the ``tune`` fixture param and the ``tune=`` op kwarg are removed.
  (fp32 is not a supported dtype for this op, so the smoke cases are fp16/bf16.)
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.attention.grouped_query_attention_sliding_window_varlen import (
    GroupedQueryAttentionSlidingWindowVarlenFwdOp,
)
from tileops.testing.test_base import TestBase
from tileops.workloads.attention import GroupedQueryAttentionSlidingWindowVarlenFwdWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    """T1: device name resolved through the device backend (GPU used "cuda")."""
    return get_device_backend().name


class GroupedQueryAttentionSlidingWindowVarlenFwdTest(
    GroupedQueryAttentionSlidingWindowVarlenFwdWorkload, TestBase
):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        """Pure-PyTorch reference: per-sample masked softmax attention.

        offset = seqlen_k - seqlen_q aligns the causal mask bottom-right
        (FA3 convention).  When seqlen_q == seqlen_k, offset=0 and the mask
        reduces to the standard causal mask.
        """
        groups = self.heads // self.heads_kv
        scale = self.dim**-0.5
        outputs = []

        for i in range(self.batch):
            q_start = cu_seqlens_q[i].item()
            q_end = cu_seqlens_q[i + 1].item()
            kv_start = cu_seqlens_k[i].item()
            kv_end = cu_seqlens_k[i + 1].item()

            q_i = q[q_start:q_end]  # [seqlen_q, heads,    dim]
            k_i = k[kv_start:kv_end]  # [seqlen_k, heads_kv, dim]
            v_i = v[kv_start:kv_end]

            seqlen_q = q_end - q_start
            seqlen_k = kv_end - kv_start
            # offset: aligns causal mask to bottom-right corner
            offset = seqlen_k - seqlen_q

            # Expand KV for GQA
            k_exp = k_i.repeat_interleave(groups, dim=1).float()  # [sk, H, D]
            v_exp = v_i.repeat_interleave(groups, dim=1).float()

            # [H, seqlen_q, seqlen_k]
            scores = (
                torch.matmul(
                    q_i.float().transpose(0, 1),  # [H, sq, D]
                    k_exp.transpose(0, 1).transpose(-2, -1),  # [H, D, sk]
                )
                * scale
            )

            # Build attention mask
            q_pos = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
            k_pos = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
            mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)
            if self.is_causal:
                mask = mask | (k_pos > q_pos + offset)
            if self.wl >= 0:
                mask = mask | (k_pos < q_pos + offset - self.wl)
            if self.wr >= 0:
                mask = mask | (k_pos > q_pos + offset + self.wr)

            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1).nan_to_num()
            out_i = torch.matmul(probs, v_exp.transpose(0, 1))  # [H, sq, D]
            outputs.append(out_i.transpose(0, 1).to(q.dtype))  # [sq, H, D]

        return torch.cat(outputs, dim=0)  # [total_q, H, D]


class GroupedQueryAttentionSlidingWindowVarlenFwdFixture(FixtureBase):
    # T3: the GPU "..., dtype, tune" param list drops the trailing ``tune`` flag.
    # Parameters: (batch, seqlens_q, seqlens_k, heads, heads_kv, dim,
    #              is_causal, wl, wr, dtype)
    PARAMS = [
        (
            "batch, seqlens_q, seqlens_k, heads, heads_kv, dim,"
            " is_causal, wl, wr, dtype",
            [
                # ── Basic correctness (smoke) ─────────────────────────────────
                pytest.param(
                    2,
                    [256, 512],
                    [256, 512],
                    8,
                    2,
                    64,
                    True,
                    -1,
                    -1,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-fwd-fp16",
                ),
                pytest.param(
                    2,
                    [256, 512],
                    [256, 512],
                    8,
                    2,
                    64,
                    True,
                    -1,
                    -1,
                    torch.bfloat16,
                    marks=[pytest.mark.smoke],
                    id="smoke-fwd-bf16",
                ),
                # ── Prefill: seqlen_q == seqlen_k (offset=0) ──────────────────
                pytest.param(
                    2, [256, 512], [256, 512], 8, 2, 64, True, 128, -1,
                    torch.float16, marks=[pytest.mark.full],
                ),
                pytest.param(
                    2, [256, 512], [256, 512], 8, 2, 64, False, -1, -1,
                    torch.float16, marks=[pytest.mark.full],
                ),
                pytest.param(
                    2, [256, 512], [256, 512], 8, 2, 64, False, 64, 64,
                    torch.float16, marks=[pytest.mark.full],
                ),
                # ── KV-cache: seqlen_k > seqlen_q (offset > 0) ────────────────
                pytest.param(
                    2, [64, 128], [256, 512], 8, 2, 64, True, -1, -1,
                    torch.float16, marks=[pytest.mark.full],
                ),
                pytest.param(
                    2, [64, 128], [256, 512], 8, 2, 64, True, 128, -1,
                    torch.float16, marks=[pytest.mark.full],
                ),
                pytest.param(
                    2, [64, 128], [256, 512], 8, 2, 64, False, 64, 64,
                    torch.float16, marks=[pytest.mark.full],
                ),
                # ── bfloat16 ──────────────────────────────────────────────────
                pytest.param(
                    2, [256, 512], [256, 512], 8, 2, 64, False, 64, 64,
                    torch.bfloat16, marks=[pytest.mark.full],
                ),
                # ── GQA ratios ────────────────────────────────────────────────
                pytest.param(
                    2, [256, 512], [256, 512], 8, 8, 64, True, -1, -1,
                    torch.float16, marks=[pytest.mark.full],
                ),
                pytest.param(
                    2, [256, 512], [256, 512], 16, 1, 64, True, -1, -1,
                    torch.float16, marks=[pytest.mark.full],
                ),
                # ── Mixed lengths within batch ────────────────────────────────
                pytest.param(
                    3, [128, 256, 384], [128, 256, 384], 8, 2, 64, True, -1, -1,
                    torch.float16, marks=[pytest.mark.full],
                ),
                # ── Right window only ─────────────────────────────────────────
                pytest.param(
                    2, [256, 512], [256, 512], 8, 2, 64, False, -1, 64,
                    torch.float16, marks=[pytest.mark.full],
                ),
                # ── wl=0 boundary ─────────────────────────────────────────────
                pytest.param(
                    2, [128, 256], [128, 256], 8, 2, 64, True, 0, -1,
                    torch.float16, marks=[pytest.mark.full],
                ),
            ],
        ),
    ]


@GroupedQueryAttentionSlidingWindowVarlenFwdFixture
def test_gqa_sliding_window_varlen_fwd_op(
    batch: int,
    seqlens_q: list[int],
    seqlens_k: list[int],
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    wl: int,
    wr: int,
    dtype: torch.dtype,
) -> None:
    test = GroupedQueryAttentionSlidingWindowVarlenFwdTest(
        batch, seqlens_q, seqlens_k, heads, heads_kv, dim, is_causal, wl, wr, dtype
    )
    op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        is_causal=is_causal,
        window_size_left=wl,
        window_size_right=wr,
        dtype=dtype,
    )
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


class TestGroupedQueryAttentionSlidingWindowVarlenFwdOpMetrics:
    """Unit tests for total_flops / total_memory / eval_roofline correctness."""

    @pytest.mark.smoke
    def test_total_flops_causal_wl0(self):
        """is_causal=True, wl=0: every query attends only to itself (eff_kv=1).

        With offset=0 (seqlen_q == seqlen_k) and wl=0, ``hi = lo = q_pos`` so
        each query contributes 1 attended key → total_attended = total_q.
        """
        B, sq, sk, H, Hkv, D = 2, [128, 128], [128, 128], 8, 2, 64
        op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(
            batch=B, heads=H, heads_kv=Hkv, dim=D, is_causal=True, window_size_left=0
        )
        op.q_lens = sq
        op.k_lens = sk
        total_attended = sum(sq)  # one attended key per query
        expected = 4 * H * total_attended * D
        assert op.total_flops == expected, f"got {op.total_flops}, expected {expected}"

    @pytest.mark.smoke
    def test_total_flops_causal_finite_window(self):
        """is_causal=True, wl=100: window limits the left context per query."""
        B, sq, sk, H, Hkv, D = 1, [256], [256], 8, 2, 64
        wl = 100
        op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(
            batch=B, heads=H, heads_kv=Hkv, dim=D, is_causal=True, window_size_left=wl
        )
        op.q_lens = sq
        op.k_lens = sk
        total_attended = sum(min(wl + 1, q + 1) for q in range(sq[0]))
        expected = 4 * H * total_attended * D
        assert op.total_flops == expected, f"got {op.total_flops}, expected {expected}"

    @pytest.mark.smoke
    def test_total_flops_kvcache_offset(self):
        """offset > 0 (seqlen_k > seqlen_q): bottom-right aligned causal window."""
        B, sq, sk, H, Hkv, D = 1, [64], [128], 8, 2, 64
        wl = 32
        op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(
            batch=B, heads=H, heads_kv=Hkv, dim=D, is_causal=True, window_size_left=wl
        )
        op.q_lens = sq
        op.k_lens = sk
        offset = sk[0] - sq[0]
        total_attended = 0
        for q_pos in range(sq[0]):
            hi = min(q_pos + offset, sk[0] - 1)
            lo = max(0, q_pos + offset - wl)
            total_attended += max(0, hi - lo + 1)
        expected = 4 * H * total_attended * D
        assert op.total_flops == expected, f"got {op.total_flops}, expected {expected}"

    @pytest.mark.smoke
    def test_total_memory_gqa(self):
        """For GQA (heads > heads_kv), Q/O use heads and K/V use heads_kv."""
        B, sq, sk, H, Hkv, D = 2, [256, 512], [256, 512], 8, 2, 64
        op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(
            batch=B, heads=H, heads_kv=Hkv, dim=D, is_causal=True
        )
        op.q_lens = sq
        op.k_lens = sk
        elem = torch.tensor([], dtype=torch.float16).element_size()
        total_q, total_k = sum(sq), sum(sk)
        expected = 2 * (total_q * H * D + total_k * Hkv * D) * elem
        assert op.total_memory == expected, f"got {op.total_memory}, expected {expected}"

    @pytest.mark.smoke
    def test_eval_roofline_matches_metrics(self):
        """eval_roofline() exposes the same arithmetic as total_flops/total_memory."""
        op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(
            batch=2, heads=8, heads_kv=2, dim=64, is_causal=True, window_size_left=32
        )
        op.q_lens = [128, 128]
        op.k_lens = [128, 128]
        assert op.eval_roofline() == (op.total_flops, op.total_memory)


class TestGroupedQueryAttentionSlidingWindowVarlenFwdOpValidation:
    """Early-error validation: __init__ and forward guard-clauses."""

    # ── window_size validation (caught in __init__, no device kernel needed) ──

    @pytest.mark.smoke
    def test_invalid_window_size_left_raises(self):
        with pytest.raises(ValueError, match="window_size_left"):
            GroupedQueryAttentionSlidingWindowVarlenFwdOp(
                batch=1, heads=4, heads_kv=2, dim=64, is_causal=True, window_size_left=-2
            )

    @pytest.mark.smoke
    def test_invalid_window_size_right_raises(self):
        with pytest.raises(ValueError, match="window_size_right"):
            GroupedQueryAttentionSlidingWindowVarlenFwdOp(
                batch=1,
                heads=4,
                heads_kv=2,
                dim=64,
                is_causal=True,
                window_size_right=-3,
            )

    @pytest.mark.smoke
    def test_heads_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            GroupedQueryAttentionSlidingWindowVarlenFwdOp(
                batch=1, heads=4, heads_kv=3, dim=64, is_causal=True
            )

    # ── forward input validation (caught before kernel dispatch) ──────────────

    @pytest.fixture(scope="class")
    def float16_op(self):
        return GroupedQueryAttentionSlidingWindowVarlenFwdOp(
            batch=1, heads=4, heads_kv=2, dim=64, is_causal=True, dtype=torch.float16
        )

    def _inputs(self, dtype=torch.float16, device=None):
        device = device or _device()
        q = torch.randn(64, 4, 64, dtype=dtype, device=device)
        k = torch.randn(64, 2, 64, dtype=dtype, device=device)
        v = torch.randn(64, 2, 64, dtype=dtype, device=device)
        cu_q = torch.tensor([0, 64], dtype=torch.int32, device=device)
        cu_k = torch.tensor([0, 64], dtype=torch.int32, device=device)
        return q, k, v, cu_q, cu_k, 64

    @pytest.mark.smoke
    def test_dtype_mismatch_raises(self, float16_op):
        q = torch.randn(64, 4, 64, dtype=torch.bfloat16, device=_device())
        k = torch.randn(64, 2, 64, dtype=torch.bfloat16, device=_device())
        v = torch.randn(64, 2, 64, dtype=torch.bfloat16, device=_device())
        cu_q = torch.tensor([0, 64], dtype=torch.int32, device=_device())
        cu_k = torch.tensor([0, 64], dtype=torch.int32, device=_device())
        with pytest.raises(ValueError, match="dtype"):
            float16_op.forward(q, k, v, cu_q, cu_k, 64)

    @pytest.mark.smoke
    def test_cpu_tensor_raises(self, float16_op):
        q, k, v, cu_q, cu_k, mq = self._inputs(device="cpu")
        with pytest.raises(ValueError, match=get_device_backend().name):
            float16_op.forward(q, k, v, cu_q, cu_k, mq)

    @pytest.mark.smoke
    def test_cu_seqlens_dtype_raises(self, float16_op):
        q, k, v, _, cu_k, mq = self._inputs()
        cu_q_bad = torch.tensor([0, 64], dtype=torch.int64, device=_device())
        with pytest.raises(ValueError, match="int32"):
            float16_op.forward(q, k, v, cu_q_bad, cu_k, mq)

    @pytest.mark.smoke
    def test_max_seqlen_q_too_small_raises(self, float16_op):
        # cu_q = [0, 64] → actual max Q length 64; passing 32 must raise.
        q, k, v, cu_q, cu_k, _ = self._inputs()
        with pytest.raises(ValueError, match="max_seqlen_q"):
            float16_op.forward(q, k, v, cu_q, cu_k, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
