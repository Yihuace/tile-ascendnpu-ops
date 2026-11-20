"""Tests for FusedMoeFwdOp (NPU) -- no correction bias.

Per-op test file: the shared GPU ``tests/ops/test_moe_fused_moe.py`` was split
so that only the ``FusedMoeFwdOp`` (no correction bias) cases run when it is
integrated; the correction-bias (Kimi K2) cases belong to the separate
``FusedMoeFwdCbFwdOp`` migration unit and now live in
``test_moe_fused_moe_cb.py``.  Cases targeting the already-migrated
``FusedTopKOp`` live in ``test_moe_fused_topk.py``; the GPU-only vLLM baseline
is dropped.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses ``_device()``
  (``get_device_backend().name``) instead of the hard-coded ``device="cuda"``.
- T2: the GPU test had no ``torch.cuda`` availability guard -- nothing to remove.
- T3/T4: no ``tune`` parametrization / autotune try-except in the GPU test.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.ops.moe import FusedMoe, FusedMoeFwdOp, FusedTopKOp
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    """Return the active device backend name (T1)."""
    return get_device_backend().name


# Reference implementations


def _ref_moe_ffn(
    hidden_states: torch.Tensor,   # [T, H]
    w_gate_up: torch.Tensor,       # [E, 2*F, H]
    w_down: torch.Tensor,          # [E, H, F]
    topk_weights: torch.Tensor,    # [T, K] float32
    topk_ids: torch.Tensor,        # [T, K] int64
) -> torch.Tensor:
    """PyTorch reference: per-expert GEMM (memory-efficient, no O(T*K*2F*H) alloc)."""
    T, H = hidden_states.shape
    E = w_gate_up.shape[0]
    ffn_size = w_gate_up.shape[1] // 2

    output = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
    for e in range(E):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        t_idx, k_idx = mask.nonzero(as_tuple=True)
        h = hidden_states[t_idx].float()
        gate_up = h @ w_gate_up[e].float().t()
        act = F.silu(gate_up[:, :ffn_size]) * gate_up[:, ffn_size:]
        down = act @ w_down[e].float().t()
        weights = topk_weights[t_idx, k_idx].float().unsqueeze(-1)
        output.index_add_(0, t_idx, down * weights)
    return output.to(hidden_states.dtype)


# Test fixture -- Qwen3 config


class Qwen3Fixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " scoring_func, renormalize, dtype",
            [
                pytest.param(
                    32, 8, 2, 64, 32, "softmax", False, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-softmax-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, "softmax", True, torch.float16,
                    marks=pytest.mark.smoke, id="smoke-softmax-renorm-fp16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, "sigmoid", True, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-sigmoid-renorm-bf16",
                ),
                pytest.param(
                    512, 128, 8, 2048, 1024, "softmax", False, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen3-small",
                ),
                pytest.param(
                    2048, 128, 8, 2048, 1024, "softmax", False, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen3-medium",
                ),
                pytest.param(
                    512, 256, 8, 2048, 1024, "softmax", True, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen35-small",
                ),
                pytest.param(
                    512, 256, 8, 2048, 1024, "sigmoid", True, torch.bfloat16,
                    marks=pytest.mark.full, id="deepseek-small",
                ),
            ],
        )
    ]


@Qwen3Fixture
def test_fused_moe_qwen3(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    scoring_func, renormalize, dtype,
) -> None:
    torch.manual_seed(42)
    dev = _device()  # T1
    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=dev)
    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device=dev)
    w_gate_up = torch.randn(
        num_experts, ffn_size * 2, hidden_size, dtype=dtype, device=dev
    ) * 0.02
    w_down = torch.randn(
        num_experts, hidden_size, ffn_size, dtype=dtype, device=dev
    ) * 0.02

    op_nopad = FusedMoe(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        scoring_func=scoring_func, renormalize=renormalize,
        dtype=dtype,
    )

    out_nopad = op_nopad(hidden, gating, w_gate_up, w_down)

    assert out_nopad.shape == (num_tokens, hidden_size)
    assert out_nopad.dtype == dtype

    # Reference using the same FusedTopKOp routing
    fk = FusedTopKOp(num_tokens, num_experts, top_k, scoring_func, renormalize)
    topk_weights, topk_ids = fk(gating)
    ref = _ref_moe_ffn(hidden, w_gate_up, w_down, topk_weights, topk_ids.long())

    torch.testing.assert_close(out_nopad.float(), ref.float(), rtol=1e-2, atol=1e-2)


# Cases for the FusedMoe non-determinism regression.  The smoke case is a fast
# path-coverage check; the nightly case repeats many times to expose any
# intermittent race in the grouped-GEMM store epilogue as it surfaces inside
# the full FusedMoe pipeline (adjacent activation/permute kernels keep the
# timing window open).
_DET_CASES = [
    pytest.param(
        dict(num_tokens=32, num_experts=8, top_k=2, hidden_size=64,
             ffn_size=32, reps=3),
        marks=pytest.mark.smoke, id="smoke",
    ),
    pytest.param(
        dict(num_tokens=2048, num_experts=128, top_k=8, hidden_size=2048,
             ffn_size=1024, reps=200),
        marks=pytest.mark.nightly, id="qwen3-medium",
    ),
]


@pytest.mark.parametrize("case", _DET_CASES)
def test_fused_moe_deterministic(case):
    """Repeated FusedMoe calls on fixed inputs must be bitwise-identical.

    Regression for an intermittent grouped-GEMM store-epilogue write->read race
    that only manifested inside the full FusedMoe pipeline.  Run the op many
    times on fixed seed-42 inputs and assert every call matches the PyTorch
    reference and is bitwise-identical to the first.
    """
    torch.manual_seed(42)
    dev = _device()  # T1
    dtype = torch.bfloat16
    nt, ne, tk = case["num_tokens"], case["num_experts"], case["top_k"]
    hs, ff, reps = case["hidden_size"], case["ffn_size"], case["reps"]
    hidden = torch.randn(nt, hs, dtype=dtype, device=dev)
    gating = torch.randn(nt, ne, dtype=dtype, device=dev)
    w_gate_up = torch.randn(ne, ff * 2, hs, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(ne, hs, ff, dtype=dtype, device=dev) * 0.02

    op = FusedMoe(
        num_tokens=nt, num_experts=ne, top_k=tk, hidden_size=hs,
        ffn_size=ff, scoring_func="softmax", renormalize=False, dtype=dtype,
    )
    fk = FusedTopKOp(nt, ne, tk, "softmax", False)
    topk_weights, topk_ids = fk(gating)
    ref = _ref_moe_ffn(hidden, w_gate_up, w_down, topk_weights, topk_ids.long())

    first = op(hidden, gating, w_gate_up, w_down)
    torch.testing.assert_close(first.float(), ref.float(), rtol=1e-2, atol=1e-2)
    for i in range(reps):
        out = op(hidden, gating, w_gate_up, w_down)
        assert torch.equal(out, first), (
            f"non-deterministic FusedMoe output on call {i + 1}/{reps}: "
            "grouped-GEMM store epilogue write->read race regressed"
        )


# expert_map local filtering test (EP simulation without All-to-All)


@pytest.mark.smoke
def test_expert_map_local_filter() -> None:
    """Simulate EP=2 on a single device: each rank owns half the experts.

    Verification: sum of outputs from rank-0 and rank-1 (with their disjoint
    expert_maps) equals the output without expert_map.
    """
    torch.manual_seed(42)
    dev = _device()  # T1
    T, E, K, H, F_ = 32, 8, 2, 64, 32
    dtype = torch.bfloat16

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, F_ * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F_, dtype=dtype, device=dev) * 0.02

    # Rank 0 owns experts 0..3, rank 1 owns experts 4..7
    expert_map_rank0 = torch.full((E,), -1, dtype=torch.int32, device=dev)
    expert_map_rank0[:E // 2] = torch.arange(E // 2, dtype=torch.int32, device=dev)

    expert_map_rank1 = torch.full((E,), -1, dtype=torch.int32, device=dev)
    expert_map_rank1[E // 2:] = torch.arange(E // 2, dtype=torch.int32, device=dev)

    # Full output (no expert_map)
    op_full = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F_, dtype=dtype,
    )
    out_full = op_full(hidden, gating, w_gate_up, w_down)

    # Rank-0 partial output (local experts 0..3)
    op_r0 = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F_, dtype=dtype,
        expert_map=expert_map_rank0,
    )
    out_r0 = op_r0(hidden, gating, w_gate_up[:E // 2], w_down[:E // 2])

    # Rank-1 partial output (local experts 4..7)
    op_r1 = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F_, dtype=dtype,
        expert_map=expert_map_rank1,
    )
    out_r1 = op_r1(hidden, gating, w_gate_up[E // 2:], w_down[E // 2:])

    # Sum of partial outputs should match full output
    out_sum = (out_r0.float() + out_r1.float()).to(dtype)
    torch.testing.assert_close(out_sum.float(), out_full.float(), rtol=1e-2, atol=1e-2)


# prepare_finalize / experts contract


@pytest.mark.smoke
def test_fused_moe_fwd_op_identity() -> None:
    """`FusedMoeFwdOp` (no correction bias) matches the `FusedMoe` reference output."""
    torch.manual_seed(42)
    dev = _device()  # T1
    T, E, K, H, F_ = 32, 8, 2, 64, 32
    dtype = torch.bfloat16

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, F_ * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F_, dtype=dtype, device=dev) * 0.02

    op = FusedMoeFwdOp(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F_, dtype=dtype,
    )
    out = op(hidden, gating, w_gate_up, w_down)
    assert out.shape == (T, H)
    assert out.dtype == dtype

    ref_op = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F_, dtype=dtype,
    )
    ref = ref_op(hidden, gating, w_gate_up, w_down)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)

    flops, nbytes = op.eval_roofline()
    assert flops > 0 and nbytes > 0


@pytest.mark.smoke
def test_prepare_finalize_without_experts_raises() -> None:
    """Supplying prepare_finalize= without experts= must raise ValueError.

    prepare_finalize can change the dispatched token count T'; the default
    experts instance is JIT-compiled for the original T and cannot be reused.
    """
    from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP

    with pytest.raises(ValueError, match="experts="):
        FusedMoe(
            num_tokens=16, num_experts=4, top_k=2,
            hidden_size=64, ffn_size=32,
            prepare_finalize=MoEPrepareAndFinalizeNoDPEP(),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
