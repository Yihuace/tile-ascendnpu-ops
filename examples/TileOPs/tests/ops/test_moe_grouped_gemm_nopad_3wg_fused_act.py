"""Op-level tests for MoeGroupedGemmNopad3WGFusedActFwdOp (fused gate/up GEMM).

Ported from the GPU TileOPs test
``tests/ops/test_fused_moe_experts.py::test_fused_act_fwd_op_shape_and_values``
(the op has no dedicated GPU test file; it is exercised only inside the
fused-MoE test module).  The ``MoeGroupedGemmNopad3WGFusedActFwdOp`` case is
ported on its own -- the surrounding FusedMoEExperts / SharedFusedMoE cases
are not migrated yet and are intentionally not carried over.

Verifies that the fused gate/up grouped GEMM produces the same per-expert
``act(A @ B_gate^T) * (A @ B_up^T)`` as a pure-PyTorch reference, across:
  - silu_and_mul and gelu_and_mul activations
  - bf16 and fp16 activations

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the workload's ``gen_inputs`` (device
  backend) instead of hard-coded ``device="cuda"``.
- T2: the GPU SM90 skip guard is replaced by nothing (the NPU kernel is
  device-agnostic; the arch check is handled by the Op base).
- T3: ``tune`` parametrization dimension removed.
- T4: no autotune try/except to remove.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.ops.moe.moe_grouped_gemm_nopad_3wg_fused_act import (
    MoeGroupedGemmNopad3WGFusedActFwdOp,
)
from tileops.workloads.moe import MoeGroupedGemmNopad3WGFusedActWorkload
from tileops.workloads.workload_base import FixtureBase


def _ref_fused_act(
    a: torch.Tensor,
    b: torch.Tensor,
    true_sizes: torch.Tensor,
    true_offsets: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """Pure-PyTorch reference for the fused gate/up activation grouped GEMM.

    Args:
        a: [numel, K] tight permuted activations.
        b: [num_experts, 2*ffn, K] gate||up expert weights.
        true_sizes: [E] int32 token count per expert.
        true_offsets: [E] int32 start offset per expert into a.
        activation: "silu_and_mul" or "gelu_and_mul".

    Returns:
        c: [numel, ffn] reference output. Rows belonging to expert e are
            ``act(a[o:o+n] @ b[e, :ffn].T) * (a[o:o+n] @ b[e, ffn:].T)``.
    """
    numel, _ = a.shape
    num_experts, two_ffn, _ = b.shape
    ffn = two_ffn // 2
    if activation == "silu_and_mul":
        act = lambda gate, up: F.silu(gate) * up  # noqa: E731
    elif activation == "gelu_and_mul":
        # Exact erf GELU, matching the kernel's 0.5*g*(1+erf(g/sqrt(2))).
        act = lambda gate, up: F.gelu(gate, approximate="none") * up  # noqa: E731
    else:
        raise ValueError(f"unsupported activation {activation!r}")

    c = torch.zeros(numel, ffn, dtype=a.dtype, device=a.device)
    sizes_l = true_sizes.tolist()
    offsets_l = true_offsets.tolist()
    for e in range(num_experts):
        size_e = sizes_l[e]
        if size_e == 0:
            continue
        off_e = offsets_l[e]
        gu = a[off_e:off_e + size_e].to(torch.float32) @ b[e].to(torch.float32).T
        c[off_e:off_e + size_e] = act(gu[:, :ffn], gu[:, ffn:]).to(a.dtype)
    return c


class MoeGroupedGemmNopad3WGFusedActFixture(FixtureBase):
    PARAMS = [
        ("numel, num_experts, ffn, k, activation, dtype", [
            # Smoke: K and ffn block-aligned, uniform distribution, silu, bf16.
            pytest.param(
                64, 4, 128, 64, "silu_and_mul", torch.bfloat16,
                marks=pytest.mark.smoke, id="smoke-silu-bf16",
            ),
            # Smoke: gelu activation + fp16 -- covers both dtype/activation branches.
            pytest.param(
                64, 4, 128, 64, "gelu_and_mul", torch.float16,
                marks=pytest.mark.smoke, id="smoke-gelu-fp16",
            ),
            # Full: larger multi-expert shape, silu, bf16.
            pytest.param(
                256, 8, 256, 128, "silu_and_mul", torch.bfloat16,
                marks=pytest.mark.full, id="full-silu-bf16",
            ),
        ]),
    ]


@MoeGroupedGemmNopad3WGFusedActFixture
def test_moe_grouped_gemm_nopad_3wg_fused_act_op(
    numel, num_experts, ffn, k, activation, dtype,
):
    workload = MoeGroupedGemmNopad3WGFusedActWorkload(
        numel, num_experts, ffn, k, dtype, activation=activation,
    )
    a, b, true_sizes, true_offsets = workload.gen_inputs()

    op = MoeGroupedGemmNopad3WGFusedActFwdOp(
        numel, num_experts, ffn, k, dtype=dtype, activation=activation,
    )
    c = op(a, b, true_sizes, true_offsets)
    c_ref = _ref_fused_act(a, b, true_sizes, true_offsets, activation)

    assert c.shape == (numel, ffn), f"expected ({numel}, {ffn}), got {tuple(c.shape)}"
    assert c.dtype == dtype

    # bf16/fp16 accumulation in fp32 -- match the kernel's accum_dtype.
    torch.testing.assert_close(
        c.float(), c_ref.float(), atol=1e-2, rtol=1e-2,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
