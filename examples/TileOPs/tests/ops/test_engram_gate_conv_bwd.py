"""Correctness tests for EngramGateConvBwdOp (NPU).

Ported from TileOPs ``tests/ops/test_engram.py`` — the ``EngramGateConvBwdOp``
subset only (the shared GPU file also covers ``EngramGateConvFwdOp`` /
``EngramDecodeOp``, which are separate NPU migrations).

NPU adaptations (T1-T4):

- T1: input tensor creation goes through
  ``EngramGateConvBwdWorkload.gen_inputs``, which resolves ``device`` from
  ``get_device_backend().name`` instead of the hard-coded ``"cuda"``.
- T2: the GPU gate-conv-bwd test has no ``torch.cuda`` availability guard --
  nothing to remove.
- T3: the ``tune`` parametrization dimension is removed.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.ops.sequence_modeling.engram_gate_conv_bwd import EngramGateConvBwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.sequence_modeling import CONV_KERNEL_SIZE, EngramGateConvBwdWorkload
from tileops.workloads.workload_base import FixtureBase


def ref_engram_gate_conv_bwd(dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                             vhat, alpha, rrms_h, rrms_k, rrms_v, eps=1e-6):
    """PyTorch reference backward via autograd."""
    M, T, d = H.shape

    H_ag = H.float().detach().requires_grad_(True)
    k_ag = k.float().detach().requires_grad_(True)
    v_ag = v.float().detach().requires_grad_(True)
    w_h_ag = rms_w_h.float().detach().requires_grad_(True)
    w_v_ag = rms_w_v.float().detach().requires_grad_(True)
    cw_ag = conv_w.float().detach().requires_grad_(True)

    def _rmsnorm(x, w):
        return x * (x ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt() * w

    h_norm = _rmsnorm(H_ag, w_h_ag)
    k_norm = _rmsnorm(k_ag, w_h_ag)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha_ag = torch.sigmoid(dot / (d ** 0.5))
    v_hat_ag = alpha_ag * v_ag
    v_hat_norm = _rmsnorm(v_hat_ag, w_v_ag)

    v_perm = v_hat_norm.permute(0, 2, 1)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))
    cw_expanded = cw_ag.T.unsqueeze(1)
    conv_out = F.conv1d(v_padded, cw_expanded, groups=d).permute(0, 2, 1)
    Y_ag = F.silu(conv_out) + v_hat_ag

    Y_ag.backward(dY.float())

    return (
        H_ag.grad.to(H.dtype),
        k_ag.grad.to(H.dtype),
        v_ag.grad.to(H.dtype),
        w_h_ag.grad,
        w_v_ag.grad,
        cw_ag.grad,
    )


class EngramGateConvBwdTest(EngramGateConvBwdWorkload, TestBase):
    def ref_program(self, dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                    vhat, alpha, rrms_h, rrms_k, rrms_v):
        return ref_engram_gate_conv_bwd(
            dY, H, k, v, rms_w_h, rms_w_v, conv_w,
            vhat, alpha, rrms_h, rrms_k, rrms_v, self.eps,
        )


class EngramGateConvBwdFixture(FixtureBase):
    PARAMS = [
        # (M, seq_len, d, dtype)
        ("M, seq_len, d, dtype", [
            pytest.param(1, 32, 256, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1, 32, 256, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.full),
            pytest.param(2, 16, 256, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(2, 512, 256, torch.float16, marks=pytest.mark.full),
        ]),
    ]


@EngramGateConvBwdFixture
def test_engram_gate_conv_bwd(M, seq_len, d, dtype):
    test = EngramGateConvBwdTest(M, seq_len, d, dtype)
    op = EngramGateConvBwdOp(M, seq_len, d, dtype)
    inputs = test.gen_inputs()
    atol = 2e-1 if dtype == torch.float16 else 3e-1
    rtol = 2e-1
    test.check(op, *inputs, atol=atol, rtol=rtol)

    # A data race varies run to run; allclose can still pass, so require two runs to match.
    run1 = [o.clone() for o in op(*inputs)]
    run2 = [o.clone() for o in op(*inputs)]
    for i, name in ((0, "dH"), (1, "dk"), (2, "dv")):
        max_err = (run1[i].float() - run2[i].float()).abs().max()
        assert torch.equal(run1[i], run2[i]), \
            f"{name} non-deterministic across runs (data race): max_err={max_err:.4e}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
