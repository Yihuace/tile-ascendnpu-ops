"""Correctness tests for EngramGateConvFwdOp (NPU).

Ported from TileOPs ``tests/ops/test_engram.py`` — the ``EngramGateConvFwdOp``
subset only (the shared GPU file also covers ``EngramGateConvBwdOp`` /
``EngramDecodeOp``, which are separate NPU migrations).

NPU adaptations (T1-T4):

- T1: input tensor creation goes through
  ``EngramGateConvFwdWorkload.gen_inputs``, which resolves ``device`` from
  ``get_device_backend().name`` instead of the hard-coded ``"cuda"``.
- T2: the GPU gate-conv-fwd test has no ``torch.cuda`` availability guard --
  nothing to remove.
- T3: the ``tune`` parametrization dimension is removed.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.ops.sequence_modeling.engram_gate_conv_fwd import EngramGateConvFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.sequence_modeling import CONV_KERNEL_SIZE, EngramGateConvFwdWorkload
from tileops.workloads.workload_base import FixtureBase


def _rmsnorm(x, w, eps=1e-6):
    """Returns (normed, rrms)."""
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = x_f * rrms * w.float()
    return normed, rrms.squeeze(-1)


def engram_gate_conv_fwd_torch(H, k, v, rms_w_h, rms_w_v, conv_w, eps=1e-6):
    """PyTorch reference for Engram GateConv forward."""
    _, _, d = H.shape

    h_norm, rrms_h = _rmsnorm(H, rms_w_h, eps)
    k_norm, rrms_k = _rmsnorm(k, rms_w_h, eps)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))

    v_hat = alpha * v.float()

    v_hat_norm, rrms_v = _rmsnorm(v_hat.to(H.dtype), rms_w_v, eps)

    v_perm = v_hat_norm.float().permute(0, 2, 1)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))
    conv_w_expanded = conv_w.float().T.unsqueeze(1)
    conv_out = F.conv1d(v_padded, conv_w_expanded, groups=d).permute(0, 2, 1)

    Y = F.silu(conv_out) + v_hat.float()

    return (
        Y.to(H.dtype),
        v_hat.to(H.dtype),
        alpha.squeeze(-1).float(),
        rrms_h.float(),
        rrms_k.float(),
        rrms_v.float(),
    )


class EngramGateConvFwdTest(EngramGateConvFwdWorkload, TestBase):
    def ref_program(self, H, k, v, rms_w_h, rms_w_v, conv_w):
        return engram_gate_conv_fwd_torch(H, k, v, rms_w_h, rms_w_v, conv_w, self.eps)


class EngramGateConvFwdFixture(FixtureBase):
    PARAMS = [
        ("M, seq_len, d, dtype", [
            pytest.param(1, 32, 256, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1, 32, 256, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.full),
            pytest.param(2, 16, 256, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


@EngramGateConvFwdFixture
def test_engram_gate_conv_fwd(M, seq_len, d, dtype):
    test = EngramGateConvFwdTest(M, seq_len, d, dtype)
    op = EngramGateConvFwdOp(M, seq_len, d, dtype)
    inputs = test.gen_inputs()
    atol = 1e-1 if dtype == torch.float16 else 2e-1
    rtol = 1e-1
    test.check(op, *inputs, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
