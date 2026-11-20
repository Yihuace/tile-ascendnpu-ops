"""Correctness tests for EngramDecodeOp (NPU).

Ported from TileOPs ``tests/ops/test_engram.py`` — the ``EngramDecodeOp``
subset only (the shared GPU file also covers ``EngramGateConvFwdOp`` /
``EngramGateConvBwdOp``, which are not migrated).

NPU adaptations (T1-T4):

- T1: manual tensor creation uses ``get_device_backend().name`` (via the
  workload's ``gen_inputs`` and ``_device()``) instead of the hard-coded
  ``"cuda"``.
- T2: the GPU decode tests have no ``torch.cuda`` availability guard --
  nothing to remove.
- T3: the ``tune`` parametrization dimension is removed.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.ops.sequence_modeling.engram_decode import EngramDecodeOp
from tileops.testing.test_base import TestBase
from tileops.workloads.sequence_modeling import EngramDecodeWorkload
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


def _rmsnorm_decode(x, w, eps=1e-6):
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f * rrms * w.float()), rrms


def engram_decode_step_torch(
    e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
    max_conv_len, dilation, eps=1e-6,
):
    """PyTorch reference for a single decode step with dilated causal conv."""
    B, d = h_t.shape
    w = conv_w.shape[0]
    L = conv_state.shape[1]

    k = e_t.float() @ W_K.float()
    v = e_t.float() @ W_V.float()

    h_norm, _ = _rmsnorm_decode(h_t.unsqueeze(1), rms_w_h)
    k_norm, _ = _rmsnorm_decode(k.unsqueeze(1).to(h_t.dtype), rms_w_h)
    h_norm = h_norm.squeeze(1)
    k_norm = k_norm.squeeze(1)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))
    v_hat = alpha * v

    v_hat_norm, _ = _rmsnorm_decode(v_hat.unsqueeze(1).to(h_t.dtype), rms_w_v)
    v_hat_norm = v_hat_norm.squeeze(1)

    if max_conv_len > L:
        padded_state = F.pad(conv_state.float(), (0, 0, max_conv_len - L, 0))
    else:
        padded_state = conv_state.float()

    conv_out = torch.zeros(B, d, device=h_t.device)
    for p in range(w - 1):
        state_idx = max_conv_len - (w - 1 - p) * dilation
        if 0 <= state_idx < max_conv_len:
            conv_out += conv_w[p].float().unsqueeze(0) * padded_state[:, state_idx, :]
    conv_out += conv_w[w - 1].float().unsqueeze(0) * v_hat_norm

    if max_conv_len > L:
        new_conv_state = torch.cat([
            conv_state,
            v_hat_norm.unsqueeze(1).to(conv_state.dtype),
        ], dim=1)
    else:
        new_conv_state = torch.cat([
            conv_state[:, 1:, :],
            v_hat_norm.unsqueeze(1).to(conv_state.dtype),
        ], dim=1)

    y_t = F.silu(conv_out) + v_hat
    return y_t.to(h_t.dtype), new_conv_state


class EngramDecodeTest(EngramDecodeWorkload, TestBase):
    def ref_program(self, e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w):
        y_ref, state_ref = engram_decode_step_torch(
            e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
            self.max_conv_len, self.dilation, self.eps,
        )
        return y_ref, state_ref


class EngramDecodeFixture(FixtureBase):
    PARAMS = [
        # (batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
        ("batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype", [
            pytest.param(1, 512, 256, 12, 4, 3, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1, 512, 256, 12, 4, 3, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4, 1024, 512, 20, 4, 5, torch.float16, marks=pytest.mark.full),
            pytest.param(8, 512, 256, 18, 4, 3, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


@EngramDecodeFixture
def test_engram_decode(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype):
    test = EngramDecodeTest(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    op = EngramDecodeOp(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype,
    )
    inputs = test.gen_inputs()
    atol = 5e-2 if dtype == torch.float16 else 1e-1
    rtol = 5e-2
    test.check(op, *inputs, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_engram_decode_multi_step():
    """Verify multi-step decode with growing conv_state and dilated conv."""
    B, d_mem, d = 2, 256, 256
    conv_kernel_size = 4
    dilation = 3
    max_conv_len = dilation * (conv_kernel_size - 1)  # = 9, minimum required
    dtype = torch.float16
    eps = 1e-6
    device = _device()

    torch.manual_seed(123)
    W_K = torch.randn(d_mem, d, dtype=dtype, device=device) * 0.02
    W_V = torch.randn(d_mem, d, dtype=dtype, device=device) * 0.02
    rms_w_h = torch.ones(d, dtype=dtype, device=device)
    rms_w_v = torch.ones(d, dtype=dtype, device=device)
    conv_w = torch.randn(conv_kernel_size, d, dtype=dtype, device=device) * 0.02

    op = EngramDecodeOp(B, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)

    # Start with empty conv_state (like empty KV cache)
    conv_state = torch.zeros(B, 0, d, dtype=dtype, device=device)
    conv_state_ref = conv_state.clone()

    num_steps = max_conv_len + 8  # go past growing phase into steady state
    for step in range(num_steps):
        e_t = torch.randn(B, d_mem, dtype=dtype, device=device) * 0.1
        h_t = torch.randn(B, d, dtype=dtype, device=device)

        y_op, conv_state = op(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)
        y_ref, conv_state_ref = engram_decode_step_torch(
            e_t, h_t, conv_state_ref, W_K, W_V, rms_w_h, rms_w_v, conv_w,
            max_conv_len, dilation, eps,
        )

        y_err = (y_op.float() - y_ref.float()).abs().max().item()
        # Compare valid portion of conv_state
        ref_len = conv_state_ref.shape[1]
        op_state_valid = conv_state[:, -ref_len:, :]
        s_err = (op_state_valid.float() - conv_state_ref.float()).abs().max().item()

        assert y_err < 0.1, f"Step {step}: y max_err={y_err:.6f}"
        assert s_err < 0.05, f"Step {step}: state max_err={s_err:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
