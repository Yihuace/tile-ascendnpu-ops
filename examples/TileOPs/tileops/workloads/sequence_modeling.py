"""Workload definitions for the sequence_modeling op family (NPU).

Ported from TileOPs ``workloads/engram.py`` — this op's workload subset.

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"`` in the original TileOPs code.
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase

CONV_KERNEL_SIZE = 4


class EngramDecodeWorkload(WorkloadBase):
    def __init__(self, batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, eps=1e-6):
        self.batch = batch
        self.d_mem = d_mem
        self.d = d
        self.max_conv_len = max_conv_len
        self.conv_kernel_size = conv_kernel_size
        self.dilation = dilation
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        device = get_device_backend().name
        e_t = torch.randn(self.batch, self.d_mem, dtype=self.dtype, device=device) * 0.1
        h_t = torch.randn(self.batch, self.d, dtype=self.dtype, device=device)
        # Full conv_state (max_conv_len entries)
        conv_state = torch.randn(
            self.batch, self.max_conv_len, self.d,
            dtype=self.dtype, device=device
        ) * 0.1
        W_K = torch.randn(self.d_mem, self.d, dtype=self.dtype, device=device) * 0.02
        W_V = torch.randn(self.d_mem, self.d, dtype=self.dtype, device=device) * 0.02
        rms_w_h = torch.ones(self.d, dtype=self.dtype, device=device)
        rms_w_v = torch.ones(self.d, dtype=self.dtype, device=device)
        conv_w = torch.randn(self.conv_kernel_size, self.d, dtype=self.dtype, device=device) * 0.02
        return e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w


class EngramGateConvFwdWorkload(WorkloadBase):
    def __init__(self, M, seq_len, d, dtype, eps=1e-6):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        device = get_device_backend().name
        H = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device=device)
        k = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device=device) * 0.1
        v = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device=device) * 0.1
        rms_w_h = torch.ones(self.d, dtype=self.dtype, device=device)
        rms_w_v = torch.ones(self.d, dtype=self.dtype, device=device)
        conv_w = torch.randn(CONV_KERNEL_SIZE, self.d, dtype=self.dtype, device=device) * 0.02
        return H, k, v, rms_w_h, rms_w_v, conv_w


class EngramGateConvBwdWorkload(WorkloadBase):
    def __init__(self, M, seq_len, d, dtype, eps=1e-6):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        """Generate inputs including saved intermediates from a reference forward."""
        device = get_device_backend().name
        H = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device=device)
        k = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device=device) * 0.1
        v = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device=device) * 0.1
        rms_w_h = torch.ones(self.d, dtype=self.dtype, device=device)
        rms_w_v = torch.ones(self.d, dtype=self.dtype, device=device)
        conv_w = torch.randn(CONV_KERNEL_SIZE, self.d, dtype=self.dtype, device=device) * 0.02
        dY = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device=device) * 0.1

        # Compute saved intermediates via reference forward
        def _rmsnorm(x, w):
            x_f = x.float()
            rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return x_f * rrms * w.float(), rrms.squeeze(-1)

        h_norm, rrms_h = _rmsnorm(H, rms_w_h)
        k_norm, rrms_k = _rmsnorm(k, rms_w_h)
        dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
        alpha = torch.sigmoid(dot / (self.d ** 0.5))
        v_hat = alpha * v.float()
        _, rrms_v = _rmsnorm(v_hat.to(self.dtype), rms_w_v)

        vhat = v_hat.to(self.dtype)
        alpha_squeezed = alpha.squeeze(-1).float()
        rrms_h = rrms_h.float()
        rrms_k = rrms_k.float()
        rrms_v = rrms_v.float()

        return (dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                vhat, alpha_squeezed, rrms_h, rrms_k, rrms_v)
