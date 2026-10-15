"""3-D convolution kernel (NPU-adapted wrapper + ``Conv3dKernel`` class).

The GPU ``Conv3dFwdOp`` selects among a dense NCDHW kernel, an NDHWC-staged
perf variant and a grouped kernel; the NDHWC variant computes the same result as
the dense path (``examples/conv3d/probe/PROBE.md`` Sec 5.3), so only the dense
implicit-GEMM kernel is migrated here and it serves every ungrouped call.

This file ports the ``Conv3dKernel`` class and imports its ``@tilelang.jit``
factory from the extracted reference file (``conv3d_kernel/_conv3d_kernel.py``),
mirroring the Conv1d package.

Adaptation summary (GPU -> NPU), the same K1-K11 applied for Conv1d:

  - Expert mode: the kernel is a Cube implicit GEMM (+ a Vector bias epilogue);
    ``ascend_mode = "Expert"`` pins every invocation regardless of the ambient
    ``TILELANG_ASCEND_MODE``.
  - K5: ``supported_archs = None``.
  - K8/K9/K11: GPU ``threads`` / ``num_stages`` / ``block_*`` CUDA execution
    knobs are not baked; only the NPU block tiling is carried.

Stage-5 glue (this file): the migrated npuir ``@T.prim_func`` takes its im2col /
polyphase workspaces (``x_pad`` / ``xdec``) and the weight re-layout (``w_flat``)
as explicit tensor parameters, prepared per call by the host-side helpers in
``_conv3d_kernel`` (a pure permutation / zero pad / width decimation) and passed
by ``forward``; the fp32 bias relay workspace ``ws`` is allocated only on the
bias side, matching the Conv1d wrapper.
"""

from typing import Optional

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils import NPUUtils

from tileops.kernels.kernel_base import Kernel

from .conv3d_kernel._conv3d_kernel import _conv3d_kernel, _host_prep, _ws_dims

__all__ = ["Conv3dKernel"]

# Cross-engine bias relay depth (Cube -> GM ws -> Vector), matching SLOTS in the
# integrated kernel module.
_SLOTS = 2

# NPU tiling defaults (design block sizes only); ``num_stages`` is a required
# factory-signature placeholder the Expert kernel ignores.
_DEFAULT_BLOCK = {"block_m": 64, "block_n": 128, "block_k": 128}
_NUM_STAGES = 1


class Conv3dKernel(Kernel):
    """Dense 3-D convolution over the full kernel window (ungrouped)."""

    # K5: CUDA SM list -> None (all architectures).
    supported_archs: Optional[list] = None

    # Cube implicit-GEMM Expert-form kernel; see module docstring.
    ascend_mode = "Expert"

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        c_out: int,
        kernel_d: int,
        kernel_h: int,
        kernel_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        pad_d: int,
        pad_h: int,
        pad_w: int,
        dtype: torch.dtype,
        dilation_d: int = 1,
        dilation_h: int = 1,
        dilation_w: int = 1,
        has_bias: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.c_out = c_out
        self.kernel_d = kernel_d
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = (stride_d, stride_h, stride_w)
        self.padding = (pad_d, pad_h, pad_w)
        self.dilation = (dilation_d, dilation_h, dilation_w)
        self.dtype = dtype
        self.has_bias = has_bias
        sd, sh, sw = self.stride
        dd, dh, dw = self.dilation
        pd, ph, pw = self.padding
        self.out_d = (d_in + 2 * pd - dd * (kernel_d - 1) - 1) // sd + 1
        self.out_h = (h_in + 2 * ph - dh * (kernel_h - 1) - 1) // sh + 1
        self.out_w = (w_in + 2 * pw - dw * (kernel_w - 1) - 1) // sw + 1
        self._factory = _conv3d_kernel(
            n,
            c_in,
            d_in,
            h_in,
            w_in,
            c_out,
            kernel_d,
            kernel_h,
            kernel_w,
            self.stride,
            self.padding,
            self.dilation,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # NPU tiling only; no CUDA execution-model knob.
        return {**_DEFAULT_BLOCK, "num_stages": _NUM_STAGES}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        weight = weight.contiguous()
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        d_pad = self.d_in + 2 * pd
        h_pad = self.h_in + 2 * ph
        need_poly, w_pad, mdec, c_in_pad = _ws_dims(self.w_in, self.c_in, pw, sw)
        x_pad, xdec, w_flat = _host_prep(
            x,
            weight,
            c_in_pad,
            d_pad,
            h_pad,
            w_pad,
            sd,
            sh,
            sw,
            pd,
            ph,
            pw,
            self.d_in,
            self.h_in,
            self.w_in,
            self.kernel_d,
            self.kernel_h,
            self.kernel_w,
            need_poly,
            mdec,
            self.dtype,
        )
        config = self.config
        program = self._factory(
            config["block_m"], config["block_n"], config["block_k"], config["num_stages"]
        )
        out_shape = (self.n, self.c_out, self.out_d, self.out_h, self.out_w)
        if bias is None:
            out = program(x_pad, w_flat, xdec)
        else:
            ws = torch.empty(
                (
                    NPUUtils.get().get_aicore_num(),
                    _SLOTS,
                    config["block_m"],
                    config["block_n"],
                ),
                dtype=torch.float32,
                device=x.device,
            )
            out = program(x_pad, w_flat, bias.contiguous(), xdec, ws)
        return out.reshape(out_shape)
