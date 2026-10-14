"""Convolution operators (host-side Op layer, NPU-adapted).

Provides:
  - Conv1dFwdOp: torch.nn.functional.conv1d

The GPU ``tileops/ops/convolution.py`` also defines ``Conv2dFwdOp`` /
``Conv3dFwdOp``; only the 1-D operator is migrated.

Adaptation from GPU (TileOPs) to NPU:

- O1: ``input.is_cuda`` device check -> ``backend.is_device_tensor(input)``.
- O2: device is never hard-coded; the kernel binds the inputs it is handed.
- O3: ``tune`` / ``target`` parameters removed; kernel constructors take no
  ``tune`` argument.
- O6: the op-specific flow (shape inference -> kernel selection -> dispatch ->
  roofline) is preserved, with the GPU ``entry_for`` / ``refusal`` /
  ``CallSpec`` machinery inlined as a plain kernel-key selection.
- ``eval_roofline`` inlines the GPU
  ``tileops.perf.formulas.conv_roofline`` (the NPU project has no tileops.perf
  module; the roofline arithmetic is device-agnostic).
"""

from typing import Dict, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.convolution import (
    Conv1dKernel,
    Conv1dPointwiseKernel,
    Conv1dUnitStrideKernel,
    DepthwiseConv1dKernel,
    GroupConv1dKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["Conv1dFwdOp"]


def _axes(value: int | Tuple[int, ...], dims: int) -> Tuple[int, ...]:
    """A per-axis parameter as one value per spatial axis."""
    return (value,) * dims if isinstance(value, int) else tuple(value)


def _out_dim(size: int, kernel: int, stride: int, before: int, after: int, dilation: int) -> int:
    return (size + before + after - dilation * (kernel - 1) - 1) // stride + 1


def _window_taps(extent: int, out: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    """Taps inside the input, summed over the windows of the axis; padded taps are none."""
    return sum(
        1
        for o in range(out)
        for j in range(kernel)
        if 0 <= o * stride - padding + j * dilation < extent
    )


def _window_reads(extent: int, out: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    """Distinct input positions the windows of the axis read."""
    return len(
        {
            p
            for o in range(out)
            for j in range(kernel)
            if 0 <= (p := o * stride - padding + j * dilation) < extent
        }
    )


class Conv1dFwdOp(Op):
    """1D convolution over an NCL input, as ``torch.nn.functional.conv1d``.

    Args:
        stride: Stride, an int or a 1-tuple (default 1).
        padding: Padding each side, an int or a 1-tuple, or ``'valid'`` /
            ``'same'`` (default 0).  ``'same'`` puts the extra element of an odd
            total on the right.
        dilation: Dilation, an int or a 1-tuple (default 1).
        groups: Number of channel groups (default 1).
        kernel_map: Optional override for kernel dispatch.
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    def __init__(
        self,
        stride: int | Tuple[int] = 1,
        padding: int | Tuple[int] | str = 0,
        dilation: int | Tuple[int] = 1,
        groups: int = 1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel: Optional[Kernel] = None
        # Forward-time shape/geometry facts (bound by ``forward``); consumed by
        # ``eval_roofline`` and the msprof Op reconstruction.
        self.n = None
        self.c_in = None
        self.l_in = None
        self.c_out = None
        self.c_in_g = None
        self.c_out_g = None
        self.kernel_l = None
        self.stride_l = None
        self.dilation_l = None
        self.pad_left = None
        self.pad_right = None
        self.out_l = None
        self.has_bias = False
        self.dtype = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "conv1d_pointwise": Conv1dPointwiseKernel,
            "conv1d": Conv1dKernel,
            "conv1d_unit_stride": Conv1dUnitStrideKernel,
            "depthwise_conv1d": DepthwiseConv1dKernel,
            "group_conv1d": GroupConv1dKernel,
        }

    def _select_kernel_key(
        self,
        c_in: int,
        c_in_g: int,
        c_out_g: int,
        kernel_l: int,
        stride: int,
        pad_left: int,
        pad_right: int,
        dilation: int,
    ) -> str:
        """Pick the most specific kernel that serves this call (GPU refusal priority)."""
        if self.groups > 1:
            if c_in_g == 1 and c_out_g == 1:
                return "depthwise_conv1d"
            return "group_conv1d"
        if (
            kernel_l == 1
            and stride == 1
            and pad_left == 0
            and pad_right == 0
            and dilation == 1
        ):
            return "conv1d_pointwise"
        if stride == 1 and kernel_l > 1 and c_in >= 16:
            return "conv1d_unit_stride"
        return "conv1d"

    def _get_kernel(
        self,
        key: str,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        c_in_g: int,
        c_out_g: int,
        kernel_l: int,
        stride: int,
        pad_l: Tuple[int, int],
        dilation: int,
        dtype: torch.dtype,
        has_bias: bool,
    ) -> Kernel:
        cache_key = (
            key,
            n,
            c_in,
            l_in,
            c_out,
            c_in_g,
            c_out_g,
            kernel_l,
            stride,
            pad_l,
            dilation,
            dtype,
            has_bias,
        )
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        cls = self.kernel_map[key]
        if key == "conv1d_pointwise":
            kernel = cls(n, c_in, l_in, c_out, dtype, has_bias=has_bias)
        elif key in ("conv1d", "conv1d_unit_stride"):
            kernel = cls(
                n, c_in, l_in, c_out, kernel_l, stride, pad_l, dtype,
                dilation_l=dilation, has_bias=has_bias,
            )
        else:  # group_conv1d / depthwise_conv1d
            kernel = cls(
                n, c_in, l_in, c_out, kernel_l, stride, pad_l, dtype,
                dilation_l=dilation, has_bias=has_bias, groups=self.groups,
                c_in_g=c_in_g, c_out_g=c_out_g,
            )
        self._kernel_cache[cache_key] = kernel
        return kernel

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the convolution.

        Args:
            input: Input, ``[N, C_in, L]``.
            weight: Weight, ``[C_out, C_in / groups, k]``.
            bias: Per-output-channel bias, ``[C_out]``, or ``None``.

        Returns:
            The convolution result, ``[N, C_out, L_out]``.
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(input):
            raise ValueError(f"input must be a {backend.name} tensor, got device {input.device}")
        if input.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"{type(self).__name__} does not support dtype {input.dtype}. "
                f"Supported: [{names}]"
            )
        if input.ndim != 3:
            raise ValueError("input must have shape [N, C_in, L_in]")
        n, c_in, l_in = input.shape
        c_out, c_in_g, kernel_l = weight.shape
        (stride,) = _axes(self.stride, 1)
        (dilation,) = _axes(self.dilation, 1)
        if self.padding == "same":
            total = dilation * (kernel_l - 1)
            pad_left, pad_right, out_l = total // 2, total - total // 2, l_in
        else:
            (pad,) = (0,) if self.padding == "valid" else _axes(self.padding, 1)
            pad_left = pad_right = pad
            out_l = _out_dim(l_in, kernel_l, stride, pad, pad, dilation)

        c_out_g = c_out // self.groups
        key = self._select_kernel_key(
            c_in, c_in_g, c_out_g, kernel_l, stride, pad_left, pad_right, dilation
        )
        kernel = self._get_kernel(
            key,
            n,
            c_in,
            l_in,
            c_out,
            c_in_g,
            c_out_g,
            kernel_l,
            stride,
            (pad_left, pad_right),
            dilation,
            input.dtype,
            bias is not None,
        )

        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.c_in_g = c_in_g
        self.c_out_g = c_out_g
        self.kernel_l = kernel_l
        self.stride_l = stride
        self.dilation_l = dilation
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.out_l = out_l
        self.has_bias = bias is not None
        self.dtype = input.dtype
        self.kernel = kernel

        input = input.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        return kernel(input, weight, bias)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the last forwarded call.

        Inlined from the GPU ``tileops.perf.formulas.conv_roofline``: a
        multiply-add per in-range tap of each output element, plus the bias add
        when present; every tensor moves once with ``input`` read only at the
        positions some window reads.
        """
        if self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        n = int(self.n)
        c_in = int(self.c_in)
        l_in = int(self.l_in)
        c_out = int(self.c_out)
        c_in_g = int(self.c_in_g)
        kernel_l = int(self.kernel_l)
        stride = int(self.stride_l)
        dilation = int(self.dilation_l)
        out_l = int(self.out_l)
        elem_bytes = self.dtype.itemsize

        taps = _window_taps(l_in, out_l, kernel_l, stride, self.pad_left, dilation)
        flops = 2 * n * c_out * c_in_g * taps
        if self.has_bias:
            flops += n * c_out * out_l

        input_bytes = n * c_in * l_in * elem_bytes
        total_bytes = (
            n * c_in * l_in
            + c_out * c_in_g * kernel_l
            + n * c_out * out_l
            + (c_out if self.has_bias else 0)
        ) * elem_bytes
        reads = _window_reads(l_in, out_l, kernel_l, stride, self.pad_left, dilation)
        nbytes = total_bytes - input_bytes + input_bytes * reads // l_in
        return int(flops), int(nbytes)
