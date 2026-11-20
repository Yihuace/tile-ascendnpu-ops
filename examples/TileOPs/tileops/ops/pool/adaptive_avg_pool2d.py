"""AdaptiveAvgPool2dFwdOp -- adaptive average pooling over CHW/NCHW inputs (NPU-adapted).

``torch.nn.functional.adaptive_avg_pool2d`` semantics: the input spatial axes
are partitioned into ``output_size`` bins ``[floor(o*in/out), ceil((o+1)*in/out))``
and each output element is the mean of its bin.  ``None`` entries in
``output_size`` keep the corresponding input axis.

Adaptation from GPU (TileOPs) to NPU:

- O1: ``x.is_cuda`` -> ``backend.is_device_tensor(x)``.
- O2: the CUDA-only error message uses ``backend.name``.
- O3: ``tune`` parameter removed; the kernel constructor takes
      ``(n, c_in, h_in, w_in, out_h, out_w, dtype, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: ``from .compile_boundary import register_instance`` and the
      ``_pool_fwd`` torch.compile dispatch boundary are removed -- ``forward``
      calls ``_eager_forward`` directly.
- O6: the op-specific flow (CHW unsqueeze -> reshape/validate -> kernel
      dispatch -> squeeze, output-shape inference, roofline) is preserved.

Only ``AdaptiveAvgPool2dFwdOp`` (single output) is ported here; the shared GPU
pool module also holds the 1d/2d/3d avg/max pool ops and the adaptive max-pool
/ indices variants, which are not migrated yet.
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.adaptive_avg_pool2d import AdaptiveAvgPool2dKernel
from tileops.ops.op_base import Op

__all__ = ["AdaptiveAvgPool2dFwdOp"]


def _device_index(tensor: torch.Tensor) -> int | None:
    return tensor.device.index


def _normalize_output_size(
    output_size: int | None | Tuple[Optional[int], Optional[int]],
) -> Tuple[Optional[int], Optional[int]]:
    """Normalize adaptive-pool ``output_size`` to a 2-tuple of ``int | None``.

    Accepts ``None``, an int, or a 2-element tuple/list of ``int | None``
    (PyTorch-style leniency for list inputs); ``None`` entries -- including a
    scalar ``None`` -- resolve to the input size.
    """
    if output_size is None:
        return (None, None)
    if isinstance(output_size, bool):
        raise TypeError("output_size must be None, an int, or a tuple of (int | None, int | None)")
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if (
        not isinstance(output_size, (tuple, list))
        or len(output_size) != 2
        or any(
            isinstance(v, bool) or (v is not None and not isinstance(v, int)) for v in output_size
        )
    ):
        raise TypeError("output_size must be None, an int, or a tuple of (int | None, int | None)")
    if any(v is not None and v <= 0 for v in output_size):
        raise ValueError("output_size entries must be positive or None")
    return tuple(output_size)


def _validate_adaptive_pool_input_dtypes(self, input: torch.Tensor) -> None:
    """Adaptive-pool dtype validator: FP16/BF16 only (bound per concrete class)."""
    if input.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"input.dtype must be float16 or bfloat16, got {input.dtype}")


def _adaptive_pool2d_roofline(op: "_AdaptivePool2dFwdOpBase", *, indices: bool) -> tuple[int, int]:
    """Shared adaptive-pool roofline mirroring the manifest formulas.

    The manifest declares the same exact adaptive-bin scan
    (``scan_H`` / ``scan_W``) so ``eval_roofline`` and the workload spec stay
    consistent.
    """
    if op._last_roofline_spec is None:
        raise RuntimeError(
            f"{type(op).__name__}.eval_roofline() requires a prior forward() "
            "call to bind input shape and dtype"
        )
    n, c_in, h_in, w_in, out_h, out_w, dtype = op._last_roofline_spec
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    # Exact adaptive-bin scan: bins are [floor(o*in/out), ceil((o+1)*in/out))
    # and adjacent bins overlap by one row/col unless out | j*in.
    scan_h = h_in + sum(1 for j in range(1, out_h) if (j * h_in) % out_h != 0)
    scan_w = w_in + sum(1 for o in range(1, out_w) if (o * w_in) % out_w != 0)
    flops = n * c_in * scan_h * scan_w
    bytes_ = (n * c_in * h_in * w_in + n * c_in * out_h * out_w) * elem_bytes
    if indices:
        bytes_ += n * c_in * out_h * out_w * 8
    return flops, bytes_


class _AdaptivePool2dFwdOpBase(Op):
    """Generic adaptive 2D pooling forward over CHW/NCHW inputs.

    Concrete subclasses set ``_kernel_slot``, supply ``default_kernel_map``,
    and keep ``eval_roofline`` / ``_validate_dtypes`` in their own class body.

    NPU adaptations: the ``tune`` parameter and the ``_pool_fwd``
    torch.compile dispatch boundary are removed (O3/O5) -- ``forward`` calls
    ``_eager_forward`` directly.
    """

    _kernel_slot: ClassVar[str] = ""

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]],
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.n = None
        self.c_in = None
        self.h_in = None
        self.w_in = None
        self.output_size = _normalize_output_size(output_size)
        self.dtype = None
        self.dispatch_kernel(kernel_map)
        if self._kernel_slot not in self.kernel_map:
            raise NotImplementedError(
                f"{type(self).__name__} requires {self._kernel_slot!r} in kernel_map"
            )
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self._last_roofline_spec: Optional[tuple] = None

    def _resolve_out_dims(self, h_in: int, w_in: int) -> tuple[int, int]:
        # Manifest parity probes call _infer_output_shapes on a mock self
        # without __init__; a missing output_size acts as (None, None),
        # i.e. pool to the input spatial size.
        output_size = getattr(self, "output_size", (None, None))
        out_h = h_in if output_size[0] is None else output_size[0]
        out_w = w_in if output_size[1] is None else output_size[1]
        return out_h, out_w

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) == 3:
            c_in, h_in, w_in = input_shape
            out_h, out_w = self._resolve_out_dims(h_in, w_in)
            full = (c_in, out_h, out_w)
        elif len(input_shape) == 4:
            n, c_in, h_in, w_in = input_shape
            out_h, out_w = self._resolve_out_dims(h_in, w_in)
            full = (n, c_in, out_h, out_w)
        else:
            raise ValueError(f"{type(self).__name__} expects input_shape to be 3D CHW or 4D NCHW")
        return {"output": full}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # O5: the GPU compile_boundary ``_pool_fwd`` custom op is removed; the
        # op dispatches straight to the eager path.
        return self._eager_forward(input)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 3:
            squeezed = True
            x = input.unsqueeze(0)
        elif input.ndim == 4:
            squeezed = False
            x = input
        else:
            raise ValueError(
                f"{type(self).__name__} expects input to be a 3D CHW or 4D NCHW tensor"
            )
        backend = get_device_backend()
        if not backend.is_device_tensor(x):  # O1
            raise ValueError(f"input must be a {backend.name} tensor")
        self._validate_dtypes(x)
        n, c_in, h_in, w_in = x.shape
        out_h, out_w = self._resolve_out_dims(h_in, w_in)
        x = x.contiguous()
        dtype = x.dtype
        key = (n, c_in, h_in, w_in, out_h, out_w, dtype, _device_index(x))
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map[self._kernel_slot](
                n=n,
                c_in=c_in,
                h_in=h_in,
                w_in=w_in,
                out_h=out_h,
                out_w=out_w,
                dtype=dtype,
            )
        kernel = self._kernel_cache[key]
        self.kernel = kernel
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.out_h = out_h
        self.out_w = out_w
        self.dtype = dtype
        self._last_roofline_spec = (n, c_in, h_in, w_in, out_h, out_w, dtype)
        out = kernel(x)
        if squeezed:
            return out.squeeze(0)
        return out


class AdaptiveAvgPool2dFwdOp(_AdaptivePool2dFwdOpBase):
    """Adaptive average pooling over PyTorch-compatible CHW/NCHW inputs."""

    _kernel_slot = "adaptive_avg_pool2d_kernel"
    _validate_dtypes = _validate_adaptive_pool_input_dtypes

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]],
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        super().__init__(output_size=output_size, kernel_map=kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "adaptive_avg_pool2d_kernel": AdaptiveAvgPool2dKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        return _adaptive_pool2d_roofline(self, indices=False)
