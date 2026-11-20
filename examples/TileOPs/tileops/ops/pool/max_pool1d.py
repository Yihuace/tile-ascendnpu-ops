"""MaxPool1dFwdOp -- max pooling over NCL inputs (NPU-adapted).

``torch.nn.functional.max_pool1d`` semantics (return_indices=False): each
output element is the max over the dilated ``kernel_size`` window placed at
``ow * stride - padding``; positions outside the padded input contribute
nothing (equivalent to ``-inf`` padding for a max reduction), a NaN anywhere in
the window wins, and ties keep the first max.  The output extent follows the
PyTorch floor/ceil rule (with the ``ceil_mode`` step-back correction) via
``pool_output_dim``.

Adaptation from GPU (TileOPs) to NPU:

- O1: ``input.is_cuda`` -> ``backend.is_device_tensor(input)``.
- O2: the CUDA-only error message uses ``backend.name``.
- O3: ``tune`` parameter removed; the kernel constructor takes
      ``(n, c_in, l_in, kernel_w, stride_w, pad_w, dilation_w, ceil_mode,
      dtype, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: ``from .compile_boundary import get_instance`` and the ``_pool_fwd``
      torch.compile dispatch boundary are removed -- ``forward`` calls
      ``_eager_forward`` directly.
- O6: the op-specific flow (3D NCL validate -> output-extent computation ->
      kernel dispatch -> roofline) is preserved.

Input rank note: ``MaxPool1dFwdOp`` accepts exactly 3D NCL input (unlike the
adaptive pool family, which also takes 2D CHW); the GPU Op rejects any other
rank and this port keeps that behaviour -- no unsqueeze.

Only ``MaxPool1dFwdOp`` (single output, no indices) is ported here; the shared
GPU pool module also holds the 1d/2d/3d avg/max pool ops and the max-pool
indices variants, which are not migrated yet.
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.max_pool1d.max_pool1d import (
    MaxPool1dKernel,
    pool_output_dim,
)
from tileops.ops.op_base import Op

__all__ = ["MaxPool1dFwdOp"]


def _device_index(tensor: torch.Tensor) -> int | None:
    return tensor.device.index


def _normalize_pool_dims(name: str, value: int | Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    """Normalize a scalar-or-tuple pooling parameter to an ``ndim``-tuple.

    Ported from GPU ``tileops/kernels/pool/common.py:_normalize_pool_dims``
    (device-agnostic, so preserved unchanged).
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int or a tuple of {ndim} ints")

    if isinstance(value, int):
        return (value,) * ndim

    if not isinstance(value, (tuple, list)) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an int or a tuple of {ndim} ints")

    if len(value) != ndim:
        raise ValueError(f"{name} must be an int or a tuple of {ndim} ints")

    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise TypeError(f"{name} must contain only ints")

    return tuple(value)


def _validate_pool_input_dtypes(self, input: torch.Tensor) -> None:
    """Pool-family dtype validator (bound per concrete class).

    GPU parity: fp16 / bf16 / fp32 are all accepted.
    """
    if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}")


def _max_pool_roofline(op: "_MaxPoolFwdOpBase", *, indices: bool) -> tuple[int, int]:
    """Shared max-pool roofline: flops = out_elems * prod(kernel); bytes in+out."""
    if op._last_roofline_spec is None:
        raise RuntimeError(
            f"{type(op).__name__}.eval_roofline() requires a prior forward() "
            "call to bind input shape and dtype"
        )
    spec = op._last_roofline_spec
    nd = op.ndim
    n, c_in = spec[0], spec[1]
    in_dims = spec[2 : 2 + nd]
    out_dims = spec[2 + nd : 2 + 2 * nd]
    dtype = spec[-1]
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    in_elems = n * c_in
    out_elems = n * c_in
    for size in in_dims:
        in_elems *= size
    for size in out_dims:
        out_elems *= size
    flops = out_elems
    for k in op.kernel_size:
        flops *= k
    bytes_ = (in_elems + out_elems) * elem_bytes
    if indices:
        bytes_ += out_elems * 8
    return flops, bytes_


# Layout token and per-axis name suffixes, indexed by spatial dimensionality.
_POOL_LAYOUTS: Dict[int, str] = {1: "NCL", 2: "NCHW", 3: "NCDHW"}
_POOL_DIM_NAMES: Dict[int, Tuple[str, ...]] = {1: ("l",), 2: ("h", "w"), 3: ("d", "h", "w")}
# Kernel-kwarg suffixes for kernel_size/stride/padding/dilation.
# Why: the 1d max-pool kernels historically name their pooling axis `w`.
_MAX_POOL_PARAM_SUFFIXES: Dict[int, Tuple[str, ...]] = {
    1: ("w",),
    2: ("h", "w"),
    3: ("d", "h", "w"),
}


class _MaxPoolFwdOpBase(Op):
    """Generic max-pooling forward, parametrized by class attributes.

    Concrete subclasses set ``ndim`` / ``_kernel_slot`` / ``_returns_indices``,
    supply ``default_kernel_map``, and keep ``eval_roofline`` /
    ``_validate_dtypes`` in their own class body.

    NPU adaptations: the ``tune`` parameter and the ``_pool_fwd``
    torch.compile dispatch boundary are removed (O3/O5) -- ``forward`` calls
    ``_eager_forward`` directly.
    """

    ndim: ClassVar[int]
    _kernel_slot: ClassVar[str] = ""
    _returns_indices: ClassVar[bool] = False

    def __init__(
        self,
        kernel_size: int | Tuple[int, ...],
        stride: Optional[int | Tuple[int, ...]] = None,
        padding: int | Tuple[int, ...] = 0,
        dilation: int | Tuple[int, ...] = 1,
        ceil_mode: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        nd = self.ndim
        self.n = None
        self.c_in = None
        for name in _POOL_DIM_NAMES[nd]:
            setattr(self, f"{name}_in", None)
        self.kernel_size = _normalize_pool_dims("kernel_size", kernel_size, nd)
        self.stride = (
            self.kernel_size if stride is None else _normalize_pool_dims("stride", stride, nd)
        )
        self.padding = _normalize_pool_dims("padding", padding, nd)
        self.dilation = _normalize_pool_dims("dilation", dilation, nd)
        if not isinstance(ceil_mode, bool):
            raise TypeError("ceil_mode must be a bool")
        self.ceil_mode = ceil_mode
        self.dtype = None
        self._validate_pool_params()
        self.dispatch_kernel(kernel_map)
        if self._kernel_slot not in self.kernel_map:
            raise NotImplementedError(
                f"{self.__class__.__name__} requires {self._kernel_slot!r} in kernel_map"
            )
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self._last_roofline_spec: Optional[tuple] = None

    def _validate_pool_params(self) -> None:
        """PyTorch-compatible pool parameter validation (device-agnostic).

        Ported verbatim from GPU
        ``tileops/kernels/pool/common.py:validate_pool_params``; the pooling
        dimensionality is taken from the concrete class's ``ndim``.
        """
        nd = self.ndim
        kernel_size, stride = self.kernel_size, self.stride
        padding, dilation = self.padding, self.dilation
        if len(kernel_size) != nd or len(stride) != nd or len(padding) != nd:
            raise ValueError("kernel_size, stride, and padding must match pooling dimensionality")
        if len(dilation) != nd:
            raise ValueError("dilation must match pooling dimensionality")

        for name, values in (
            ("kernel_size", kernel_size),
            ("stride", stride),
            ("padding", padding),
            ("dilation", dilation),
        ):
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
                raise TypeError(f"{name} must contain only ints")

        if any(v <= 0 for v in kernel_size):
            raise ValueError("kernel_size must be greater than zero")

        if any(v <= 0 for v in stride):
            raise ValueError("stride must be greater than zero")

        if any(v <= 0 for v in dilation):
            raise ValueError("dilation must be greater than zero")

        if any(v < 0 for v in padding):
            raise ValueError("padding must be non-negative")

        for pad, kernel in zip(padding, kernel_size, strict=True):
            if pad > kernel // 2:
                raise ValueError("padding must be at most half of the effective kernel size")

    def _resolve_input(self, input: torch.Tensor) -> tuple:
        nd = self.ndim
        if input.ndim != nd + 2:
            raise ValueError(
                f"{self.__class__.__name__} expects input to be a "
                f"{nd + 2}D {_POOL_LAYOUTS[nd]} tensor"
            )
        n, c_in, *in_dims = input.shape
        backend = get_device_backend()
        if not backend.is_device_tensor(input):  # O1
            raise ValueError(f"input must be a {backend.name} tensor")  # O2
        self._validate_dtypes(input)
        out_dims = tuple(
            pool_output_dim(
                size,
                self.kernel_size[k],
                self.stride[k],
                self.padding[k],
                self.ceil_mode,
                self.dilation[k],
            )
            for k, size in enumerate(in_dims)
        )
        if any(v <= 0 for v in out_dims):
            raise ValueError(
                f"{self.__class__.__name__} calculated output size must be greater than zero, "
                f"got {out_dims}"
            )
        return (n, c_in, *in_dims, *out_dims, input.dtype)

    def _get_kernel(
        self,
        n: int,
        c_in: int,
        in_dims: Tuple[int, ...],
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (
            n,
            c_in,
            *in_dims,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            dtype,
            device_index,
        )
        if key not in self._kernel_cache:
            kernel_kwargs: Dict[str, object] = dict(
                n=n,
                c_in=c_in,
                ceil_mode=self.ceil_mode,
                dtype=dtype,
            )
            for k, name in enumerate(_POOL_DIM_NAMES[self.ndim]):
                kernel_kwargs[f"{name}_in"] = in_dims[k]
            for k, name in enumerate(_MAX_POOL_PARAM_SUFFIXES[self.ndim]):
                kernel_kwargs[f"kernel_{name}"] = self.kernel_size[k]
                kernel_kwargs[f"stride_{name}"] = self.stride[k]
                kernel_kwargs[f"pad_{name}"] = self.padding[k]
                kernel_kwargs[f"dilation_{name}"] = self.dilation[k]
            self._kernel_cache[key] = self.kernel_map[self._kernel_slot](**kernel_kwargs)
        return self._kernel_cache[key]

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        nd = self.ndim
        if len(input_shape) != nd + 2:
            raise ValueError(
                f"{self.__class__.__name__} expects input_shape to be {nd + 2}D {_POOL_LAYOUTS[nd]}"
            )
        n, c_in, *in_dims = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        dilation = getattr(self, "dilation", (1,) * nd)
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            zero = (n, c_in) + (0,) * nd
            if self._returns_indices:
                return {"output": zero, "indices": zero}
            return {"output": zero}
        out_dims = tuple(
            pool_output_dim(size, kernel_size[k], stride[k], padding[k], ceil_mode, dilation[k])
            for k, size in enumerate(in_dims)
        )
        full = (n, c_in, *out_dims)
        if self._returns_indices:
            return {"output": full, "indices": full}
        return {"output": full}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # O5: the GPU compile_boundary ``_pool_fwd`` custom op is removed; the
        # op dispatches straight to the eager path.
        return self._eager_forward(input)

    def _eager_forward(self, input: torch.Tensor):
        resolved = self._resolve_input(input)
        input = input.contiguous()
        nd = self.ndim
        n, c_in = resolved[0], resolved[1]
        in_dims = resolved[2 : 2 + nd]
        out_dims = resolved[2 + nd : 2 + 2 * nd]
        dtype = resolved[-1]
        kernel = self._get_kernel(n, c_in, in_dims, dtype, _device_index(input))
        self.kernel = kernel
        self.n = n
        self.c_in = c_in
        for name, size in zip(_POOL_DIM_NAMES[nd], in_dims, strict=True):
            setattr(self, f"{name}_in", size)
        for name, size in zip(_POOL_DIM_NAMES[nd], out_dims, strict=True):
            setattr(self, f"out_{name}", size)
        self.dtype = dtype
        self._last_roofline_spec = resolved
        return kernel(input)


class MaxPool1dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCL inputs (return_indices=False)."""

    ndim = 1
    _kernel_slot = "max_pool1d_kernel"
    _validate_dtypes = _validate_pool_input_dtypes

    def __init__(
        self,
        kernel_size: int | Tuple[int],
        stride: Optional[int | Tuple[int]] = None,
        padding: int | Tuple[int] = 0,
        dilation: int | Tuple[int] = 1,
        ceil_mode: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            kernel_map=kernel_map,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool1d_kernel": MaxPool1dKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        return _max_pool_roofline(self, indices=False)
