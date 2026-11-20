"""GPT-NeoX style Rotary Position Embedding op (NPU-adapted).

Ported from TileOPs ``{gpu_repo_root}/tileops/ops/rope.py`` -- ``RopeNeoxOp``
(variant ``rope_neox`` with standard ``theta = base^(-2k/d)`` frequencies).

The op computes the variant-specific frequency tables (``cos`` / ``sin``)
lazily at forward time on the **same device as the input tensor** and delegates
the actual rotation to :class:`RopeNeoxKernel`, which receives
``(x, cos, sin)``.

Two layouts are supported:

- ``"1d"``: input ``(seq_len, head_dim)``
- ``"2d"``: input ``(batch, seq_len, num_heads, head_dim)``

Adaptation from GPU (TileOPs) to NPU:

- O1: ``x.is_cuda`` device check -> ``backend.is_device_tensor(x)``.
- O3: the ``tune`` parameter is removed; the kernel constructor takes
      ``(seq_len, head_dim, dtype, layout=..., batch=..., num_heads=...,
      config=None, device_index=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: the ``torch.compile`` custom-op registry (``_OP_REGISTRY`` /
      ``_wrapped`` / ``_register_rope_custom_op``) is removed -- NPU ops
      dispatch through ``forward`` directly.
- O6: the op-specific flow (validate -> bind shapes -> compute cos/sin ->
      kernel dispatch -> reshape output) is preserved unchanged.
- ``eval_roofline``: the GPU ``tileops.perf.formulas.rope_roofline``
  re-implemented inline (the NPU project has no ``tileops.perf`` module; the
  roofline arithmetic is device-agnostic).
"""

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.position_encoding.rope_neox import RopeNeoxKernel
from tileops.ops.op_base import Op

__all__ = ["RopeNeoxOp"]


def _base_freqs(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard RoPE cos/sin tables.

    Args:
        head_dim: Head dimension (must be even).
        seq_len: Sequence length.
        base: Frequency base (default 10000).
        dtype: Output dtype.
        device: Torch device the tables are created on (resolved from the
            input tensor by the caller -- no hard-coded ``"cuda"``).

    Returns:
        ``(cos, sin)`` each of shape ``(seq_len, head_dim // 2)``.
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


class RopeNeoxOp(Op):
    """GPT-NeoX style RoPE op with standard theta frequencies.

    Applies the rotation ``y = x * cos + rotate_half(x) * sin`` where
    ``rotate_half`` splits the head dimension at the midpoint.  The cos/sin
    tables are computed internally from ``base`` and cached per
    ``(seq_len, head_dim, dtype, device)``.  Reference: GPT-NeoX / HuggingFace
    ``transformers`` ``RotaryEmbedding``.

    Args:
        layout: ``"1d"`` for ``(seq_len, head_dim)`` or ``"2d"`` for
            ``(batch, seq_len, num_heads, head_dim)``.
        base: Frequency base (default 10000.0).
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "rope_neox"
    kernel_cls = RopeNeoxKernel

    def __init__(
        self,
        layout: str = "1d",
        base: float = 10000.0,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.base = base
        self.layout = layout
        self.seq_len: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None
        self.batch: Optional[int] = None
        self.num_heads: Optional[int] = None

        self._freq_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x + cos + sin + write y."""
        if self.seq_len is None or self.head_dim is None or self.dtype is None:
            raise RuntimeError("RoPE memory stats are available after the first forward()")
        half = self.head_dim // 2
        elem = self.dtype.itemsize
        cos_sin_elems = self.seq_len * half * 2
        if self.layout == "1d":
            x_elems = self.seq_len * self.head_dim
        else:
            x_elems = self.batch * self.seq_len * self.num_heads * self.head_dim
        return (2 * x_elems + cos_sin_elems) * elem

    def _get_cos_sin(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached cos/sin tables, recomputing if device changed."""
        key = (self.seq_len, self.head_dim, self.dtype, device)
        if key not in self._freq_cache:
            self._freq_cache[key] = _base_freqs(
                self.head_dim, self.seq_len, base=self.base,
                dtype=self.dtype, device=device,
            )
        return self._freq_cache[key]

    def _get_kernel(self, device_index: int | None) -> Kernel:
        key = (
            self.seq_len,
            self.head_dim,
            self.dtype,
            self.layout,
            self.batch,
            self.num_heads,
            device_index,
        )
        if key not in self._kernel_cache:
            # O3: no `tune` argument on the NPU kernel constructor.
            self._kernel_cache[key] = self.kernel_map[self._op_name](
                seq_len=self.seq_len,
                head_dim=self.head_dim,
                dtype=self.dtype,
                layout=self.layout,
                batch=self.batch,
                num_heads=self.num_heads,
                device_index=device_index,
            )
        return self._kernel_cache[key]

    def _validate_and_prepare(self, x: torch.Tensor) -> torch.Tensor:
        """Validate input shape/dtype/device and return a contiguous tensor.

        Raises:
            ValueError: If ``x`` is not on the device, has the wrong dtype, or
                the wrong rank for the configured layout.
        """
        backend = get_device_backend()
        # O1: device check via the device backend instead of ``x.is_cuda``.
        if not backend.is_device_tensor(x):
            raise ValueError(f"Input must be a {backend.name} tensor")
        if self.layout == "1d":
            if x.ndim != 2:
                raise ValueError("RoPE 1d layout expects input shape [seq_len, head_dim]")
            self.seq_len, self.head_dim = x.shape
            self.batch = 1
            self.num_heads = 1
        else:
            if self.layout != "2d":
                raise ValueError(f"Unsupported RoPE layout {self.layout!r}")
            if x.ndim != 4:
                raise ValueError(
                    "RoPE 2d layout expects input shape [batch, seq_len, num_heads, head_dim]"
                )
            self.batch, self.seq_len, self.num_heads, self.head_dim = x.shape
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError("head_dim must be positive and even")
        self.dtype = x.dtype
        self.kernel = self._get_kernel(x.device.index)
        return x.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the RoPE rotation using internally computed cos/sin tables.

        Args:
            x: Input tensor. Shape depends on layout:
                - 1d: ``(seq_len, head_dim)``
                - 2d: ``(batch, seq_len, num_heads, head_dim)``

        Returns:
            Rotated output tensor with the same shape as ``x``.
        """
        x = self._validate_and_prepare(x)
        cos, sin = self._get_cos_sin(x.device)
        return self.kernel(x, cos, sin)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inlined from the GPU ``tileops.perf.formulas.rope_roofline`` (the NPU
        project has no ``tileops.perf`` module; the arithmetic is
        device-agnostic)::

            outer = batch * num_heads if layout == "2d" else 1
            x_elems = outer * seq_len * head_dim
            cos_sin_elems = seq_len * (head_dim // 2) * 2
            flops = 4 * x_elems
            bytes = (2 * x_elems + cos_sin_elems) * elem_bytes

        ``seq_len`` / ``head_dim`` / ``dtype`` (and ``batch`` / ``num_heads``
        for the 2d layout) are bound by the last ``forward()`` call.
        """
        if self.seq_len is None or self.head_dim is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind the dynamic shapes and dtype"
            )
        seq_len = int(self.seq_len)
        head_dim = int(self.head_dim)
        batch = int(self.batch or 1)
        num_heads = int(self.num_heads or 1)
        elem = int(self.dtype.itemsize)
        outer = batch * num_heads if self.layout == "2d" else 1
        x_elems = outer * seq_len * head_dim
        cos_sin_elems = seq_len * (head_dim // 2) * 2
        flops = 4 * x_elems
        nbytes = (2 * x_elems + cos_sin_elems) * elem
        return int(flops), int(nbytes)
