"""Llama 3.1 (piecewise-scaled) Rotary Position Embedding op (NPU-adapted).

Ported from TileOPs ``{gpu_repo_root}/tileops/ops/rope.py`` --
``RopeLlama31Op`` (variant ``rope_llama31``).  Llama 3.1 shares the neox
rotation with ``RopeNeoxOp`` and differs only in the frequency tables: the
inverse frequencies are piecewise-scaled based on wavelength thresholds
(``low_freq_factor`` / ``high_freq_factor`` / ``original_max_position``) with a
low-frequency ``scale_factor``.

The op computes the variant-specific frequency tables (``cos`` / ``sin``)
lazily at forward time on the **same device as the input tensor** and delegates
the actual rotation to :class:`RopeLlama31Kernel`, which receives
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

import math
from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.position_encoding.rope_llama31 import RopeLlama31Kernel
from tileops.ops.op_base import Op

__all__ = ["RopeLlama31Op"]


def _llama31_freqs(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    scale_factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    original_max_position: int = 8192,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Llama 3.1 piecewise-scaled frequency computation.

    Implements the canonical Llama 3.1 frequency scaling: for each inverse
    frequency the wavelength is compared against the high/low frequency
    wavelength thresholds; high-frequency dimensions are kept, low-frequency
    dimensions are divided by ``scale_factor``, and intermediate dimensions are
    smoothly interpolated between the two.

    Reference: Meta Llama 3.1 model implementation.

    Args:
        head_dim: Head dimension.
        seq_len: Sequence length.
        base: Frequency base.
        scale_factor: Scaling factor for low frequencies.
        low_freq_factor: Threshold for low-frequency wavelengths.
        high_freq_factor: Threshold for high-frequency wavelengths.
        original_max_position: Original maximum position length.
        dtype: Output dtype.
        device: Torch device the tables are created on (resolved from the input
            tensor by the caller -- no hard-coded ``"cuda"``).

    Returns:
        ``(cos, sin)`` each of shape ``(seq_len, head_dim // 2)``.
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))

    low_freq_wavelen = original_max_position / low_freq_factor
    high_freq_wavelen = original_max_position / high_freq_factor

    scaled_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq.item()
        if wavelen < high_freq_wavelen:
            scaled_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            scaled_freqs.append(freq / scale_factor)
        else:
            smooth = (original_max_position / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor)
            scaled_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

    freqs = torch.stack(scaled_freqs)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


class RopeLlama31Op(Op):
    """Llama 3.1 RoPE op with piecewise frequency scaling.

    Applies the rotation ``y = x * cos + rotate_half(x) * sin`` where
    ``rotate_half`` splits the head dimension at the midpoint.  The cos/sin
    tables are computed internally from the Llama 3.1 piecewise-scaled
    frequencies and are cached per ``(seq_len, head_dim, dtype, device)``.
    Reference: Meta Llama 3.1 model implementation.

    Args:
        layout: ``"1d"`` for ``(seq_len, head_dim)`` or ``"2d"`` for
            ``(batch, seq_len, num_heads, head_dim)``.
        base: Frequency base (default 10000.0).
        scale_factor: Scaling factor for low frequencies (default 8.0).
        low_freq_factor: Low-frequency wavelen threshold (default 1.0).
        high_freq_factor: High-frequency wavelen threshold (default 4.0).
        original_max_position: Original max position (default 8192).
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "rope_llama31"
    kernel_cls = RopeLlama31Kernel

    def __init__(
        self,
        layout: str = "1d",
        base: float = 10000.0,
        scale_factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position: int = 8192,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.base = base
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position = original_max_position
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
            self._freq_cache[key] = _llama31_freqs(
                self.head_dim, self.seq_len, base=self.base,
                scale_factor=self.scale_factor,
                low_freq_factor=self.low_freq_factor,
                high_freq_factor=self.high_freq_factor,
                original_max_position=self.original_max_position,
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
        """Apply the Llama 3.1 RoPE rotation using internally computed cos/sin.

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
