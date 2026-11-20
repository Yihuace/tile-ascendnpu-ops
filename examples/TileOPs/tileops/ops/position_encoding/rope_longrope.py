"""LongRoPE (per-dimension rescaled) Rotary Position Embedding op (NPU-adapted).

Ported from TileOPs ``{gpu_repo_root}/tileops/ops/rope.py`` --
``RopeLongRopeOp`` (variant ``rope_longrope``).  LongRoPE shares the neox
rotation with ``RopeNeoxOp`` and differs only in the frequency tables: per-
dimension ``rescale_factors`` (``ext_factors``) multiply the inverse-frequency
divisor, and a scale-dependent amplitude factor (``scaling_factor``) multiplies
the ``cos`` / ``sin`` output.

The op computes the variant-specific frequency tables (``cos`` / ``sin``)
lazily at forward time on the **same device as the input tensor** and delegates
the actual rotation to :class:`RopeLongRopeKernel`, which receives
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
from tileops.kernels.position_encoding.rope_longrope import RopeLongRopeKernel
from tileops.ops.op_base import Op

__all__ = ["RopeLongRopeOp"]


def _longrope_freqs(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    rescale_factors: Optional[torch.Tensor] = None,
    max_position_embeddings: int = 4096,
    original_max_position_embeddings: int = 4096,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LongRoPE per-dimension rescaled frequency computation.

    Implements the canonical LongRoPE formula:

    1. ``divisor = ext_factors[k] * base^(2k/d)`` (``ext_factors`` multiply the
       divisor in the inverse-frequency computation)
    2. ``scaling_factor = sqrt(1 + log(scale) / log(orig_max_pos))`` where
       ``scale = max_pos / orig_max_pos`` (amplitude factor applied to cos/sin
       output when ``scale > 1``)

    Reference: TVM ``rope_freq_longrope`` in ``position_embedding.py``; Ding et
    al., "LongRoPE: Extending LLM Context Window Beyond 2M Tokens".

    Args:
        head_dim: Head dimension.
        seq_len: Sequence length.
        base: Frequency base.
        rescale_factors: Per-dimension rescale factors (``ext_factors``) of
            shape ``(head_dim // 2,)``.  These multiply the divisor in the
            inverse-frequency formula.
        max_position_embeddings: Extended max position length.
        original_max_position_embeddings: Original max position length.
        dtype: Output dtype.
        device: Torch device the tables are created on (resolved from the input
            tensor by the caller -- no hard-coded ``"cuda"``).

    Returns:
        ``(cos, sin)`` each of shape ``(seq_len, head_dim // 2)``.
    """
    half = head_dim // 2
    dim_indices = torch.arange(0, half, device=device, dtype=torch.float32)
    divisor = base ** (dim_indices / half)

    # ext_factors multiply the divisor (matching canonical formula)
    if rescale_factors is not None:
        rf = rescale_factors.to(device=device, dtype=torch.float32)
        divisor = rf * divisor

    freqs = 1.0 / divisor

    # Compute amplitude scaling factor
    scale = max_position_embeddings / original_max_position_embeddings
    if scale > 1.0:
        scaling_factor = math.sqrt(
            1.0 + math.log(scale) / math.log(original_max_position_embeddings)
        )
    else:
        scaling_factor = 1.0

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return (
        (torch.cos(angles) * scaling_factor).to(dtype),
        (torch.sin(angles) * scaling_factor).to(dtype),
    )


class RopeLongRopeOp(Op):
    """LongRoPE op with per-dimension frequency rescaling.

    Applies the rotation ``y = x * cos + rotate_half(x) * sin`` where
    ``rotate_half`` splits the head dimension at the midpoint.  The cos/sin
    tables are computed internally from ``rescale_factors`` (which multiply the
    inverse-frequency divisor) and an amplitude ``scaling_factor``, and are
    cached per ``(seq_len, head_dim, dtype, device)``.  Reference: TVM
    ``rope_freq_longrope``; Ding et al., "LongRoPE".

    Args:
        layout: ``"1d"`` for ``(seq_len, head_dim)`` or ``"2d"`` for
            ``(batch, seq_len, num_heads, head_dim)``.
        base: Frequency base (default 10000.0).
        rescale_factors: Per-dimension rescale factors (``ext_factors``) of
            shape ``(head_dim // 2,)``; ``None`` means no rescaling.
        max_position_embeddings: Extended max position length (default 4096).
        original_max_position_embeddings: Original max position length
            (default 4096).
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "rope_longrope"
    kernel_cls = RopeLongRopeKernel

    def __init__(
        self,
        layout: str = "1d",
        base: float = 10000.0,
        rescale_factors: Optional[torch.Tensor] = None,
        max_position_embeddings: int = 4096,
        original_max_position_embeddings: int = 4096,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.base = base
        self.rescale_factors = rescale_factors
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
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
            self._freq_cache[key] = _longrope_freqs(
                self.head_dim, self.seq_len, base=self.base,
                rescale_factors=self.rescale_factors,
                max_position_embeddings=self.max_position_embeddings,
                original_max_position_embeddings=self.original_max_position_embeddings,
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
        """Apply the LongRoPE rotation using internally computed cos/sin tables.

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
