"""NPU kernel package for the position-encoding family.

Ported ops are registered here one at a time.  RopeNeoxKernel (GPT-NeoX style
RoPE) is the first ported kernel, followed by RopeLlama31Kernel (piecewise-
scaled Llama 3.1 RoPE) and RopeLongRopeKernel (per-dimension rescaled LongRoPE);
both variants reuse the neox rotation.
"""

from tileops.kernels.position_encoding.rope_llama31 import RopeLlama31Kernel
from tileops.kernels.position_encoding.rope_longrope import RopeLongRopeKernel
from tileops.kernels.position_encoding.rope_neox import RopeNeoxKernel

__all__ = ["RopeLlama31Kernel", "RopeLongRopeKernel", "RopeNeoxKernel"]
