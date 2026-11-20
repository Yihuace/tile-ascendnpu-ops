"""NPU op package for the position-encoding family.

Ported ops are registered here one at a time.  RopeNeoxOp (GPT-NeoX style
RoPE) is the first ported op, followed by RopeLlama31Op (piecewise-scaled Llama
3.1 RoPE) and RopeLongRopeOp (per-dimension rescaled LongRoPE); both variants
reuse the neox rotation.
"""

from tileops.ops.position_encoding.rope_llama31 import RopeLlama31Op
from tileops.ops.position_encoding.rope_longrope import RopeLongRopeOp
from tileops.ops.position_encoding.rope_neox import RopeNeoxOp

__all__ = ["RopeLlama31Op", "RopeLongRopeOp", "RopeNeoxOp"]
