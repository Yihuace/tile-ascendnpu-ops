"""Activation registry for MoE expert pipelines (NPU).

Ported from GPU TileOPs ``tileops/ops/moe/_activation.py`` to the NPU backend.

This module is **host glue only** -- it builds the NPU-adapted activation
sub-Ops and contains no device-specific code beyond the activation registry.

Adaptation from GPU (TileOPs) to NPU:

- The GPU module imported ``FusedGatedOp`` from the ``tileops.ops.elementwise``
  package ``__init__``.  In the NPU project ``FusedGatedOp`` lives in
  ``tileops.ops.elementwise.silu_and_mul`` (the GPU ``_base.py`` base is
  inlined there), so the import is routed to the module that defines it.
- ``SiluAndMulFwdOp`` / ``GeluAndMulFwdOp`` are imported from the NPU
  elementwise package; both are already migrated.
- O6: ``build_activation_op`` logic (registry lookup, ``ValueError`` on an
  unknown key, construction contract) is preserved verbatim.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.ops.elementwise.gelu_and_mul import GeluAndMulFwdOp
from tileops.ops.elementwise.silu_and_mul import FusedGatedOp, SiluAndMulFwdOp

__all__ = ["build_activation_op"]

_ACTIVATION_REGISTRY: Dict[str, type] = {
    "silu_and_mul": SiluAndMulFwdOp,
    "gelu_and_mul": GeluAndMulFwdOp,
}


def build_activation_op(
    activation: str,
    M: int,
    N: int,
    dtype: torch.dtype,
    kernel_map: Optional[Dict[str, Kernel]] = None,
) -> FusedGatedOp:
    """Construct the activation sub-Op for an MoE experts pipeline.

    Args:
        activation: One of the keys in ``_ACTIVATION_REGISTRY``.
        M: Row count of the (M, 2N) gate_up tensor.
        N: Half column dim -- the activation output width (= ffn_size).
        dtype: Activation/output dtype.
        kernel_map: Forwarded to the inner activation op for kernel dispatch.

    Returns:
        A ``FusedGatedOp`` instance configured for ``(M, 2N) -> (M, N)``.

    Raises:
        ValueError: If ``activation`` is not a registered key.
    """
    if activation not in _ACTIVATION_REGISTRY:
        allowed = ", ".join(sorted(_ACTIVATION_REGISTRY))
        raise ValueError(
            f"activation must be one of [{allowed}], got {activation!r}"
        )
    return _ACTIVATION_REGISTRY[activation](
        M=M, N=N, dtype=dtype, kernel_map=kernel_map,
    )
