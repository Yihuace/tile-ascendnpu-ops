"""No-DP/EP prepare-and-finalize: quantization pass-through, local weighted reduction.

Ported from GPU TileOPs ``tileops/ops/moe/prepare_finalize/no_dp_ep.py`` to the
NPU backend.

This module is **host glue only** -- it contains no TileLang kernel and no
device-specific code.  It is a strategy class (``FusedMoEPrepareAndFinalize``
subclass) used by ``FusedMoe``; its compute is delegated to the
``WeightedReduce`` instance supplied by the experts op at ``finalize()`` time.

Adaptation from GPU (TileOPs) to NPU:

- O4/O6: the GPU imported the ABCs from
  ``tileops.ops.moe.routed_expert.abc``; the NPU project flattens the GPU
  ``routed_expert/`` sub-package, so the ABCs live in
  :mod:`tileops.ops.moe._routed_expert_abc`.  The class body
  (``prepare`` pass-through and ``finalize`` delegation) is preserved verbatim
  -- it is device-agnostic.
"""

from __future__ import annotations

from torch import Tensor

from tileops.ops.moe._routed_expert_abc import (
    FusedMoEPrepareAndFinalize,
    PrepareResult,
    WeightedReduce,
)

__all__ = ["MoEPrepareAndFinalizeNoDPEP"]


class MoEPrepareAndFinalizeNoDPEP(FusedMoEPrepareAndFinalize):
    """No Data-Parallel / Expert-Parallel prepare and finalize.

    prepare(): bf16/fp16 pass-through (no quantization, no dispatch).
    finalize(): delegates to weight_and_reduce (typically WeightedReduceNoOp).

    For EP, replace with a subclass that adds All-to-All dispatch/gather.
    """

    def prepare(
        self,
        hidden: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        num_experts: int,
        expert_map: Tensor | None,
    ) -> PrepareResult:
        return PrepareResult(
            hidden_q=hidden,
            scale=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    def finalize(
        self,
        output: Tensor,
        expert_out: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        weight_and_reduce: WeightedReduce,
    ) -> None:
        weight_and_reduce.apply(output, expert_out, topk_weights, topk_ids)
