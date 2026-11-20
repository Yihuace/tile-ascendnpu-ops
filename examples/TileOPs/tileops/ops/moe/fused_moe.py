"""Routed Mixture-of-Experts (MoE) FFN operator (NPU).

Ported from GPU TileOPs ``tileops/ops/moe/fused_moe.py``.  The GPU file defines
two manifest identities that share one composite implementation:

- ``FusedMoeFwdOp`` -- routing + expert FFN without a routing correction bias
  (Qwen3 / DeepSeek-V3 style).
- ``FusedMoeFwdCbFwdOp`` -- same flow but accepts a per-expert correction bias
  applied during top-k selection (Kimi K2 style).

Both manifest identities are migrated here.  They share the ``FusedMoe`` base
class below and differ only in the pinned correction-bias flag (and the
default ``scoring_func``).

The shared core (``FusedMoe``) is ported in full because both subclasses
inherit it.  It wires ``FusedTopKOp`` (routing),
``MoEPrepareAndFinalizeNoDPEP`` (quantization / EP dispatch), and
``FusedMoEExpertsNopadPersistent3WGFwdOp`` (permute + GEMM + unpermute).  This
op is a **composition orchestrator**: it owns no TileLang kernel of its own --
every sub-kernel is owned by the inner Ops it wires.

Adaptation from GPU (TileOPs) to NPU:

- O4: ``from tileops.utils import get_sm_version`` was never used here; the
  manifest-codegen path is replaced by an explicit ``eval_roofline`` on the
  base class (see below).  ``from ..op_base import Op`` becomes
  ``from tileops.ops.op_base import Op``.
- O5: the GPU op did not use ``compile_boundary`` (nothing to remove).
- O6: constructor signature, ``default_kernel_map``, validation guards and the
  ``forward`` pipeline (top-k -> prepare -> experts -> finalize) are preserved
  unchanged; only the import paths are adapted.  The GPU project's
  ``routed_expert/`` sub-package is flattened into
  :mod:`tileops.ops.moe.fused_moe_experts_nopad_persistent_3wg` and
  :mod:`tileops.ops.moe._routed_expert_abc`; the ``prepare_finalize``
  sub-package keeps the GPU layout.
- Roofline: the GPU manifest declared ``func:
  "tileops.perf.formulas.fused_moe_fwd_bytes"``.  The NPU project has no
  ``tileops.perf`` package, so :meth:`FusedMoe.eval_roofline` implements that
  formula inline (below) and the NPU manifest stores the same formula as
  explicit ``flops``/``bytes`` expressions (roofline arithmetic is
  device-agnostic).
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.ops.moe._routed_expert_abc import (
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalize,
)
from tileops.ops.moe.fused_moe_experts_nopad_persistent_3wg import (
    FusedMoEExpertsNopadPersistent3WGFwdOp,
)
from tileops.ops.moe.fused_topk import FusedTopKOp
from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP
from tileops.ops.op_base import Op

__all__ = ["FusedMoe", "FusedMoeFwdCbFwdOp", "FusedMoeFwdOp"]


class FusedMoe(Op):
    """Shared composite implementation for routed MoE FFN ops.

    Concrete manifest identities (``FusedMoeFwdOp`` and
    ``FusedMoeFwdCbFwdOp``) subclass this to pin the correction-bias variant;
    both share the routing-and-expert pipeline below.

    Args:
        num_tokens: T -- number of input tokens.
        num_experts: E -- total number of experts (global count).
        top_k: K -- experts selected per token.
        hidden_size: H -- model hidden dimension.
        ffn_size: F -- per-expert intermediate dimension.
        scoring_func: "softmax" (Qwen3) or "sigmoid" (Kimi K2 / DeepSeek-V3).
        renormalize: Renormalize top-k weights to sum to 1.
        with_correction_bias: Accept `correction_bias` during routing.
        routed_scaling_factor: Multiplier on expert output (Kimi K2: 2.827).
        dtype: Activation and weight dtype.
        expert_map: [E_global] int32 for Expert Parallel local filtering.
        prepare_finalize: Override the PrepareAndFinalize implementation.
        experts: Override the Experts implementation.
        kernel_map: Override the dispatched kernel map.
    """

    # msprof kernel-name filter (see tileops.benchmark.msprof).  This op is a
    # composition orchestrator -- ``forward`` dispatches fused_topk -> prepare ->
    # experts (permute / 3WG GEMM / activation / unpermute) -> finalize.  msprof's
    # ``--kernel-name`` accepts a single name, so the filter is pinned to the
    # composition's dominant device entry (the mix AIC entry of the grouped GEMM).
    # The reported latency/roofline therefore covers the GEMM side only; a
    # whole-op number requires events mode or a multi-kernel metric-layer
    # aggregation.
    msprof_kernel_name = "main_mix_aic"

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        with_correction_bias: bool = False,
        routed_scaling_factor: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None,
        experts: Optional[FusedMoEExpertsModular] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        activation: str = "silu_and_mul",
        use_fused_activation: bool = False,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.dtype = dtype
        self.expert_map = expert_map

        self.dispatch_kernel(kernel_map)

        self._fused_topk = FusedTopKOp(
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            with_correction_bias=with_correction_bias,
            kernel_map=kernel_map,
        )

        self._prepare: FusedMoEPrepareAndFinalize = (
            prepare_finalize if prepare_finalize is not None
            else MoEPrepareAndFinalizeNoDPEP()
        )

        if prepare_finalize is not None and experts is None:
            raise ValueError(
                "prepare_finalize may change the dispatched token count (T'); "
                "you must also supply a matching experts= instance sized for T'."
            )

        if experts is not None:
            # use_fused_activation only configures the default experts
            # implementation; an injected experts instance must be pre-configured
            # by the caller. Reject the combination rather than silently ignoring
            # the flag.
            if use_fused_activation:
                raise ValueError(
                    "use_fused_activation=True cannot be combined with an injected "
                    "experts= instance; the flag only configures the default experts "
                    "implementation. Build the experts instance with "
                    "use_fused_activation=True yourself instead."
                )
            # All in-tree FusedMoEExperts*FwdOp set self.activation in __init__.
            # We require it to be present rather than falling back silently --
            # a missing attribute on a third-party experts implementation would
            # otherwise let a non-matching `activation` argument be silently
            # accepted, producing a wrong-activation pipeline.
            if not hasattr(experts, "activation"):
                raise ValueError(
                    f"injected experts instance ({type(experts).__name__}) "
                    "is missing the required `.activation` attribute. "
                    "Set it in __init__ to the activation string this experts "
                    "implementation uses (e.g. 'silu_and_mul')."
                )
            experts_activation = experts.activation
            # Reject only conflicting non-default values. Passing the default
            # ("silu_and_mul") alongside experts= is silently accepted because
            # it cannot be distinguished from the bare experts= call. Passing
            # an explicit value that matches the injected experts' activation
            # is also accepted.
            if activation != "silu_and_mul" and activation != experts_activation:
                raise ValueError(
                    "activation conflicts with the injected experts instance: "
                    f"got activation={activation!r}, "
                    f"experts.activation={experts_activation!r} "
                    f"(experts={type(experts).__name__}). "
                    "Either omit activation or pass the same value."
                )
            self.activation = experts_activation
            self._experts: FusedMoEExpertsModular = experts
        else:
            self.activation = activation
            self._experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                activation=activation,
                routed_scaling_factor=routed_scaling_factor,
                dtype=dtype,
                expert_map=expert_map,
                kernel_map=kernel_map,
                use_fused_activation=use_fused_activation,
            )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {}

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inlines the GPU manifest roofline function
        ``tileops.perf.formulas.fused_moe_fwd_bytes`` (device-agnostic): byte
        traffic mixes float32 gating (and, for the Cb variant, float32
        correction bias) with hidden-states / weights at ``self.dtype``, so a
        single ``elem_bytes`` cannot express the total.
        """
        T = int(self.num_tokens)
        E = int(self.num_experts)
        K = int(self.top_k)
        H = int(self.hidden_size)
        F = int(self.ffn_size)
        elem_bytes = self.dtype.itemsize

        flops = T * K * 6 * F * H
        weight_bytes = E * 3 * F * H * elem_bytes
        token_bytes = 2 * T * H * elem_bytes
        gating_bytes = T * E * 4  # float32 logits
        bias_bytes = E * 4 if getattr(self, "with_correction_bias", False) else 0
        return int(flops), int(weight_bytes + token_bytes + gating_bytes + bias_bytes)

    def forward(
        self,
        hidden_states: torch.Tensor,                    # [T, H]
        gating_output: torch.Tensor,                    # [T, E]
        w_gate_up: torch.Tensor,                        # [E, 2*F, H]
        w_down: torch.Tensor,                           # [E, H, F]
        correction_bias: Optional[torch.Tensor] = None, # [E] float32
    ) -> torch.Tensor:                                  # [T, H]
        topk_weights, topk_ids = self._fused_topk(gating_output, correction_bias)

        r = self._prepare.prepare(
            hidden_states, topk_weights, topk_ids,
            self.num_experts, expert_map=self.expert_map,
        )

        T_prime = r.hidden_q.shape[0]
        ws1_shape, ws2_shape = self._experts.workspace_shapes(
            T_prime, self.ffn_size, self.hidden_size,
            self.top_k, self.num_experts,
        )
        ws1 = hidden_states.new_empty(ws1_shape)
        ws2 = hidden_states.new_empty(ws2_shape)

        output = hidden_states.new_empty(hidden_states.shape)
        expert_out_shape = self._experts.output_shape(T_prime, self.hidden_size)
        expert_out = output if expert_out_shape == tuple(hidden_states.shape) else hidden_states.new_empty(expert_out_shape)
        self._experts.forward(
            expert_out, r.hidden_q, w_gate_up, w_down,
            r.topk_weights, r.topk_ids,
            expert_map=self.expert_map,
            workspace1=ws1, workspace2=ws2,
            num_experts=self.num_experts,
        )

        self._prepare.finalize(
            output, expert_out,
            r.topk_weights, r.topk_ids,
            self._experts.make_weighted_reduce(),
        )
        return output


class FusedMoeFwdOp(FusedMoe):
    """Routed MoE FFN without a routing correction bias.

    Covers Qwen3 (softmax) and DeepSeek-V3 (sigmoid) style configurations
    where top-k is computed directly from the gating scores.
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        routed_scaling_factor: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None,
        experts: Optional[FusedMoEExpertsModular] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        activation: str = "silu_and_mul",
        use_fused_activation: bool = False,
    ):
        super().__init__(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            scoring_func=scoring_func,
            renormalize=renormalize,
            with_correction_bias=False,
            routed_scaling_factor=routed_scaling_factor,
            dtype=dtype,
            expert_map=expert_map,
            prepare_finalize=prepare_finalize,
            experts=experts,
            kernel_map=kernel_map,
            activation=activation,
            use_fused_activation=use_fused_activation,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(hidden_states, gating_output, w_gate_up, w_down, None)


class FusedMoeFwdCbFwdOp(FusedMoe):
    """Routed MoE FFN with a per-expert routing correction bias.

    Covers Kimi K2 style configurations: top-k is selected from
    ``sigmoid(score) + correction_bias`` while the final weights use the
    original (unbiased) sigmoid scores, renormalized.  This is the
    correction-bias variant of :class:`FusedMoeFwdOp`; the pipeline is
    identical apart from the extra ``correction_bias`` input and the pinned
    ``with_correction_bias=True`` / default ``scoring_func="sigmoid"``.
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        scoring_func: str = "sigmoid",
        renormalize: bool = False,
        routed_scaling_factor: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None,
        experts: Optional[FusedMoEExpertsModular] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        activation: str = "silu_and_mul",
        use_fused_activation: bool = False,
    ):
        super().__init__(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            scoring_func=scoring_func,
            renormalize=renormalize,
            with_correction_bias=True,
            routed_scaling_factor=routed_scaling_factor,
            dtype=dtype,
            expert_map=expert_map,
            prepare_finalize=prepare_finalize,
            experts=experts,
            kernel_map=kernel_map,
            activation=activation,
            use_fused_activation=use_fused_activation,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(
            hidden_states, gating_output, w_gate_up, w_down, correction_bias,
        )
