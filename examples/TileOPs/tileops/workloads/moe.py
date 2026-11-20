"""Workload definitions for the moe (routed-expert) op family (NPU).

Ported from TileOPs ``workloads/moe.py`` -- the ``MoeGroupedGemmNopadWorkload``,
``MoePermuteWorkload``, ``MoeUnpermuteWorkload``, ``FusedTopKWorkload``,
``MoeExpertsWorkload`` and ``FusedMoeWorkload`` subsets (the shared GPU file
also defines MoePermuteAlign / SharedFusedMoe / MoeFusedActivation /
MoeSharedExpertMlp workloads, which are not migrated).

``MoeGroupedGemmNopad3WGFusedActWorkload`` is additionally defined here: the
GPU repo has no dedicated workload for the fused-activation grouped GEMM, so
it is inferred from the op contract (same generation as
``MoeGroupedGemmNopadWorkload`` with ``n = 2 * ffn``).

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"`` in the original TileOPs code.
"""

from __future__ import annotations

import contextlib

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase

def make_expert_sizes_offsets(
    numel: int,
    num_experts: int,
    distribution: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (true_sizes, true_offsets) for a fixed token-to-expert distribution.

    Args:
        numel: Total token-expert pairs (T * top_k).
        num_experts: Number of experts E.
        distribution: "uniform" -- evenly split; "skewed" -- most tokens on first
            20% of experts (one-token floor for the rest).
        device: Device string (backend-resolved; W1).

    Returns:
        true_sizes [E] int32, true_offsets [E] int32.
    """
    if distribution == "uniform":
        base = max(1, numel // num_experts)
        sizes = torch.full((num_experts,), base, dtype=torch.int32, device=device)
        sizes[-1] = numel - base * (num_experts - 1)
    elif distribution == "skewed":
        sizes = torch.ones(num_experts, dtype=torch.int32, device=device)
        extra = numel - num_experts
        top_experts = max(1, num_experts // 5)
        per_top = extra // top_experts
        sizes[:top_experts] += per_top
        sizes[0] += extra - per_top * top_experts
    else:
        raise ValueError(f"unknown distribution: {distribution}")

    offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    assert int(sizes.sum().item()) == numel
    return sizes, offsets


class MoePermuteWorkload(WorkloadBase):
    """Two-input workload for the tight (non-padded) MoE token permute.

    Ported from TileOPs ``workloads/moe.py`` ``MoePermuteWorkload``.  The
    random expert assignment (``randint`` over ``[0, num_experts)``) matches
    the GPU generation logic; only the device is backend-resolved (W1).
    """

    def __init__(self, total_tokens, top_k, num_experts, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        # W1: device resolved from the backend instead of hard-coded "cuda".
        dev = get_device_backend().name
        hidden_states = torch.randn(
            self.total_tokens, self.hidden_size, dtype=self.dtype, device=dev
        )
        topk_ids = torch.randint(
            0, self.num_experts,
            (self.total_tokens, self.top_k),
            dtype=torch.int32, device=dev,
        )
        return hidden_states, topk_ids


class MoeUnpermuteWorkload(WorkloadBase):
    """Three-input workload for the MoE token unpermute (weighted scatter-add).

    Ported from TileOPs ``workloads/moe.py`` ``MoeUnpermuteWorkload``.  The
    padded down-proj output ``mm2_pad`` is generated with ``numel = T*K`` rows
    (no inter-expert padding for this standalone workload), ``fwd_idx`` is a
    random valid permutation of ``[0, numel)``, and ``topk_weights`` are
    float32 in ``[0, 1)``.  Only the device is backend-resolved (W1).
    """

    def __init__(self, total_tokens, top_k, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved from the backend instead of hard-coded "cuda".
        dev = get_device_backend().name
        numel = self.total_tokens * self.top_k
        mm2_pad = torch.randn(numel, self.hidden_size, dtype=self.dtype, device=dev)
        # fwd_idx: simulate a valid mapping: random shuffle of [0, numel)
        fwd_idx = torch.randperm(numel, dtype=torch.int32, device=dev)
        topk_weights = torch.rand(
            self.total_tokens, self.top_k, dtype=torch.float32, device=dev
        )
        return mm2_pad, fwd_idx, topk_weights


class FusedTopKWorkload(WorkloadBase):
    """Single-input workload for the MoE fused top-k routing op.

    Ported from TileOPs ``workloads/moe.py`` ``FusedTopKWorkload``.  The router
    logits ``gating_output`` are drawn from ``torch.randn`` under a fixed seed
    (matching the GPU generation logic); only the device is backend-resolved
    (W1).  ``scoring_func`` / ``renormalize`` are carried for the workload
    description but do not affect input generation.
    """

    def __init__(self, num_tokens, num_experts, top_k, scoring_func, renormalize, dtype):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        torch.manual_seed(42)
        # W1: device resolved from the backend instead of hard-coded "cuda".
        dev = get_device_backend().name
        return (
            torch.randn(self.num_tokens, self.num_experts, dtype=self.dtype, device=dev),
        )


class MoeGroupedGemmNopad3WGFusedActWorkload(WorkloadBase):
    """Four-input workload for the no-pad 3WG grouped GEMM with fused activation.

    The GPU repo has no dedicated workload class for this op (it is exercised
    only inside the fused-MoE tests), so this is inferred from the op contract
    ``A[numel, K] @ B[E, 2*ffn, K]^T -> act(gate) * up``: the input generation
    is identical to :class:`MoeGroupedGemmNopadWorkload` with ``n = 2 * ffn``
    (tight ``a``, per-expert ``b``, and the expert size/offset tables).  Only
    the device is backend-resolved (W1).
    """

    def __init__(
        self,
        numel: int,
        num_experts: int,
        ffn: int,
        k: int,
        dtype: torch.dtype,
        activation: str = "silu_and_mul",
        distribution: str = "uniform",
    ):
        self.numel = numel
        self.num_experts = num_experts
        self.ffn = ffn
        self.k = k
        self.dtype = dtype
        self.activation = activation
        self.distribution = distribution

    def gen_inputs(self):
        torch.manual_seed(42)
        # W1: device resolved from the backend instead of hard-coded "cuda".
        dev = get_device_backend().name
        true_sizes, true_offsets = make_expert_sizes_offsets(
            self.numel, self.num_experts, self.distribution, dev
        )
        # Small scale keeps fp16 accumulation well within the parity tolerance.
        a = torch.randn(self.numel, self.k, dtype=self.dtype, device=dev) * 0.02
        b = torch.randn(
            self.num_experts, 2 * self.ffn, self.k, dtype=self.dtype, device=dev
        ) * 0.02
        return a, b, true_sizes, true_offsets


class MoeExpertsWorkload(WorkloadBase):
    """Five-input workload for the fused MoE experts pipeline.

    Ported from TileOPs ``workloads/moe.py`` ``MoeExpertsWorkload``: the
    generation logic matches the GPU source (seeded ``randn`` hidden states,
    ``* 0.02`` weights, softmax router weights, random expert ids); only the
    device is backend-resolved (W1).
    """

    def __init__(self, num_tokens, num_experts, top_k, hidden_size, ffn_size, dtype):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        # W1: device resolved from the backend instead of hard-coded "cuda".
        dev = get_device_backend().name
        hidden = torch.randn(
            self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev
        )
        # Scale IN PLACE.  ``torch.randn(...) * 0.02`` materialises a second
        # full-size device tensor (the out-of-place mul result) and the NPU
        # caching allocator keeps the freed block reserved.  At DeepSeek shapes
        # (E=256, F=2048, H=7168) that is 21 GiB of weights + 21 GiB transient ->
        # 42 GiB reserved on a 61 GiB card, which is what makes msprof's device
        # memory snapshot fail (B3).  In-place scaling removes the transient and
        # is bit-identical for bf16/fp16 (same elementwise op, same dtype).
        w1 = torch.randn(
            self.num_experts, self.ffn_size * 2, self.hidden_size,
            dtype=self.dtype, device=dev,
        )
        w1.mul_(0.02)
        # Release NOW, not only at the end: measured on NPU, a single trailing
        # empty_cache() does NOT reclaim the generator's cached transient once a
        # later tensor has been carved out of it -- the pool must be flushed
        # between the large allocations for the release to take effect.
        _release_device_cache()
        w2 = torch.randn(
            self.num_experts, self.hidden_size, self.ffn_size,
            dtype=self.dtype, device=dev,
        )
        w2.mul_(0.02)
        _release_device_cache()
        topk_weights = torch.softmax(
            torch.randn(self.num_tokens, self.top_k, dtype=torch.float32, device=dev),
            dim=-1,
        )
        topk_ids = torch.randint(
            0, self.num_experts, (self.num_tokens, self.top_k),
            dtype=torch.int32, device=dev,
        )
        # Release the generator's transient back to the device.  On NPU,
        # ``torch.randn`` / ``Tensor.normal_`` materialise ~2x the output as an
        # internal temporary; the caching allocator then keeps that freed block
        # RESERVED.  At DeepSeek shapes (E=256, F=2048, H=7168) the weights are
        # 21 GiB, so the process sits on ~42 GiB reserved even though only 21 GiB
        # is live -- which is what starves msprof's device memory snapshot (B3).
        _release_device_cache()
        return hidden, w1, w2, topk_weights, topk_ids


class FusedMoeWorkload(WorkloadBase):
    """Five-input workload for a single FusedMoe benchmark configuration.

    Ported from TileOPs ``workloads/moe.py`` ``FusedMoeWorkload``: the
    generation logic matches the GPU source (seeded ``randn`` hidden states,
    float32 router logits, ``* 0.02`` weights, optional float32 correction
    bias); only the device is backend-resolved (W1).

    Returns ``(hidden, gating, correction_bias, w_gate_up, w_down)`` where
    ``correction_bias`` is ``None`` when ``with_correction_bias`` is False.
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        scoring_func: str,
        renormalize: bool,
        with_correction_bias: bool,
        routed_scaling_factor: float,
        dtype: torch.dtype,
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

    def gen_inputs(self):
        torch.manual_seed(42)
        # W1: device resolved from the backend instead of hard-coded "cuda".
        dev = get_device_backend().name
        hidden = torch.randn(
            self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev,
        )
        gating = torch.randn(
            self.num_tokens, self.num_experts, dtype=torch.float32, device=dev,
        )
        correction_bias = (
            torch.randn(self.num_experts, dtype=torch.float32, device=dev) * 0.1
            if self.with_correction_bias else None
        )
        # In-place scaling: see MoeExpertsWorkload.gen_inputs for why (the
        # out-of-place ``* 0.02`` doubles the device peak at DeepSeek shapes and
        # breaks msprof's device memory snapshot).
        w_gate_up = torch.randn(
            self.num_experts, self.ffn_size * 2, self.hidden_size,
            dtype=self.dtype, device=dev,
        )
        w_gate_up.mul_(0.02)
        _release_device_cache()
        w_down = torch.randn(
            self.num_experts, self.hidden_size, self.ffn_size,
            dtype=self.dtype, device=dev,
        )
        w_down.mul_(0.02)
        _release_device_cache()
        return hidden, gating, correction_bias, w_gate_up, w_down


class MoeGroupedGemmNopadWorkload(WorkloadBase):
    """Tight A, per-expert weights B, and the expert size/offset tables."""

    def __init__(
        self,
        numel: int,
        num_experts: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        distribution: str = "uniform",
    ):
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.dtype = dtype
        self.distribution = distribution

    def gen_inputs(self):
        torch.manual_seed(42)
        # W1: device resolved from the backend instead of hard-coded "cuda".
        dev = get_device_backend().name
        true_sizes, true_offsets = make_expert_sizes_offsets(
            self.numel, self.num_experts, self.distribution, dev
        )
        # Small scale keeps fp16 accumulation well within the parity tolerance.
        a = torch.randn(self.numel, self.k, dtype=self.dtype, device=dev) * 0.02
        b = torch.randn(self.num_experts, self.n, self.k, dtype=self.dtype, device=dev) * 0.02
        return a, b, true_sizes, true_offsets
