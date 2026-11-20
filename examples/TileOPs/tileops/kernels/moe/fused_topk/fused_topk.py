"""MoE fused top-k routing kernel package (NPU-adapted).

Ported from GPU TileOPs ``tileops/kernels/moe/fused_topk.py`` -- the
``FusedTopKKernel`` class (fused scoring + top-k selection, one warp per token).

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory ``_fused_topk_kernel`` is extracted from the
    GPU repo via ``extract_tl_kernel.py`` (pattern B) and imported as-is.  It
    serves as the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 / K11 adaptations (decorator, grid/sync
    semantics, ``threads`` removal, warp-shuffle rewrite) are handled by the NPU
    kernel component during re-implementation.

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8: ``tune`` parameter removed; heuristic config selection only
        (``init_config(config)``).  ``TOKENS_PER_BLOCK`` is pinned to the
        one-warp-per-token mapping, so ``default_config`` exposes it directly.
    Device check in ``forward`` routed through ``get_device_backend()``
        instead of the CUDA-specific ``gating_output.is_cuda``.

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation -- its inner callable is ``@tilelang.jit`` (CUDA).  It becomes
  callable once the NPU kernel component rewrites it for ``target="npuir"``
  (K1-K4).  Running ``forward`` before that rewrite fails -- this is the
  expected scaffold state.
"""

from typing import Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel

# --- baseline (Stage 3, correction-bias variant) -----------------------------
# The with_correction_bias=True path dispatches to this sibling factory; the
# standard factory raises NotImplementedError for it (DESIGN.md R-1).
from .fused_topk_cb_kernel import _fused_topk_kernel_cb

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/moe/fused_topk.py.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .fused_topk_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .fused_topk_kernel import _fused_topk_kernel

# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .fused_topk_kernel.perf_opt._fused_topk_kernel import _fused_topk_kernel

__all__ = ["FusedTopKKernel"]

_SCORING_FUNCS = ("softmax", "sigmoid")


class FusedTopKKernel(Kernel):
    """MoE top-k routing kernel -- fused scoring + top-k, zero __syncthreads().

    Uses a per-warp algorithm: each warp of 32 lanes independently handles one
    token, keeping expert scores in local registers.  All reductions (softmax
    max/sum, argmax) use warp shfl_xor -- no shared memory, no __syncthreads__.

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Number of experts E.
        top_k: Number of experts to select per token K.
        scoring_func: "softmax" or "sigmoid".
        renormalize: If True, normalize selected weights to sum to 1 (done in PyTorch).
        with_correction_bias: If True, accept a per-expert correction_bias tensor in
            forward(). Adds bias to sigmoid scores for expert selection while writing
            unbiased sigmoid scores to topk_weights. Requires scoring_func="sigmoid".
        config: Optional kernel config dict (key: "TOKENS_PER_BLOCK").

    Note:
        ``dtype`` does not apply. Routing arithmetic is fixed at float32 and
        the outputs are float32 weights plus int32 expert ids; ``forward()``
        up-casts ``gating_output`` of any float dtype before the launch, so no
        element type is selected at construction.
    """

    ascend_mode = "Developer"

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        with_correction_bias: bool = False,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if scoring_func not in _SCORING_FUNCS:
            raise ValueError(
                f"Unsupported scoring_func '{scoring_func}'. "
                f"Expected one of {_SCORING_FUNCS}."
            )
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
        if with_correction_bias and scoring_func != "sigmoid":
            raise ValueError(
                "with_correction_bias=True requires scoring_func='sigmoid'. "
                f"Got scoring_func='{scoring_func}'."
            )

        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias

        self._kernel_fn = (
            _fused_topk_kernel_cb(
                num_tokens, num_experts, top_k, scoring_func, with_correction_bias
            )
            if with_correction_bias
            else _fused_topk_kernel(
                num_tokens, num_experts, top_k, scoring_func, with_correction_bias
            )
        )

        # K8: ``tune`` removed -- heuristic config selection only.
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # TOKENS_PER_BLOCK = 16 gives 512 threads/block (16 warps/block).
        return {"TOKENS_PER_BLOCK": 16}

    def forward(
        self,
        gating_output: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run scoring + top-k selection.

        Args:
            gating_output: [T, E] router logits, any float dtype.
            correction_bias: [E] float32 per-expert bias (required when
                with_correction_bias=True, must be None otherwise).

        Returns:
            topk_weights: [T, K] float32 routing weights.
            topk_ids:     [T, K] int32 expert indices.
        """
        assert gating_output.shape == (self.num_tokens, self.num_experts), (
            f"Expected gating_output shape ({self.num_tokens}, {self.num_experts}), "
            f"got {tuple(gating_output.shape)}"
        )
        assert get_device_backend().is_device_tensor(gating_output), (
            "gating_output must be a device tensor"
        )

        if self.with_correction_bias:
            assert correction_bias is not None, (
                "correction_bias must be provided when with_correction_bias=True"
            )
            assert correction_bias.shape == (self.num_experts,), (
                f"Expected correction_bias shape ({self.num_experts},), "
                f"got {tuple(correction_bias.shape)}"
            )
        else:
            assert correction_bias is None, (
                "correction_bias must be None when with_correction_bias=False"
            )

        logits_f32 = gating_output.to(torch.float32)

        fn = self._kernel_fn(self.config["TOKENS_PER_BLOCK"])
        if self.with_correction_bias:
            bias_f32 = correction_bias.to(torch.float32)
            topk_weights, topk_ids = fn(logits_f32, bias_f32)
        else:
            # The NPU kernel allocates its outputs via @tilelang.jit(out_idx=[1, 2]):
            # call with the single input and unpack the returned (weights, ids).
            topk_weights, topk_ids = fn(logits_f32)

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids
