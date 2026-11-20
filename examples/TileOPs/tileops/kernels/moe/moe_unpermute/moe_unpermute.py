"""MoE token unpermute kernel package (cutlass path, NPU-adapted).

Ported from GPU TileOPs ``tileops/kernels/moe/unpermute.py`` -- the
``MoeUnpermuteKernel`` class (weighted scatter-add of padded expert outputs
back to token order).

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory ``_make_unpermute_kernel`` is extracted
    from the GPU repo via ``extract_tl_kernel.py`` (pattern B) and imported
    as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync semantics, ``threads`` removal, padding strategy) are handled
    by the NPU kernel component during re-implementation.

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8: ``tune`` parameter removed; heuristic config selection only
        (``init_config(config)``).  This kernel exposes no tunable knobs, so
        ``default_config`` stays empty on both backends.
    Device check in ``forward`` routed through ``get_device_backend()``
        instead of the CUDA-specific ``mm2_pad.is_cuda``.
    The buffer-overlap guard (GPU ``tileops.kernels.buffer_utils.tensors_overlap``,
    which the NPU project does not ship) is ported verbatim into this module as
    ``_tensors_overlap`` -- it is pure device-agnostic address arithmetic.

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

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/moe/unpermute.py.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .moe_unpermute_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .moe_unpermute_kernel import _make_unpermute_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .moe_unpermute_kernel.perf_opt._make_unpermute_kernel import _make_unpermute_kernel

__all__ = ["MoeUnpermuteKernel"]


def _tensors_overlap(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """True if two tensors share storage AND their byte intervals overlap.

    Device-agnostic port of the GPU helper ``tileops.kernels.buffer_utils.
    tensors_overlap`` (not shipped in the NPU project).  Sharing one storage
    with disjoint intervals is safe and common -- serving frameworks (e.g.
    vLLM) slice inputs, weights, and outputs from a single pre-allocated
    workspace -- so storage identity alone must not be rejected.
    """
    # Pointers live in per-device address spaces, so two tensors on different
    # devices can share a numeric data_ptr by coincidence -- guard before any
    # interval math.
    if t1.device != t2.device:
        return False
    if t1.untyped_storage().data_ptr() != t2.untyped_storage().data_ptr():
        return False

    def _byte_span(t: torch.Tensor) -> int:
        if t.numel() == 0:
            return 0
        max_off = sum((s - 1) * st for s, st in zip(t.shape, t.stride(), strict=True))
        return (max_off + 1) * t.element_size()

    t1_start, t1_end = t1.data_ptr(), t1.data_ptr() + _byte_span(t1)
    t2_start, t2_end = t2.data_ptr(), t2.data_ptr() + _byte_span(t2)
    return t1_start < t2_end and t2_start < t1_end


class MoeUnpermuteKernel(Kernel):
    """MoE token unpermute kernel (cutlass path).

    Scatters padded expert outputs back to original token order with
    weighted reduction using the forward index mapping from moe_permute.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        hidden_size: Hidden dimension H.
        padded_batch_sum: Size of the padded mm2_pad buffer (>= T*K).
        scaling: Scalar multiplied into the reduced output before the
            cast/store (folds ``routed_scaling_factor``).  Defaults to 1.0
            (no scaling).
        dtype: Data type of mm2_pad and output (bf16 or fp16).
        config: Optional config dict.  This kernel exposes no tunable knobs.

    Example:
        >>> kernel = MoeUnpermuteKernel(num_tokens=4, top_k=2, hidden_size=128, padded_batch_sum=512)
        >>> output = kernel(mm2_pad, fwd_idx, topk_weights)
    """

    ascend_mode = "Developer"

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        padded_batch_sum: int,
        scaling: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.padded_batch_sum = padded_batch_sum
        self.dtype = dtype
        self.numel = num_tokens * top_k

        self._unpermute_fn = _make_unpermute_kernel(
            num_tokens, top_k, hidden_size, padded_batch_sum, self.dtype_str, scaling
        )

        # K8: ``tune`` removed -- heuristic config selection only.
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # The GPU kernel exposed no tunable knobs, so the config stays empty.
        return {}

    def forward(
        self,
        mm2_pad: torch.Tensor,
        fwd_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run moe_unpermute.

        Args:
            mm2_pad: [padded_batch_sum, H] bf16/fp16 down-proj output (padded layout).
            fwd_idx: [T*K] int32 forward mapping: flat_idx -> padded slot.
            topk_weights: [T, K] float32 routing weights.
            out: optional [T, H] output buffer to write into and reuse across
                calls. Allocated internally with ``torch.empty`` if omitted.

        Returns:
            output: [T, H] bf16/fp16 (``out`` if provided).
        """
        assert fwd_idx.dtype == torch.int32
        assert topk_weights.dtype == torch.float32
        assert get_device_backend().is_device_tensor(mm2_pad)

        dev = mm2_pad.device
        if out is None:
            output = torch.empty(
                (self.num_tokens, self.hidden_size), dtype=self.dtype, device=dev)
        else:
            if tuple(out.shape) != (self.num_tokens, self.hidden_size):
                raise ValueError(
                    f"out shape must be {(self.num_tokens, self.hidden_size)}, "
                    f"got {tuple(out.shape)}")
            if out.dtype != self.dtype:
                raise ValueError(f"out dtype must be {self.dtype}, got {out.dtype}")
            # The kernel writes ``out`` on mm2_pad's device as a row-major compact
            # tensor; a cross-device or non-contiguous ``out`` would scatter the
            # store to the wrong memory. Reject rather than corrupt silently.
            if out.device != dev:
                raise ValueError(f"out device must be {dev}, got {out.device}")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            # The kernel reads ``mm2_pad`` (gathered per token) while writing
            # ``out`` concurrently, so an ``out`` overlapping ``mm2_pad`` in
            # memory races. Disjoint slices of one workspace buffer are allowed.
            if _tensors_overlap(out, mm2_pad):
                raise ValueError("out must not overlap mm2_pad in memory")
            output = out

        # S5 glue: the integrated NPU factory returns the compiled kernel
        # directly (call it with the tensors).  The Stage-0 scaffold treated
        # ``self._unpermute_fn`` as a zero-arg factory and called it as
        # ``self._unpermute_fn()(tensors)`` (GPU convention), which raises on
        # the NPU JITKernel ("Expected 4 inputs, got 0").
        self._unpermute_fn(mm2_pad, fwd_idx, topk_weights, output)

        return output
