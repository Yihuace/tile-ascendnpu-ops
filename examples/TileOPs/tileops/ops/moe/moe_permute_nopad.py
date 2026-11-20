"""MoE permute op (no-pad variant): counting sort + tight gather without block_m padding.

Ported from GPU TileOPs ``tileops/ops/moe/routed_expert/permute_nopad.py`` to
the NPU backend.

Adaptation from GPU (TileOPs) to NPU:

- O1: the ``x.is_cuda`` device checks in ``forward`` are routed through
  :func:`get_device_backend` via ``backend.is_device_tensor(x)``.
- O3: no ``tune`` parameter existed in the GPU op, so nothing to remove; the
  kernel constructor is called with the GPU positional argument list.
- O4: ``from ...op_base import Op`` becomes the NPU package import
  ``from tileops.ops.op_base import Op``.
- O5: the GPU op did not use ``compile_boundary`` (nothing to remove).
- O6: the op-specific flow (constructor signature, validation, ``_get_kernel``
  cache, ``forward`` pass-through, ``eval_roofline``) is preserved unchanged.

``eval_roofline()`` implements the GPU manifest-declared roofline
(``flops = 0``, ``bytes = (T*H + T*K*H) * elem_bytes + (E+1)*8 + 2*T*K*4 +
E*8``) inline -- roofline arithmetic is device-agnostic.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.moe_permute_nopad import MoePermuteNopadKernel
from tileops.ops.op_base import Op

__all__ = ["MoePermuteNopadFwdOp"]


class MoePermuteNopadFwdOp(Op):
    """Route tokens to tight (non-padded) expert-contiguous layout.

    The output perm_h has exactly T*K rows with no
    inter-expert padding, enabling smaller intermediate tensors throughout
    the MoE pipeline.

    Args:
        total_tokens: Optional committed number of input tokens T. Preferred
            API infers it from ``hidden_states.shape[0]``.
        top_k: Optional committed number of experts selected per token K.
            Preferred API infers it from ``topk_ids.shape[1]``.
        num_experts: Total number of experts E (global count).
        hidden_size: Optional committed hidden dimension H. Preferred API
            infers it from ``hidden_states.shape[1]``.
        dtype: Data type of hidden_states (bf16 or fp16). If omitted,
            inferred from ``hidden_states``.
        expert_map: Optional [E_global] int32 tensor mapping global expert ids
            to local ids (-1 = not on this rank).  When provided, only local
            token-expert pairs are counted; non-local positions get fwd_idx = -1.
        kernel_map: Optional kernel override dict.

    Example:
        >>> op = MoePermuteNopadFwdOp(num_experts=8)
        >>> perm_h, offsets, sizes, expert_offset, fwd_idx = op(hidden_states, topk_ids)
    """

    def __init__(
        self,
        total_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        num_experts: Optional[int] = None,
        hidden_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        expert_map: Optional[torch.Tensor] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype
        self._committed_total_tokens = total_tokens
        self._committed_top_k = top_k
        self._committed_hidden_size = hidden_size
        self._committed_dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.expert_map = expert_map
        self._kernel_cache: Dict[tuple[int, int, int, int, torch.dtype, int | None], Kernel] = {}

    def eval_roofline(self) -> tuple[int, int]:
        if (
            not hasattr(self, "hidden_states_shape")
            or not hasattr(self, "topk_ids_shape")
            or self.dtype is None
            or self.num_experts is None
        ):
            raise ValueError(
                "MoePermuteNopadFwdOp.eval_roofline() requires a prior forward() "
                "to bind hidden_states_shape, topk_ids_shape, dtype, and num_experts"
            )
        total_tokens, hidden_size = self.hidden_states_shape
        top_k = self.topk_ids_shape[1]
        elem_bytes = self.dtype.itemsize
        nbytes = (
            (total_tokens * hidden_size + total_tokens * top_k * hidden_size)
            * elem_bytes
            + (self.num_experts + 1) * 8
            + 2 * total_tokens * top_k * 4
            + self.num_experts * 8
        )
        return 0, int(nbytes)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"permute_nopad_kernel": MoePermuteNopadKernel}

    def _get_kernel(
        self,
        total_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (total_tokens, top_k, num_experts, hidden_size, dtype, device_index)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["permute_nopad_kernel"](
                total_tokens, top_k, num_experts, hidden_size, dtype, self.expert_map
            )
        return self._kernel_cache[key]

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run moe_permute without padding.

        Args:
            hidden_states: [T, H] input activations (bf16/fp16).
            topk_ids: [T, K] int32 expert assignments (global ids).

        Returns:
            perm_h:                    [T*K, H] tight hidden states
            true_offsets:              [E_local] int32 tight start per local expert
            true_sizes:                [E_local] int32 true token count per local expert
            expert_first_token_offset: [E_local+1] int64 non-padded prefix-sum
            fwd_idx:                   [T*K] int32 forward mapping: flat_idx -> tight slot
                                       (-1 for non-local pairs when expert_map is set)
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(hidden_states):
            raise ValueError(f"hidden_states must be a {backend.name} tensor")
        if not backend.is_device_tensor(topk_ids):
            raise ValueError(f"topk_ids must be a {backend.name} tensor")
        if hidden_states.device != topk_ids.device:
            raise ValueError(
                f"Expected hidden_states and topk_ids to be on the same device, "
                f"got {hidden_states.device} and {topk_ids.device}"
            )
        if hidden_states.ndim != 2:
            raise ValueError(
                f"Expected hidden_states to be 2D [T, H], got {hidden_states.ndim}D"
            )
        if topk_ids.ndim != 2:
            raise ValueError(
                f"Expected topk_ids to be 2D [T, K], got {topk_ids.ndim}D"
            )
        total_tokens, hidden_size = hidden_states.shape
        topk_tokens, top_k = topk_ids.shape
        if topk_tokens != total_tokens:
            raise ValueError(
                f"Expected topk_ids.shape[0] == hidden_states.shape[0] "
                f"({total_tokens}), got {topk_tokens}"
            )
        if (
            self._committed_total_tokens is not None
            and total_tokens != self._committed_total_tokens
        ):
            raise ValueError(
                f"Expected total_tokens={self._committed_total_tokens}, got {total_tokens}"
            )
        if self._committed_top_k is not None and top_k != self._committed_top_k:
            raise ValueError(f"Expected top_k={self._committed_top_k}, got {top_k}")
        if (
            self._committed_hidden_size is not None
            and hidden_size != self._committed_hidden_size
        ):
            raise ValueError(
                f"Expected hidden_size={self._committed_hidden_size}, got {hidden_size}"
            )
        if self.num_experts is None:
            raise ValueError("num_experts must be provided at construction time")
        if (
            self._committed_dtype is not None
            and hidden_states.dtype != self._committed_dtype
        ):
            raise ValueError(
                f"Expected hidden_states.dtype {self._committed_dtype}, got {hidden_states.dtype}"
            )
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "Expected hidden_states.dtype to be torch.float16 or "
                f"torch.bfloat16, got {hidden_states.dtype}"
            )
        if topk_ids.dtype != torch.int32:
            raise ValueError(f"Expected topk_ids.dtype torch.int32, got {topk_ids.dtype}")

        dtype = hidden_states.dtype
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.hidden_states_shape = tuple(hidden_states.shape)
        self.topk_ids_shape = tuple(topk_ids.shape)
        kernel = self._get_kernel(
            total_tokens, top_k, self.num_experts, hidden_size, dtype,
            hidden_states.device.index,
        )
        return kernel(hidden_states, topk_ids)
