"""MoE token permute kernel package (no-pad variant, NPU-adapted).

Ported from GPU TileOPs ``tileops/kernels/moe/permute_nopad.py`` -- the
``MoePermuteNopadKernel`` class (counting-sort scan + tight gather) and its
``_atomic_helper.h`` scatter helper.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (migrated + integrated):
    The three TileLang kernel factories (``_make_scan_kernel_nopad``,
    ``_make_scan_kernel_nopad_ep``, ``_make_gather_kernel_nopad``) are the
    NPU (``target="npuir"``) Stage-3 products re-implemented by the NPU kernel
    component and integrated under ``.moe_permute_nopad_kernel``.  Their
    Stage-0 GPU forms (extracted from the GPU repo via
    ``extract_tl_kernel.py``) are kept only as the reference the NPU products
    were derived from.

  **Part B -- Kernel class** (ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8: ``tune`` parameter removed; heuristic config selection only
        (``init_config(config)``).
    K9: ``threads`` removed from ``default_config`` and from the scan
        factory dispatch (the GPU scan callable took ``threads``;
        ``_SCAN_THREADS`` is the removal target on the NPU side).
    Device checks in ``forward`` routed through ``get_device_backend()``
    instead of the CUDA-specific ``tensor.is_cuda``.

  Both scan dispatches (the non-EP ``_make_scan_kernel_nopad`` and the EP
  ``_make_scan_kernel_nopad_ep`` variant) now bind their respective integrated
  NPU factory; ``forward`` sizes the per-core scratch tables from the active
  factory's ``VECTOR_CORES`` and routes ``expert_map`` through the EP variant.
"""

from typing import Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factories extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/moe/permute_nopad.py.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .moe_permute_nopad_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .moe_permute_nopad_kernel import (
    _make_gather_kernel_nopad,
    _make_scan_kernel_nopad,
    _make_scan_kernel_nopad_ep,
)
# S5 glue: each integrated scan factory partitions its grid as
# ``NB = min(VECTOR_CORES, numel)`` and requires one ``[NB, E_local]`` scratch
# table per core (bh / slot_base / run_buf).  The wrapper must size those tables
# exactly like the factory does, so bind the same constant each active baseline
# factory uses (the non-EP and EP variants live in separate modules).
from .moe_permute_nopad_kernel._make_scan_kernel_nopad import (
    scan_num_blocks as _scan_num_blocks,
)
from .moe_permute_nopad_kernel._make_scan_kernel_nopad_ep import (
    scan_num_blocks_ep as _scan_num_blocks_ep,
)
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .moe_permute_nopad_kernel.perf_opt._make_gather_kernel_nopad import _make_gather_kernel_nopad
# from .moe_permute_nopad_kernel.perf_opt._make_scan_kernel_nopad import _make_scan_kernel_nopad
# from .moe_permute_nopad_kernel.perf_opt._make_scan_kernel_nopad_ep import _make_scan_kernel_nopad_ep

__all__ = ["MoePermuteNopadKernel"]


class MoePermuteNopadKernel(Kernel):
    """MoE token permute kernel without block_m-aligned padding.

    Sorts tokens into tight expert-contiguous order (no inter-expert gaps).
    Output perm_h has exactly M_local rows, where M_local = T*K when
    expert_map is None (all experts local), or M_local <= T*K when expert_map
    filters out non-local experts.  Non-local positions get fwd_idx = -1.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        num_experts: Total number of experts E (global count).
        hidden_size: Hidden dimension H.
        dtype: Data type of hidden_states.
        expert_map: Optional [E_global] int32 tensor mapping global expert ids
            to local ids (-1 = not on this rank).  When provided, only local
            token-expert pairs are included in the permuted output.
        config: Optional config dict.  The GPU scan thread count is removed on
            NPU (K9), so the default config is empty.

    Example:
        >>> kernel = MoePermuteNopadKernel(num_tokens=4, top_k=2, num_experts=8, hidden_size=128)
        >>> perm_h, offsets, sizes, expert_offset, fwd_idx = kernel(hidden_states, topk_ids)
    """

    ascend_mode = "Developer"

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.numel = num_tokens * top_k
        self.expert_map = expert_map

        if expert_map is not None:
            assert expert_map.dtype == torch.int32
            self.num_experts_local = int((expert_map >= 0).sum().item())
            # S5 re-integration: the NPU EP scan variant
            # (``_make_scan_kernel_nopad_ep``) is now migrated and integrated,
            # so the ``expert_map`` path binds the NPU factory instead of
            # rejecting the path.  Same 4-arg factory signature as the Stage-0
            # GPU form -- ``(numel, e_global, e_local, top_k)`` -- where
            # ``e_local`` is the local expert count derived above.
            self._scan_fn = _make_scan_kernel_nopad_ep(
                self.numel, num_experts, self.num_experts_local, top_k
            )
        else:
            self.num_experts_local = num_experts
            self._scan_fn = _make_scan_kernel_nopad(self.numel, num_experts, top_k)

        self._gather_fn = _make_gather_kernel_nopad(
            num_tokens, self.numel, hidden_size, self.dtype_str
        )

        # K8: ``tune`` removed -- heuristic config selection only.
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # K9: the GPU ``{"threads": _SCAN_THREADS}`` config is removed -- NPU
        # has no CUDA thread count; the scan algorithm is re-expressed by the
        # NPU kernel component.  No heuristic config parameter remains here.
        return {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run moe_permute without padding.

        Args:
            hidden_states: [T, H] input activations.
            topk_ids: [T, K] int32 expert assignments (global ids).

        Returns:
            perm_h:                    [T*K, H] tight hidden states.  When expert_map
                                       is set, only rows 0..M_local-1 contain valid
                                       data; rows M_local..T*K-1 are safe but unused.
            true_offsets:              [E_local] int32 tight start per local expert
            true_sizes:                [E_local] int32 true token count per local expert
            expert_first_token_offset: [E_local+1] int64 non-padded exclusive prefix-sum
            fwd_idx:                   [T*K] int32 forward mapping: flat_idx -> tight slot
                                       (-1 for non-local token-expert pairs when expert_map set)
        """
        assert topk_ids.dtype == torch.int32
        backend = get_device_backend()
        assert backend.is_device_tensor(hidden_states) and backend.is_device_tensor(topk_ids)
        assert topk_ids.numel() == self.numel

        dev = hidden_states.device
        flat_ids = topk_ids.flatten().contiguous()
        E_local = self.num_experts_local

        expert_first_token_offset = torch.empty(E_local + 1, dtype=torch.int64, device=dev)
        true_offsets = torch.empty(E_local, dtype=torch.int32, device=dev)
        true_sizes = torch.empty(E_local, dtype=torch.int32, device=dev)
        # Zero-init so unwritten slots (non-local pairs in EP mode) default to row 0
        permuted_idx = torch.zeros(self.numel, dtype=torch.int32, device=dev)
        fwd_idx = torch.empty(self.numel, dtype=torch.int32, device=dev)

        # S5 glue: the integrated NPU scan factories return a plain ``run``
        # callable that takes the per-core scratch tables (bh / slot_base /
        # run_buf, shape [NB, E_local]) in addition to the outputs.  The
        # Stage-0 scaffold expected a zero-arg factory plus a single
        # ``write_offsets`` table (GPU form).  NB mirrors the active factory's
        # grid, and the EP variant additionally consumes ``expert_map`` as its
        # 2nd positional argument (before the scratch tables).
        # ``bh`` / ``run_buf`` are RMW scratch and must be host-zeroed (scan
        # DESIGN sec 4.2); ``permuted_idx`` is zeroed above.
        if self.expert_map is not None:
            nb = _scan_num_blocks_ep(self.numel)
        else:
            nb = _scan_num_blocks(self.numel)
        bh = torch.zeros((nb, E_local), dtype=torch.int32, device=dev)
        run_buf = torch.zeros((nb, E_local), dtype=torch.int32, device=dev)
        slot_base = torch.empty((nb, E_local), dtype=torch.int32, device=dev)
        if self.expert_map is not None:
            expert_map = self.expert_map.flatten().contiguous()
            self._scan_fn(
                flat_ids, expert_map, bh, true_offsets, true_sizes,
                expert_first_token_offset, slot_base, run_buf, fwd_idx, permuted_idx,
            )
        else:
            self._scan_fn(
                flat_ids, bh, true_offsets, true_sizes,
                expert_first_token_offset, slot_base, run_buf, fwd_idx, permuted_idx,
            )

        # perm_h always has numel rows; downstream ops (GEMM/SiluAndMul) are compiled
        # with numel.  In EP mode, rows M_local..numel-1 contain row-0 duplicates
        # (safe dummy data) that the tile scheduler never accesses.
        perm_h = torch.empty(
            (self.numel, self.hidden_size), dtype=self.dtype, device=dev
        )
        # S5 glue: the integrated gather factory returns the compiled kernel
        # directly (call it with the tensors); the Stage-0 ``self._gather_fn()``
        # zero-arg form raises on the NPU JITKernel.
        self._gather_fn(hidden_states, permuted_idx, perm_h)

        return perm_h, true_offsets, true_sizes, expert_first_token_offset, fwd_idx
