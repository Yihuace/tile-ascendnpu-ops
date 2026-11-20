"""MoE no-pad 3WG persistent grouped GEMM with fused gate/up activation (NPU-adapted).

Ported from GPU TileOPs
``tileops/kernels/moe/moe_grouped_gemm_persistent_3wg_fused_act.py`` -- the
``MoeGroupedGemmPersistent3WGFusedActKernel`` class (warp-specialized 3-WG
persistent grouped GEMM whose epilogue fuses ``act(gate) * up`` so the
``[numel, 2*ffn]`` gate_up intermediate never reaches global memory).

Adaptation summary (GPU -> NPU, see docs/gpu_to_npu_adaptation.md):

  **Part A -- TileLang kernel functions** (imported from the integrated
  Stage-3 product):
    The GPU scaffold's single ``_fused_act_kernel`` factory was superseded by
    the NPU Stage-3 product ``_fused_act_kernel.py`` (see the co-located
    ``_fused_act_kernel_DESIGN.md``), which splits the op into two kernels
    launched back-to-back on the same stream:

      * ``_gemm_gate_up`` -- Expert single-``T.Scope("Cube")`` grouped GEMM
        with two fp32 L0C accumulators, writing the packed
        ``gate_up[numel, 2*ffn]`` (fp32, no-pad dynamic extents);
      * ``_act_packed``   -- pure Vector activation
        ``C = act(gate) * up`` with a single round-back to the activation
        dtype.

    Both factories are imported from the integrated kernel package; the tile
    schedule the GEMM kernel consumes is materialised on the host (see the
    integration note below).

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = [90]`` (Hopper SM90) -> ``None`` (all archs).
    K8: ``autotune()`` / ``autotune_configs`` / ``tune`` param removed;
        heuristic config selection only (``init_config(config)``).
    K9: ``threads`` removed from ``default_config`` and from the ``forward``
        dispatch (``num_stages`` / ``group_size_m`` / ``block_*`` retained --
        algorithmic, not CUDA thread parameters).
    K6/O1-O2: the persistent grid size is resolved inside the two NPU kernels
        from the physical AI-core counts, so ``sm_count`` is retained only as
        a compat parameter and is no longer forwarded to a factory.

  Integration note (Stage 5): the Stage-3 GEMM kernel reads a GM tile-schedule
  table that the sibling ``_tile_scheduler_kernel`` producer would supply.
  Until that sibling lands, ``forward`` builds the same table on the host from
  ``true_sizes`` (DESIGN sec 4.1 contract: clustered experts, per-expert
  ``ceil(s_e / block_m)`` tiles, ``(-1, 0)`` sentinel padding, row offsets in
  rows).  This costs one host round-trip per call and is replaced by the
  on-device producer once available.
"""

from typing import Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.utils import device_ascend_mode

# Part A: NPU Stage-3 kernel factories, imported by integrate_kernel.py from
# examples/moe_grouped_gemm_nopad_3wg_fused_act/_fused_act_kernel/_fused_act_kernel.py
# (the GPU scaffold's ``_fused_act_kernel`` placeholder is superseded by these
# two kernels -- see the module docstring).  NOTE: integrate_kernel.py consumes
# the names imported from this statement to emit the baseline/perf_opt block
# below, so the two names must be the real Stage-3 exports.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .moe_grouped_gemm_nopad_3wg_fused_act_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .moe_grouped_gemm_nopad_3wg_fused_act_kernel import _gemm_gate_up, _act_packed
# S5 glue: reuse the kernel's own bounded fresh-compile retry (the documented
# reliable net for the flaky bishengir-compile SIGSEGV; see the kernel DESIGN /
# ``_run_kernel``).
from .moe_grouped_gemm_nopad_3wg_fused_act_kernel._fused_act_kernel import _run_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .moe_grouped_gemm_nopad_3wg_fused_act_kernel.perf_opt._fused_act_kernel import _gemm_gate_up
# from .moe_grouped_gemm_nopad_3wg_fused_act_kernel.perf_opt._fused_act_kernel import _act_packed

__all__ = ["MoeGroupedGemmPersistent3WGFusedActKernel"]

# K9: "threads" removed from the GPU default config (was ``threads=384``,
# 1 producer + 2 consumer WGs) -- NPU has no CUDA thread concept.
_DEFAULT_CONFIG = {
    "block_m": 128,        # cooperative template (split-A, shared dual-B)
    "block_n": 128,        # ffn N-tile; dual fp32 accum caps this at 128
    "block_k": 64,         # K-aligned only (K % block_k == 0)
    "num_stages": 3,
    "group_size_m": 1,
}

#: Activations fused into the GEMM epilogue (compile-time specialized).
_SUPPORTED_ACTIVATIONS = ("silu_and_mul", "gelu_and_mul")


def _resolve_core_count(device_index: Optional[int] = None) -> int:
    """Resolve the persistent grid size (NPU adaptation, K6/O1-O2).

    The GPU version reads ``multi_processor_count`` from the CUDA device
    properties.  NPU has no SM concept; the persistent grid maps to the AI
    core count.  The backend device-properties object is probed for the
    attributes exposed by the Ascend runtime, falling back to 1.  The NPU
    kernel component must confirm the exact attribute when it retargets the
    kernel to ``target="npuir"``.
    """
    backend = get_device_backend()
    try:
        # Some torch_npu builds reject ``get_device_properties(None)`` with
        # "Invalid device id"; resolve the current device explicitly first.
        dev = device_index
        if dev is None and backend.is_available():
            dev = backend.current_device()
        props = backend.get_device_properties(dev) if dev is not None else None
    except Exception:  # noqa: BLE001 - value is compat-only (see below)
        props = None
    if props is not None:
        for attr in ("multi_processor_count", "cube_core_num", "aicore_num",
                     "ai_core_num", "core_num"):
            val = getattr(props, attr, None)
            if val:
                return int(val)
    return 1


class MoeGroupedGemmPersistent3WGFusedActKernel(Kernel):
    """3WG persistent grouped GEMM with fused gate/up activation (K-aligned only).

    Computes ``A[numel, K] @ B[E, 2*ffn, K]^T -> act(gate) * up -> C[numel, ffn]``
    where the gate (first ``ffn`` rows of each expert weight) and up (next
    ``ffn`` rows) projections share the same ``A`` rows and are fused in the
    GEMM epilogue.

    Args:
        numel: T * top_k tight row count.
        num_experts: Number of local experts E.
        N: FFN width (output column count); B has ``2*N`` rows (gate||up).
        K: Hidden size.
        dtype: Activation and weight dtype (bf16 or fp16).
        activation: Fused activation ("silu_and_mul" or "gelu_and_mul").
        sm_count: Optional persistent grid size override (defaults to the
            device AI-core count).
        config: Optional config dict with block_m/n/k, num_stages,
            group_size_m.
        device_index: Optional device index for multi-device dispatch.

    Example:
        >>> kernel = MoeGroupedGemmPersistent3WGFusedActKernel(
        ...     numel=16384, num_experts=256, N=768, K=2048,
        ...     dtype=torch.bfloat16, activation="silu_and_mul")
        >>> C = kernel(A, B, true_sizes, true_offsets)  # [numel, ffn]
    """

    # Expert is required on 910B (Developer persistent+gemm segfaults,
    # CG-2026-0010) but broken on A5/CANN 9.2 (Expert codegen emits a MIX
    # module with an empty AIV sub-block -> bishengir "Unknown PIPE!").
    # ``device_ascend_mode`` downgrades Expert -> Developer on A5 only;
    # __init__ resolves it per instance so the pin matches the device.
    ascend_mode = "Expert"

    # K5: [90] (CUDA SM90 / Hopper) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        numel: int,
        num_experts: int,
        N: int,
        K: int,
        dtype: torch.dtype = torch.bfloat16,
        activation: str = "silu_and_mul",
        sm_count: Optional[int] = None,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ):
        super().__init__()
        if activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_SUPPORTED_ACTIVATIONS}, got {activation!r}"
            )
        self.numel = numel
        self.num_experts = num_experts
        self.N = N            # ffn (output width), NOT 2*ffn
        self.K = K
        self.dtype = dtype
        self.activation = activation
        self.device_index = device_index
        # A5: Expert codegen is broken for this kernel, use Developer there.
        self.ascend_mode = device_ascend_mode(type(self).ascend_mode)
        # K6/O1-O2: device-backend core count instead of torch.cuda.
        if sm_count is None:
            sm_count = _resolve_core_count(device_index)
        self.sm_count = sm_count
        # K8: ``tune`` removed -- heuristic config selection only.
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # K9: no "threads" key.
        return dict(_DEFAULT_CONFIG)

    @staticmethod
    def _build_tile_schedule(sizes_l, block_m, max_tiles):
        """Host-built GM tile schedule (DESIGN sec 4.1 contract).

        Mirrors ``_fused_act_kernel.build_schedule``: experts are clustered in
        order, each contributing ``ceil(s_e / block_m)`` tiles; ``rows`` are
        in-expert row offsets (already multiplied by ``block_m``).  The tables
        are padded to the kernel's compile-time ``max_tiles`` with the
        ``(-1, 0)`` dead-tile sentinel, and the returned ``total`` is the number
        of live tiles.  Replaced by the on-device ``_tile_scheduler_kernel``
        producer once it lands.
        """
        ids, rows = [], []
        for e, s in enumerate(sizes_l):
            for j in range((int(s) + block_m - 1) // block_m):
                ids.append(e)
                rows.append(j * block_m)
        total = len(ids)
        if total > max_tiles:
            raise ValueError(
                f"tile schedule overflow: total={total} > max_tiles={max_tiles}"
            )
        while len(ids) < max_tiles:
            ids.append(-1)
            rows.append(0)
        return ids, rows, total

    def forward(
        self,
        A: torch.Tensor,             # [numel, K]
        B: torch.Tensor,             # [num_experts, 2*ffn, K]
        true_sizes: torch.Tensor,    # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run the fused gate_up GEMM + activation.

        Launches the two Stage-3 kernels back-to-back on the default stream:
        ``_gemm_gate_up`` (Expert Cube, packed fp32 ``gate_up``) then
        ``_act_packed`` (Vector activation, single round-back).

        Args:
            A: [numel, K] tight permuted activations (not M-padded).
            B: [num_experts, 2*ffn, K] gate||up expert weights (NT layout).
            true_sizes: [E] int32 token count per expert.
            true_offsets: [E] int32 tight start offset per expert in A.

        Returns:
            C: [numel, ffn] activated gate_up output.
        """
        bm = self.config["block_m"]
        bn = self.config["block_n"]
        bk = self.config["block_k"]
        if self.K % bk != 0:
            raise ValueError(f"K-aligned only: K={self.K}, block_k={bk}")
        if self.N % bn != 0:
            raise ValueError(f"ffn must be divisible by block_n: N={self.N}, block_n={bn}")

        # Compile-time M-tile upper bound (DESIGN sec 4.1 shared constant).
        max_tiles = self.numel // bm + self.num_experts
        sizes_l = [int(s) for s in true_sizes.detach().cpu().tolist()]
        ids_l, rows_l, total = self._build_tile_schedule(sizes_l, bm, max_tiles)
        device = A.device
        tile_ids = torch.tensor(ids_l, dtype=torch.int32, device=device)
        tile_rows = torch.tensor(rows_l, dtype=torch.int32, device=device)
        total_tiles = torch.tensor([total], dtype=torch.int32, device=device)

        # Kernel G: Expert Cube grouped GEMM -> packed fp32 gate_up [numel, 2*ffn].
        # S5 glue: bounded fresh-compile retry (``_run_kernel``) -- the kernel's
        # own harness recovers from the flaky bishengir-compile SIGSEGV; the
        # factory call sits inside the lambda so a retry recompiles.
        gate_up = _run_kernel(
            lambda: _gemm_gate_up(
                self.numel, max_tiles, self.num_experts, self.N, self.K,
                self.dtype_str, bm, bn, bk,
            ),
            (A, B, tile_ids, tile_rows, true_offsets, true_sizes, total_tiles),
        )

        # Kernel V: Vector activation -> C [numel, ffn].
        return _run_kernel(
            lambda: _act_packed(
                self.numel, self.N, self.dtype_str, self.activation,
            ),
            (gate_up,),
        )
