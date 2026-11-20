"""Grouped GEMM persistent 3-WG kernel package (NPU-adapted).

Ported from GPU TileOPs
``tileops/kernels/grouped_gemm/grouped_gemm_persistent_3wg.py`` -- the
``GroupedGemmPersistent3WGKernel`` class (V2 persistent grouped GEMM:
warp-specialized 3-WG + static-wave scheduler + TMA store).

Migration-unit note
-------------------
This kernel has **no standalone manifest op entry** and **no dedicated Op
class** in the GPU repo.  It is a pure kernel referenced only through the
``kernel_map`` of the MoE fusion ops
(``FusedMoEExpertsNopadPersistent3WGFwdOp`` / ``FusedMoeFwdOp`` /
``FusedMoeFwdCbFwdOp``) and by the grouped-GEMM baseline benchmark.  It is
therefore scaffolded as a **kernel-only package** (no manifest entry, no
workload / Op class / test / benchmark are fabricated).  Its standalone
contract is ``C[e] = A[e] @ B[e]^T`` for every expert, matching the already
migrated ``MoeGroupedGemmNopadFwdOp`` grouped-GEMM semantics.

Adaptation summary (GPU -> NPU, see docs/gpu_to_npu_adaptation.md):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang JIT factory ``_persistent_grouped_gemm_v2_kernel`` plus
    its two build helpers (``_make_pingpong_kernel`` /
    ``_make_cooperative_kernel``) are extracted from the GPU repo into
    ``_grouped_gemm_persistent_3wg_kernels.py`` and imported as-is.  They are
    the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 / K11 adaptations (decorator target, grid/sync
    semantics, ``threads`` removal) are handled by the NPU kernel component
    during re-implementation -- **not** by this scaffolding step.

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = [90]`` (Hopper SM90) -> ``None`` (all archs).
    K8: ``autotune()`` / ``autotune_configs`` / ``tune`` param removed;
        heuristic config selection only (``init_config(config)``).
    K9: ``threads`` removed from ``default_config`` and from the ``forward``
        dispatch.  ``num_stages`` / ``group_size_m`` / ``block_*`` are
        retained (they are algorithmic, not CUDA thread parameters).
    O1/O2: raw ``torch.cuda.get_device_properties(...).multi_processor_count``
        replaced by the device backend (``get_device_backend()``); the
        ``out`` overlap guard's ``tensors_overlap`` helper is inlined because
        the NPU project does not ship ``tileops/kernels/buffer_utils.py``.
        The helper itself is pure tensor-storage arithmetic (device-agnostic).

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation.  Its inner callable takes
  ``(block_m, block_n, block_k, num_stages, threads, group_size_m)``; the
  dispatch below passes the NPU-shaped argument list (no ``threads``).  It
  becomes callable once the NPU kernel component rewrites the factory for
  ``target="npuir"`` and drops ``threads`` from the callable signature (K3).
  Running ``forward`` before that rewrite fails -- this is the expected
  scaffold state.
"""

from typing import Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.utils import device_ascend_mode

# Part A: NPU Stage-3 kernel factory.  S5 glue: the Stage-0 scaffold imported
# the GPU extracted module ``._grouped_gemm_persistent_3wg_kernels``; the
# integrated NPU factory lives in ``.grouped_gemm_persistent_3wg_kernel``.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .grouped_gemm_persistent_3wg_kernel/perf_opt/{func}.py and pass their L0/L1
# regression; the baseline (Stage 3) source is active otherwise.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .grouped_gemm_persistent_3wg_kernel import _persistent_grouped_gemm_v2_kernel
# S5 glue: reuse the kernel's own bounded fresh-compile retry (the documented
# reliable net for the flaky bishengir-compile SIGSEGV; see the kernel DESIGN /
# ``_run_gemm``).  Imported from the ACTIVE module (baseline or perf_opt), whose
# drop-in ships the same helper.
# from .grouped_gemm_persistent_3wg_kernel._persistent_grouped_gemm_v2_kernel import (
#     _run_gemm,
# )
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# Round 7 (2026-10-22): the final module is the gate-contract name
# ``perf_opt/GroupedGemmPersistent3WG.py`` (statectl stage 4 hard-codes
# ``perf_opt/{operator_name}.py``).  It keeps the same factory signature and
# schedule contract as the Stage-3 module; the Stage-4 changes inside it are
# the D2 paired/double-buffered K body plus the shape-gated ``block_n`` bump
# (both are compile-time constant folds -- nothing is dispatched here).
# v1 ``perf_opt/_persistent_grouped_gemm_v2_kernel.py`` is kept on disk,
# unmodified, as the previous current-best (revert = re-point these two lines).
from .grouped_gemm_persistent_3wg_kernel.perf_opt.GroupedGemmPersistent3WG import (
    _persistent_grouped_gemm_v2_kernel,
)
from .grouped_gemm_persistent_3wg_kernel.perf_opt.GroupedGemmPersistent3WG import (
    _run_gemm,
)
from .grouped_gemm_persistent_3wg_kernel.perf_opt.GroupedGemmPersistent3WG import (
    TUNED_DEFAULT_CONFIG as _TUNED_DEFAULT_CONFIG,
)

__all__ = ["GroupedGemmPersistent3WGKernel"]

# K9: "threads" removed from the GPU default config (was ``threads=384``,
# 1 producer + 2 consumer WGs) -- NPU has no CUDA thread concept.
_DEFAULT_CONFIG = {
    # cooperative template (split-A, shared-B)
    "block_m": _TUNED_DEFAULT_CONFIG["block_m"],
    # Stage-4 tuned (was block_n=256 / block_k=64): block_k is the dominant
    # knob for this persistent grouped GEMM -- measured on the qwen3-235b
    # shapes, gate_up 14037.7 -> 10201.3 us (-27.3%) and down 6911.8 ->
    # 4969.7 us (-28.1%).  block_n=128 also wins slightly over 256 at
    # block_k=256 (10201.3 vs 10335.2 us).
    # Round 7: the kernel re-resolves block_n per shape at compile time
    # (128 -> 256 only for A-reread-dominated shapes), so this value is the
    # small-shape base, not a global constant.
    "block_n": _TUNED_DEFAULT_CONFIG["block_n"],
    "block_k": _TUNED_DEFAULT_CONFIG["block_k"],
    "num_stages": _TUNED_DEFAULT_CONFIG["num_stages"],
    # Threadblock swizzle: group this many consecutive m-tiles so concurrent
    # CTAs in a wave share B[e] columns in L2.
    "group_size_m": 8,
}


def _tensors_overlap(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """True if two tensors share storage AND their byte intervals overlap.

    Inlined from the GPU ``tileops/kernels/buffer_utils.py`` helper (the NPU
    project does not ship that module).  Pure tensor-storage arithmetic, so it
    is device-agnostic.
    """
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


def _resolve_core_count(device_index: Optional[int] = None) -> int:
    """Resolve the persistent grid size (NPU adaptation, K9/O1-O2).

    The GPU version reads ``multi_processor_count`` from the CUDA device
    properties.  NPU has no SM concept; the persistent grid maps to the AI
    core count.  The backend device-properties object is probed for the
    attributes exposed by the Ascend runtime, falling back to 1.  The NPU
    kernel component must confirm the exact attribute when it retargets the
    kernel to ``target="npuir"``.
    """
    backend = get_device_backend()
    try:
        dev = device_index
        if dev is None and backend.is_available():
            dev = backend.current_device()
        props = backend.get_device_properties(dev) if dev is not None else None
    except Exception:  # noqa: BLE001 - value is compat-only (see below)
        props = None
    for attr in ("multi_processor_count", "cube_core_num", "aicore_num",
                 "ai_core_num", "core_num"):
        val = getattr(props, attr, None)
        if val:
            return int(val)
    return 1


class GroupedGemmPersistent3WGKernel(Kernel):
    """V2 persistent grouped-GEMM kernel (K-aligned only)."""

    # S5 glue: the Stage-3 kernel is a single ``T.Scope("Cube")`` Expert-form
    # kernel (``ASCEND_MODE = "Expert"``), but the Stage-0 scaffold class did
    # not declare ``ascend_mode``.  Under the ambient (unset) mode the kernel
    # compiled into a form whose codegen segfaulted in ``bishengir-compile``
    # (in-process SIGSEGV); pinning Expert matches the kernel DESIGN and the
    # class-level mode used by every other Expert kernel in this repo.
    # ``device_ascend_mode`` downgrades Expert -> Developer on A5 only.
    ascend_mode = "Expert"

    # K5: [90] (CUDA SM90 / Hopper) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(self, numel, num_experts, N, K,
                 dtype=torch.bfloat16, sm_count=None,
                 config=None, device_index=None):
        super().__init__()
        self.ascend_mode = device_ascend_mode(type(self).ascend_mode)
        self.numel = numel
        self.num_experts = num_experts
        self.N = N
        self.K = K
        self.dtype = dtype
        self.device_index = device_index
        if sm_count is None:
            sm_count = _resolve_core_count(device_index)
        self.sm_count = sm_count
        self.kernel = lambda: _persistent_grouped_gemm_v2_kernel(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.sm_count, _DEFAULT_CONFIG["block_k"])
        # K8: ``tune`` removed -- heuristic config selection only.
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    def forward(self, A, B, true_sizes, true_offsets, out=None):
        """Run the grouped GEMM ``C[e] = A[e] @ B[e]^T`` for every expert.

        Args:
            A: ``[numel, K]`` tokens, grouped by expert (tight, no padding).
            B: ``[num_experts, N, K]`` expert weights (NT layout).
            true_sizes: ``[num_experts]`` int32 row count per expert.
            true_offsets: ``[num_experts]`` int32 start offset per expert in A.
            out: optional ``[numel, N]`` output buffer to write into and reuse
                across calls. Allocated internally with ``torch.empty`` if omitted.
                Must be contiguous and must not overlap ``A`` or ``B`` in memory
                (the kernel reads inputs and writes the output concurrently).
                Disjoint slices of a shared workspace buffer are allowed.

        Returns:
            The ``[numel, N]`` output (``out`` if provided).
        """
        block_m = self.config["block_m"]
        block_n = self.config["block_n"]
        block_k = self.config["block_k"]
        if self.K % block_k != 0:
            raise ValueError(f"K-aligned only: K={self.K}, block_k={block_k}")
        if self.N % block_n != 0:
            raise ValueError(f"N-aligned only: N={self.N}, block_n={block_n}")

        if not (B.device == true_sizes.device == true_offsets.device == A.device):
            raise ValueError(
                "A, B, true_sizes, true_offsets must be on the same device")

        if out is not None:
            if tuple(out.shape) != (self.numel, self.N):
                raise ValueError(
                    f"out shape must be {(self.numel, self.N)}, got {tuple(out.shape)}")
            if out.dtype != self.dtype:
                raise ValueError(f"out dtype must be {self.dtype}, got {out.dtype}")
            if out.device != A.device:
                raise ValueError(f"out device must be {A.device}, got {out.device}")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            if _tensors_overlap(out, A) or _tensors_overlap(out, B):
                raise ValueError("out must not overlap A or B in memory")

        # S5 glue: the integrated NPU factory consumes a host-built GM tile
        # schedule (DESIGN sec 4.1) and returns the compiled kernel directly
        # (``out_idx=[2]``).  The Stage-0 scaffold passed the GPU factory
        # signature ``(numel, num_experts, N, K, dtype, sm_count, block_k)`` and
        # re-called the result with the GPU block tuple, neither of which match
        # the NPU factory ``(numel, max_tiles, E, N, K, dtype, block_m, block_n,
        # block_k)``.  Schedule construction mirrors the sibling
        # ``MoeGroupedGemmPersistent3WGFusedActKernel._build_tile_schedule``.
        max_tiles = self.numel // block_m + self.num_experts
        sizes_l = [int(s) for s in true_sizes.detach().cpu().tolist()]
        ids_l, rows_l = [], []
        for e, s in enumerate(sizes_l):
            for j in range((s + block_m - 1) // block_m):
                ids_l.append(e)
                rows_l.append(j * block_m)
        total = len(ids_l)
        if total > max_tiles:
            raise ValueError(
                f"tile schedule overflow: total={total} > max_tiles={max_tiles}")
        while len(ids_l) < max_tiles:
            ids_l.append(-1)
            rows_l.append(0)
        tile_ids = torch.tensor(ids_l, dtype=torch.int32, device=A.device)
        tile_rows = torch.tensor(rows_l, dtype=torch.int32, device=A.device)
        total_tiles = torch.tensor([total], dtype=torch.int32, device=A.device)

        # S5 glue: use the kernel's bounded fresh-compile retry (``_run_gemm``,
        # PGG_COMPILE_ATTEMPTS) -- the kernel's own harness treats the flaky
        # bishengir-compile SIGSEGV as recoverable; the harness path must too or
        # the composed op fails nondeterministically.
        C = _run_gemm(
            lambda: _persistent_grouped_gemm_v2_kernel(
                self.numel, max_tiles, self.num_experts, self.N, self.K,
                self.dtype_str, block_m, block_n, block_k,
            ),
            (A, B, tile_ids, tile_rows, true_offsets, true_sizes, total_tiles),
        )
        if out is not None:
            out.copy_(C)
            return out
        return C
