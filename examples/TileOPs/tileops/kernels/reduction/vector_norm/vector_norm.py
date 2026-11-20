"""L1 / L2 / Inf vector-norm kernel using TileLang (NPU-adapted).

Reduces a ``(M, N)`` tensor along the last dim, producing ``(M,)``:

    l1  -> sum(|x|)
    l2  -> sqrt(sum(x^2))
    inf -> max(|x|)

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + reimplemented):
    The GPU kernel functions (``_vector_norm_kernel``,
    ``_vector_norm_kernel_tiled``) are extracted via
    ``.agents/skills/add-npu-op/scripts/extract_tl_kernel.py`` and preserved in
    ``_inf_norm_fwd_kernels.py`` as the porting reference.  The npuir
    re-implementation of the single-tile variant lives in
    ``_vector_norm_kernel/``.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: autotune removed -- heuristic config selection only.
    K9: ``threads`` removed from ``default_config`` and ``forward``.

  **SCOPE LIMITATION (documented, not hidden)**: only the **single-tile**
  variant is migrated.  The GPU ``_vector_norm_kernel_tiled`` exists to work
  around a CUDA single-fragment column limit (``MAX_SINGLE_TILE_COLS``); on
  NPU the equivalent constraint is the UB budget: the single-tile path needs
  ``block_m * N * (2*elem_bytes + 4)`` bytes for its two dtype tiles plus the
  fp32 shadow.  For ``N`` so large that even ``block_m = 1`` does not fit,
  the constructor raises ``NotImplementedError`` naming the missing tiled
  path rather than silently mis-computing.  See the migration log §9.3.
"""

import functools
from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    UB_SAFETY_RESERVE_BYTES,
    device_smem_budget,
)

# Only the single-tile (baseline) source is migrated; see the module docstring.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .vector_norm_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .vector_norm_kernel.perf_opt._vector_norm_kernel import _vector_norm_kernel

# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .vector_norm_kernel.perf_opt._vector_norm_kernel import _vector_norm_kernel

_VECTOR_NORM_KINDS = ("l1", "l2", "inf")

__all__ = ["VectorNormKernel"]


def _elem_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


@functools.lru_cache(maxsize=64)
def _vector_norm_fwd_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Cached factory builder (mirrors the GPU ``functools.lru_cache`` use)."""
    return _vector_norm_kernel(M, N, op_kind, dtype)


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::vector_norm_fwd", mutates_args=())
def _vector_norm_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _vector_norm_fwd_kernel(M, N, op_kind, dtype_str)(block_m)(x)


@_vector_norm_fwd_wrapped.register_fake
def _(M: int, N: int, op_kind: str, dtype_str: str, block_m: int, x: torch.Tensor):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5, K8, K9 adaptations)
# ---------------------------------------------------------------------------


class VectorNormKernel(Kernel):
    """L1 / L2 / Inf norm forward kernel.

    Output dtype matches the input dtype; internal computation is fp32.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``; the GPU
      ``threads``/``npt`` launch geometry has no NPU counterpart.

    Args:
        M: Number of rows (product of all dims except the last).
        N: Hidden dimension (last dim).
        op_kind: One of ``"l1"``, ``"l2"``, ``"inf"``.
        dtype: Input data type (float16, bfloat16, float32).
        config: Optional kernel configuration dict (e.g. ``{"block_m": 4}``).
        device_index: Device index for the shared-memory budget query.
            ``None`` uses the current device.
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    ascend_mode = "Developer"

    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ):
        super().__init__()
        if op_kind not in _VECTOR_NORM_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_VECTOR_NORM_KINDS)}."
            )
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self._elem_bytes = _elem_bytes(dtype)
        self._smem_budget = device_smem_budget(device_index)
        self._ub_budget = max(self._smem_budget - UB_SAFETY_RESERVE_BYTES, 16 * 1024)

        # Single-tile path requires the whole row in UB together with its fp32
        # shadow.  Fail loudly rather than silently mis-computing.
        if self._ub_bytes_for(1) > self._ub_budget:
            raise NotImplementedError(
                f"VectorNormKernel: N={N} does not fit the single-tile UB budget "
                f"({self._ub_bytes_for(1)} B needed for block_m=1 vs "
                f"{self._ub_budget} B available). The N-tiled variant "
                f"(_vector_norm_kernel_tiled) is not migrated yet -- see the "
                f"module docstring."
            )

        self.kernel = _vector_norm_fwd_kernel(self.M, self.N, self.op_kind, self.dtype_str)
        # 契约变更（2026-11-06）：bf16 输入在 Op 层先上采成 fp16 再进 kernel。
        # 依据：本机 bf16 的向量算力只有 fp16 的一半（64 vs 128 op/cycle），而 kernel
        # 内部不允许 bf16<->fp16 转换（vcast 不支持），只能在 Op 层做。
        # 数值精确：bf16 只有 8 位尾数，转 fp16（10 位尾数）无舍入，|x| ∈ [6.1e-5, 65504]。
        # 代价（不计入被测 kernel）：每次调用多一趟 bf16->fp16 与一趟 fp16->bf16 的显存读写；
        # 按本机可达带宽 1.49 TB/s 折算，上采 33.55 MB ≈ 22.52 µs，见 perf_opt/opt_log.md 第 11 轮。
        self._kernel_f16 = (
            _vector_norm_fwd_kernel(self.M, self.N, self.op_kind, "float16")
            if dtype is torch.bfloat16
            else None
        )
        self.init_config(config)

    def _ub_bytes_for(self, block_m: int) -> int:
        """UB bytes for a given block_m: two dtype tiles + fp32 shadow + row acc."""
        return block_m * self.N * (2 * self._elem_bytes + 4) + block_m * (self._elem_bytes + 4)

    @property
    def default_config(self) -> dict:
        """Pick the largest block_m whose tiles fit in the UB budget."""
        for cand in (16, 8, 4, 2, 1):
            if self._ub_bytes_for(cand) <= self._ub_budget:
                return {"block_m": min(cand, self.M)}
        return {"block_m": 1}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the vector-norm kernel on an ``(M, N)`` tensor.

        K9: no ``threads`` argument (NPU has no CUDA threads).
        """
        if x.dtype is torch.bfloat16 and self._kernel_f16 is not None:
            # 上采到 fp16 域计算（128 op/cycle），结果再写回 bf16。
            x_h = x.to(torch.float16)
            y_h = self._kernel_f16(self.config["block_m"])(x_h)
            return y_h.to(torch.bfloat16)
        return _vector_norm_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self.config["block_m"],
            x,
        )
