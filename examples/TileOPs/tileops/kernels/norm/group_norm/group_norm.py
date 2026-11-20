"""GroupNorm forward kernel using TileLang (NPU-adapted).

y = (x - mean) / sqrt(var + eps) * weight + bias

where mean and var are computed over ``(C/G, *spatial)`` dimensions for each
of the ``G`` groups independently.  The input ``(N, C, *spatial)`` is reshaped
to ``(N*G, D)`` where ``D = (C/G) * spatial_size``, enabling row-wise
normalization identical to LayerNorm.  This module ports the **affine**
primary (``GroupNormFwdOp`` -> ``GroupNormKernel``); the no-affine variant
(``GroupNormNoAffineFwdOp`` / ``GroupNormNoAffineKernel``) is out of scope and
is not migrated.

Affine layering (preserved from GPU, do not collapse):

  - **Kernel layer**: ``_group_norm_kernel``'s prim_func is
    ``main(x, weight, bias, y)`` (four tensors, ``out_idx=[3]``) where
    ``weight``/``bias`` are **row-broadcast** vectors of shape ``(D_padded,)``.
    ``GroupNormKernel.forward`` applies an identity affine (unit weight /
    zero bias of width ``D_padded``) when the caller does not supply one.
  - **Op layer**: ``GroupNormFwdOp.forward(x, weight, bias)`` takes
    **per-channel** ``(C,)`` weight/bias.  Because the kernel's affine is
    row-broadcast and GroupNorm's per-channel affine does not fit that layout,
    the op calls the kernel *without* an affine and applies the per-channel
    ``y * weight.reshape(...) + bias.reshape(...)`` outside the kernel.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_group_norm_kernel``) is extracted from
    the GPU repo via ``extract_tl_kernel.py`` and imported as-is, together with
    its ``_align_up`` helper and the ``ALIGNMENT`` constant.  It serves as the
    reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 adaptations (decorator, grid/sync semantics,
    ``threads`` removal, and padding strategy) are handled by the NPU component
    during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7: ``custom_op("npub::group_norm_fwd")`` (was ``"top::group_norm_fwd"``).
    K8: ``autotune_configs`` property and ``tune`` parameter removed;
        heuristic config selection only, ``init_config(config)``.
    K9: ``threads`` removed from the custom_op wrapper signature, the
        factory-callable invocation, ``default_config``, and the ``forward``
        dispatch (row-reduction family: the meaningful knob is ``block_m``,
        not CUDA thread counts).
"""

from typing import Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .group_norm_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .group_norm_kernel import ALIGNMENT, _align_up, _group_norm_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .group_norm_kernel.perf_opt._group_norm_kernel import ALIGNMENT, _align_up, _group_norm_kernel

__all__ = ["GroupNormKernel"]


# ---------------------------------------------------------------------------
# Row-config selection.
#
# Ported from GPU ``tileops/kernels/norm/_config.py:select_row_config``.
# The GPU helper pins ``block_m=1`` (with one row per CTA the per-row
# uniformity the row reduction needs always holds, so the cross-thread
# AllReduce collapse is structurally impossible) plus ``threads=128``
# (measured CUDA optimum).  NPU adaptation K9 drops the ``threads`` key,
# leaving the structural ``block_m=1`` default.  The GPU
# ``select_row_configs`` autotune-space generator is not ported (K8).
# ---------------------------------------------------------------------------

_DEFAULT_BLOCK_M = 1


def _select_row_config(n_padded: int) -> dict:
    """Structurally collapse-free default ``{block_m}`` for a row reduction."""
    return {"block_m": _DEFAULT_BLOCK_M}


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::group_norm_fwd", mutates_args=())
def _group_norm_wrapped(
    M: int,
    D: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _group_norm_kernel(M, D, eps, dtype_str)(block_m)(x, weight, bias)


@_group_norm_wrapped.register_fake
def _(M, D, eps, dtype_str, block_m, x, weight, bias):
    D_padded = _align_up(D, ALIGNMENT)
    return torch.empty((M, D_padded), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class GroupNormKernel(Kernel):
    """GroupNorm forward kernel (affine primary).

    Normalizes each group's ``(C/G, *spatial)`` slice independently.
    Input is pre-reshaped to ``(M, D)`` where ``M = N*G``,
    ``D = (C/G)*spatial_size``.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed (no ``autotune_configs``, no ``tune`` parameter);
      heuristic config selection only, ``init_config(config)``.
    - K9: ``threads`` removed from ``default_config`` and the ``forward``
      dispatch.

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional tile config dict.
    """

    ascend_mode = "Developer"

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        M: int,
        D: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.M = M
        self.D = D
        self.eps = eps
        self.dtype = dtype
        self.D_padded = _align_up(D, ALIGNMENT)
        self.kernel = _group_norm_kernel(self.M, self.D, self.eps, self.dtype_str)
        self._identity_affine: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Default config (K9: ``threads`` key removed)."""
        return _select_row_config(self.D_padded)

    def _identity_affine_for(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached unit-scale / zero-shift buffers of the launch width."""
        buffers = self._identity_affine.get(device)
        if buffers is None:
            buffers = (
                torch.ones(self.D_padded, dtype=self.dtype, device=device),
                torch.zeros(self.D_padded, dtype=self.dtype, device=device),
            )
            self._identity_affine[device] = buffers
        return buffers

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize ``(M, D)`` rows with a row-broadcast affine.

        Args:
            x: Input of shape ``(M, D)``.
            weight: Affine scale of shape ``(D,)``. Omit together with *bias*
                to normalize without an affine.
            bias: Affine shift of shape ``(D,)``. Omit together with *weight*
                to normalize without an affine.

        Returns:
            Tensor of shape ``(M, D)``. The alignment padding the prim_func
            requires is applied and trimmed here.

        Raises:
            ValueError: If exactly one of *weight* / *bias* is omitted.
        """
        if (weight is None) != (bias is None):
            raise ValueError("weight and bias must be supplied together")
        pad = self.D_padded - self.D
        if pad:
            x = F.pad(x, (0, pad))
        if weight is None:
            weight, bias = self._identity_affine_for(x.device)
        elif pad:
            weight = F.pad(weight, (0, pad))
            bias = F.pad(bias, (0, pad))
        y = _group_norm_wrapped(
            self.M,
            self.D,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            x,
            weight,
            bias,
        )
        return y[:, : self.D] if pad else y
