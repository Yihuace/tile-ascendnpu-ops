"""Abs forward kernel using TileLang (NPU-adapted).

Implements the elementwise absolute value:
    y = |x|

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + reimplemented):
    The GPU TileLang kernel function (``abs_fwd_kernel``) is extracted from
    the GPU repo via ``.agents/skills/add-npu-op/scripts/extract_tl_kernel.py``
    and preserved verbatim in ``_abs_fwd_kernels.py`` as the porting
    reference.  The NPU re-implementation for ``target="npuir"`` lives in
    ``abs_kernel/abs.py`` and is what this wrapper dispatches to
    (K1-K4, K11 handled there).

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7:  ``custom_op("npub::...")`` (was ``"top::..."``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed.
    K9:  ``threads`` removed from ``default_config`` and ``forward``.
    K11: ``threads * npt`` collapsed into a single ``block_size`` param.

  **Dtype scope**: the GPU manifest declares integer dtypes for AbsFwdOp, but
  the GPU kernel itself (``FloatUnaryKernel``) is float-only — integers were
  short-circuited at the GPU op layer.  The NPU port is float-only for the
  same reason and matches the ``MishFwdKernel`` precedent.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source
#
# Stage 0 produced the extracted GPU reference (``_abs_fwd_kernels.py``); the
# npuir re-implementation lives in the ``abs_kernel/`` integration package.
# Stage 5 repoints this import when a ``perf_opt/`` variant is added, as done
# for ``mish``.
# ---------------------------------------------------------------------------
from .abs_kernel.perf_opt.abs_fwd_kernel import abs_fwd_kernel

# K11: GPU threads(256) * npt(8) = 2048 (fp16/bf16); 256 * 4 = 1024 (fp32).
_DEFAULT_BLOCK_SIZE = {"float32": 16384, "float16": 16384, "bfloat16": 16384}

__all__ = ["AbsFwdKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed, K11: block_size)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::abs_fwd", mutates_args=())
def _abs_fwd_wrapped(
    N: int,
    dtype_str: str,
    block_size: int,
    x: torch.Tensor,
) -> torch.Tensor:
    # NOTE: the perf_opt/ kernel takes (x, y) -- no `shape` scalar, unlike the
    # baseline and unlike ceil/cos/exp/floor -- so it is called with x only.
    return abs_fwd_kernel(N, dtype_str, block_size=block_size)(x)


@_abs_fwd_wrapped.register_fake
def _(N: int, dtype_str: str, block_size: int, x: torch.Tensor):
    return torch.empty((N,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9, K11 adaptations)
# ---------------------------------------------------------------------------


class AbsFwdKernel(Kernel):
    """Abs forward kernel: y = |x|.

    A single-input, single-output elementwise op.  The input is flattened to
    a 1-D vector of ``N_total`` elements and the pointwise absolute value is
    applied.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` product collapsed into ``block_size``.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 2048}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer mode
    # regardless of ambient TILELANG_ASCEND_MODE.
    ascend_mode = "Developer"

    # Float-only dtypes (inherited from the GPU FloatUnaryKernel).
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(input)
        # Build the factory callable (compilation is deferred to forward()
        # via the custom_op wrapper).
        self.kernel = abs_fwd_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * npt)."""
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the abs kernel on a 1-D contiguous tensor of ``N_total`` elements."""
        return _abs_fwd_wrapped(
            self.N_total,
            self.dtype_str,
            self.config["block_size"],
            x,
        )
