"""Exp forward kernel using TileLang (NPU-adapted).

y = exp(x)

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + reimplemented):
    The GPU kernel function (``exp_fwd_kernel``) is extracted via
    ``.agents/skills/add-npu-op/scripts/extract_tl_kernel.py`` and preserved in
    ``_exp_fwd_kernels.py`` as the porting reference.  The npuir
    re-implementation lives in ``exp_kernel/exp.py``.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None``.
    K7: ``custom_op("npub::...")``.
    K8: autotune removed.
    K9: ``threads`` removed.
    K11: ``threads * npt`` collapsed into ``block_size``.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Kernel source: npuir integration package (Stage 5 repoints this import if a
# perf_opt/ variant is added, as done for `mish`).
from .exp_kernel.perf_opt.exp_fwd_kernel import exp_fwd_kernel, _BIG_MIN_N

# K11: GPU threads(256) * npt(8) = 2048.
# Round 4: block size is now differentiated by dtype.  The kernel measures 12 B per
# element of block_size at fp16/bf16 (x=2, f=4, e=4, y=2) so 16384 needs exactly 192 KB
# and compiles; at fp32 it is 16 B (all four buffers 4 B) so 16384 would need 256 KB and
# does not fit.  The tuning sweep already measured 16384 as 7.9% faster on fp16/bf16.
_DEFAULT_BLOCK_SIZE = {"float32": 8192, "float16": 16384, "bfloat16": 16384}

__all__ = ["ExpFwdKernel"]


@torch.library.custom_op("npub::exp_fwd", mutates_args=())
def _exp_fwd_wrapped(
    N: int,
    dtype_str: str,
    block_size: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return exp_fwd_kernel(N, dtype_str, block_size=block_size)(x, N)


@_exp_fwd_wrapped.register_fake
def _(N: int, dtype_str: str, block_size: int, x: torch.Tensor):
    return torch.empty((N,), dtype=x.dtype, device=x.device)


class ExpFwdKernel(Kernel):
    """Exp forward kernel: y = exp(x).

    Float-only (inherited from the GPU ``FloatUnaryKernel``); the GPU manifest
    declares no integer dtypes for ExpFwdOp.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict (e.g. ``{"block_size": 2048}``).
    """

    supported_archs: Optional[list] = None
    ascend_mode = "Developer"
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
        self.kernel = exp_fwd_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # Same rule as the kernel factory (exp_fwd_kernel): the big block only pays off
        # for fp16/bf16 AND only when N fills the final grid round.  Measured on 1m fp16,
        # bs=16384 is 8.47% SLOWER than 8192 (64 blocks over 48 cores), so fall back.
        _big_ok = (self.dtype_str != "float32" and self.N_total >= _BIG_MIN_N)
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str] if _big_ok else 8192}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _exp_fwd_wrapped(
            self.N_total,
            self.dtype_str,
            self.config["block_size"],
            x,
        )
