"""Floor forward kernel using TileLang (NPU-adapted).

y = floor(x)

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + reimplemented):
    The GPU kernel function (``floor_fwd_kernel``) is extracted via
    ``.agents/skills/add-npu-op/scripts/extract_tl_kernel.py`` and preserved in
    ``_floor_fwd_kernels.py`` as the porting reference.  The npuir
    re-implementation lives in ``floor_kernel/floor.py``.

  **API substitution**: this npuir build has no unary ``floor``.  The kernel
  expresses it through ``npuir_cast`` with ``round_mode="floor"`` (int32
  round-trip).  See ``floor_kernel/floor.py`` for the API inventory check and
  the ``|x| < 2**31`` range caveat.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5/K7/K8/K9/K11 as per the elementwise adaptation set.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

from .floor_kernel.perf_opt.floor_fwd_kernel import floor_fwd_kernel

# K11: GPU threads(256) * npt(8) = 2048.
# Stage-4 round 4: same buffer eliminations as ceil (tmp_ub folded into
# t_ub; the SAT_MAG constant shrunk to (1,1)) let block_size reach 8192.
# Must match perf_opt/floor_fwd_kernel.py.
_DEFAULT_BLOCK_SIZE = {"float32": 8192, "float16": 8192, "bfloat16": 8192}

__all__ = ["FloorFwdKernel"]


@torch.library.custom_op("npub::floor_fwd", mutates_args=())
def _floor_fwd_wrapped(
    N: int,
    dtype_str: str,
    block_size: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return floor_fwd_kernel(N, dtype_str, block_size=block_size)(x, N)


@_floor_fwd_wrapped.register_fake
def _(N: int, dtype_str: str, block_size: int, x: torch.Tensor):
    return torch.empty((N,), dtype=x.dtype, device=x.device)


class FloorFwdKernel(Kernel):
    """Floor forward kernel: y = floor(x).

    Float-only, implemented via an int32 cast round-trip (no unary floor in this
    npuir build).  Output equals the mathematical floor for ``|x| < 2**31``.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict.
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
        self.kernel = floor_fwd_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _floor_fwd_wrapped(
            self.N_total,
            self.dtype_str,
            self.config["block_size"],
            x,
        )
