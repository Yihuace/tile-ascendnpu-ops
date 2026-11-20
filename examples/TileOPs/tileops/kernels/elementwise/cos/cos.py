"""Cos forward kernel using TileLang (NPU-adapted).

y = cos(x)

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + reimplemented):
    The GPU kernel function (``cos_fwd_kernel``) is extracted via
    ``.agents/skills/add-npu-op/scripts/extract_tl_kernel.py`` and preserved in
    ``_cos_fwd_kernels.py`` as the porting reference.  The npuir
    re-implementation lives in ``cos_kernel/cos.py`` (``T.vcos``).

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5/K7/K8/K9/K11 as per the elementwise adaptation set.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

from .cos_kernel.perf_opt.cos_fwd_kernel import cos_fwd_kernel

# K11: GPU threads(256) * npt(8) = 2048.
# Stage 4（r6，2026-11-02）：tuned 档位。必须与 kernel 侧
# `perf_opt/cos_fwd_kernel.py` 的 `TUNED_DEFAULT_BLOCK_SIZE` 成对一致——
# fp16/bf16 走单流（12288 = 12 B/元素 × 12288 = 147 KB），
# fp32 走双流（8192，编译器多缓冲实需 24 B/元素 ⇒ 192 KB 压线）。
# 4096 档只能拿到 ratio 54.75%（16m fp16），12288 档为 63.82%。
_DEFAULT_BLOCK_SIZE = {"float32": 8192, "float16": 12288, "bfloat16": 12288}

__all__ = ["CosFwdKernel"]


@torch.library.custom_op("npub::cos_fwd", mutates_args=())
def _cos_fwd_wrapped(
    N: int,
    dtype_str: str,
    block_size: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return cos_fwd_kernel(N, dtype_str, block_size=block_size)(x, N)


@_cos_fwd_wrapped.register_fake
def _(N: int, dtype_str: str, block_size: int, x: torch.Tensor):
    return torch.empty((N,), dtype=x.dtype, device=x.device)


class CosFwdKernel(Kernel):
    """Cos forward kernel: y = cos(x).

    Float-only.  Signed output: cos is not sign-symmetric, so the test suite
    compares raw values (no abs() trick).

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
        self.kernel = cos_fwd_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _cos_fwd_wrapped(
            self.N_total,
            self.dtype_str,
            self.config["block_size"],
            x,
        )
