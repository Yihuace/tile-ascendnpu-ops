"""GLA single-token decode kernel (NPU wrapper + ``GLADecodeKernel``).

The GPU ``GLADecodeKernel`` (and the fp32 variant ``GLADecodeFP32Kernel``) is a
Vector recurrence; the NPU port keeps the same contract and pins
``ascend_mode = "Developer"`` (no Cube / ``T.gemm``).  See the extracted factory
``gla_decode_kernel/_gla_decode_kernel.py`` and the feasibility probe under
``examples/gla_decode/probe/``.
"""

from typing import Optional

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# perf_opt source (perf_opt/GLADecodeOp.py, k_tile default 32) is active after
# Stage 4 L0/L1 regression passed; the baseline (Stage 3, k_tile default 16)
# source is commented out.  ``pytest tests/ops/`` and ``benchmarks/ops/``
# dispatch through whichever source is active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3, k_tile default 16) -----------------------------------
# from .gla_decode_kernel._gla_decode_kernel import _ceil_ktile_fit, _gla_decode_kernel
# --- perf_opt (Stage 4 tuned, k_tile default 32) ------------------------------
from .gla_decode_kernel.perf_opt.GLADecodeOp import _ceil_ktile_fit, _gla_decode_kernel

__all__ = ["GLADecodeKernel"]


class GLADecodeKernel(Kernel):
    """Single-step (decode) gated linear attention, layout BHD."""

    supported_archs: Optional[list] = None
    ascend_mode = "Developer"

    def __init__(
        self,
        B: int,
        H: int,
        DK: int,
        DV: int,
        dtype: torch.dtype = torch.bfloat16,
        scale: float = -1.0,
        k_tile: Optional[int] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.B = B
        self.H = H
        self.DK = DK
        self.DV = DV
        self.dtype = dtype
        # GPU contract: a non-positive ``scale`` means DK ** -0.5.
        self.scale = float(scale) if scale is not None and scale > 0 else DK ** -0.5
        if k_tile is None:
            k_tile = _ceil_ktile_fit(DK, DV)
        self.k_tile = k_tile
        self._factory = _gla_decode_kernel(B, H, DK, DV, self.dtype_str, self.scale)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"k_tile": self.k_tile}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, gk, state = (t.contiguous() for t in (q, k, v, gk, state))
        program = self._factory(self.config["k_tile"])
        return program(q, k, v, gk, state)
