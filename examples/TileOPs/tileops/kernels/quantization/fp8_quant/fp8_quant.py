"""FP8 per-group quantization kernel using TileLang (NPU-adapted).

Quantizes a ``[B, S_kv, G, D]`` index tensor along the last (index) dimension:
each ``(b, s, g)`` group gets a scale ``amax / 448`` (amax clamped to
``>= 1e-4``) and the values are clamped to ``[-448, 448]`` and cast to
``float8_e4m3fn``.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_fp8_quant_kernel``) is extracted from
    the GPU repo via ``extract_tl_kernel.py`` and imported as-is.  It serves as
    the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 adaptations (decorator target, grid/sync
    semantics, ``threads`` removal, padding strategy) are handled by the NPU
    component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[90]``, a CUDA SM integer).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` property and ``tune`` parameter removed;
        heuristic config selection only, ``init_config(config)``.
    K9: no ``threads`` key in ``default_config``/``forward`` (the GPU callable
        also carries no ``threads``; the CUDA ``threads=128`` lives inside the
        prim_func body and is removed by the NPU component, K2).
    K11: no ``npt`` -- this op is not in the ``threads * npt`` elementwise
        family; its knobs are ``num_stages`` and ``block_m``.
"""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .fp8_quant_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .fp8_quant_kernel import _fp8_quant_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .fp8_quant_kernel.perf_opt._fp8_quant_kernel import _fp8_quant_kernel

__all__ = ["FP8QuantKernel"]


@torch.library.custom_op("npub::fp8_quant_wrapped_kernel", mutates_args=())
def _fp8_quant_wrapped_kernel(
    batch: int,
    seq_len_kv: int,
    kv_group: int,
    index_dim: int,
    in_dtype: str,
    num_stages: int,
    block_m: int,
    input_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # NPU glue: the integrated kernel has the explicit-output signature
    # ``(X(R, D), Scale(R,), OutU8(R, D))`` -- there is no ``out_idx``
    # (uint8 out_idx silently falls back to float32, CG-2026-0019), so the
    # host side allocates the fp32 scale and the uint8 byte carrier, runs the
    # kernel on the flattened ``[B*S_kv*G, D]`` metadata view, and restores the
    # externally observable contract (fp32 scale ``[B,S,G]`` + fp8_e4m3
    # ``[B,S,G,D]``) via reshape + ``.view(torch.float8_e4m3fn)``.
    kernel = _fp8_quant_kernel(batch, seq_len_kv, kv_group, index_dim, in_dtype)(
        num_stages, block_m
    )
    rows = batch * seq_len_kv * kv_group
    x = input_tensor.contiguous().reshape(rows, index_dim)
    scale = torch.empty(rows, dtype=torch.float32, device=x.device)
    out_u8 = torch.empty(rows, index_dim, dtype=torch.uint8, device=x.device)
    kernel(x, scale, out_u8)
    return (
        scale.reshape(batch, seq_len_kv, kv_group),
        out_u8.reshape(batch, seq_len_kv, kv_group, index_dim).view(torch.float8_e4m3fn),
    )


@_fp8_quant_wrapped_kernel.register_fake
def _(batch, seq_len_kv, kv_group, index_dim, in_dtype, num_stages, block_m, *inputs):
    return torch.empty(
        (batch, seq_len_kv, kv_group), dtype=torch.float32, device=inputs[0].device
    ), torch.empty(
        (batch, seq_len_kv, kv_group, index_dim), dtype=torch.float8_e4m3fn, device=inputs[0].device
    )


class FP8QuantKernel(Kernel):
    """Per-group fp8 quantization of a ``[B, S_kv, G, D]`` index tensor.

    Args:
        batch: Batch size.
        seq_len_kv: Key/value sequence length.
        kv_group: Number of key/value groups.
        index_dim: Index (head) dimension.
        dtype: Torch dtype of the tensor being quantized. The output element
            type is fixed at ``torch.float8_e4m3fn`` and the scale at
            ``torch.float32``, so this is the kernel's only free dtype.
        config: Optional dict with "num_stages" and "block_m".
    """

    ascend_mode = "Developer"

    # K5: [90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        batch: int,
        seq_len_kv: int,
        kv_group: int,
        index_dim: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.batch = batch
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.index_dim = index_dim
        self.dtype = dtype
        self.kernel = _fp8_quant_kernel(
            self.batch, self.seq_len_kv, self.kv_group, self.index_dim, self.dtype_str
        )
        # K8: no ``tune`` argument (NPU uses heuristic config selection only).
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"num_stages": 0, "block_m": 32}

    def forward(self, input_tensor: torch.Tensor):
        return _fp8_quant_wrapped_kernel(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            self.dtype_str,
            self.config["num_stages"],
            self.config["block_m"],
            input_tensor,
        )
