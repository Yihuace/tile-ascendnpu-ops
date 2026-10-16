"""1-D convolution kernels (NPU-adapted wrapper + Kernel classes).

The GPU ``Conv1dFwdOp`` selects among five kernel classes:
``Conv1dPointwiseKernel`` (1x1 GEMM), ``Conv1dKernel`` (full implicit-GEMM
window), ``Conv1dUnitStrideKernel`` (stride-1, tap-rectangle im2col),
``DepthwiseConv1dKernel`` and ``GroupConv1dKernel`` (per-group implicit GEMM).
This file ports the five Kernel classes and imports their ``@tilelang.jit``
builders from the extracted reference file.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factories live in the self-contained reference
    ``_conv1d_fwd_kernels.py`` (extracted Pattern B -- 5 raw builders).  They
    are the GPU (CUDA-target) implementations the NPU kernel component
    rewrites for ``target="npuir"``.  K1-K4 (decorator, grid/sync, ``threads``
    removal, padding strategy) are applied during that re-implementation.

  **Cube / Expert mode**: every Conv1d program is an implicit GEMM (except the
    depthwise-direct tap loop), i.e. a Cube op on npuir.  The feasibility probe
    ``examples/conv1d/probe/conv1d_geomm_probe.py`` confirmed the GEMM form
    compiles/runs on the npuir Cube path in Expert mode (``T.Scope("Cube")``,
    ``T.alloc_L1`` / ``T.alloc_L0C`` / ``T.gemm``, slice-form ``T.copy`` for the
    strided/overlapping im2col stage; ``load_nd2nz`` avoided per CG-2026-0004).
    Each Kernel class therefore pins ``ascend_mode = "Expert"`` (mirroring the
    MHA / SSD Cube siblings), so every invocation is scoped to Expert mode
    regardless of the ambient ``TILELANG_ASCEND_MODE``.

  **Part B -- Kernel classes** (ported from the GPU classes):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` -- removed; heuristic
         config selection only (``init_config(config)``).
    K9:  GPU ``threads`` (and ``num_stages`` / ``block_*`` / ``taps`` /
         ``enable_rasterization``) are CUDA execution-model knobs, not baked
         here.
    K11: the wrapper ``default_config`` returns ``{}`` and ``forward`` invokes
         the factory with no baked CUDA execution parameters, letting the
         kernel's own NPU default govern (do not bake the GPU config).

  **Part C -- Stage 5 glue** (this file): the imported factories are the
    migrated ``target="npuir"`` kernels.  Unlike the GPU flow, each NPU
    ``@T.prim_func`` takes its im2col / polyphase / cross-engine workspaces
    (``w_flat``, ``x_pad`` / ``x_gpad``, ``xdec``, ``poly_idx``, the fp32 bias
    relay ``ws``) as explicit tensor parameters, so ``forward`` allocates and
    passes them (reusing each kernel module's own shape planner) and calls the
    factory with NPU block tiling from ``default_config``.  No CUDA execution
    knob (``threads`` / ``enable_rasterization``) is baked; ``num_stages`` is a
    factory-signature placeholder the Expert kernels ignore (mirroring the
    ``ssd_chunk_scan`` wrapper), not a GPU pipeline value.  The host-side
    ``weight.permute`` K-reorder of the GPU ``Conv1dKernel.forward`` is dropped:
    the npuir kernels take the native ``(c_out, c_in, k)`` weight and re-order
    it in-kernel.
"""

from typing import Optional, Tuple

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils import NPUUtils

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .conv1d_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .conv1d_kernel import (
    _conv1d_direct_kernel,
    _conv1d_group_kernel,
    _conv1d_kernel,
    _conv1d_pointwise_kernel,
)
# --- perf_opt (Stage 4 tuned): unit_stride dropped the in-kernel Vector prelude
# (weight re-layout + x_pad), so the wrapper fills w_flat/x_pad on the host.
from .conv1d_kernel.perf_opt._conv1d_unit_stride_kernel import _conv1d_unit_stride_kernel
_HOST_LAYOUT = True

__all__ = [
    "Conv1dKernel",
    "Conv1dPointwiseKernel",
    "Conv1dUnitStrideKernel",
    "DepthwiseConv1dKernel",
    "GroupConv1dKernel",
]

# ---------------------------------------------------------------------------
# Workspace / plan helpers reused from the integrated NPU kernel modules.
#
# The migrated npuir kernels take their im2col / polyphase / cross-engine
# workspaces as explicit ``T.Tensor`` parameters (allocated by the wrapper),
# so the wrapper reuses each kernel module's own shape planner and constant
# index builder instead of re-deriving the layout.  These are pure host-side
# helpers (Python / torch CPU) with no device dependency.
# ---------------------------------------------------------------------------
from .conv1d_kernel._conv1d_kernel import (  # noqa: E402
    _build_poly_idx as _general_poly_idx,
    _ws_dims as _general_ws_dims,
)
from .conv1d_kernel._conv1d_unit_stride_kernel import _plan as _unit_stride_plan  # noqa: E402
from .conv1d_kernel._conv1d_group_kernel import (  # noqa: E402
    _build_poly_idx as _group_poly_idx,
    _plan as _group_plan,
)
from .conv1d_kernel._conv1d_direct_kernel import (  # noqa: E402
    _build_poly_idx as _direct_poly_idx,
    _plan as _direct_plan,
)

# Cross-engine bias relay depth (Cube -> GM ws -> Vector), matching SLOTS in
# the integrated kernel modules.
_SLOTS = 2

# NPU tiling defaults: the design's block sizes only (mirror each integrated
# module's module-level ``BLOCK_*`` constants).  The GPU ``threads`` /
# ``enable_rasterization`` knobs are NOT ported.
#
# ``num_stages`` is a required positional of every integrated factory signature
# (``(block_m, block_n, block_k[, taps], num_stages)``) but the Expert kernels
# hardcode their pipeline depth and ignore it; it is carried as a
# source-compatible placeholder (same convention as the ``ssd_chunk_scan``
# wrapper), never a GPU-derived value.
_DEFAULT_BLOCK = {"block_m": 64, "block_n": 128, "block_k": 128}
_NUM_STAGES = 1


def _aicore_num() -> int:
    """Physical AI-core count used to size the per-core bias relay workspace."""
    return NPUUtils.get().get_aicore_num()


def _run(program, x, weight, bias, workspaces) -> torch.Tensor:
    """Dispatch one traced program: ``bias`` first when present, ws last.

    A program is traced for one side of ``has_bias``; the no-bias variant lacks
    the bias parameter.  The bias relay workspace ``ws`` (fp32, per-core) only
    exists on the bias side and always trails the argument list.
    """
    if bias is None:
        return program(x, weight, *workspaces)
    return program(x, weight, bias, *workspaces)


class Conv1dPointwiseKernel(Kernel):
    """Dense 1x1 Conv1d, which lowers to a pointwise GEMM over the channel axis."""

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Cube (implicit GEMM) Expert-form kernel; see module docstring.
    ascend_mode = "Expert"

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        dtype: torch.dtype,
        has_bias: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_l = l_in
        self.k_total = c_in
        self.kernel = _conv1d_pointwise_kernel(
            n,
            c_in,
            l_in,
            c_out,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # NPU tiling only (design block sizes); no CUDA execution-model knobs.
        return {**_DEFAULT_BLOCK, "num_stages": _NUM_STAGES}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight_2d = weight[:, :, 0].contiguous()
        config = self.config
        program = self.kernel(**config)
        if bias is None:
            return _run(program, x.contiguous(), weight_2d, None, [])
        ws = torch.empty(
            (_aicore_num(), _SLOTS, config["block_m"], config["block_n"]),
            dtype=torch.float32,
            device=x.device,
        )
        return _run(program, x.contiguous(), weight_2d, bias.contiguous(), [ws])


class Conv1dKernel(Kernel):
    """Dense Conv1d over the full kernel window; the general ungrouped kernel."""

    supported_archs: Optional[list] = None
    ascend_mode = "Expert"

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_l: int,
        stride_l: int,
        pad_l: Tuple[int, int],
        dtype: torch.dtype,
        dilation_l: int = 1,
        has_bias: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_l = kernel_l
        self.stride_l = stride_l
        self.pad_l = pad_l
        self.pad_left, self.pad_right = pad_l
        self.dilation_l = dilation_l
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_l = (l_in + sum(pad_l) - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
        self.m = n * self.out_l
        self.k_total = c_in * kernel_l
        self.kernel = _conv1d_kernel(
            n,
            c_in,
            l_in,
            c_out,
            kernel_l,
            stride_l,
            self.pad_left,
            self.pad_right,
            dilation_l,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # NPU tiling only; the GPU threads / enable_rasterization knobs (and the
        # host ``weight.permute`` K-reorder) are not ported -- the npuir kernel
        # re-orders the weight in-kernel from the native (c_out, c_in, k) layout.
        return {**_DEFAULT_BLOCK, "num_stages": _NUM_STAGES}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        weight = weight.contiguous()
        config = self.config
        need_poly, l_pad, mdec, c_in_pad, _ch_xp, cg = _general_ws_dims(
            self.l_in,
            self.c_in,
            self.pad_left,
            self.pad_right,
            self.kernel_l,
            self.dilation_l,
            self.stride_l,
        )
        w_flat = torch.empty((self.kernel_l, self.c_out, c_in_pad), dtype=x.dtype, device=x.device)
        x_pad = torch.zeros((self.n, c_in_pad, l_pad), dtype=x.dtype, device=x.device)
        xdec = torch.empty((self.n, c_in_pad, l_pad), dtype=x.dtype, device=x.device)
        if need_poly:
            poly_idx = _general_poly_idx(mdec, self.stride_l, cg, l_pad).to(x.device)
        else:
            poly_idx = torch.zeros((cg, l_pad), dtype=torch.int32, device=x.device)
        workspaces = [w_flat, x_pad, xdec, poly_idx]
        bias_t = None
        if bias is not None:
            bias_t = bias.contiguous()
            workspaces.append(
                torch.empty(
                    (_aicore_num(), _SLOTS, config["block_m"], config["block_n"]),
                    dtype=torch.float32,
                    device=x.device,
                )
            )
        program = self.kernel(**config)
        return _run(program, x, weight, bias_t, workspaces)


class Conv1dUnitStrideKernel(Conv1dKernel):
    """Dense stride-1 Conv1d whose k tiles are one tap over a run of input channels."""

    supported_archs: Optional[list] = None
    ascend_mode = "Expert"

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_l: int,
        stride_l: int,
        pad_l: Tuple[int, int],
        dtype: torch.dtype,
        dilation_l: int = 1,
        has_bias: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(
            n,
            c_in,
            l_in,
            c_out,
            kernel_l,
            stride_l,
            pad_l,
            dtype,
            dilation_l,
            has_bias,
            config=None,
        )
        self.kernel = _conv1d_unit_stride_kernel(
            n,
            c_in,
            l_in,
            c_out,
            kernel_l,
            stride_l,
            self.pad_left,
            self.pad_right,
            dilation_l,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # NPU tiling only; ``taps`` groups taps per k tile (unit-stride tap
        # rectangle), matching the integrated module's TAPS baseline.
        return {**_DEFAULT_BLOCK, "taps": 1, "num_stages": _NUM_STAGES}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        weight = weight.contiguous()
        config = self.config
        l_pad, c_in_pad, _ch_xp = _unit_stride_plan(
            self.l_in, self.c_in, self.pad_left, self.pad_right
        )
        w_flat = torch.zeros((self.kernel_l, self.c_out, c_in_pad), dtype=x.dtype, device=x.device)
        x_pad = torch.zeros((self.n, c_in_pad, l_pad), dtype=x.dtype, device=x.device)
        if _HOST_LAYOUT:
            # host pre-layout (perf_opt): x_pad[n,ci,pad_left+i]=x[n,ci,i];
            # w_flat[kw,co,ci]=weight[co,ci,kw]
            x_pad[:, : self.c_in, self.pad_left : self.pad_left + self.l_in] = x
            w_flat[:, :, : self.c_in] = weight.permute(2, 0, 1).to(x.dtype)
        workspaces = [w_flat, x_pad]
        bias_t = None
        if bias is not None:
            bias_t = bias.contiguous()
            workspaces.append(
                torch.empty(
                    (_aicore_num(), _SLOTS, config["block_m"], config["block_n"]),
                    dtype=torch.float32,
                    device=x.device,
                )
            )
        program = self.kernel(**config)
        return _run(program, x, weight, bias_t, workspaces)


class GroupConv1dKernel(Kernel):
    """Grouped Conv1d: one implicit GEMM per group over that group's channels."""

    supported_archs: Optional[list] = None
    ascend_mode = "Expert"

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_l: int,
        stride_l: int,
        pad_l: Tuple[int, int],
        dtype: torch.dtype,
        dilation_l: int = 1,
        has_bias: bool = False,
        groups: int = 1,
        c_in_g: Optional[int] = None,
        c_out_g: Optional[int] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_l = kernel_l
        self.stride_l = stride_l
        self.pad_l = pad_l
        self.pad_left, self.pad_right = pad_l
        self.dilation_l = dilation_l
        self.groups = groups
        self.c_in_g = c_in_g if c_in_g is not None else c_in // groups
        self.c_out_g = c_out_g if c_out_g is not None else c_out // groups
        self.dtype = dtype
        self.has_bias = has_bias
        if self.groups <= 1:
            raise ValueError(f"{type(self).__name__} requires groups > 1")
        self._build_program()
        self.init_config(config)

    def _build_program(self) -> None:
        """Compile the grouped implicit GEMM over this call's group shape."""
        if self.c_in % self.groups or self.c_out % self.groups:
            raise ValueError(
                f"{type(self).__name__} requires c_in and c_out divisible by groups; "
                f"got c_in={self.c_in}, c_out={self.c_out}, groups={self.groups}"
            )
        self.kernel = _conv1d_group_kernel(
            self.n,
            self.c_in,
            self.l_in,
            self.c_out,
            self.kernel_l,
            self.stride_l,
            self.pad_left,
            self.pad_right,
            self.dilation_l,
            self.has_bias,
            self.dtype_str,
            self.groups,
            self.c_in_g,
            self.c_out_g,
        )

    @property
    def default_config(self) -> dict:
        # NPU tiling only; the per-group weight K-reorder is done in-kernel.
        return {**_DEFAULT_BLOCK, "num_stages": _NUM_STAGES}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        weight = weight.contiguous()
        config = self.config
        need_poly, l_pad, mdec, c_in_g_pad, _ch_xp, cg = _group_plan(
            self.l_in, self.c_in_g, self.pad_left, self.pad_right, self.stride_l
        )
        w_flat = torch.zeros(
            (self.groups, self.kernel_l, self.c_out_g, c_in_g_pad), dtype=x.dtype, device=x.device
        )
        x_gpad = torch.zeros(
            (self.n, self.groups, c_in_g_pad, l_pad), dtype=x.dtype, device=x.device
        )
        xdec_g = torch.empty(
            (self.n, self.groups, c_in_g_pad, l_pad), dtype=x.dtype, device=x.device
        )
        if need_poly:
            poly_idx = _group_poly_idx(mdec, self.stride_l, cg, l_pad).to(x.device)
        else:
            poly_idx = torch.zeros((cg, l_pad), dtype=torch.int32, device=x.device)
        workspaces = [w_flat, x_gpad, xdec_g, poly_idx]
        bias_t = None
        if bias is not None:
            bias_t = bias.contiguous()
            workspaces.append(
                torch.empty(
                    (_aicore_num(), _SLOTS, config["block_m"], config["block_n"]),
                    dtype=torch.float32,
                    device=x.device,
                )
            )
        program = self.kernel(**config)
        return _run(program, x, weight, bias_t, workspaces)


class DepthwiseConv1dKernel(GroupConv1dKernel):
    """Depthwise Conv1d: one input channel per group, one output channel per group.

    A different program from :class:`GroupConv1dKernel`: with a single channel
    per group there is no contraction to tile, so each output position
    accumulates over the kernel window instead of running an implicit GEMM.
    """

    supported_archs: Optional[list] = None
    ascend_mode = "Expert"

    def _build_program(self) -> None:
        self.kernel = _conv1d_direct_kernel(
            self.n,
            self.c_in,
            self.l_in,
            self.c_out,
            self.kernel_l,
            self.stride_l,
            self.pad_left,
            self.pad_right,
            self.dilation_l,
            self.has_bias,
            self.dtype_str,
        )

    @property
    def default_config(self) -> dict:
        # NPU tiling only: the depthwise-direct path is a pure-Vector kernel
        # whose only tile knob is the L-direction width (module BLOCK_N).
        return {"block_n": _DEFAULT_BLOCK["block_n"]}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        weight = weight.contiguous()
        _out_l, l_pad2, mdec, need_poly, _r_kw, _q_kw = _direct_plan(
            self.n,
            self.c_in,
            self.c_out,
            self.l_in,
            self.kernel_l,
            self.stride_l,
            self.pad_left,
            self.pad_right,
            self.dilation_l,
        )
        xpad = torch.zeros((self.n, self.c_in, l_pad2), dtype=x.dtype, device=x.device)
        if need_poly:
            xdec = torch.empty((self.n, self.c_in, l_pad2), dtype=x.dtype, device=x.device)
            poly_idx = _direct_poly_idx(l_pad2, mdec, self.stride_l, self.c_in).to(x.device)
        else:
            xdec = torch.empty((1, 1, 1), dtype=x.dtype, device=x.device)
            poly_idx = torch.zeros((1, 1), dtype=torch.int32, device=x.device)
        workspaces = [xpad, xdec, poly_idx]
        program = self.kernel(**self.config)
        return _run(program, x, weight, bias.contiguous() if bias is not None else None, workspaces)
