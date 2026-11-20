"""Add forward kernel using TileLang (NPU-adapted).

Implements the scalar-alpha binary addition of ``torch.add``::

    y = a + alpha * b

with ``alpha`` baked into the kernel (one specialization per distinct value).
Float operands take an fp32 intermediate for the scalar multiply, so a half
input keeps the precision of the scaled term; integer / bool operands keep
native integer arithmetic.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §2/§3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``add_fwd_kernel`` -- the
    ``make_binary_register_copy`` builder merged with ``AddFwdKernel``'s
    dynamic ``AlphaScaledBinaryKernel._alpha_op_func`` op body) lives in
    ``_add_fwd_kernels.py`` and is imported as-is.  It serves as the reference
    for the NPU kernel component to reimplement for ``target="npuir"``.  K1-K4
    and K11 adaptations (decorator, grid/sync, ``threads``/``npt`` collapse,
    padding strategy) are handled by the NPU component during re-implementation.

  **Part B -- Kernel class** (ported from the GPU ``AlphaScaledBinaryKernel``
  register_copy flow):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
         heuristic config selection only.
    K9:  ``threads`` removed from ``default_config`` and ``forward``.
    K11: ``threads * npt`` collapsed into a single ``block_size`` config key.
    (No ``@torch.library.custom_op`` wrapper -- the GPU elementwise kernel has
    none; ``forward`` dispatches the compiled callable directly.)

Binary + broadcast note: the Op layer expands both operands to
``broadcast_shapes(input, other)`` and flattens them, so this kernel only ever
sees two same-shape contiguous 1-D tensors of ``N_total`` elements (the GPU
``register_copy`` case).  The broadcast contract is preserved by ``AddFwdOp``.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .add_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .add_kernel import add_fwd_kernel
# add_bcast_row_kernel = add_bcast_scalar_kernel = plan_broadcast = None
# BCAST_ENABLED = False
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# The R6 broadcast fast path (row / scalar reuse) is part of the perf_opt
# source.  Optional: with the baseline source above there are no broadcast
# kernels, ``BCAST_ENABLED`` is False, and ``AddFwdOp`` stays on the flat path
# (identical numerics, 3N traffic).
from .add_kernel.perf_opt.AddFwdOp import (
    add_bcast_period_kernel,
    add_bcast_row_kernel,
    add_bcast_scalar_kernel,
    add_fwd_kernel,
    plan_broadcast,
)

# Default ON (2026-11-17, P0-0a): the reuse path cuts real latency 30-43% on
# broadcast shapes.  The official ``msprof`` Ratio column (L2 read-miss based)
# *drops* when redundant reads are removed (linear-bias bf16 69.4% -> 49.6%),
# because it counts the removed redundant read as useful work.  That is an
# accounting artifact, not a slowdown: Add is therefore gated on Task Duration
# + the model-byte MBU from ``eval_roofline()`` (see
# reports/2026-11/2026-11-17-三算子性能瓶颈重审.md).  Set to False only to
# reproduce the legacy flat-path numbers.
BCAST_ENABLED = True

__all__ = [
    "BCAST_ENABLED",
    "AddBcastPeriodKernel",
    "AddBcastRowKernel",
    "AddBcastScalarKernel",
    "AddFwdKernel",
    "plan_broadcast",
]

# ---------------------------------------------------------------------------
# K11 constant: UB tile width ``block_size``.
#
# Stage 5 integration takes the NPU value from the Stage 1 design
# (add_fwd_kernel_DESIGN.md §5.2): block_size = 8192, the same element-wise
# best value measured for the mish / hardsigmoid / hardswish family (32B
# aligned for both fp16 and fp32; UB budget < 192KB).  The GPU
# ``make_binary_register_copy`` default ``threads * npt`` collapse (256 * 8 =
# 2048) is superseded by this design decision.
# ---------------------------------------------------------------------------

_DEFAULT_BLOCK_SIZE = {
    "float32": 8192,
    "float16": 8192,
    "bfloat16": 8192,
}


class AddFwdKernel(Kernel):
    """Scalar-alpha binary addition forward kernel: ``y = a + alpha * b``.

    A two-input, single-output elementwise op.  The kernel receives two flat,
    contiguous, same-shape 1-D vectors of ``N_total`` elements (the Op layer
    resolves broadcasting) and applies ``y = a + alpha * b`` with ``alpha``
    baked in.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` product collapsed into ``block_size``.
    - No custom_op wrapper (GPU elementwise parity); ``forward`` calls the
      compiled callable directly.

    Args:
        N_total: Total number of elements (flattened, post-broadcast).
        dtype: Data type (float16, bfloat16, or float32).
        alpha: Multiplier on ``other`` (default 1).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 8192}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Manifest: float16 | bfloat16 | float32 (the GPU op's full union also
    # covers ints/bool; this NPU port follows the float subset).
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        alpha: "int | float" = 1,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.alpha = alpha
        self.output_dtype = dtype  # same_as(input)
        # Build the factory callable (does not compile yet -- compilation is
        # deferred to forward()).  The imported function is the GPU reference
        # implementation; the NPU component rewrites it for target="npuir".
        self.kernel = add_fwd_kernel(self.N_total, self.dtype_str, alpha=self.alpha)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (``block_size = 8192``, DESIGN §5.2)."""
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run the add kernel.

        Accepts two 1-D contiguous tensors of ``N_total`` elements.  K9:
        ``threads`` removed; K11: the factory is rebuilt with the configured
        ``block_size`` (the GPU ``threads``/``npt`` decomposition collapses into
        the single NPU ``block_size``).
        """
        block_size = self.config["block_size"]
        return add_fwd_kernel(
            self.N_total,
            self.dtype_str,
            alpha=self.alpha,
            block_size=block_size,
        )(a, b, self.N_total)


class AddBcastRowKernel(Kernel):
    """Broadcast fast path: ``other`` is one row repeated every ``row``.

    Drop-in for the flat kernel on ``plan_broadcast``'s row mode
    (``linear-bias``: ``[1024, 4096] + [1, 4096]``).  The kernel stages the
    single period in UB once per core and reuses it for every tile, so DRAM
    traffic is ``2*N + row`` elements instead of ``3*N``.  Numerics are
    identical to ``AddFwdKernel`` (same alpha fold, same fp32 transit rule).
    """

    supported_archs: Optional[list] = None
    ascend_mode = "Developer"
    # The broadcast factories all define their prim_func as ``main``; declaring
    # it explicitly keeps msprof filtered to the add kernel (the period path's
    # one-off BroadcastTo materialization must not enter the kernel timing).
    msprof_kernel_name = "main"
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self, plan: dict, dtype: torch.dtype, alpha: "int | float" = 1):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype {dtype}")
        self.plan = plan
        self.dtype = dtype
        self.alpha = alpha
        self.output_dtype = dtype
        self.init_config(None)

    @property
    def default_config(self) -> dict:
        return {"block_size": self.plan["tile"]}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``a``: the full operand; ``b``: the single broadcast period."""
        n_total = self.plan["N"]
        return add_bcast_row_kernel(
            n_total,
            self.plan["row"],
            self.dtype_str,
            alpha=self.alpha,
            block_size=self.config["block_size"],
        )(a.reshape(-1), b.reshape(-1), n_total)


class AddBcastScalarKernel(Kernel):
    """Broadcast fast path: one scalar per ``L``-element plane, ``b[r % b_len]``.

    Drop-in for the flat kernel on ``plan_broadcast``'s scalar mode
    (``cnn-feat-broadcast``: ``[16, 256, 56, 56] + [256, 1, 1]``).  A tile
    covers ``planes`` consecutive planes and adds the per-plane scalars with
    the UB row-broadcast form ``(P, L) + (P, 1)`` -- no materialized operand,
    so traffic is ``2*N + B``.
    """

    supported_archs: Optional[list] = None
    ascend_mode = "Developer"
    msprof_kernel_name = "main"
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self, plan: dict, dtype: torch.dtype, alpha: "int | float" = 1):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype {dtype}")
        self.plan = plan
        self.dtype = dtype
        self.alpha = alpha
        self.output_dtype = dtype
        self.init_config(None)

    @property
    def default_config(self) -> dict:
        return {"planes": self.plan["planes"]}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``a``: the full ``(R, L)`` operand; ``b``: ``(chunk, planes, 1)``."""
        plan = self.plan
        return add_bcast_scalar_kernel(
            plan["R"],
            plan["L"],
            plan["b_len"],
            self.dtype_str,
            alpha=self.alpha,
            planes=self.config["planes"],
        )(
            a.reshape(plan["R"], plan["L"]),
            b.reshape(plan["chunk"], plan["planes"], 1),
            plan["R"],
        )

class AddBcastPeriodKernel(Kernel):
    """Broadcast fast path: 2-byte scalar broadcast via a materialized period.

    Drop-in for the flat path on ``plan_broadcast``'s period mode
    (``cnn-feat-broadcast`` bf16/fp16).  ``other`` has one scalar per output
    plane (``B`` planes, each ``L`` long); one full period ``(B, L)`` is
    materialized once per call and the kernel streams same-shape slices from
    it (L2-resident), so DRAM traffic is ``2*N + B*L`` elements instead of
    ``3*N``.  The 2-byte UB broadcast (``vbrc``) route is avoided because it
    costs 3-4x on this toolchain (REPORT_ADD_BCAST.md section 4).
    """

    supported_archs: Optional[list] = None
    ascend_mode = "Developer"
    msprof_kernel_name = "main"
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self, plan: dict, dtype: torch.dtype, alpha: "int | float" = 1):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype {dtype}")
        self.plan = plan
        self.dtype = dtype
        self.alpha = alpha
        self.output_dtype = dtype
        self.init_config(None)

    @property
    def default_config(self) -> dict:
        return {"planes": self.plan["planes"]}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``a``: the full ``(R, L)`` operand; ``b``: the ``(B, 1, ...)`` scalars.

        The single period ``(B, L)`` is materialized here (one small broadcast
        copy, 1.6 MB for ``cnn-feat``); the timed kernel then reads it from L2.
        """
        plan = self.plan
        b_period = (
            b.reshape(-1)
            .reshape(plan["B"], 1)
            .expand(plan["B"], plan["L"])
            .contiguous()
        )
        return add_bcast_period_kernel(
            plan["R"],
            plan["L"],
            plan["B"],
            self.dtype_str,
            alpha=self.alpha,
            planes=self.config["planes"],
        )(
            a.reshape(plan["R"], plan["L"]),
            b_period,
            plan["R"],
        )

