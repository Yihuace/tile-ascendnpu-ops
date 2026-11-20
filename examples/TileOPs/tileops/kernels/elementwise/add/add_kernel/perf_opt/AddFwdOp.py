# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Add (scalar-alpha) forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``AddFwdKernel`` (TileOPs,
``elementwise/arithmetic.py`` -> ``AlphaScaledBinaryKernel`` register_copy
path, ``_add_fwd_kernels.py``) to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y = input + alpha * other          (torch.add(input, other, *, alpha=1))
    broadcast(y.shape) = broadcast_shapes(input, other)
on two pre-broadcast, flattened, contiguous same-shape 1-D ``(N,)``
vectors of dtype float16 | bfloat16 | float32; output ``(N,)`` same dtype.
``alpha`` is a construction-time ``int | float`` scalar (default 1) baked
into the kernel (one specialization per distinct value).

Broadcast contract (DESIGN.md section 0.6 R2): the Op layer resolves numpy
broadcasting via ``broadcast_shapes`` + ``expand`` + flatten, so this
kernel only ever sees two same-shape contiguous 1-D vectors (the GPU
``register_copy`` case).  The kernel is broadcast-agnostic.

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel(num_kernels, is_npu=True)`` + v-prefix vector
        chain (vcast -> vmul -> vadd -> vcast) replaces the CUDA
        threads x npt two-level parallel; ``block_size`` folds threads*npt.
    R2: fp32 transit -- fp16/bf16 have no native Vector arithmetic for
        fp32 opmath (bf16 has none at all), so both are upcast with
        ``T.vcast(rint)`` -> fp32 compute -> single ``T.vcast(rint)`` back,
        matching the golden's fp32 opmath rounding path.
    R3: ``alpha == 1`` identity fold at trace time (``1*b == b`` exactly for
        every representable value incl. NaN/+-inf/-0.0); ``alpha == 0`` is
        NOT folded (``0*inf = NaN``).
    R4: persistent core split: ``num_kernels = min(num_logical,
        vector_cores)`` with an in-core grid-stride loop and an if-guard for
        out-of-range tiles (static bounds).

Stage-4 tuning (this file is the tuned drop-in; see ``perf_opt/opt_log.md``):
    T1: the task space is split into *full* static-extent tiles plus one
        dynamic tail tile owned by core 0, so the hot loop no longer computes
        ``T.min`` / dynamic copy extents per tile.
    T2: the full-tile loop is software-pipelined (``T.Pipelined`` 2 stages),
        overlapping the next tile's MTE2 loads with the current tile's VEC
        chain.  ``TUNED_BLOCK_SIZE`` is the measured winner (8192).
    T3 (A6): fp16 with ``alpha == 1`` computes natively in fp16.  A single
        add of two fp16 values has an exact fp32-representable sum, so
        "fp32 add then round to fp16" and "native fp16 add" are both one
        correct rounding and are bitwise equal (L0/L1 max_diff = 0).  This
        removes 3 of the 4 vector passes and 2/3 of the UB traffic.
        bf16 keeps the fp32 transit: native bf16 vadd compiles and is
        bit-exact on CANN 9.0.0, but its measured latency is inconsistent
        (probe -2%, official retest +3.8%), so it is not adopted.
        ``alpha != 1`` keeps the fp32 chain (the multiply would double-round).
    Rejected candidates (kept in ``perf_opt/candidates/`` for evidence):
    block_size 12288/16384 (load imbalance +5~7%), element-balanced contiguous
    per-core chunks (-3% vs grid-stride), and output-buffer aliasing
    (``y_ub = a_ub``; trace-scope compile error).

Broadcast reuse (R6/P0-0a): ``plan_broadcast`` turns three broadcast families
    into ``2*N + B`` traffic instead of the materialized ``3*N``, dispatched to
    ``add_bcast_row_kernel`` (``other`` = a repeated row, e.g. ``[1,4096]``),
    ``add_bcast_scalar_kernel`` (one scalar per output plane, fp32, e.g.
    ``[256,1,1]``), or ``add_bcast_period_kernel`` (2-byte scalar broadcast).
    The scalar kernel's UB broadcast (``vbrc`` / broadcast ``vadd``) costs 3-4x
    the same-shape form on 2-byte dtypes (``REPORT_ADD_BCAST.md`` §4), so 2-byte
    scalar broadcast goes through the period path instead: one full period
    ``(B, L)`` is materialized and streamed from L2, which charges
    ``2*N + B*L``.  Every other broadcast shape falls back to the flat path,
    and the roofline bytes follow the same decision.

Run the embedded hierarchical tests:
    python AddFwdOp.py --level L0
    python AddFwdOp.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no manual
# sync points (DESIGN.md section 7).  This literal mirrors the wrapper's
# ``ascend_mode = "Developer"`` and is read by Stage 5 integration.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
ASCEND_MODE = "Developer"

import argparse
import time
from math import prod

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = [
    "add_bcast_period_kernel",
    "add_bcast_row_kernel",
    "add_bcast_scalar_kernel",
    "add_fwd_kernel",
    "golden_add",
    "plan_broadcast",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608

# Stage-4 tuned default (perf_opt/opt_log.md, Round 2/3).  Exposed at module
# level so the Stage-5 wrapper switch block can pair its default assignment
# with this value instead of the Stage-3 8192 literal.
TUNED_BLOCK_SIZE = 8192

# --- broadcast fast path (R6) ----------------------------------------------
# Plan/tile guards.  ``_UB_SAFETY`` keeps the fp32-transit UB footprint under
# the 192 KB Vector UB; ``_MAX_ROW_REPS`` bounds the number of UB->UB copies
# used to replicate one broadcast period across a tile (b_len == 1 uses a
# single ``vbrc`` instead).
_UB_SAFETY = 0.90
_MAX_ROW_REPS = 16
# Period path (2-byte scalar broadcast, P0-0a): one full broadcast period is
# materialized (``B * L`` elements) and must stay L2-resident.  Cap its size
# and the rows per tile (UB budget is charged separately by ``_ub_bytes``).
_PERIOD_MAX_BYTES = 4 << 20
# The 2-stage pipeline makes the compiler duplicate the three tile buffers
# (observed ~8x the single tile for this shape), so keep the period tile at
# two planes: that is the form measured at the ~22.5 us 2D-copy floor in
# REPORT_ADD_BCAST.md section 4.  Larger tiles overflow the 192 KB UB.
_PERIOD_MAX_PLANES = 2

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Precision tolerances (DESIGN.md section 8.2, identical to the source-repo
# test_add.py ``_get_tolerances``).
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
}

# Per-(dtype, alpha) worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_add(input, other, alpha=1):
    """Add reference: ``y = input + alpha * other``, torch broadcast + fp32 opmath.

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_add.py: ``ref = torch.add(a.float(), b.float(), alpha=self.alpha)
    .to(a.dtype)``).  Independent of the NPU algorithm (no vcast / vmul /
    vadd fp32 chain / persistent structure / Op-layer flattening):
    broadcasting is handled natively by ``torch.add``.  Runs on CPU and does
    not require an NPU device.
    """
    a_cpu = input.detach().cpu()
    b_cpu = other.detach().cpu()
    y_f32 = torch.add(a_cpu.float(), b_cpu.float(), alpha=alpha)
    return y_f32.to(a_cpu.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[2], target="npuir")
def add_fwd_kernel(N, dtype, alpha=1, output_dtype=None, block_size=TUNED_BLOCK_SIZE, num_cores=None):
    """Add (scalar-alpha) forward kernel: ``y = a + alpha * b``.

    Compile-time params:
        N: flattened element count (each input is a contiguous ``(N,)`` vector).
        dtype: input dtype, "float16" | "bfloat16" | "float32".
        alpha: multiplier on ``b`` (int | float, default 1); baked in.
        output_dtype: optional output dtype (defaults to ``dtype``).
        block_size: UB tile width (folds GPU threads*npt); default
            ``TUNED_BLOCK_SIZE`` = 8192 (Stage-4 winner, opt_log.md).
        num_cores: optional explicit kernel count override (else section 5.5).

    The returned ``main(a, b, y, shape)`` is invoked as ``kernel(a, b, N)``;
    ``y`` is auto-allocated (``out_idx=[2]``).
    """
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"

    # R1-A4: split the task space (static full tiles + one dynamic tail) and
    # software-pipeline the full-tile loop.  The hot loop then has static
    # extents (no per-tile T.min), which lets the compiler keep the copy
    # lengths compile-time and gives T.Pipelined a clean 2-stage body.
    num_full = N // block_size           # static full tiles
    tail = N - num_full * block_size     # 0 <= tail < block_size
    num_logical = num_full + (1 if tail else 0)
    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a pure
    # Vector op; re-queried per build so other devices adapt automatically).
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_full, num_kernels)  # static bound (full tiles)

    # Trace-time (Python) branches: alpha == 1 identity fold; the scalar is a
    # plain Python float (the NPU binary-op front end takes a raw scalar
    # operand, not a cast expression).
    fold = alpha == 1
    alpha_scalar = float(alpha)

    @T.prim_func
    def main(
        a: T.Tensor((N,), dtype),
        b: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # UB buffers (DESIGN.md section 4.3).  All allocated
            # unconditionally; a32 doubles as the in-place result chain.
            a_ub = T.alloc_ub((block_size,), dtype)
            b_ub = T.alloc_ub((block_size,), dtype)
            a32 = T.alloc_ub((block_size,), compute_dtype)
            b32 = T.alloc_ub((block_size,), compute_dtype)
            # NOTE: an explicit y_ub is kept.  R1-A2 (aliasing y_ub = a_ub)
            # was rejected: expressed with a Python if/else inside the
            # prim_func it is traced as a runtime branch and y_ub stays scoped
            # inside the branch ("Undefined variable: y_ub",
            # TRAP-tvm-parser-rules), and the else-less variant with an
            # expression condition fails the same way.
            y_ub = T.alloc_ub((block_size,), out_dtype)

            # Persistent full-tile loop: static extents, grid-stride mapping,
            # if-guard masks out-of-range tiles.  Only executed when N has at
            # least one full tile.
            if num_full > 0:
                for i in T.Pipelined(num_local_tasks, num_stages=2):
                    block_id = i * num_kernels + cid
                    if block_id < num_full:
                        offset = block_id * block_size

                        # 1. Load GM -> UB (full tile, static extent).
                        T.copy(a[offset : offset + block_size], a_ub)
                        T.copy(b[offset : offset + block_size], b_ub)

                        # A6: native-fp16 fast path.  For alpha==1 the op is a
                        # SINGLE add of two fp16 values: the exact sum fits in
                        # fp32, so "fp32 add then round to fp16" and "native
                        # fp16 add" are both one correct rounding -> bitwise
                        # equal.  Skips 2 up-casts + 1 down-cast (4 vector
                        # passes -> 1) and 2/3 of the UB traffic.
                        # alpha!=1 keeps the fp32 chain (mul would double-round);
                        # bf16 keeps fp32 transit (no native bf16 v-prefix).
                        if dtype == "float16" and fold and out_dtype == "float16":
                            T.vadd(a_ub, b_ub, y_ub)
                        else:
                            # 2. Cast to fp32 (fp16/bf16) or copy (fp32).
                            if dtype != compute_dtype:
                                T.vcast(a_ub, a32, round_mode="rint")
                                T.vcast(b_ub, b32, round_mode="rint")
                            else:
                                T.copy(a_ub, a32)
                                T.copy(b_ub, b32)

                            # 3. Compute in fp32: fold alpha==1, else scaled add.
                            if fold:
                                T.vadd(a32, b32, a32)  # y = a + b (identity fold)
                            else:
                                T.vmul(b32, alpha_scalar, b32)  # t = alpha * b
                                T.vadd(a32, b32, a32)           # y = a + t

                            # 4. Cast back to output dtype (fp16/bf16) or copy.
                            if out_dtype != compute_dtype:
                                T.vcast(a32, y_ub, round_mode="rint")
                            else:
                                T.copy(a32, y_ub)

                        # 5. Store UB -> GM (full tile, static extent).
                        T.copy(y_ub, y[offset : offset + block_size])

            # Tail epilogue: at most one partial tile, owned by core 0.
            if tail > 0:
                if cid == 0:
                    offset = num_full * block_size
                    T.copy(a[offset : offset + tail], a_ub[0:tail])
                    T.copy(b[offset : offset + tail], b_ub[0:tail])
                    if dtype == "float16" and fold and out_dtype == "float16":
                        T.vadd(a_ub, b_ub, y_ub)
                    else:
                        if dtype != compute_dtype:
                            T.vcast(a_ub, a32, round_mode="rint")
                            T.vcast(b_ub, b32, round_mode="rint")
                        else:
                            T.copy(a_ub, a32)
                            T.copy(b_ub, b32)
                        if fold:
                            T.vadd(a32, b32, a32)
                        else:
                            T.vmul(b32, alpha_scalar, b32)
                            T.vadd(a32, b32, a32)
                        if out_dtype != compute_dtype:
                            T.vcast(a32, y_ub, round_mode="rint")
                        else:
                            T.copy(a32, y_ub)
                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


# ---------------------------------------------------------------------------
# Broadcast fast path (R6): row-reuse kernels
#
# ``AddFwdOp`` resolves broadcasting before dispatch (``_expand_flat``), so the
# flat kernel above always reads a *materialized* copy of ``other`` at the
# post-broadcast size: 3xN traffic.  When ``other`` is periodic in the flattened
# output index -- which holds for every right-aligned broadcast whose non-unit
# dims form one contiguous block -- only ``2xN + B`` bytes are semantically
# required.  Two kernels implement that:
#
#   row    : the broadcast pattern has period ``row`` (``other`` = a row that
#            repeats every ``row`` elements); one period is staged in UB once
#            per core and replicated across a full tile, then reused by every
#            tile of that core.  ``linear-bias`` ([1, 4096] vs [1024, 4096]).
#   scalar : each broadcast ``plane`` of ``L`` elements takes one scalar,
#            ``other[r % b_len]``; a tile covers ``planes`` consecutive planes
#            and the add uses the row-broadcast form ``(P,L) + (P,1)``.
#            ``cnn-feat-broadcast`` ([256, 1, 1] vs [16, 256, 56, 56]).
#
# Both keep the flat kernel's alpha semantics: ``alpha == 1`` folds the
# multiply; float16/bfloat16 with ``alpha == 1`` use a native single add (one
# correct rounding of an exact sum -> bitwise equal to the fp32 transit); every
# other case goes through the fp32 transit chain.
# ---------------------------------------------------------------------------


def _ub_bytes(elem_size, elems, alpha=1, extra=0):
    """UB footprint of one broadcast tile (dtype buffers + fp32 scratch)."""
    total = elem_size * (3 * elems + extra)
    if alpha != 1:
        # fp32 scaled-b scratch, plus the a32 transit for 2-byte dtypes.
        total += 4 * elems
        if elem_size != 4:
            total += 4 * elems
    return total


def plan_broadcast(
    input_shape,
    other_shape,
    elem_size,
    alpha=1,
    block_size=TUNED_BLOCK_SIZE,
):
    """Return a broadcast fast-path plan, or ``None`` to use the flat path.

    The plan is a pure function of the two *pre-broadcast* shapes, so the Op
    layer can use it both for dispatch and for roofline accounting (the byte
    count must not claim the re-use win when the flat path is taken).

    Structure of the accepted case: right-align ``other`` against the output
    shape (``E``), require ``input`` to already be the output shape, and
    require the non-unit dims of ``E`` to form one contiguous block.  Under
    those conditions the flattened ``other`` value at output index ``f`` is
    ``other_flat[f % B]`` -- fully determined by ``B = prod(other.shape)`` and
    the trailing run length ``L`` (the last dims of ``E`` that are 1):

    - ``L == 1`` -> row mode (period ``B``);
    - ``L > 1``  -> scalar mode (one scalar per ``L``-element plane).

    Returns a dict with ``mode`` plus the kernel parameters, or ``None``.
    """
    in_shape = tuple(int(x) for x in input_shape)
    oth_shape = tuple(int(x) for x in other_shape)
    out_shape = tuple(torch.broadcast_shapes(in_shape, oth_shape))
    # The fast path never gathers/scatters ``input``; a broadcast ``input``
    # would need one, so it stays on the flat path.
    if in_shape != out_shape:
        return None
    n_total = prod(out_shape) if out_shape else 1
    b_len = prod(oth_shape) if oth_shape else 1
    ndim = len(out_shape)
    if len(oth_shape) > ndim:
        return None
    extended = [1] * (ndim - len(oth_shape)) + list(oth_shape)
    unit_dims = [j for j in range(ndim) if extended[j] != 1]
    plan = {"N": n_total, "b_len": b_len, "out_shape": out_shape}
    # Per-tile UB budget, charged with the pipeline's own (dtype, alpha) so a
    # plan never overflows the 192 KB Vector UB.
    budget = _UB_CAPACITY_BYTES * _UB_SAFETY

    if not unit_dims:
        # Whole tensor is one scalar: value is ``other_flat[0]`` everywhere.
        if _ub_bytes(elem_size, block_size, alpha) > budget:
            return None
        plan.update(mode="row", row=1, tile=block_size, reps=0)
        return plan

    if unit_dims != list(range(unit_dims[0], unit_dims[-1] + 1)):
        # A unit dim sits between two real dims -> the pattern is not a
        # plain period; fall back to the flat path.
        return None

    run = 1
    for j in range(unit_dims[-1] + 1, ndim):
        run *= out_shape[j]

    if run == 1:
        # Row mode: ``other_flat`` repeats with period ``b_len``.
        if b_len == 1:
            tile, reps = block_size, 0
        else:
            mult = block_size // b_len
            if mult < 1:
                mult = 1
            # One UB->UB copy per replication: cap the count and shrink the
            # tile instead (rows below ~512 elements).
            mult = min(mult, _MAX_ROW_REPS)
            tile, reps = b_len * mult, mult
        if _ub_bytes(elem_size, tile, alpha, extra=b_len) > budget:
            return None
        plan.update(mode="row", row=b_len, tile=tile, reps=reps)
        return plan

    # Scalar mode: ``planes`` consecutive planes of ``L`` share one scalar each.
    lanes = run
    rows = n_total // lanes
    if elem_size != 4:
        # 2-byte dtypes cannot use the UB row-broadcast form (vbrc /
        # broadcast vadd costs 3-4x the copy floor: cnn bf16 66.8 us vs
        # 22.5 us).  Use the period path instead: materialize one full
        # broadcast period ``(B, L)`` once and stream same-shape slices from
        # it (L2-resident), i.e. the "copy2d + same-shape vadd" form measured
        # at ~22.5 us.  alpha != 1 keeps the flat path (the multiply would
        # need extra fp32 scratch and is not on any serving shape).
        if alpha != 1:
            return None
        if b_len * lanes * elem_size > _PERIOD_MAX_BYTES:
            return None
        max_period_planes = int(budget) // (3 * lanes * elem_size)
        planes = 0
        for cand in range(min(max_period_planes, _PERIOD_MAX_PLANES), 0, -1):
            if b_len % cand == 0 and rows % cand == 0:
                planes = cand
                break
        if planes < 1:
            return None
        if _ub_bytes(elem_size, planes * lanes, alpha) > budget:
            return None
        plan.update(
            mode="period",
            L=lanes,
            R=rows,
            B=b_len,
            planes=planes,
            period=b_len * lanes,
        )
        return plan
    max_planes = max(1, block_size // lanes)
    planes = 1
    for cand in range(max_planes, 0, -1):
        if b_len % cand == 0:
            planes = cand
            break
    if rows % planes:
        return None
    # 4 dtype tiles (a, b_tile, y + the broadcast source) + fp32 scratch.
    if _ub_bytes(elem_size, planes * lanes, alpha, extra=planes * lanes) > budget:
        return None
    plan.update(
        mode="scalar",
        L=lanes,
        R=rows,
        planes=planes,
        chunk=b_len // planes,
    )
    return plan


@tilelang.jit(out_idx=[2], target="npuir")
def add_bcast_row_kernel(N, row, dtype, alpha=1, block_size=TUNED_BLOCK_SIZE, num_cores=None):
    """``y = a + alpha * b`` where ``b`` is one row repeated every ``row``.

    ``a``/``y`` are flat ``(N,)``; ``b`` is the single broadcast period
    ``(row,)``.  Each core stages ``b`` once (``row`` elements) and replicates
    it across a ``block_size`` tile in UB, then walks its tiles with a
    grid-stride loop, so DRAM traffic is ``2*N + row`` instead of ``3*N``.
    """
    if row < 1 or block_size % row != 0:
        raise ValueError(f"block_size ({block_size}) must be a multiple of row ({row})")
    reps = block_size // row
    if row > 1 and reps > _MAX_ROW_REPS:
        raise ValueError(f"row {row} needs {reps} replications; use the flat path")
    compute_dtype = "float32"

    num_full = N // block_size
    tail = N - num_full * block_size
    num_logical = num_full + (1 if tail else 0)
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_full, num_kernels)

    fold = alpha == 1
    alpha_scalar = float(alpha)
    # alpha == 1: a single add of two dtype values is exact in fp32, so the
    # native add and "fp32 add + one rounding" agree bitwise (A6 finding).
    native = fold and dtype in ("float16", "bfloat16")
    # UB budget: the fp32 transit buffers exist only for the fp32-opmath path
    # (fp16/bf16 with alpha != 1); fp32-direct needs a scaled-b scratch only.
    a32_elems = block_size if (dtype != compute_dtype and not fold) else 1
    b32_elems = block_size if not fold else 1

    @T.prim_func
    def main(
        a: T.Tensor((N,), dtype),
        b: T.Tensor((row,), dtype),
        y: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((block_size,), dtype)
            b_ub = T.alloc_ub((block_size,), dtype)
            row_ub = T.alloc_ub((row,), dtype)
            a32 = T.alloc_ub((a32_elems,), compute_dtype)
            b32 = T.alloc_ub((b32_elems,), compute_dtype)
            y_ub = T.alloc_ub((block_size,), dtype)

            # Stage one period per core, then replicate it over the tile once.
            T.copy(b[0:row], row_ub)
            if row == 1:
                T.vbrc(row_ub, b_ub)
            else:
                for r in T.serial(reps):
                    T.copy(row_ub[0:row], b_ub[r * row : (r + 1) * row])

            if num_full > 0:
                for i in T.Pipelined(num_local_tasks, num_stages=2):
                    block_id = i * num_kernels + cid
                    if block_id < num_full:
                        offset = block_id * block_size
                        T.copy(a[offset : offset + block_size], a_ub)
                        if dtype == compute_dtype:
                            if fold:
                                T.vadd(a_ub, b_ub, y_ub)
                            else:
                                T.vmul(b_ub, alpha_scalar, b32)
                                T.vadd(a_ub, b32, y_ub)
                        elif native:
                            T.vadd(a_ub, b_ub, y_ub)
                        else:
                            T.vcast(a_ub, a32, round_mode="rint")
                            T.vcast(b_ub, b32, round_mode="rint")
                            T.vmul(b32, alpha_scalar, b32)
                            T.vadd(a32, b32, a32)
                            T.vcast(a32, y_ub, round_mode="rint")
                        T.copy(y_ub, y[offset : offset + block_size])

            if tail > 0:
                if cid == 0:
                    offset = num_full * block_size
                    T.copy(a[offset : offset + tail], a_ub[0:tail])
                    if dtype == compute_dtype:
                        if fold:
                            T.vadd(a_ub, b_ub, y_ub)
                        else:
                            T.vmul(b_ub, alpha_scalar, b32)
                            T.vadd(a_ub, b32, y_ub)
                    elif native:
                        T.vadd(a_ub, b_ub, y_ub)
                    else:
                        T.vcast(a_ub, a32, round_mode="rint")
                        T.vcast(b_ub, b32, round_mode="rint")
                        T.vmul(b32, alpha_scalar, b32)
                        T.vadd(a32, b32, a32)
                        T.vcast(a32, y_ub, round_mode="rint")
                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


@tilelang.jit(out_idx=[2], target="npuir")
def add_bcast_scalar_kernel(R, L, b_len, dtype, alpha=1, planes=1, num_cores=None):
    """``y[r, :] = a[r, :] + alpha * b[r % b_len]`` over ``R`` planes of ``L``.

    ``a``/``y`` are ``(R, L)``; ``b`` is viewed as ``(b_len // planes,
    planes, 1)`` so one tile's scalars are a single contiguous load.  The add
    is the UB row-broadcast form ``(planes, L) + (planes, 1)``: no per-element
    index arithmetic and no materialized broadcast operand.
    """
    if planes < 1 or b_len % planes:
        raise ValueError(f"planes ({planes}) must divide b_len ({b_len})")
    if R % planes:
        raise ValueError(f"planes ({planes}) must divide R ({R})")
    compute_dtype = "float32"
    chunk = b_len // planes
    groups = R // planes

    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > groups:
        num_kernels = min(groups, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(groups, num_kernels)

    fold = alpha == 1
    alpha_scalar = float(alpha)
    native = fold and dtype in ("float16", "bfloat16")
    a32_elems = (planes, L) if (dtype != compute_dtype and not fold) else (1, 1)

    @T.prim_func
    def main(
        a: T.Tensor((R, L), dtype),
        b: T.Tensor((chunk, planes, 1), dtype),
        y: T.Tensor((R, L), dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((planes, L), dtype)
            b_ub = T.alloc_ub((planes, 1), dtype)
            b_tile = T.alloc_ub((planes, L), dtype)
            a32 = T.alloc_ub(a32_elems, compute_dtype)
            b32 = T.alloc_ub((planes, L), compute_dtype)
            y_ub = T.alloc_ub((planes, L), dtype)

            for i in T.Pipelined(num_local_tasks, num_stages=2):
                group = i * num_kernels + cid
                if group < groups:
                    # b index: (group * planes) % b_len == (group % chunk) * planes
                    T.copy(a[group * planes : (group + 1) * planes, 0:L], a_ub)
                    T.copy(b[group % chunk, 0:planes, 0:1], b_ub)
                    # UB row-broadcast ``vadd((P,L), (P,1))`` is far slower than
                    # the same-shape form for 2-byte dtypes (measured 2.8x on
                    # cnn bf16), so the per-plane scalars are materialized into
                    # the tile with the dedicated broadcast op instead: one
                    # extra vector pass, fully hidden behind the MTE2 stream.
                    T.vbrc(b_ub, b_tile)
                    if dtype == compute_dtype:
                        if fold:
                            T.vadd(a_ub, b_tile, y_ub)
                        else:
                            T.vmul(b_tile, alpha_scalar, b32)
                            T.vadd(a_ub, b32, y_ub)
                    elif native:
                        T.vadd(a_ub, b_tile, y_ub)
                    else:
                        T.vcast(a_ub, a32, round_mode="rint")
                        T.vcast(b_tile, b32, round_mode="rint")
                        T.vmul(b32, alpha_scalar, b32)
                        T.vadd(a32, b32, a32)
                        T.vcast(a32, y_ub, round_mode="rint")
                    T.copy(y_ub, y[group * planes : (group + 1) * planes, 0:L])

    return main


@tilelang.jit(out_idx=[2], target="npuir")
def add_bcast_period_kernel(R, L, B, dtype, alpha=1, planes=1, num_cores=None):
    """``y[r, :] = a[r, :] + b[(r % B), :]`` via a materialized period.

    For 2-byte scalar broadcast (``cnn-feat``: ``(16,256,56,56) + (256,1,1)``):
    ``a``/``y`` are ``(R, L)`` and ``b`` is one full broadcast period
    ``(B, L)`` whose row ``j`` is the scalar of output plane ``j % B``.  A tile
    covers ``planes`` consecutive planes; the period slice is read from GM
    (L2-resident) and added with the same-shape ``vadd`` -- no 2-byte UB
    broadcast, which costs 3-4x on this toolchain (REPORT_ADD_BCAST.md §4).

    Preconditions (enforced by ``plan_broadcast``): ``alpha == 1``,
    ``planes | B``, ``planes | R``, and the period fits the L2 cap.
    """
    if alpha != 1:
        raise ValueError("the period path requires alpha == 1")
    if planes < 1 or B % planes or R % planes:
        raise ValueError(f"planes ({planes}) must divide B ({B}) and R ({R})")
    groups = R // planes
    chunk = B // planes

    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > groups:
        num_kernels = min(groups, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(groups, num_kernels)

    @T.prim_func
    def main(
        a: T.Tensor((R, L), dtype),
        b: T.Tensor((B, L), dtype),
        y: T.Tensor((R, L), dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((planes, L), dtype)
            b_ub = T.alloc_ub((planes, L), dtype)
            y_ub = T.alloc_ub((planes, L), dtype)

            for i in T.Pipelined(num_local_tasks, num_stages=2):
                group = i * num_kernels + cid
                if group < groups:
                    p = (group % chunk) * planes
                    T.copy(a[group * planes : (group + 1) * planes, 0:L], a_ub)
                    T.copy(b[p : p + planes, 0:L], b_ub)
                    T.vadd(a_ub, b_ub, y_ub)
                    T.copy(y_ub, y[group * planes : (group + 1) * planes, 0:L])

    return main


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
def _max_diff_metric(out, ref):
    """Return the max absolute difference between out and ref.

    NaN/Inf aware: positions where both are NaN (or both equal infinities)
    count as equal and contribute zero difference.
    """
    eq = out == ref
    eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    return diff.max().item()


def _record_stat(dtype_str, alpha, max_diff):
    if max_diff == max_diff:  # NaN guard
        key = f"{dtype_str}/alpha={alpha}"
        if max_diff > _STATS_MAX_DIFF.get(key, 0.0):
            _STATS_MAX_DIFF[key] = max_diff


def _run_tensor(a, b, dtype_str, tag, alpha=1, block_size=None, equal_nan=False):
    """Run kernel on pre-made same-shape 1-D NPU tensors and compare vs golden.

    Raises AssertionError on tolerance violation.  Returns max_diff.
    """
    N = a.numel()
    atol, rtol = _TOLERANCE[dtype_str]
    kwargs = {}
    if block_size is not None:
        kwargs["block_size"] = block_size
    kernel = add_fwd_kernel(N, dtype_str, alpha=alpha, **kwargs)
    y_cpu = kernel(a, b, N).cpu()
    golden = golden_add(a, b, alpha)
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol, equal_nan=equal_nan)
    max_diff = _max_diff_metric(y_cpu, golden)
    _record_stat(dtype_str, alpha, max_diff)
    print(
        f"[{tag}] PASS: N={N} dtype={dtype_str} alpha={alpha} "
        f"block_size={block_size} max_diff={max_diff:.3e}"
    )
    return max_diff


def _run_case(N, dtype_str, tag, alpha=1, block_size=None, scale=1.0):
    """Run kernel on random data of shape (N,) with the given dtype."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    a = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    b = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    return _run_tensor(a, b, dtype_str, tag, alpha=alpha, block_size=block_size)


def _run_broadcast_case(a_shape, b_shape, dtype_str, tag, alpha=1):
    """Broadcast case: Op layer expands/flattens, kernel sees same-shape 1-D.

    Golden uses torch's native broadcast on the original shapes (independent
    of the NPU-side Op-layer flattening).
    """
    torch_dtype = _TORCH_DTYPE[dtype_str]
    out_shape = tuple(torch.broadcast_shapes(a_shape, b_shape))
    a = torch.randn(a_shape, dtype=torch.float32).to(torch_dtype)
    b = torch.randn(b_shape, dtype=torch.float32).to(torch_dtype)
    golden = golden_add(a, b, alpha)
    N = 1
    for d in out_shape:
        N *= d
    a_flat = a.expand(out_shape).contiguous().view(-1).npu()
    b_flat = b.expand(out_shape).contiguous().view(-1).npu()
    kernel = add_fwd_kernel(N, dtype_str, alpha=alpha)
    y_cpu = kernel(a_flat, b_flat, N).cpu().view(out_shape)
    atol, rtol = _TOLERANCE[dtype_str]
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol)
    max_diff = _max_diff_metric(y_cpu, golden)
    _record_stat(dtype_str, alpha, max_diff)
    print(
        f"[{tag}] PASS: A={tuple(a_shape)} B={tuple(b_shape)} -> {out_shape} "
        f"N={N} dtype={dtype_str} alpha={alpha} max_diff={max_diff:.3e}"
    )
    return max_diff


def _run_reuse_case(a_shape, b_shape, dtype_str, tag, alpha=1):
    """Broadcast reuse path: row/scalar kernel vs golden and vs the flat path.

    Exercises ``plan_broadcast`` -> ``add_bcast_row_kernel`` /
    ``add_bcast_scalar_kernel`` end to end, including the bitwise identity
    with the flat (expand-and-flatten) kernel.
    """
    torch_dtype = _TORCH_DTYPE[dtype_str]
    out_shape = tuple(torch.broadcast_shapes(a_shape, b_shape))
    a = torch.randn(a_shape, dtype=torch.float32).to(torch_dtype)
    b = torch.randn(b_shape, dtype=torch.float32).to(torch_dtype)
    golden = golden_add(a, b, alpha)
    plan = plan_broadcast(a_shape, b_shape, torch_dtype.itemsize, alpha=alpha)
    if plan is None:
        print(
            f"[{tag}] FALLBACK: A={tuple(a_shape)} B={tuple(b_shape)} dtype={dtype_str} "
            f"alpha={alpha} -> flat path"
        )
        return _run_broadcast_case(a_shape, b_shape, dtype_str, tag, alpha=alpha)
    N = 1
    for d in out_shape:
        N *= d
    a_npu = a.expand(out_shape).contiguous().npu()
    b_npu = b.npu()
    if plan["mode"] == "row":
        kernel = add_bcast_row_kernel(
            plan["N"], plan["row"], dtype_str, alpha=alpha, block_size=plan["tile"]
        )
        y = kernel(a_npu.view(-1), b_npu.view(-1), plan["N"])
    else:
        kernel = add_bcast_scalar_kernel(
            plan["R"],
            plan["L"],
            plan["b_len"],
            dtype_str,
            alpha=alpha,
            planes=plan["planes"],
        )
        y = kernel(
            a_npu.view(plan["R"], plan["L"]),
            b_npu.view(plan["chunk"], plan["planes"], 1),
            plan["R"],
        )
    y_cpu = y.cpu().view(out_shape)
    atol, rtol = _TOLERANCE[dtype_str]
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol)
    max_diff = _max_diff_metric(y_cpu, golden)
    # Bitwise identity with the flat path (same rounding, same alpha fold).
    flat = add_fwd_kernel(N, dtype_str, alpha=alpha)(
        a.expand(out_shape).contiguous().view(-1).npu(),
        b.expand(out_shape).contiguous().view(-1).npu(),
        N,
    )
    int_view = torch.int32 if dtype_str == "float32" else torch.int16
    bitwise = torch.equal(
        y_cpu.reshape(-1).view(int_view),
        flat.cpu().reshape(-1).view(int_view),
    )
    _record_stat(dtype_str, alpha, max_diff)
    print(
        f"[{tag}] PASS: A={tuple(a_shape)} B={tuple(b_shape)} -> {out_shape} "
        f"mode={plan['mode']} N={N} dtype={dtype_str} alpha={alpha} "
        f"max_diff={max_diff:.3e} bitwise_vs_flat={bitwise}"
    )
    if not bitwise:
        raise AssertionError(f"{tag}: reuse path differs bitwise from the flat path")
    return max_diff


def _try_case(failures, case_id, fn, **kwargs):
    """Run a case, collecting (not re-raising) per-case failures."""
    try:
        fn(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


# ---------------------------------------------------------------------------
# Layer suites
# ---------------------------------------------------------------------------
_L0_N = 1048576  # 2^20, shape [1024, 1024] same-shape (DESIGN.md section 8.3)


def run_L0():
    """L0 gate: all dtypes x alpha, non-divisible tail, broadcast path.

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # L0-1: smoke, all dtypes, N=2^20 same-shape, default alpha.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _try_case(failures, f"L0-1/{dtype_str}", _run_case, N=_L0_N, dtype_str=dtype_str, tag="L0-1")

    # L0-2: nonzero alpha (scalar multiply path), all dtypes.
    for dtype_str in ("float32", "float16", "bfloat16"):
        for alpha in (2.0, -1.5):
            _try_case(
                failures,
                f"L0-2/{dtype_str}/alpha={alpha}",
                _run_case,
                N=_L0_N,
                dtype_str=dtype_str,
                tag="L0-2",
                alpha=alpha,
            )

    # L0-3: non-divisible tail (partial tile + guard).
    for dtype_str in ("float16", "float32"):
        for N in (10000, 4097):
            _try_case(
                failures,
                f"L0-3/{dtype_str}/N={N}",
                _run_case,
                N=N,
                dtype_str=dtype_str,
                tag="L0-3",
                alpha=2.0,
            )

    # L0-4: broadcast path (Op layer expand/flatten).
    _try_case(
        failures,
        "L0-4/fp32/[64,128]x[1,128]",
        _run_broadcast_case,
        a_shape=(64, 128),
        b_shape=(1, 128),
        dtype_str="float32",
        tag="L0-4",
    )
    _try_case(
        failures,
        "L0-4/fp16/[2,512,768]x[1,1,768]",
        _run_broadcast_case,
        a_shape=(2, 512, 768),
        b_shape=(1, 1, 768),
        dtype_str="float16",
        tag="L0-4",
        alpha=1.0,
    )

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x alpha x shape x block_size).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # L1-1: alpha sweep (incl. 0.0 and negative) x all dtypes.
    for dtype_str in ("float16", "bfloat16", "float32"):
        for alpha in (0.0, -1.0, 0.5, 3):
            _try_case(
                failures,
                f"L1-1/{dtype_str}/alpha={alpha}",
                _run_case,
                N=1 << 16,
                dtype_str=dtype_str,
                tag="L1-1",
                alpha=alpha,
            )

    # L1-2: shape / tail coverage (exact multiple, tail==1-ish, odd tail).
    for dtype_str in ("float16", "bfloat16", "float32"):
        for N in (8192, 8191, 1, 100):
            _try_case(
                failures,
                f"L1-2/{dtype_str}/N={N}",
                _run_case,
                N=N,
                dtype_str=dtype_str,
                tag="L1-2",
            )

    # L1-3: non-default block_size (UB budget still holds).
    _try_case(
        failures,
        "L1-3/fp16/bs=2048",
        _run_case,
        N=1 << 20,
        dtype_str="float16",
        tag="L1-3",
        block_size=2048,
    )
    _try_case(
        failures,
        "L1-3/fp32/bs=4096",
        _run_case,
        N=1 << 20,
        dtype_str="float32",
        tag="L1-3",
        block_size=4096,
    )

    # L1-4: broadcast coverage across dtypes/shapes.
    _try_case(
        failures,
        "L1-4/bf16/[1,4096]x[1024,4096]",
        _run_broadcast_case,
        a_shape=(1, 4096),
        b_shape=(1024, 4096),
        dtype_str="bfloat16",
        tag="L1-4",
    )
    _try_case(
        failures,
        "L1-4/fp16/[2,512,768]x[2,512,1]",
        _run_broadcast_case,
        a_shape=(2, 512, 768),
        b_shape=(2, 512, 1),
        dtype_str="float16",
        tag="L1-4",
    )
    _try_case(
        failures,
        "L1-4/fp32/[4,1,8]x[1,8,1]",
        _run_broadcast_case,
        a_shape=(4, 1, 8),
        b_shape=(1, 8, 1),
        dtype_str="float32",
        tag="L1-4",
    )

    # L1-5: broadcast reuse kernels (row / scalar / whole-tensor scalar).
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(
            failures,
            f"L1-5/{dtype_str}/linear-bias",
            _run_reuse_case,
            a_shape=(1024, 4096),
            b_shape=(1, 4096),
            dtype_str=dtype_str,
            tag="L1-5",
        )
    _try_case(
        failures,
        "L1-5/bf16/cnn-feat",
        _run_reuse_case,
        a_shape=(16, 256, 56, 56),
        b_shape=(256, 1, 1),
        dtype_str="bfloat16",
        tag="L1-5",
    )
    _try_case(
        failures,
        "L1-5/fp16/cnn-feat/alpha=2",
        _run_reuse_case,
        a_shape=(16, 256, 56, 56),
        b_shape=(256, 1, 1),
        dtype_str="float16",
        tag="L1-5",
        alpha=2.0,
    )
    # scalar mode is fp32-only (2-byte UB broadcast is 3-4x slower); this is
    # the case that actually exercises it.
    _try_case(
        failures,
        "L1-5/fp32/cnn-feat",
        _run_reuse_case,
        a_shape=(16, 256, 56, 56),
        b_shape=(256, 1, 1),
        dtype_str="float32",
        tag="L1-5",
    )
    _try_case(
        failures,
        "L1-5/fp16/[64,128]x[1,128]",
        _run_reuse_case,
        a_shape=(64, 128),
        b_shape=(1, 128),
        dtype_str="float16",
        tag="L1-5",
    )
    _try_case(
        failures,
        "L1-5/fp16/[1024,1024]x[1,1]",
        _run_reuse_case,
        a_shape=(1024, 1024),
        b_shape=(1, 1),
        dtype_str="float16",
        tag="L1-5",
    )
    _try_case(
        failures,
        "L1-5/bf16/col-bcast",
        _run_reuse_case,
        a_shape=(1024, 4096),
        b_shape=(1024, 1),
        dtype_str="bfloat16",
        tag="L1-5",
    )

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes (warn only, non-blocking)."""
    for N, dtype_str in ((1, "float32"), (2, "float16"), (10, "bfloat16"), (1 << 24, "float16")):
        try:
            _run_case(N, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking).

    Covers IEEE NaN/+-inf propagation, the alpha==0 no-fold requirement
    (``0*inf = NaN``), and the alpha==1 identity fold on special values.
    """
    N = 4096

    # (+-inf / NaN in a and b) x alpha={1 fold, 2.0 scale}: NaN/Inf propagate.
    for dtype_str in ("float32", "float16", "bfloat16"):
        for alpha in (1, 2.0):
            try:
                torch_dtype = _TORCH_DTYPE[dtype_str]
                a = torch.randn(N, dtype=torch.float32)
                b = torch.randn(N, dtype=torch.float32)
                a[0::8] = float("inf")
                a[1::8] = float("-inf")
                a[2::8] = float("nan")
                b[3::8] = float("inf")
                b[4::8] = float("-inf")
                b[5::8] = float("nan")
                a[6::8] = float("inf")
                b[6::8] = float("-inf")
                a[7::8] = float("-inf")
                b[7::8] = float("inf")
                _run_tensor(
                    a.to(torch_dtype).npu(),
                    b.to(torch_dtype).npu(),
                    dtype_str,
                    "Boundary-infnan",
                    alpha=alpha,
                    equal_nan=True,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"[Boundary] WARN (Record without blocking): inf-nan "
                    f"dtype={dtype_str} alpha={alpha} err={e}"
                )

    # alpha == 0 must NOT fold: b=inf -> 0*inf=NaN -> y=NaN (torch semantics).
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            a = torch.randn(N, dtype=torch.float32)
            b = torch.randn(N, dtype=torch.float32)
            b[0::4] = float("inf")
            a = a.to(torch_dtype).npu()
            b = b.to(torch_dtype).npu()
            kernel = add_fwd_kernel(N, dtype_str, alpha=0.0)
            y_cpu = kernel(a, b, N).cpu()
            golden = golden_add(a, b, alpha=0.0)
            torch.testing.assert_close(y_cpu, golden, atol=0.0, rtol=0.0, equal_nan=True)
            print(
                f"[Boundary] PASS: alpha==0 no-fold (0*inf=NaN) dtype={dtype_str} "
                f"nan_out={int(torch.isnan(y_cpu).sum().item())}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): alpha=0 dtype={dtype_str} err={e}")

    # -0.0 / subnormal magnitudes (alpha=2.0).
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            a = (torch.randn(N, dtype=torch.float32) * 1e-6).to(torch_dtype)
            b = torch.randn(N, dtype=torch.float32).to(torch_dtype)
            a[0:4] = torch.tensor([0.0, -0.0, 0.0, -0.0], dtype=torch_dtype)
            _run_tensor(a.npu(), b.npu(), dtype_str, "Boundary-signedzero", alpha=2.0)
        except Exception as e:  # noqa: BLE001
            print(
                f"[Boundary] WARN (Record without blocking): signed-zero "
                f"dtype={dtype_str} err={e}"
            )

    # Large magnitudes (alpha=-1.5, cancellation corners within tolerance).
    try:
        _run_case(1 << 20, "float16", "Boundary-large", alpha=-1.5, scale=100.0)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large err={e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    t0 = time.time()
    failures += run_L0()
    print(f"[L0] elapsed {time.time() - t0:.1f}s")

    if args.level == "all":
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    print(
        "Summary max_diff per (dtype, alpha): "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
