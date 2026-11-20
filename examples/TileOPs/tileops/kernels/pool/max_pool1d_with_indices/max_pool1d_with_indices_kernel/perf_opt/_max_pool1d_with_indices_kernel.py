# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Stage 4 final -- windowed max pooling 1D with int64 indices (NPU).

Derived from the read-only Stage 3 baseline.  Round 1 (910B4 / tilelang
e8b5d51) produced the C-axis general grid with the fp32-domain epilogue
decisions and the exact UB footprint (CH -> C); round 2 (2026-10-20,
**Ascend910B3 / tilelang d2783bd**) adds two structural paths on top of it:

  * **phase-decimated wide-tile path** (``s_w == 2``, ``p_w``/``d_w`` even,
    ``l_in`` even): the stride-2 decimation is one ``T.deinterleave``, so the
    window becomes a stride-``d_w/2`` sliding window in the even-phase plane and
    each tap costs *one* wide vector op instead of ``bl`` narrow ones.
    ecg 339.71 -> 100.20 us on this device.
  * **1-D flat store** for ``L_out == 1``: the channel block is contiguous in GM,
    so the value/index store drops the ``(bl,CH) -> (CH,bl)`` transpose and the
    1-element row region.  Measured on the store alone: 85.60 -> 3.44 us,
    ``UB_to_GM`` 96 KB -> 3.8 KB (real write).  textcnn 133.54 -> 107.46 us.
  * the general grid is unchanged and still handles every other shape
    (sincnet k3/s3 stays on it: 496.4 us before and after, non-regression).

Round 3 (2026-10-20, same 910B3 / d2783bd) adds the third structural path:

  * **`L_out == 1` halving-reduce path** (``k_w == l_in``, ``s_w == 1``,
    ``p_w == 0``, ``d_w == 1``, ``l_in`` a power of two, ``l_in <= 128``): the
    single window *is* the whole line, so the ``k_w * bl`` narrow-op forest
    (2 passes x k taps x 3 ops) collapses into ``log2(l_in)`` wide halving
    stages over the transposed ``(l_in, CH)`` tile.  Values fold with ``vmax``;
    the first hit is *not* folded alongside the value (the halving tree pairs
    ``i`` with ``i + l_in/2``, so "left child wins" is not "smallest index
    wins") -- it is recovered as ``min index among the lanes equal to the
    folded maximum`` (``vbrc`` + ``vcmp eq`` + ``vselect(ramp, BIG)`` +
    ``vmin`` fold).  textcnn 107.46 us -> see perf_opt/opt_log.md section 12.

See perf_opt/opt_log.md for the per-round evidence and the residual list.

Original header:
Windowed max pooling 1D forward kernel with int64 indices (NPU, Developer).

Migrated from the GPU repo's ``_max_pool1d_with_indices_kernel`` (TileOPs
``MaxPool1dIndicesFwdOp``, ``ref_api = torch.nn.functional.max_pool1d(...,
return_indices=True)``) to TileLang ``target="npuir"``.  The value-only sibling
in the same source file is out of scope (already migrated by a previous task).

Semantics (DESIGN.md section 0.1, frozen; matches
``torch.nn.functional.max_pool1d(..., return_indices=True)``):
    out[n, c, ow]     = max over the kW window anchored at (ow*sW - pW + kw*dW)
    indices[n, c, ow] = iw (int64, the position along the length axis)
    Out-of-range window positions behave as -inf padding (the source kernel
    skips those lanes; max(-inf, x) == x -- machine-verified as E1 in
    ``verify_equiv.py``).  Ties keep the first hit (strict ``>`` record).  A NaN
    anywhere in the window wins: the value is NaN and the index is the *last*
    NaN visited (``max`` of the NaN positions, because the visit order is
    strictly increasing).  ``{+inf, -inf}`` without NaN keeps the IEEE maximum
    (``+inf``).  Empty windows (no in-bounds lane; possible only when
    ``d_W >= 2 and p_W > 0``) align with the source state machine:
    ``(value, index) = (-inf, 0)``; torch returns an out-of-range index there,
    so torch is not an oracle for that regime (DESIGN.md sections 0.1 / 8.3
    L0-15).
    Exactly 3D NCL, float16 / bfloat16 / float32, same dtype out, int64
    indices.  The output extent comes from the source helper
    ``pool_output_dim`` (including both ceil_mode corrections) and is never
    re-derived.

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel`` + ``(n, o_blk, c_block)`` logical units with a
        persistent in-core ``T.serial`` grid-stride loop and an
        ``if uid < NUM_LOGICAL`` tail guard (static bounds).  The length axis is
        chunked into output blocks ``BL`` because ``L_in`` reaches 4096 (a whole
        ``(CH, L_in)`` line does not fit UB).
    R2: UB fp32 vector accumulators on the channel axis (the vector axis is C).
        Maximum = in-place ``T.vmax`` on a C-vector slice; NaN tracking = an
        explicit IEEE ``x == x`` eq mask multiplied into a flag accumulator
        seeded to 1 (0 absorbing, 1 neutral).  The kernel never relies on the
        undocumented ``T.vmax`` NaN behaviour.
    R3: int32 vector index accumulators.  ``max_idx`` = first-hit = the minimum
        ``iw`` whose value equals the window maximum (a *second* pass with
        ``vcmp(eq)`` + ``vselect`` + ``vmin``; the value accumulator must be
        final before it runs).  ``nan_idx`` = last-NaN = the maximum ``iw`` over
        NaN lanes (first pass, ``vcmp(eq)`` + ``vselect`` + ``vmax``).  Both
        outputs share one NaN decision ``cfinal = (nan_idx >= 0)``; the int32
        indices are cast to int64 (``rint``) before the store.
    R4: contiguous clamped GM span -> UB ``T.copy`` + in-core two-axis
        ``T.transpose`` ([1, 0]) so the channel axis becomes the contiguous
        vector axis; outputs are transposed back before the span store.
    R5: no geometric specialisation.  C4 (whole-line reduction) / C4b
        (identity) are mathematically sound for the *value* half but the
        index half would need a ``T.arange``-derived in-line index pattern fed
        into a ``vselect`` + reduce chain, which is a chain verified broken on
        this toolchain (``TRAP-npuir-arange-pattern-with-whole-plane-reduce``:
        values stay exact while the index output is corrupted).  Every shape --
        including ``textcnn-global`` with ``L_out == 1`` -- therefore takes the
        general grid, whose correctness is identical (single code path, no
        branch).

Frozen implementation-order constraints (DESIGN.md sections 1.6.1 N3 / 3.3 /
6.2 and section 9.2):
    * ``T.vbrc`` output must be a whole buffer; a scalar broadcast into a buffer
      region with a runtime offset is silently dropped by this toolchain, and
      the int32 form then reads uninitialised UB (vector-core fault).  Every
      per-``owl`` operand is a standalone whole ``(1, CH)`` buffer here.
    * A ``T.vbrc`` scalar operand must be a typed PrimExpr (``T.int32(iw)``,
      ``T.float32(1.0)``); a raw Python number fails the frontend rank assert.
    * The block-level ``vcmp(..., "eq")`` + ``vcast`` NaN mask must be computed
      before the in-loop ``T.vmul``.
    * The first-hit pass must run after *every* first-pass ``vmax`` / ``vmul`` /
      ``vmax(nan_idx)`` (the value accumulator must be final).
    * ``vcmp(nan_idx, zero_i32, "ge")`` runs after both passes, and the
      empty-window clause runs after ``out32`` is selected (it reuses the
      ``cfinal`` bool buffer).
    * ``T.vcast`` with target == source dtype miscompiles on this toolchain, so
      float32 inputs use ``T.copy`` for the same-shape same-dtype transfers.
    * ``@tilelang.jit`` on this branch does not accept ``compile_flags`` (the
      source's ``["-O3", "-DENABLE_BF16"]`` is dropped; the verified same-family
      kernels do the same).

Round-2 constraints (measured on 910B3 / d2783bd, 2026-10-20):
    * ``T.deinterleave`` only with ``channel_nums=2`` (3 and 4 both SIGABRT
      ``bishengir-compile``), and never inside a runtime ``if/else`` (the
      ``vdeinterleave`` is silently dropped and the phase plane reads
      uninitialised UB).
    * ``T.vselect`` accepts **no** sliced operand (``hivm.hir.vsel`` reports
      ``operands' shape are inconsistent``) while ``vcmp`` / ``vmax`` / ``vmin``
      / ``vadd`` accept same-shape slices -- the phase path's first-hit mask is
      therefore materialised into a whole ``(bl, CH)`` bool and folded with
      ``vadd`` on a staged copy of the index-ramp slice.
    * ``T.vcast`` with target == source dtype miscompiles (both directions are
      branched on ``is_f32`` in the phase path and the general grid).
    * The phase path's UB footprint proxy (``_phase_bytes``) is calibrated
      against the backend: 258 KB naive sum compiles, 319 KB overflows.
    * ``CH == 32`` is excluded from the phase-path channel candidates (measured
      wrong index at that width; see ``_PHASE_CH_EXCLUDED``).

Factory contract (source interface unchanged):
    _max_pool1d_with_indices_kernel(n, c_in, l_in, kernel_w, stride_w, pad_w,
                                    dilation_w, ceil_mode, dtype="float16")
      -> build(block_m=None) -> main
    where ``main(x)`` returns ``(out, indices)`` (``out_idx=[1, 2]``).

Run the embedded hierarchical tests:
    python _max_pool1d_with_indices_kernel.py --level L0
    python _max_pool1d_with_indices_kernel.py --level all
"""

import os

# Developer mode: alloc_ub maps to UB, the compiler auto-synchronises
# (DESIGN.md sections 2.1 and 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import time

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

from tileops.kernels._factory_cache import kernel_factory_cache

# Programming mode actually verified in Stage 3 (Stage 5 syncs the wrapper's
# Kernel.ascend_mode from this literal).
ASCEND_MODE = "Developer"

__all__ = [
    "ASCEND_MODE",
    "_max_pool1d_with_indices_kernel",
    "golden_max_pool1d_with_indices",
    "pool_output_dim",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Unified Buffer capacity per Vector core (developer guide: 192 KB).
_UB_CAPACITY_BYTES = 196_608

# Multi-buffer inflation factor for the UB budget guard (DESIGN.md section 4.5,
# default tier: auto-multi-buffer enabled).
_UB_MULTIBUFFER_FACTOR = 1.7

# Supported dtypes (the source Op layer rejects float64).
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# Precision tolerances (DESIGN.md section 8.2): max selects an input element and
# performs no accumulation, so the value output is bit-exact; the index
# comparison is exact int64.
_TOLERANCE = {
    "float16": (torch.float16, 0.0, 0.0),
    "bfloat16": (torch.bfloat16, 0.0, 0.0),
    "float32": (torch.float32, 0.0, 0.0),
}

# fp16/bf16 vector width in elements (alignment of the vector axis).
_VECTOR_WIDTH = 8

# Target channel-vector-block ceiling (Stage 4 tiling re-derivation).
# C is 80 / 256 / 128 for the three manifest full workloads, so 128 lets the two
# C=128-shaped workloads run a single channel block (no padding lanes and half
# the logical units); the capacity inequality below then picks BL.
_CH_MAX = 128

# Output-position block ceiling (DESIGN.md section 5.2; halo shrinks with BL,
# capacity caps it).
_BL_MAX = 32

# int32 index carrier bound (the length index must fit int32 before the cast).
_INT32_INDEX_LIMIT = 2**31


def _elem_bytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2


# ---------------------------------------------------------------------------
# Source helper (verbatim from the GPU repo's kernels/pool/common.py, inlined
# into the Stage 0 extraction package).  The output extent must never be
# re-derived anywhere else.
# ---------------------------------------------------------------------------
def pool_output_dim(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
    dilation: int = 1,
) -> int:
    """Output extent of one pooling axis (PyTorch max_pool1d rule)."""
    effective_kernel = dilation * (kernel_size - 1) + 1
    if ceil_mode:
        out = (input_size + 2 * padding - effective_kernel + stride - 1) // stride + 1
    else:
        out = (input_size + 2 * padding - effective_kernel) // stride + 1

    if ceil_mode and out > 0 and (out - 1) * stride >= input_size + padding:
        out -= 1

    return max(out, 0)


# ---------------------------------------------------------------------------
# Tiling plan (host/trace side): CH/BL joint selection, core split
# ---------------------------------------------------------------------------
def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _span(bl: int, k_w: int, s_w: int, d_w: int, l_in: int) -> int:
    """Contiguous input span covered by one BL output block, clamped to L_in."""
    return min((bl - 1) * s_w + (k_w - 1) * d_w + 1, l_in)


def _general_bytes(ch: int, lspan: int, bl: int, elem_bytes: int) -> int:
    """Per-core UB footprint of the general path (bytes, no inflation).

    Stage 4 recount (tiling re-derivation): the Stage 3 closed form
    ``G = CH*LSPAN*(2*eb+9) + BL*CH*(69+2*eb) + 256`` budgeted the buffers of the
    *two-pass + dead NaN-flag* implementation.  After the dead-code removal and
    the fp32-domain NaN/empty-window decisions the live set is exactly

        G = CH*LSPAN*(2*eb + 4) + BL*CH*(57 + 2*eb) + 14*CH

    ``2*eb + 4``   = row_src(eb) + row_dst(eb) + row_f32(4)
    ``57 + 2*eb``  = acc / infbuf / nanbuf / out32 / nan_f32 / idx_f32 (6 x f32 = 24)
                     + cfinal (bool 1) + max_idx / nan_idx / idx32 / zeros_i /
                       bigv (5 x i32 = 20) + idx_t (4) + idx64 (8)
                     + acc16 + acc_t (2*eb)
    ``14*CH``      = nanc / fvec / neg1 (3 x i32 = 12) + eqb / eq2b (2 x bool)

    The per-buffer sum is asserted against the closed form by
    ``_general_bytes_selfcheck()`` at import time.
    """
    per_span = 2 * elem_bytes + 4
    per_out = 57 + 2 * elem_bytes
    return ch * lspan * per_span + bl * ch * per_out + 14 * ch


def _general_bytes_selfcheck() -> None:
    """Per-buffer sum == closed form (guards the capacity inequality)."""
    for eb in (2, 4):
        live_out = (
            6 * 4  # acc, infbuf, nanbuf, out32, nan_f32, idx_f32
            + 1  # cfinal
            + 5 * 4  # max_idx, nan_idx, idx32, zeros_i, bigv
            + 4  # idx_t
            + 8  # idx64
            + 2 * eb  # acc16, acc_t
        )
        live_span = eb + eb + 4  # row_src, row_dst, row_f32
        live_vec = 3 * 4 + 2  # nanc, fvec, neg1, eqb, eq2b
        assert live_out == 57 + 2 * eb, live_out
        assert live_span == 2 * eb + 4, live_span
        assert live_vec == 14, live_vec


def _choose_bl(ch, l_in, out_l, k_w, s_w, d_w, elem_bytes):
    """Largest BL (<= min(L_out, BL_MAX)) whose general footprint fits UB.

    Returns 0 when even BL=1 does not fit at this CH.
    """
    bl = min(out_l, _BL_MAX)
    while bl >= 1:
        lspan = _span(bl, k_w, s_w, d_w, l_in)
        if _general_bytes(ch, lspan, bl, elem_bytes) * _UB_MULTIBUFFER_FACTOR <= _UB_CAPACITY_BYTES:
            return bl
        bl -= 1
    return 0


def _choose_ch_bl(c_in, l_in, out_l, k_w, s_w, d_w, elem_bytes):
    """Channel vector block + output block (DESIGN.md sections 4.5 / 5.2).

    CH is the largest multiple of 8 <= min(C, 64); BL is the largest block at
    that CH satisfying the capacity inequality (BL <= min(L_out, 32)).  CH is
    halved (never below 8) while no BL fits.  Returns BL=0 when even CH=8/BL=1
    is infeasible (the caller rejects such shapes).
    """
    ch = max(_VECTOR_WIDTH, _round_up(min(c_in, _CH_MAX), _VECTOR_WIDTH))
    while True:
        bl = _choose_bl(ch, l_in, out_l, k_w, s_w, d_w, elem_bytes)
        if bl >= 1:
            return ch, bl
        if ch == _VECTOR_WIDTH:
            return ch, 0
        ch = max(_VECTOR_WIDTH, _round_up(ch // 2, _VECTOR_WIDTH))


# ---------------------------------------------------------------------------
# Phase-decimated wide-tile path (Stage 4 round 2, s_w == 2 specialisation)
# ---------------------------------------------------------------------------
# Algebra (exact, no approximation): with ``s_w == 2``, ``p_w`` even and
# ``d_w`` even, every tap of output ``ow`` sits at
#
#     iw = 2*ow - p_w + t*d_w          (t = 0 .. k_w-1)
#
# which is *even* for every ``t``.  In the even-phase plane
# ``ph[q] = x[2q]`` (``q = iw/2``) the window therefore degenerates into a
# stride-``dd = d_w/2`` sliding window over ``q = ow - p_w/2 + t*dd``: one wide
# vector op per tap instead of ``bl`` narrow ops per tap (the general grid costs
# ``k_w * bl`` narrow visits per unit, this path costs ``k_w`` wide ones).
#
# Data flow per logical unit (no 3-D transpose anywhere -- only the 2-axis
# forms that the feasibility re-check measured as free):
#
#   GM span (ch, lspan) --GMDMA--> src --T.deinterleave(cn=2)--> even phase
#     --2D transpose--> (pp, ch) --vcast--> pf (float32, lane = channel)
#   out[ow, :] = max_t pf[pad + off + t*dd, :]
#
# ``pf`` is armed once with -inf and ``pramp`` (the parallel phase-index ramp)
# with a BIG sentinel, so the pad rows behave exactly like the general path's
# out-of-bounds guard: the unit loop only ever writes the *data* rows
# ``[pad, pad+pp)``.
#
# Index semantics preserved bit-exactly:
#   * first hit  = the smallest ``q`` whose value equals the window maximum
#                  (running ``vmin`` over the per-tap index candidates);
#   * NaN        = the largest ``q`` holding NaN (running ``vmax`` over a
#                  pre-masked ``nanpos`` plane), and the value is NaN;
#   * empty window (all taps out of range) -> ``(value, index) = (-inf, 0)``
#                  (only emitted for shapes where it is reachable).
#
# Frozen constraints of this path (all measured on 910B3 / d2783bd):
#   * ``T.deinterleave`` only with ``channel_nums=2`` (>2 SIGABRTs), and never
#     inside a runtime if/else (silently dropped);
#   * ``T.vselect`` takes **no** sliced operand (``operands' shape are
#     inconsistent``), while ``vcmp``/``vmax``/``vmin``/``vadd`` do -- so the
#     first-hit mask is folded in with a whole-buffer ``vselect`` + ``vadd``;
#   * ``T.vbrc`` never targets a runtime-offset region (device fault 507057),
#     so the pad/ramp arming happens once, outside the unit loop.
_PHASE_CH_STEPS = (128, 96, 64, 48, 32, 24, 16, 8)
_PHASE_BL_CANDIDATES = (64, 48, 40, 32, 24, 16)
# Store rows narrower than 32 B are amplified by the GM write granularity
# (measured 2.5x at 16 x f16 and 21x at 1 element, opt_log sections 3.3 / 4), so
# the phase path never picks bl below _PHASE_MIN_BL.
_PHASE_MIN_BL = 16
# ch == 32 is excluded: on 910B3/d2783bd the phase path returns
# ``idx = 2*q0 + 0`` for the trailing three output columns of the last block
# (304 of 2 095 104 elements), while ch = 8/16/24/40/48/56/64/80 are bit-exact
# on the same data and the same plan.  Root cause not identified (the armed
# ``pramp``/``pf`` planes dump correctly at ch == 32, and the empty-window
# clause does not fire); the planner therefore never selects it.
_PHASE_CH_EXCLUDED = (32,)
# Naive per-buffer UB sum budget.  The compiler budget itself is 196 608 B but
# it reuses non-overlapping buffers, so the *naive* sum runs ~1.6x the backend's
# own requirement.  Calibration (910B3 / tilelang d2783bd, ecg shape, 40 AIV):
#     naive sum  backend result                          outcome
#     217 KB      compiles                              110.41 us, exact
#     309 KB      compiles (ch64/bl40)                   91.18 us, exact
#     314 KB      compiles (ch96/bl32)                  185.58 us, exact
#     342 KB      ub overflow (1800192 bits, ch64/bl64)  -
#     425 KB      ub overflow (1896448 bits, ch128/bl32) -
#     500 KB      ub overflow (2551808 bits, ch128/bl40) -
# 300 KB accepts every config measured to compile (<= 296 KB) and rejects every
# config measured to overflow (>= 319 KB).
_PHASE_UB_SUM_BUDGET_BYTES = 300_000
# Stage 4 tuning knobs (0/unset = auto planner), e.g.
#   MAXPOOL1D_PHASE_CH=64 MAXPOOL1D_PHASE_BL=40 python ...
_PHASE_CH_OVERRIDE = int(os.environ.get("MAXPOOL1D_PHASE_CH", "0")) or None
_PHASE_BL_OVERRIDE = int(os.environ.get("MAXPOOL1D_PHASE_BL", "0")) or None


def _phase_bytes(ch, pp, rows, bl, elem_bytes):
    """Naive per-buffer UB footprint of the phase path (bytes).

    Keep in sync with the ``T.alloc_ub`` list of the phase ``prim_func``; the
    budget constant above is calibrated against this sum, so a new buffer must
    be added here too.
    """
    f32 = 4
    return (
        ch * 2 * pp * elem_bytes  # src (ch, 2*pp)
        + 2 * ch * pp * elem_bytes  # pe, po
        + pp * ch * elem_bytes  # pft
        + pp * ch * f32  # pff
        + 2 * rows * ch * f32  # pf, pramp
        + rows * ch  # isnum (bool)
        + 2 * rows * ch * f32  # nanpos, negr
        + 6 * bl * ch * f32  # acc, nan_idx, idx, cand, bigm, prb
        + 2 * bl * ch  # cf, eqb (bool)
        + 2 * bl * ch * f32  # bigb, zerob
        + pp * ch * f32  # rramp_i (int32)
        + bl * ch * elem_bytes  # acc16
        + bl * ch * f32  # idx32
        + ch * bl * elem_bytes  # st
        + ch * bl * f32  # idx_t
        + ch * bl * 8  # idx64
    )


def _phase_ch_candidates(c_in):
    """Channel blocks worth trying: exact divisors of C, plus one-block C.

    Partial channel blocks are excluded.  Measured on the ecg shape (C = 128,
    same plan otherwise): a partial tail costs much more than the extra lanes
    buy -- ch=96 (96 + 32) 185.58 us and ch=48 (48 + 48 + 32) 113.24 us against
    91.18 us for ch=64 (2 x 64), while ch=64 with a *smaller* bl is also worse
    (110.41 us at bl=32).  Exact divisors are the only clean choice.
    """
    cands = []
    for ch in _PHASE_CH_STEPS:
        if ch < _VECTOR_WIDTH or ch > c_in or ch in _PHASE_CH_EXCLUDED:
            continue
        if c_in % ch == 0:
            cands.append(ch)
    whole = _round_up(c_in, _VECTOR_WIDTH)
    if whole <= 128 and whole not in cands and whole not in _PHASE_CH_EXCLUDED:
        cands.append(whole)  # single channel block, <= 7 padding lanes
    return cands


def _phase_plan(n, c_in, l_in, out_l, k_w, s_w, p_w, d_w, ceil_mode, dtype, vector_cores=None):
    """Phase-path tiling plan, or ``None`` when the specialisation does not apply.

    Applicability is purely algebraic (``s_w == 2``, ``p_w`` even, ``d_w`` even,
    ``l_in`` even) plus "the phase plane fits the UB budget at
    ``bl >= _PHASE_MIN_BL``".  Every other shape keeps the general grid, so no
    correctness depends on this path being taken.

    Selection rule: maximise the vector lanes per wide op (``bl * ch``) subject
    to the UB budget; ties prefer the larger ``bl`` (wider store rows) and then
    the larger ``ch``.  Measured ecg-shape ranking (910B3, 40 AIV):
    (ch64, bl40) 91.18 us < (ch64, bl32) 110.41 < (ch128, bl16) 131.88 <
    (ch96, bl32) 185.58.
    """
    if s_w != 2 or p_w % 2 or d_w % 2 or l_in % 2 or k_w < 1:
        return None
    dd = d_w // 2
    ph = p_w // 2
    nq = l_in // 2  # number of even-phase positions
    tap_span = (k_w - 1) * dd
    pp_max = (nq // _VECTOR_WIDTH) * _VECTOR_WIDTH
    bl_max = min(min(out_l, _PHASE_BL_CANDIDATES[0]), pp_max - tap_span)
    if bl_max < _PHASE_MIN_BL:
        return None
    elem_bytes = _elem_bytes(dtype)
    empty_possible = any(
        all(not (0 <= ow - ph + t * dd <= nq - 1) for t in range(k_w)) for ow in range(out_l)
    )
    ch_cands = _phase_ch_candidates(c_in)
    if _PHASE_CH_OVERRIDE is not None:
        ch_cands = [c for c in ch_cands if c == _PHASE_CH_OVERRIDE]

    best = None
    for bl in _PHASE_BL_CANDIDATES:
        if bl > bl_max or bl < _PHASE_MIN_BL:
            continue
        if _PHASE_BL_OVERRIDE is not None and bl != _PHASE_BL_OVERRIDE:
            continue
        pp = _round_up(bl + tap_span, _VECTOR_WIDTH)
        if pp > pp_max:
            continue
        lspan = 2 * pp
        s0_max = 2 * (nq - pp)
        nob = (out_l + bl - 1) // bl
        off_min, off_max = 10**9, -(10**9)
        for oblk in range(nob):
            bq = oblk * bl - ph
            s0 = min(max(2 * bq, 0), s0_max)
            off = bq - s0 // 2
            off_min = min(off_min, off)
            off_max = max(off_max, off)
        pad = ph + 1
        if pad + off_min < 0:
            continue
        max_row = pad + off_max + (bl - 1) + tap_span
        rows = max(pad + pp + 2, max_row + 1)
        for ch in ch_cands:
            nb_bytes = _phase_bytes(ch, pp, rows, bl, elem_bytes)
            if nb_bytes > _PHASE_UB_SUM_BUDGET_BYTES:
                continue
            lanes = bl * ch
            if best is not None and lanes <= best["lanes"]:
                continue
            num_cblk = (c_in + ch - 1) // ch
            num_logical = n * nob * num_cblk
            if vector_cores is None:
                vector_cores = NPUUtils.get().get_aicore_num() * 2
            num_kernels = min(num_logical, vector_cores)
            best = {
                "ch": ch,
                "bl": bl,
                "lanes": lanes,
                "pp": pp,
                "lspan": lspan,
                "pad": pad,
                "rows": rows,
                "s0_max": s0_max,
                "dd": dd,
                "ph": ph,
                "nq": nq,
                "nob": nob,
                "num_cblk": num_cblk,
                "out_l": out_l,
                "elem_bytes": elem_bytes,
                "num_logical": num_logical,
                "num_kernels": num_kernels,
                "num_local": (num_logical + num_kernels - 1) // num_kernels,
                "empty_possible": empty_possible,
                "bytes": nb_bytes,
            }
    return best


# ---------------------------------------------------------------------------
# Whole-line halving-reduce path (Stage 4 round 3, `L_out == 1` specialisation)
# ---------------------------------------------------------------------------
# Algebra: with ``k_w == l_in``, ``s_w == 1``, ``p_w == 0`` and ``d_w == 1`` the
# output extent is exactly 1 and the single window covers the whole line
# ``0 .. l_in-1``.  The ``vmax`` over that window is a *tree reduce*, so a
# ``log2(l_in)``-level halving fold replaces the general grid's
# ``k_w * bl`` narrow visited lanes (2 passes x 3 ops per tap).
#
# Data flow per logical unit (one ``(n, channel-block)`` pair):
#
#   GM (ch, l_in) --T.copy--> src --T.transpose[1,0]--> tr (l_in, ch)
#     values : tr -> v1 .. v7   (vmax of two contiguous halves per level)
#     first  : vbrc(v7 -> mb); vcmp(tr, mb, eq); vselect(eq, ramp, BIG, cand);
#              cand -> o1 .. o7 (vmin fold)
#     lastNaN: vcmp(tr, tr, isnum, eq); vselect(isnum, ramp, -1, aux);
#              aux -> n1 .. n7 (vmax fold)
#     epilogue on (1, ch): has_nan = (n7 >= 0); value = has_nan ? NaN : v7;
#              index = has_nan ? n7 : o7; empty-window guard; int64 flat store
#
# Why the index is *not* folded with the value (measured, probe round 3): the
# halving tree pairs row ``i`` with ``i + l_in/2``, so the left subtree's index
# set ``{i, i + l_in/2}`` is not all-smaller than the right subtree's
# ``{i + l_in/4, i + 3*l_in/4}``; "keep the left child on a tie" therefore does
# not keep the first hit (measured: 15 of 35 ties wrong while the values stayed
# exact).  The ``eq`` + minimum-index fold has no tie logic at all and is exact,
# including ONES/``+-inf``/all-equal windows.
#
# Frozen constraints of this path (same toolchain as round 2, re-verified):
#   * every fold operand is a contiguous whole-row block of a buffer whose row
#     stride equals the buffer row stride -- a region whose stride differs is
#     lowered to UB->GM->UB and scalarised (measured 77x on this operator);
#   * ``T.vselect`` takes no sliced operand, so the mask/ramp/sentinel planes
#     are whole ``(l_in, CH)`` buffers (``vmin``/``vmax``/``vcmp`` do accept
#     same-shape slices);
#   * the index ramp is armed once per core from a **1-D** ``T.arange`` +
#     ``T.reshape`` + ``T.vbrc`` row broadcast (the 2-D ``T.arange`` form is
#     documented as a scalar pipeline write without PIPE_S->PIPE_V sync, and a
#     static *row* ramp fed into vcmp/vselect chains is a known crash pattern;
#     the int32 ramp never enters a compare here -- the index domain is f32,
#     exact for ``l_in <= 2**24``);
#   * the buffers shrink with the level (``l_in -> l_in/2 -> ... -> 1`` rows), so
#     there is no pair of resident ``(l_in, CH)`` output planes: the naive
#     per-buffer sum stays at the same order as the pre-existing paths (the
#     whole-plane dual-output form is what overflows UB at CH >= 16).
_FOLD_MAX_L_IN = 128  # 7 halving levels, see _FOLD_STAGES
_FOLD_MAX_INDEX = 2**24  # f32 index domain is exact below this
# Channel-block candidates: exact divisors of C first (no padding lanes), then
# the single rounded-up whole-C block (<= 128 lanes, <= 7 padding lanes).
_FOLD_CH_STEPS = (128, 96, 64, 48, 32, 24, 16, 8)
# Naive per-buffer UB sum budget.  Calibrated on the same backend/axis as
# `_PHASE_UB_SUM_BUDGET_BYTES` (the backend reuses non-overlapping buffers, so
# the naive sum runs ~1.6x its own requirement): the round-3 probes compile at
# a 205 KB naive sum, ch=64 of this path sums to ~243 KB (same order as the
# phase path's 296 KB that compiles) and ch=128 sums to ~487 KB, well past the
# 425 KB that was measured to overflow.
_FOLD_UB_SUM_BUDGET_BYTES = 260_000
# Stage 4 tuning knob (0/unset = planner): force a channel block for the
# one-shot CH neighbourhood scan; the UB budget check is then bypassed and a
# backend overflow falls back to the general grid.
_FOLD_CH_OVERRIDE = int(os.environ.get("MAXPOOL1D_FOLD_CH", "0")) or None


def _fold_stages(l_in):
    """``log2(l_in)`` for the power-of-two line lengths this path accepts."""
    return l_in.bit_length() - 1


def _fold_bytes(ch, l_in, elem_bytes):
    """Naive per-buffer UB footprint of the halving-reduce path (bytes).

    Keep in sync with the ``T.alloc_ub`` list of the fold ``prim_func``.  The
    three halving trees each shrink the row count, so they contribute
    ``(l_in - 1) * ch`` lanes per tree (not ``l_in * ch``).
    """
    f32 = 4
    tree = (l_in - 1) * ch
    return (
        ch * l_in * elem_bytes  # src (ch, l_in)
        + l_in * ch * elem_bytes  # tr (l_in, ch)
        + tree * elem_bytes  # v1 .. v7 (value tree)
        + l_in * ch * elem_bytes  # mb (broadcast of the folded maximum)
        + 2 * l_in * ch  # eq, isnum (bool planes)
        + 3 * l_in * ch * f32  # pramp, aux, cand
        + 2 * tree * f32  # o1..o7 (first hit), n1..n7 (last NaN)
        + 2 * l_in * f32  # ramp_f (l_in,) + its (l_in, 1) view
        + 40 * ch  # (1, CH) epilogue buffers
    )


def _fold_ch_candidates(c_in):
    """Channel blocks worth trying for the halving-reduce path."""
    cands = [ch for ch in _FOLD_CH_STEPS if ch >= _VECTOR_WIDTH and ch <= c_in and c_in % ch == 0]
    whole = _round_up(c_in, _VECTOR_WIDTH)
    if whole <= _FOLD_MAX_L_IN and whole not in cands:
        cands.append(whole)
    return cands


def _fold_plan(n, c_in, l_in, out_l, k_w, s_w, p_w, d_w, ceil_mode, dtype, vector_cores=None):
    """Halving-reduce plan, or ``None`` when the specialisation does not apply.

    Applicability is purely algebraic and checked in this order:
    ``out_l == 1`` and ``k_w == l_in`` and ``s_w == 1`` and ``p_w == 0`` and
    ``d_w == 1`` and ``l_in`` a power of two in ``[2, _FOLD_MAX_L_IN]``.
    Every other shape keeps the general grid (or the phase path), so no
    correctness depends on this path being taken and a *near miss* is
    functionally lossless.
    """
    if out_l != 1 or k_w != l_in or s_w != 1 or p_w != 0 or d_w != 1:
        return None
    if l_in < 2 or l_in > _FOLD_MAX_L_IN or l_in > _FOLD_MAX_INDEX:
        return None
    if l_in & (l_in - 1):
        return None

    nstages = _fold_stages(l_in)
    elem_bytes = _elem_bytes(dtype)
    ch_cands = _fold_ch_candidates(c_in)
    if _FOLD_CH_OVERRIDE is not None:
        ch_cands = [c for c in ch_cands if c == _FOLD_CH_OVERRIDE]
        if not ch_cands:
            return None

    best = None
    for ch in ch_cands:  # descending: widest ops, fewest units
        nb_bytes = _fold_bytes(ch, l_in, elem_bytes)
        if _FOLD_CH_OVERRIDE is None and nb_bytes > _FOLD_UB_SUM_BUDGET_BYTES:
            continue
        num_cblk = (c_in + ch - 1) // ch
        num_logical = n * num_cblk
        if vector_cores is None:
            vector_cores = NPUUtils.get().get_aicore_num() * 2
        num_kernels = min(num_logical, vector_cores)
        best = {
            "ch": ch,
            "l_in": l_in,
            "stages": nstages,
            "num_cblk": num_cblk,
            "num_logical": num_logical,
            "num_kernels": num_kernels,
            "num_local": (num_logical + num_kernels - 1) // num_kernels,
            "elem_bytes": elem_bytes,
            "bytes": nb_bytes,
        }
        break
    return best


def _plan(n, c_in, l_in, k_w, s_w, p_w, d_w, ceil_mode, dtype, vector_cores=None):
    """Derive the compile-time tiling plan for one shape/dtype combination.

    All shapes take the general grid (DESIGN.md section 0.6 R5: C4 / C4b are
    not adopted), so the logical unit is always ``(n, o_blk, c_block)`` and
    ``num_logical = N * ceil(L_out/BL) * ceil(C/CH)`` has a single definition
    source.  ``vector_cores`` defaults to the device query (pure-Vector op ->
    AI cores x 2).
    """
    out_l = pool_output_dim(l_in, k_w, s_w, p_w, ceil_mode, d_w)
    if out_l < 1:
        raise ValueError(f"output size must be greater than zero, got out_l={out_l}")
    elem_bytes = _elem_bytes(dtype)
    ch, bl = _choose_ch_bl(c_in, l_in, out_l, k_w, s_w, d_w, elem_bytes)
    if bl < 1:
        raise ValueError(
            f"no UB-feasible plan for (l_in={l_in}, kernel_w={k_w}, "
            f"stride_w={s_w}, dilation_w={d_w}, c_in={c_in}): the general-grid "
            f"span is at least {(k_w - 1) * d_w + 1} elements wide, which does "
            f"not fit the 192 KiB UB budget even at CH=8/BL=1"
        )
    lspan = _span(bl, k_w, s_w, d_w, l_in)
    num_cblk = (c_in + ch - 1) // ch
    num_oblk = (out_l + bl - 1) // bl
    num_logical = n * num_oblk * num_cblk
    if vector_cores is None:
        vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_kernels = min(num_logical, vector_cores)
    num_local = (num_logical + num_kernels - 1) // num_kernels

    return {
        "out_l": out_l,
        "ch": ch,
        "bl": bl,
        "lspan": lspan,
        "num_cblk": num_cblk,
        "num_oblk": num_oblk,
        "num_logical": num_logical,
        "num_kernels": num_kernels,
        "num_local": num_local,
        "general_bytes": _general_bytes(ch, lspan, bl, elem_bytes),
    }


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_max_pool1d_with_indices(
    x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):
    """PyTorch reference for max_pool1d(return_indices=True), per DESIGN.md 0.1.

    Directly uses the source-repo test benchmark
    (``F.max_pool1d(..., return_indices=True)``).  Exactly 3D NCL; independent
    of the NPU algorithm (no transpose / no C-vectorisation / no span-load /
    no index accumulator / no specialisation).  Runs on CPU and does not
    require an NPU device.  The fp32 opmath domain is loss-free for a pure
    selection, so the returned values are bit-exact and the int64 indices are
    compared exactly.
    """
    vals, idx = F.max_pool1d(
        x.detach().to(torch.float32),  # fp32 opmath domain for fp16/bf16 CPU ref
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    return vals.to(x.dtype), idx


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------
@kernel_factory_cache
def _max_pool1d_with_indices_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    """Build the windowed max pooling 1D kernel factory (values + indices).

    Returns a build callable taking a single ``block_m`` (the CUDA ``threads``
    parameter is removed, K9); ``build(block_m)`` compiles and returns ``main``;
    ``main(x)`` returns ``(out, indices)`` (``out_idx=[1, 2]``).
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"MaxPool1dIndicesFwdOp supports float16, bfloat16, and float32, got {dtype}"
        )
    if min(n, c_in, l_in) < 1:
        raise ValueError(f"input shape must be positive, got {(n, c_in, l_in)}")
    if kernel_w < 1:
        raise ValueError(f"kernel_size must be greater than zero, got {kernel_w}")
    if stride_w < 1:
        raise ValueError(f"stride must be greater than zero, got {stride_w}")
    if dilation_w < 1:
        raise ValueError(f"dilation must be greater than zero, got {dilation_w}")
    if pad_w < 0:
        raise ValueError(f"padding must be non-negative, got {pad_w}")
    if pad_w > kernel_w // 2:
        raise ValueError(
            "padding must be at most half of kernel_size, got "
            f"padding={pad_w} kernel_size={kernel_w}"
        )
    if l_in >= _INT32_INDEX_LIMIT:
        raise ValueError(f"int32 index carrier requires l_in < 2**31, got l_in = {l_in}")

    # The output extent is validated before the tiling plan is derived: a
    # non-positive extent makes the core-split arithmetic degenerate, so the Op
    # layer's "output size must be greater than zero" guard runs first.
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    if out_l < 1:
        raise ValueError(
            f"output size must be greater than zero, got {out_l} for input {l_in} kernel {kernel_w}"
        )

    plan = _plan(n, c_in, l_in, kernel_w, stride_w, pad_w, dilation_w, ceil_mode, dtype)
    assert plan["out_l"] == out_l

    # The phase-decimated path replaces the general grid whenever its algebraic
    # preconditions hold (s_w == 2, p_w / d_w even, l_in even) and a UB-feasible
    # wide tile exists; otherwise the general grid runs unchanged.
    phase = _phase_plan(
        n, c_in, l_in, out_l, kernel_w, stride_w, pad_w, dilation_w, ceil_mode, dtype
    )

    # The whole-line halving reduce replaces both other paths when the single
    # window *is* the whole line (k_w == l_in, s_w == 1, p_w == 0, d_w == 1,
    # l_in a power of two); it is algebraically disjoint from the phase path
    # (which needs s_w == 2).
    fold = _fold_plan(n, c_in, l_in, out_l, kernel_w, stride_w, pad_w, dilation_w, ceil_mode, dtype)

    ch = plan["ch"]
    bl = plan["bl"]
    lspan = plan["lspan"]
    num_cblk = plan["num_cblk"]
    num_oblk = plan["num_oblk"]
    num_logical = plan["num_logical"]
    num_kernels = plan["num_kernels"]
    num_local = plan["num_local"]
    is_f32 = dtype == "float32"

    def _build(block_m=None):
        if block_m is not None and block_m < 1:
            raise ValueError(f"block_m must be positive, got block_m={block_m}")

        # ---------------- general grid (every shape) ----------------
        @tilelang.jit(out_idx=[1, 2], target="npuir")
        def _func_general(ch_):
            @T.prim_func
            def main(
                x: T.Tensor((n, c_in, l_in), dtype),
                out: T.Tensor((n, c_in, out_l), dtype),
                indices: T.Tensor((n, c_in, out_l), "int64"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # Round-2 (branch r2_flatstore): `L_out == 1` stores the whole
                    # channel block *contiguously* in GM, so the value/index store
                    # uses a 1-D destination region fed by the (1, CH) upload
                    # buffer directly -- no (bl,CH)->(CH,bl) transpose on the
                    # store path.  Measured (probe_store_r2.py, 910B3): the
                    # transposed `(real_c, 1)` region store costs
                    # UB_to_GM = 96 KB/launch vs 3.8 KB (real write) for the 1-D
                    # form, i.e. mte3 66.9 us -> 0.9 us.
                    flat_store = out_l == 1
                    # Vector accumulators on the channel axis (vector axis = C).
                    acc = T.alloc_ub((bl, ch), "float32")
                    infbuf = T.alloc_ub((bl, ch), "float32")
                    nanbuf = T.alloc_ub((bl, ch), "float32")
                    out32 = T.alloc_ub((bl, ch), "float32")
                    cfinal = T.alloc_ub((bl, ch), "bool")
                    max_idx = T.alloc_ub((bl, ch), "int32")
                    nan_idx = T.alloc_ub((bl, ch), "int32")
                    idx32 = T.alloc_ub((bl, ch), "int32")
                    nan_f32 = T.alloc_ub((bl, ch), "float32")
                    idx_f32 = T.alloc_ub((bl, ch), "float32")
                    zeros_i = T.alloc_ub((bl, ch), "int32")
                    bigv = T.alloc_ub((bl, ch), "int32")
                    idx_t = T.alloc_ub((ch, bl), "int32")
                    idx64 = T.alloc_ub((ch, bl), "int64")
                    acc16 = T.alloc_ub((bl, ch), dtype)
                    acc_t = T.alloc_ub((ch, bl), dtype)
                    # 1-D store staging (only used when flat_store, i.e. bl == 1,
                    # so the (bl, ch) shape costs ch * 8 B).
                    idx64r = T.alloc_ub((bl, ch), "int64")
                    # Contiguous span staging: NCL span -> transposed (C innermost).
                    row_src = T.alloc_ub((ch, lspan), dtype)
                    row_dst = T.alloc_ub((lspan, ch), dtype)
                    row_f32 = T.alloc_ub((lspan, ch), "float32")
                    # Per-element (1, CH) operands.  They must be *whole* buffers:
                    # a T.vbrc into a region with a runtime offset is silently
                    # dropped by this toolchain (and the int32 form then reads
                    # uninitialised UB -> vector-core fault).
                    cand = T.alloc_ub((1, ch), "int32")
                    nanc = T.alloc_ub((1, ch), "int32")
                    fvec = T.alloc_ub((1, ch), "int32")
                    neg1 = T.alloc_ub((1, ch), "int32")
                    big1 = T.alloc_ub((1, ch), "int32")
                    eqb = T.alloc_ub((1, ch), "bool")
                    eq2b = T.alloc_ub((1, ch), "bool")

                    # max identity is -inf (not 0); the NaN-free flag accumulator
                    # starts at 1 (0 is the absorbing value); the first-hit
                    # accumulator starts at the BIG = L_in sentinel (also the
                    # empty-window sentinel); the last-NaN accumulator starts at
                    # -1 (no NaN seen).
                    neg_inf = -T.infinity("float32")
                    pos_inf = T.infinity("float32")
                    big_i = T.int32(l_in)
                    neg1_i = T.int32(-1)
                    zero_i = T.int32(0)
                    zero_f32 = T.float32(0.0)
                    big_f32 = T.float32(l_in)

                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // (num_oblk * num_cblk)
                            rem = uid % (num_oblk * num_cblk)
                            oblk = rem // num_cblk
                            cb = rem % num_cblk
                            real_c = T.min(ch, c_in - cb * ch)
                            ow0 = oblk * bl
                            # Clamped contiguous span (DESIGN.md 1.6.1 C1): the
                            # start is a runtime value but always in bounds
                            # (0 <= s0 and s0 + LSPAN <= L_in) and the extent is
                            # the static LSPAN, so no dynamic-offset subview is
                            # formed.  j = iw - s0 stays in [0, LSPAN) whenever
                            # the in-bounds guard holds.
                            base = ow0 * stride_w - pad_w
                            s0 = T.max(0, T.min(base, l_in - lspan))

                            T.vbrc(neg_inf, acc)
                            T.vbrc(big_i, bigv)
                            T.vbrc(big_i, big1)
                            T.vbrc(neg1_i, neg1)
                            T.vbrc(zero_i, zeros_i)
                            T.vbrc(big_i, max_idx)
                            T.vbrc(neg1_i, nan_idx)
                            T.vbrc(pos_inf, infbuf)
                            T.vsub(infbuf, infbuf, nanbuf)  # NaN = inf - inf

                            T.copy(
                                x[nb, cb * ch : cb * ch + real_c, s0 : s0 + lspan],
                                row_src[0:real_c, 0:lspan],
                            )
                            T.transpose(row_src, row_dst, permutation=[1, 0])
                            if is_f32:
                                T.copy(row_dst, row_f32)
                            else:
                                T.vcast(row_dst, row_f32, round_mode="rint")

                            # Pass 1: value + NaN flag + last-NaN index.
                            for kw in T.serial(kernel_w):
                                for owl in T.serial(bl):
                                    iw = base + owl * stride_w + kw * dilation_w
                                    j = iw - s0
                                    if iw >= 0 and iw < l_in:
                                        T.vmax(
                                            acc[owl : owl + 1, 0:ch],
                                            row_f32[j : j + 1, 0:ch],
                                            acc[owl : owl + 1, 0:ch],
                                        )
                                        T.vbrc(T.int32(iw), fvec)
                                        # last-NaN: non-NaN -> -1, NaN -> iw, then
                                        # vmax accumulates the largest NaN iw.
                                        T.vcmp(
                                            row_f32[j : j + 1, 0:ch],
                                            row_f32[j : j + 1, 0:ch],
                                            eq2b,
                                            "eq",
                                        )
                                        T.vselect(eq2b, neg1, fvec, nanc)
                                        T.vmax(
                                            nan_idx[owl : owl + 1, 0:ch],
                                            nanc,
                                            nan_idx[owl : owl + 1, 0:ch],
                                        )

                            # Pass 2: first-hit.  The value accumulator is final,
                            # so eq(row, acc) marks every lane whose value equals
                            # the window maximum; vmin then keeps the smallest iw.
                            for kw in T.serial(kernel_w):
                                for owl in T.serial(bl):
                                    iw = base + owl * stride_w + kw * dilation_w
                                    j = iw - s0
                                    if iw >= 0 and iw < l_in:
                                        T.vbrc(T.int32(iw), fvec)
                                        T.vcmp(
                                            row_f32[j : j + 1, 0:ch],
                                            acc[owl : owl + 1, 0:ch],
                                            eqb,
                                            "eq",
                                        )
                                        T.vselect(eqb, fvec, big1, cand)
                                        T.vmin(
                                            max_idx[owl : owl + 1, 0:ch],
                                            cand,
                                            max_idx[owl : owl + 1, 0:ch],
                                        )

                            # Both outputs share one NaN decision:
                            # cfinal = (nan_idx >= 0) <=> the window holds a NaN.
                            # The comparison is executed in the fp32 domain: an
                            # int32 ``vcmp`` is lowered to a per-lane scalar loop
                            # on this toolchain (~16 ns/lane, measured) and costs
                            # more than the whole window scan, while an fp32 vcmp
                            # is a vector op.  The index range (|iw| < 2**24) makes
                            # the int32 -> fp32 cast exact.
                            T.vcast(nan_idx, nan_f32, round_mode="rint")
                            T.vcmp(nan_f32, zero_f32, cfinal, "ge")
                            T.vselect(cfinal, nanbuf, acc, out32)
                            T.vselect(cfinal, nan_idx, max_idx, idx32)
                            # Empty-window clause (DESIGN.md 0.1, alignment with
                            # the source state machine): idx32 == BIG <=> no NaN
                            # and no in-bounds lane -> idx = 0.  It reuses the
                            # cfinal bool buffer, so it must run *after* out32 is
                            # selected.  fp32-domain comparison for the same reason
                            # as above.
                            T.vcast(idx32, idx_f32, round_mode="rint")
                            T.vcmp(idx_f32, big_f32, cfinal, "eq")
                            T.vselect(cfinal, zeros_i, idx32, idx32)

                            if is_f32:
                                T.copy(out32, acc16)
                            else:
                                T.vcast(out32, acc16, round_mode="rint")
                            bl_real = T.min(bl, out_l - ow0)
                            if flat_store:
                                # out / indices are (n, c, 1): the channel block
                                # [cb*CH, cb*CH+real_c) is one contiguous GM run,
                                # so a 1-D region write is a single burst.
                                T.vcast(idx32, idx64r, round_mode="rint")
                                T.copy(
                                    acc16[0, 0:real_c],
                                    out[nb, cb * ch : cb * ch + real_c, 0],
                                )
                                T.copy(
                                    idx64r[0, 0:real_c],
                                    indices[nb, cb * ch : cb * ch + real_c, 0],
                                )
                            else:
                                T.transpose(acc16, acc_t, permutation=[1, 0])
                                T.transpose(idx32, idx_t, permutation=[1, 0])
                                T.vcast(idx_t, idx64, round_mode="rint")  # i32 -> i64
                                T.copy(
                                    acc_t[0:real_c, 0:bl_real],
                                    out[nb, cb * ch : cb * ch + real_c, ow0 : ow0 + bl_real],
                                )
                                T.copy(
                                    idx64[0:real_c, 0:bl_real],
                                    indices[nb, cb * ch : cb * ch + real_c, ow0 : ow0 + bl_real],
                                )

            return main

        if fold is not None:

            @tilelang.jit(out_idx=[1, 2], target="npuir")
            def _func_fold():
                # ---- whole-line halving-reduce path (see the header block) ----
                f_ch = fold["ch"]
                f_nk = fold["num_kernels"]
                f_nlog = fold["num_logical"]
                f_nloc = fold["num_local"]
                f_ncb = fold["num_cblk"]
                f_l = l_in
                h1 = max(1, f_l >> 1)
                h2 = max(1, f_l >> 2)
                h3 = max(1, f_l >> 3)
                h4 = max(1, f_l >> 4)
                h5 = max(1, f_l >> 5)
                h6 = max(1, f_l >> 6)

                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, l_in), dtype),
                    out: T.Tensor((n, c_in, out_l), dtype),
                    indices: T.Tensor((n, c_in, out_l), "int64"),
                ):
                    with T.Kernel(f_nk, is_npu=True) as (cid, _):
                        src = T.alloc_ub((f_ch, f_l), dtype)
                        tr = T.alloc_ub((f_l, f_ch), dtype)
                        # value tree (shrinks: f_l/2 -> 1 rows)
                        v1 = T.alloc_ub((h1, f_ch), dtype)
                        v2 = T.alloc_ub((h2, f_ch), dtype)
                        v3 = T.alloc_ub((h3, f_ch), dtype)
                        v4 = T.alloc_ub((h4, f_ch), dtype)
                        v5 = T.alloc_ub((h5, f_ch), dtype)
                        v6 = T.alloc_ub((h6, f_ch), dtype)
                        v7 = T.alloc_ub((1, f_ch), dtype)
                        mb = T.alloc_ub((f_l, f_ch), dtype)
                        eq = T.alloc_ub((f_l, f_ch), "bool")
                        isnum = T.alloc_ub((f_l, f_ch), "bool")
                        pramp = T.alloc_ub((f_l, f_ch), "float32")
                        aux = T.alloc_ub((f_l, f_ch), "float32")
                        cand = T.alloc_ub((f_l, f_ch), "float32")
                        # first-hit tree (min fold)
                        o1 = T.alloc_ub((h1, f_ch), "float32")
                        o2 = T.alloc_ub((h2, f_ch), "float32")
                        o3 = T.alloc_ub((h3, f_ch), "float32")
                        o4 = T.alloc_ub((h4, f_ch), "float32")
                        o5 = T.alloc_ub((h5, f_ch), "float32")
                        o6 = T.alloc_ub((h6, f_ch), "float32")
                        o7 = T.alloc_ub((1, f_ch), "float32")
                        # last-NaN tree (max fold)
                        n1 = T.alloc_ub((h1, f_ch), "float32")
                        n2 = T.alloc_ub((h2, f_ch), "float32")
                        n3 = T.alloc_ub((h3, f_ch), "float32")
                        n4 = T.alloc_ub((h4, f_ch), "float32")
                        n5 = T.alloc_ub((h5, f_ch), "float32")
                        n6 = T.alloc_ub((h6, f_ch), "float32")
                        n7 = T.alloc_ub((1, f_ch), "float32")
                        # 1-D ramp source + its (f_l, 1) view for the row broadcast
                        ramp_f = T.alloc_ub((f_l,), "float32")
                        ramp_c = T.alloc_ub((f_l, 1), "float32")
                        # (1, CH) epilogue buffers
                        v7f = T.alloc_ub((1, f_ch), "float32")
                        nanb = T.alloc_ub((1, f_ch), "float32")
                        out32b = T.alloc_ub((1, f_ch), "float32")
                        out16 = T.alloc_ub((1, f_ch), dtype)
                        idxv = T.alloc_ub((1, f_ch), "float32")
                        idx32 = T.alloc_ub((1, f_ch), "int32")
                        i64b = T.alloc_ub((1, f_ch), "int64")
                        zero1 = T.alloc_ub((1, f_ch), "float32")
                        neginf1 = T.alloc_ub((1, f_ch), "float32")
                        cf = T.alloc_ub((1, f_ch), "bool")
                        cf2 = T.alloc_ub((1, f_ch), "bool")

                        neg_inf = -T.infinity("float32")
                        pos_inf = T.infinity("float32")
                        zero_f = T.float32(0.0)
                        neg1_f = T.float32(-1.0)
                        # Index sentinel: strictly greater than every row index.
                        # It is also the empty-window sentinel of the general grid.
                        big_f = T.float32(f_l)

                        # Armed once per core, outside the unit loop: the index
                        # ramp (1-D arange -> reshape -> row broadcast; the int32
                        # ramp never enters a compare) and the constant planes.
                        T.arange(ramp_f, [1], 0)
                        T.reshape(ramp_f, ramp_c)
                        T.vbrc(ramp_c, pramp)
                        T.vbrc(big_f, aux)
                        T.vbrc(pos_inf, nanb)
                        T.vsub(nanb, nanb, nanb)  # NaN = inf - inf
                        T.vbrc(zero_f, zero1)
                        T.vbrc(neg_inf, neginf1)

                        for t in T.serial(f_nloc):
                            uid = t * f_nk + cid
                            if uid < f_nlog:
                                nb = uid // f_ncb
                                cb = uid % f_ncb
                                real_c = T.min(f_ch, c_in - cb * f_ch)

                                T.copy(
                                    x[nb, cb * f_ch : cb * f_ch + real_c, 0:f_l],
                                    src[0:real_c, 0:f_l],
                                )
                                T.transpose(src, tr, permutation=[1, 0])

                                # ---- values: log2(f_l) halving stages ----
                                # Every fold operand is a contiguous whole-row
                                # block (row stride == buffer row stride), and
                                # the buffers shrink per level, so no residency
                                # of two full-size output planes is needed.
                                T.vmax(tr[0:h1, 0:f_ch], tr[h1 : 2 * h1, 0:f_ch], v1)
                                if f_l >= 4:
                                    T.vmax(v1[0:h2, 0:f_ch], v1[h2 : 2 * h2, 0:f_ch], v2)
                                else:
                                    T.copy(v1, v2)
                                if f_l >= 8:
                                    T.vmax(v2[0:h3, 0:f_ch], v2[h3 : 2 * h3, 0:f_ch], v3)
                                else:
                                    T.copy(v2, v3)
                                if f_l >= 16:
                                    T.vmax(v3[0:h4, 0:f_ch], v3[h4 : 2 * h4, 0:f_ch], v4)
                                else:
                                    T.copy(v3, v4)
                                if f_l >= 32:
                                    T.vmax(v4[0:h5, 0:f_ch], v4[h5 : 2 * h5, 0:f_ch], v5)
                                else:
                                    T.copy(v4, v5)
                                if f_l >= 64:
                                    T.vmax(v5[0:h6, 0:f_ch], v5[h6 : 2 * h6, 0:f_ch], v6)
                                else:
                                    T.copy(v5, v6)
                                if f_l >= 128:
                                    T.vmax(v6[0:1, 0:f_ch], v6[1:2, 0:f_ch], v7)
                                else:
                                    T.copy(v6, v7)

                                # ---- first hit = min index among the equal
                                # lanes (no tie logic: the value tree's sibling
                                # order is not the index order) ----
                                T.vbrc(v7, mb)
                                T.vcmp(tr, mb, eq, "eq")
                                T.vselect(eq, pramp, aux, cand)
                                T.vmin(cand[0:h1, 0:f_ch], cand[h1 : 2 * h1, 0:f_ch], o1)
                                if f_l >= 4:
                                    T.vmin(o1[0:h2, 0:f_ch], o1[h2 : 2 * h2, 0:f_ch], o2)
                                else:
                                    T.copy(o1, o2)
                                if f_l >= 8:
                                    T.vmin(o2[0:h3, 0:f_ch], o2[h3 : 2 * h3, 0:f_ch], o3)
                                else:
                                    T.copy(o2, o3)
                                if f_l >= 16:
                                    T.vmin(o3[0:h4, 0:f_ch], o3[h4 : 2 * h4, 0:f_ch], o4)
                                else:
                                    T.copy(o3, o4)
                                if f_l >= 32:
                                    T.vmin(o4[0:h5, 0:f_ch], o4[h5 : 2 * h5, 0:f_ch], o5)
                                else:
                                    T.copy(o4, o5)
                                if f_l >= 64:
                                    T.vmin(o5[0:h6, 0:f_ch], o5[h6 : 2 * h6, 0:f_ch], o6)
                                else:
                                    T.copy(o5, o6)
                                if f_l >= 128:
                                    T.vmin(o6[0:1, 0:f_ch], o6[1:2, 0:f_ch], o7)
                                else:
                                    T.copy(o6, o7)

                                # ---- last NaN = max index among the NaN lanes
                                # (-1 when the window holds no NaN).  ``cand`` is
                                # reused as the masked plane after the first-hit
                                # fold consumed it, and ``aux`` re-armed to -1, so
                                # the two folds share the full-size planes instead
                                # of adding a fourth one. ----
                                T.vcmp(tr, tr, isnum, "eq")
                                T.vbrc(neg1_f, aux)
                                # isnum (x == x) -> -1, NaN -> its row index
                                T.vselect(isnum, aux, pramp, cand)
                                T.vmax(cand[0:h1, 0:f_ch], cand[h1 : 2 * h1, 0:f_ch], n1)
                                if f_l >= 4:
                                    T.vmax(n1[0:h2, 0:f_ch], n1[h2 : 2 * h2, 0:f_ch], n2)
                                else:
                                    T.copy(n1, n2)
                                if f_l >= 8:
                                    T.vmax(n2[0:h3, 0:f_ch], n2[h3 : 2 * h3, 0:f_ch], n3)
                                else:
                                    T.copy(n2, n3)
                                if f_l >= 16:
                                    T.vmax(n3[0:h4, 0:f_ch], n3[h4 : 2 * h4, 0:f_ch], n4)
                                else:
                                    T.copy(n3, n4)
                                if f_l >= 32:
                                    T.vmax(n4[0:h5, 0:f_ch], n4[h5 : 2 * h5, 0:f_ch], n5)
                                else:
                                    T.copy(n4, n5)
                                if f_l >= 64:
                                    T.vmax(n5[0:h6, 0:f_ch], n5[h6 : 2 * h6, 0:f_ch], n6)
                                else:
                                    T.copy(n5, n6)
                                if f_l >= 128:
                                    T.vmax(n6[0:1, 0:f_ch], n6[1:2, 0:f_ch], n7)
                                else:
                                    T.copy(n6, n7)

                                # ---- epilogue: one NaN decision per lane ----
                                # has_nan <=> n7 >= 0 (the fp32 index domain keeps
                                # the int32 vcmp per-lane scalarisation out).
                                T.vcmp(n7, zero_f, cf, "ge")
                                if is_f32:
                                    T.copy(v7, v7f)
                                else:
                                    T.vcast(v7, v7f, round_mode="rint")
                                T.vselect(cf, nanb, v7f, out32b)
                                T.vselect(cf, n7, o7, idxv)
                                # Empty-window clause, identical to the general
                                # grid (idx == BIG <=> no in-bounds lane).  It is
                                # unreachable here (k_w == l_in and p_w == 0 make
                                # every window full) but kept so both paths share
                                # one contract.
                                T.vcmp(idxv, big_f, cf2, "eq")
                                T.vselect(cf2, zero1, idxv, idxv)
                                T.vselect(cf2, neginf1, out32b, out32b)

                                if is_f32:
                                    T.copy(out32b, out16)
                                else:
                                    T.vcast(out32b, out16, round_mode="rint")
                                T.vcast(idxv, idx32, round_mode="rint")
                                T.vcast(idx32, i64b, round_mode="rint")
                                # out_l == 1: the channel block is one contiguous
                                # GM run, so the store is a single 1-D burst.
                                T.copy(
                                    out16[0, 0:real_c],
                                    out[nb, cb * f_ch : cb * f_ch + real_c, 0],
                                )
                                T.copy(
                                    i64b[0, 0:real_c],
                                    indices[nb, cb * f_ch : cb * f_ch + real_c, 0],
                                )

                return main

            try:
                return _func_fold()
            except Exception as exc:  # noqa: BLE001
                # Build-time feasibility guard (the UB accounting is a proxy), not
                # a performance selector: an overflow falls back to the phase /
                # general path, which are bit-exact for the same shape.
                print(
                    "[_max_pool1d_with_indices_kernel] fold path rejected by the "
                    f"backend for this shape, using the general grid: "
                    f"{str(exc).splitlines()[0][:160]}"
                )

        if phase is not None:

            @tilelang.jit(out_idx=[1, 2], target="npuir")
            def _func_phase():
                # ---- phase-decimated wide-tile path (see the header block) ----
                p_ch = phase["ch"]
                p_bl = phase["bl"]
                p_pp = phase["pp"]
                p_lspan = phase["lspan"]
                p_pad = phase["pad"]
                p_rows = phase["rows"]
                p_dd = phase["dd"]
                p_ph = phase["ph"]
                p_nq = phase["nq"]
                p_nob = phase["nob"]
                p_ncb = phase["num_cblk"]
                p_nlog = phase["num_logical"]
                p_nk = phase["num_kernels"]
                p_nloc = phase["num_local"]
                p_empty = phase["empty_possible"]

                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, l_in), dtype),
                    out: T.Tensor((n, c_in, out_l), dtype),
                    indices: T.Tensor((n, c_in, out_l), "int64"),
                ):
                    with T.Kernel(p_nk, is_npu=True) as (cid, _):
                        src = T.alloc_ub((p_ch, p_lspan), dtype)
                        pe = T.alloc_ub((p_ch, p_pp), dtype)
                        po = T.alloc_ub((p_ch, p_pp), dtype)
                        pft = T.alloc_ub((p_pp, p_ch), dtype)
                        pff = T.alloc_ub((p_pp, p_ch), "float32")
                        pf = T.alloc_ub((p_rows, p_ch), "float32")
                        pramp = T.alloc_ub((p_rows, p_ch), "float32")
                        isnum = T.alloc_ub((p_rows, p_ch), "bool")
                        nanpos = T.alloc_ub((p_rows, p_ch), "float32")
                        negr = T.alloc_ub((p_rows, p_ch), "float32")
                        acc = T.alloc_ub((p_bl, p_ch), "float32")
                        bigm = T.alloc_ub((p_bl, p_ch), "float32")
                        prb = T.alloc_ub((p_bl, p_ch), "float32")
                        bigb = T.alloc_ub((p_bl, p_ch), "float32")
                        zerob = T.alloc_ub((p_bl, p_ch), "float32")
                        nan_idx = T.alloc_ub((p_bl, p_ch), "float32")
                        idx = T.alloc_ub((p_bl, p_ch), "float32")
                        cand = T.alloc_ub((p_bl, p_ch), "float32")
                        nanb = T.alloc_ub((p_bl, p_ch), "float32")
                        cf = T.alloc_ub((p_bl, p_ch), "bool")
                        eqb = T.alloc_ub((p_bl, p_ch), "bool")
                        rramp_i = T.alloc_ub((p_pp, p_ch), "int32")
                        acc16 = T.alloc_ub((p_bl, p_ch), dtype)
                        idx32 = T.alloc_ub((p_bl, p_ch), "int32")
                        st = T.alloc_ub((p_ch, p_bl), dtype)
                        idx_t = T.alloc_ub((p_ch, p_bl), "int32")
                        idx64 = T.alloc_ub((p_ch, p_bl), "int64")

                        neg_inf = -T.infinity("float32")
                        pos_inf = T.infinity("float32")
                        zero_f = T.float32(0.0)
                        neg1_f = T.float32(-1.0)
                        big_f = T.float32(1.0e9)
                        # Armed once, outside the unit loop: the pad rows of
                        # ``pf`` keep -inf and their ``pramp`` keeps the BIG
                        # sentinel, because the loop only rewrites the data rows
                        # ``[pad, pad+pp)``.  This is what makes out-of-range
                        # window taps behave like the general path's
                        # ``0 <= iw < l_in`` guard without a per-lane branch.
                        T.vbrc(neg_inf, pf)
                        T.vbrc(big_f, pramp)
                        T.arange(rramp_i, [2, 0], 0)
                        T.vcast(rramp_i, pff, round_mode="rint")
                        T.copy(pff, pramp[p_pad : p_pad + p_pp, 0:p_ch])
                        T.vbrc(neg1_f, negr)
                        T.vbrc(zero_f, zerob)
                        T.vbrc(big_f, bigb)
                        T.vbrc(pos_inf, nanb)
                        T.vsub(nanb, nanb, nanb)  # NaN = inf - inf

                        for t in T.serial(p_nloc):
                            uid = t * p_nk + cid
                            if uid < p_nlog:
                                nb = uid // (p_nob * p_ncb)
                                rem = uid % (p_nob * p_ncb)
                                oblk = rem // p_ncb
                                cb = rem % p_ncb
                                real_c = T.min(p_ch, c_in - cb * p_ch)
                                ow0 = oblk * p_bl
                                bq = ow0 - p_ph  # first phase position needed
                                q0 = T.max(0, T.min(bq, p_nq - p_pp))
                                r0 = p_pad + bq - q0
                                base2 = T.float32(2 * q0)

                                T.copy(
                                    x[
                                        nb,
                                        cb * p_ch : cb * p_ch + real_c,
                                        2 * q0 : 2 * q0 + p_lspan,
                                    ],
                                    src[0:real_c, 0:p_lspan],
                                )
                                # Stride-2 decimation is exactly one 2-phase
                                # deinterleave; the odd phase is unused.
                                T.deinterleave(src, pe, po, channel_nums=2)
                                T.transpose(pe, pft, permutation=[1, 0])
                                if is_f32:
                                    # T.vcast with target == source dtype
                                    # miscompiles on this toolchain (see the
                                    # header), so fp32 stays a plain copy.
                                    T.copy(pft, pff)
                                else:
                                    T.vcast(pft, pff, round_mode="rint")
                                T.copy(pff, pf[p_pad : p_pad + p_pp, 0:p_ch])
                                # one NaN mask + one masked index plane per unit
                                T.vcmp(pf, pf, isnum, "eq")
                                T.vselect(isnum, negr, pramp, nanpos)

                                T.vbrc(neg_inf, acc)
                                T.vbrc(big_f, idx)
                                T.vbrc(neg1_f, nan_idx)
                                for tt in range(kernel_w):
                                    rr = r0 + tt * p_dd
                                    T.vmax(pf[rr : rr + p_bl, 0:p_ch], acc, acc)
                                    T.vmax(nan_idx, nanpos[rr : rr + p_bl, 0:p_ch], nan_idx)
                                # first hit = smallest phase position whose
                                # value equals the final window maximum.  vsel
                                # takes no sliced operand, so the miss mask is
                                # materialised whole-buffer and folded in with
                                # vadd on a staged copy of the ramp slice.
                                for tt in range(kernel_w):
                                    rr = r0 + tt * p_dd
                                    T.vcmp(pf[rr : rr + p_bl, 0:p_ch], acc, eqb, "eq")
                                    T.vselect(eqb, zerob, bigb, bigm)
                                    T.copy(pramp[rr : rr + p_bl, 0:p_ch], prb)
                                    T.vadd(prb, bigm, cand)
                                    T.vmin(idx, cand, idx)
                                # NaN wins the value and the (last-NaN) index.
                                T.vcmp(nan_idx, zero_f, cf, "ge")
                                T.vselect(cf, nanb, acc, acc)
                                T.vselect(cf, nan_idx, idx, idx)
                                if p_empty:
                                    # empty window -> (value, index) = (-inf, 0)
                                    T.vbrc(zero_f, cand)
                                    T.vcmp(idx, big_f, cf, "eq")
                                    T.vselect(cf, cand, idx, idx)
                                T.vbrc(base2, cand)
                                T.vadd(idx, cand, idx)
                                bl_real = T.min(p_bl, out_l - ow0)
                                if is_f32:
                                    T.copy(acc, acc16)  # same-dtype vcast trap
                                else:
                                    T.vcast(acc, acc16, round_mode="rint")
                                T.vcast(idx, idx32, round_mode="rint")
                                T.transpose(acc16, st, permutation=[1, 0])
                                T.transpose(idx32, idx_t, permutation=[1, 0])
                                T.vcast(idx_t, idx64, round_mode="rint")
                                T.copy(
                                    st[0:real_c, 0:bl_real],
                                    out[nb, cb * p_ch : cb * p_ch + real_c, ow0 : ow0 + bl_real],
                                )
                                T.copy(
                                    idx64[0:real_c, 0:bl_real],
                                    indices[
                                        nb, cb * p_ch : cb * p_ch + real_c, ow0 : ow0 + bl_real
                                    ],
                                )

                return main

            try:
                return _func_phase()
            except Exception as exc:  # noqa: BLE001
                # Build-time feasibility guard, not a performance selector: the
                # UB accounting above is a proxy, so if the backend still
                # rejects this tile (measured failure mode: ``ub overflow ...
                # requires N bits``) the shape falls back to the general grid
                # instead of failing to build.  Both paths are bit-exact.
                print(
                    "[_max_pool1d_with_indices_kernel] phase path rejected by "
                    f"the backend for this shape, using the general grid: "
                    f"{str(exc).splitlines()[0][:160]}"
                )

        return _func_general(ch)

    return _build


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_KERNEL_CACHE = {}

# Worst-case statistics over the whole run (final summary).
_STATS = {"value_max_diff": {}, "idx_mismatch": 0, "idx_total": 0}


def _as_int(value):
    """Accept a scalar or a 1-element tuple (source uses int scalars)."""
    if isinstance(value, (tuple, list)):
        assert len(value) == 1, f"expected a 1-element pooling spec, got {value}"
        return int(value[0])
    return int(value)


def _normalise(kernel_size, stride, padding, dilation):
    k_w = _as_int(kernel_size)
    s_w = k_w if stride is None else _as_int(stride)
    p_w = _as_int(padding)
    d_w = _as_int(dilation)
    return k_w, s_w, p_w, d_w


def _get_kernel(n, c_in, l_in, k_w, s_w, p_w, d_w, ceil_mode, dtype_str, block_m=None):
    """Cache compiled kernels (one per shape/dtype, mirroring the wrapper's
    per-shape kernel cache)."""
    key = (n, c_in, l_in, k_w, s_w, p_w, d_w, bool(ceil_mode), dtype_str, block_m)
    if key not in _KERNEL_CACHE:
        factory = _max_pool1d_with_indices_kernel(
            n,
            c_in,
            l_in,
            k_w,
            s_w,
            p_w,
            d_w,
            bool(ceil_mode),
            dtype_str,
        )
        _KERNEL_CACHE[key] = factory(block_m)
    return _KERNEL_CACHE[key]


def _value_stats(out, ref, equal_nan):
    """Return (max_abs_diff, violation_count) between two value tensors."""
    eq = out == ref
    if equal_nan:
        eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(eq, torch.zeros((), dtype=torch.float32), (out.float() - ref.float()).abs())
    return diff.max().item(), int((~eq).sum().item())


def _record_value_diff(dtype_str, max_diff):
    if max_diff == max_diff:  # NaN guard
        best = _STATS["value_max_diff"].get(dtype_str, -1.0)
        if max_diff > best:
            _STATS["value_max_diff"][dtype_str] = max_diff


def _run_case(
    n,
    c_in,
    l_in,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    dtype_str,
    tag="L0",
    x=None,
    equal_nan=False,
):
    """Run one kernel case and compare both outputs against the golden.

    ``x`` may be pre-built as a CPU tensor (special-value cases); by default
    randn data is generated on CPU, cast and moved to NPU (device-independent
    data generation).  Raises AssertionError on a value or index violation.
    """
    torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
    k_w, s_w, p_w, d_w = _normalise(kernel_size, stride, padding, dilation)
    out_l = pool_output_dim(l_in, k_w, s_w, p_w, ceil_mode, d_w)

    if x is None:
        x = torch.randn(n, c_in, l_in, dtype=torch.float32).to(torch_dtype)
    else:
        assert x.dtype == torch_dtype, f"input dtype {x.dtype} != {torch_dtype}"
        assert tuple(x.shape) == (n, c_in, l_in), f"input shape {x.shape}"

    kernel = _get_kernel(n, c_in, l_in, k_w, s_w, p_w, d_w, bool(ceil_mode), dtype_str)
    out, idx = kernel(x.npu())
    out = out.cpu()
    idx = idx.cpu()

    ref, ref_idx = golden_max_pool1d_with_indices(x, k_w, s_w, p_w, d_w, bool(ceil_mode))

    assert idx.dtype == torch.int64, f"indices dtype {idx.dtype} != int64"
    assert tuple(out.shape) == (n, c_in, out_l), (
        f"{tag}: value shape {tuple(out.shape)} != {(n, c_in, out_l)}"
    )
    assert tuple(idx.shape) == tuple(out.shape), (
        f"{tag}: indices shape {tuple(idx.shape)} != out shape {tuple(out.shape)}"
    )

    max_diff, violations = _value_stats(out, ref, equal_nan)
    _record_value_diff(dtype_str, max_diff)
    idx_mm = int((idx != ref_idx).sum().item())
    idx_total = int(idx.numel())
    _STATS["idx_mismatch"] += idx_mm
    _STATS["idx_total"] += idx_total

    try:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
        assert idx_mm == 0, f"index mismatch in {idx_mm}/{idx_total} elements"
    except AssertionError:
        print(
            f"[{tag}] FAIL: shape=({n},{c_in},{l_in}) out={out_l} "
            f"k={kernel_size} s={stride} p={padding} d={dilation} "
            f"ceil={ceil_mode} dtype={dtype_str} max_diff={max_diff:.3e} "
            f"value_violations={violations}/{out.numel()} "
            f"idx_mismatch={idx_mm}/{idx_total}"
        )
        raise
    print(
        f"[{tag}] PASS: shape=({n},{c_in},{l_in}) out={out_l} "
        f"dtype={dtype_str} max_diff={max_diff:.3e} idx_exact"
    )
    return max_diff, idx_mm


def _try_case(failures, case_id, **kwargs):
    """Run ``_run_case``, collecting (not re-raising) per-case failures.

    AssertionError is a precision (or shape) failure; any other exception is
    recorded as a runtime error (both block the L0/L1 gate).
    """
    try:
        _run_case(**kwargs)
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


def _expect_value_error(failures, case_id, kwargs):
    """Assert that the factory rejects the given arguments with ValueError."""
    try:
        _max_pool1d_with_indices_kernel(**kwargs)
    except ValueError as exc:
        print(f"[{case_id}] PASS: rejected ({exc})")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} WRONG-ERROR: {exc!r}")
    else:
        failures.append((case_id, AssertionError("invalid arguments accepted")))
        print(f"[collected] {case_id} FAILED: invalid arguments were accepted")


# ---------------------------------------------------------------------------
# Semantics probes (NaN / +-inf / ties / first-hit order / empty window)
# ---------------------------------------------------------------------------
# Probe shape: (1, 16, 13), k=3, s=1, p=1 -> out_l = 13.  Window of output ow
# covers iw in {ow-1, ow, ow+1} (in-bounds lanes only).
_SEM_SHAPE = (1, 16, 13)
_SEM_K, _SEM_S, _SEM_P = 3, 1, 1


def _sem_probe_x(dtype):
    """Special-value input for the fp16/bf16 semantic gates (same collection).

    Per-channel expectations (window of output ow covers {ow-1, ow, ow+1}):
      0: NaN at iw=0            -> (NaN, 0)          at ow=0
      1: NaN at iw=2 and iw=4   -> (NaN, 4)          at ow=3   (last NaN)
      2: {+inf at 0, -inf at 1} -> (+inf, 0)         at ow=1   (no NaN)
      3: all -inf               -> (-inf, 4)         at ow=5   (first valid)
      4: 9 at iw=2 and iw=4     -> (9, 2)            at ow=3   (tie -> first)
      5: 9 at iw=3 only         -> (9, 3)            at ow=2   (max not first)
      6: +inf at 5, NaN at 6    -> (NaN, 6)          at ow=5   (NaN beats +inf)
      7: NaN at iw=9            -> (-3, 1) at ow=2;  (NaN, 9) at ow=9
    Rows outside the asserted windows keep low finite values so the full-tensor
    golden comparison stays meaningful.
    """
    n, c_in, l_in = _SEM_SHAPE
    x = torch.full((n, c_in, l_in), -3.0, dtype=torch.float32)
    x[0, 0, 0] = float("nan")
    x[0, 1, 2] = float("nan")
    x[0, 1, 4] = float("nan")
    x[0, 2, 0] = float("inf")
    x[0, 2, 1] = float("-inf")
    x[0, 3, :] = float("-inf")
    x[0, 4, 2] = 9.0
    x[0, 4, 4] = 9.0
    x[0, 5, 3] = 9.0
    x[0, 6, 5] = float("inf")
    x[0, 6, 6] = float("nan")
    x[0, 7, 9] = float("nan")
    return x.to(dtype)


def _sem_expected():
    """Hand-derived (value, is_nan, index) per (channel, output position)."""
    return {
        (0, 0): (None, True, 0),
        (1, 3): (None, True, 4),
        (2, 1): (float("inf"), False, 0),
        (3, 5): (float("-inf"), False, 4),
        (4, 3): (9.0, False, 2),
        (5, 2): (9.0, False, 3),
        (6, 5): (None, True, 6),
        (7, 2): (-3.0, False, 1),
        (7, 9): (None, True, 9),
    }


def _run_semantic_gate(failures, dtype_str, case_id):
    """Semantics gate (DESIGN.md 8.3 L0-8 / L0-9 / L0-10) with mechanism
    evidence.

    Asserts the hand-derived per-position expectations (value, NaN-ness and the
    exact index) *and* a full-tensor comparison against the golden.  The
    hand-derived assertions are what make the first-hit / last-NaN mechanisms
    observable (a value-only check can pass through ``vmax`` NaN propagation).
    """
    try:
        torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
        x = _sem_probe_x(torch_dtype)
        n, c_in, l_in = _SEM_SHAPE
        pool_output_dim(l_in, _SEM_K, _SEM_S, _SEM_P, False, 1)
        kernel = _get_kernel(n, c_in, l_in, _SEM_K, _SEM_S, _SEM_P, 1, False, dtype_str)
        out, idx = kernel(x.npu())
        out = out.cpu()
        idx = idx.cpu()
        ref, ref_idx = golden_max_pool1d_with_indices(x, _SEM_K, _SEM_S, _SEM_P, 1, False)

        for (ch, ow), (want_val, want_nan, want_idx) in _sem_expected().items():
            got = out[0, ch, ow]
            got_nan = bool(torch.isnan(got))
            assert got_nan == want_nan, (
                f"ch={ch} ow={ow} NaN-ness {got_nan} != {want_nan} (got {got})"
            )
            got_idx = int(idx[0, ch, ow].item())
            assert got_idx == want_idx, (
                f"ch={ch} ow={ow} index {got_idx} != {want_idx} (first-hit / last-NaN mechanism)"
            )
            if not want_nan:
                assert float(got) == float(want_val), (
                    f"ch={ch} ow={ow} value {float(got)} != {want_val}"
                )

        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, equal_nan=True)
        max_diff, _ = _value_stats(out, ref, True)
        _record_value_diff(dtype_str, max_diff)
        idx_mm = int((idx != ref_idx).sum().item())
        _STATS["idx_mismatch"] += idx_mm
        _STATS["idx_total"] += int(idx.numel())
        assert idx_mm == 0, f"index mismatch in {idx_mm}/{idx.numel()} elements"
        print(
            f"[{case_id}] PASS: last-NaN index; tie -> first hit; max-not-first "
            f"-> correct index; {{+inf,-inf}} -> +inf; all -inf -> first "
            f"element; full-tensor match vs golden (equal_nan=True, idx exact) "
            f"dtype={dtype_str}"
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


def _run_first_hit_order(failures):
    """First-hit record-order check with hand-derived indices (mechanism evidence).

    ``x = arange(13)`` with k=3, s=1, p=1: the window of output ``ow`` covers
    ``{ow-1, ow, ow+1}`` and its maximum is the *last* visited element, so the
    first-hit index must be ``min(ow+1, 12)``.  A fused single-pass form (first
    hit against a running maximum) would report the first visited element
    instead.  ``x = [9, 1, 9, 1]`` with k=3, s=1, p=0 pins the tie direction
    (output 0 records iw 0, output 1 records iw 2).
    """
    case_id = "L0-8/first-hit-order"
    try:
        x = torch.arange(13, dtype=torch.float32).reshape(1, 1, 13)
        kernel = _get_kernel(1, 1, 13, 3, 1, 1, 1, False, "float32")
        out, idx = kernel(x.npu())
        out = out.cpu()
        idx = idx.cpu()
        ref, ref_idx = golden_max_pool1d_with_indices(x, 3, 1, 1, 1, False)
        for ow in range(13):
            want = min(ow + 1, 12)
            got_i = int(idx[0, 0, ow].item())
            assert got_i == want, (
                f"arange first-hit at ow={ow}: index {got_i} != {want} "
                f"(a running-maximum form would give the first visited {max(ow - 1, 0)})"
            )
            assert float(out[0, 0, ow]) == float(want), (
                f"arange value at ow={ow}: {float(out[0, 0, ow])} != {want}"
            )
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
        assert torch.equal(idx, ref_idx), "arange index != golden index"

        xt = torch.tensor([9.0, 1.0, 9.0, 1.0]).reshape(1, 1, 4)
        kt = _get_kernel(1, 1, 4, 3, 1, 0, 1, False, "float32")
        outt, idxt = kt(xt.npu())
        outt = outt.cpu()
        idxt = idxt.cpu()
        assert int(idxt[0, 0, 0]) == 0, "tie output 0 must keep the first hit (0)"
        assert int(idxt[0, 0, 1]) == 2, "tie output 1 must keep the first hit (2)"
        print(
            "[L0-8] PASS: first-hit order (arange max is the last visited "
            "element; ties keep the first hit)"
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


def _fold_path_taken(n, c_in, l_in, k_w, s_w, p_w, d_w, ceil_mode, dtype_str):
    """True when the planner selects the halving-reduce path for this shape."""
    out_l = pool_output_dim(l_in, k_w, s_w, p_w, ceil_mode, d_w)
    return _fold_plan(n, c_in, l_in, out_l, k_w, s_w, p_w, d_w, ceil_mode, dtype_str) is not None


def _run_fold_semantics(failures):
    """NaN / last-NaN / tie / all-(-inf) semantics on the halving-reduce path.

    The window is the whole line (``k == l_in == 8``), so each channel is one
    window and the expectations are hand-derived:
      0: NaN at 2 and 5      -> (NaN, 5)    last NaN
      1: NaN at 0            -> (NaN, 0)
      2: all -inf            -> (-inf, 0)   no NaN; first lane
      3: +inf at 1 and 6     -> (+inf, 1)   tie -> first hit
      4: 9 at 3 and 7        -> (9, 3)      tie -> first hit
      5: 9 at 0, NaN at 7    -> (NaN, 7)    NaN wins the *value*
      6: all NaN             -> (NaN, 7)    last NaN
      7: constant -3         -> (-3, 0)
    """
    case_id = "L0-17e/fold-nan-tie-semantics"
    try:
        x = torch.full((1, 8, 8), -3.0, dtype=torch.float32)
        x[0, 0, 2] = float("nan")
        x[0, 0, 5] = float("nan")
        x[0, 1, 0] = float("nan")
        x[0, 2, :] = float("-inf")
        x[0, 3, 1] = float("inf")
        x[0, 3, 6] = float("inf")
        x[0, 4, 3] = 9.0
        x[0, 4, 7] = 9.0
        x[0, 5, 0] = 9.0
        x[0, 5, 7] = float("nan")
        x[0, 6, :] = float("nan")
        expected = {
            0: (None, True, 5),
            1: (None, True, 0),
            2: (float("-inf"), False, 0),
            3: (float("inf"), False, 1),
            4: (9.0, False, 3),
            5: (None, True, 7),
            6: (None, True, 7),
            7: (-3.0, False, 0),
        }
        kernel = _get_kernel(1, 8, 8, 8, 1, 0, 1, False, "float16")
        out, idx = kernel(x.to(torch.float16).npu())
        out = out.cpu()
        idx = idx.cpu()
        assert _fold_path_taken(1, 8, 8, 8, 1, 0, 1, False, "float16"), (
            "the fold semantics case fell back to another path"
        )
        for ch, (want_val, want_nan, want_idx) in expected.items():
            got = out[0, ch, 0]
            assert bool(torch.isnan(got)) == want_nan, (
                f"ch={ch} NaN-ness {bool(torch.isnan(got))} != {want_nan} (got {got})"
            )
            got_i = int(idx[0, ch, 0].item())
            assert got_i == want_idx, (
                f"ch={ch} index {got_i} != {want_idx} (first-hit / last-NaN mechanism)"
            )
            if not want_nan:
                assert float(got) == float(want_val), f"ch={ch} value {float(got)} != {want_val}"
        ref, ref_idx = golden_max_pool1d_with_indices(x.to(torch.float16), 8, 1, 0, 1, False)
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0, equal_nan=True)
        assert torch.equal(idx, ref_idx), "fold-path index != golden index"
        print(
            "[L0-17e] PASS: fold path last-NaN / tie->first / all -inf / "
            "NaN-beats-max / all-NaN (values + int64 indices exact)"
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


def _run_fold_path_selection(failures):
    """Path-selection gate: fold path taken exactly on its algebraic domain.

    Each entry is (tag, shape, k, s, p, d, dtype, expected).  The negative cases
    are the *near misses* documented in ``_fold_plan``; they must fall back (to
    the phase path or the general grid) and stay bit-exact.
    """
    case_id = "L0-17f/fold-path-selection"
    try:
        entries = [
            ("textcnn", (64, 256, 128), 128, 1, 0, 1, "float16", True),
            ("l64", (1, 16, 64), 64, 1, 0, 1, "float16", True),
            ("l2", (1, 8, 2), 2, 1, 0, 1, "float16", True),
            ("cblk-tail", (2, 20, 32), 32, 1, 0, 1, "bfloat16", True),
            ("c48-whole-c-fp32", (1, 48, 128), 128, 1, 0, 1, "float32", True),
            ("c-not-multiple", (1, 20, 128), 128, 1, 0, 1, "float16", True),
            # near misses -> must NOT take the fold path
            ("k<l", (64, 256, 128), 64, 1, 0, 1, "float16", False),
            ("l-not-pow2", (1, 8, 96), 96, 1, 0, 1, "float16", False),
            ("p>0", (1, 8, 32), 32, 1, 1, 1, "float16", False),
            ("d>1", (1, 8, 32), 16, 1, 0, 2, "float16", False),
            ("s>1", (1, 8, 30), 30, 2, 0, 1, "float16", False),
            ("l=1", (1, 8, 1), 1, 1, 0, 1, "float16", False),
            ("l>128", (1, 8, 256), 256, 1, 0, 1, "float16", False),
            ("ceil-d2-out2", (1, 8, 32), 16, 1, 0, 2, "float16", False),
            ("ceil-fold", (2, 16, 64), 64, 1, 0, 1, "float16", True),
        ]
        for tag, shape, k, s, p, d, dt, want in entries:
            n, c, l = shape
            got = _fold_path_taken(n, c, l, k, s, p, d, False, dt)
            assert got == want, (
                f"{tag}: fold path taken={got}, expected {want} "
                f"(shape={shape} k={k} s={s} p={p} d={d} {dt})"
            )
            print(f"[L0-17f] PASS: {tag}: fold={got} (expected {want})")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


def _run_empty_window(failures):
    """Empty-window contract (DESIGN.md 0.1 / 8.3 L0-15, negative control group).

    ``d_W >= 2 and p_W > 0`` can make a window hold no in-bounds lane.  The
    design aligns with the *source state machine*: ``(value, index) = (-inf, 0)``.
    torch returns an out-of-range index for that regime (a CPU-kernel
    loop-artifact), so torch is explicitly **not** used as an oracle here -- the
    assertion is the contract itself, plus a documentation check that torch
    indeed goes out of range.
    """
    case_id = "L0-15/empty-window-contract"
    try:
        configs = [
            # (tag, l_in, kernel, stride, padding, dilation, ceil)
            ("EW-min", 1, 2, 1, 1, 2, False),
            ("EW-ceil", 1, 2, 3, 1, 3, True),
        ]
        for tag, l_in, k, s, p, d, ceil in configs:
            n, c_in = 1, 2
            out_l = pool_output_dim(l_in, k, s, p, ceil, d)
            assert out_l == 1, f"{tag}: expected out_l=1, got {out_l}"
            x = torch.tensor([[5.0], [7.0]]).to(torch.float16)
            kernel = _get_kernel(n, c_in, l_in, k, s, p, d, ceil, "float16")
            out, idx = kernel(x.npu())
            out = out.cpu()
            idx = idx.cpu()
            assert bool((out == float("-inf")).all()), (
                f"{tag}: empty-window value {out.flatten().tolist()} != -inf"
            )
            assert bool((idx == 0).all()), (
                f"{tag}: empty-window index {idx.flatten().tolist()} != 0 (source-aligned contract)"
            )
            # torch is not an oracle here: it returns an out-of-range index.
            _, t_idx = golden_max_pool1d_with_indices(x, k, s, p, d, ceil)
            assert int(t_idx.min()) >= l_in or int(t_idx.max()) >= l_in, (
                f"{tag}: torch index {t_idx.flatten().tolist()} unexpectedly "
                f"in range; the empty-window premise no longer holds"
            )
            print(
                f"[L0-15] PASS: {tag} (l_in={l_in} k={k} s={s} p={p} d={d} "
                f"ceil={ceil}) -> (val, idx) = (-inf, 0); "
                f"torch idx {t_idx.flatten().tolist()} is out of range"
            )
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


# ---------------------------------------------------------------------------
# L0 (gate suite -- DESIGN.md section 8.3) -- blocking
# ---------------------------------------------------------------------------
def run_L0():
    """L0 gate: manifest workloads across all three dtypes, the frozen
    general-grid path, the max-specific NaN / last-NaN / +-inf / tie
    semantics, the empty-window contract and the validation guards."""
    failures = []

    cases = [
        # n, c, l, kernel, stride, padding, dilation, ceil, dtype, tag
        # -- L0-1..L0-2: manifest smoke (3 dtypes) --
        (1, 8, 32, 3, 2, 1, 1, False, "float32", "L0-1/smoke-fp32"),
        (2, 8, 64, 3, 2, 1, 1, False, "float16", "L0-2a/smoke-fp16"),
        (2, 8, 64, 3, 2, 1, 1, False, "bfloat16", "L0-2b/smoke-bf16"),
        # -- L0-3..L0-5: manifest workloads --
        (32, 80, 4096, 3, 3, 0, 1, False, "float16", "L0-3/sincnet"),
        (64, 256, 128, 128, 1, 0, 1, False, "float16", "L0-4a/textcnn-global-fp16"),
        (64, 256, 128, 128, 1, 0, 1, False, "float32", "L0-4b/textcnn-global-fp32"),
        (16, 128, 2048, 5, 2, 2, 2, True, "bfloat16", "L0-5/ecg-dilated"),
        # -- L0-6..L0-7: stride=None + dilation, non-overlapping + ceil --
        (1, 4, 63, 3, None, 1, 2, False, "float16", "L0-6/stride-none-d2"),
        (1, 4, 97, 5, 3, 2, 1, True, "float16", "L0-7a/k5s3p2-ceil-fp16"),
        (1, 4, 97, 5, 3, 2, 1, True, "bfloat16", "L0-7b/k5s3p2-ceil-bf16"),
        # -- L0-12: channel-block tails (C=96 -> CH=64 + real_c=32; C=4 -> CH=8) --
        (3, 96, 64, 3, 2, 1, 1, False, "float16", "L0-12a/cblk-tail-96"),
        (1, 4, 14, 3, 3, 1, 1, False, "float16", "L0-12b/cblk-tail-4"),
        # -- L0-16: phase-decimated path (s_w == 2, p_w / d_w even, l_in even).
        # The three manifest shapes above only exercise the general grid, so the
        # specialisation needs its own coverage: k / d / ceil variants, all three
        # dtypes, a partial channel block (C=20 -> CH=24, real_c=20) and the
        # near-miss shapes that must fall back to the general grid.
        (2, 16, 64, 3, 2, 0, 2, False, "float16", "L0-16a/phase-k3d2"),
        (2, 24, 128, 5, 2, 2, 2, True, "bfloat16", "L0-16b/phase-k5d2-ceil"),
        (2, 8, 96, 3, 2, 0, 4, False, "float32", "L0-16c/phase-k3d4-fp32"),
        (1, 16, 64, 1, 2, 0, 2, False, "float16", "L0-16d/phase-k1"),
        (2, 20, 80, 2, 2, 0, 2, False, "bfloat16", "L0-16e/phase-cblk-tail-20"),
        (2, 16, 128, 5, 2, 2, 2, True, "float16", "L0-16f/phase-k5d2-ceil-fp16"),
        # near misses: odd l_in, odd p_w, odd d_w -> general grid
        (2, 16, 65, 3, 2, 0, 2, False, "float16", "L0-16g/fallback-odd-l"),
        (2, 16, 64, 3, 2, 1, 2, False, "float16", "L0-16h/fallback-odd-p"),
        (2, 16, 64, 3, 2, 0, 3, False, "float16", "L0-16i/fallback-odd-d"),
        # -- L0-17: whole-line halving-reduce path (L_out == 1, k == l_in,
        # l_in a power of two).  The manifest shapes only hit it at
        # l_in == 128, so the specialisation needs its own coverage: both
        # widths, all three dtypes, a partial channel block (C=20) and the
        # near-miss shapes that must fall back.
        (1, 16, 64, 64, 1, 0, 1, False, "float16", "L0-17a/fold-l64-fp16"),
        (2, 8, 32, 32, 1, 0, 1, False, "bfloat16", "L0-17b/fold-l32-bf16"),
        (1, 4, 16, 16, 1, 0, 1, False, "float32", "L0-17c/fold-l16-fp32"),
        (2, 20, 32, 32, 1, 0, 1, False, "bfloat16", "L0-17d/fold-cblk-tail"),
        (1, 8, 96, 96, 1, 0, 1, False, "float16", "L0-17g/fallback-l-not-pow2"),
        (1, 8, 32, 16, 1, 0, 2, False, "float16", "L0-17h/fallback-d2"),
        (1, 8, 1, 1, 1, 0, 1, False, "float16", "L0-17i/fallback-l1"),
        # ceil_mode does not change L_out when k == l_in / s == 1 / p == 0, so
        # the fold path must stay *eligible* and exact under ceil_mode.
        (2, 16, 64, 64, 1, 0, 1, True, "float16", "L0-17j/fold-ceil-mode"),
        # ... while a ceil_mode shape whose L_out != 1 must fall back.
        (1, 8, 32, 16, 1, 0, 2, True, "float16", "L0-17k/fallback-ceil-d2"),
    ]
    for (
        n,
        c_in,
        l_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype_str,
        case_id,
    ) in cases:
        _try_case(
            failures,
            case_id,
            n=n,
            c_in=c_in,
            l_in=l_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            dtype_str=dtype_str,
            tag=case_id.split("/")[0],
        )

    # L0-8/9/10: NaN (last-NaN), tie (first hit) and +-inf semantics with
    # hand-derived mechanism evidence, on fp16 and bf16.
    _run_semantic_gate(failures, "float16", "L0-8-9-10/fp16-semantics")
    _run_semantic_gate(failures, "bfloat16", "L0-8-9-10/bf16-semantics")
    _run_first_hit_order(failures)

    # L0-16j: NaN / last-NaN / tie semantics on the *phase* path (s_w == 2).
    # The semantic gate above uses k3/s1/p1, which stays on the general grid, so
    # this case is what keeps the phase path's NaN plumbing covered.
    case_id = "L0-16j/phase-nan-tie"
    try:
        x = torch.full((1, 16, 96), -3.0, dtype=torch.float32)
        x[0, 0, 4] = float("nan")  # even phase (q=2)
        x[0, 0, 60] = float("nan")  # even phase (q=30), later -> last-NaN wins
        x[0, 1, 2] = float("nan")  # even phase, single NaN
        x[0, 2, 8] = float("inf")  # tie on +inf at iw 8 / 10 -> first hit
        x[0, 2, 10] = float("inf")
        x[0, 3, 0] = float("-inf")
        x[0, 3, 1] = float("-inf")  # all -inf window keeps the first in-range
        _run_case(
            n=1,
            c_in=16,
            l_in=96,
            kernel_size=3,
            stride=2,
            padding=0,
            dilation=2,
            ceil_mode=False,
            dtype_str="float16",
            tag="L0-16",
            x=x.to(torch.float16),
            equal_nan=True,
        )
        print("[L0-16j] PASS: phase-path NaN/last-NaN/+inf-tie/-inf semantics")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-17e/f: the halving-reduce path's own mechanisms and its exact domain.
    _run_fold_semantics(failures)
    _run_fold_path_selection(failures)

    # L0-11: non-contiguous input -- the Op layer normalises with
    # ``.contiguous()`` before the kernel sees the tensor (DESIGN.md 0.1).
    case_id = "L0-11/non-contiguous"
    try:
        base = torch.randn(2, 8, 32, dtype=torch.float32)
        xn = base[:, :, ::2]  # (2, 8, 16), non-contiguous view
        assert tuple(xn.shape) == (2, 8, 16)
        assert not xn.is_contiguous(), "test input unexpectedly contiguous"
        xc = xn.contiguous()  # Op-layer normalisation (metadata copy)
        _run_case(
            n=2,
            c_in=8,
            l_in=16,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            ceil_mode=False,
            dtype_str="float16",
            tag="L0-11",
            x=xc.to(torch.float16),
        )
        print("[L0-11] PASS: non-contiguous view -> .contiguous() -> kernel")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-13: front padding makes the window start at iw < 0 (covered by the
    # smoke cases); additionally check the index contract directly.
    case_id = "L0-13/front-padding-index-range"
    try:
        x = torch.randn(1, 8, 32, dtype=torch.float32).to(torch.float16)
        kernel = _get_kernel(1, 8, 32, 3, 2, 1, 1, False, "float16")
        _, idx = kernel(x.npu())
        idx = idx.cpu()
        assert int(idx.min()) >= 0 and int(idx.max()) < 32, (
            f"index out of [0, L_in): [{int(idx.min())}, {int(idx.max())}]"
        )
        print("[L0-13] PASS: front-padding indices stay in [0, L_in)")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-14: parameter / input validation (Op-layer guards mirrored by the
    # factory so the rejection paths are exercised here too).
    _expect_value_error(
        failures,
        "L0-14/dtype-fp64",
        dict(
            n=1,
            c_in=8,
            l_in=32,
            kernel_w=3,
            stride_w=2,
            pad_w=1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float64",
        ),
    )
    _expect_value_error(
        failures,
        "L0-14/zero-batch",
        dict(
            n=0,
            c_in=8,
            l_in=32,
            kernel_w=3,
            stride_w=2,
            pad_w=1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-14/zero-len",
        dict(
            n=1,
            c_in=8,
            l_in=0,
            kernel_w=3,
            stride_w=2,
            pad_w=1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-14/zero-stride",
        dict(
            n=1,
            c_in=8,
            l_in=32,
            kernel_w=3,
            stride_w=0,
            pad_w=1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-14/zero-dilation",
        dict(
            n=1,
            c_in=8,
            l_in=32,
            kernel_w=3,
            stride_w=2,
            pad_w=1,
            dilation_w=0,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-14/negative-padding",
        dict(
            n=1,
            c_in=8,
            l_in=32,
            kernel_w=3,
            stride_w=2,
            pad_w=-1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-14/padding-too-large",
        dict(
            n=1,
            c_in=8,
            l_in=32,
            kernel_w=3,
            stride_w=2,
            pad_w=2,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-14/zero-kernel",
        dict(
            n=1,
            c_in=8,
            l_in=32,
            kernel_w=0,
            stride_w=2,
            pad_w=1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-14/non-positive-output-size",
        dict(
            n=1,
            c_in=1,
            l_in=2,
            kernel_w=5,
            stride_w=1,
            pad_w=0,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-14/index-width-guard",
        dict(
            n=1,
            c_in=1,
            l_in=2**31,
            kernel_w=1,
            stride_w=1,
            pad_w=0,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )

    # L0-15: empty-window contract (negative control group).
    _run_empty_window(failures)

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print(
            "[L0] ALL PASS (manifest shapes x dtypes + general path + "
            "NaN/last-NaN/tie/+-inf + empty window + guards)"
        )
    return failures


# ---------------------------------------------------------------------------
# L1 (full functional coverage) -- blocking
# ---------------------------------------------------------------------------
def run_L1():
    """L1: dtype x shape x parameter matrix, including the remaining manifest
    dtype/size combinations, disjoint windows, dilation/ceil combinations and
    the capacity-pressed tiling."""
    failures = []

    cases = [
        # n, c, l, kernel, stride, padding, dilation, ceil, dtype, tag
        (32, 80, 4096, 3, 3, 0, 1, False, "bfloat16", "L1/bf16/sincnet"),
        (64, 256, 128, 128, 1, 0, 1, False, "bfloat16", "L1/bf16/textcnn"),
        (16, 128, 2048, 5, 2, 2, 2, True, "float32", "L1/fp32/ecg-dilated"),
        (1, 1, 7, 2, 2, 1, 1, False, "float16", "L1/fp16/c-eq-1"),
        (2, 3, 17, 3, 2, 1, 2, False, "bfloat16", "L1/bf16/asym-k-s-p-d"),
        (1, 7, 9, 3, 2, 1, 1, True, "float32", "L1/fp32/odd-ceil"),
        (1, 8, 30, 3, None, 1, 1, False, "float16", "L1/fp16/default-stride"),
        (2, 16, 14, 2, 2, 0, 1, False, "float16", "L1/fp16/disjoint"),
        (1, 6, 15, 5, 3, 2, 1, False, "float16", "L1/fp16/pad-only"),
        (1, 8, 19, 3, 2, 1, 3, True, "bfloat16", "L1/bf16/dilation-ceil"),
        (1, 8, 23, 4, 2, 2, 2, False, "float32", "L1/fp32/pad-dilation"),
        # Capacity-pressed general tiling (BL < BL_MAX) and a long L_in.
        (1, 8, 2048, 1024, 1, 0, 1, False, "float16", "L1/fp16/long-kernel"),
        (1, 8, 16384, 3, 2, 1, 1, False, "float16", "L1/fp16/long-l"),
        # Output-block tail: sincnet L_out=1365 is not a multiple of BL=16.
        (8, 80, 4096, 3, 3, 0, 1, False, "float16", "L1/fp16/sincnet-n8"),
    ]
    for (
        n,
        c_in,
        l_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype_str,
        case_id,
    ) in cases:
        _try_case(
            failures,
            case_id,
            n=n,
            c_in=c_in,
            l_in=l_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            dtype_str=dtype_str,
            tag="L1",
        )

    # NaN semantics on the general path in bf16 and fp32.
    for dtype_str in ("bfloat16", "float32"):
        case_id = f"L1/{dtype_str}/nan-window"
        try:
            tdt = _TOLERANCE[dtype_str][0]
            x = torch.full((1, 8, 13), -3.0, dtype=torch.float32)
            x[0, 0, 0] = float("nan")
            x[0, 1, 2] = float("nan")
            x[0, 1, 4] = float("nan")
            x[0, 2, :] = float("-inf")
            _run_case(
                n=1,
                c_in=8,
                l_in=13,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                ceil_mode=False,
                dtype_str=dtype_str,
                tag="L1",
                x=x.to(tdt),
                equal_nan=True,
            )
        except AssertionError as exc:
            failures.append((case_id, exc))
            print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
        except Exception as exc:  # noqa: BLE001
            failures.append((case_id, exc))
            print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # The wrapper dispatches with block_m=256; the value is otherwise ignored
    # (K9 kept the knob).
    case_id = "L1/block_m-256-dispatch"
    try:
        kernel = _max_pool1d_with_indices_kernel(1, 8, 32, 3, 2, 1, 1, False, "float16")(256)
        xx = torch.randn(1, 8, 32, dtype=torch.float16)
        out, idx = kernel(xx.npu())
        out = out.cpu()
        idx = idx.cpu()
        ref, ref_idx = golden_max_pool1d_with_indices(xx, 3, 2, 1, 1, False)
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
        assert torch.equal(idx, ref_idx), "index mismatch on the block_m=256 dispatch"
        print("[L1] PASS: block_m=256 dispatch (wrapper contract)")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # block_m=0 is rejected.
    try:
        _max_pool1d_with_indices_kernel(1, 8, 32, 3, 2, 1, 1, False, "float16")(0)
    except ValueError as exc:
        print(f"[L1] PASS: block_m guard rejects 0 ({exc})")
    else:
        failures.append(("L1/block_m-guard", AssertionError("block_m=0 accepted")))
        print("[collected] L1/block_m-guard FAILED: block_m=0 accepted")

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


# ---------------------------------------------------------------------------
# L2 (extreme sizes -- warn only, non-blocking)
# ---------------------------------------------------------------------------
def run_L2():
    """L2: extreme sizes (warn only)."""
    cases = [
        (1, 5, 7, 2, 2, 0, 1, False, "float16"),
        (1, 1, 1, 1, 1, 0, 1, False, "bfloat16"),
        (3, 1024, 3, 1, 1, 0, 1, False, "float16"),
        (1, 16, 512, 3, 2, 1, 1, False, "float32"),
        (1, 8, 1024, 2, 2, 0, 1, False, "float16"),
        (1, 8, 64, 32, 32, 0, 1, False, "float16"),
        (2, 7, 13, 5, 3, 2, 1, True, "bfloat16"),
        (1, 3, 5, 5, 1, 2, 1, False, "float16"),
    ]
    for n, c_in, l_in, kernel_size, stride, padding, dilation, ceil_mode, dtype_str in cases:
        try:
            _run_case(
                n=n,
                c_in=c_in,
                l_in=l_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                dtype_str=dtype_str,
                tag="L2",
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[L2] WARN (record only): shape=({n},{c_in},{l_in}) "
                f"k={kernel_size} s={stride} p={padding} d={dilation} "
                f"ceil={ceil_mode} dtype={dtype_str}: {str(exc)[:200]}"
            )


# ---------------------------------------------------------------------------
# Boundary (special-value inputs -- warn only, non-blocking)
# ---------------------------------------------------------------------------
def run_boundary():
    """Boundary: zeros / constant / large / subnormal / all-NaN / +-inf only /
    k=1 single-element windows."""
    common = dict(
        n=2,
        c_in=16,
        l_in=21,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        ceil_mode=False,
        dtype_str="float16",
    )

    # Zeros: exact zeros in, zeros out.
    try:
        _run_case(tag="Boundary-zeros", x=torch.zeros(2, 16, 21, dtype=torch.float16), **common)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): zeros: {str(exc)[:200]}")

    # Constant line: every window max equals the constant.
    try:
        _run_case(
            tag="Boundary-const", x=torch.full((2, 16, 21), 3.7, dtype=torch.float16), **common
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): const: {str(exc)[:200]}")

    # Large magnitudes (max selects, never overflows).
    try:
        _run_case(
            tag="Boundary-large",
            x=(torch.randn(2, 16, 21, dtype=torch.float32) * 1000.0).to(torch.float16),
            **common,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large: {str(exc)[:200]}")

    # Subnormal fp16 magnitudes (the eq mask must not clip them).
    try:
        _run_case(
            tag="Boundary-subnormal",
            x=(torch.randn(2, 16, 21, dtype=torch.float32) * 1e-5).to(torch.float16),
            **common,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {str(exc)[:200]}")

    # +-inf without NaN must never produce NaN (equal_nan=False).
    try:
        x = torch.full((2, 16, 21), float("-inf"), dtype=torch.float32)
        x[0, 1, 0] = float("inf")
        x[1, 2, :] = float("inf")
        _run_case(tag="Boundary-pm-inf-only", x=x.to(torch.float16), **common)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): pm-inf-only: {str(exc)[:200]}")

    # All-NaN window on the general path -> NaN + last-NaN index.
    try:
        x = torch.full((2, 16, 21), -3.0, dtype=torch.float32)
        x[0, 0, :] = float("nan")
        x[0, 3, 5] = float("nan")
        _run_case(tag="Boundary-all-nan-window", x=x.to(torch.float16), equal_nan=True, **common)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): all-nan-window: {str(exc)[:200]}")

    # k=1 single-element windows (a NaN survives, index = own position).
    try:
        x = torch.randn(1, 8, 12, dtype=torch.float32)
        x[0, 0, 5] = float("nan")
        x[0, 1, 7] = float("-inf")
        _run_case(
            n=1,
            c_in=8,
            l_in=12,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            ceil_mode=False,
            dtype_str="float16",
            tag="Boundary-k1",
            x=x.to(torch.float16),
            equal_nan=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): k1: {str(exc)[:200]}")

    # Non-overlapping windows on an odd length with ceil_mode.
    try:
        x = torch.randn(1, 8, 17, dtype=torch.float32)
        x[0, 2, 8] = float("nan")
        _run_case(
            n=1,
            c_in=8,
            l_in=17,
            kernel_size=4,
            stride=4,
            padding=0,
            dilation=2,
            ceil_mode=True,
            dtype_str="bfloat16",
            tag="Boundary-disjoint-ceil-d2",
            x=x.to(torch.bfloat16),
            equal_nan=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): disjoint-ceil-d2: {str(exc)[:200]}")


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
        "Summary: worst value max_diff per dtype: "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS["value_max_diff"].items()))
        + f"; index mismatches {_STATS['idx_mismatch']}/{_STATS['idx_total']}"
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
