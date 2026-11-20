# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Windowed max pooling 1D forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``_max_pool1d_kernel`` (TileOPs ``MaxPool1dFwdOp``,
``ref_api = torch.nn.functional.max_pool1d(return_indices=False)``) to TileLang
``target="npuir"``.  The ``with_indices`` variant of the source file is out of
scope for this migration.

Semantics (DESIGN.md section 0.1, frozen):
    out[n, c, ow] = max over the kW window anchored at
                    (ow*sW - pW + kw*dW)
    Out-of-range window positions behave as -inf padding (the source kernel
    skips those lanes; max(-inf, x) == x, machine-verified as E1 in
    ``verify_equiv.py``).  A NaN anywhere in the window wins (out = NaN);
    {+inf, -inf} in one window resolves to +inf (IEEE max, never NaN).
    Exactly 3D NCL, float16 / bfloat16 / float32, same dtype out.
    The output extent comes from the source helper ``pool_output_dim``
    (including both ceil_mode corrections) and is never re-derived.

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel`` + ``(n, o_blk, c_block)`` logical units with a
        persistent in-core ``T.serial`` grid-stride loop and an
        ``if uid < NUM_LOGICAL`` tail guard (static bounds).  Unlike the 2D
        sibling the length axis is chunked into output blocks ``BL`` because
        ``L_in`` reaches 4096 (a whole ``(CH, L_in)`` line does not fit UB).
    R2: UB fp32 vector accumulator on the channel axis.  The window loops are
        ``T.serial(kW)`` / ``T.serial(BL)`` with static upper bounds; the single
        spatial guard ``0 <= iw < L_in`` is per lane (1D has only one spatial
        axis, so there is no second-axis guard to hoist).  Maximum = ``T.vmax``
        on a C-vector slice; NaN tracking = explicit IEEE ``x == x`` eq mask
        multiplied into a flag accumulator seeded to 1 (0 absorbing, 1 neutral),
        final ``has_nan ? NaN : max`` via ``T.vcmp(flag, 0, "eq")`` +
        ``T.vselect``.  The kernel never relies on the undocumented ``T.vmax`` /
        ``T.reduce_max`` NaN behaviour.
    R3: contiguous clamped GM span -> UB ``T.copy`` + in-core two-axis
        ``T.transpose`` ([1, 0]) so the channel axis becomes the contiguous
        vector axis; the output is transposed back before the span store.
    R4: degenerate-grid specialisation: C4 (whole-line ``T.reduce_max`` plus an
        independent eq-mask NaN reduction) and C4b (identity UB transit copy),
        both guarded by the frozen sufficient conditions AND the UB capacity
        criterion (a pure optimisation -- an oversized line falls back to the
        general path when that path is UB-feasible).

C4 sufficient condition (DESIGN.md sections 0.6 R4 / 1.6.1, REVIEW.md problem 1):
``out_l == 1 and p_W == 0 and d_W == 1 and (k_W-1)*d_W + 1 >= L_in``.
``d_W == 1`` is mandatory: the window's coverage is ``{k_w*d_W} & [0, L_in)``,
which equals the whole line only for ``d == 1`` (a ``d > 1`` window such as
``L=5, k=2, s=1, p=0, d=4`` covers only positions {0, 4}, so a whole-line
reduction would return the wrong value).  ``verify_equiv.py`` C4x locks this
with machine-checked case-groups.

C4b sufficient condition: ``k_W == 1 and d_W == 1 and p_W == 0 and s_W == 1``
(the window is exactly one element).

1D-vs-2D differences implemented here (DESIGN.md section 1.6.1, RETROSPECTIVE):
    * Output-position blocking ``BL`` + ``span = (BL-1)*sW + (kW-1)*dW + 1``,
      ``LSPAN = min(span, L_in)``, clamped load start
      ``s0 = max(0, min(base, L_in - LSPAN))`` with ``base = ow0*sW - pW``.
      The load start is a runtime value but the extent is the static ``LSPAN``;
      ``j = iw - s0`` stays in ``[0, LSPAN)`` whenever the guard holds.
    * No second-axis guard hoisting (only one spatial axis exists).

Stage 4 round 4 (structure family change -- see perf_opt/opt_log.md
"Round-4 continuation"): two compile-time specialisations replace the general
grid for the manifest workloads, selected by pure algebra in ``_plan``:
    * **group** (``s_w == k_w``, ``d_w == 1``, ``p_w == 0``, ``k_w*out_l <= l_in``):
      the windows are the contiguous non-overlapping groups ``[k*ow, k*ow+k)``,
      so the pooling is a reduce over the group axis.  One span DMA + one
      2-axis transpose + ``T.reshape`` to ``(bl, k, ch)`` +
      ``T.reduce_max(dim=1)`` per logical unit replaces the general grid's
      ``2*k*bl`` narrow ``vmax`` ops (``2*k*out_l*N`` for the whole operator,
      independent of ``bl``).  NaN tracking is a second reduce over the same
      tile (``reduce_min`` of a 1.0/0.0 mask), and the final NaN select is done
      in float32 (``inf - inf``; ``vsub`` has no fp16/bf16 form).  Declines for
      bfloat16 (no ``vreduce`` bf16 form) and for channel blocks that are not a
      multiple of 32 B (the VEC instruction faults on unaligned reduce lanes).
      It is the sincnet path: 159.01 -> 61.99 us on 910B2.
    * **phase** (``s_w == 2``, ``p_w`` even, ``d_w`` even, ``l_in`` even): every
      tap ``iw = 2*ow - p_w + t*d_w`` is even, so on the even-phase plane
      ``ph[q] = x[2q]`` the window degenerates into a stride-``d_w/2`` sliding
      window: one 2-phase ``T.deinterleave`` + ``T.transpose`` + one **wide**
      ``vmax`` per tap instead of ``bl`` narrow ones.  It is the ecg path:
      94.34 -> 38.79 us on 910B2.
Both paths keep the tiered tests bit-exact (``max_diff = 0`` for fp16/bf16/fp32)
and every specialisation asserts that the *built* kernel really took the planned
path (a backend rejection falls back to the general grid, which is still
bit-exact but slower, so the plan alone is not evidence).

Implementation notes (Stage 3, deviations from the DESIGN.md section 3.3
pseudocode and their justification):
    * ``@tilelang.jit`` on this branch does not accept ``compile_flags`` (its
      signature is ``out_idx / target / target_host / execution_backend /
      verbose / pass_configs / debug_root_path``), so the source's
      ``["-O3", "-DENABLE_BF16"]`` is dropped -- the verified same-family kernels
      ``_max_pool2d_kernel`` / ``_adaptive_max_pool2d_kernel`` do the same.
    * ``T.vcast`` with target dtype == source dtype **miscompiles** on this
      toolchain.  Whenever the input dtype is float32 the same-shape
      same-dtype transfers therefore use ``T.copy`` instead.
    * The C4 NaN judgement reduces the eq mask with ``T.reduce_min`` (0 iff any
      element is NaN) and compares it to 0.  ``T.reduce_max`` of the same mask
      is 1 iff *at least one* element is non-NaN, so ``reduce_max`` + ``eq 0``
      would report NaN only for an all-NaN line.
    * Channel-block tails (``C % CH != 0``) use dynamic
      ``real_c = min(CH, C - cb*CH)`` copy extents; padded lanes never reach GM.
    * Output-block tails use ``bl_real = min(BL, L_out - ow0)`` dynamic extents.
    * ``block_m`` is accepted by the returned callable for source interface
      compatibility (K9 kept the wrapper's ``block_m`` config knob); NPU tiling
      is shape-derived, so the value is validated and otherwise ignored.

Factory contract (source interface unchanged):
    _max_pool1d_kernel(n, c_in, l_in, kernel_w, stride_w, pad_w, dilation_w,
                       ceil_mode, dtype="float16")
      -> build(block_m=None) -> main
    where ``main(x)`` returns ``out`` (``out_idx=[1]``).

Run the embedded hierarchical tests:
    python _max_pool1d_kernel.py --level L0
    python _max_pool1d_kernel.py --level all
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
    "_max_pool1d_kernel",
    "golden_max_pool1d",
    "pool_output_dim",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Unified Buffer capacity per Vector core (developer guide: 192 KB).
_UB_CAPACITY_BYTES = 196_608

# Conservative multi-buffer inflation factor for the UB budget guard
# (DESIGN.md section 4.5 default tier; auto-multi-buffer enabled).
_UB_MULTIBUFFER_FACTOR = 1.7

# Stage 4 round 1 (R1b) C4 footprint coefficient: the whole-line path's main
# term is CH*L_in*(eb + _C4_BYTES_PER_ELEM) instead of (eb + 9) -- the eq_mask
# / eq_f32 / nmin planes are gone.
_C4_BYTES_PER_ELEM = 4

# Supported dtypes: the windowed max-pool op accepts float32 as well, unlike
# the adaptive pool family.
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# Precision tolerances (DESIGN.md section 8.2): max selects an input element
# and performs no accumulation, so the result is bit-exact
# (test_pool.py: "Max is exact (no accumulation): compare bit-for-bit").
_TOLERANCE = {
    "float16": (torch.float16, 0.0, 0.0),
    "bfloat16": (torch.bfloat16, 0.0, 0.0),
    "float32": (torch.float32, 0.0, 0.0),
}

# fp16/bf16 vector width in elements (alignment of the vector axis).
_VECTOR_WIDTH = 8

# Target channel-vector-block ceiling (DESIGN.md sections 5.2 / 1.6.3).
# Stage 4 round 1 (R1a): raised 64 -> 128.  With CH pinned at 64 the
# channel-block tail (`real_c = C - cb*CH`) still costs *full-CH* window-op
# lanes for every padded lane, so sincnet (C=80, CH=64) ran 2 channel blocks
# with the second only 16/64 useful (~37% wasted lanes, ~2x logical units).
# The capacity inequalities below are unchanged, so BL shrinks instead.
_CH_MAX_TUNED = 128
_CH_MAX = _CH_MAX_TUNED

# Output-position block ceiling (DESIGN.md section 5.2; halo shrinks with BL,
# capacity caps it).
_BL_MAX = 32


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
# Tiling plan (host/trace side): CH/BL joint selection, degenerate-path
# selection, core split
# ---------------------------------------------------------------------------
def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _span(bl: int, k_w: int, s_w: int, d_w: int, l_in: int) -> int:
    """Contiguous input span covered by one BL output block, clamped to L_in."""
    return min((bl - 1) * s_w + (k_w - 1) * d_w + 1, l_in)


def _general_bytes(ch: int, lspan: int, bl: int, elem_bytes: int) -> int:
    """Per-core UB footprint of the general path (bytes, no inflation).

    Matches DESIGN.md 4.5 ``G = CH*LSPAN*(2*eb+9) + BL*CH*(25+2*eb)``:
    ``2*eb+9`` = row_src + row_dst + row_f32 + eq_mask + eq_f32;
    ``25+2*eb`` = acc / flag / zeros / infbuf / nanbuf / out32 (6 x f32) +
    cfinal (bool) + acc16 + acc_t.
    """
    return (
        2 * ch * lspan * elem_bytes  # row_src, row_dst
        + 2 * lspan * ch * 4  # row_f32, eq_f32
        + lspan * ch  # eq_mask (bool)
        + 6 * bl * ch * 4  # acc, flag, zeros, infbuf, nanbuf, out32
        + bl * ch  # cfinal (bool)
        + 2 * bl * ch * elem_bytes  # acc16, acc_t
    )


def _global_bytes(ch: int, l_in: int, elem_bytes: int) -> int:
    """Per-core UB footprint of the C4 whole-line path (bytes, no inflation).

    Main term ``CH*L_in*(eb+9)`` = flat (eb) + flat32 (4) + eq_mask (1) +
    eq_f32 (4); small term ``CH*(25+eb)`` covers gmax/nmin/zeros/infbuf/nanbuf/
    out32 (CH*1 f32 x6), cond (CH bool) and gout (CH*eb).
    """
    return ch * l_in * (elem_bytes + _C4_BYTES_PER_ELEM) + ch * (25 + elem_bytes)


def _identity_bytes(ch: int, l_in: int, elem_bytes: int) -> int:
    """Per-core UB footprint of the C4b identity path (bytes, no inflation)."""
    return ch * l_in * elem_bytes


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
# Group-reduce path (Stage 4 round 4, ``s_w == k_w and d_w == 1 and p_w == 0``)
# ---------------------------------------------------------------------------
# Algebra (exact, no approximation): with ``s_w == k_w``, ``d_w == 1`` and
# ``p_w == 0`` the windows are the contiguous non-overlapping groups
# ``[k_w*ow, k_w*ow + k_w)`` -- the pooling is literally "view L as
# ``(out_l, k_w)`` and reduce the size-``k_w`` axis".  ``k_w*out_l <= l_in``
# always holds in floor mode, so a block's span is exactly
# ``[k_w*ow0, k_w*(ow0+bl_real))`` with **no halo and no clamp** (the manifest
# sincnet case wastes exactly one input element at the end of every line).
#
# Data flow per logical unit -- no transpose anywhere:
#
#   GM span (ch, k*bl) --GMDMA--> src --T.reshape--> s3 (ch, bl, k)
#   T.reduce_max(s3, lacc3, dim=2)     -> lacc2 (ch, bl)   # already store layout
#   T.reduce_min(nm3,  nacc3, dim=2)   -> nacc2 (ch, bl)   # 1.0 = no NaN
#   out[ow, :] = has_nan ? NaN : max
#
# Why this is a *structural* win and not a tiling tweak: the round-1 general grid
# emits ``2*k_w*bl`` narrow (1 x ch) vector ops per unit -- ``2*k_w*out_l*N``
# for the whole kernel, independent of ``bl`` -- while this path emits two
# reduce ops over the full (ch, bl, k) tile per unit, i.e. ``2*N*ceil(out_l/bl)``
# wide ops.  For sincnet that is 524 160 narrow ops -> 1 152 reduce ops (455x
# fewer op issues) at the same GM traffic.
#
# Toolchain notes (re-verified on this machine, see opt_log round-4 section):
#   * ``T.reduce_max/min(..., dim=2)`` on a 3-D UB tile whose reduced axis is the
#     innermost one compiles to a real wide reduce (the ``(ch, bl, k)`` view of
#     the natural GM layout) -- and is bit-exact;
#   * ``T.reshape`` (``npuir_reshape``) is the documented way to give a UB buffer
#     a different rank; both the 2-D -> 3-D and the 3-D -> 2-D direction work;
#   * ``T.vcast`` with target dtype == source dtype miscompiles -> ``T.copy``;
#   * the NaN value itself is built in fp32 (``inf - inf``): ``vsub`` has no
#     fp16/bf16 form, so the final select is done in fp32 and cast back.
_GROUP_CH_STEPS = (128, 96, 80, 64, 48, 40, 32, 24, 16, 8)
_GROUP_BL_CANDIDATES = (128, 96, 80, 64, 48, 32, 24, 16)
# Same naive-sum budget convention as the phase path (the backend reuses
# non-overlapping buffers, so the naive sum runs above the 196 608 B hardware
# limit).
_GROUP_UB_SUM_BUDGET_BYTES = int(os.environ.get("MAXPOOL1D_GROUP_UB_BUDGET", "0")) or 240_000
# Largest bl measured to compile *and* run without a device fault (see the
# selection-rule note in _group_plan).
_GROUP_BL_SAFE_MAX = 96
_GROUP_CH_OVERRIDE = int(os.environ.get("MAXPOOL1D_GROUP_CH", "0")) or None
_GROUP_BL_OVERRIDE = int(os.environ.get("MAXPOOL1D_GROUP_BL", "0")) or None


def _group_bytes(ch, bl, k_w, elem_bytes):
    """Naive per-buffer UB footprint of the group-reduce path (bytes).

    Keep in sync with the ``T.alloc_ub`` list of ``_func_group``.
    """
    f32 = 4
    lanes = ch * bl * k_w
    return (
        2 * lanes * elem_bytes  # src (ch, k*bl), t2 (k*bl, ch)
        + lanes * elem_bytes  # t3 (bl, k, ch)
        + lanes  # isb3 (bl, k, ch) bool
        + lanes * elem_bytes  # nm3 (bl, k, ch) mask, 1.0 = number
        + 2 * ch * bl * elem_bytes  # acc3 (bl,1,ch), nacc3
        + 2 * ch * bl * elem_bytes  # acc2, nacc2
        + ch * bl  # cf (bool)
        + 2 * ch * bl * f32  # acc32, nanb
        + ch * bl * elem_bytes  # st (ch, bl)
    )


def _group_plan(n, c_in, l_in, out_l, k_w, s_w, p_w, d_w, dtype, vector_cores=None):
    """Group-reduce tiling plan, or ``None`` when the specialisation does not apply.

    Applicability is purely algebraic (``s_w == k_w``, ``d_w == 1``, ``p_w == 0``,
    ``k_w * out_l <= l_in``) plus "the tile fits the UB budget at some
    ``bl >= 16``".  Every other shape keeps the general grid.
    """
    if s_w != k_w or d_w != 1 or p_w != 0 or k_w < 1:
        return None
    if k_w * out_l > l_in:
        return None
    # Toolchain limits measured on this machine (910B2 / f947fb1):
    #   * ``hivm.hir.vreduce`` has no bfloat16 form (same gap the C4 path works
    #     around by casting to float32 first), so bf16 keeps the general grid;
    #   * the innermost extent of the 3-D reduce tile must be >= 32 B, otherwise
    #     the VEC instruction faults with "UB address ... is not aligned"
    #     (measured: ch=8 fp16 = 16 B faults, ch=8 fp32 = 32 B is fine).
    if dtype == "bfloat16":
        return None
    elem_bytes = _elem_bytes(dtype)
    ch_cands = []
    for ch in _GROUP_CH_STEPS:
        if ch < _VECTOR_WIDTH or ch > c_in:
            continue
        if c_in % ch == 0:
            ch_cands.append(ch)
    # Only exact channel blocks: a *partial* c-block (real_c < ch) was measured
    # to fault the device at runtime inside this 3-D reduce path even though the
    # identical shape works on the phase/general paths, so the group path keeps
    # the strict "ch divides C" invariant and otherwise falls through.
    whole = _round_up(c_in, _VECTOR_WIDTH)
    if whole == c_in and whole <= 128 and whole not in ch_cands:
        ch_cands.append(whole)
    if _GROUP_CH_OVERRIDE is not None:
        ch_cands = [c for c in ch_cands if c == _GROUP_CH_OVERRIDE]

    feasible = []
    for ch in ch_cands:
        for bl in _GROUP_BL_CANDIDATES:
            if bl < 16 or k_w * bl > l_in or bl > out_l:
                continue
            # 32 B lane granularity: the 3-D reduce faults ("UB address ...
            # is not aligned") unless the innermost extent is a multiple of 32 B
            # -- measured: ch*eb = 16 B and 48 B fault, 32 B and 160 B are fine.
            if ch * elem_bytes % 32:
                continue
            if _GROUP_BL_OVERRIDE is not None and bl != _GROUP_BL_OVERRIDE:
                continue
            nb = _group_bytes(ch, bl, k_w, elem_bytes)
            if nb > _GROUP_UB_SUM_BUDGET_BYTES:
                continue
            feasible.append((ch, bl, nb, bl * ch))
    if not feasible:
        return None
    # Selection rule: stay within 80% of the widest tile, then take the longest
    # DMA burst (largest bl); ties prefer the wider channel block.  Measured on
    # the sincnet shape: the widest tile and the longest burst disagree, and the
    # burst wins (the whole path is DMA-bound once the reduce is wide).
    # Selection rule: widest tile wins, ties prefer the wider channel block
    # (single c-block => no c-block split).  ``bl`` is capped at
    # _GROUP_BL_SAFE_MAX: tiles with bl > 96 were measured to *fault the device*
    # at runtime (silent UB over-allocation, not a compile error) even when the
    # naive byte sum is identical to a working tile -- e.g. (ch 40, bl 128) and
    # (ch 16, bl 160) fault while (ch 80, bl 64) at the same byte count runs
    # bit-exactly.  Measured on 910B2 / f947fb1.
    pool = [f for f in feasible if f[1] <= _GROUP_BL_SAFE_MAX]
    if not pool:
        return None
    ch, bl, nb, lanes = max(pool, key=lambda f: (f[3], f[0]))
    nob = (out_l + bl - 1) // bl
    num_cblk = (c_in + ch - 1) // ch
    num_logical = n * nob * num_cblk
    if vector_cores is None:
        vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_kernels = min(num_logical, vector_cores)
    return {
        "ch": ch,
        "bl": bl,
        "lanes": lanes,
        "k_w": k_w,
        "lspan": k_w * bl,
        "nob": nob,
        "num_cblk": num_cblk,
        "out_l": out_l,
        "elem_bytes": elem_bytes,
        "num_logical": num_logical,
        "num_kernels": num_kernels,
        "num_local": (num_logical + num_kernels - 1) // num_kernels,
        "bytes": nb,
    }


# ---------------------------------------------------------------------------
# Phase-decimated wide-tile path (Stage 4 round 4, ``s_w == 2`` specialisation)

# ---------------------------------------------------------------------------
# Algebra (exact, no approximation, checked before any API is chosen):
# with ``s_w == 2``, ``p_w`` even and ``d_w`` even, every tap of output ``ow``
# sits at
#
#     iw = 2*ow - p_w + t*d_w            (t = 0 .. k_w-1)
#
# which is *even* for every ``t``.  In the even-phase plane ``ph[q] = x[2q]``
# (``q = iw/2``) the window therefore degenerates into a stride-``dd = d_w/2``
# sliding window over ``q = ow - p_w/2 + t*dd``: **one wide vector op per tap**
# instead of the general grid's ``bl`` narrow ops per tap (``k_w*bl`` narrow
# visits per unit -> ``k_w`` wide ones).
#
# Data flow per logical unit (only the 2-axis forms that were measured as free
# in round 1 -- no 3-D transpose, no gather, no strided extract anywhere):
#
#   GM span (ch, 2*pp) --GMDMA--> src --T.deinterleave(cn=2)--> even phase pe
#     --2D transpose--> pft (pp, ch) --vcast--> pf (float32, lane = channel)
#   out[ow, :] = max_t pf[pad + off + t*dd, :]
#
# ``pf`` is armed once with -inf and ``nmask`` (1.0 = NaN) / ``nacc`` (armed 1.0)
# carry the NaN semantics with one extra wide ``vmin`` per tap: a NaN anywhere in
# the window makes ``nacc == 0`` for that output, and the final whole-buffer
# ``vselect`` writes NaN.  Out-of-range taps land on the armed ``pad`` rows and
# are therefore *neutral* for both reductions -- the analogue of the general
# path's ``0 <= iw < l_in`` guard, without any per-lane branch.
#
# Frozen toolchain constraints reused here (each re-verified on this machine, see
# opt_log.md section 4 of the round-4 continuation):
#   * ``T.deinterleave`` supports ``channel_nums=2`` only (>2 SIGABRTs), and must
#     not sit inside a runtime if/else that the tracer can fold away;
#   * ``T.vselect`` accepts **no** sliced operand (whole buffers only), while
#     ``vmax``/``vmin``/``vcmp``/``vadd``/``vcast`` accept same-shape slices;
#   * ``T.vcast`` with target dtype == source dtype miscompiles -> ``T.copy``;
#   * ``T.vbrc`` never targets a runtime-offset region, so the arming happens
#     once, outside the unit loop.
_PHASE_CH_STEPS = (128, 96, 64, 48, 32, 24, 16, 8)
_PHASE_BL_CANDIDATES = (64, 128, 48, 40, 96, 32, 24, 16)
# Store rows narrower than ~32 B are amplified by the GM write granularity, so
# the phase path never picks bl below _PHASE_MIN_BL.
_PHASE_MIN_BL = 16
# bl ceiling kept at the largest tile measured to compile and run (bl 80/96/128
# tiles exceed the backend's UB budget on the ecg shape).  (An earlier apparent
# (ch 64, bl 64) fault was a planner bug -- a stale `nob` made num_logical too
# large, so tail units ran out of bounds -- not a tile property.)
_PHASE_BL_SAFE_MAX = 64
# ch == 32 is excluded: on 910B3/d2783bd the sibling phase path returned wrong
# trailing columns at exactly ch == 32 (all other widths bit-exact).  Not
# re-derived here -- the planner simply never selects it (same exclusion as the
# sibling) and ch == 32 is covered by an explicit L0 case on the general path.
_PHASE_CH_EXCLUDED = (32,)
# Naive per-buffer UB sum budget.  The backend reuses non-overlapping buffers, so
# the naive sum runs well above the hardware's 196 608 B (calibrated on the
# sibling task: naive 309 KB compiles, 342 KB overflows).  300 KB accepts every
# configuration measured to compile there.
_PHASE_UB_SUM_BUDGET_BYTES = int(os.environ.get("MAXPOOL1D_PHASE_UB_BUDGET", "0")) or 280_000
# Stage 4 tuning knobs (0/unset = auto planner), e.g.
#   MAXPOOL1D_PHASE_CH=64 MAXPOOL1D_PHASE_BL=40 python ...
_PHASE_CH_OVERRIDE = int(os.environ.get("MAXPOOL1D_PHASE_CH", "0")) or None
_PHASE_BL_OVERRIDE = int(os.environ.get("MAXPOOL1D_PHASE_BL", "0")) or None


def _phase_bytes(ch, pp, rows, bl, elem_bytes):
    """Naive per-buffer UB footprint of the phase path (bytes).

    Keep in sync with the ``T.alloc_ub`` list of ``_func_phase``; the budget
    constant above is calibrated against this sum.
    """
    f32 = 4
    return (
        ch * 2 * pp * elem_bytes  # src (ch, 2*pp)
        + 2 * ch * pp * elem_bytes  # pe, po (deinterleave outputs)
        + pp * ch * elem_bytes  # pft (pp, ch)
        + pp * ch * f32  # pff (pp, ch) float32
        + rows * ch * f32  # pf (rows, ch), -inf armed
        + rows * ch  # isnum (bool)
        + rows * ch * f32  # nmask (rows, ch), 1.0 = NaN
        + 3 * bl * ch * f32  # acc, nacc, out32
        + bl * ch * f32  # nanb (NaN source for vselect)
        + bl * ch  # cf (bool)
        + bl * ch * elem_bytes  # acc16
        + ch * bl * elem_bytes  # st (ch, bl)
    )


def _phase_ch_candidates(c_in):
    """Channel blocks worth trying: exact divisors of C, plus one-block C.

    Partial channel blocks are excluded: the sibling measured that a partial
    tail costs much more than the extra lanes buy (ch=96 -> 185.58 us and
    ch=48 -> 113.24 us against 91.18 us for an exact divisor on the same shape).
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


def _phase_plan(n, c_in, l_in, out_l, k_w, s_w, p_w, d_w, dtype, vector_cores=None):
    """Phase-path tiling plan, or ``None`` when the specialisation does not apply.

    Applicability is purely algebraic (``s_w == 2``, ``p_w`` even, ``d_w`` even,
    ``l_in`` even) plus "the phase plane fits the UB budget at
    ``bl >= _PHASE_MIN_BL``".  Every other shape keeps the general grid, so no
    correctness depends on this path being taken.

    Selection rule: among the tiles within 80% of the widest one and inside the
    UB budget, take the largest ``bl`` (longest DMA bursts, widest store rows),
    then the larger ``ch``.
    """
    if s_w != 2 or p_w % 2 or d_w % 2 or l_in % 2 or k_w < 1:
        return None
    dd = d_w // 2
    ph = p_w // 2
    nq = l_in // 2  # number of even-phase positions
    tap_span = (k_w - 1) * dd
    pp_max = (nq // _VECTOR_WIDTH) * _VECTOR_WIDTH
    bl_max = min(min(out_l, max(_PHASE_BL_CANDIDATES)), pp_max - tap_span)
    if bl_max < _PHASE_MIN_BL:
        return None
    elem_bytes = _elem_bytes(dtype)
    ch_cands = _phase_ch_candidates(c_in)
    if _PHASE_CH_OVERRIDE is not None:
        ch_cands = [c for c in ch_cands if c == _PHASE_CH_OVERRIDE]

    feasible = []
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
        # Headroom: `pad` rows absorb every negative offset, and 2 trailing rows
        # absorb the clamp at the top end.
        pad = ph + 1
        if pad + off_min < 0:
            continue
        max_row = pad + off_max + (bl - 1) + tap_span
        rows = max(pad + pp + 2, max_row + 1)
        for ch in ch_cands:
            nb_bytes = _phase_bytes(ch, pp, rows, bl, elem_bytes)
            if nb_bytes > _PHASE_UB_SUM_BUDGET_BYTES:
                continue
            feasible.append((ch, bl, nb_bytes, bl * ch, pp, lspan, pad, rows))
    if not feasible:
        return None
    # Selection rule: stay within 80% of the widest tile, then take the longest
    # DMA burst (largest bl); ties prefer the wider channel block.  Measured on
    # the ecg shape (910B2, msprof op): (ch64, bl64) 37.34 us beats the widest
    # tile (ch128, bl40) 39.97 us and (ch64, bl80) does not compile -- the path
    # is DMA-burst-bound once the wide ops are cheap.
    pool = [f for f in feasible if f[1] <= _PHASE_BL_SAFE_MAX]
    if not pool:
        return None
    lane_max = max(f[3] for f in pool)
    pool = [f for f in pool if f[3] * 5 >= lane_max * 4]
    ch, bl, nb_bytes, lanes, pp, lspan, pad, rows = max(pool, key=lambda f: (f[1], f[0]))
    # Recompute every bl-derived quantity from the SELECTED bl: `nob`/`s0_max`
    # from the candidate loop must never leak into the plan (a stale nob makes
    # num_logical too large, so the tail units read/write outside the tensor).
    nob = (out_l + bl - 1) // bl
    s0_max = 2 * (nq - pp)
    num_cblk = (c_in + ch - 1) // ch
    num_logical = n * nob * num_cblk
    if vector_cores is None:
        vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_kernels = min(num_logical, vector_cores)
    return {
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
        "bytes": nb_bytes,
    }


def _degenerate_conditions(out_l, l_in, k_w, s_w, p_w, d_w):
    """Frozen sufficient conditions of the C4 / C4b specialisations."""
    identity = k_w == 1 and d_w == 1 and p_w == 0 and s_w == 1
    global_line = out_l == 1 and p_w == 0 and d_w == 1 and (k_w - 1) * d_w + 1 >= l_in
    return identity, global_line


def _plan(n, c_in, l_in, k_w, s_w, p_w, d_w, ceil_mode, dtype, vector_cores=None):
    """Derive the compile-time tiling plan for one shape/dtype combination.

    Returns a dict with the output extent, the channel/output blocks, the
    degenerate path tag and the persistent core-split constants.  ``vector_cores``
    defaults to the device query (pure-Vector op -> AI cores x 2).
    """
    out_l = pool_output_dim(l_in, k_w, s_w, p_w, ceil_mode, d_w)
    if out_l < 1:
        raise ValueError(f"output size must be greater than zero, got out_l={out_l}")
    elem_bytes = _elem_bytes(dtype)
    ch, bl = _choose_ch_bl(c_in, l_in, out_l, k_w, s_w, d_w, elem_bytes)
    num_cblk = (c_in + ch - 1) // ch

    id_cond, gl_cond = _degenerate_conditions(out_l, l_in, k_w, s_w, p_w, d_w)
    is_identity = (
        id_cond
        and _identity_bytes(ch, l_in, elem_bytes) * _UB_MULTIBUFFER_FACTOR <= _UB_CAPACITY_BYTES
    )
    is_global = (
        gl_cond
        and not is_identity
        and _global_bytes(ch, l_in, elem_bytes) * _UB_MULTIBUFFER_FACTOR <= _UB_CAPACITY_BYTES
    )
    path = "identity" if is_identity else ("global" if is_global else "general")
    # Round-4 R4-c: the degenerate (identity / C4) paths were inheriting the
    # *general* grid's channel block, whose UB price per element is much higher
    # (2*eb+9 vs eb+_C4_BYTES_PER_ELEM), so textcnn ran with CH=64 (4 channel
    # blocks) even though its whole-line footprint allows CH=128 (2 blocks).
    # Re-derive CH from the footprint of the path that is actually taken.
    if path in ("identity", "global"):

        def _deg_bytes(c):
            if path == "identity":
                return _identity_bytes(c, l_in, elem_bytes)
            return _global_bytes(c, l_in, elem_bytes)

        wide = _round_up(min(c_in, _CH_MAX), _VECTOR_WIDTH)
        while wide > ch:
            if _deg_bytes(wide) * _UB_MULTIBUFFER_FACTOR <= _UB_CAPACITY_BYTES:
                ch = wide
                break
            wide = max(_VECTOR_WIDTH, _round_up(wide // 2, _VECTOR_WIDTH))
        num_cblk = (c_in + ch - 1) // ch
    # Stage 4 round 4: the group-reduce path (disjoint contiguous windows) and
    # the phase-decimated path (s_w == 2) replace the general grid when their
    # algebra applies.  They are mutually exclusive (group needs d_w == 1, phase
    # needs d_w even).
    group = None
    phase = None
    if path == "general":
        group = _group_plan(n, c_in, l_in, out_l, k_w, s_w, p_w, d_w, dtype, vector_cores)
        if group is not None:
            path = "group"
    if path == "general":
        phase = _phase_plan(n, c_in, l_in, out_l, k_w, s_w, p_w, d_w, dtype, vector_cores)
        if phase is not None:
            path = "phase"
    if path == "general" and bl < 1:
        # No UB-feasible plan: not reachable for the manifest workloads (all
        # have a small kernel / span).  Reject instead of emitting a kernel that
        # overflows UB at compile time.
        raise ValueError(
            f"no UB-feasible plan for (l_in={l_in}, kernel_w={k_w}, "
            f"stride_w={s_w}, dilation_w={d_w}, c_in={c_in}): the general-grid "
            f"span is at least {(k_w - 1) * d_w + 1} elements wide, which does "
            f"not fit the 192 KiB UB budget even at CH=8/BL=1"
        )

    # The general-grid span is always derived, even when the phase path is
    # selected: it is the build-time fallback's tile if the backend rejects the
    # phase tile (a zero here would silently build a kernel that DMAs nothing).
    lspan = _span(bl, k_w, s_w, d_w, l_in) if bl >= 1 else 0

    # Logical unit granularity is the single definition source of NUM_LOGICAL
    # (DESIGN.md 5.5 item 1): general unit = (n, o_blk, c_block); C4 / C4b unit =
    # the whole (n, c_block) line.  The general-grid counts are kept in the plan
    # even when the phase path is selected, because the phase build falls back to
    # the general grid if the backend rejects the phase tile; the phase unit
    # count lives in plan["phase"]["num_logical"].
    if path in ("identity", "global"):
        num_oblk = None
        num_logical = n * num_cblk
    else:
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
        "path": path,
        "phase": phase,
        "group": group,
        "elem_bytes": elem_bytes,
        "num_logical": num_logical,
        "num_kernels": num_kernels,
        "num_local": num_local,
        "general_bytes": _general_bytes(ch, lspan, bl, elem_bytes) if bl >= 1 else 0,
        "global_bytes": _global_bytes(ch, l_in, elem_bytes),
        "identity_bytes": _identity_bytes(ch, l_in, elem_bytes),
    }


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_max_pool1d(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    """PyTorch reference for max_pool1d, per DESIGN.md section 0.1.

    Directly uses the source-repo test benchmark (``MaxPool1dTest.ref_program``
    in examples/TileOPs/tests/ops/test_pool.py).  Exactly 3D NCL; independent of
    the NPU algorithm (no transpose / no C-vectorisation / no span-load /
    no C4-C4b specialisation).  Runs on CPU and does not require an NPU device.
    """
    return F.max_pool1d(
        x.detach().to(torch.float32),  # fp32 opmath domain for fp16/bf16 CPU ref
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    ).to(x.dtype)


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------
@kernel_factory_cache
def _max_pool1d_kernel(
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
    """Build the windowed max pooling 1D kernel factory.

    Returns a build callable taking a single ``block_m`` (the CUDA ``threads``
    parameter is removed, K9); ``build(block_m)`` compiles and returns ``main``;
    ``main(x)`` returns ``out`` (``out_idx=[1]``).
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"MaxPool1dFwdOp supports float16, bfloat16, and float32, got {dtype}")
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

    # The output extent is validated before the tiling plan is derived: a
    # non-positive extent makes the core-split arithmetic degenerate, so the
    # Op layer's "output size must be greater than zero" guard runs first.
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    if out_l < 1:
        raise ValueError(
            f"output size must be greater than zero, got {out_l} for input {l_in} kernel {kernel_w}"
        )

    plan = _plan(n, c_in, l_in, kernel_w, stride_w, pad_w, dilation_w, ceil_mode, dtype)
    assert plan["out_l"] == out_l

    ch = plan["ch"]
    bl = plan["bl"]
    lspan = plan["lspan"]
    num_cblk = plan["num_cblk"]
    num_oblk = plan["num_oblk"]
    path = plan["path"]
    num_logical = plan["num_logical"]
    num_kernels = plan["num_kernels"]
    num_local = plan["num_local"]
    is_f32 = dtype == "float32"

    def _build(block_m=None):
        if block_m is not None and block_m < 1:
            raise ValueError(f"block_m must be positive, got block_m={block_m}")

        if path == "group":
            # ---- group-reduce path (disjoint contiguous windows) ----
            # The reduce MUST be the *middle*-axis form (bl, k, ch) with
            # reduce_dims = [1]: the innermost-group form (ch, bl, k) with
            # reduce_dims = [2] compiles but is scalarised (measured 2234 us vs
            # 35.79 us, probe_sinc3_r4.py variants `last` / `reduce3`).
            g_ch = plan["group"]["ch"]
            g_bl = plan["group"]["bl"]
            g_lspan = plan["group"]["lspan"]
            g_k = plan["group"]["k_w"]
            g_nob = plan["group"]["nob"]
            g_ncb = plan["group"]["num_cblk"]
            g_nlog = plan["group"]["num_logical"]
            g_nk = plan["group"]["num_kernels"]
            g_nloc = plan["group"]["num_local"]

            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_group():
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, l_in), dtype),
                    out: T.Tensor((n, c_in, out_l), dtype),
                ):
                    with T.Kernel(g_nk, is_npu=True) as (cid, _):
                        src = T.alloc_ub((g_ch, g_lspan), dtype)
                        t2 = T.alloc_ub((g_lspan, g_ch), dtype)
                        t3 = T.alloc_ub((g_bl, g_k, g_ch), dtype)
                        acc3 = T.alloc_ub((g_bl, 1, g_ch), dtype)
                        acc2 = T.alloc_ub((g_bl, g_ch), dtype)
                        isb3 = T.alloc_ub((g_bl, g_k, g_ch), "bool")
                        nm3 = T.alloc_ub((g_bl, g_k, g_ch), dtype)
                        nacc3 = T.alloc_ub((g_bl, 1, g_ch), dtype)
                        nacc2 = T.alloc_ub((g_bl, g_ch), dtype)
                        cf = T.alloc_ub((g_bl, g_ch), "bool")
                        acc32 = T.alloc_ub((g_bl, g_ch), "float32")
                        nanb = T.alloc_ub((g_bl, g_ch), "float32")
                        st = T.alloc_ub((g_ch, g_bl), dtype)

                        pos_inf = T.infinity("float32")
                        one_val = T.float32(1.0)

                        for t in T.serial(g_nloc):
                            uid = t * g_nk + cid
                            if uid < g_nlog:
                                nb = uid // (g_nob * g_ncb)
                                rem = uid % (g_nob * g_ncb)
                                oblk = rem // g_ncb
                                cb = rem % g_ncb
                                real_c = T.min(g_ch, c_in - cb * g_ch)
                                ow0 = oblk * g_bl
                                bl_real = T.min(g_bl, out_l - ow0)
                                # Disjoint contiguous windows: the span is
                                # exactly [k*ow0, k*(ow0+bl_real)), no halo and no
                                # clamp (the plan is guarded by k_w*out_l <= l_in).
                                T.copy(
                                    x[
                                        nb,
                                        cb * g_ch : cb * g_ch + real_c,
                                        ow0 * g_k : ow0 * g_k + g_k * bl_real,
                                    ],
                                    src[0:real_c, 0 : g_k * bl_real],
                                )
                                # (ch, k*bl) -> (k*bl, ch) -> (bl, k, ch): the tap
                                # axis becomes a middle axis, one wide reduce.
                                T.transpose(src, t2, permutation=[1, 0])
                                T.reshape(t2, t3)
                                T.reduce_max(t3, acc3, dim=1, clear=True)
                                T.reshape(acc3, acc2)
                                # NaN: 1.0 = number / 0.0 = NaN, min over the tap
                                # axis -> 0.0 iff any lane of the window is NaN.
                                T.vcmp(t3, t3, isb3, "eq")
                                T.vcast(isb3, nm3, round_mode="rint")
                                T.reduce_min(nm3, nacc3, dim=1, clear=True)
                                T.reshape(nacc3, nacc2)
                                T.vcmp(nacc2, one_val, cf, "lt")
                                if is_f32:
                                    T.copy(acc2, acc32)  # same-dtype vcast trap
                                else:
                                    T.vcast(acc2, acc32, round_mode="rint")
                                # NaN = inf - inf; vsub has no fp16/bf16 form, so
                                # the select stays in fp32.
                                T.vbrc(pos_inf, nanb)
                                T.vsub(nanb, nanb, nanb)
                                T.vselect(cf, nanb, acc32, acc32)
                                if is_f32:
                                    T.copy(acc32, acc2)
                                else:
                                    T.vcast(acc32, acc2, round_mode="rint")
                                T.transpose(acc2, st, permutation=[1, 0])
                                T.copy(
                                    st[0:real_c, 0:bl_real],
                                    out[nb, cb * g_ch : cb * g_ch + real_c, ow0 : ow0 + bl_real],
                                )

                return main

            try:
                _build.built_path = "group"
                return _func_group()
            except Exception as exc:  # noqa: BLE001
                print(
                    "[_max_pool1d_kernel] group-reduce path rejected by the "
                    f"backend for this shape, using the general grid: "
                    f"{str(exc).splitlines()[0][:160]}"
                )

        if path == "phase":
            # ---- phase-decimated wide-tile path (see the header block) ----
            p_ch = plan["phase"]["ch"]
            p_bl = plan["phase"]["bl"]
            p_pp = plan["phase"]["pp"]
            p_lspan = plan["phase"]["lspan"]
            p_pad = plan["phase"]["pad"]
            p_rows = plan["phase"]["rows"]
            p_dd = plan["phase"]["dd"]
            p_ph = plan["phase"]["ph"]
            p_nq = plan["phase"]["nq"]
            p_nob = plan["phase"]["nob"]
            p_ncb = plan["phase"]["num_cblk"]
            p_nlog = plan["phase"]["num_logical"]
            p_nk = plan["phase"]["num_kernels"]
            p_nloc = plan["phase"]["num_local"]

            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_phase():
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, l_in), dtype),
                    out: T.Tensor((n, c_in, out_l), dtype),
                ):
                    with T.Kernel(p_nk, is_npu=True) as (cid, _):
                        src = T.alloc_ub((p_ch, p_lspan), dtype)
                        pe = T.alloc_ub((p_ch, p_pp), dtype)
                        po = T.alloc_ub((p_ch, p_pp), dtype)
                        pft = T.alloc_ub((p_pp, p_ch), dtype)
                        pff = T.alloc_ub((p_pp, p_ch), "float32")
                        pf = T.alloc_ub((p_rows, p_ch), "float32")
                        isnum = T.alloc_ub((p_rows, p_ch), "bool")
                        nmask = T.alloc_ub((p_rows, p_ch), "float32")
                        acc = T.alloc_ub((p_bl, p_ch), "float32")
                        nacc = T.alloc_ub((p_bl, p_ch), "float32")
                        cf = T.alloc_ub((p_bl, p_ch), "bool")
                        out32 = T.alloc_ub((p_bl, p_ch), "float32")
                        nanb = T.alloc_ub((p_bl, p_ch), "float32")
                        acc16 = T.alloc_ub((p_bl, p_ch), dtype)
                        st = T.alloc_ub((p_ch, p_bl), dtype)

                        neg_inf = -T.infinity("float32")
                        pos_inf = T.infinity("float32")
                        one_f = T.float32(1.0)
                        # Armed once, outside the unit loop: the pad rows of `pf`
                        # keep -inf (neutral for the value max) and their `nmask`
                        # rows are re-derived as 1.0 = "not NaN" (neutral for the
                        # NaN min) because `-inf == -inf` is true.  This is what
                        # makes out-of-range window taps behave like the general
                        # path's `0 <= iw < l_in` guard with no per-lane branch.
                        T.vbrc(neg_inf, pf)
                        T.vbrc(pos_inf, nanb)
                        T.vsub(nanb, nanb, nanb)  # NaN = inf - inf (no bf16 form)

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
                                # Only the even phase is consumed, but the DMA is
                                # one contiguous (ch, 2*pp) span: the deinterleave
                                # below is the stride-2 decimation.
                                T.copy(
                                    x[
                                        nb,
                                        cb * p_ch : cb * p_ch + real_c,
                                        2 * q0 : 2 * q0 + p_lspan,
                                    ],
                                    src[0:real_c, 0:p_lspan],
                                )
                                T.deinterleave(src, pe, po, channel_nums=2)
                                T.transpose(pe, pft, permutation=[1, 0])
                                if is_f32:
                                    # T.vcast with target == source dtype
                                    # miscompiles (header note).
                                    T.copy(pft, pff)
                                else:
                                    T.vcast(pft, pff, round_mode="rint")
                                T.copy(pff, pf[p_pad : p_pad + p_pp, 0:p_ch])
                                # One NaN plane per unit: 1.0 = number, 0.0 = NaN.
                                T.vcmp(pf, pf, isnum, "eq")
                                T.vcast(isnum, nmask, round_mode="rint")

                                T.vbrc(neg_inf, acc)
                                T.vbrc(one_f, nacc)
                                for tt in range(kernel_w):
                                    rr = r0 + tt * p_dd
                                    T.vmax(pf[rr : rr + p_bl, 0:p_ch], acc, acc)
                                    T.vmin(nmask[rr : rr + p_bl, 0:p_ch], nacc, nacc)
                                # A NaN anywhere in the window wins (IEEE max
                                # would silently drop it).
                                T.vcmp(nacc, one_f, cf, "lt")
                                # vselect takes no sliced operand: whole buffers.
                                T.vselect(cf, nanb, acc, out32)
                                bl_real = T.min(p_bl, out_l - ow0)
                                if is_f32:
                                    T.copy(out32, acc16)  # same-dtype vcast trap
                                else:
                                    T.vcast(out32, acc16, round_mode="rint")
                                T.transpose(acc16, st, permutation=[1, 0])
                                T.copy(
                                    st[0:real_c, 0:bl_real],
                                    out[nb, cb * p_ch : cb * p_ch + real_c, ow0 : ow0 + bl_real],
                                )

                return main

            try:
                _build.built_path = "phase"
                return _func_phase()
            except Exception as exc:  # noqa: BLE001
                # Build-time feasibility guard, not a performance selector: the
                # UB accounting is a proxy, so if the backend still rejects this
                # tile the shape must fall back to the general grid (both paths
                # are bit-exact) instead of failing to build.
                print(
                    "[_max_pool1d_kernel] phase path rejected by the backend for "
                    f"this shape, using the general grid: "
                    f"{str(exc).splitlines()[0][:160]}"
                )

        if path == "identity":
            # C4b: the window is a single element and out == in, so out == x
            # (pure UB transit, same dtype, NaN preserved).
            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_identity(ch_):
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, l_in), dtype),
                    out: T.Tensor((n, c_in, out_l), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        stage = T.alloc_ub((ch, l_in), dtype)
                        for t in T.serial(num_local):
                            uid = t * num_kernels + cid
                            if uid < num_logical:
                                nb = uid // num_cblk
                                cb = uid % num_cblk
                                real_c = T.min(ch, c_in - cb * ch)
                                T.copy(
                                    x[nb, cb * ch : cb * ch + real_c, 0:l_in],
                                    stage[0:real_c, 0:l_in],
                                )
                                T.copy(
                                    stage[0:real_c, 0:l_in],
                                    out[nb, cb * ch : cb * ch + real_c, 0:l_in],
                                )

                return main

            _build.built_path = "identity"
            return _func_identity(ch)

        if path == "global":
            # C4: out_l == 1 -> the window is the whole line; one whole-line max
            # reduction plus an independent eq-mask NaN reduction (never
            # reduce_max's NaN behaviour).
            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_global(ch_):
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, l_in), dtype),
                    out: T.Tensor((n, c_in, out_l), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        flat = T.alloc_ub((ch, l_in), dtype)
                        flat32 = T.alloc_ub((ch, l_in), "float32")
                        gmax = T.alloc_ub((ch, 1), "float32")
                        nsum = T.alloc_ub((ch, 1), "float32")
                        okbuf = T.alloc_ub((ch, 1), "bool")
                        T.alloc_ub((ch, 1), "float32")
                        infbuf = T.alloc_ub((ch, 1), "float32")
                        nanbuf = T.alloc_ub((ch, 1), "float32")
                        T.alloc_ub((ch, 1), "bool")
                        out32 = T.alloc_ub((ch, 1), "float32")
                        gout = T.alloc_ub((ch, 1), dtype)
                        pos_inf = T.infinity("float32")
                        for t in T.serial(num_local):
                            uid = t * num_kernels + cid
                            if uid < num_logical:
                                nb = uid // num_cblk
                                cb = uid % num_cblk
                                real_c = T.min(ch, c_in - cb * ch)
                                T.copy(
                                    x[nb, cb * ch : cb * ch + real_c, 0:l_in],
                                    flat[0:real_c, 0:l_in],
                                )
                                if is_f32:
                                    T.copy(flat, flat32)
                                else:
                                    T.vcast(flat, flat32, round_mode="rint")
                                # Whole-line maximum (clear=True: the reduction
                                # target must not accumulate stale data).
                                T.reduce_max(flat32, gmax, dim=1, clear=True)
                                # Stage 4 round 1 (R1b): equal-cost NaN tracking.
                                # has_nan(line) <=> isnan(sum(x*x)): x*x is NaN
                                # only for NaN inputs and is otherwise >= 0, so
                                # the sum can never fabricate a NaN (+inf - inf
                                # cannot arise) -- PL-1.22 (8).  The square MUST
                                # be in place (flat32 is dead after reduce_max):
                                # that keeps the BishengHIR reduction-chain UB
                                # price at the in-place-square band (32 vs
                                # 96-128 B/plane element).
                                T.vmul(flat32, flat32, flat32)
                                T.reduce_sum(flat32, nsum, dim=1, clear=True)
                                # ok = not-NaN (ordered eq is IEEE-correct).
                                T.vcmp(nsum, nsum, okbuf, "eq")
                                # NaN = inf - inf (IEEE, built in fp32 because
                                # vsub / reduce_max have no bf16 form).
                                T.vbrc(pos_inf, infbuf)
                                T.vsub(infbuf, infbuf, nanbuf)
                                T.vselect(okbuf, gmax, nanbuf, out32)
                                if is_f32:
                                    T.copy(out32, gout)
                                else:
                                    T.vcast(out32, gout, round_mode="rint")
                                T.copy(
                                    gout[0:real_c, 0:1],
                                    out[nb, cb * ch : cb * ch + real_c, 0:1],
                                )

                return main

            _build.built_path = "global"
            return _func_global(ch)

        # ---------------- general grid ----------------
        @tilelang.jit(out_idx=[1], target="npuir")
        def _func_general(ch_):
            @T.prim_func
            def main(
                x: T.Tensor((n, c_in, l_in), dtype),
                out: T.Tensor((n, c_in, out_l), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # Vector accumulators on the channel axis (vector axis = C).
                    acc = T.alloc_ub((bl, ch), "float32")
                    flag = T.alloc_ub((bl, ch), "float32")
                    zeros = T.alloc_ub((bl, ch), "float32")
                    infbuf = T.alloc_ub((bl, ch), "float32")
                    nanbuf = T.alloc_ub((bl, ch), "float32")
                    out32 = T.alloc_ub((bl, ch), "float32")
                    cfinal = T.alloc_ub((bl, ch), "bool")
                    # Contiguous span staging: NCL span -> transposed (C innermost).
                    row_src = T.alloc_ub((ch, lspan), dtype)
                    row_dst = T.alloc_ub((lspan, ch), dtype)
                    row_f32 = T.alloc_ub((lspan, ch), "float32")
                    eq_mask = T.alloc_ub((lspan, ch), "bool")
                    eq_f32 = T.alloc_ub((lspan, ch), "float32")
                    acc16 = T.alloc_ub((bl, ch), dtype)
                    acc_t = T.alloc_ub((ch, bl), dtype)

                    # max identity is -inf (not 0); the NaN-free flag
                    # accumulator starts at 1 (0 is the absorbing value).
                    neg_inf = -T.infinity("float32")
                    pos_inf = T.infinity("float32")
                    one_val = T.float32(1.0)

                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // (num_oblk * num_cblk)
                            rem = uid % (num_oblk * num_cblk)
                            oblk = rem // num_cblk
                            cb = rem % num_cblk
                            real_c = T.min(ch, c_in - cb * ch)
                            ow0 = oblk * bl
                            # Clamped contiguous span (DESIGN.md 1.6.1 C1):
                            # the start is a runtime value but always in bounds
                            # (0 <= s0 and s0 + LSPAN <= L_in) and the extent is
                            # the static LSPAN, so no dynamic-offset subview is
                            # formed.
                            base = ow0 * stride_w - pad_w
                            s0 = T.max(0, T.min(base, l_in - lspan))
                            T.vbrc(neg_inf, acc)
                            T.vbrc(one_val, flag)
                            T.clear(zeros)
                            T.copy(
                                x[nb, cb * ch : cb * ch + real_c, s0 : s0 + lspan],
                                row_src[0:real_c, 0:lspan],
                            )
                            T.transpose(row_src, row_dst, permutation=[1, 0])
                            if is_f32:
                                T.copy(row_dst, row_f32)
                            else:
                                T.vcast(row_dst, row_f32, round_mode="rint")
                            # Block-level NaN probe materialised as an fp32
                            # {0, 1} mask: x == x is 1 for numbers and 0 for NaN
                            # (the "ne" mode is an ordered not-equal, always
                            # false for NaN).
                            T.vcmp(row_f32, row_f32, eq_mask, "eq")
                            T.vcast(eq_mask, eq_f32, round_mode="rint")
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
                                        T.vmul(
                                            flag[owl : owl + 1, 0:ch],
                                            eq_f32[j : j + 1, 0:ch],
                                            flag[owl : owl + 1, 0:ch],
                                        )
                            # has_nan ? NaN : max (per channel lane); the flag
                            # accumulator reaches 0 iff some element was NaN.
                            T.vcmp(flag, zeros, cfinal, "eq")
                            T.vbrc(pos_inf, infbuf)
                            T.vsub(infbuf, infbuf, nanbuf)
                            T.vselect(cfinal, nanbuf, acc, out32)
                            if is_f32:
                                T.copy(out32, acc16)
                            else:
                                T.vcast(out32, acc16, round_mode="rint")
                            bl_real = T.min(bl, out_l - ow0)
                            T.transpose(acc16, acc_t, permutation=[1, 0])
                            T.copy(
                                acc_t[0:real_c, 0:bl_real],
                                out[nb, cb * ch : cb * ch + real_c, ow0 : ow0 + bl_real],
                            )

            return main

        _build.built_path = "general"
        return _func_general(ch)

    return _build


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_KERNEL_CACHE = {}

# Stage 4 round 4: the path each cached kernel was *actually* built with.  A
# specialisation whose tile the backend rejects falls back to the general grid
# (still bit-exact), so the plan alone cannot prove which structure ran; every
# tiered case therefore asserts plan["path"] == built path.
_BUILT_PATH = {}
_LAST_BUILT_PATH = None

# Per-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


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
        factory = _max_pool1d_kernel(
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
        _BUILT_PATH[key] = getattr(factory, "built_path", None)
    global _LAST_BUILT_PATH
    _LAST_BUILT_PATH = _BUILT_PATH.get(key)
    return _KERNEL_CACHE[key]


def _compare(out, ref, atol, rtol, equal_nan=False):
    """Return (max_diff, violation_count) between out and ref.

    NaN/Inf aware: positions where both are NaN count as equal when
    ``equal_nan`` is set; equal infinities count as equal.
    """
    eq = out == ref
    if equal_nan:
        eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    max_diff = diff.max().item()
    violations = int((diff > (atol + rtol * ref.float().abs())).sum().item())
    return max_diff, violations


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
    expect_path=None,
):
    """Run one kernel case and compare against the golden.

    ``x`` may be pre-built as a CPU tensor (special-value cases); by default
    randn data is generated on CPU, cast, and moved to NPU (device-independent
    data generation).  ``expect_path`` asserts which compile-time branch the
    tiling plan selects.  Raises AssertionError on tolerance or path violation.
    """
    torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
    k_w, s_w, p_w, d_w = _normalise(kernel_size, stride, padding, dilation)
    out_l = pool_output_dim(l_in, k_w, s_w, p_w, ceil_mode, d_w)

    plan = _plan(n, c_in, l_in, k_w, s_w, p_w, d_w, bool(ceil_mode), dtype_str)
    if expect_path is not None:
        assert plan["path"] == expect_path, (
            f"{tag}: expected the {expect_path} path, plan selected "
            f"{plan['path']} (CH={plan['ch']}, BL={plan['bl']}, "
            f"num_logical={plan['num_logical']})"
        )

    if x is None:
        x = torch.randn(n, c_in, l_in, dtype=torch.float32).to(torch_dtype)
    else:
        assert x.dtype == torch_dtype, f"input dtype {x.dtype} != expected {torch_dtype}"
        assert tuple(x.shape) == (n, c_in, l_in), (
            f"input shape {tuple(x.shape)} != {(n, c_in, l_in)}"
        )

    kernel = _get_kernel(n, c_in, l_in, k_w, s_w, p_w, d_w, bool(ceil_mode), dtype_str)
    if expect_path is not None and expect_path != _LAST_BUILT_PATH:
        raise AssertionError(
            f"{tag}: plan selected the {expect_path} path but the kernel was "
            f"built with {_LAST_BUILT_PATH} (backend rejected the "
            f"{expect_path} tile)"
        )
    out = kernel(x.npu()).cpu()

    ref = golden_max_pool1d(x, k_w, s_w, p_w, d_w, bool(ceil_mode))
    assert tuple(out.shape) == (n, c_in, out_l), (
        f"{tag}: output shape {tuple(out.shape)} != {(n, c_in, out_l)}"
    )

    max_diff, violations = _compare(out, ref, atol, rtol, equal_nan=equal_nan)
    if max_diff == max_diff:  # NaN guard (all-equal cases give 0.0)
        best = _STATS_MAX_DIFF.get(dtype_str, -1.0)
        if max_diff > best:
            _STATS_MAX_DIFF[dtype_str] = max_diff

    try:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    except AssertionError:
        print(
            f"[{tag}] FAIL: shape=({n},{c_in},{l_in}) out={out_l} "
            f"k={kernel_size} s={stride} p={padding} d={dilation} "
            f"ceil={ceil_mode} dtype={dtype_str} path={plan['path']} "
            f"CH={plan['ch']} BL={plan['bl']} max_diff={max_diff:.3e} "
            f"violations={violations}/{out.numel()}"
        )
        raise
    if plan["path"] == "phase":
        tile = (
            f"CH={plan['phase']['ch']} BL={plan['phase']['bl']} "
            f"PP={plan['phase']['pp']} rows={plan['phase']['rows']} "
            f"units={plan['phase']['num_logical']}"
        )
    elif plan["path"] == "group":
        tile = (
            f"CH={plan['group']['ch']} BL={plan['group']['bl']} "
            f"units={plan['group']['num_logical']}"
        )
    else:
        tile = f"CH={plan['ch']} BL={plan['bl']}"
    print(
        f"[{tag}] PASS: shape=({n},{c_in},{l_in}) out={out_l} "
        f"dtype={dtype_str} path={plan['path']} {tile} "
        f"max_diff={max_diff:.3e}"
    )
    return max_diff, violations


def _try_case(failures, case_id, **kwargs):
    """Run ``_run_case``, collecting (not re-raising) per-case failures.

    AssertionError is a precision (or path) failure; any other exception is
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
        _max_pool1d_kernel(**kwargs)
    except ValueError as exc:
        print(f"[{case_id}] PASS: rejected ({exc})")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} WRONG-ERROR: {exc!r}")
    else:
        failures.append((case_id, AssertionError("invalid arguments accepted")))
        print(f"[collected] {case_id} FAILED: invalid arguments were accepted")


def _nan_window_x(n=1, c_in=8, l_in=13):
    """NaN injection for the *windowed* semantics: NaNs at the first position,
    mid, a full channel line and the last position (every window touching them
    must yield NaN)."""
    x = torch.randn(n, c_in, l_in, dtype=torch.float32)
    x[0, 0, 0] = float("nan")  # leading position
    x[0, 1, 5] = float("nan")  # interior
    x[0, 2, :] = float("nan")  # full channel line
    x[0, 3, l_in - 1] = float("nan")  # trailing position
    return x


# ---------------------------------------------------------------------------
# L0 (gate suite -- DESIGN.md section 8.3)
# ---------------------------------------------------------------------------
def run_L0():
    """L0 gate: manifest workloads across all three dtypes, the frozen
    degenerate specialisations (including their dilation/ceil guards and the
    capacity routing) and the max-specific NaN / +-inf semantics."""
    failures = []

    cases = [
        # n, c, l, kernel, stride, padding, dilation, ceil, dtype, tag, path
        # -- L0-1..L0-2: manifest smoke (3 dtypes) --
        (1, 8, 32, 3, 2, 1, 1, False, "float32", "L0-1/smoke-fp32", "general"),
        (2, 8, 64, 3, 2, 1, 1, False, "float16", "L0-2a/smoke-fp16", "general"),
        (2, 8, 64, 3, 2, 1, 1, False, "bfloat16", "L0-2b/smoke-bf16", "general"),
        # -- L0-3..L0-5: manifest workloads --
        (32, 80, 4096, 3, 3, 0, 1, False, "float16", "L0-3/sincnet", "group"),
        (64, 256, 128, 128, 1, 0, 1, False, "float16", "L0-4/textcnn-global", "global"),
        (16, 128, 2048, 5, 2, 2, 2, True, "bfloat16", "L0-5/ecg-dilated", "phase"),
        # -- L0-6..L0-7: stride=None + dilation, non-overlapping + ceil --
        (1, 4, 63, 3, None, 1, 2, False, "float16", "L0-6/stride-none-d2", "general"),
        (1, 4, 97, 5, 3, 2, 1, True, "float16", "L0-7a/k5s3p2-ceil-fp16", "general"),
        (1, 4, 97, 5, 3, 2, 1, True, "bfloat16", "L0-7b/k5s3p2-ceil-bf16", "general"),
        # -- L0-11c: ceil_mode step-back --
        (1, 8, 8, 3, 4, 0, 1, True, "float16", "L0-11c/ceil-step-back", "general"),
        # -- L0-11a3: capacity-pressed general path (BL < BL_MAX) --
        (1, 8, 2048, 1024, 1, 0, 1, False, "float16", "L0-11a3/long-kernel", "general"),
        # -- L0-11b: C4 dilation guard (out==1 but d>1 -> general) --
        (1, 8, 5, 2, 1, 0, 4, False, "float16", "L0-11b/d4-guard", "general"),
        (1, 8, 7, 3, 1, 0, 3, False, "float16", "L0-11b2/d3-guard", "general"),
        # -- L0-12a: C4b identity --
        (1, 8, 32, 1, 1, 0, 1, False, "float16", "L0-12a/identity", "identity"),
        # -- L0-12b: C4b capacity fallback -> general (the identity tile does not
        # fit UB at L_in=16384; the group path also declines because ch=8 fp16 is
        # 16 B < the 32 B reduce-lane floor) --
        (1, 8, 16384, 1, 1, 0, 1, False, "float16", "L0-12b/identity-fallback", "general"),
        # -- L0-13: channel-block tails (C=96 -> CH=64 + real_c=32; C=4 -> CH=8) --
        (3, 96, 64, 3, 2, 1, 1, False, "float16", "L0-13a/cblk-tail-96", "general"),
        (1, 4, 14, 3, 3, 1, 1, False, "float16", "L0-13b/cblk-tail-4", "general"),
        # -- L0-18: phase-decimated path (s_w == 2, p_w / d_w even, l_in even) --
        # k3/d2 (dd=1), plain sliding window in the phase plane.
        (2, 16, 64, 3, 2, 0, 2, False, "float16", "L0-18a/phase-k3d2", "phase"),
        # k5/d2 with p=2 and ceil_mode: out-of-range taps on BOTH ends land on
        # the -inf armed pad rows; out_l=63 with bl=48 -> partial last block.
        (2, 24, 128, 5, 2, 2, 2, True, "bfloat16", "L0-18b/phase-k5d2-ceil", "phase"),
        # d4 (dd=2) in float32: the same-dtype vcast trap -> T.copy branch.
        (2, 8, 96, 3, 2, 0, 4, False, "float32", "L0-18c/phase-k3d4-fp32", "phase"),
        # k1: tap_span = 0, the window is one phase position.
        (1, 16, 64, 1, 2, 0, 2, False, "float16", "L0-18d/phase-k1", "phase"),
        # C=20 -> ch=24 (single block with 4 padding lanes): channel tail.
        (2, 20, 80, 2, 2, 0, 2, False, "bfloat16", "L0-18e/phase-cblk-tail-20", "phase"),
        # Same geometry as the manifest ecg workload but fp16 and half the length.
        (2, 16, 128, 5, 2, 2, 2, True, "float16", "L0-18f/phase-k5d2-ceil-fp16", "phase"),
        # -- L0-18g..i: algebraic guards, each MUST fall back to the general grid --
        # odd l_in: the even phase would not cover the last position.
        (1, 8, 23, 4, 2, 2, 2, False, "float32", "L0-18g/guard-odd-l", "general"),
        # odd p_w: taps alternate between the two phases.
        (1, 8, 32, 3, 2, 1, 2, False, "float16", "L0-18h/guard-odd-p", "general"),
        # odd d_w: taps alternate between the two phases.
        (1, 8, 32, 3, 2, 0, 1, False, "float16", "L0-18i/guard-odd-d", "general"),
        # -- L0-19: group-reduce path (s_w == k_w, d_w == 1, p_w == 0) --
        (2, 16, 96, 3, 3, 0, 1, False, "float16", "L0-19a/group-k3s3", "group"),
        (2, 16, 64, 2, 2, 0, 1, True, "float16", "L0-19b/group-k2s2-ceil", "group"),
        (2, 8, 128, 4, 4, 0, 1, False, "float32", "L0-19c/group-k4s4-fp32", "group"),
        # bf16 declines the group path (vreduce has no bf16 form) -> general.
        (2, 24, 160, 5, 5, 0, 1, False, "bfloat16", "L0-19d/group-k5s5-bf16", "general"),
        # same geometry in fp16 to keep the k=5 group case covered (C=32 so the
        # channel block is 64 B, a multiple of the 32 B reduce-lane granule).
        (2, 32, 160, 5, 5, 0, 1, False, "float16", "L0-19d2/group-k5s5-fp16", "group"),
        # C=20 has no exact channel divisor, so the group path deliberately
        # declines (a partial c-block faults inside its 3-D reduce) and the
        # shape keeps the general grid; C=24 covers the single-block case above.
        (2, 20, 96, 3, 3, 0, 1, False, "float16", "L0-19e/group-cblk-tail-20", "general"),
        # out_l=50 with bl=48 -> a 2-output partial tier (bl_real < bl).
        (2, 16, 150, 3, 3, 0, 1, False, "float16", "L0-19g/group-partial-tier", "group"),
        # ceil overrun: k*out_l = 102 > l_in = 100 -> MUST fall back to general.
        (1, 8, 100, 3, 3, 0, 1, True, "float16", "L0-19h/guard-ceil-overrun", "general"),
        # s != k -> not a group reduce (disjoint but not contiguous).
        (1, 8, 96, 3, 4, 0, 1, False, "float16", "L0-19i/guard-s-ne-k", "general"),
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
        path,
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
            expect_path=path,
        )

    # L0-11a: C4 whole-line specialisation on the manifest shape.
    _try_case(
        failures,
        "L0-11a/c4-128",
        n=1,
        c_in=8,
        l_in=128,
        kernel_size=128,
        stride=1,
        padding=0,
        dilation=1,
        ceil_mode=False,
        dtype_str="float16",
        tag="L0-11a",
        expect_path="global",
    )
    _try_case(
        failures,
        "L0-11a2/c4-fp32",
        n=1,
        c_in=8,
        l_in=4,
        kernel_size=4,
        stride=1,
        padding=0,
        dilation=1,
        ceil_mode=False,
        dtype_str="float32",
        tag="L0-11a2",
        expect_path="global",
    )

    # L0-11a4: an oversized C4 candidate falls back by capacity.  For a
    # whole-line window the general-grid span equals L_in, and the general path
    # costs more per element than C4 (2*eb+9 vs eb+9), so such a shape has no
    # UB-feasible plan at all -> the factory rejects it (documented in
    # RETROSPECTIVE; never reached by any manifest workload).
    case_id = "L0-11a4/oversized-c4-rejected"
    try:
        gl_cond = _degenerate_conditions(
            pool_output_dim(4096, 4096, 1, 0, False, 1), 4096, 4096, 1, 0, 1
        )[1]
        assert gl_cond, "test premise: the shape must satisfy the C4 condition"
        _expect_value_error(
            failures,
            case_id,
            dict(
                n=1,
                c_in=8,
                l_in=4096,
                kernel_w=4096,
                stride_w=1,
                pad_w=0,
                dilation_w=1,
                ceil_mode=False,
                dtype="float16",
            ),
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")

    # L0-11b3: the d>1 out==1 configuration must return the window max, not the
    # whole-line max (bit-exact check).
    case_id = "L0-11b3/d4-bit-exact"
    try:
        x = torch.zeros(1, 8, 5, dtype=torch.float16)
        x[0, :, 2] = 1.0  # the only nonzero lies outside the {0,4} window
        _run_case(
            n=1,
            c_in=8,
            l_in=5,
            kernel_size=2,
            stride=1,
            padding=0,
            dilation=4,
            ceil_mode=False,
            dtype_str="float16",
            tag="L0-11b3",
            x=x,
            expect_path="general",
        )
        out = _get_kernel(1, 8, 5, 2, 1, 0, 4, False, "float16")(x.npu()).cpu()
        assert bool((out == 0.0).all()), (
            f"d>1 out=1 config must return 0.0, got {out.flatten()[:4].tolist()}"
        )
        print("[L0-11b3] PASS: d=4 out=1 returns 0.0 (whole-line reduction correctly excluded)")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-8: NaN semantics -- a NaN anywhere in the window wins.
    case_id = "L0-8/nan-window"
    try:
        x = _nan_window_x().to(torch.float16)
        _run_case(
            n=1,
            c_in=8,
            l_in=13,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            ceil_mode=False,
            dtype_str="float16",
            tag="L0-8",
            x=x,
            equal_nan=True,
            expect_path="general",
        )
        out = _get_kernel(1, 8, 13, 3, 2, 1, 1, False, "float16")(x.npu()).cpu()
        # window(ow=0) = iw {-1,0,1} contains the leading NaN.
        assert bool(torch.isnan(out[0, 0, 0])), (
            f"leading NaN did not win: out[0,0,0]={out[0, 0, 0]}"
        )
        # window(ow=3) = iw {5,6,7} contains the interior NaN at 5.
        assert bool(torch.isnan(out[0, 1, 3])), (
            f"interior NaN did not win: out[0,1,3]={out[0, 1, 3]}"
        )
        # window(ow=6) = iw {11,12,13}: no NaN in range (13 out of bounds).
        assert not bool(torch.isnan(out[0, 0, 6])), "NaN-free window produced NaN"
        print("[L0-8] PASS: NaN wins in-window; NaN-free window stays finite")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-9: +-inf semantics -- {+inf, -inf} in one window gives +inf (never
    # NaN); an all -inf window gives -inf.  equal_nan=False so a spurious NaN
    # output is caught.
    case_id = "L0-9/+-inf-semantics"
    try:
        x = torch.full((1, 8, 13), float("-inf"), dtype=torch.float32)
        x[0, 1, 0] = float("inf")  # ch1: one +inf among -inf
        x[0, 2, :] = float("inf")  # ch2: all +inf
        x = x.to(torch.float16)
        _run_case(
            n=1,
            c_in=8,
            l_in=13,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            ceil_mode=False,
            dtype_str="float16",
            tag="L0-9",
            x=x,
            equal_nan=False,
            expect_path="general",
        )
        out = _get_kernel(1, 8, 13, 3, 2, 1, 1, False, "float16")(x.npu()).cpu()
        assert out[0, 1, 0].item() == float("inf"), (
            f"{{+inf,-inf}} window gave {out[0, 1, 0]} (expected +inf)"
        )
        assert out[0, 1, 6].item() == float("-inf"), (
            f"all -inf window gave {out[0, 1, 6]} (expected -inf)"
        )
        assert out[0, 2, 4].item() == float("inf"), (
            f"all +inf window gave {out[0, 2, 4]} (expected +inf)"
        )
        print("[L0-9] PASS: {+inf,-inf} -> +inf; all -inf -> -inf; no spurious NaN")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-18j: NaN and -inf padding semantics on the phase path (s_w == 2).
    case_id = "L0-18j/phase-nan"
    try:
        x = torch.randn(1, 8, 64, dtype=torch.float32)
        x[0, 0, 4] = float("nan")  # even index -> phase q=2 (interior NaN)
        x[0, 1, 60] = float("nan")  # phase q=30, last output block
        x[0, 2, 0] = float("nan")  # phase q=0, first window
        x[0, 3, :] = float("inf")  # all +inf line stays +inf, never NaN
        x = x.to(torch.float16)
        _run_case(
            n=1,
            c_in=8,
            l_in=64,
            kernel_size=3,
            stride=2,
            padding=0,
            dilation=2,
            ceil_mode=False,
            dtype_str="float16",
            tag="L0-18j",
            x=x,
            equal_nan=True,
            expect_path="phase",
        )
        out = _get_kernel(1, 8, 64, 3, 2, 0, 2, False, "float16")(x.npu()).cpu()
        # window(ow) covers iw {2*ow, 2*ow+2, 2*ow+4}; q = iw/2.
        assert bool(torch.isnan(out[0, 0, 0])) and bool(torch.isnan(out[0, 0, 2])), (
            "interior NaN did not win on the phase path: "
            f"out[0,0,0]={out[0, 0, 0]} out[0,0,2]={out[0, 0, 2]}"
        )
        assert bool(torch.isnan(out[0, 1, 29])), (
            f"last-block NaN did not win: out[0,1,29]={out[0, 1, 29]}"
        )
        assert bool(torch.isnan(out[0, 2, 0])), (
            f"leading NaN did not win: out[0,2,0]={out[0, 2, 0]}"
        )
        assert out[0, 3, 7].item() == float("inf"), (
            f"all +inf line gave {out[0, 3, 7]} (expected +inf)"
        )
        assert not bool(torch.isnan(out[0, 4:8]).any()), "NaN leaked into a NaN-free channel"
        print("[L0-18j] PASS: phase-path NaN wins, +inf line stays +inf")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-19j: NaN semantics on the group-reduce path (s_w == k_w).
    case_id = "L0-19j/group-nan"
    try:
        x = torch.randn(2, 16, 96, dtype=torch.float32)
        x[0, 0, 4] = float("nan")  # inside group ow=1
        x[0, 1, 90] = float("nan")  # last group (ow=30)
        x[0, 2, 0] = float("nan")  # first group (ow=0)
        x[0, 3, :] = float("inf")  # all +inf line stays +inf
        x[0, 4, :] = float("-inf")  # all -inf line stays -inf
        x = x.to(torch.float16)
        _run_case(
            n=2,
            c_in=16,
            l_in=96,
            kernel_size=3,
            stride=3,
            padding=0,
            dilation=1,
            ceil_mode=False,
            dtype_str="float16",
            tag="L0-19j",
            x=x,
            equal_nan=True,
            expect_path="group",
        )
        out = _get_kernel(2, 16, 96, 3, 3, 0, 1, False, "float16")(x.npu()).cpu()
        assert bool(torch.isnan(out[0, 0, 1])), (
            f"in-group NaN did not win: out[0,0,1]={out[0, 0, 1]}"
        )
        assert bool(torch.isnan(out[0, 1, 30])), (
            f"last-group NaN did not win: out[0,1,30]={out[0, 1, 30]}"
        )
        assert bool(torch.isnan(out[0, 2, 0])), (
            f"first-group NaN did not win: out[0,2,0]={out[0, 2, 0]}"
        )
        assert out[0, 3, 5].item() == float("inf"), (
            f"all +inf line gave {out[0, 3, 5]} (expected +inf)"
        )
        assert out[0, 4, 5].item() == float("-inf"), (
            f"all -inf line gave {out[0, 4, 5]} (expected -inf)"
        )
        assert not bool(torch.isnan(out[0, 5:16]).any()), "NaN leaked into a NaN-free channel"
        print("[L0-19j] PASS: group-path NaN wins, +-inf lines stay finite")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-18j2: an all -inf line with p=2 / ceil must stay -inf (the armed pad rows
    # are the only source of -inf here, so a spurious NaN would be a padding bug).
    case_id = "L0-18j2/phase-inf-padding"
    try:
        x = torch.full((1, 8, 64), float("-inf"), dtype=torch.float16)
        _run_case(
            n=1,
            c_in=8,
            l_in=64,
            kernel_size=5,
            stride=2,
            padding=2,
            dilation=2,
            ceil_mode=True,
            dtype_str="float16",
            tag="L0-18j2",
            x=x,
            equal_nan=False,
            expect_path="phase",
        )
        out = _get_kernel(1, 8, 64, 5, 2, 2, 2, True, "float16")(x.npu()).cpu()
        assert bool((out == float("-inf")).all()), (
            "an all -inf line must give -inf everywhere on the phase path"
        )
        print("[L0-18j2] PASS: phase-path -inf padding is neutral (no NaN)")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-10: non-contiguous input -- the Op layer normalises with
    # ``.contiguous()`` before the kernel sees the tensor (DESIGN.md 0.1).
    case_id = "L0-10/non-contiguous"
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
            tag="L0-10",
            x=xc.to(torch.float16),
            expect_path="general",
        )
        print("[L0-10] PASS: non-contiguous view -> .contiguous() -> kernel")
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

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (manifest shapes x dtypes + C4/C4b + guards + NaN/Inf + validation)")
    return failures


# ---------------------------------------------------------------------------
# L1 (full functional coverage -- blocking)
# ---------------------------------------------------------------------------
def run_L1():
    """L1: dtype x shape x path matrix, including the fp32 manifest sizes, the
    degenerate specialisations and NaN on the C4 reduction path."""
    failures = []

    cases = [
        # n, c, l, kernel, stride, padding, dilation, ceil, dtype, tag, path
        (2, 8, 16, 3, 2, 1, 1, False, "bfloat16", "L1/bf16/smoke", "general"),
        # bfloat16 has no vreduce form on this toolchain -> general grid.
        (32, 80, 4096, 3, 3, 0, 1, False, "bfloat16", "L1/bf16/sincnet", "general"),
        (16, 128, 2048, 5, 2, 2, 2, True, "float32", "L1/fp32/ecg-dilated", "phase"),
        (64, 256, 128, 128, 1, 0, 1, False, "float32", "L1/fp32/textcnn-global", "global"),
        (1, 1, 7, 2, 2, 1, 1, False, "float16", "L1/fp16/c-eq-1", "general"),
        (2, 3, 17, 3, 2, 1, 2, False, "bfloat16", "L1/bf16/asym-k-s-p-d", "general"),
        (1, 7, 9, 3, 2, 1, 1, True, "float32", "L1/fp32/odd-ceil", "general"),
        (1, 8, 30, 3, None, 1, 1, False, "float16", "L1/fp16/default-stride", "general"),
        # -- degenerate specialisations --
        (1, 33, 6, 1, 1, 0, 1, False, "float16", "L1/fp16/identity-cblk-tail", "identity"),
        (2, 4, 11, 1, 1, 0, 1, False, "bfloat16", "L1/bf16/identity", "identity"),
        (1, 8, 7, 1, 1, 0, 1, False, "float32", "L1/fp32/identity", "identity"),
        (2, 16, 7, 7, 1, 0, 1, False, "bfloat16", "L1/bf16/global", "global"),
        (1, 8, 8, 8, 1, 0, 1, False, "float32", "L1/fp32/global", "global"),
        (1, 16, 5, 5, 1, 0, 1, False, "float16", "L1/fp16/global", "global"),
        (1, 8, 20, 20, 1, 0, 1, False, "float16", "L1/fp16/global-20", "global"),
        # -- disjoint windows (s == k) and a stride-1 overlap --
        (2, 16, 14, 2, 2, 0, 1, False, "float16", "L1/fp16/disjoint", "general"),
        (1, 8, 13, 3, 1, 1, 1, False, "float16", "L1/fp16/overlap-s1", "general"),
        # -- padding>0 and dilation>0 combinations --
        (1, 6, 15, 5, 3, 2, 1, False, "float16", "L1/fp16/pad-only", "general"),
        (1, 8, 19, 3, 2, 1, 3, True, "bfloat16", "L1/bf16/dilation-ceil", "general"),
        (1, 8, 23, 4, 2, 2, 2, False, "float32", "L1/fp32/pad-dilation", "general"),
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
        path,
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
            expect_path=path,
        )

    # NaN / +-inf on the C4 whole-line reduction path (blocking: the C4 NaN
    # judgement is an independent eq-mask reduction, never reduce_max's NaN
    # behaviour).
    case_id = "L1/fp16/global-nan-inf"
    try:
        x = torch.randn(2, 16, 5, dtype=torch.float32)
        x[0, 0, :] = float("inf")
        x[0, 1, :] = float("nan")
        x[1, 3, 0] = float("-inf")
        x[1, 3, 2] = float("nan")
        x[1, 5, :] = float("-inf")
        x = x.to(torch.float16)
        _run_case(
            n=2,
            c_in=16,
            l_in=5,
            kernel_size=5,
            stride=1,
            padding=0,
            dilation=1,
            ceil_mode=False,
            dtype_str="float16",
            tag="L1",
            x=x,
            equal_nan=True,
            expect_path="global",
        )
        out = _get_kernel(2, 16, 5, 5, 1, 0, 1, False, "float16")(x.npu()).cpu()
        assert out[0, 0, 0].item() == float("inf"), f"+inf line gave {out[0, 0, 0]} (expected +inf)"
        assert bool(torch.isnan(out[0, 1, 0])), "all-NaN line did not give NaN"
        assert bool(torch.isnan(out[1, 3, 0])), "NaN-in-line did not win"
        assert out[1, 5, 0].item() == float("-inf"), (
            f"all -inf line gave {out[1, 5, 0]} (expected -inf)"
        )
        print("[L1] PASS: C4 NaN/Inf semantics (inf line -> inf, NaN wins)")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # NaN semantics on a non-degenerate window in bf16 and fp32.
    for dtype_str in ("bfloat16", "float32"):
        case_id = f"L1/{dtype_str}/nan-window"
        try:
            tdt = _TOLERANCE[dtype_str][0]
            x = _nan_window_x().to(tdt)
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
                x=x,
                equal_nan=True,
                expect_path="general",
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
        kernel = _max_pool1d_kernel(1, 8, 32, 3, 2, 1, 1, False, "float16")(256)
        xx = torch.randn(1, 8, 32, dtype=torch.float16)
        out = kernel(xx.npu()).cpu()
        torch.testing.assert_close(
            out,
            golden_max_pool1d(xx, 3, 2, 1, 1, False),
            rtol=0.0,
            atol=0.0,
        )
        print("[L1] PASS: block_m=256 dispatch (wrapper contract)")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # block_m=0 is rejected.
    try:
        _max_pool1d_kernel(1, 8, 32, 3, 2, 1, 1, False, "float16")(0)
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
        (1, 2, 37, 5, 5, 2, 1, False, "float16"),
        (2, 7, 13, 3, 2, 1, 1, True, "bfloat16"),
        (3, 1024, 3, 1, 1, 0, 1, False, "float16"),
        (1, 16, 512, 3, 2, 1, 1, False, "float32"),
        (1, 8, 1024, 2, 2, 0, 1, False, "float16"),
        (1, 8, 64, 32, 32, 0, 1, False, "float16"),
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
                f"k={kernel_size} dtype={dtype_str}: {exc}"
            )


# ---------------------------------------------------------------------------
# Boundary (special-value inputs -- warn only, non-blocking)
# ---------------------------------------------------------------------------
def run_boundary():
    """Boundary: zeros / constant / large / subnormal / all-NaN / +-inf only."""
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
        print(f"[Boundary] WARN (record only): zeros: {exc}")

    # Constant line: every window max equals the constant.
    try:
        _run_case(
            tag="Boundary-const", x=torch.full((2, 16, 21), 3.7, dtype=torch.float16), **common
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): const: {exc}")

    # Large magnitudes (max selects, never overflows).
    try:
        _run_case(
            tag="Boundary-large",
            x=(torch.randn(2, 16, 21, dtype=torch.float32) * 1000.0).to(torch.float16),
            **common,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large: {exc}")

    # Subnormal fp16 magnitudes.
    try:
        _run_case(
            tag="Boundary-subnormal",
            x=(torch.randn(2, 16, 21, dtype=torch.float32) * 1e-5).to(torch.float16),
            **common,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {exc}")

    # +-inf without NaN must never produce NaN (equal_nan=False).
    try:
        x = torch.full((2, 16, 21), float("-inf"), dtype=torch.float32)
        x[0, 1, 0] = float("inf")
        x[1, 2, :] = float("inf")
        _run_case(tag="Boundary-pm-inf-only", x=x.to(torch.float16), **common)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): pm-inf-only: {exc}")

    # All-NaN window on the general path.
    try:
        x = torch.randn(2, 16, 21, dtype=torch.float32)
        x[0, 0, :] = float("nan")
        _run_case(tag="Boundary-all-nan-window", x=x.to(torch.float16), equal_nan=True, **common)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): all-nan-window: {exc}")

    # NaN / +-inf on the C4 whole-line path.
    try:
        x = torch.randn(2, 16, 5, dtype=torch.float32)
        x[0, 0, :] = float("inf")
        x[0, 4, :] = float("-inf")
        x[1, 3, 2] = float("nan")
        _run_case(
            n=2,
            c_in=16,
            l_in=5,
            kernel_size=5,
            stride=1,
            padding=0,
            dilation=1,
            ceil_mode=False,
            dtype_str="float16",
            tag="Boundary-inf-nan-global",
            x=x.to(torch.float16),
            equal_nan=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): inf-nan-global: {exc}")

    # NaN under identity (C4b) -- NaN must survive the pure copy.
    try:
        x = torch.randn(1, 8, 6, dtype=torch.float32)
        x[0, 2, 3] = float("nan")
        _run_case(
            n=1,
            c_in=8,
            l_in=6,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            ceil_mode=False,
            dtype_str="float16",
            tag="Boundary-identity-nan",
            x=x.to(torch.float16),
            equal_nan=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): identity-nan: {exc}")


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
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
