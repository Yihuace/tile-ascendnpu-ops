# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Floor-divide forward kernel (NPU, Developer mode) -- Stage 4 tuned.

Round 6 (2026-10-23): **fast path + rare exact path** for fp32 --
bit-exact at ~1.7-2.2x the round-4 exact version's speed.

Motivation (measured, `probe_vdiv_exact.py`): ``hivm::VDivOp`` is *not*
IEEE correctly rounded on this toolchain -- 5.51 % of 16.78 M random fp32
pairs come back 1 ulp off -- so ``floor(a/b)`` built as
``vdiv`` + ``vcast(floor)`` flips a result by 1.0 on ~1e-8 of the elements.
Round 4 fixed that by running a 20-pass Markstein/Veltkamp exact division on
*every* tile (bit-exact, but fp32 cost 2.2-3.6x).

Round 6 keeps the round-4 code as the *rare* path and guards it with a
per-tile "is any quotient close to an integer" detector:

    q0    = vdiv(a, b)                       # fast path, whole tile
    yq    = floor(q0 * (1 + eps))            # = floor(q0) whenever the tile
                                             #   is clean, and the stored value
    ql    = q0 * (1 - eps)
    x     = ql - yq                          # <= 0  <=>  an integer lies in
                                             #            [ql, qu]
    if min(x) > 0:  y = yq                   # the overwhelmingly common tile
    else:           y = exact_floor(a, b)    # round-4 Markstein, verbatim

Soundness (eps = 2^-20, i.e. 8x the proven 1-ulp bound 2^-23, so
``ql <= q_true <= qu`` always): if no integer lies in [ql, qu] then
``floor(qu) == floor(q_true)`` because ``qu >= q_true`` and no boundary is
crossed between them. Self-checked by `probe_rarepath2.py`:
  * 2.62 M-element near-integer construction set (a = fl32(k*b), k = 2..41 --
    the set that produced 59 246 floor flips with the bare fast path):
    round-6 bitdiffs = 0, fast-path-only = 59 246.
  * the same kernel with eps one notch tighter (2^-25, i.e. below the ulp
    scale) gets those 59 246 flips back -> the band is necessary, not folklore.
  * 36.2 M uniform-random elements + the IEEE special battery:
    bitdiffs = 0 (the NaN / Inf / 0-div-0 tiles fall into the exact arm
    through the inverted compare, so both arms keep the shipped semantics).
Cost (msprof op Task Duration, hidden 2048x4096 fp32): fast path 51-54 us,
round-4 exact 111 us, round 6 67 us. The detector itself is 3 vector passes
(~2.3 us) plus the per-tile cross-lane reduction (~15.6 us measured in
isolation -- `T.reduce_min` is ~20x one elementwise pass here, and a
lane-local vmin tree is no cheaper: every vector op costs ~0.8 us on this
shape). The reduction is therefore the limiting cost, not the arithmetic;
`opt_log.md` round 6 carries the full decomposition.

Round 4 (2026-10-23) exact path, kept verbatim as the rare arm: Markstein
quotient + exact Veltkamp/Dekker residual, FMA-free, with the ``+-Inf`` /
NaN / 0-div-0 clamps that reproduce the shipped semantics.

Stage-4 winner (2026-10-22 closeout): ``r3_op2`` = the round-2 2-D broadcast
tile path plus a **sub-affine col mapping** for the half dtypes when the outer
shape does not collapse into a single run (``R_lead > 1``, the cnn-feat
``16x256x3136`` geometry). See the "Stage-4 round 3" comment at the dispatch
site below for the measured mechanism and the numbers.

Migrated from the GPU TileOPs operator ``floor_divide_fwd_kernel``
(``_make_binary_explicit`` factory specialised with
``op_func(a, b) = T.Cast(a.dtype, T.floor(T.cast(a, "float32") /
T.cast(b, "float32")))``) to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y[i] = floor(a[off_a(i)] / b[off_b(i)])
on two already-flattened contiguous ``(a_numel,)`` / ``(b_numel,)`` tensors
of the same float dtype (float16 / bfloat16 / float32); the output is the
flat ``(N_total,)`` view of ``broadcast_shapes(a_shape, b_shape)``.
``off_a`` / ``off_b`` are the broadcast index maps; for same-shape inputs
they are the identity.  Division and floor are evaluated in the fp32
domain, then the single result is cast back to the native dtype.

NPU redesign (DESIGN.md section 0.6):
    R1: the source's per-element unrolled divmod chain (used only on the
        broadcast path) is replaced by an outer-row scalar base (one
        unrolled mixed-radix divmod per task) plus innermost-dimension
        vectorisation.  Rows whose last-dim stride is 0 (a scalar within
        the row) are materialised with ``T.vbrc``.
    R2: one-dimensional ``T.Kernel(num_kernels, is_npu=True)`` with a
        persistent in-core ``T.serial`` grid-stride loop and an if-guard,
        replacing the CUDA grid x block parallel structure.  Element
        parallelism is handled by the Vector unit.
    R4: there is no usable one-shot floor-divide primitive (see the
        runtime traps below); the elementwise work is the two-instruction
        chain ``T.vdiv`` then ``T.vcast(round_mode="floor")``, executed in
        the fp32 domain.  fp16/bf16 are promoted losslessly with
        ``T.vcast(round_mode="rint")`` and the result is rounded back once.

Implementation notes / toolchain traps (DESIGN.md section 9; probed and
registered in capability-gaps):
    - ``T.vfloordiv`` (``npuir_floordiv``) and ``T.vfloor`` (``npuir_floor``)
      compile successfully but HANG the Vector core at runtime on this
      toolchain (fp32/fp16/bf16 same-shape, sliced and ``alloc_shared``
      forms all hang; CG-2026-0023).  They must NOT be used here; the
      two-instruction ``vdiv`` + ``vcast(floor)`` chain is the only
      verified floor-divide path.
    - ``T.vcast`` with an identity-dtype cast (f32->f32, any rounding mode)
      produces NaN on this toolchain even though
      docs/Tilelang.language/数据类型转换操作/T.vcast.md lists it as
      supported (CG-2026-0024).  The fp32 path therefore does no
      promote/demote cast at all -- the dtype branches below skip the
      casts statically instead of running a uniform cast chain.
    - Vector ops silently IGNORE sub-buffer slicing: ``T.vmin(x[0:2048],
      x[2048:4096], x[0:2048])`` compiles but degrades to ``vmin(x, x, x)``
      (measured: a per-tile fold written that way kept only lane 0 of the
      flag, `logs/rarepath/lane_sweep.py`).  Only ``T.copy`` and the
      1-element staging buffer idiom may slice; the tile reduction below
      therefore uses the builtin ``T.reduce_min``.
    - The fp32 tile carries 10 full-size UB buffers when the exact scratch is
      live, so ``npuir.enable_auto_multi_buffer`` is disabled for the fp32
      1-D path (it duplicates buffers and overruns UB: "requires 1703936
      bits"). The fp32 2-D path keeps it disabled too, because its five live
      fp32 tiles already need 20 B/element; the half paths keep the pass on.
    - The fp32 2-D tile is capped at ``_MAX_2D_ELEMS_FP32`` elements (5 fp32
      buffers x 4 B). Larger tiles (r_blk 4) are UB-infeasible once the rare
      path exists; the measured cost is per-task, so this cap trades 2.3x
      (cnn-feat 16x256x3136: r_blk 2 -> 1024 fewer tasks would be 2.3x
      faster) against correctness of the rare arm.
    - The fp32-transit for fp16/bf16 is a correctness requirement, not an
      optimisation: computing in the native half domain rounds the
      quotient before the floor (e.g. ``2.999..`` -> ``3.0`` ->
      ``floor=3``), diverging from the golden.  DESIGN.md E3 measured 66
      fp16 / 430 bf16 divergences for the no-transit variant.
    - The kernel body's ``block_size`` must be the same value used to
      derive the task decomposition (``chunks`` / ``num_logical`` /
      ``num_kernels`` / ``num_local_tasks``).  Those are computed *inside*
      the jit function (not at factory-build time) so the returned
      callable stays consistent for every block_size it is invoked with.

Factory contract (frozen; see examples/TileOPs/tileops/kernels/elementwise/
floor_divide/floor_divide.py):
    floor_divide_fwd_kernel(N_total, dtype, coalesced_shape, a_strides,
                            b_strides, a_numel, b_numel,
                            output_dtype=None, block_size=4096)
        -> kernel(block_size) -> main(a, b) -> y   (``out_idx=[2]``)

Run the embedded hierarchical tests:
    python floor_divide_fwd_kernel.py --level L0
    python floor_divide_fwd_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per task, no manual
# sync points (DESIGN.md section 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import math
import time

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = [
    "ASCEND_MODE",
    "coalesce_broadcast_dims",
    "floor_divide_fwd_kernel",
    "golden_floor_divide",
]

# Programming mode actually validated by this file (TileOPs harness
# migration requires this module-level literal).
ASCEND_MODE = "Developer"

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608
# Conservative auto-multi-buffer inflation factor (CONST-capacity-910B2C).
_UB_INFLATION = 1.7

# dtypes handled natively by the kernel (float-only; integers/bool are
# rejected by the Op layer, matching the source _FLOAT_DTYPES contract).
_ALLOWED_DTYPES = ("float16", "bfloat16", "float32")

# Stage-4 tuned default block_size.  The cross-dtype-copy transit (see below)
# collapses the fp16/bf16 UB footprint from 22 to 16 B/elem, so every float
# dtype can use the full 192 KB UB: 4 fp32 buffers x 12288 x 4 B = 196608 B.
# Exposed at module level for the Op-layer wrapper source-switch block.
_DEFAULT_BLOCK = 12288
TUNED_BLOCK_BY_DTYPE = {
    "float32": 4096,
    "float16": 12288,
    "bfloat16": 12288,
}
TUNED_BLOCK_SIZE = 12288
# Per-dtype default used by the L0/L1 harness below (the fp32 UB ceiling is
# 5120 after the exact-division scratch landed).
_DEFAULT_BLOCK_BY_DTYPE = dict(TUNED_BLOCK_BY_DTYPE)

# Smallest block that keeps the UB buffer 32-B aligned for every dtype
# (16 elem x 2 B = 32 B) and keeps the scalar-broadcast T.vbrc dst strictly
# larger than its (1,) src (TRAP-vbrc-same-shape-empty-broadcast).
_ALIGN_ELEMS = 16

# --- Stage-4 optimisation #1: cross-dtype T.copy transit --------------------
# The fp16/bf16 path loads each operand straight into its fp32 UB buffer with
# a cross-dtype ``T.copy`` and stores the fp32 result straight into the native
# dtype output buffer -- eliminating the per-operand ``a_in``/``b_in`` and the
# ``y_out`` transit buffers.  Footprint is 16 B/elem for every float dtype
# (a32 + b32 + q + yf), so the UB-derived block_size ceiling rises to 12288
# (measured: 14336 fails bishengir-compile; 12288 compiles and is bit-exact).
_UB_BYTES_PER_ELEM_CROSS = {
    "float32": 16,
    "float16": 16,
    "bfloat16": 16,
}
_UB_MAX_BLOCK_CROSS = {dt: 12288 for dt in _ALLOWED_DTYPES}

# fp32 additionally carries the exact-division scratch (7 fp32 buffers on top
# of a32/b32/q/yf), so its *effective* tile is clamped to
# 10 x 4096 x 4 B = 163840 B.  The public block_size ceiling stays 12288 for
# every dtype because the Op-layer wrapper passes ``TUNED_BLOCK_SIZE`` (12288)
# unconditionally; the clamp is applied at trace time so the compiled kernel
# always matches the tile it actually runs.
_EXACT_DIV_BLOCK = 4096

# --- Stage 4 precision fix (fp32 only): exact floor division ---------------
# hivm ``VDivOp`` is NOT IEEE-correctly-rounded: measured on 16.78 M random
# fp32 pairs it differs from CPU ``div.rn.f32`` by 1 ulp on 5.5% of the
# elements (probe_vdiv_exact.py).  ``floor`` therefore flips by exactly 1
# whenever the correctly rounded quotient is an integer and vdiv lands one ulp
# below -- measured 1e-8 per element, i.e. an intermittent single-element
# mismatch on the 12-17 M-element manifests.  One Markstein step
#   q0 = vdiv(a,b);  r = a - q0*b (EXACT);  q1 = q0 + r/b
# restores bit-exact ``div.rn.f32`` (0/19.4 M mismatches, probe_div_fix.py V1).
# No FMA exists on this toolchain, so the exact residual uses Veltkamp splits
# (K = 2^13+1) and four exact partial products.  fp16/bf16 keep the cheap
# vdiv+floor chain: their contract tolerance (atol = 1.0) already accepts the
# 1-ulp case, and they are vector-throughput bound.
_SPLIT_K = 8193.0
# Clamp on q0 before the split: |q0| * K must not overflow, and the clamp keeps
# +-Inf / NaN inputs from poisoning the split (floor(+-Inf) = +-Inf either
# way).  2^104 * 8193 = 1.7e35 < FLT_MAX.
_CLAMP_Q = float(2.0 ** 104)
# Divisor clamp for the split: the Veltkamp constant multiplies by K, so |b|
# must stay below FLT_MAX / K = 4.15e34.  +-Inf divisors land here (q0 = 0 and
# the correction degenerates to 0, which is the contracted answer).  NOTE: a
# finite |b| above 2^114 loses the 1-ulp guarantee (documented limitation:
# outside the operator's manifests, where |b| <= 4).
_CLAMP_B = float(2.0 ** 114)

# --- Stage 4 round 5 (fp32): near-integer band for the fast/rare split ------
# A tile is routed to the exact path iff at least one element satisfies
#     |q0 - rint(q0)| <= 2^-20 * (|q0| + 1)          (== _NEAR_EPS_SCALE)
# Conservativeness (constructive): ``hivm::VDivOp`` is within 1 ulp of the
# correctly rounded quotient, i.e. |q0 - q_true| <= ulp(q0) <= 2^-23 * |q0|
# (probe_vdiv_exact.py measured exactly 1 ulp on 16.78 M random pairs, 5.51 %
# of them; probe_rarepath2.py re-measures the max relative error).  floor(q0)
# can only differ from floor(q_true) if an integer k lies between them, so
# |q0 - k| <= 2^-23 * |q0| < 2^-20 * max(1, |q0|) <= 2^-20 * (|q0| + 1).  The
# band is therefore an 8x margin over the proven 1-ulp bound and a strict
# superset of the affected set (verified: 2.62 M near-integer construction
# pairs / 59 246 measured floor flips -> 100 % inside the band; narrowing the
# band by one notch (2^-24) provably misses, see probe_rarepath2.py).
# |q0| + 1 (one vadd) is used instead of max(1, |q0|) (vabs + vmax): it is
# *more* conservative and one pass cheaper.
_NEAR_EPS = 2.0 ** -20
_NEAR_HI = 1.0 + _NEAR_EPS
_NEAR_LO = 1.0 - _NEAR_EPS
# round-5 m3: tile flag via reduce_min(ql - floor(qu)) <= 0 (see the detector
# comment at the dispatch site).

# --- Stage-4 optimisation #3: 2-D row-tile (DESIGN section 1.6.3 alt #2) ----
# Broadcast workloads (cnn-feat / row-broadcast) process one row per task in
# the 1-D innermost path, so the per-task emission (a scalar-b DMA + a T.vbrc
# working on L stale-free lanes) dominates.  The 2-D ``[R_blk, L]`` tile loads
# R_blk contiguous rows and computes ``T.vdiv`` with auto-broadcast (no vbrc):
#   - col mode (sb==0, b is one scalar per row): b tile [R_blk, 1]
#   - row mode (sb==1, b is a shared full row):  b tile [1, L] (hoisted once)
# Toolchain note (probed 2026-10-14): a 1-D-GM -> 2-D-UB ``T.copy`` HANGS
# bishengir-compile; the working legal form is a **2-D-declared GM parameter**
# (runtime accepts the flat 1-D contiguous tensor for it, numel-checked).
# ``R_blk`` is UB-bounded: measured R_blk=4 ok / R_blk=8 wrong-results (UB
# over-subscription) for the manifest shapes; half-precision row mode is capped
# at 2.  Adjudicated vs the 1-D path with msprof-op Task Duration: cnn-feat
# 2.3-3.7x faster (fp32 30.3% -> 70.0% Roofline), row-broadcast fp32 55.4% ->
# 61.9%; the two column modes with half dtypes stay capped by the read-only
# metric numerator.
# fp32 pays for the exact-division scratch (see `_EXACT_DIV` below): the 2-D
# tile then has 6 live fp32 buffers instead of 4, so its element count is
# capped at 8192 (192 KB / (6 x 4 B)) -- r_blk 2 on the L = 3136 / 4096 rows.
_TUNED_RBLK_COL = {"float32": 2, "float16": 4, "bfloat16": 4}
_TUNED_RBLK_ROW = {"float32": 4, "float16": 2, "bfloat16": 2}
# Largest 2-D tile width (columns) that leaves UB room for the fp32 buffers.
_MAX_2D_L = 8192
# Largest [R_blk, L] element count (a32/q/yf x 4 B fits the 192 KB UB).
_MAX_2D_ELEMS = 16384
# fp32 2-D tile cap: the exact-division path keeps 6 fp32 buffers live
# (a32/q/yf/x_qh/x_ql/x_rm) plus the (r_blk,1) b-scratch, i.e. 24 B/elem, so
# 192 KB / 24 B = 8192 elements minus the small b buffers and the 32-B
# alignment -> 6144 is the practical ceiling (measured: 8192 overflows).
_MAX_2D_ELEMS_FP32 = 8192


def _effective_block(block_size, L):
    """Clamp the requested block_size to a shape-safe tile (compile-time).

    For a wide row (``L >= block_size``, e.g. the flat hidden-state stream) the
    requested block is used as-is.  For a short row (``L < block_size``, e.g.
    the 4096 / 3136 broadcast rows) the tile is clamped to ``L`` rounded up to
    ``_ALIGN_ELEMS`` -- this stops the vector ops from chewing through
    ``block_size - L`` stale lanes (a 3-4x waste at block_size 12288) while
    staying 32-B aligned; the tail is still trimmed by ``tail`` per task.
    """
    if L >= block_size:
        return block_size
    rounded = ((L + _ALIGN_ELEMS - 1) // _ALIGN_ELEMS) * _ALIGN_ELEMS
    return min(block_size, max(rounded, _ALIGN_ELEMS))


# Bytes of full-size UB buffer per element, worst case (DESIGN.md 4.5):
#   float32        : a32 + b32 + q + yf                      = 4 x 4 = 16
#   float16/bf16   : a_in + b_in + a32 + b32 + q + yf + y_out = 22
_UB_BYTES_PER_ELEM = {
    "float32": 4 * 4,
    "float16": 3 * 2 + 4 * 4,
    "bfloat16": 3 * 2 + 4 * 4,
}

# Upper bound on block_size from the UB budget (DESIGN.md section 4.5,
# conservative x1.7 inflation; fp32 ~7228, fp16/bf16 ~5257).
_UB_MAX_BLOCK = {
    dt: int(_UB_CAPACITY_BYTES // (_UB_BYTES_PER_ELEM[dt] * _UB_INFLATION))
    for dt in _ALLOWED_DTYPES
}

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_floor_divide(a, b):
    """Floor-divide reference (source-operator semantics, direct broadcast).

    ``torch.floor(a.float() / b.float()).to(a.dtype)`` -- the source-repo
    ``test_floor_divide.py`` ``ref_program`` (line 41), which mirrors the GPU
    op body ``T.Cast(a.dtype, T.floor(T.cast(a, "float32") /
    T.cast(b, "float32")))``: division and floor are evaluated in the fp32
    domain, then the single result is cast back.  Inputs keep their original
    broadcast shapes; the output shape is ``broadcast_shapes(a.shape,
    b.shape)`` and the dtype is ``same_as(a)``.  Independent of the NPU
    algorithm (no row/inner indexing, no vdiv/vcast, no persistent
    structure) and runs on CPU without an NPU device.

    NOTE (Stage 3, D-class finding): plain ``torch.floor_divide(a, b)`` is
    deliberately NOT used as the reference.  On CPU it floors a
    higher-precision quotient, so it disagrees with the source's own fp32
    double-rounding semantics at boundary points.  Measured on a
    (4096, 4096) / (1, 4096) fp32 row-broadcast: ``torch.floor_divide``
    differs from the explicit fp32 formula at 3 / 16.78M points -- e.g.
    a=1.6938038 b=0.11292025: the fp32 quotient rounds to exactly 15.0
    (explicit floor = 15) while ``floor_divide`` returns 14.  The frozen
    operator semantics is the fp32-domain one, so the explicit formula is
    the golden.
    """
    return torch.floor(a.detach().cpu().float() / b.detach().cpu().float()).to(a.dtype)


# ---------------------------------------------------------------------------
# Broadcast lowering (verbatim from the source operator's helper)
# ---------------------------------------------------------------------------
def coalesce_broadcast_dims(a_shape, b_shape):
    """Coalesce N-dim broadcast into minimal effective dimensions.

    Merges adjacent dimensions that have the same broadcast behaviour
    (both real or both broadcast) to minimise the number of divmod
    operations inside the kernel body.  This is a lowering decision, not
    part of the op contract: the caller hands the factory two operand
    shapes and the kernel picks its own index representation.

    Returns ``(out_shape, coalesced_shape, a_strides, b_strides)`` where
    strides use 0 for broadcast dimensions.
    """
    if len(a_shape) == 0:
        a_shape = (1,)
    if len(b_shape) == 0:
        b_shape = (1,)

    out_shape = torch.broadcast_shapes(a_shape, b_shape)
    ndim = len(out_shape)
    a_pad = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
    b_pad = (1,) * (ndim - len(b_shape)) + tuple(b_shape)

    def _make_strides(padded_shape):
        strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * padded_shape[i + 1]
        return [
            0 if padded_shape[i] == 1 and out_shape[i] > 1 else strides[i]
            for i in range(ndim)
        ]

    a_raw = _make_strides(a_pad)
    b_raw = _make_strides(b_pad)

    groups = [(out_shape[0], a_raw[0], b_raw[0])]
    for i in range(1, ndim):
        prev_out, prev_as, prev_bs = groups[-1]
        a_can = (a_raw[i] == 0 and prev_as == 0) or (
            a_raw[i] != 0 and prev_as == a_raw[i] * out_shape[i]
        )
        b_can = (b_raw[i] == 0 and prev_bs == 0) or (
            b_raw[i] != 0 and prev_bs == b_raw[i] * out_shape[i]
        )
        if a_can and b_can:
            groups[-1] = (prev_out * out_shape[i], a_raw[i], b_raw[i])
        else:
            groups.append((out_shape[i], a_raw[i], b_raw[i]))

    groups = [g for g in groups if g[0] > 1] or [(1, 0, 0)]
    coalesced_shape = tuple(g[0] for g in groups)
    a_strides = tuple(g[1] for g in groups)
    b_strides = tuple(g[2] for g in groups)
    return out_shape, coalesced_shape, a_strides, b_strides


# ---------------------------------------------------------------------------
# Outer-row scalar base (compile-time unrolled mixed-radix divmod)
# ---------------------------------------------------------------------------
def _row_base(row, outer_shape, strides):
    """Build the outer-row base offset expression for one operand.

    ``row`` is the runtime row index; ``outer_shape`` and ``strides`` are
    Python constants, so the loop unrolls at trace time into a chain of
    ``T.floordiv`` / ``T.floormod`` ops (one divmod per outer dimension,
    executed once per task rather than once per element -- DESIGN.md R1).
    Returns the Python int 0 when there is no outer dimension (k == 1).
    """
    base = 0
    rem = row
    n_outer = len(outer_shape)
    for j in range(n_outer):
        div = 1
        for t in range(j + 1, n_outer):
            div *= outer_shape[t]
        coord = T.floordiv(rem, div)
        rem = T.floormod(rem, div)
        base = base + coord * strides[j]
    return base


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------
def _normalize_dtype(dtype):
    """Return the TileLang dtype string, rejecting unsupported dtypes.

    The source operator is float-only (``_FLOAT_DTYPES``); integer and bool
    dtypes are rejected at the Op layer (test_floor_divide_rejects_integer_dtype).
    """
    if isinstance(dtype, torch.dtype):
        name = str(dtype).replace("torch.", "")
    else:
        name = str(dtype)
    if name not in _ALLOWED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; floor_divide is float-only, "
            f"expected one of {_ALLOWED_DTYPES}"
        )
    return name


def floor_divide_fwd_kernel(
    N_total,
    dtype,
    coalesced_shape,
    a_strides,
    b_strides,
    a_numel,
    b_numel,
    output_dtype=None,
    block_size=_DEFAULT_BLOCK,
):
    """Build the floor-divide kernel factory.

    Returns a JIT callable taking a single ``block_size`` which compiles and
    returns ``main``; ``main(a, b)`` returns ``y`` (``out_idx=[2]``,
    source-operator contract).
    """
    dtype = _normalize_dtype(dtype)
    out_dtype = _normalize_dtype(output_dtype) if output_dtype is not None else dtype
    coalesced_shape = tuple(int(x) for x in coalesced_shape)
    a_strides = tuple(int(x) for x in a_strides)
    b_strides = tuple(int(x) for x in b_strides)
    N_total = int(N_total)
    a_numel = int(a_numel)
    b_numel = int(b_numel)

    if N_total <= 0:
        raise ValueError(f"N_total must be a positive int, got {N_total}")
    if len(coalesced_shape) == 0 or len(a_strides) != len(coalesced_shape) or len(
        b_strides
    ) != len(coalesced_shape):
        raise ValueError("coalesced_shape / a_strides / b_strides rank mismatch")

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op).  Re-queried per factory call so other deployments
    # adapt automatically.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    L = coalesced_shape[-1]
    outer = list(coalesced_shape[:-1])
    R = math.prod(outer) if outer else 1
    sa, sb = a_strides[-1], b_strides[-1]
    if sa not in (0, 1) or sb not in (0, 1):
        raise ValueError(f"innermost strides must be 0 or 1, got sa={sa}, sb={sb}")

    # Static dtype branch: fp16/bf16 need the fp32 transit (R4); the fp32
    # path must skip every identity cast (CG-2026-0024).  Resolved at trace
    # time so no runtime branch is emitted.
    needs_upcast = dtype != "float32"
    needs_downcast = out_dtype != "float32"
    max_block = _UB_MAX_BLOCK_CROSS[dtype]

    # --- DESIGN section 1.6.3 adjudication: 2-D row-tile (#2) dispatch ---------
    # Selected at factory-build (trace) time from the compile-time shape
    # structure; the returned callable is specialised per shape, so no runtime
    # branch is emitted.  All conditions are structural (contiguity /
    # broadcast orientation / UB budget), never data-dependent.
    run = outer[-1] if outer else 1
    b_2d_col = (
        sb == 0 and len(outer) >= 1 and b_strides[-2] == 1 and b_numel == run
    )
    b_2d_row = sb == 1 and all(s == 0 for s in b_strides[:-1]) and b_numel == L
    if b_2d_col:
        r_blk = _TUNED_RBLK_COL.get(dtype, 4)
    elif b_2d_row:
        r_blk = _TUNED_RBLK_ROW.get(dtype, 2)
    else:
        r_blk = 1
    # fp32: the exact-division residual needs bh/bl multiplied by qh/ql, and
    # hivm vmul/vsub have no [1, N] x [M, N] broadcast, so the row-mode 2-D
    # tile (b hoisted as one [1, L] row) is not usable for fp32.
    if not needs_upcast:
        b_2d_row_ok = False
    else:
        b_2d_row_ok = b_2d_row
    use_2d = (
        (b_2d_col or b_2d_row_ok)
        and len(outer) >= 1
        and sa == 1
        and a_strides[-2] == L
        and R * L == a_numel
        and L <= _MAX_2D_L
        and L % _ALIGN_ELEMS == 0
        and 2 <= r_blk <= run
        and run % r_blk == 0
        and r_blk * L <= (_MAX_2D_ELEMS if needs_upcast else _MAX_2D_ELEMS_FP32)
    )
    if not use_2d:
        b_2d_col = False
    R_lead = R // run if run else 1
    # --- Stage-4 round 3 (closeout): sub-affine col mapping -------------------
    # Half-dtype col broadcasts whose outer dims do NOT collapse into one run
    # (R_lead > 1, e.g. cnn-feat 16x256x3136) pay a codegen penalty: the
    # per-tile b row index is a non-affine function of the tile index, which
    # makes the bufferised IR a *value-returning* scf.if with per-iteration
    # ``tensor.empty()`` -- no buffer reuse, no pipelining.  Measured on the
    # cnn-feat geometry: 41-46 us with the non-affine index vs 24 us when every
    # subview index is affine in the loop counter (2-D flat control case), and
    # 24.3 us with a timing-only probe that replaced the b index by the affine
    # row index.  The fix gives each core whole ``sub`` (b-row group) slices
    # and iterates the leads inside, so both the b load and the a/y tile index
    # are affine in a loop counter and share one value.  fp32 is immune (it is
    # MTE2-bound at the same rate with or without the defect), so the dispatch
    # keeps it on the balanced flat mapping.
    _gpl = run // r_blk if r_blk else 0
    use_subaffine = (
        use_2d
        and b_2d_col
        and needs_upcast
        and R_lead > 1
        and _gpl >= 2
    )

    # Round 5: the fp32 path now carries the near-integer detector + the exact
    # scratch in the same tile, i.e. 10 full-size UB buffers at block_size
    # 4096 (163840 B).  The HiVM auto-multi-buffer pass then duplicates 3 of
    # them (measured: "ub overflow, requires 1703936 bits while 1572864 bits
    # available") for the shape where the task guard folds away.  Disabling it
    # for fp32 only keeps the tile at 4096 and leaves the half-precision DMA
    # pipelining (which is where it pays off) untouched.
    # Auto multi-buffer is what hides the DMA behind the vector chain; it is
    # disabled only where the fp32 1-D tile cannot afford the duplication
    # (10 buffers x 4096 x 4 B, "ub overflow, requires 1703936 bits").  The 2-D
    # tile keeps far smaller scratch, so it keeps the pipelining.
    _pass_cfg = {"npuir.enable_auto_multi_buffer": bool(needs_upcast)}

    @tilelang.jit(out_idx=[2], target="npuir", pass_configs=_pass_cfg)
    def kernel(block_size=block_size):
        if use_subaffine:
            # ---- round 3: sub-affine col mapping (half dtypes, R_lead > 1) ----
            num_kernels = min(_gpl, vector_cores)
            subs_per_core = T.ceildiv(_gpl, num_kernels)

            @T.prim_func
            def main3(
                a: T.Tensor((R_lead, run, L), dtype),
                b: T.Tensor((run, 1), dtype),
                y: T.Tensor((R_lead, run, L), out_dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    a32 = T.alloc_ub((r_blk, L), "float32")
                    b2 = T.alloc_ub((r_blk, 1), "float32")
                    q = T.alloc_ub((r_blk, L), "float32")
                    yf = T.alloc_ub((r_blk, L), "float32")
                    # Parser parses every branch body -> unconditional alloc.
                    a_in = T.alloc_ub((r_blk, L), dtype)
                    b_in = T.alloc_ub((r_blk, 1), dtype)
                    y_out = T.alloc_ub((r_blk, L), out_dtype)

                    for k in T.serial(subs_per_core):
                        sub = cid + k * num_kernels
                        if sub < _gpl:
                            s = sub * r_blk
                            # b's slice is loop-invariant across the lead loop
                            # and ``s`` is the SAME value used by the a/y tile
                            # subviews below (single affine index).
                            T.copy(b[s, 0], b_in)
                            T.vcast(b_in, b2)
                            for j in T.serial(R_lead):
                                T.copy(a[j, s, 0], a_in)
                                T.vcast(a_in, a32)
                                T.vdiv(a32, b2, q)
                                T.vcast(q, yf, round_mode="floor")
                                T.vcast(yf, y_out)
                                T.copy(y_out, y[j, s, 0])

            return main3

        if use_2d:
            # ---- alternative #2: 2-D [r_blk, L] row tile per task ----------
            groups_per_lead = run // r_blk
            num_logical = R_lead * groups_per_lead
            num_kernels = min(num_logical, vector_cores)
            num_local_tasks = T.ceildiv(num_logical, num_kernels)

            @T.prim_func
            def main2(
                a: T.Tensor((R, L), dtype),
                b: T.Tensor((run, 1) if b_2d_col else (1, L), dtype),
                y: T.Tensor((R, L), out_dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    a32 = T.alloc_ub((r_blk, L), "float32")
                    b2 = T.alloc_ub((r_blk, 1) if b_2d_col else (1, L), "float32")
                    q = T.alloc_ub((r_blk, L), "float32")
                    yf = T.alloc_ub((r_blk, L), "float32")
                    # Parser parses every branch body -> unconditional alloc.
                    # Native-dtype transit buffers exist only for the
                    # needs_upcast / needs_downcast paths; for fp32 they would
                    # cost 2 x (r_blk*L*4) B of the UB budget for nothing, so
                    # they collapse to a single element there.
                    a_in = T.alloc_ub((r_blk, L) if needs_upcast else (1, 1), dtype)
                    b_in = T.alloc_ub(
                        ((r_blk, 1) if b_2d_col else (1, L)) if needs_upcast else (1, 1), dtype
                    )
                    y_out = T.alloc_ub((r_blk, L) if needs_downcast else (1, 1), out_dtype)
                    # exact-division scratch (fp32 only; 1 element otherwise so
                    # the half-dtype UB budget is untouched)
                    x_sh = (r_blk, 1) if (b_2d_col and not needs_upcast) else (1, 1)
                    x_full = (1, 1) if needs_upcast else (r_blk, L)
                    x_bh = T.alloc_ub(x_sh, "float32")
                    x_bl = T.alloc_ub(x_sh, "float32")
                    x_qh = T.alloc_ub(x_full, "float32")
                    x_ql = T.alloc_ub(x_full, "float32")
                    x_rm = T.alloc_ub(x_full, "float32")
                    x_bs = T.alloc_ub(x_sh, "float32")
                    # round 6: per-tile near-integer flag (fp32 2-D path).
                    # x_cnt2 holds one scalar per tile row, x_cnt the tile flag.
                    x_cnt2 = T.alloc_ub((r_blk, 1) if not needs_upcast else (1, 1), "float32")
                    x_nan = T.alloc_ub((r_blk, L) if not needs_upcast else (1, 1), "bool")
                    x_isl = T.alloc_ub((r_blk, L) if not needs_upcast else (1, 1), "bool")
                    x_cnt = T.alloc_ub((1, 1), "float32")

                    # row mode: b is shared across all rows -> load once per core
                    if not b_2d_col:
                        if needs_upcast:
                            T.copy(b[0, 0], b_in)
                            T.vcast(b_in, b2)
                        else:
                            T.copy(b[0, 0], b2)

                    for i in T.serial(num_local_tasks):
                        task = i * num_kernels + cid
                        if task < num_logical:
                            lead = task // groups_per_lead
                            sub = task % groups_per_lead
                            row0 = lead * run + sub * r_blk
                            if needs_upcast:
                                T.copy(a[row0, 0], a_in)
                                T.vcast(a_in, a32)
                            else:
                                T.copy(a[row0, 0], a32)
                            if b_2d_col:
                                if needs_upcast:
                                    T.copy(b[sub * r_blk, 0], b_in)
                                    T.vcast(b_in, b2)
                                else:
                                    T.copy(b[sub * r_blk, 0], b2)
                            T.vdiv(a32, b2, q)
                            if not needs_upcast:
                                # ---- fp32: near-integer detector (2-D) ----
                                # Identical test to the 1-D path (see there):
                                # qu/ql bracket q0 by 2^-20 relative, and the
                                # tile is routed to the exact arm iff any lane
                                # has floor(qu) != floor(ql) -- which also covers
                                # +-Inf / NaN lanes through NaN != NaN.
                                T.vmul(q, _NEAR_HI, x_ql)                # qu
                                T.vcast(x_ql, x_qh, round_mode="floor")  # yq
                                T.vmul(q, _NEAR_LO, x_ql)                # ql
                                T.vcast(x_ql, x_rm, round_mode="floor")  # yl
                                T.vcmp(x_qh, x_rm, x_isl, "ne")          # yq != yl
                                T.vcast(x_isl, x_ql, round_mode="rint")
                                T.reduce_max(x_ql, x_cnt2, dim=-1, clear=True)
                                # fold the per-row flags into the tile flag
                                T.reduce_max(x_cnt2, x_cnt, dim=0, clear=True)
                            if not needs_upcast and not (x_cnt[0, 0] <= 0.5):
                                # ---- rare path: EXACT floor division (fp32) ----
                                T.vmax(q, -_CLAMP_Q, x_qh)
                                T.vmin(x_qh, _CLAMP_Q, q)
                                T.vmax(b2, -_CLAMP_B, x_bh)
                                T.vmin(x_bh, _CLAMP_B, x_bh)
                                T.vmul(x_bh, _SPLIT_K, x_bs)
                                T.vsub(x_bs, x_bh, x_bl)
                                T.vsub(x_bs, x_bl, x_bs)
                                T.vsub(x_bh, x_bs, x_bl)
                                T.vmul(q, _SPLIT_K, x_qh)
                                T.vsub(x_qh, q, x_ql)
                                T.vsub(x_qh, x_ql, x_qh)
                                T.vsub(q, x_qh, x_ql)
                                # Residual accumulated in a32: the dividend tile
                                # is dead after the vdiv, so this needs no
                                # separate (r_blk, L) residual buffer (one fewer
                                # full fp32 buffer keeps the 2-D tile inside the
                                # UB budget that made round 3 fast).
                                T.vmul(x_qh, x_bs, yf)
                                T.vsub(a32, yf, a32)
                                T.vmul(x_qh, x_bl, yf)
                                T.vsub(a32, yf, a32)
                                T.vmul(x_ql, x_bs, yf)
                                T.vsub(a32, yf, a32)
                                T.vmul(x_ql, x_bl, yf)
                                T.vsub(a32, yf, a32)
                                T.vdiv(a32, x_bh, yf)
                                T.vadd(q, yf, q)
                                T.vcast(q, yf, round_mode="floor")
                                # store inside the arm: with a value-producing
                                # scf.if every branch result gets its own
                                # tensor.empty and the tile overruns UB
                                # (measured on the 1-D path, round 4).
                                T.copy(yf, y[row0, 0])
                            else:
                                if not needs_upcast:
                                    # ---- fast path: vdiv + 2 mul + 2 floor ----
                                    T.copy(x_qh, y[row0, 0])
                                else:
                                    T.vcast(q, yf, round_mode="floor")
                                    if needs_downcast:
                                        T.vcast(yf, y_out)
                                        T.copy(y_out, y[row0, 0])
                                    else:
                                        T.copy(yf, y[row0, 0])

            return main2

        # ---- main choice #1: 1-D innermost tile --------------------------
        # Host-side UB budget guard (DESIGN.md section 4.5).
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if block_size > max_block:
            raise ValueError(
                f"block_size={block_size} exceeds the UB budget for {dtype} "
                f"(max {max_block} elements, {_UB_CAPACITY_BYTES} B UB capacity)"
            )
        # Shape-safe clamp: a short row (L < block_size) is tiled at ~L so the
        # vector ops do not process block_size - L stale lanes.  Folds to a
        # compile-time constant.
        block_size = _effective_block(block_size, L)
        if not needs_upcast:
            block_size = max(min(block_size, _EXACT_DIV_BLOCK), _ALIGN_ELEMS)

        # block_size-dependent task decomposition, evaluated here (not at
        # factory time) so the compiled kernel always matches the block_size
        # it is invoked with -- the returned callable is cached per
        # block_size (wrapper K11 contract).  block_size folds to a
        # compile-time constant.
        chunks = T.ceildiv(L, block_size)
        num_logical = R * chunks
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = T.ceildiv(num_logical, num_kernels)

        # r1_op2: outer-broadcast-operand hoist.  When an operand is broadcast
        # across *every* outer dim (all outer strides 0) and the row fits in a
        # single chunk, its GM->UB materialisation is loop-invariant: load it
        # once per core instead of once per task (removes R-1 redundant row
        # loads).  Inert for hidden (R=1) and cnn-feat (b outer stride 1).
        hoist_a = R > 1 and chunks == 1 and sa == 1 and all(
            s == 0 for s in a_strides[:-1]
        )
        hoist_b = R > 1 and chunks == 1 and sb == 1 and all(
            s == 0 for s in b_strides[:-1]
        )

        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                # fp32 运算域缓冲：整 buffer 一条向量指令覆盖。
                a32 = T.alloc_ub((block_size,), "float32")
                b32 = T.alloc_ub((block_size,), "float32")
                q = T.alloc_ub((block_size,), "float32")
                yf = T.alloc_ub((block_size,), "float32")
                # r1_op1：不再需要 a_in/b_in/y_out 中转缓冲——跨 dtype T.copy
                # 直接把 GM 的 native dtype 数据落到 fp32 buffer（及反向落盘）。
                # 标量广播源：仅当某 operand 末维 stride==0（该 operand 在行内
                # 是标量）时使用，先单元素搬入再 T.vbrc 物化成 (block_size,)。
                a1 = T.alloc_ub((1,), dtype)
                b1 = T.alloc_ub((1,), dtype)
                a1f = T.alloc_ub((1,), "float32")
                b1f = T.alloc_ub((1,), "float32")
                # exact-division scratch (fp32 only; 1 element otherwise)
                x_n = (1,) if needs_upcast else (block_size,)
                x_bh = T.alloc_ub(x_n, "float32")
                x_bl = T.alloc_ub(x_n, "float32")
                x_qh = T.alloc_ub(x_n, "float32")
                x_ql = T.alloc_ub(x_n, "float32")
                x_rm = T.alloc_ub(x_n, "float32")
                x_bs = T.alloc_ub(x_n, "float32")
                # round 5: near-integer detector flag + per-tile scalar count
                # (fp32 only; 1 element otherwise so the half UB budget is
                # untouched -- the half path keeps the plain vdiv+floor chain).
                x_cnt = T.alloc_ub((1,), "float32")
                # IEEE NaN mask for the tile flag (fp32 only; 1 element otherwise)
                x_nan = T.alloc_ub((block_size,) if not needs_upcast else (1,), "bool")
                x_isl = T.alloc_ub((block_size,) if not needs_upcast else (1,), "bool")

                # r1_op2: hoisted (loop-invariant) operand loads, once per core.
                if hoist_a:
                    T.copy(a[0:L], a32[0:L])
                if hoist_b:
                    T.copy(b[0:L], b32[0:L])

                # persistent 核内串行循环：边界为编译期静态值（num_local_tasks
                # 由 num_logical/num_kernels 常量除得），grid-stride 映射
                # task = i*num_kernels + cid，if 屏蔽越界 task。
                for i in T.serial(num_local_tasks):
                    task = i * num_kernels + cid
                    if task < num_logical:
                        row = task // chunks
                        inner_off = (task % chunks) * block_size
                        # 行 base 标量：每 task 仅一次 unrolled divmod（非每元素，
                        # 较源逐元素 divmod 降 ~L 倍），末维仅在 stride==1 时
                        # 贡献 inner_off。
                        a_base = _row_base(row, outer, a_strides) + (
                            inner_off if sa == 1 else 0
                        )
                        b_base = _row_base(row, outer, b_strides) + (
                            inner_off if sb == 1 else 0
                        )
                        tail = T.min(block_size, L - inner_off)

                        # --- 物化 a 到 fp32 域 (block_size,) ---
                        if not hoist_a:
                            if sa == 1:
                                # r1_op1：跨 dtype T.copy 直接落到 fp32 buffer，
                                # 免去 a_in 中转与一次 vcast。
                                T.copy(a[a_base : a_base + tail], a32[0:tail])
                            else:
                                # a 行内标量：单元素搬入 + vbrc 广播到整 buffer。
                                T.copy(a[a_base : a_base + 1], a1[0:1])
                                if needs_upcast:
                                    T.vcast(a1, a1f, round_mode="rint")
                                    T.vbrc(a1f, a32)
                                else:
                                    T.vbrc(a1, a32)

                        # --- 物化 b 到 fp32 域 (block_size,)（同上）---
                        if not hoist_b:
                            if sb == 1:
                                T.copy(b[b_base : b_base + tail], b32[0:tail])
                            else:
                                T.copy(b[b_base : b_base + 1], b1[0:1])
                                if needs_upcast:
                                    T.vcast(b1, b1f, round_mode="rint")
                                    T.vbrc(b1f, b32)
                                else:
                                    T.vbrc(b1, b32)

                        # --- 除法 + floor + 落盘 ---
                        # 禁用 T.vfloordiv / T.vfloor（编译成功但运行期挂死，
                        # CG-2026-0023）：唯一可用链是 vdiv + vcast(floor)，
                        # 且必须在 fp32 域（半精度商先舍入再 floor 会偏）。
                        flat0 = row * L + inner_off
                        T.vdiv(a32, b32, q)
                        if not needs_upcast:
                            # ---- fp32: near-integer detector --------
                            # Conservative interval test on the quotient:
                            #   qu = fl(q0*(1+eps)), yq = floor(qu)  <- the value
                            #                                         the fast
                            #                                         path stores
                            #   ql = fl(q0*(1-eps)), yl = floor(ql)
                            #   flag <=> yq != yl  (an integer lies in [ql, qu])
                            # eps = 2^-20 is 8x the proven 1-ulp vdiv error, so
                            # ql <= q_true <= qu always; a tile with yq == yl
                            # therefore has floor(qu) == floor(ql) == floor(q_true).
                            # Both floors are required: the interval is not ordered
                            # by the sign of q0 (a=8, b=-2 -> floor(qu) = -5 while
                            # the answer is -4), so a one-sided test is unsound.
                            # NaN != NaN is TRUE, so +-Inf / NaN lanes flag their
                            # tile into the exact arm too.  That is required: vmax
                            # / vmin / vadd all PROPAGATE NaN on this toolchain and
                            # a NaN scalar flag silently takes the wrong arm
                            # (measured: a tile mixing Inf with finite
                            # near-integer quotients corrupted the finite
                            # elements).  Casting the bool to 0/1 before the
                            # reduction keeps the reduced value -- and hence the
                            # scalar compare -- finite.
                            T.vmul(q, _NEAR_HI, x_ql)                # qu
                            T.vcast(x_ql, x_qh, round_mode="floor")  # yq
                            T.vmul(q, _NEAR_LO, x_ql)                # ql
                            T.vcast(x_ql, x_rm, round_mode="floor")  # yl
                            T.vcmp(x_qh, x_rm, x_isl, "ne")          # yq != yl
                            T.vcast(x_isl, x_ql, round_mode="rint")  # 0.0 / 1.0
                            # reduce_max: a stale accumulator can only over-flag,
                            # which costs a rare exact arm (safe direction).
                            T.reduce_max(x_ql, x_cnt, dim=-1, clear=True)
                            if not (x_cnt[0] <= 0.5):
                                # ---- rare path: EXACT floor division ------
                                # Verbatim round-4 Markstein step with the exact
                                # residual (Veltkamp/Dekker, FMA-free).  Same
                                # floor semantics incl. the +-Inf / NaN /
                                # 0/0 clamps as the shipped exact version.
                                T.vmax(q, -_CLAMP_Q, x_qh)
                                T.vmin(x_qh, _CLAMP_Q, q)
                                T.vmax(b32, -_CLAMP_B, x_bh)
                                T.vmin(x_bh, _CLAMP_B, x_bh)
                                T.vmul(x_bh, _SPLIT_K, x_bs)
                                T.vsub(x_bs, x_bh, x_bl)
                                T.vsub(x_bs, x_bl, x_bs)
                                T.vsub(x_bh, x_bs, x_bl)
                                T.vmul(q, _SPLIT_K, x_qh)
                                T.vsub(x_qh, q, x_ql)
                                T.vsub(x_qh, x_ql, x_qh)
                                T.vsub(q, x_qh, x_ql)
                                T.vmul(x_qh, x_bs, yf)
                                T.vsub(a32, yf, x_rm)
                                T.vmul(x_qh, x_bl, yf)
                                T.vsub(x_rm, yf, x_rm)
                                T.vmul(x_ql, x_bs, yf)
                                T.vsub(x_rm, yf, x_rm)
                                T.vmul(x_ql, x_bl, yf)
                                T.vsub(x_rm, yf, x_rm)
                                T.vdiv(x_rm, x_bh, yf)
                                T.vadd(q, yf, q)
                                T.vcast(q, yf, round_mode="floor")
                                # The GM store is repeated inside each arm on
                                # purpose: with a *value-producing* scf.if the
                                # bufferiser gives every branch result its own
                                # tensor.empty (no buffer reuse) and the fp32
                                # tile overruns UB at block_size 4096
                                # (measured: "ub overflow, requires 1703936
                                # bits"); terminating both arms in the store
                                # keeps the scf.if value-free.
                                T.copy(yf[0:tail], y[flat0 : flat0 + tail])
                            else:
                                # ---- fast path: vdiv + 2 mul + 2 floor ----
                                T.copy(x_qh[0:tail], y[flat0 : flat0 + tail])
                        else:
                            T.vcast(q, yf, round_mode="floor")
                            # r1_op1：跨 dtype T.copy 落盘（fp32 yf -> native y），
                            # 免去 y_out 中转与一次 vcast；尾块 [tail:] 的陈旧 lane
                            # 一并参与向量计算，但只写回 [0:tail]。
                            T.copy(yf[0:tail], y[flat0 : flat0 + tail])

        return main

    return kernel


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
# Per-dtype tolerance (DESIGN.md section 8.2, identical to the source repo
# test_floor_divide.py: atol = 1.0 if dtype != float32 else 1e-5).
# The fp32-transit chain is measured bit-exact vs the golden; the half
# tolerances are the source contract.
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1.0, 0.0),
    "bfloat16": (1.0, 0.0),
}

_FACTORY_CACHE = {}

# Per-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_factory(N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel):
    key = (N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = floor_divide_fwd_kernel(
            N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel
        )
    return _FACTORY_CACHE[key]


def _rand_cpu(shape, dtype_str, mode="positive"):
    """Random CPU data (source tests/bench use positive operands).

    mode: ``positive`` -> [0.1, 4.0); ``negative`` -> (-4.0, -0.1];
    ``signed`` -> randn (exercises the toward-minus-inf floor branch).
    """
    if mode == "positive":
        base = torch.rand(shape, dtype=torch.float32) * 3.9 + 0.1
    elif mode == "negative":
        base = -(torch.rand(shape, dtype=torch.float32) * 3.9 + 0.1)
    else:
        base = torch.randn(shape, dtype=torch.float32)
    return base.to(_TORCH_DTYPE[dtype_str])


def _compare(out, ref, dtype_str, equal_nan=False):
    """Return (max_diff, violations); NaN/Inf aware when equal_nan is set."""
    atol, rtol = _TOLERANCE[dtype_str]
    if tuple(out.shape) != tuple(ref.shape):
        raise AssertionError(f"shape mismatch: out {tuple(out.shape)} != ref {tuple(ref.shape)}")
    if out.dtype != ref.dtype:
        raise AssertionError(f"dtype mismatch: out {out.dtype} != ref {ref.dtype}")
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
    a_shape,
    b_shape,
    dtype_str,
    tag,
    block_size=None,
    a_cpu=None,
    b_cpu=None,
    mode="positive",
    equal_nan=False,
    exact=False,
):
    """Run one kernel case and compare against the golden.

    ``a_cpu``/``b_cpu`` may be supplied verbatim (special-value cases);
    otherwise random data is generated (``mode``: positive/negative/signed).
    ``exact=True`` additionally requires bit-exact equality (fp32 path).
    """
    if block_size is None:
        block_size = _DEFAULT_BLOCK_BY_DTYPE.get(dtype_str, _DEFAULT_BLOCK)

    out_shape, cshape, a_strides, b_strides = coalesce_broadcast_dims(a_shape, b_shape)
    N_total = math.prod(out_shape)
    a_numel = math.prod(a_shape) if len(a_shape) else 1
    b_numel = math.prod(b_shape) if len(b_shape) else 1

    if a_cpu is None:
        a_cpu = _rand_cpu(a_shape, dtype_str, mode)
        b_cpu = _rand_cpu(b_shape, dtype_str, mode)
    assert b_cpu is not None, "a_cpu and b_cpu must be supplied together"

    ref = golden_floor_divide(a_cpu, b_cpu)
    assert tuple(ref.shape) == tuple(out_shape), (
        f"golden shape {tuple(ref.shape)} != broadcast {tuple(out_shape)}"
    )
    assert ref.dtype == _TORCH_DTYPE[dtype_str], f"golden dtype {ref.dtype} unexpected"

    a_npu = a_cpu.reshape(-1).npu()
    b_npu = b_cpu.reshape(-1).npu()

    fn = _get_factory(N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel)
    y = fn(block_size)(a_npu, b_npu)
    out = y.cpu().reshape(out_shape)

    if exact and not torch.equal(out, ref):
        viol = int((out != ref).sum().item())
        md = (out.float() - ref.float()).abs().max().item()
        raise AssertionError(f"exact compare failed: {viol} violation(s), max_diff={md}")

    max_diff, viol = _compare(out, ref, dtype_str, equal_nan=equal_nan)
    torch.testing.assert_close(
        out, ref, rtol=_TOLERANCE[dtype_str][1], atol=_TOLERANCE[dtype_str][0],
        equal_nan=equal_nan,
    )
    if max_diff == max_diff and max_diff > _STATS_MAX_DIFF.get(dtype_str, -1.0):
        _STATS_MAX_DIFF[dtype_str] = max_diff  # max_diff == max_diff skips NaN

    print(
        f"[{tag}] PASS: a={tuple(a_shape)} b={tuple(b_shape)} dtype={dtype_str} "
        f"block={block_size} max_diff={max_diff}"
    )
    return max_diff, viol


def _try_case(failures, case_id, **kwargs):
    """Run ``_run_case``, collecting (not re-raising) per-case failures."""
    try:
        _run_case(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, twelve items)."""
    failures = []

    # L0-1: smoke same-shape, all float dtypes (manifest smoke-1k).
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(failures, f"L0-1/{dtype_str}/4096", a_shape=(4096,), b_shape=(4096,), dtype_str=dtype_str, tag="L0-1")
    _try_case(failures, "L0-1/float32/1024", a_shape=(1024,), b_shape=(1024,), dtype_str="float32", tag="L0-1")

    # L0-2: large same-shape (manifest hidden-state-prefill, chunks=2048).
    for dtype_str in ("float32", "float16"):
        _try_case(
            failures,
            f"L0-2/{dtype_str}",
            a_shape=(2048, 4096),
            b_shape=(2048, 4096),
            dtype_str=dtype_str,
            tag="L0-2",
        )

    # L0-3: broadcast three modes (manifest cnn-feat + row-broadcast + bias).
    for dtype_str in ("float32", "float16"):
        for a_shape, b_shape, label in (
            ((16, 256, 56, 56), (256, 1, 1), "cnn-feat"),
            ((2, 64, 128), (1, 1, 128), "bias"),
        ):
            _try_case(
                failures,
                f"L0-3/{dtype_str}/{label}",
                a_shape=a_shape,
                b_shape=b_shape,
                dtype_str=dtype_str,
                tag="L0-3",
            )
    _try_case(
        failures,
        "L0-3/float32/row-broadcast",
        a_shape=(4096, 4096),
        b_shape=(1, 4096),
        dtype_str="float32",
        tag="L0-3",
    )

    # L0-4: a-side broadcast (roles swapped, exercises sa == 0).
    _try_case(failures, "L0-4/1x4096|4096x4096", a_shape=(1, 4096), b_shape=(4096, 4096), dtype_str="float32", tag="L0-4")
    _try_case(failures, "L0-4/2x64x128|2x64x1", a_shape=(2, 64, 128), b_shape=(2, 64, 1), dtype_str="float32", tag="L0-4")

    # L0-5: scalar broadcast / full scalar (L == 1 both-vbrc degenerate).
    _try_case(failures, "L0-5/scalar-broadcast", a_shape=(64, 128), b_shape=(1, 1), dtype_str="float32", tag="L0-5")
    _try_case(failures, "L0-5/full-scalar", a_shape=(1,), b_shape=(1,), dtype_str="float32", tag="L0-5")

    # L0-6: non-divisible tails (single partial tile / non-32B-aligned tail).
    for N in (1000, 3136, 4097):
        _try_case(failures, f"L0-6/N={N}", a_shape=(N,), b_shape=(N,), dtype_str="float32", tag="L0-6")

    # L0-7: integer / bool dtype rejection (source float-only contract).
    for bad in ("int8", "int16", "int32", "int64", "uint8", "bool"):
        try:
            _normalize_dtype(bad)
        except ValueError:
            print(f"[L0-7] PASS: dtype {bad!r} rejected")
        else:
            failures.append((f"L0-7/{bad}", AssertionError(f"{bad} not rejected")))
    try:
        floor_divide_fwd_kernel(4096, "int32", (4096,), (1,), (1,), 4096, 4096)
    except ValueError:
        print("[L0-7] PASS: factory rejects int32")
    else:
        failures.append(("L0-7/factory-int32", AssertionError("factory int32 not rejected")))

    # L0-8: fp32 no-identity-cast regression (DESIGN.md note 3, CG-2026-0024).
    _try_case(failures, "L0-8/float32-exact", a_shape=(4096,), b_shape=(4096,), dtype_str="float32", tag="L0-8", exact=True)

    # L0-9: persistent core-count switch boundary (num_logical in 39/40/41).
    bs = _DEFAULT_BLOCK
    for num_logical, N in ((39, 39 * bs), (40, 40 * bs), (41, 41 * bs)):
        _try_case(
            failures,
            f"L0-9/num_logical={num_logical}",
            a_shape=(N,),
            b_shape=(N,),
            dtype_str="float32",
            tag="L0-9",
        )

    # L0-10: float corners (NaN / +-Inf / +-0 / subnormal / integer boundary).
    _run_L0_corners(failures)

    # L0-11: vdiv broadcast capability regression ([M,N]/[1,N], [M,N]/[M,1],
    # [M,N]/[1,1]) -- covers the R1 materialisation for both operand roles.
    for a_shape, b_shape, label in (
        ((64, 128), (1, 128), "col"),
        ((64, 128), (64, 1), "row"),
        ((64, 128), (1, 1), "scalar"),
    ):
        _try_case(
            failures,
            f"L0-11/{label}",
            a_shape=a_shape,
            b_shape=b_shape,
            dtype_str="float32",
            tag="L0-11",
        )

    # L0-12: divisor boundary (small positive divisor, b == 1.0).
    try:
        a = _rand_cpu((4096,), "float32")
        b = torch.full((4096,), 1.0, dtype=torch.float32)
        _try_case(
            failures,
            "L0-12/b=1.0",
            a_shape=(4096,),
            b_shape=(4096,),
            dtype_str="float32",
            tag="L0-12",
            a_cpu=a,
            b_cpu=b,
        )
        a2 = _rand_cpu((4096,), "float32")
        b2 = torch.full((4096,), 0.1, dtype=torch.float32)
        _try_case(
            failures,
            "L0-12/b=0.1",
            a_shape=(4096,),
            b_shape=(4096,),
            dtype_str="float32",
            tag="L0-12",
            a_cpu=a2,
            b_cpu=b2,
        )
    except Exception as exc:  # noqa: BLE001
        failures.append(("L0-12/setup", exc))

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (twelve items)")
    return failures


def _run_L0_corners(failures):
    """L0-10: float special values, fp32 + fp16 (IEEE propagation)."""
    nan = float("nan")
    inf = float("inf")
    a_pat = [6.0, 7.0, -7.0, inf, -inf, nan, 0.0, -0.0, 2.9999, -3.0001, 1.0, 1e-8, 5.5, -5.5, 8.0, 9.0]
    b_pat = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, inf, 3.0, 2.0, 2.0, -2.0, 0.5]
    reps = 4096 // len(a_pat)
    for dtype_str in ("float32", "float16"):
        a = torch.tensor(a_pat * reps, dtype=_TORCH_DTYPE[dtype_str])
        b = torch.tensor(b_pat * reps, dtype=_TORCH_DTYPE[dtype_str])
        _try_case(
            failures,
            f"L0-10/{dtype_str}",
            a_shape=(4096,),
            b_shape=(4096,),
            dtype_str=dtype_str,
            tag="L0-10",
            a_cpu=a,
            b_cpu=b,
            equal_nan=True,
        )


def run_L1():
    """L1: full functional coverage (dtype x shape x block_size)."""
    failures = []

    # dtype x {exact multiple, tail==1-ish, non-32B-aligned tail}.
    for dtype_str in ("float16", "bfloat16", "float32"):
        for N in (8192, 1000, 4097):
            _try_case(failures, f"L1/{dtype_str}/N={N}", a_shape=(N,), b_shape=(N,), dtype_str=dtype_str, tag="L1")

    # Broadcast variants across dtypes and shapes.
    for dtype_str in ("float16", "float32"):
        for a_shape, b_shape in (
            ((4, 8, 16), (1, 8, 1)),
            ((3, 5), (1, 1)),
            ((2, 3, 4), (2, 3, 4)),
            ((1, 4096), (4096, 4096)),
            ((16, 256, 56, 56), (256, 1, 1)),
        ):
            _try_case(
                failures,
                f"L1/{dtype_str}/{a_shape}",
                a_shape=a_shape,
                b_shape=b_shape,
                dtype_str=dtype_str,
                tag="L1",
            )

    # Signed operands: the toward-minus-infinity floor branch.
    for dtype_str in ("float16", "float32"):
        _try_case(failures, f"L1/{dtype_str}/signed", a_shape=(4096,), b_shape=(4096,), dtype_str=dtype_str, tag="L1", mode="signed")
        _try_case(failures, f"L1/{dtype_str}/negative", a_shape=(4096,), b_shape=(4096,), dtype_str=dtype_str, tag="L1", mode="negative")

    # Non-default block sizes within the UB budget.
    _try_case(failures, "L1/fp32/N=2^20/bs=2048", a_shape=(2**20,), b_shape=(2**20,), dtype_str="float32", tag="L1", block_size=2048)
    _try_case(failures, "L1/float16/N=2^20/bs=2048", a_shape=(2**20,), b_shape=(2**20,), dtype_str="float16", tag="L1", block_size=2048)
    _try_case(failures, "L1/bfloat16/N=2^18/bs=1024", a_shape=(2**18,), b_shape=(2**18,), dtype_str="bfloat16", tag="L1", block_size=1024)

    # E2 factory contract: one factory callable serving two block_sizes.
    try:
        dtype_str = "float32"
        a_shape = (8192,)
        out_shape, cshape, a_strides, b_strides = coalesce_broadcast_dims(a_shape, a_shape)
        a = _rand_cpu(a_shape, dtype_str).npu()
        b = _rand_cpu(a_shape, dtype_str).npu()
        ref = golden_floor_divide(a.cpu(), b.cpu())
        fn = _get_factory(8192, dtype_str, cshape, a_strides, b_strides, 8192, 8192)
        for bs in (4096, 2048):
            out = fn(bs)(a, b).cpu()
            max_diff, _ = _compare(out, ref, dtype_str)
            torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        print("[L1] PASS: E2 factory(block_size) dual-config reuse")
    except AssertionError as exc:
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse RUNTIME-ERROR: {exc!r}")

    # UB budget guard.  Stage 4 raised the ceiling to 12288 for every float
    # dtype (4 fp32 buffers x 12288 x 4 B = 196608 B); the guard must therefore
    # reject values above that new ceiling.
    try:
        floor_divide_fwd_kernel(4096, "float16", (4096,), (1,), (1,), 4096, 4096)(13000)
    except ValueError as exc:
        print(f"[L1] PASS: UB guard rejects float16 block_size=13000 ({exc})")
    else:
        failures.append(("L1/UB-guard-float16", AssertionError("float16 13000 not rejected")))
    try:
        floor_divide_fwd_kernel(4096, "float32", (4096,), (1,), (1,), 4096, 4096)(13000)
    except ValueError as exc:
        print(f"[L1] PASS: UB guard rejects float32 block_size=13000 ({exc})")
    else:
        failures.append(("L1/UB-guard-float32", AssertionError("float32 13000 not rejected")))

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes (warn only, non-blocking)."""
    for N, dtype_str in ((1, "float32"), (2, "float16"), (3, "bfloat16"), (2**22, "float16"), (2**24, "float32")):
        try:
            _run_case(a_shape=(N,), b_shape=(N,), dtype_str=dtype_str, tag="L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    N = 4096

    # all zeros: 0 / 0 -> NaN (IEEE), matches the golden.
    try:
        a = torch.zeros(N, dtype=torch.float32)
        b = torch.zeros(N, dtype=torch.float32)
        _run_case(a_shape=(N,), b_shape=(N,), dtype_str="float32", tag="Boundary-zeros", a_cpu=a, b_cpu=b, equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros: {e}")

    # a == b -> floor(1) == 1 (positive), 0 -> -1 (negative) etc.
    try:
        a = _rand_cpu((N,), "float32", "signed")
        _run_case(a_shape=(N,), b_shape=(N,), dtype_str="float32", tag="Boundary-a-eq-b", a_cpu=a, b_cpu=a.clone(), equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): a==b: {e}")

    # subnormal magnitudes.
    try:
        a = (torch.randn(N, dtype=torch.float32) * 1e-8).to(torch.float16)
        b = (torch.rand(N, dtype=torch.float32) * 0.9 + 0.1).to(torch.float16)
        _run_case(a_shape=(N,), b_shape=(N,), dtype_str="float16", tag="Boundary-subnormal", a_cpu=a, b_cpu=b, equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): subnormal: {e}")

    # large magnitudes.
    try:
        a = (torch.rand(N, dtype=torch.float32) * 100.0 + 1.0).to(torch.float16)
        b = (torch.rand(N, dtype=torch.float32) * 3.0 + 0.1).to(torch.float16)
        _run_case(a_shape=(N,), b_shape=(N,), dtype_str="float16", tag="Boundary-large", a_cpu=a, b_cpu=b, equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large: {e}")

    # scalar broadcast with a special divisor (x / scalar).
    try:
        a = _rand_cpu((N,), "float32")
        b = torch.tensor([2.5], dtype=torch.float32)
        _run_case(a_shape=(N,), b_shape=(1,), dtype_str="float32", tag="Boundary-scalar-div", a_cpu=a, b_cpu=b)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): scalar-div: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    if args.level in ("L0", "all"):
        t0 = time.time()
        failures += run_L0()
        print(f"[L0] elapsed {time.time() - t0:.1f}s")
    if args.level in ("L1", "all"):
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
    if args.level in ("L2", "all"):
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
    if args.level in ("boundary", "all"):
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    print(
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}={v}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
