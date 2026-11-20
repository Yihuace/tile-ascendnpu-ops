# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Windowed max pooling 1D forward kernel with int64 indices (NPU, Developer).

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
import functools
import time

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

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

# Target channel-vector-block ceiling (DESIGN.md sections 5.2 / 1.6.3).
_CH_MAX = 64

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

    The frozen DESIGN.md section 4.5 criterion (revision v1, byte coefficient
    ``69 + 2*eb`` after the v0 undercount of idx32 / idx_t / idx64 was fixed):

        G = CH*LSPAN*(2*eb + 9) + BL*CH*(69 + 2*eb) + 256

    ``2*eb + 9``  = row_src(eb) + row_dst(eb) + row_f32(4) + eq_mask(1) +
                    eq_f32(4);
    ``69 + 2*eb`` = acc / flag / infbuf / nanbuf / out32 / zeros (6 x f32 = 24)
                    + cfinal (bool 1) + acc16 + acc_t (2*eb)
                    + max_idx / nan_idx / cand / nanc / bigv / neg1v / zero_i32
                      (7 x i32 = 28)
                    + idx32 / idx_t (8) + idx64 (8);
    ``256``       = fv (1, CH) int32 upper bound for CH <= 64.

    It is reproduced verbatim (per-buffer sum == closed form, the v1 self-check)
    so the CH/BL selection matches the design tables exactly (sincnet CH=64 /
    BL=16 / LSPAN=48; ecg CH=64 / BL=17 / LSPAN=41).  The implementation folds
    the per-``owl`` ``(1, CH)`` staging operands into small whole buffers, so the
    real footprint is at or below this bound.
    """
    per_elem = 2 * elem_bytes + 9  # row_src + row_dst + row_f32 + eq_mask + eq_f32
    base = 6 * 4 + 1 + 2 * elem_bytes + 7 * 4 + 2 * 4 + 8  # = 69 + 2*eb
    return ch * lspan * per_elem + bl * ch * base + 256


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
@functools.lru_cache(maxsize=32)
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
                    # Vector accumulators on the channel axis (vector axis = C).
                    acc = T.alloc_ub((bl, ch), "float32")
                    flag = T.alloc_ub((bl, ch), "float32")
                    infbuf = T.alloc_ub((bl, ch), "float32")
                    nanbuf = T.alloc_ub((bl, ch), "float32")
                    out32 = T.alloc_ub((bl, ch), "float32")
                    cfinal = T.alloc_ub((bl, ch), "bool")
                    max_idx = T.alloc_ub((bl, ch), "int32")
                    nan_idx = T.alloc_ub((bl, ch), "int32")
                    idx32 = T.alloc_ub((bl, ch), "int32")
                    zeros_i = T.alloc_ub((bl, ch), "int32")
                    bigv = T.alloc_ub((bl, ch), "int32")
                    idx_t = T.alloc_ub((ch, bl), "int32")
                    idx64 = T.alloc_ub((ch, bl), "int64")
                    acc16 = T.alloc_ub((bl, ch), dtype)
                    acc_t = T.alloc_ub((ch, bl), dtype)
                    # Contiguous span staging: NCL span -> transposed (C innermost).
                    row_src = T.alloc_ub((ch, lspan), dtype)
                    row_dst = T.alloc_ub((lspan, ch), dtype)
                    row_f32 = T.alloc_ub((lspan, ch), "float32")
                    eq_mask = T.alloc_ub((lspan, ch), "bool")
                    eq_f32 = T.alloc_ub((lspan, ch), "float32")
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
                    one_val = T.float32(1.0)
                    big_i = T.int32(l_in)
                    neg1_i = T.int32(-1)
                    zero_i = T.int32(0)

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
                            T.vbrc(one_val, flag)
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
                            # Block-level NaN probe materialised as an fp32
                            # {0, 1} mask: x == x is 1 for numbers and 0 for NaN
                            # (the "ne" mode is an ordered not-equal and is always
                            # false for NaN, so it can never detect NaN).
                            T.vcmp(row_f32, row_f32, eq_mask, "eq")
                            T.vcast(eq_mask, eq_f32, round_mode="rint")

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
                                        T.vmul(
                                            flag[owl : owl + 1, 0:ch],
                                            eq_f32[j : j + 1, 0:ch],
                                            flag[owl : owl + 1, 0:ch],
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
                            T.vcmp(nan_idx, zeros_i, cfinal, "ge")
                            T.vselect(cfinal, nanbuf, acc, out32)
                            T.vselect(cfinal, nan_idx, max_idx, idx32)
                            # Empty-window clause (DESIGN.md 0.1, alignment with
                            # the source state machine): idx32 == BIG <=> no NaN
                            # and no in-bounds lane -> idx = 0.  It reuses the
                            # cfinal bool buffer, so it must run *after* out32 is
                            # selected.
                            T.vcmp(idx32, bigv, cfinal, "eq")
                            T.vselect(cfinal, zeros_i, idx32, idx32)

                            if is_f32:
                                T.copy(out32, acc16)
                            else:
                                T.vcast(out32, acc16, round_mode="rint")
                            bl_real = T.min(bl, out_l - ow0)
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
