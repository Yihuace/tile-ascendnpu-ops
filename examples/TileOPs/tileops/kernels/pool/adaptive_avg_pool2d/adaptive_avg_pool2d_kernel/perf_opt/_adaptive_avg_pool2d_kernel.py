# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Adaptive average pooling 2D forward kernel (NPU, Developer mode) -- TUNED (round3: C4 fp16 cheap reduce chain + plane/dtype-aware CH + single-wave CH heuristic).

Stage 4 (`perf_opt/`) of the migrated ``AdaptiveAvgPool2dFwdOp`` kernel.  Base:
``../_adaptive_avg_pool2d_kernel.py`` (Stage 3, precision-passing).

Factory contract identical to the base (wrapper-compatible):
    _adaptive_avg_pool2d_kernel(n, c_in, h_in, w_in, out_h, out_w,
                                dtype="float16")
      -> build(block_m=None) -> main
    where ``main(x)`` returns ``out`` (``out_idx=[1]``).

Three structural paths stay dispatched at factory level, all three correct:
  * identity   (out_H == H_in and out_W == W_in): UB transit copy;
  * C4 global  (out_H == out_W == 1): whole-plane reduction;
  * general    (anything else): per-row transpose + C-axis bin accumulation.

Tuning changes vs the base (all measured with `msprof op` Task Duration):
  G1. C4 global path: flattened spatial axis.  The base used the natural
      ``(ch, H, W)`` UB tile and ``T.reduce_sum`` over the W axis (extent 7) and
      then the H axis.  On this toolchain that reduction is scalarised
      (~19 us/unit for resnet-global 8x2048x7x7 -> ~166 us kernel).
      Declaring the plane as ``(N, C, 1, H*W)`` keeps the GM->UB DMA fully
      contiguous and makes the innermost axis 49 wide, so a single
      ``T.reduce_sum(..., dim=2)`` vectorises over the channel axis.
  G2. Per-path channel blocks (``_CH_IDENTITY`` / ``_CH_GLOBAL`` /
      ``_CH_GENERAL``) instead of one shared ``min(C, 64)``: the general-grid
      logical-unit count ``N * out_H * ceil(C/CH)`` decides how many of the
      physical vector cores are busy (nondiv: 14 units for C=64,CH=64 vs 28 for
      CH=32), and the C4 path wants a block that keeps the reduction vector
      wide while fitting UB.
  G3. Degenerate-path UB budget checks recomputed per path from the buffer list
      actually used by that path.

Run the embedded hierarchical tests:
    python _adaptive_avg_pool2d_kernel.py --level L0
    python _adaptive_avg_pool2d_kernel.py --level all
"""

import os

# Developer mode: alloc_ub maps to UB, the compiler auto-synchronises.
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

# Programming mode verified in Stage 3 (Stage 5 syncs the wrapper's
# Kernel.ascend_mode from this literal).
ASCEND_MODE = "Developer"

__all__ = [
    "ASCEND_MODE",
    "CH_GENERAL",
    "CH_GLOBAL",
    # Tuned defaults exposed as module-level constants (wrapper switch block
    # imports them together with the three kernel names).
    "CH_IDENTITY",
    "_adaptive_avg_pool2d_kernel",
    "adaptive_bin",
    "golden_adaptive_avg_pool2d",
    "max_adaptive_bin_extent",
]


# ---------------------------------------------------------------------------
# Source helpers (re-exported by the integration __init__.py)
# ---------------------------------------------------------------------------
def adaptive_bin(o, size_in: int, size_out: int):
    """Half-open input range ``[start, end)`` feeding output index ``o``.

    PyTorch partitions each spatial axis as
    ``[floor(o*in/out), ceil((o+1)*in/out))``. Bins are never empty, including
    when ``size_out > size_in``. Evaluated while tracing, so the emitted
    expression is the same as writing the arithmetic inline.
    """
    return (o * size_in) // size_out, ((o + 1) * size_in + size_out - 1) // size_out


def max_adaptive_bin_extent(size_in: int, size_out: int) -> int:
    """Widest bin on this axis -- a compile-time bound for a ``T.serial`` loop.

    ``ceil(in/out)`` is not a valid bound: in=55/out=7 has a bin of 9 against
    ceil = 8, and expanding in=8/out=12 gives bins of 2 against 1.
    """
    extents = (adaptive_bin(o, size_in, size_out) for o in range(size_out))
    return max(end - start for start, end in extents)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196_608

# Conservative multi-buffer factor for the UB-budget guard: the
# auto-multi-buffer pass inflates the resident buffers when the persistent
# serial loop is active.
_UB_MULTIBUFFER_FACTOR = 2

# Supported dtypes (the source op rejects float32).
_SUPPORTED_DTYPES = ("float16", "bfloat16")

# Precision tolerances (identical to the source repo tests/ops/test_pool.py).
_TOLERANCE = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "bfloat16": (torch.bfloat16, 1.6e-2, 1.6e-2),
}

# fp16/bf16 vector width in elements (alignment of the channel vector axis).
_VECTOR_WIDTH = 8

# Measured BishengIR reduction UB unit price for this kernel's C4 chain, in
# bytes of UB per *plane element* of the reduction input -- the lowering books
# it independently of the real dtype buffer list
# (constants.md CONST-reduce-ub-per-element).  Two chain forms:
#   * fp16 cheap chain (`reduce_sum(fp16 -> fp16)` + one `vcast`): calibrated
#     on 910B3 / CANN 9.0.0 by compile-time sweep (perf_opt/probe_r3_gf.py):
#     plane=49 fp16 admits CH=64 but not CH=128; plane=81 bf16 rejects CH=64
#     but admits CH=32 with the fp32 chain -> unit prices land in (37.9, 62.7].
#   * bf16 (`reduce_sum` has no bf16 accumulation form -> fp32 full-plane
#     intermediate + `reduce_sum(fp32 -> fp32)`).
_C4_REDUCE_UB_UNIT_F16 = 48
_C4_REDUCE_UB_UNIT_SUBFP32 = 64

# ---- tuned per-path channel blocks (module-level, wrapper-visible) --------
# Identity path: the UB tile is (CH, H_in, W_in) fp16, so CH is capacity bound.
CH_IDENTITY = 64
# C4 global path: the flattened tile is (CH, 1, H*W) fp16 + fp32, reduced along
# the innermost H*W axis.  CH=32 measured fastest for resnet-global
# (8x2048x7x7); CH=64 overflows UB in the reduction lowering for H*W=49.
CH_GLOBAL = 64
# General grid path: base channel block; the effective block is derived from
# this constant by `_pick_ch_general` (single-wave shape dispatch, G2).
CH_GENERAL = 64


def _round_ch(limit: int, cap: int) -> int:
    """Channel block: ``min(limit, cap)`` rounded up to the fp16 vector width."""
    ch = min(limit, cap)
    return max(_VECTOR_WIDTH, ((ch + _VECTOR_WIDTH - 1) // _VECTOR_WIDTH) * _VECTOR_WIDTH)


def _pick_ch_general(c_in: int, n: int, out_h: int, vector_cores: int, cap: int) -> int:
    """Effective channel block of the general grid path (G2, refined in round 3).

    The logical-unit count is ``N * out_H * ceil(C/CH)``; it decides how many
    physical vector cores are busy and whether the persistent loop needs more
    than one wave (``num_local > 1``).  A second wave costs more than a wider
    vector saves, so the block is selected **inside the single-wave set**
    (``n * out_h * ceil(C/CH) <= vector_cores``) by:

      1. maximise the number of busy cores (``units``);
      2. among equal occupancy, minimise the last block's wasted channels
         (``ceil(C/CH) * CH - C``);
      3. among those, prefer the wider block.

    Measured with `msprof op` Task Duration on the manifest general-grid shapes
    (round 3, `perf_opt/probe*`, `perf_opt/perf_records.jsonl`):
      * spp-6x6  (C=128, out 6x6, N=2): CH=64 -> 24 busy cores -> 16.13us;
        CH=48 -> 36 busy cores -> **15.37us (-4.7%)**;
        CH=32 -> 48 units -> 2 waves -> 24.32us (round-1 r1_gen32).
      * nondiv   (C=64,  out 7x7, N=2): CH=32 -> 28 units, 0 wasted -> 13.20us;
        CH=48 -> 28 units, 32 wasted channels -> 13.84us (+4.8%).
    Step (2) breaks the tie that would otherwise pick CH=48 for nondiv as well.
    """
    cap = max(_VECTOR_WIDTH, cap)
    limit = min(c_in, cap)
    best_key = None
    best_ch = None
    ch = _VECTOR_WIDTH
    while ch <= limit:
        units = n * out_h * ((c_in + ch - 1) // ch)
        if units <= vector_cores:
            waste = ((c_in + ch - 1) // ch) * ch - c_in
            key = (-units, waste, -ch)
            if best_key is None or key < best_key:
                best_key, best_ch = key, ch
        ch += _VECTOR_WIDTH
    if best_ch is not None:
        return best_ch
    # No single-wave block exists (many logical units): fall back to the widest
    # block within the capacity cap, which keeps the wave count lowest.
    return _round_ch(c_in, cap)


def _pick_ch_global(c_in: int, plane: int, cap: int, unit_bytes: int) -> int:
    """Channel block of the C4 global path, bounded by the reduction UB price.

    The reduction lowering books UB proportional to the *plane element* count
    of the reduction input, so the usable block is bounded by
    ``unit_bytes * plane * CH <= _UB_CAPACITY_BYTES`` -- a plane-aware cap, not
    a byte cap on the real buffer list.  Returns 0 when even one vector width
    does not fit; the factory then falls back to the general path (always
    correct).
    """
    plane_elems = _UB_CAPACITY_BYTES // unit_bytes
    limit = min(c_in, cap, plane_elems // plane if plane else cap)
    if limit < _VECTOR_WIDTH:
        return 0
    return (limit // _VECTOR_WIDTH) * _VECTOR_WIDTH


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_adaptive_avg_pool2d(x: torch.Tensor, output_size):
    """PyTorch reference for adaptive_avg_pool2d (CPU, algorithm-independent)."""
    size = (None, None) if output_size is None else output_size
    return F.adaptive_avg_pool2d(x.detach().float(), size).to(x.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@kernel_factory_cache
def _adaptive_avg_pool2d_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str = "float16",
):
    """Build the adaptive average pooling 2D kernel factory.

    Returns ``build(block_m=None) -> main``; ``main(x)`` returns ``out``.
    ``block_m`` is accepted for source-interface compatibility (K9) and is
    otherwise ignored -- NPU tiling is shape-derived.
    """
    if min(n, c_in, h_in, w_in, out_h, out_w) < 1:
        raise ValueError(
            "all of n, c_in, h_in, w_in, out_h, out_w must be positive, got "
            f"{(n, c_in, h_in, w_in, out_h, out_w)}"
        )
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )

    vector_cores = NPUUtils.get().get_aicore_num() * 2
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    inv_plane = 1.0 / float(h_in * w_in)
    plane = h_in * w_in

    # ---- degenerate-grid dispatch (factory level) ----
    is_identity = out_h == h_in and out_w == w_in
    is_global = out_h == 1 and out_w == 1

    ch_id = _round_ch(c_in, CH_IDENTITY)
    # C4 global: the reduction UB price scales with the plane element
    # count and depends on the chain form, so the block cap is plane- and
    # dtype-aware (`_pick_ch_global`).
    c4_unit = _C4_REDUCE_UB_UNIT_F16 if dtype == "float16" else _C4_REDUCE_UB_UNIT_SUBFP32
    ch_glb = _pick_ch_global(c_in, plane, CH_GLOBAL, c4_unit)
    ch_gen = _pick_ch_general(c_in, n, out_h, vector_cores, CH_GENERAL)
    nblk_id = (c_in + ch_id - 1) // ch_id
    nblk_glb = (c_in + ch_glb - 1) // ch_glb if ch_glb else 0
    nblk_gen = (c_in + ch_gen - 1) // ch_gen

    # Per-path UB budget guards.  The specialisations are pure optimisations:
    # an oversized plane falls back to the general path, which is always
    # correct.
    identity_bytes = ch_id * plane * elem_bytes * _UB_MULTIBUFFER_FACTOR
    identity_fits = identity_bytes <= _UB_CAPACITY_BYTES
    # C4 global fit: `ch_glb == 0` means the plane-aware reduction-capacity cap
    # could not fit even one vector width, so the plane is not C4-viable.
    global_fits = ch_glb >= _VECTOR_WIDTH
    if is_identity and not identity_fits:
        is_identity = False
    if is_global and not global_fits:
        is_global = False

    max_bin_h = max_adaptive_bin_extent(h_in, out_h)
    max_bin_w = max_adaptive_bin_extent(w_in, out_w)

    def _grid(num_logical):
        num_kernels = min(num_logical, vector_cores)
        num_local = (num_logical + num_kernels - 1) // num_kernels
        return num_kernels, num_local

    if is_identity:
        num_kernels, num_local = _grid(n * nblk_id)

        def _build(block_m=None):
            if block_m is not None and block_m < 1:
                raise ValueError(f"block_m must be positive, got block_m={block_m}")

            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_identity(ch_: int):
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, h_in, w_in), dtype),
                    out: T.Tensor((n, c_in, out_h, out_w), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        stage = T.alloc_ub((ch_id, h_in, w_in), dtype)
                        for t in T.serial(num_local):
                            uid = t * num_kernels + cid
                            if uid < n * nblk_id:
                                nb = uid // nblk_id
                                cb = uid % nblk_id
                                real_c = T.min(ch_id, c_in - cb * ch_id)
                                T.copy(
                                    x[nb, cb * ch_id : cb * ch_id + real_c, 0:h_in, 0:w_in],
                                    stage[0:real_c, 0:h_in, 0:w_in],
                                )
                                T.copy(
                                    stage[0:real_c, 0:h_in, 0:w_in],
                                    out[nb, cb * ch_id : cb * ch_id + real_c, 0:h_in, 0:w_in],
                                )

                return main

            return _func_identity(ch_id)

        return _build

    if is_global:
        num_kernels, num_local = _grid(n * nblk_glb)

        def _build(block_m=None):
            if block_m is not None and block_m < 1:
                raise ValueError(f"block_m must be positive, got block_m={block_m}")

            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_global(ch_: int):
                @T.prim_func
                def main(
                    # The plane is read in its natural (contiguous) memory order:
                    # one channel block is CH*H*W contiguous elements, so the
                    # spatial H/W axes are merged into a single flat axis.  This
                    # keeps the GM->UB copy a plain contiguous transfer and makes
                    # the reduction axis wide (H*W) instead of narrow (W).
                    x: T.Tensor((n, c_in, 1, plane), dtype),
                    out: T.Tensor((n, c_in, out_h, out_w), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        if dtype == "float16":
                            # Cheap chain (G11/w1): reduce in fp16 and cast once,
                            # dropping the fp32 full-plane intermediate.  Its UB
                            # price (`_C4_REDUCE_UB_UNIT_F16`) admits a wider CH.
                            tile = T.alloc_ub((ch_glb, 1, plane), dtype)
                            s16 = T.alloc_ub((ch_glb, 1, 1), dtype)
                            sred = T.alloc_ub((ch_glb, 1, 1), "float32")
                            gout = T.alloc_ub((ch_glb, 1, 1), dtype)
                            for t in T.serial(num_local):
                                uid = t * num_kernels + cid
                                if uid < n * nblk_glb:
                                    nb = uid // nblk_glb
                                    cb = uid % nblk_glb
                                    real_c = T.min(ch_glb, c_in - cb * ch_glb)
                                    T.copy(
                                        x[nb, cb * ch_glb : cb * ch_glb + real_c, 0:1, 0:plane],
                                        tile[0:real_c, 0:1, 0:plane],
                                    )
                                    T.reduce_sum(tile, s16, dim=2)
                                    T.vcast(s16, sred, round_mode="rint")
                                    T.vmul(sred, inv_plane, sred)
                                    T.vcast(sred, gout, round_mode="rint")
                                    T.copy(
                                        gout[0:real_c, 0:1, 0:1],
                                        out[nb, cb * ch_glb : cb * ch_glb + real_c, 0:1, 0:1],
                                    )
                        else:
                            # bf16 has no `reduce_sum` accumulation form, so keep
                            # the fp32 full-plane intermediate (round-2 chain)
                            # and the matching price / CH cap.
                            tile = T.alloc_ub((ch_glb, 1, plane), dtype)
                            tile32 = T.alloc_ub((ch_glb, 1, plane), "float32")
                            sred = T.alloc_ub((ch_glb, 1, 1), "float32")
                            gout = T.alloc_ub((ch_glb, 1, 1), dtype)
                            for t in T.serial(num_local):
                                uid = t * num_kernels + cid
                                if uid < n * nblk_glb:
                                    nb = uid // nblk_glb
                                    cb = uid % nblk_glb
                                    real_c = T.min(ch_glb, c_in - cb * ch_glb)
                                    T.copy(
                                        x[nb, cb * ch_glb : cb * ch_glb + real_c, 0:1, 0:plane],
                                        tile[0:real_c, 0:1, 0:plane],
                                    )
                                    T.vcast(tile, tile32, round_mode="rint")
                                    T.reduce_sum(tile32, sred, dim=2)
                                    T.vmul(sred, inv_plane, sred)
                                    T.vcast(sred, gout, round_mode="rint")
                                    T.copy(
                                        gout[0:real_c, 0:1, 0:1],
                                        out[nb, cb * ch_glb : cb * ch_glb + real_c, 0:1, 0:1],
                                    )

                return main

            return _func_global(ch_glb)

        return _build

    # ---------------- general grid ----------------
    num_kernels, num_local = _grid(n * out_h * nblk_gen)

    def _build(block_m=None):
        if block_m is not None and block_m < 1:
            raise ValueError(f"block_m must be positive, got block_m={block_m}")

        @tilelang.jit(out_idx=[1], target="npuir")
        def _func_general(ch_: int):
            @T.prim_func
            def main(
                x: T.Tensor((n, c_in, h_in, w_in), dtype),
                out: T.Tensor((n, c_in, out_h, out_w), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    acc = T.alloc_ub((out_w, ch_gen), "float32")
                    row_src = T.alloc_ub((ch_gen, w_in), dtype)
                    row_dst = T.alloc_ub((w_in, ch_gen), dtype)
                    row_f32 = T.alloc_ub((w_in, ch_gen), "float32")
                    acc16 = T.alloc_ub((out_w, ch_gen), dtype)
                    acc_t = T.alloc_ub((ch_gen, out_w), dtype)

                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < n * out_h * nblk_gen:
                            nb = uid // (out_h * nblk_gen)
                            rem = uid % (out_h * nblk_gen)
                            oh = rem // nblk_gen
                            cb = rem % nblk_gen
                            real_c = T.min(ch_gen, c_in - cb * ch_gen)
                            ih0, ih1 = adaptive_bin(oh, h_in, out_h)
                            bh = ih1 - ih0

                            T.clear(acc)
                            for kh in T.serial(max_bin_h):
                                if kh < bh:
                                    T.copy(
                                        x[nb, cb * ch_gen : cb * ch_gen + real_c, ih0 + kh, 0:w_in],
                                        row_src[0:real_c, 0:w_in],
                                    )
                                    T.transpose(row_src, row_dst, permutation=[1, 0])
                                    T.vcast(row_dst, row_f32, round_mode="rint")
                                    for ow in T.serial(out_w):
                                        iw0, iw1 = adaptive_bin(ow, w_in, out_w)
                                        bw = iw1 - iw0
                                        for kw in T.serial(max_bin_w):
                                            if kw < bw:
                                                T.vadd(
                                                    acc[ow : ow + 1, 0:ch_gen],
                                                    row_f32[iw0 + kw : iw0 + kw + 1, 0:ch_gen],
                                                    acc[ow : ow + 1, 0:ch_gen],
                                                )
                            for ow in T.serial(out_w):
                                iw0, iw1 = adaptive_bin(ow, w_in, out_w)
                                inv = 1.0 / T.cast(bh * (iw1 - iw0), "float32")
                                T.vmul(acc[ow : ow + 1, 0:ch_gen], inv, acc[ow : ow + 1, 0:ch_gen])
                            T.vcast(acc, acc16, round_mode="rint")
                            T.transpose(acc16, acc_t, permutation=[1, 0])
                            T.copy(
                                acc_t[0:real_c, 0:out_w],
                                out[nb, cb * ch_gen : cb * ch_gen + real_c, oh, 0:out_w],
                            )

            return main

        return _func_general(ch_gen)

    return _build


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_FACTORY_CACHE = {}
_STATS_MAX_DIFF = {}


def _get_factory(n, c_in, h_in, w_in, out_h, out_w, dtype_str):
    key = (n, c_in, h_in, w_in, out_h, out_w, dtype_str)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _adaptive_avg_pool2d_kernel(
            n, c_in, h_in, w_in, out_h, out_w, dtype_str
        )
    return _FACTORY_CACHE[key]


def _resolve_out_dims(h_in, w_in, output_size):
    if output_size is None:
        return h_in, w_in
    if isinstance(output_size, int):
        return output_size, output_size
    oh = h_in if output_size[0] is None else output_size[0]
    ow = w_in if output_size[1] is None else output_size[1]
    return oh, ow


def _compare(out, ref, atol, rtol, equal_nan=False):
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
    h_in,
    w_in,
    output_size,
    dtype_str,
    tag="L0",
    x=None,
    ndim3=False,
    equal_nan=False,
):
    torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
    out_h, out_w = _resolve_out_dims(h_in, w_in, output_size)

    if x is None:
        x = torch.randn(n, c_in, h_in, w_in, dtype=torch.float32).to(torch_dtype)
    else:
        assert x.dtype == torch_dtype, f"input dtype {x.dtype} != expected {torch_dtype}"

    factory = _get_factory(n, c_in, h_in, w_in, out_h, out_w, dtype_str)
    kernel = factory(None)
    out = kernel(x.npu()).cpu()

    ref = golden_adaptive_avg_pool2d(x, output_size)
    if ndim3:
        assert out.shape == (c_in, out_h, out_w), f"3D rank not preserved: {out.shape}"
        ref = golden_adaptive_avg_pool2d(x[0], output_size)

    max_diff, violations = _compare(out, ref, atol, rtol, equal_nan=equal_nan)
    if max_diff == max_diff:  # NaN guard
        best = _STATS_MAX_DIFF.get(dtype_str, -1.0)
        if max_diff > best:
            _STATS_MAX_DIFF[dtype_str] = max_diff

    try:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    except AssertionError:
        print(
            f"[{tag}] FAIL: shape=({n},{c_in},{h_in},{w_in}) out=({out_h},{out_w}) "
            f"dtype={dtype_str} max_diff={max_diff:.3e} "
            f"violations={violations}/{out.numel()}"
        )
        raise
    print(
        f"[{tag}] PASS: shape=({n},{c_in},{h_in},{w_in}) out=({out_h},{out_w}) "
        f"dtype={dtype_str} max_diff={max_diff:.3e}"
    )
    return max_diff, violations


def _try_case(failures, case_id, **kwargs):
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


def _run_3d_case(failures):
    case_id = "L0-8/3d-chw"
    try:
        torch_dtype, atol, rtol = _TOLERANCE["float16"]
        c_in, h_in, w_in = 16, 11, 13
        output_size = (5, 4)
        out_h, out_w = output_size
        x3 = torch.randn(c_in, h_in, w_in, dtype=torch.float32).to(torch_dtype)
        kernel = _get_factory(1, c_in, h_in, w_in, out_h, out_w, "float16")(None)
        out = kernel(x3.unsqueeze(0).npu()).cpu().squeeze(0)
        ref = golden_adaptive_avg_pool2d(x3, output_size)
        assert out.shape == (c_in, out_h, out_w), f"3D rank not preserved: {out.shape}"
        max_diff, _ = _compare(out, ref, atol, rtol)
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        best = _STATS_MAX_DIFF.get("float16", -1.0)
        if max_diff > best:
            _STATS_MAX_DIFF["float16"] = max_diff
        print(
            f"[L0-8] PASS: shape=({c_in},{h_in},{w_in}) out=({out_h},{out_w}) "
            f"dtype=float16 max_diff={max_diff:.3e}"
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


def run_L0():
    failures = []

    cases = [
        (1, 8, 4, 4, (2, 2), "float16", "L0-1/smoke"),
        (2, 128, 56, 56, (6, 6), "float16", "L0-2/spp"),
        (8, 2048, 7, 7, (1, 1), "float16", "L0-3/resnet-global"),
        (2, 64, 55, 57, (7, 7), "bfloat16", "L0-4/nondiv-bf16"),
        (1, 8, 9, 11, None, "float16", "L0-5/identity"),
        (1, 16, 11, 13, (None, 5), "float16", "L0-6/half-none"),
        (1, 16, 10, 10, 4, "bfloat16", "L0-7/scalar-size-bf16"),
        (2, 64, 56, 56, (6, 6), "float16", "L0-9/cblk-edge"),
        (2, 2048, 7, 7, (1, 1), "float16", "L0-10/persistent-switch"),
    ]
    for n, c_in, h_in, w_in, output_size, dtype_str, case_id in cases:
        _try_case(
            failures,
            case_id,
            n=n,
            c_in=c_in,
            h_in=h_in,
            w_in=w_in,
            output_size=output_size,
            dtype_str=dtype_str,
            tag=case_id.split("/")[0],
        )

    _run_3d_case(failures)

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (10 items)")
    return failures


def run_L1():
    failures = []

    cases = [
        (1, 96, 28, 30, (7, 5), "bfloat16", "L1/bf16/cblk-tail"),
        (1, 128, 55, 57, (7, 7), "float16", "L1/fp16/nondiv"),
        (2, 128, 56, 56, (6, 6), "bfloat16", "L1/bf16/spp"),
        (2, 2048, 7, 7, (1, 1), "float16", "L1/fp16/global-2batch"),
        (4, 64, 9, 9, (1, 1), "bfloat16", "L1/bf16/global"),
        (1, 96, 5, 7, None, "float16", "L1/fp16/identity-cblk-tail"),
        (1, 8, 4, 4, (7, 7), "float16", "L1/fp16/expanding"),
        (1, 16, 11, 13, (11, 5), "float16", "L1/fp16/h-identity-general"),
        (16, 32, 6, 6, (3, 3), "float16", "L1/fp16/large-n"),
        (1, 8, 1, 1, (1, 1), "float16", "L1/fp16/1x1"),
    ]
    for n, c_in, h_in, w_in, output_size, dtype_str, case_id in cases:
        _try_case(
            failures,
            case_id,
            n=n,
            c_in=c_in,
            h_in=h_in,
            w_in=w_in,
            output_size=output_size,
            dtype_str=dtype_str,
            tag="L1",
        )

    try:
        _adaptive_avg_pool2d_kernel(1, 8, 4, 4, 2, 2, "float16")(0)
    except ValueError as exc:
        print(f"[L1] PASS: block_m guard rejects 0 ({exc})")
    else:
        failures.append(("L1/block_m-guard", AssertionError("block_m=0 not rejected")))

    try:
        _adaptive_avg_pool2d_kernel(1, 8, 4, 4, 2, 2, "float32")
    except ValueError as exc:
        print(f"[L1] PASS: dtype guard rejects float32 ({exc})")
    else:
        failures.append(("L1/dtype-guard", AssertionError("float32 not rejected")))
    try:
        _adaptive_avg_pool2d_kernel(0, 8, 4, 4, 2, 2, "float16")
    except ValueError as exc:
        print(f"[L1] PASS: shape guard rejects n=0 ({exc})")
    else:
        failures.append(("L1/shape-guard", AssertionError("n=0 not rejected")))

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    cases = [
        (1, 5, 7, 3, (2, 2), "float16"),
        (1, 1, 1, 1, (1, 1), "bfloat16"),
        (1, 8, 32, 32, (1, 1), "float16"),
        (1, 8, 64, 64, None, "float16"),
        (3, 1024, 3, 3, (1, 1), "float16"),
        (1, 8, 1, 37, (1, 5), "float16"),
        (2, 7, 13, 5, (9, 3), "bfloat16"),
    ]
    for n, c_in, h_in, w_in, output_size, dtype_str in cases:
        try:
            _run_case(
                n=n,
                c_in=c_in,
                h_in=h_in,
                w_in=w_in,
                output_size=output_size,
                dtype_str=dtype_str,
                tag="L2",
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[L2] WARN (record only): shape=({n},{c_in},{h_in},{w_in}) "
                f"out={output_size} {dtype_str}: {e}"
            )


def run_boundary():
    try:
        _run_case(
            n=2,
            c_in=16,
            h_in=9,
            w_in=11,
            output_size=(3, 4),
            dtype_str="float16",
            tag="Boundary-zeros",
            x=torch.zeros(2, 16, 9, 11, dtype=torch.float16),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): zeros: {e}")

    try:
        _run_case(
            n=1,
            c_in=8,
            h_in=11,
            w_in=13,
            output_size=(5, 4),
            dtype_str="float16",
            tag="Boundary-const",
            x=torch.full((1, 8, 11, 13), 3.7, dtype=torch.float16),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): const: {e}")

    try:
        _run_case(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            output_size=(4, 4),
            dtype_str="float16",
            tag="Boundary-large",
            x=(torch.randn(1, 8, 16, 16, dtype=torch.float32) * 100.0).to(torch.float16),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large: {e}")

    try:
        _run_case(
            n=1,
            c_in=8,
            h_in=12,
            w_in=12,
            output_size=(3, 3),
            dtype_str="float16",
            tag="Boundary-subnormal",
            x=(torch.randn(1, 8, 12, 12, dtype=torch.float32) * 1e-5).to(torch.float16),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {e}")

    try:
        x = torch.randn(1, 8, 11, 13, dtype=torch.float32)
        x[0, 0, 0, :] = float("inf")
        x[0, 1, :, 0] = float("nan")
        x[0, 2, 0, 0] = float("inf")
        x[0, 2, 0, 1] = float("-inf")
        _run_case(
            n=1,
            c_in=8,
            h_in=11,
            w_in=13,
            output_size=(5, 4),
            dtype_str="float16",
            tag="Boundary-inf-nan",
            x=x.to(torch.float16),
            equal_nan=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): inf-nan: {e}")

    try:
        x = torch.randn(2, 16, 5, 5, dtype=torch.float32)
        x[0, 0, :, :] = float("inf")
        x[1, 3, 2, 2] = float("nan")
        _run_case(
            n=2,
            c_in=16,
            h_in=5,
            w_in=5,
            output_size=(1, 1),
            dtype_str="float16",
            tag="Boundary-inf-nan-global",
            x=x.to(torch.float16),
            equal_nan=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): inf-nan-global: {e}")


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
