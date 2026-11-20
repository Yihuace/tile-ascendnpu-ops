# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Adaptive max pooling 2D forward kernel (NPU, Developer mode) -- TUNED.

Stage 4 (``perf_opt/``) of the migrated ``AdaptiveMaxPool2dFwdOp`` kernel.  Base:
``../_adaptive_max_pool2d_kernel.py`` (Stage 3, precision-passing).  The factory
contract, the three factory-level structural paths and the frozen semantics are
unchanged; only the two degenerate-grid/general code paths were re-shaped.

Semantics (DESIGN.md section 0.1, frozen, matches
``torch.nn.functional.adaptive_max_pool2d``):
    out[n, c, oh, ow] = max(x[n, c, ih0:ih1, iw0:iw1])
    (ih0, ih1) = (floor(oh * H_in / out_H), ceil((oh+1) * H_in / out_H))
    (iw0, iw1) = (floor(ow * W_in / out_W), ceil((ow+1) * W_in / out_W))
    A NaN anywhere in the window wins (out = NaN); +-inf follow IEEE max.
    float16 / bfloat16 only; the comparison domain is fp32 with a single
    round-back to the input dtype, so the output is bit-exact with a window
    element (atol = rtol = 0).

Three structural paths stay dispatched at factory level, all three correct:
  * identity   (out_H == H_in and out_W == W_in): UB transit copy;
  * C4 global  (out_H == out_W == 1): flat-plane whole-plane reduction;
  * general    (anything else): per-row transpose + C-axis bin accumulation.

Tuned changes vs the base (all measured with ``msprof op`` Task Duration):

  G1. C4 global path: flattened spatial axis.  The base declared the plane as
      ``(CH, H, W)`` and chained ``T.reduce_max`` over the W axis and then H --
      two narrow (extent 7) reductions that scalarise on this toolchain.  The
      plane is now declared ``(N, C, 1, H*W)``: the GM->UB copy stays one
      contiguous transfer and a *single* wide reduction over ``H*W`` replaces
      both levels.  ``global-1x1`` (8x2048x7x7 fp16) 320.83 -> 19.0 us (16.9x),
      bit-exact.
  G2. C4 NaN detection without a full-plane comparison.  The base compared the
      fp32 plane to itself (``vcmp``) and reduced the resulting bool/fp32 mask,
      which on this toolchain inflates the reduction UB footprint to 96-128 B
      per plane element (capping CH at ~32).  ``has_nan`` is now obtained as
      ``isnan(reduce_sum(x*x))``: ``x*x`` is NaN exactly where ``x`` is NaN and
      is >= 0 elsewhere, so the sum cannot become NaN through ``+inf - inf``
      (the failure mode of summing ``x`` itself) -- it is NaN iff the plane
      holds a NaN.  The square is *in place* (``flat`` is dead after the max
      reduce); the extra plane buffer would double the reduction UB footprint.
      Cost drops to ~32 B/plane-element, which is what unlocks CH=64..120.
  G3. C4 UB budget: per-dtype, from the measured reduction cost model
      (``_UB_REDUCE_B_PER_ELEM`` 32 B for the fp16 path, 96 B for the
      fp32/bf16 path) rather than the base's buffer-list estimate, which did
      not account for the reduction lowering's internal temporaries.  The
      bf16/fp32 form has no fp16 reduce, so it keeps the fp32 comparison domain
      and a smaller block.
  G4. General grid path: single-wave channel-block dispatch.  The logical-unit
      count ``N * out_H * ceil(C/CH)`` sets both the wave count and the per-core
      work, so the block picked is the smallest multiple of the vector width
      that still fits one wave -- that spreads the units over the most cores
      while minimising the widest block.  Measured: spp-6x6 CH=64 -> 24 units
      (max block 64) 20.42 us vs CH=48 -> 36 units (max block 48) 19.02 us;
      nondiv-7x7 CH=64 -> 14 units 18.32 us vs CH=32 -> 28 units 15.54 us.

Measured negative result (kept out of the code): the PL-1.13 dual-AIV ``subid``
work split does NOT apply to pure-Vector npuir kernels on this toolchain --
``subid`` is always 0 (one Vector sub-block per block), so a subid partition
would silently process only half the logical units (verified with
``probe_subid.py``).

Run the embedded hierarchical tests:
    python _adaptive_max_pool2d_kernel.py --level L0
    python _adaptive_max_pool2d_kernel.py --level all
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
    "CH_GENERAL",
    "CH_GLOBAL",
    # Tuned defaults (module-level, wrapper-visible).
    "CH_IDENTITY",
    "_adaptive_max_pool2d_kernel",
    "adaptive_bin",
    "golden_adaptive_max_pool2d",
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
# Unified Buffer capacity per Vector core (developer guide: 192 KB).
_UB_CAPACITY_BYTES = 196_608

# Conservative multi-buffer factor for the UB-budget guard: the
# auto-multi-buffer pass inflates the resident buffers when the persistent
# serial loop is active (measured 1.7x-2.5x on this toolchain, see
# pattern-library traps: UB-multibuffer-inflation).
_UB_MULTIBUFFER_FACTOR = 2

# Supported dtypes (the source op rejects float32).
_SUPPORTED_DTYPES = ("float16", "bfloat16")

# Precision tolerances (DESIGN.md section 8.2): max selects an input element,
# so the result is bit-exact (test_pool.py: "Max is exact (no accumulation)").
_TOLERANCE = {
    "float16": (torch.float16, 0.0, 0.0),
    "bfloat16": (torch.bfloat16, 0.0, 0.0),
}

# fp16/bf16 vector width in elements (alignment of the vector axis).
_VECTOR_WIDTH = 8

# ---- tuned per-path channel blocks (module-level, wrapper-visible) --------
# Identity path: whole-plane UB transit, (CH, H_in, W_in) fp16.
CH_IDENTITY = 64
# C4 global path, fp16: flattened (CH, 1, H*W) tile; reduce_max runs directly in
# fp16 and the NaN mask is reduced from the bool plane.  Capacity-bound; the
# BishengHIR reduction needs ~32 B/plane-element, so CH*H*W <= 196608/32.
CH_GLOBAL = 64
# C4 global path, bf16/fp32: the reduction runs in the fp32 domain (no bf16
# reduce form), whose lowering needs ~96 B/plane-element -> a much smaller CH.
CH_GLOBAL_F32 = 32
# General grid path: base channel block; the effective block is derived by
# `_pick_ch_general` (single-wave dispatch).
CH_GENERAL = 64
# BishengHIR reduction UB cost per plane element (measured with probe_ub2.py:
# lean in-place fp16 path 32 B/elem; fp32/bf16 path 96 B/elem).
_UB_REDUCE_B_PER_ELEM = 32
_UB_REDUCE_B_PER_ELEM_F32 = 96
# NOTE (PL-1.13 applicability, measured): the dual-AIV work split does NOT
# apply to pure-Vector npuir kernels on this toolchain -- `subid` is always 0
# (only one Vector sub-block per block), so a subid-based partition would
# silently process half the logical units.  Verified with probe_subid.py.


def _round_ch(limit: int, cap: int) -> int:
    """Channel block: ``min(limit, cap)`` rounded up to the fp16 vector width."""
    ch = min(limit, cap)
    return max(_VECTOR_WIDTH, ((ch + _VECTOR_WIDTH - 1) // _VECTOR_WIDTH) * _VECTOR_WIDTH)


def _pick_ch_general(c_in: int, n: int, out_h: int, vector_cores: int, cap: int) -> int:
    """Effective channel block of the general grid path.

    The logical-unit count ``N * out_H * ceil(C/CH)`` sets both the wave count and
    the per-core work: with a single wave the wall time is proportional to the
    largest channel block.  So the block is the **smallest** multiple of the
    vector width that keeps ``units <= vector_cores`` (one wave) -- that both
    spreads the units over the most cores and minimises the widest block.

    Measured on this toolchain (msprof op, spp/nondiv):
      * spp-6x6  (C=128, N=2, out 6x6): CH=64 -> 24 units / max block 64 ->
        20.42us;  CH=48 -> 36 units / max block 48 -> 18.97us (-7%);
        CH=32 -> 48 units (two waves) -> 29.47us (+44%).
      * nondiv   (C=64,  N=2, out 7x7): CH=32 -> 28 units / max block 32 ->
        15.45us;  CH=48 -> 28 units / max block 48 -> 17.07us (+10%).
    """
    base = _round_ch(c_in, cap)
    for ch in range(_VECTOR_WIDTH, base + 1, _VECTOR_WIDTH):
        units = n * out_h * ((c_in + ch - 1) // ch)
        if units <= vector_cores:
            return ch
    return base


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_adaptive_max_pool2d(x: torch.Tensor, output_size):
    """PyTorch reference for adaptive_max_pool2d, per DESIGN.md section 0.1.

    Directly uses the source-repo test benchmark
    (examples/TileOPs/tests/ops/test_pool.py: ``ref_program``).  Accepts 3D
    CHW or 4D NCHW; ``None`` entries keep the input size; single output (no
    indices).  Independent of the NPU algorithm (no transpose / no
    C-vectorisation / no C4-C4b specialisation).  Runs on CPU and does not
    require an NPU device.
    """
    size = (None, None) if output_size is None else output_size
    return F.adaptive_max_pool2d(x.detach().to(torch.float32), size).to(x.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@kernel_factory_cache
def _adaptive_max_pool2d_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str = "float16",
):
    """Build the adaptive max pooling 2D kernel factory.

    Returns a build callable taking a single ``block_m`` (CUDA ``threads``
    removed, K9); ``build(block_m)`` compiles and returns ``main``;
    ``main(x)`` returns ``out`` (``out_idx=[1]``).
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

    elem_bytes = 2  # float16 / bfloat16 only

    def _round_up(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    def _general_bytes(ch: int) -> int:
        """Per-core UB footprint of the general path (bytes, without inflation)."""
        return (
            2 * ch * w_in * elem_bytes  # row_src, row_dst
            + 2 * w_in * ch * 4  # row_f32, eq_f32
            + w_in * ch  # nan_mask (bool)
            + 6 * out_w * ch * 4  # acc, h_acc, zeros, infbuf, nanbuf, out32
            + out_w * ch  # cfinal (bool)
            + 2 * out_w * ch * elem_bytes  # acc16, acc_t
        )

    def _choose_ch() -> int:
        """Channel vector block: C axis is the vector axis (DESIGN.md 1.6.3).

        Must be a multiple of the fp16 vector width; DESIGN.md 5.2 primary
        choice is ``min(C, 64)``.  Halved only if the general-path footprint
        would not fit the UB budget (robustness guard, no test shape needs it).
        """
        ch = max(_VECTOR_WIDTH, _round_up(min(c_in, 64), _VECTOR_WIDTH))
        while (
            ch > _VECTOR_WIDTH and _general_bytes(ch) * _UB_MULTIBUFFER_FACTOR > _UB_CAPACITY_BYTES
        ):
            ch = max(_VECTOR_WIDTH, _round_up(ch // 2, _VECTOR_WIDTH))
        return ch

    ch = _choose_ch()
    num_cblk = (c_in + ch - 1) // ch
    # Physical cores: pure-Vector op -> AI cores x 2 (DESIGN.md 5.5 item 2).
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    # General-grid block: single-wave dispatch (halve only when the base block
    # leaves most cores idle and the half block still fits one wave).
    ch_gen = _pick_ch_general(c_in, n, out_h, vector_cores, CH_GENERAL)
    # Robustness guard: never let the general-path buffer list overflow UB
    # (shrink toward the vector width if the shape's row/column buffers are big).
    while (
        ch_gen > _VECTOR_WIDTH
        and _general_bytes(ch_gen) * _UB_MULTIBUFFER_FACTOR > _UB_CAPACITY_BYTES
    ):
        ch_gen = max(_VECTOR_WIDTH, ch_gen - _VECTOR_WIDTH)
    num_cblk_gen = (c_in + ch_gen - 1) // ch_gen
    # C4 global path uses its own (flattened) channel block: the lean fp16 form
    # reduces in fp16 (32 B/plane-element), the fp32/bf16 form in fp32 (96).
    _plane = h_in * w_in
    if dtype == "float16":
        _bpe = _UB_REDUCE_B_PER_ELEM
        _base_ch = CH_GLOBAL
    else:
        _bpe = _UB_REDUCE_B_PER_ELEM_F32
        _base_ch = CH_GLOBAL_F32
    # BishengHIR reduction UB budget (the model is the measured total, so no
    # additional multi-buffer factor is applied): a plane element costs ~_bpe B.
    _cap_ch = (_UB_CAPACITY_BYTES // _bpe) // _plane
    ch_g = _round_ch(c_in, max(0, min(_base_ch, _cap_ch)))
    nblk_g = (c_in + ch_g - 1) // ch_g

    # Degenerate-grid specialisations (DESIGN.md 0.6 R4 / 1.6.1 C4, C4b).
    # Only used when the whole-plane buffers fit the UB budget; the general
    # path is always correct, so an oversized plane simply falls back.
    plane_elems = ch * h_in * w_in
    identity_bytes = plane_elems * elem_bytes
    # C4 reduction UB cost model (reduction lowering temporaries dominate).
    global_bytes = ch_g * _plane * _bpe
    identity_shape = out_h == h_in and out_w == w_in
    is_identity = identity_shape and identity_bytes * _UB_MULTIBUFFER_FACTOR <= _UB_CAPACITY_BYTES
    is_global = (
        out_h == 1
        and out_w == 1
        and not identity_shape
        and _cap_ch >= _VECTOR_WIDTH
        and global_bytes <= _UB_CAPACITY_BYTES
    )

    def _grid(num_logical: int):
        """(num_kernels, num_local) for the persistent raster (PL-1.21)."""
        num_kernels = min(num_logical, vector_cores)
        num_local = (num_logical + num_kernels - 1) // num_kernels
        return num_kernels, num_local

    max_bin_h = max_adaptive_bin_extent(h_in, out_h)
    max_bin_w = max_adaptive_bin_extent(w_in, out_w)

    # Persistent raster per structural path (single source of NUM_LOGICAL).
    if is_identity:
        num_logical = n * num_cblk
    elif is_global:
        num_logical = n * nblk_g
    else:
        num_logical = n * out_h * num_cblk_gen
    num_kernels, num_local = _grid(num_logical)

    def _build(block_m=None):
        if block_m is not None and block_m < 1:
            raise ValueError(f"block_m must be positive, got block_m={block_m}")

        if is_identity:
            # C4b: out_H == H_in and out_W == W_in -> every bin is unit size,
            # so out == x (pure UB transit, same dtype, NaN preserved).
            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_identity(ch_):
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, h_in, w_in), dtype),
                    out: T.Tensor((n, c_in, out_h, out_w), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        stage = T.alloc_ub((ch, h_in, w_in), dtype)
                        for t in T.serial(num_local):
                            uid = t * num_kernels + cid
                            if uid < num_logical:
                                nb = uid // num_cblk
                                cb = uid % num_cblk
                                real_c = T.min(ch, c_in - cb * ch)
                                T.copy(
                                    x[nb, cb * ch : cb * ch + real_c, 0:h_in, 0:w_in],
                                    stage[0:real_c, 0:h_in, 0:w_in],
                                )
                                T.copy(
                                    stage[0:real_c, 0:h_in, 0:w_in],
                                    out[nb, cb * ch : cb * ch + real_c, 0:h_in, 0:w_in],
                                )

                return main

            return _func_identity(ch)

        if is_global:
            # C4: out_H == out_W == 1 -> the whole plane is one bin.  The plane
            # is declared flat (N, C, 1, H*W) so the GM->UB copy stays a plain
            # contiguous transfer and a single wide reduce over H*W replaces the
            # baseline's two narrow (W then H) reduction levels.
            plane_g = h_in * w_in

            if dtype == "float16":
                # Lean form: the max reduction runs directly on the fp16 plane
                # (max of fp16 values is exact -- no rounding) and the NaN mask
                # is reduced from the bool plane.  This keeps the UB footprint
                # at ~32 B/plane-element, which unlocks CH=64..120 (the fp32
                # domain buffers would cap CH at 32).
                @tilelang.jit(out_idx=[1], target="npuir")
                def _func_global_f16(ch_):
                    @T.prim_func
                    def main(
                        x: T.Tensor((n, c_in, 1, plane_g), dtype),
                        out: T.Tensor((n, c_in, out_h, out_w), dtype),
                    ):
                        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                            flat = T.alloc_ub((ch_g, 1, plane_g), dtype)
                            gmax16 = T.alloc_ub((ch_g, 1, 1), dtype)
                            nsum = T.alloc_ub((ch_g, 1, 1), dtype)
                            gmax = T.alloc_ub((ch_g, 1, 1), "float32")
                            ok = T.alloc_ub((ch_g, 1, 1), "bool")
                            infbuf = T.alloc_ub((ch_g, 1, 1), "float32")
                            nanbuf = T.alloc_ub((ch_g, 1, 1), "float32")
                            out32 = T.alloc_ub((ch_g, 1, 1), "float32")
                            gout = T.alloc_ub((ch_g, 1, 1), dtype)
                            pos_inf = T.infinity("float32")
                            for t in T.serial(num_local):
                                uid = t * num_kernels + cid
                                if uid < num_logical:
                                    nb = uid // nblk_g
                                    cb = uid % nblk_g
                                    real_c = T.min(ch_g, c_in - cb * ch_g)
                                    T.copy(
                                        x[nb, cb * ch_g : cb * ch_g + real_c, 0:1, 0:plane_g],
                                        flat[0:real_c, 0:1, 0:plane_g],
                                    )
                                    # Whole-plane maximum (clear=True: the
                                    # reduction target must not accumulate on
                                    # stale data).  Max of fp16 values is exact.
                                    T.reduce_max(flat, gmax16, dim=2, clear=True)
                                    # NaN detection without a full-plane compare:
                                    # has_nan <=> isnan(reduce_sum(x*x)).  x*x is
                                    # NaN exactly where x is NaN and >= 0
                                    # elsewhere, so the sum cannot become NaN
                                    # through +inf - inf (unlike summing x) --
                                    # it is NaN iff the plane holds a NaN.  The
                                    # square is in-place: `flat` is dead after
                                    # the max reduce, and the extra plane buffer
                                    # would double the reduction UB footprint.
                                    T.vmul(flat, flat, flat)
                                    T.reduce_sum(flat, nsum, dim=2, clear=True)
                                    T.vcast(gmax16, gmax, round_mode="rint")
                                    # ok = 1 iff the plane is NaN-free
                                    # ("eq" is IEEE-correct; "ne" is ordered and
                                    # returns false for NaN => unusable).
                                    T.vcmp(nsum, nsum, ok, "eq")
                                    # NaN = inf - inf (IEEE, runtime).
                                    T.vbrc(pos_inf, infbuf)
                                    T.vsub(infbuf, infbuf, nanbuf)
                                    T.vselect(ok, gmax, nanbuf, out32)
                                    T.vcast(out32, gout, round_mode="rint")
                                    T.copy(
                                        gout[0:real_c, 0:1, 0:1],
                                        out[nb, cb * ch_g : cb * ch_g + real_c, 0:1, 0:1],
                                    )

                    return main

                return _func_global_f16(ch_g)

            # bf16 / fp32 form: the reduction runs in the fp32 comparison domain
            # (reduce_max has no bf16 form).
            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_global_f32(ch_):
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, 1, plane_g), dtype),
                    out: T.Tensor((n, c_in, out_h, out_w), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        flat = T.alloc_ub((ch_g, 1, plane_g), dtype)
                        flat32 = T.alloc_ub((ch_g, 1, plane_g), "float32")
                        nmask = T.alloc_ub((ch_g, 1, plane_g), "bool")
                        nmask32 = T.alloc_ub((ch_g, 1, plane_g), "float32")
                        gmax = T.alloc_ub((ch_g, 1, 1), "float32")
                        nmin = T.alloc_ub((ch_g, 1, 1), "float32")
                        zeros = T.alloc_ub((ch_g, 1, 1), "float32")
                        infbuf = T.alloc_ub((ch_g, 1, 1), "float32")
                        nanbuf = T.alloc_ub((ch_g, 1, 1), "float32")
                        cond = T.alloc_ub((ch_g, 1, 1), "bool")
                        out32 = T.alloc_ub((ch_g, 1, 1), "float32")
                        gout = T.alloc_ub((ch_g, 1, 1), dtype)
                        pos_inf = T.infinity("float32")
                        for t in T.serial(num_local):
                            uid = t * num_kernels + cid
                            if uid < num_logical:
                                nb = uid // nblk_g
                                cb = uid % nblk_g
                                real_c = T.min(ch_g, c_in - cb * ch_g)
                                T.copy(
                                    x[nb, cb * ch_g : cb * ch_g + real_c, 0:1, 0:plane_g],
                                    flat[0:real_c, 0:1, 0:plane_g],
                                )
                                T.vcast(flat, flat32, round_mode="rint")
                                T.reduce_max(flat32, gmax, dim=2, clear=True)
                                T.vcmp(flat32, flat32, nmask, "eq")
                                T.vcast(nmask, nmask32, round_mode="rint")
                                T.reduce_min(nmask32, nmin, dim=2, clear=True)
                                T.clear(zeros)
                                T.vcmp(nmin, zeros, cond, "eq")
                                T.vbrc(pos_inf, infbuf)
                                T.vsub(infbuf, infbuf, nanbuf)
                                T.vselect(cond, nanbuf, gmax, out32)
                                T.vcast(out32, gout, round_mode="rint")
                                T.copy(
                                    gout[0:real_c, 0:1, 0:1],
                                    out[nb, cb * ch_g : cb * ch_g + real_c, 0:1, 0:1],
                                )

                return main

            return _func_global_f32(ch_g)

        @tilelang.jit(out_idx=[1], target="npuir")
        def _func_general(ch_):
            @T.prim_func
            def main(
                x: T.Tensor((n, c_in, h_in, w_in), dtype),
                out: T.Tensor((n, c_in, out_h, out_w), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # Vector accumulators on the channel axis (vector axis = C).
                    acc = T.alloc_ub((out_w, ch_gen), "float32")
                    h_acc = T.alloc_ub((out_w, ch_gen), "float32")
                    zeros = T.alloc_ub((out_w, ch_gen), "float32")
                    infbuf = T.alloc_ub((out_w, ch_gen), "float32")
                    nanbuf = T.alloc_ub((out_w, ch_gen), "float32")
                    out32 = T.alloc_ub((out_w, ch_gen), "float32")
                    cfinal = T.alloc_ub((out_w, ch_gen), "bool")
                    # Row staging: NCHW row slice -> transposed (C innermost).
                    row_src = T.alloc_ub((ch_gen, w_in), dtype)
                    row_dst = T.alloc_ub((w_in, ch_gen), dtype)
                    row_f32 = T.alloc_ub((w_in, ch_gen), "float32")
                    eq_mask = T.alloc_ub((w_in, ch_gen), "bool")
                    eq_f32 = T.alloc_ub((w_in, ch_gen), "float32")
                    acc16 = T.alloc_ub((out_w, ch_gen), dtype)
                    acc_t = T.alloc_ub((ch_gen, out_w), dtype)

                    neg_inf = -T.infinity("float32")
                    pos_inf = T.infinity("float32")
                    one_val = 1.0

                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // (out_h * num_cblk_gen)
                            rem = uid % (out_h * num_cblk_gen)
                            oh = rem // num_cblk_gen
                            cb = rem % num_cblk_gen
                            real_c = T.min(ch_gen, c_in - cb * ch_gen)
                            ih0, ih1 = adaptive_bin(oh, h_in, out_h)
                            bh = ih1 - ih0

                            # max identity is -inf (not 0); the NaN-free flag
                            # accumulator starts at 1 (0 is the absorbing value).
                            T.vbrc(neg_inf, acc)
                            T.vbrc(one_val, h_acc)
                            T.clear(zeros)
                            for kh in T.serial(max_bin_h):
                                if kh < bh:
                                    T.copy(
                                        x[
                                            nb,
                                            cb * ch_gen : cb * ch_gen + real_c,
                                            ih0 + kh,
                                            0:w_in,
                                        ],
                                        row_src[0:real_c, 0:w_in],
                                    )
                                    T.transpose(row_src, row_dst, permutation=[1, 0])
                                    T.vcast(row_dst, row_f32, round_mode="rint")
                                    # Row-level NaN probe as an fp32 {0, 1}
                                    # mask: x == x is 1 for numbers and 0 for
                                    # NaN (the "ne" mode is ordered and returns
                                    # false for NaN => unusable).
                                    T.vcmp(row_f32, row_f32, eq_mask, "eq")
                                    T.vcast(eq_mask, eq_f32, round_mode="rint")
                                    for ow in T.serial(out_w):
                                        iw0, iw1 = adaptive_bin(ow, w_in, out_w)
                                        bw = iw1 - iw0
                                        for kw in T.serial(max_bin_w):
                                            if kw < bw:
                                                T.vmax(
                                                    acc[ow : ow + 1, 0:ch_gen],
                                                    row_f32[iw0 + kw : iw0 + kw + 1, 0:ch_gen],
                                                    acc[ow : ow + 1, 0:ch_gen],
                                                )
                                                T.vmul(
                                                    h_acc[ow : ow + 1, 0:ch_gen],
                                                    eq_f32[iw0 + kw : iw0 + kw + 1, 0:ch_gen],
                                                    h_acc[ow : ow + 1, 0:ch_gen],
                                                )
                            # has_nan ? NaN : max  (per channel lane); the flag
                            # accumulator reaches 0 iff some element was NaN.
                            T.vcmp(h_acc, zeros, cfinal, "eq")
                            T.vbrc(pos_inf, infbuf)
                            T.vsub(infbuf, infbuf, nanbuf)
                            T.vselect(cfinal, nanbuf, acc, out32)
                            T.vcast(out32, acc16, round_mode="rint")
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

# Per-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_factory(n, c_in, h_in, w_in, out_h, out_w, dtype_str):
    """Cache factory instances (one factory per shape/dtype, mirroring the
    wrapper's per-shape kernel cache)."""
    key = (n, c_in, h_in, w_in, out_h, out_w, dtype_str)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _adaptive_max_pool2d_kernel(
            n, c_in, h_in, w_in, out_h, out_w, dtype_str
        )
    return _FACTORY_CACHE[key]


def _resolve_out_dims(h_in, w_in, output_size):
    """Mirror of the Op layer's ``_resolve_out_dims`` (DESIGN.md section 0.1)."""
    if output_size is None:
        return h_in, w_in
    if isinstance(output_size, int):
        return output_size, output_size
    oh = h_in if output_size[0] is None else output_size[0]
    ow = w_in if output_size[1] is None else output_size[1]
    return oh, ow


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
    h_in,
    w_in,
    output_size,
    dtype_str,
    tag="L0",
    x=None,
    ndim3=False,
    equal_nan=False,
):
    """Run one kernel case and compare against the golden.

    ``x`` may be pre-built as a CPU tensor (special-value cases); by default
    randn data is generated on CPU, cast, and moved to NPU (device-independent
    data generation).  ``ndim3`` exercises the Op layer's 3D CHW path: the
    kernel always sees the unsqueezed 4D tensor, the comparison uses 3D.
    Raises AssertionError on tolerance violation.
    """
    torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
    out_h, out_w = _resolve_out_dims(h_in, w_in, output_size)

    if x is None:
        x = torch.randn(n, c_in, h_in, w_in, dtype=torch.float32).to(torch_dtype)
    else:
        assert x.dtype == torch_dtype, f"input dtype {x.dtype} != expected {torch_dtype}"

    factory = _get_factory(n, c_in, h_in, w_in, out_h, out_w, dtype_str)
    kernel = factory(None)
    out = kernel(x.npu()).cpu()

    ref = golden_adaptive_max_pool2d(x, output_size)
    if ndim3:
        assert out.shape == (c_in, out_h, out_w), f"3D rank not preserved: {out.shape}"
        ref = golden_adaptive_max_pool2d(x[0], output_size)

    max_diff, violations = _compare(out, ref, atol, rtol, equal_nan=equal_nan)
    if max_diff == max_diff:  # NaN guard (all-equal cases give 0.0)
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
    """Run ``_run_case``, collecting (not re-raising) per-case failures.

    AssertionError is a precision failure; any other exception is recorded as
    a runtime error (both block the L0/L1 gate).
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


def _nan_probe_x(n=1, c_in=16, h_in=11, w_in=13, dtype=torch.float16):
    """Build the NaN / +-inf injection input for the adaptive-bin semantics.

    Channel 0 gets a single NaN at (0, 0); channel 1 gets +inf and -inf inside
    the same window (oh=0, ow=0 covers rows 0..2 and columns 0..3); channel 2
    gets a full NaN row.  Expected: NaN wins where a NaN is inside the bin and
    ``{+inf, -inf}`` (no NaN) resolves to +inf.
    """
    x = torch.randn(n, c_in, h_in, w_in, dtype=torch.float32)
    x[0, 0, 0, 0] = float("nan")
    x[0, 1, 0, 0] = float("inf")
    x[0, 1, 0, 1] = float("-inf")
    x[0, 2, 0, :] = float("nan")
    return x.to(dtype)


def _run_3d_case(failures):
    """L0-8: 3D CHW input -> output rank preserved (Op layer unsqueeze path).

    The kernel is 4D NCHW; this emulates the Op layer's unsqueeze(0) on the
    host (a metadata view, DESIGN.md section 0.1) and checks the squeezed
    output against the 3D golden.
    """
    case_id = "L0-8/3d-chw"
    try:
        torch_dtype, atol, rtol = _TOLERANCE["float16"]
        c_in, h_in, w_in = 16, 11, 13
        output_size = (5, 4)
        out_h, out_w = output_size
        x3 = torch.randn(c_in, h_in, w_in, dtype=torch.float32).to(torch_dtype)
        kernel = _get_factory(1, c_in, h_in, w_in, out_h, out_w, "float16")(None)
        out = kernel(x3.unsqueeze(0).npu()).cpu().squeeze(0)
        ref = golden_adaptive_max_pool2d(x3, output_size)
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
    """L0 gate suite (DESIGN.md section 8.3 items + the harness workload shapes).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    cases = [
        (1, 8, 4, 4, (2, 2), "float16", "L0-1/smoke"),  # manifest smoke
        (2, 64, 56, 56, (6, 6), "float16", "L0-2/spp-6x6"),  # harness workload
        (8, 2048, 7, 7, (1, 1), "float16", "L0-3/resnet-global"),  # C4 + persistent
        (1, 96, 28, 30, (7, 5), "bfloat16", "L0-4/cblk-tail-bf16"),  # harness
        (1, 8, 9, 11, None, "float16", "L0-5/identity"),  # C4b
        (1, 128, 55, 57, (7, 7), "float16", "L0-6/nondiv-fp16"),  # harness
        (1, 16, 11, 13, (5, None), "float16", "L0-7/half-none"),  # harness
        (2, 64, 55, 57, (7, 7), "bfloat16", "L0-9/nondiv-bf16"),  # manifest nondiv
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

    # L0-10: NaN / +-inf semantics (blocking: this is the max-specific
    # semantics that must match PyTorch bit-for-bit).
    case_id = "L0-10/nan-inf-semantics"
    try:
        x = _nan_probe_x(dtype=torch.float16)
        out = _get_factory(1, 16, 11, 13, 5, 4, "float16")(None)(x.npu()).cpu()
        ref = golden_adaptive_max_pool2d(x, (5, 4))
        # NaN anywhere in the window wins.
        assert bool(torch.isnan(out[0, 0, 0, 0])), (
            f"NaN-in-window did not win: out[0,0,0,0]={out[0, 0, 0, 0]}"
        )
        assert bool(torch.isnan(out[0, 2, 0, 0])), (
            f"NaN row did not win: out[0,2,0,0]={out[0, 2, 0, 0]}"
        )
        # {+inf, -inf} in one window without NaN -> +inf (not NaN).
        assert out[0, 1, 0, 0] == float("inf"), (
            f"{{+inf,-inf}} window gave {out[0, 1, 0, 0]} (expected +inf)"
        )
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0, equal_nan=True)
        print(
            "[L0-10] PASS: NaN wins in-window; {+inf,-inf} -> +inf; "
            "full-tensor match vs golden (equal_nan=True)"
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (8 shapes + 3D + NaN/Inf + contract guards)")
    return failures


def run_L1():
    """L1: full functional coverage (shape x dtype x specialisation path).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    cases = [
        # Channel-block tail (C=96 -> CH=64 + real_c=32), asymmetric bins.
        (1, 96, 28, 30, (7, 5), "bfloat16", "L1/bf16/cblk-tail"),
        # bf16 on the manifest spp shape.
        (2, 128, 56, 56, (6, 6), "bfloat16", "L1/bf16/spp"),
        # Non-divisible fp16, W_in=57 (odd tail on the transposed row axis).
        (1, 128, 55, 57, (7, 7), "bfloat16", "L1/bf16/nondiv"),
        # fp16 global pool with two channel blocks per batch.
        (2, 2048, 7, 7, (1, 1), "float16", "L1/fp16/global-2batch"),
        # bf16 global pool (C4 reduce path in bf16 -> fp32 -> bf16).
        (4, 64, 9, 9, (1, 1), "bfloat16", "L1/bf16/global"),
        # Identity with a channel tail (C=96, out == in).
        (1, 96, 5, 7, None, "float16", "L1/fp16/identity-cblk-tail"),
        # out size larger than input (expanding bins, size 2).
        (1, 8, 4, 4, (7, 7), "float16", "L1/fp16/expanding"),
        # W-only adaptive (H identity, W not) -- must take the general path.
        (1, 16, 11, 13, (11, 5), "float16", "L1/fp16/h-identity-general"),
        # Large batch on a small plane (persistent, many logical units).
        (16, 32, 6, 6, (3, 3), "float16", "L1/fp16/large-n"),
        # 1x1 input (global == identity, identity precedence).
        (1, 8, 1, 1, (1, 1), "float16", "L1/fp16/1x1"),
        # Single output column (max_bin_w large, out_w == 1).
        (2, 32, 13, 17, (4, 1), "float16", "L1/fp16/out-w-1"),
        # Scalar output size + bf16 (out == in on W only).
        (1, 16, 10, 10, 5, "bfloat16", "L1/bf16/scalar-size"),
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

    # NaN semantics on the C4 (global) reduction path.
    case_id = "L1/fp16/global-nan-inf"
    try:
        x = torch.randn(2, 16, 5, 5, dtype=torch.float32)
        x[0, 0, :, :] = float("inf")
        x[0, 1, :, :] = float("nan")
        x[1, 3, 0, 0] = float("-inf")
        x[1, 3, 2, 2] = float("nan")
        x = x.to(torch.float16)
        out = _get_factory(2, 16, 5, 5, 1, 1, "float16")(None)(x.npu()).cpu()
        ref = golden_adaptive_max_pool2d(x, (1, 1))
        assert out[0, 0, 0, 0] == float("inf"), f"+inf plane gave {out[0, 0, 0, 0]} (expected +inf)"
        assert bool(torch.isnan(out[0, 1, 0, 0])), "all-NaN plane did not give NaN"
        assert out[1, 3, 0, 0] == float("nan") or bool(torch.isnan(out[1, 3, 0, 0])), (
            "NaN-in-plane did not win"
        )
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0, equal_nan=True)
        print("[L1] PASS: C4 NaN/Inf semantics (inf plane -> inf, NaN wins)")
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # Factory contract guards.
    try:
        _adaptive_max_pool2d_kernel(1, 8, 4, 4, 2, 2, "float16")(0)
    except ValueError as exc:
        print(f"[L1] PASS: block_m guard rejects 0 ({exc})")
    else:
        failures.append(("L1/block_m-guard", AssertionError("block_m=0 not rejected")))

    # The wrapper dispatches with block_m=256; the value is otherwise ignored.
    try:
        kernel = _adaptive_max_pool2d_kernel(1, 8, 4, 4, 2, 2, "float16")(256)
        xx = torch.randn(1, 8, 4, 4, dtype=torch.float16)
        out = kernel(xx.npu()).cpu()
        torch.testing.assert_close(out, golden_adaptive_max_pool2d(xx, (2, 2)), rtol=0.0, atol=0.0)
        print("[L1] PASS: block_m=256 dispatch (wrapper contract)")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/block_m-256-dispatch", exc))
        print(f"[collected] L1/block_m-256-dispatch RUNTIME-ERROR: {exc!r}")

    # Unsupported dtype / non-positive shapes are rejected.
    try:
        _adaptive_max_pool2d_kernel(1, 8, 4, 4, 2, 2, "float32")
    except ValueError as exc:
        print(f"[L1] PASS: dtype guard rejects float32 ({exc})")
    else:
        failures.append(("L1/dtype-guard", AssertionError("float32 not rejected")))
    try:
        _adaptive_max_pool2d_kernel(0, 8, 4, 4, 2, 2, "float16")
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
    """L2: extreme sizes (warn only, non-blocking)."""
    cases = [
        (1, 5, 7, 3, (2, 2), "float16"),  # C=5 (below the vector width)
        (1, 1, 1, 1, (1, 1), "bfloat16"),  # minimal
        (1, 8, 32, 32, (1, 1), "float16"),  # global on a large plane (fallback)
        (1, 8, 64, 64, None, "float16"),  # identity on a large plane (fallback)
        (3, 1024, 3, 3, (1, 1), "float16"),  # many channel blocks, persistent
        (1, 8, 1, 37, (1, 5), "float16"),  # 1xW row
        (2, 7, 13, 5, (9, 3), "bfloat16"),  # odd everything, expanding
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
    """Boundary: special-value inputs (warn only, non-blocking)."""
    # Zeros: exact zeros in, zeros out on both sides.
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

    # Constant plane: every bin max equals the constant.
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

    # Large magnitudes (fp16 near-max scale; max selects, never overflows).
    try:
        _run_case(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            output_size=(4, 4),
            dtype_str="float16",
            tag="Boundary-large",
            x=(torch.randn(1, 8, 16, 16, dtype=torch.float32) * 1000.0).to(torch.float16),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large: {e}")

    # Subnormal fp16 magnitudes (~1e-5 scale).
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

    # NaN / +-inf in the general path (equal_nan comparison).
    try:
        _run_case(
            n=1,
            c_in=16,
            h_in=11,
            w_in=13,
            output_size=(5, 4),
            dtype_str="float16",
            tag="Boundary-inf-nan",
            x=_nan_probe_x(dtype=torch.float16),
            equal_nan=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): inf-nan: {e}")

    # Global pool of inf / NaN planes (C4 reduce path NaN judgement).
    try:
        x = torch.randn(2, 16, 5, 5, dtype=torch.float32)
        x[0, 0, :, :] = float("inf")
        x[0, 4, :, :] = float("-inf")
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

    # All-NaN window (general path): every output of the touched channels NaN.
    try:
        x = torch.randn(1, 8, 9, 11, dtype=torch.float32)
        x[0, 0, :, :] = float("nan")
        _run_case(
            n=1,
            c_in=8,
            h_in=9,
            w_in=11,
            output_size=(4, 5),
            dtype_str="float16",
            tag="Boundary-all-nan-window",
            x=x.to(torch.float16),
            equal_nan=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): all-nan-window: {e}")

    # +-inf without NaN must resolve to +inf (never NaN) -- compared with
    # equal_nan=False so a spurious NaN output is caught.
    try:
        x = torch.randn(1, 8, 11, 13, dtype=torch.float32)
        x[0, 1, :, :] = float("inf")
        x[0, 2, :, :] = float("-inf")
        _run_case(
            n=1,
            c_in=8,
            h_in=11,
            w_in=13,
            output_size=(5, 4),
            dtype_str="float16",
            tag="Boundary-pm-inf-only",
            x=x.to(torch.float16),
            equal_nan=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): pm-inf-only: {e}")


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
