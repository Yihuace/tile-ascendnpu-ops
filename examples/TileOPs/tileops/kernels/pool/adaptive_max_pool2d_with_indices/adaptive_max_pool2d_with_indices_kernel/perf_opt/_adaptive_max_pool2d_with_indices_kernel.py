# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Adaptive max pooling 2D forward kernel with int64 indices (NPU, Developer).

Migrated from the GPU repo's ``_adaptive_max_pool2d_with_indices_kernel``
(TileOPs ``AdaptiveMaxPool2dIndicesFwdOp``, ``return_indices=True``) to
TileLang ``target="npuir"``.

Semantics (DESIGN.md section 0.1, frozen; matches
``torch.nn.functional.adaptive_max_pool2d(..., return_indices=True)``):
    out[n, c, oh, ow]      = max(x[n, c, ih0:ih1, iw0:iw1])
    indices[n, c, oh, ow]  = ih * W_in + iw   (row-major flat plane index, int64)
    (ih0, ih1) = (floor(oh * H_in / out_H), ceil((oh+1) * H_in / out_H))
    (iw0, iw1) = (floor(ow * W_in / out_W), ceil((ow+1) * W_in / out_W))
Ties keep the first hit (strict ``>`` record); a NaN anywhere in the window makes
the value NaN and records the *last* NaN visited (``max`` of the NaN flat
indices, because the visit order is strictly increasing); ``{+inf, -inf}``
without NaN keeps the IEEE maximum (``+inf``).  float16 / bfloat16 only; the
comparison domain is fp32 with a single round-back, so the value output is
bit-exact with a window element (atol = rtol = 0).

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel`` + ``(n, o_h, c_block)`` logical units, persistent
        in-core ``T.serial`` grid-stride loop and an ``if uid < NUM_LOGICAL``
        guard (static bounds).
    R2: C-axis vector accumulators in UB (fp32 value accumulator, multiplicative
        eq-mask NaN flag); the comparison domain is fp32.
    R3: int32 vector index accumulators: ``max_idx`` (first-hit = max flat index
        over the strict-``>`` record set) and ``nan_idx`` (last-NaN = max flat
        index over the NaN set); ``cfinal = (nan_idx >= 0)`` decides the NaN
        branch for *both* outputs.
    R4: per-row GM -> UB ``T.copy`` + in-core two-axis ``T.transpose`` so the
        channel axis becomes the contiguous vector axis; outputs are transposed
        back before the row-slice store.
    R5: degenerate-grid specialisation, used only when the whole-plane buffers
        fit the UB budget: C4b (``out == in``) identity transit copy + the
        constant flat-index pattern.  Its channel block is the capacity-driven
        ``CH_deg = min(C, 32)`` (DESIGN.md section 4.5); an oversized plane
        falls back to the general path, which is always correct.
        The C4 whole-plane ``reduce_max`` / ``reduce_min`` argmax chain
        (``out_H == out_W == 1``) is **disabled on this toolchain**: the
        in-kernel ``arange``-derived flat-index pattern is read back corrupted
        when the whole-plane reduce chain lives in the same kernel (see
        ``repro/TRAP-npuir-arange-pattern-with-whole-plane-reduce.py``), which
        silently breaks the index output while the values stay exact.  ``1x1``
        output grids therefore take the general path -- DESIGN.md section 4.5's
        sanctioned "fall back to the general path" branch, whose semantics and
        precision are identical (the specialisation is a pure optimisation).
        The general path already covers ``out_H == out_W == 1`` exactly
        (``num_logical = N * out_H * num_cblk``, ``out_H == 1``).

Frozen implementation-order constraint (DESIGN.md sections 1.6.1 N3 / 3.3 / 6.2):
    ``T.vcmp(row_f32[iw], acc[ow], gt_b, "gt")`` must run *before* the in-place
    ``T.vmax(acc[ow], row_f32[iw], acc[ow])`` so that ``gt`` compares against the
    accumulator value from *before* this element (i.e. the running maximum of the
    previously visited elements).  Swapping the two makes ``gt`` identically
    false and degenerates ``max_idx`` to the seed (the first visited element).

Stage 3 platform findings (probes, see repro/ and RETROSPECTIVE.md):
    * A scalar ``T.vbrc`` whose destination is a *buffer region* with a runtime
      offset is silently dropped on this toolchain (the destination keeps its
      stale contents).  Every per-``o_w`` seeded value is therefore produced by a
      full-buffer ``T.vbrc`` into a ``(1, CH)`` staging buffer followed by a
      ``T.copy`` of that staging buffer into the region (verified equivalent).
    * A scalar operand of ``T.vbrc`` must be a typed PrimExpr (``T.int32(-1)``,
      ``T.float32(1.0)``); a raw Python number fails the frontend rank assert.
    * NaN probing uses the IEEE-correct equality mode (``x == x``) and inverts
      it; ``cmp_mod="ne"`` is an *ordered* not-equal and is false for NaN.
    * ``bool`` (i1) results stay on-chip; bf16 is up-cast to fp32 before any
      comparison (``T.vmax`` / ``T.reduce_max`` / ``T.sub`` have no bf16 form).
    * ``@T.prim_func`` Python ``range`` is not unrolled at trace time, so every
      window / segment loop is ``T.serial(max_extent)`` + ``if k < extent``.

Factory contract (source interface unchanged):
    _adaptive_max_pool2d_with_indices_kernel(n, c_in, h_in, w_in, out_h, out_w,
                                             dtype="float16")
      -> build(block_m=None) -> main
    where ``main(x)`` returns ``(out, indices)`` (``out_idx=[1, 2]``).

Run the embedded hierarchical tests:
    python _adaptive_max_pool2d_with_indices_kernel.py --level L0
    python _adaptive_max_pool2d_with_indices_kernel.py --level all
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
    "_adaptive_max_pool2d_with_indices_kernel",
    "adaptive_bin",
    "golden_adaptive_max_pool2d_with_indices",
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
# serial loop is active (measured 1.7x-2.5x on this toolchain).
_UB_MULTIBUFFER_FACTOR = 2

# Supported dtypes (the source Op layer rejects float32).
_SUPPORTED_DTYPES = ("float16", "bfloat16")

# Precision tolerances (DESIGN.md section 8.2): max selects an input element,
# so the value output is bit-exact and the index comparison is exact.
_TOLERANCE = {
    "float16": (torch.float16, 0.0, 0.0),
    "bfloat16": (torch.bfloat16, 0.0, 0.0),
}

# fp16/bf16 vector width in elements (alignment of the vector axis).
_VECTOR_WIDTH = 8

# ---- tuned per-path channel-block caps (module-level, wrapper-visible) -----
# Identity path (C4b): whole-plane UB transit, capacity-driven (DESIGN 4.5).
CH_IDENTITY = 32
# Flat-plane global path (out_H == out_W == 1): capacity-driven block cap; the
# scan is an O(plane) serial loop, so the block is the largest that fits UB.
CH_GLOBAL = 104
# General grid path: cap for the single-wave dispatch search (the effective
# block is derived by `_pick_ch_general`).
CH_GENERAL = 64

# int32 index carrier bound (source ``idx_dtype`` rule: h_in * w_in < 2**31).
_INT32_INDEX_LIMIT = 2**31


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_adaptive_max_pool2d_with_indices(x: torch.Tensor, output_size):
    """PyTorch reference for adaptive_max_pool2d(return_indices=True).

    Directly uses the source-repo test benchmark (``ref_program`` in
    examples/TileOPs/tests/ops/test_pool.py), per DESIGN.md section 0.1.
    Accepts 3D CHW or 4D NCHW; ``None`` entries keep the input size.  Returns
    ``(values, indices)`` where ``indices`` is int64 and holds the flat
    ``ih * W_in + iw`` index (ties keep the first hit; a NaN in the window makes
    the value NaN and records the last NaN visited).  Independent of the NPU
    algorithm (no transpose / no C-vectorisation / no index accumulator / no
    degenerate-grid specialisation).  Runs on CPU and does not require an NPU
    device.
    """
    size = (None, None) if output_size is None else output_size
    val, idx = F.adaptive_max_pool2d(x.detach().to(torch.float32), size, return_indices=True)
    return val.to(x.dtype), idx


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@kernel_factory_cache
def _adaptive_max_pool2d_with_indices_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str = "float16",
):
    """Build the adaptive max pooling 2D kernel factory (values + indices).

    Returns a build callable taking a single ``block_m`` (the CUDA ``threads``
    parameter is removed, K9); ``build(block_m)`` compiles and returns ``main``;
    ``main(x)`` returns ``(out, indices)`` (``out_idx=[1, 2]``).
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
    if h_in * w_in >= _INT32_INDEX_LIMIT:
        raise ValueError(
            f"int32 index carrier requires h_in * w_in < 2**31, got h_in * w_in = {h_in * w_in}"
        )

    elem_bytes = 2  # float16 / bfloat16 only

    def _round_up(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    def _general_bytes(ch: int) -> int:
        """Per-core UB footprint of the general path (bytes, no inflation)."""
        return (
            2 * ch * w_in * elem_bytes  # row_src, row_dst
            + w_in * ch * 4  # row_f32
            + w_in * ch  # eq_mask (bool)
            + 4 * out_w * ch * 4  # acc, infbuf, nanbuf, out32
            + 4 * out_w * ch * 4  # max_idx, nan_idx, zeros_i, idx32
            + out_w * ch * 4  # idx_t
            + out_w * ch  # cfinal (bool)
            + 2 * out_w * ch * elem_bytes  # acc16, acc_t
            + 4 * ch * 4  # cand, nanc, fvec, neg1
            + ch * 4  # seed staging
            + 2 * ch  # gt_b, isn_b (bool)
        )

    def _identity_bytes(ch: int) -> int:
        """Per-core UB footprint of the C4b identity path (DESIGN.md 4.5)."""
        plane = ch * h_in * w_in
        return (
            plane * (elem_bytes + 4 + 8)  # stage(dtype), ibl(int32), idx64(int64)
            + plane * 4  # iw (the (ch, h*w) int32 index pattern)
            + h_in * w_in * 4 * 2  # i1 (1-D arange) + i2 (reshaped view)
        )

    def _choose_ch() -> int:
        """General-path channel vector block (DESIGN.md 5.2: min(C, 64)).

        Must be a multiple of the fp16 vector width.  Halved only if the
        general-path footprint would not fit the UB budget (robustness guard).
        """
        ch = max(_VECTOR_WIDTH, _round_up(min(c_in, 64), _VECTOR_WIDTH))
        while (
            ch > _VECTOR_WIDTH and _general_bytes(ch) * _UB_MULTIBUFFER_FACTOR > _UB_CAPACITY_BYTES
        ):
            ch = max(_VECTOR_WIDTH, _round_up(ch // 2, _VECTOR_WIDTH))
        return ch

    def _pick_ch_general(cap: int) -> int:
        """Effective general-grid channel block: smallest single-wave block.

        The logical-unit count ``N * out_H * ceil(C/CH)`` sets both the wave
        count and the per-core work.  Picking the **smallest** multiple of the
        vector width that still fits one wave spreads the units over the most
        cores while minimising the widest block (borrowed from the sibling
        adaptive-max-pool Stage 4 ``_pick_ch_general``; the dual-output form has
        a >3x higher per-element vector cost, so more cores help more here).
        """
        base = max(_VECTOR_WIDTH, _round_up(min(c_in, cap), _VECTOR_WIDTH))
        vector_cores = NPUUtils.get().get_aicore_num() * 2
        for ch in range(_VECTOR_WIDTH, base + 1, _VECTOR_WIDTH):
            units = n * out_h * ((c_in + ch - 1) // ch)
            if units <= vector_cores:
                return ch
        return base

    ch = _choose_ch()  # noqa: F841
    # General-grid block: single-wave dispatch (smallest block that keeps the
    # logical-unit count within the physical core count).
    ch_gen = _pick_ch_general(CH_GENERAL)
    # Degenerate paths use the capacity-driven channel block (DESIGN.md 4.5).
    ch_deg = max(_VECTOR_WIDTH, _round_up(min(c_in, CH_IDENTITY), _VECTOR_WIDTH))

    identity_shape = out_h == h_in and out_w == w_in
    is_identity = (
        identity_shape and _identity_bytes(ch_deg) * _UB_MULTIBUFFER_FACTOR <= _UB_CAPACITY_BYTES
    )

    # ---- flat-plane global specialisation (out_H == out_W == 1) -------------
    # The whole plane is a single bin, so declare x as (N, C, 1, H*W): the
    # GM->UB read is one contiguous transfer (the general path's per-row
    # strides + per-row transpose propagate a layout that inflates GM_to_UB
    # ~8x and UB_to_GM ~21x on this toolchain).  One (C, plane) -> (plane, C)
    # transpose then makes every scan row contiguous.
    _plane = h_in * w_in

    def _global_bytes(ch: int) -> int:
        """Per-core UB footprint of the flat-plane global path (bytes).

        Only dtype-width plane buffers (``flat``, ``flatT``) are declared, so
        the footprint is ~4 B/plane-element plus the O(ch) accumulators.
        """
        return (
            2 * ch * _plane * elem_bytes  # flat, flatT (dtype)
            + ch * (3 * elem_bytes + 2 * 4 + 8 * 4 + 4 + 8 + 3 + 2 * 4)
            # (1,ch) dtype x3 / fp32 x2 / int32 x8 / bool x3 ; (ch,1) int32,
            # int64, dtype
        )

    def _choose_ch_global() -> int:
        ch_g = max(_VECTOR_WIDTH, _round_up(min(c_in, CH_GLOBAL), _VECTOR_WIDTH))
        while (
            ch_g > _VECTOR_WIDTH
            and _global_bytes(ch_g) * _UB_MULTIBUFFER_FACTOR > _UB_CAPACITY_BYTES
        ):
            ch_g = max(_VECTOR_WIDTH, _round_up(ch_g // 2, _VECTOR_WIDTH))
        return ch_g

    ch_g = _choose_ch_global()
    # fp16 only: the flat-plane scan accumulates in the *input* dtype (the fp16
    # plane buffers are what keep the UB footprint -- and therefore the poisoned
    # plane-transpose round trip -- small), and this toolchain has no bf16
    # `T.vmax` form.  bf16 1x1 grids therefore take the general path, which
    # up-casts to fp32: a pure optimisation fallback, semantics unchanged.
    is_global = (
        dtype == "float16"
        and out_h == 1
        and out_w == 1
        and not identity_shape
        and _global_bytes(ch_g) * _UB_MULTIBUFFER_FACTOR <= _UB_CAPACITY_BYTES
    )

    # Logical unit granularity is the single definition source of NUM_LOGICAL
    # (DESIGN.md 5.5 item 1): general (n, o_h, c_block); global / identity the
    # whole (n, c_block) plane.
    if is_identity:
        num_cblk = (c_in + ch_deg - 1) // ch_deg
        num_logical = n * num_cblk
    elif is_global:
        num_cblk_g = (c_in + ch_g - 1) // ch_g
        num_logical = n * num_cblk_g
    else:
        num_cblk_gen = (c_in + ch_gen - 1) // ch_gen
        num_logical = n * out_h * num_cblk_gen

    # Physical cores: pure-Vector op -> AI cores x 2 (DESIGN.md 5.5 item 2).
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_kernels = min(num_logical, vector_cores)
    num_local = (num_logical + num_kernels - 1) // num_kernels

    max_bin_h = max_adaptive_bin_extent(h_in, out_h)
    max_bin_w = max_adaptive_bin_extent(w_in, out_w)

    def _build(block_m=None):
        if block_m is not None and block_m < 1:
            raise ValueError(f"block_m must be positive, got block_m={block_m}")

        if is_identity:
            # C4b (DESIGN.md 0.6 R5 / 1.6.1 C4b): out_H == H_in and
            # out_W == W_in -> every bin is unit size, so out == x (pure UB
            # transit, NaN preserved) and the index is the constant flat-index
            # pattern ih * W_in + iw (task invariant, built once).
            @tilelang.jit(out_idx=[1, 2], target="npuir")
            def _func_identity(ch_):
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, h_in, w_in), dtype),
                    out: T.Tensor((n, c_in, out_h, out_w), dtype),
                    indices: T.Tensor((n, c_in, out_h, out_w), "int64"),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        stage = T.alloc_ub((ch_, h_in, w_in), dtype)
                        i1 = T.alloc_ub((h_in * w_in,), "int32")
                        i2 = T.alloc_ub((1, h_in * w_in), "int32")
                        iw = T.alloc_ub((ch_, h_in * w_in), "int32")
                        ibl = T.alloc_ub((ch_, h_in, w_in), "int32")
                        idx64 = T.alloc_ub((ch_, h_in, w_in), "int64")
                        # 1-D arange -> reshape -> vbrc -> reshape: the
                        # flat-index constant pattern (metadata-only reshape).
                        T.arange(i1, [1], 0)
                        T.reshape(i1, i2)
                        T.vbrc(i2, iw)
                        T.reshape(iw, ibl)
                        T.vcast(ibl, idx64, round_mode="rint")
                        for t in T.serial(num_local):
                            uid = t * num_kernels + cid
                            if uid < num_logical:
                                nb = uid // num_cblk
                                cb = uid % num_cblk
                                real_c = T.min(ch_, c_in - cb * ch_)
                                T.copy(
                                    x[nb, cb * ch_ : cb * ch_ + real_c, 0:h_in, 0:w_in],
                                    stage[0:real_c, 0:h_in, 0:w_in],
                                )
                                T.copy(
                                    stage[0:real_c, 0:h_in, 0:w_in],
                                    out[nb, cb * ch_ : cb * ch_ + real_c, 0:h_in, 0:w_in],
                                )
                                T.copy(
                                    idx64[0:real_c, 0:h_in, 0:w_in],
                                    indices[nb, cb * ch_ : cb * ch_ + real_c, 0:h_in, 0:w_in],
                                )

                return main

            return _func_identity(ch_deg)

        if is_global:
            # C4c: out_H == out_W == 1 -> the whole plane is one bin.  Flat
            # plane read in one contiguous DMA and transposed once to
            # (plane, C); the record scan then runs on contiguous (1, C) rows.
            # ONLY dtype-width buffers participate in the scan: the fp32
            # plane-sized comparison buffers of the first shape are dropped, so
            # the UB->GM->UB round trip that the plane transpose costs on this
            # toolchain moves ~6 KB/unit instead of ~31 KB/unit.
            # The accumulator is seeded from the first plane element (no -inf
            # needed); the seed record index 0 matches the general path.
            @tilelang.jit(out_idx=[1, 2], target="npuir")
            def _func_global(ch_):
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, 1, _plane), dtype),
                    out: T.Tensor((n, c_in, out_h, out_w), dtype),
                    indices: T.Tensor((n, c_in, out_h, out_w), "int64"),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        flat = T.alloc_ub((ch_, _plane), dtype)
                        flatT = T.alloc_ub((_plane, ch_), dtype)
                        acc16 = T.alloc_ub((1, ch_), dtype)
                        nan16 = T.alloc_ub((1, ch_), dtype)
                        gout = T.alloc_ub((1, ch_), dtype)
                        gout_t = T.alloc_ub((ch_, 1), dtype)
                        max_idx = T.alloc_ub((1, ch_), "int32")
                        nan_idx = T.alloc_ub((1, ch_), "int32")
                        fvec = T.alloc_ub((1, ch_), "int32")
                        neg1 = T.alloc_ub((1, ch_), "int32")
                        cand = T.alloc_ub((1, ch_), "int32")
                        nanc = T.alloc_ub((1, ch_), "int32")
                        zeros_i = T.alloc_ub((1, ch_), "int32")
                        idx32 = T.alloc_ub((1, ch_), "int32")
                        idx_t = T.alloc_ub((ch_, 1), "int32")
                        idx64 = T.alloc_ub((ch_, 1), "int64")
                        gt_b = T.alloc_ub((1, ch_), "bool")
                        isn_b = T.alloc_ub((1, ch_), "bool")
                        cfinal = T.alloc_ub((1, ch_), "bool")
                        infbuf = T.alloc_ub((1, ch_), "float32")
                        nanbuf = T.alloc_ub((1, ch_), "float32")

                        T.vbrc(T.int32(-1), neg1)
                        T.vbrc(T.int32(0), zeros_i)
                        T.vbrc(T.infinity("float32"), infbuf)
                        T.vsub(infbuf, infbuf, nanbuf)
                        T.vcast(nanbuf, nan16, round_mode="rint")
                        for t in T.serial(num_local):
                            uid = t * num_kernels + cid
                            if uid < num_logical:
                                nb = uid // num_cblk_g
                                cb = uid % num_cblk_g
                                real_c = T.min(ch_, c_in - cb * ch_)
                                T.copy(
                                    x[nb, cb * ch_ : cb * ch_ + real_c, 0, 0:_plane],
                                    flat[0:real_c, 0:_plane],
                                )
                                T.transpose(flat, flatT, permutation=[1, 0])
                                # seed: acc = flatT[0] (max identity + first
                                # visited element is both the seed value and the
                                # seed record index 0)
                                T.copy(flatT[0:1, 0:ch_], acc16)
                                T.vbrc(T.int32(0), max_idx)
                                T.vbrc(T.int32(-1), nan_idx)
                                for k in T.serial(_plane):
                                    T.vbrc(T.int32(k), fvec)
                                    # first-hit: gt compares against acc from
                                    # BEFORE this element's vmax(acc).
                                    T.vcmp(flatT[k : k + 1, 0:ch_], acc16, gt_b, "gt")
                                    T.vselect(gt_b, fvec, neg1, cand)
                                    T.vmax(max_idx, cand, max_idx)
                                    T.vmax(acc16, flatT[k : k + 1, 0:ch_], acc16)
                                    # isn_b = (x == x) is 1 for numbers and 0
                                    # for NaN, so the NaN branch takes the flat
                                    # index and the number branch takes -1.
                                    T.vcmp(
                                        flatT[k : k + 1, 0:ch_],
                                        flatT[k : k + 1, 0:ch_],
                                        isn_b,
                                        "eq",
                                    )
                                    T.vselect(isn_b, neg1, fvec, nanc)
                                    T.vmax(nan_idx, nanc, nan_idx)
                                T.vcmp(nan_idx, zeros_i, cfinal, "ge")
                                T.vselect(cfinal, nan_idx, max_idx, idx32)
                                T.transpose(idx32, idx_t, permutation=[1, 0])
                                T.vcast(idx_t, idx64, round_mode="rint")
                                T.copy(
                                    idx64[0:real_c, 0:1],
                                    indices[nb, cb * ch_ : cb * ch_ + real_c, 0, 0:1],
                                )
                                T.vselect(cfinal, nan16, acc16, gout)
                                T.transpose(gout, gout_t, permutation=[1, 0])
                                T.copy(
                                    gout_t[0:real_c, 0:1],
                                    out[nb, cb * ch_ : cb * ch_ + real_c, 0, 0:1],
                                )

                return main

            return _func_global(ch_g)

        # ---------------- general grid ----------------
        @tilelang.jit(out_idx=[1, 2], target="npuir")
        def _func_general(ch_):
            @T.prim_func
            def main(
                x: T.Tensor((n, c_in, h_in, w_in), dtype),
                out: T.Tensor((n, c_in, out_h, out_w), dtype),
                indices: T.Tensor((n, c_in, out_h, out_w), "int64"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # C-axis vector accumulators (vector axis = C, DESIGN.md 1.6.3).
                    acc = T.alloc_ub((out_w, ch_), "float32")
                    infbuf = T.alloc_ub((out_w, ch_), "float32")
                    nanbuf = T.alloc_ub((out_w, ch_), "float32")
                    out32 = T.alloc_ub((out_w, ch_), "float32")
                    cfinal = T.alloc_ub((out_w, ch_), "bool")
                    zeros_i = T.alloc_ub((out_w, ch_), "int32")
                    max_idx = T.alloc_ub((out_w, ch_), "int32")
                    nan_idx = T.alloc_ub((out_w, ch_), "int32")
                    idx32 = T.alloc_ub((out_w, ch_), "int32")
                    idx_t = T.alloc_ub((ch_, out_w), "int32")
                    idx64 = T.alloc_ub((ch_, out_w), "int64")
                    acc16 = T.alloc_ub((out_w, ch_), dtype)
                    acc_t = T.alloc_ub((ch_, out_w), dtype)
                    # Row staging: NCHW row slice -> transposed (C innermost).
                    row_src = T.alloc_ub((ch_, w_in), dtype)
                    row_dst = T.alloc_ub((w_in, ch_), dtype)
                    row_f32 = T.alloc_ub((w_in, ch_), "float32")
                    eq_mask = T.alloc_ub((w_in, ch_), "bool")
                    eq_f32 = T.alloc_ub((w_in, ch_), "float32")
                    # Per-element operands / masks (all on-chip).
                    cand = T.alloc_ub((1, ch_), "int32")
                    nanc = T.alloc_ub((1, ch_), "int32")
                    fvec = T.alloc_ub((1, ch_), "int32")
                    neg1 = T.alloc_ub((1, ch_), "int32")
                    seed_stg = T.alloc_ub((1, ch_), "int32")
                    gt_b = T.alloc_ub((1, ch_), "bool")
                    isn_b = T.alloc_ub((1, ch_), "bool")
                    zero_e = T.alloc_ub((1, ch_), "float32")

                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // (out_h * num_cblk_gen)
                            rem = uid % (out_h * num_cblk_gen)
                            oh = rem // num_cblk_gen
                            cb = rem % num_cblk_gen
                            real_c = T.min(ch_, c_in - cb * ch_)
                            ih0, ih1 = adaptive_bin(oh, h_in, out_h)
                            bh = ih1 - ih0

                            # max identity is -inf (not 0); the last-NaN
                            # accumulator starts at -1 (no NaN seen) and is
                            # also the single NaN decision source for both
                            # outputs: cfinal = (nan_idx >= 0).
                            T.vbrc(-T.infinity("float32"), acc)
                            T.vbrc(T.float32(0.0), zero_e)
                            T.vbrc(T.int32(-1), neg1)
                            T.vbrc(T.int32(-1), nan_idx)
                            T.vbrc(T.int32(0), zeros_i)
                            # first-hit seed = flat index of the first visited
                            # element of each window: ih0 * w_in + iw0(ow).  The
                            # seed is applied through a staging buffer + T.copy:
                            # a scalar vbrc straight into a region is dropped by
                            # this toolchain (see the module docstring).
                            for ow in T.serial(out_w):
                                iw0, iw1 = adaptive_bin(ow, w_in, out_w)
                                T.vbrc(T.int32(ih0 * w_in + iw0), seed_stg)
                                T.copy(seed_stg, max_idx[ow : ow + 1, 0:ch_])
                            for kh in T.serial(max_bin_h):
                                if kh < bh:
                                    T.copy(
                                        x[
                                            nb,
                                            cb * ch_ : cb * ch_ + real_c,
                                            ih0 + kh,
                                            0:w_in,
                                        ],
                                        row_src[0:real_c, 0:w_in],
                                    )
                                    T.transpose(row_src, row_dst, permutation=[1, 0])
                                    T.vcast(row_dst, row_f32, round_mode="rint")
                                    # Row-level NaN probe: x == x is 1 for
                                    # numbers and 0 for NaN ("ne" is ordered
                                    # and returns false for NaN => unusable).
                                    T.vcmp(row_f32, row_f32, eq_mask, "eq")
                                    T.vcast(eq_mask, eq_f32, round_mode="rint")
                                    for ow in T.serial(out_w):
                                        iw0, iw1 = adaptive_bin(ow, w_in, out_w)
                                        bw = iw1 - iw0
                                        for kw in T.serial(max_bin_w):
                                            if kw < bw:
                                                T.vbrc(
                                                    T.int32((ih0 + kh) * w_in + iw0 + kw),
                                                    fvec,
                                                )
                                                # first-hit: the gt mask must be
                                                # compared against the acc value
                                                # from BEFORE this element, so
                                                # this vcmp must precede the
                                                # in-place vmax(acc) below.
                                                T.vcmp(
                                                    row_f32[iw0 + kw : iw0 + kw + 1, 0:ch_],
                                                    acc[ow : ow + 1, 0:ch_],
                                                    gt_b,
                                                    "gt",
                                                )
                                                T.vselect(gt_b, fvec, neg1, cand)
                                                T.vmax(
                                                    max_idx[ow : ow + 1, 0:ch_],
                                                    cand,
                                                    max_idx[ow : ow + 1, 0:ch_],
                                                )
                                                # value (in-place vmax after gt)
                                                T.vmax(
                                                    acc[ow : ow + 1, 0:ch_],
                                                    row_f32[iw0 + kw : iw0 + kw + 1, 0:ch_],
                                                    acc[ow : ow + 1, 0:ch_],
                                                )
                                                # last-NaN (max flat index)
                                                T.vcmp(
                                                    eq_f32[iw0 + kw : iw0 + kw + 1, 0:ch_],
                                                    zero_e,
                                                    isn_b,
                                                    "eq",
                                                )
                                                T.vselect(isn_b, fvec, neg1, nanc)
                                                T.vmax(
                                                    nan_idx[ow : ow + 1, 0:ch_],
                                                    nanc,
                                                    nan_idx[ow : ow + 1, 0:ch_],
                                                )
                            # Both outputs share one NaN decision:
                            # cfinal = (nan_idx >= 0) <=> the window holds a NaN.
                            T.vcmp(nan_idx, zeros_i, cfinal, "ge")
                            T.vselect(cfinal, nan_idx, max_idx, idx32)
                            T.transpose(idx32, idx_t, permutation=[1, 0])
                            T.vcast(idx_t, idx64, round_mode="rint")
                            T.copy(
                                idx64[0:real_c, 0:out_w],
                                indices[nb, cb * ch_ : cb * ch_ + real_c, oh, 0:out_w],
                            )
                            T.vbrc(T.infinity("float32"), infbuf)
                            T.vsub(infbuf, infbuf, nanbuf)
                            T.vselect(cfinal, nanbuf, acc, out32)
                            T.vcast(out32, acc16, round_mode="rint")
                            T.transpose(acc16, acc_t, permutation=[1, 0])
                            T.copy(
                                acc_t[0:real_c, 0:out_w],
                                out[nb, cb * ch_ : cb * ch_ + real_c, oh, 0:out_w],
                            )

            return main

        return _func_general(ch_gen)

    return _build


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_FACTORY_CACHE = {}

# Worst-case statistics over the whole run (final summary).
_STATS = {"value_max_diff": {}, "idx_mismatch": 0, "idx_total": 0}


def _get_factory(n, c_in, h_in, w_in, out_h, out_w, dtype_str):
    """Cache factory instances (one per shape/dtype, as the wrapper does)."""
    key = (n, c_in, h_in, w_in, out_h, out_w, dtype_str)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _adaptive_max_pool2d_with_indices_kernel(
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


def _value_stats(out, ref, equal_nan):
    """Return (max_abs_diff, violation_count) between two value tensors."""
    eq = out == ref
    if equal_nan:
        eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(eq, torch.zeros((), dtype=torch.float32), (out.float() - ref.float()).abs())
    return diff.max().item(), int((~eq).sum().item())


def _idx_stats(idx, ref_idx):
    """Return (mismatch_count, total) for the exact int64 index comparison."""
    mismatch = int((idx != ref_idx).sum().item())
    return mismatch, int(idx.numel())


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
    """Run one kernel case and compare both outputs against the golden.

    ``x`` may be pre-built as a CPU tensor (special-value cases); by default
    randn data is generated on CPU, cast and moved to NPU.  ``ndim3`` exercises
    the Op layer's 3D CHW path: the kernel always sees the unsqueezed 4D tensor
    and the comparison uses the 3D golden.  Raises AssertionError on a value or
    index violation.
    """
    torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
    out_h, out_w = _resolve_out_dims(h_in, w_in, output_size)

    if x is None:
        x = torch.randn(n, c_in, h_in, w_in, dtype=torch.float32).to(torch_dtype)
    else:
        assert x.dtype == torch_dtype, f"input dtype {x.dtype} != {torch_dtype}"
        assert tuple(x.shape) == (n, c_in, h_in, w_in), f"input shape {x.shape}"

    kernel = _get_factory(n, c_in, h_in, w_in, out_h, out_w, dtype_str)(None)
    out, idx = kernel(x.npu())
    out = out.cpu()
    idx = idx.cpu()

    ref, ref_idx = golden_adaptive_max_pool2d_with_indices(x, output_size)
    if ndim3:
        assert tuple(out.shape) == (c_in, out_h, out_w), f"3D rank not preserved: {out.shape}"
        assert tuple(idx.shape) == (c_in, out_h, out_w), f"3D index rank not preserved: {idx.shape}"
        ref, ref_idx = golden_adaptive_max_pool2d_with_indices(x[0], output_size)

    assert idx.dtype == torch.int64, f"indices dtype {idx.dtype} != int64"
    assert tuple(idx.shape) == tuple(out.shape), (
        f"indices shape {tuple(idx.shape)} != out shape {tuple(out.shape)}"
    )

    max_diff, violations = _value_stats(out, ref, equal_nan)
    if max_diff == max_diff:
        best = _STATS["value_max_diff"].get(dtype_str, -1.0)
        if max_diff > best:
            _STATS["value_max_diff"][dtype_str] = max_diff
    idx_mm, idx_total = _idx_stats(idx, ref_idx)
    _STATS["idx_mismatch"] += idx_mm
    _STATS["idx_total"] += idx_total

    try:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
        assert idx_mm == 0, f"index mismatch in {idx_mm}/{idx_total} elements"
    except AssertionError:
        print(
            f"[{tag}] FAIL: shape=({n},{c_in},{h_in},{w_in}) out=({out_h},{out_w}) "
            f"dtype={dtype_str} max_diff={max_diff:.3e} value_violations={violations}"
            f"/{out.numel()} idx_mismatch={idx_mm}/{idx_total}"
        )
        raise
    print(
        f"[{tag}] PASS: shape=({n},{c_in},{h_in},{w_in}) out=({out_h},{out_w}) "
        f"dtype={dtype_str} max_diff={max_diff:.3e} idx_exact"
    )
    return max_diff, idx_mm


def _try_case(failures, case_id, **kwargs):
    """Run ``_run_case``, collecting (not re-raising) per-case failures."""
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


# ---------------------------------------------------------------------------
# Semantics probes (NaN / +-inf / ties / first-hit order)
# ---------------------------------------------------------------------------
# Window (oh=0, ow=0) of an (11, 13) input -> out (5, 4) covers rows 0..2 and
# columns 0..3, i.e. 12 elements; flat index = ih * 13 + iw (W_in = 13).
_SEM_SHAPE = (1, 16, 11, 13)
_SEM_SIZE = (5, 4)
_SEM_WIN = (0, 3, 0, 4)  # ih0, ih1, iw0, iw1


def _flat(ih, iw, w_in=13):
    return ih * w_in + iw


def _sem_probe_x(dtype):
    """Special-value input for the fp16/bf16 semantic gates (same collection).

    Channels (window (0,0) = rows 0..2 x cols 0..3):
      0: one NaN at (0, 0)                       -> value NaN, idx flat(0,0)=0
      1: two NaN at (0,1) and (2,3)              -> value NaN, idx flat(2,3)=29
      2: {+inf at (0,0), -inf at (0,1)}          -> value +inf, idx 0
      3: all -inf inside the window              -> value -inf, idx 0
      4: all equal inside the window (5.0)       -> value 5.0, idx 0 (first hit)
      5: single max placed at (2,3) (not first)  -> value 9.0, idx flat(2,3)=29
      6: the max appears twice ((0,0) and (2,3)) -> value 9.0, idx 0 (first hit)
      7: NaN at (1,2) and +inf at (0,0)          -> value NaN, idx flat(1,2)=15
    """
    n, c_in, h_in, w_in = _SEM_SHAPE
    x = torch.randn(n, c_in, h_in, w_in, dtype=torch.float32)
    ih0, ih1, iw0, iw1 = _SEM_WIN
    # Rows/cols outside the probe window keep small random values (still inside
    # other windows; those are compared against the golden, not hand-asserted).
    x[0, 0, 0, 0] = float("nan")
    x[0, 1, 0, 1] = float("nan")
    x[0, 1, 2, 3] = float("nan")
    x[0, 2, 0, 0] = float("inf")
    x[0, 2, 0, 1] = float("-inf")
    x[0, 3, ih0:ih1, iw0:iw1] = float("-inf")
    x[0, 4, ih0:ih1, iw0:iw1] = 5.0
    x[0, 5, ih0:ih1, iw0:iw1] = -3.0
    x[0, 5, 2, 3] = 9.0
    x[0, 6, ih0:ih1, iw0:iw1] = -3.0
    x[0, 6, 0, 0] = 9.0
    x[0, 6, 2, 3] = 9.0
    x[0, 7, ih0:ih1, iw0:iw1] = -3.0
    x[0, 7, 0, 0] = float("inf")
    x[0, 7, 1, 2] = float("nan")
    return x.to(dtype)


def _sem_expected():
    """Hand-derived (value_is_nan, index) per probe channel for window (0, 0)."""
    return {
        0: (True, _flat(0, 0)),
        1: (True, _flat(2, 3)),
        2: (False, _flat(0, 0)),  # +inf wins (no NaN in the window)
        3: (False, _flat(0, 0)),  # all -inf -> the first visited element
        4: (False, _flat(0, 0)),  # ties -> first hit
        5: (False, _flat(2, 3)),  # max is not the first element
        6: (False, _flat(0, 0)),  # tie -> first hit
        7: (True, _flat(1, 2)),  # NaN beats +inf; last/only NaN index
    }


def _run_semantic_gate(failures, dtype_str, case_id):
    """Semantics gate (DESIGN.md 8.3 L0-9 / L0-9b) with mechanism evidence.

    Asserts the hand-derived per-channel expectations (value NaN-ness and the
    exact flat index) *and* a full-tensor comparison against the golden.  The
    hand-derived assertions are what make the NaN-index / first-hit mechanisms
    observable (a value-only check can pass through ``vmax`` NaN propagation).
    """
    n, c_in, h_in, w_in = _SEM_SHAPE
    out_h, out_w = _SEM_SIZE
    try:
        torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
        x = _sem_probe_x(torch_dtype)
        kernel = _get_factory(n, c_in, h_in, w_in, out_h, out_w, dtype_str)(None)
        out, idx = kernel(x.npu())
        out = out.cpu()
        idx = idx.cpu()
        ref, ref_idx = golden_adaptive_max_pool2d_with_indices(x, _SEM_SIZE)

        for ch, (want_nan, want_idx) in _sem_expected().items():
            got = out[0, ch, 0, 0]
            got_nan = bool(torch.isnan(got))
            assert got_nan == want_nan, (
                f"ch={ch} window(0,0) NaN-ness {got_nan} != {want_nan} (got {got})"
            )
            got_idx = int(idx[0, ch, 0, 0].item())
            assert got_idx == want_idx, (
                f"ch={ch} window(0,0) index {got_idx} != {want_idx} "
                f"(first-hit / last-NaN mechanism)"
            )
            if not want_nan:
                assert float(got) == float(ref[0, ch, 0, 0].item()), (
                    f"ch={ch} window(0,0) value {float(got)} != golden "
                    f"{float(ref[0, ch, 0, 0].item())}"
                )

        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, equal_nan=True)
        idx_mm, idx_total = _idx_stats(idx, ref_idx)
        _STATS["idx_mismatch"] += idx_mm
        _STATS["idx_total"] += idx_total
        assert idx_mm == 0, f"index mismatch in {idx_mm}/{idx_total} elements"
        print(
            f"[{case_id}] PASS: NaN wins in-window; last-NaN index; first-hit "
            f"tie; {{+inf,-inf}} -> +inf; all--inf -> first element; "
            f"full-tensor match vs golden (equal_nan=True, idx exact) "
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
    """Implementation-level first-hit check (DESIGN.md 8.3 mechanism evidence).

    ``arange(12).reshape(1,1,3,4) -> (2,2)``: bin(1,1) covers rows 1..2 x
    columns 2..3 = flat {6,7,10,11}, so the maximum 11 lives at flat 11 -- the
    window's *last* visited element.  The v0 ordering (vmax(acc) before
    vcmp(gt)) degenerates the record mask to false and yields the seed, flat 6.
    bin(1,0) covers flat {4,5,8,9} -> 9 at flat 9.
    """
    case_id = "L0-12/first-hit-order"
    try:
        x = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4).to(torch.float16)
        kernel = _get_factory(1, 1, 3, 4, 2, 2, "float16")(None)
        out, idx = kernel(x.npu())
        out = out.cpu()
        idx = idx.cpu()
        ref, ref_idx = golden_adaptive_max_pool2d_with_indices(x, (2, 2))
        assert float(out[0, 0, 0, 0]) == 5.0 and int(idx[0, 0, 0, 0]) == 5, (
            f"bin(0,0): value {float(out[0, 0, 0, 0])} idx {int(idx[0, 0, 0, 0])} != 5/5"
        )
        assert float(out[0, 0, 0, 1]) == 7.0 and int(idx[0, 0, 0, 1]) == 7, (
            f"bin(0,1): value {float(out[0, 0, 0, 1])} idx {int(idx[0, 0, 0, 1])} != 7/7"
        )
        assert float(out[0, 0, 1, 0]) == 9.0 and int(idx[0, 0, 1, 0]) == 9, (
            f"bin(1,0): value {float(out[0, 0, 1, 0])} idx {int(idx[0, 0, 1, 0])} != 9/9 "
            f"(record-order bug would give the seed 4)"
        )
        assert float(out[0, 0, 1, 1]) == 11.0 and int(idx[0, 0, 1, 1]) == 11, (
            f"bin(1,1): value {float(out[0, 0, 1, 1])} idx {int(idx[0, 0, 1, 1])} != 11/11 "
            f"(the v0 record-order defect would give the seed 6)"
        )
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
        assert torch.equal(idx, ref_idx), "index != golden index"
        print("[L0-12] PASS: first-hit record order (bin(1,1) index 11, not seed 6)")
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


def _run_factory_guards(failures):
    """block_m contract, dtype gate and index-width gate (L0-10 / L1)."""
    try:
        _adaptive_max_pool2d_with_indices_kernel(1, 8, 4, 4, 2, 2, "float16")(0)
    except ValueError as exc:
        print(f"[L0-10] PASS: block_m guard rejects 0 ({exc})")
    else:
        failures.append(("L0-10/block_m-guard", AssertionError("block_m=0 not rejected")))

    try:
        kernel = _adaptive_max_pool2d_with_indices_kernel(1, 8, 4, 4, 2, 2, "float16")(256)
        xx = torch.randn(1, 8, 4, 4, dtype=torch.float16)
        o, i = kernel(xx.npu())
        o = o.cpu()
        i = i.cpu()
        r, ri = golden_adaptive_max_pool2d_with_indices(xx, (2, 2))
        torch.testing.assert_close(o, r, rtol=0.0, atol=0.0)
        assert torch.equal(i, ri), "index mismatch on the block_m=256 dispatch"
        print("[L0-10] PASS: block_m=256 dispatch (wrapper contract)")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L0-10/block_m-256-dispatch", exc))
        print(f"[collected] L0-10/block_m-256-dispatch RUNTIME-ERROR: {exc!r}")

    for bad_dtype in ("float32", "float64"):
        try:
            _adaptive_max_pool2d_with_indices_kernel(1, 8, 4, 4, 2, 2, bad_dtype)
        except ValueError as exc:
            print(f"[L0-10] PASS: dtype guard rejects {bad_dtype} ({exc})")
        else:
            failures.append(
                (f"L0-10/dtype-guard-{bad_dtype}", AssertionError(f"{bad_dtype} not rejected"))
            )
    try:
        _adaptive_max_pool2d_with_indices_kernel(0, 8, 4, 4, 2, 2, "float16")
    except ValueError as exc:
        print(f"[L0-10] PASS: shape guard rejects n=0 ({exc})")
    else:
        failures.append(("L0-10/shape-guard", AssertionError("n=0 not rejected")))
    try:
        _adaptive_max_pool2d_with_indices_kernel(1, 8, 2**16, 2**15, 1, 1, "float16")
    except ValueError as exc:
        print(f"[L0-10] PASS: index-width guard rejects h_in*w_in >= 2**31 ({exc})")
    else:
        failures.append(
            ("L0-10/index-width-guard", AssertionError("h_in*w_in >= 2**31 not rejected"))
        )


def _run_3d_case(failures):
    """L0-8: 3D CHW input -> both outputs keep rank 3 (Op unsqueeze path)."""
    case_id = "L0-8/3d-chw"
    try:
        torch_dtype = torch.float16
        c_in, h_in, w_in = 16, 11, 13
        output_size = (5, 4)
        out_h, out_w = output_size
        x3 = torch.randn(c_in, h_in, w_in, dtype=torch.float32).to(torch_dtype)
        kernel = _get_factory(1, c_in, h_in, w_in, out_h, out_w, "float16")(None)
        out, idx = kernel(x3.unsqueeze(0).npu())
        out = out.cpu().squeeze(0)
        idx = idx.cpu().squeeze(0)
        ref, ref_idx = golden_adaptive_max_pool2d_with_indices(x3, output_size)
        assert tuple(out.shape) == (c_in, out_h, out_w), f"3D rank: {out.shape}"
        assert tuple(idx.shape) == (c_in, out_h, out_w), f"3D idx rank: {idx.shape}"
        assert idx.dtype == torch.int64
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
        assert torch.equal(idx, ref_idx), "index mismatch"
        print(
            f"[L0-8] PASS: shape=({c_in},{h_in},{w_in}) out=({out_h},{out_w}) "
            f"3D rank preserved (values + int64 indices)"
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3) -- blocking."""
    failures = []

    cases = [
        (2, 64, 56, 56, (6, 6), "float16", "L0-1/spp-6x6"),
        (1, 96, 28, 30, (7, 5), "bfloat16", "L0-2/asym-bf16-cblk-tail"),
        (1, 128, 55, 57, (7, 7), "float16", "L0-3/nondiv-fp16"),
        (1, 16, 11, 13, (5, None), "float16", "L0-4/partial-none"),
        (8, 2048, 7, 7, (1, 1), "float16", "L0-5/global-1x1-persistent"),
        (1, 8, 9, 11, None, "float16", "L0-6/identity"),
        (1, 8, 4, 4, (7, 7), "float16", "L0-7/expanding"),
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
    _run_semantic_gate(failures, "float16", "L0-9/nan-inf-tie-fp16")
    _run_semantic_gate(failures, "bfloat16", "L0-9b/nan-inf-tie-bf16")
    _run_factory_guards(failures)
    _run_first_hit_order(failures)

    # L0-11: indices contract on a representative shape.
    case_id = "L0-11/indices-contract"
    try:
        x = torch.randn(2, 32, 17, 19, dtype=torch.float32).to(torch.float16)
        kernel = _get_factory(2, 32, 17, 19, 6, 5, "float16")(None)
        out, idx = kernel(x.npu())
        out = out.cpu()
        idx = idx.cpu()
        assert idx.dtype == torch.int64, f"indices dtype {idx.dtype}"
        assert tuple(idx.shape) == (2, 32, 6, 5), f"indices shape {idx.shape}"
        assert tuple(idx.shape) == tuple(out.shape), "indices shape != value shape"
        assert int(idx.min()) >= 0 and int(idx.max()) < 17 * 19, (
            f"flat index out of plane range [{int(idx.min())}, {int(idx.max())}]"
        )
        print("[L0-11] PASS: indices int64, same shape as values, flat-plane range")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (7 shapes + 3D + 2 semantic gates + guards + order)")
    return failures


def run_L1():
    """L1: full functional coverage (shape x dtype x specialisation path)."""
    failures = []

    cases = [
        # General path, bf16, channel-block tail (C=96 -> ch=64 + real_c=32).
        (1, 96, 28, 30, (7, 5), "bfloat16", "L1/bf16/cblk-tail"),
        # bf16 on the manifest spp shape.
        (2, 128, 56, 56, (6, 6), "bfloat16", "L1/bf16/spp"),
        # fp16 non-divisible with the odd transposed row length.
        (1, 128, 55, 57, (7, 7), "float16", "L1/fp16/nondiv"),
        # 1x1 output grid (whole plane) on the general path, several c-blocks.
        (4, 64, 9, 9, (1, 1), "bfloat16", "L1/bf16/global-1x1"),
        (2, 32, 5, 5, (1, 1), "float16", "L1/fp16/global-1x1-small-plane"),
        # Identity (C4b) with a channel tail.
        (1, 96, 5, 7, None, "float16", "L1/fp16/identity-cblk-tail"),
        (2, 32, 6, 6, None, "bfloat16", "L1/bf16/identity"),
        # W-only adaptive (H identity, W not) -> must take the general path.
        (1, 16, 11, 13, (11, 5), "float16", "L1/fp16/h-identity-general"),
        (1, 8, 4, 4, (3, 3), "float16", "L1/fp16/expanding-small"),
        # Large batch on a small plane (persistent, many logical units).
        (16, 32, 6, 6, (3, 3), "float16", "L1/fp16/large-n"),
        # 1x1 input (identity and global coincide; identity wins).
        (1, 8, 1, 1, (1, 1), "float16", "L1/fp16/1x1"),
        # Single output column (max_bin_w large, out_w == 1).
        (2, 32, 13, 17, (4, 1), "float16", "L1/fp16/out-w-1"),
        # Single output row.
        (1, 16, 9, 11, (1, 5), "float16", "L1/fp16/out-h-1"),
        # Scalar output size, bf16.
        (1, 16, 10, 10, 5, "bfloat16", "L1/bf16/scalar-size"),
        # Shrinking bins (out < in, non-divisible).
        (3, 24, 13, 13, (4, 5), "float16", "L1/fp16/shrink-nondiv"),
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

    # NaN semantics on the whole-plane (1x1 output) path: inf plane -> inf,
    # all-NaN plane -> NaN with the last-NaN index, -inf only -> -inf.
    case_id = "L1/fp16/global-1x1-nan-inf"
    try:
        x = torch.randn(2, 16, 5, 5, dtype=torch.float32)
        x[0, 0, :, :] = float("inf")
        x[0, 1, :, :] = float("nan")
        x[1, 3, :, :] = float("-inf")
        x[1, 5, 2, 2] = float("nan")
        x = x.to(torch.float16)
        kernel = _get_factory(2, 16, 5, 5, 1, 1, "float16")(None)
        out, idx = kernel(x.npu())
        out = out.cpu()
        idx = idx.cpu()
        ref, ref_idx = golden_adaptive_max_pool2d_with_indices(x, (1, 1))
        assert float(out[0, 0, 0, 0]) == float("inf"), (
            f"+inf plane gave {float(out[0, 0, 0, 0])} (expected +inf)"
        )
        assert int(idx[0, 0, 0, 0]) == 0, "+inf plane first-hit index != 0"
        assert bool(torch.isnan(out[0, 1, 0, 0])), "all-NaN plane did not give NaN"
        assert int(idx[0, 1, 0, 0]) == 24, (
            f"all-NaN plane last-NaN index {int(idx[0, 1, 0, 0])} != 24 (last element)"
        )
        assert float(out[1, 3, 0, 0]) == float("-inf"), "all -inf plane != -inf"
        assert int(idx[1, 3, 0, 0]) == 0, "all -inf plane first-hit index != 0"
        assert bool(torch.isnan(out[1, 5, 0, 0])), "single NaN in plane did not win"
        assert int(idx[1, 5, 0, 0]) == 12, "single NaN index != 12"
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0, equal_nan=True)
        assert torch.equal(idx, ref_idx), "1x1-output index != golden index"
        print(
            "[L1] PASS: 1x1-output NaN/Inf semantics (inf plane -> inf, all-NaN "
            "-> NaN + last-NaN index, all -inf -> -inf + first index)"
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # Identity (C4b) path preserves NaN and emits the constant index pattern.
    case_id = "L1/fp16/identity-nan-index-pattern"
    try:
        x = torch.randn(1, 8, 9, 11, dtype=torch.float32)
        x[0, 0, 3, 4] = float("nan")
        x[0, 1, 0, 0] = float("-inf")
        x = x.to(torch.float16)
        kernel = _get_factory(1, 8, 9, 11, 9, 11, "float16")(None)
        out, idx = kernel(x.npu())
        out = out.cpu()
        idx = idx.cpu()
        ref, ref_idx = golden_adaptive_max_pool2d_with_indices(x, None)
        assert bool(torch.isnan(out[0, 0, 3, 4])), "identity path lost the NaN"
        pattern = torch.arange(9 * 11, dtype=torch.int64).reshape(9, 11)
        assert torch.equal(idx[0], pattern.expand(8, 9, 11)), (
            "identity index pattern != row-major flat index"
        )
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0, equal_nan=True)
        assert torch.equal(idx, ref_idx), "identity index != golden index"
        print("[L1] PASS: C4b identity NaN passthrough + constant flat index")
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # Degenerate-grid capacity fallback: a big plane must take the general path
    # (the identity specialisation is gated on the UB budget) and stay exact.
    case_id = "L1/fp16/degenerate-capacity-fallback"
    try:
        x = torch.randn(1, 8, 32, 32, dtype=torch.float32).to(torch.float16)
        kernel = _get_factory(1, 8, 32, 32, 1, 1, "float16")(None)
        out, idx = kernel(x.npu())
        out = out.cpu()
        idx = idx.cpu()
        ref, ref_idx = golden_adaptive_max_pool2d_with_indices(x, (1, 1))
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
        assert torch.equal(idx, ref_idx), "fallback index != golden index"
        print("[L1] PASS: oversized identity plane takes the general path")
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes (warn only, non-blocking)."""
    cases = [
        (1, 5, 7, 3, (2, 2), "float16"),  # C below the vector width
        (1, 1, 1, 1, (1, 1), "bfloat16"),  # minimal
        (1, 8, 64, 64, None, "float16"),  # identity on a large plane (fallback)
        (3, 1024, 3, 3, (1, 1), "float16"),  # many channel blocks, persistent
        (1, 8, 1, 37, (1, 5), "float16"),  # 1 x W input row
        (2, 7, 13, 5, (9, 3), "bfloat16"),  # odd everything, expanding
        (1, 16, 31, 33, (1, 7), "float16"),  # out_h == 1 only
        (1, 16, 33, 31, (7, 1), "float16"),  # out_w == 1 only
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
                f"out={output_size} {dtype_str}: {str(e)[:200]}"
            )


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    probes = []

    probes.append(
        (
            "zeros",
            dict(
                n=2,
                c_in=16,
                h_in=9,
                w_in=11,
                output_size=(3, 4),
                dtype_str="float16",
                x=torch.zeros(2, 16, 9, 11, dtype=torch.float16),
            ),
        )
    )
    probes.append(
        (
            "const",
            dict(
                n=1,
                c_in=8,
                h_in=11,
                w_in=13,
                output_size=(5, 4),
                dtype_str="float16",
                x=torch.full((1, 8, 11, 13), 3.7, dtype=torch.float16),
            ),
        )
    )
    probes.append(
        (
            "large-magnitude",
            dict(
                n=1,
                c_in=8,
                h_in=16,
                w_in=16,
                output_size=(4, 4),
                dtype_str="bfloat16",
                x=(torch.randn(1, 8, 16, 16) * 1000.0).to(torch.bfloat16),
            ),
        )
    )
    probes.append(
        (
            "subnormal",
            dict(
                n=1,
                c_in=8,
                h_in=12,
                w_in=12,
                output_size=(3, 3),
                dtype_str="float16",
                x=(torch.randn(1, 8, 12, 12) * 1e-5).to(torch.float16),
            ),
        )
    )
    probes.append(
        (
            "nan-inf-general",
            dict(
                n=1,
                c_in=16,
                h_in=11,
                w_in=13,
                output_size=(5, 4),
                dtype_str="float16",
                x=_sem_probe_x(torch.float16),
                equal_nan=True,
            ),
        )
    )
    probes.append(
        (
            "global-inf-nan",
            dict(
                n=2,
                c_in=16,
                h_in=5,
                w_in=5,
                output_size=(1, 1),
                dtype_str="float16",
                x=_global_probe_x(torch.float16),
                equal_nan=True,
            ),
        )
    )
    probes.append(
        (
            "all-nan-window",
            dict(
                n=1,
                c_in=8,
                h_in=9,
                w_in=11,
                output_size=(4, 5),
                dtype_str="float16",
                x=_all_nan_x(torch.float16),
                equal_nan=True,
            ),
        )
    )
    probes.append(
        (
            "pm-inf-only",
            dict(
                n=1,
                c_in=8,
                h_in=11,
                w_in=13,
                output_size=(5, 4),
                dtype_str="float16",
                x=_pm_inf_x(torch.float16),
                equal_nan=False,
            ),
        )
    )
    probes.append(
        (
            "identity-nan",
            dict(
                n=1,
                c_in=8,
                h_in=7,
                w_in=9,
                output_size=None,
                dtype_str="bfloat16",
                x=_all_nan_x(torch.bfloat16, shape=(1, 8, 7, 9)),
                equal_nan=True,
            ),
        )
    )

    for name, kwargs in probes:
        try:
            _run_case(tag=f"Boundary-{name}", **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): {name}: {str(e)[:200]}")


def _global_probe_x(dtype):
    x = torch.randn(2, 16, 5, 5, dtype=torch.float32)
    x[0, 0, :, :] = float("inf")
    x[0, 4, :, :] = float("-inf")
    x[1, 3, 2, 2] = float("nan")
    x[1, 6, :, :] = float("-inf")
    return x.to(dtype)


def _all_nan_x(dtype, shape=(1, 8, 9, 11)):
    x = torch.randn(*shape, dtype=torch.float32)
    x[0, 0, :, :] = float("nan")
    x[0, 3, 4, 5] = float("nan")
    return x.to(dtype)


def _pm_inf_x(dtype):
    x = torch.randn(1, 8, 11, 13, dtype=torch.float32)
    x[0, 1, :, :] = float("inf")
    x[0, 2, :, :] = float("-inf")
    return x.to(dtype)


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
