# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Windowed max pooling 2D forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``_max_pool2d_kernel`` (TileOPs ``MaxPool2dFwdOp``,
``ref_api = torch.nn.functional.max_pool2d(return_indices=False)``) to TileLang
``target="npuir"``.

Semantics (DESIGN.md section 0.1, frozen):
    out[n, c, oh, ow] = max over the kH x kW window anchored at
                        (oh*sH - pH + kh*dH, ow*sW - pW + kw*dW)
    Out-of-range window positions behave as -inf padding (the source kernel
    skips those lanes; max(-inf, x) == x, machine-verified as E1 in
    ``verify_equiv.py``).  A NaN anywhere in the window wins (out = NaN);
    {+inf, -inf} in one window resolves to +inf (IEEE max, never NaN).
    Exactly 4D NCHW, float16 / bfloat16 / float32, same dtype out.
    The output extent comes from the source helper ``pool_output_dim``
    (including both ceil_mode corrections) and is never re-derived.

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel`` + ``(n, o_h, c_block)`` logical units with a
        persistent in-core ``T.serial`` grid-stride loop and an
        ``if uid < NUM_LOGICAL`` tail guard (static bounds).
    R2: UB fp32 vector accumulator on the channel axis.  The window loops are
        ``T.serial(kH)`` / ``T.serial(kW)`` / ``T.serial(out_W)`` with static
        upper bounds; the H in-bounds guard is hoisted to once per row (C1)
        and the W guard stays per lane.  Maximum = ``T.vmax`` on a C-vector
        slice; NaN tracking = explicit IEEE ``x == x`` eq mask multiplied into
        a flag accumulator seeded to 1 (0 absorbing, 1 neutral), final
        ``has_nan ? NaN : max`` via ``T.vcmp(flag, 0, "eq")`` +
        ``T.vselect``.  The kernel never relies on the undocumented
        ``T.vmax`` / ``T.reduce_max`` NaN behaviour.
    R3: per-row GM -> UB ``T.copy`` + in-core two-axis ``T.transpose`` ([1, 0])
        so the channel axis becomes the contiguous vector axis; the output is
        transposed back before the row-slice store.
    R4: degenerate-grid specialisation: C4 (whole-plane ``T.reduce_max`` chain
        plus an independent eq-mask NaN reduction) and C4b (identity UB transit
        copy), both guarded by the frozen sufficient conditions AND the UB
        capacity criterion (a pure optimisation -- an oversized plane simply
        falls back to the general path, which is always correct).

C4 sufficient condition (DESIGN.md sections 0.6 R4 / 1.6.1, REVIEW.md problem 1):
``out_H == out_W == 1 and p_H == p_W == 0 and d_H == d_W == 1 and
(k_H-1)*d_H + 1 >= H_in and (k_W-1)*d_W + 1 >= W_in``.  ``d_H == d_W == 1`` is
mandatory: the window's H coverage is ``{k_h*d_H} & [0, H_in)``, which equals
the whole plane only for ``d == 1`` (a ``d > 1`` window such as ``H=W=5,
k=2, s=1, p=0, d=4`` covers only rows/cols {0, 4}, so a whole-plane reduction
would return the wrong value).  ``verify_equiv.py`` C4x locks this with six
machine-checked case-groups.

Implementation notes (Stage 3, deviations from the DESIGN.md section 3.3
pseudocode and their justification):
    * ``@tilelang.jit`` on this branch does not accept ``compile_flags`` (its
      signature is ``out_idx / target / target_host / execution_backend /
      verbose / pass_configs / debug_root_path``), so the source's
      ``["-O3", "-DENABLE_BF16"]`` is dropped -- the verified same-family
      kernel ``_adaptive_max_pool2d_kernel`` does the same.
    * The NaN probe is evaluated **once per input row** (a ``(W_in, CH)`` eq
      mask consumed by column slice inside the window loops) instead of once
      per window element.  This is the form actually verified on this
      toolchain by ``_adaptive_max_pool2d_kernel`` (PL-1.2 family), it is
      element-wise identical, and it removes ``2 * kH * kW * out_W`` vector ops
      per unit from the design's highest-risk term (emission count).
    * ``T.vcast`` with target dtype == source dtype **miscompiles** on this
      toolchain (probe: a f32 -> f32 ``vcast(round_mode="rint")`` in the C4
      path returned wrong values).  Whenever the input dtype is float32 the
      same-shape same-dtype transfers therefore use ``T.copy`` instead.
    * The C4 NaN judgement reduces the eq mask with ``T.reduce_min`` (0 iff any
      element is NaN) and compares it to 0.  ``T.reduce_max`` of the same mask
      is 1 iff *at least one* element is non-NaN, so the design's
      ``reduce_max`` + ``eq 0`` combination would report NaN only for an
      all-NaN plane (and would otherwise depend on the undocumented
      ``reduce_max`` NaN propagation).  ``reduce_min`` preserves the frozen
      intent -- an independent eq-mask reduction, never ``reduce_max``'s NaN
      behaviour -- and is the form verified by ``_adaptive_max_pool2d_kernel``.
    * Channel-block tails (``C % CH != 0``, e.g. C=96 with CH=64) use dynamic
      ``real_c = min(CH, C - cb*CH)`` copy extents; padded lanes never reach GM.
    * ``block_m`` is accepted by the returned callable for source interface
      compatibility (K9 kept the wrapper's ``block_m`` config knob); NPU tiling
      is shape-derived, so the value is validated and otherwise ignored.

Factory contract (source interface unchanged):
    _max_pool2d_kernel(n, c_in, h_in, w_in, kernel_h, kernel_w, stride_h,
                       stride_w, pad_h, pad_w, dilation_h, dilation_w,
                       ceil_mode, dtype="float16")
      -> build(block_m=None) -> main
    where ``main(x)`` returns ``out`` (``out_idx=[1]``).

Run the embedded hierarchical tests:
    python _max_pool2d_kernel.py --level L0
    python _max_pool2d_kernel.py --level all
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
    "_max_pool2d_kernel",
    "golden_max_pool2d",
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
_CH_MAX = 64


def _elem_bytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2


# ---------------------------------------------------------------------------
# Source helper (verbatim from the GPU repo's kernels/pool/common.py, inlined
# into the Stage 0 extraction package and re-exported here).  The output
# extent must never be re-derived anywhere else.
# ---------------------------------------------------------------------------
def pool_output_dim(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
    dilation: int = 1,
) -> int:
    """Output extent of one pooling axis (PyTorch max_pool2d rule)."""
    effective_kernel = dilation * (kernel_size - 1) + 1
    if ceil_mode:
        out = (input_size + 2 * padding - effective_kernel + stride - 1) // stride + 1
    else:
        out = (input_size + 2 * padding - effective_kernel) // stride + 1

    if ceil_mode and out > 0 and (out - 1) * stride >= input_size + padding:
        out -= 1

    return max(out, 0)


# ---------------------------------------------------------------------------
# Tiling plan (host/trace side): CH, degenerate-path selection, core split
# ---------------------------------------------------------------------------
def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _general_bytes(ch: int, w_in: int, out_w: int, elem_bytes: int) -> int:
    """Per-core UB footprint of the general path (bytes, no inflation)."""
    return (
        2 * ch * w_in * elem_bytes  # row_src, row_dst
        + 2 * w_in * ch * 4  # row_f32, eq_f32
        + w_in * ch  # eq_mask (bool)
        + 6 * out_w * ch * 4  # acc, flag, zeros, infbuf, nanbuf, out32
        + out_w * ch  # cfinal (bool)
        + 2 * out_w * ch * elem_bytes  # acc16, acc_t
    )


def _global_bytes(ch: int, h_in: int, w_in: int, elem_bytes: int) -> int:
    """Per-core UB footprint of the C4 whole-plane path (bytes, no inflation).

    Main term ``CH*H_in*W_in*(elem_bytes + 9)`` = flat (elem_bytes) + flat32 (4)
    + eq_mask (1) + eq_f32 (4); small term covers mid/midn (CH*H_in*1 f32 x2),
    gmax/nmin/zeros/infbuf/nanbuf/out32 (CH*1*1 f32 x6), gout (CH*elem_bytes)
    and cond (CH bool).
    """
    return (
        ch * h_in * w_in * (elem_bytes + 9) + 2 * ch * h_in * 4 + 7 * ch * 4 + ch * elem_bytes + ch
    )


def _identity_bytes(ch: int, h_in: int, w_in: int, elem_bytes: int) -> int:
    """Per-core UB footprint of the C4b identity path (bytes, no inflation)."""
    return ch * h_in * w_in * elem_bytes


def _choose_ch(c_in: int, w_in: int, out_w: int, elem_bytes: int) -> int:
    """Channel vector block: C axis is the vector axis (DESIGN.md 1.6.3).

    Largest multiple of 8 that is <= min(C, 64) and whose general-path footprint
    fits the UB budget; halved (never below 8) while it does not.
    """
    ch = max(_VECTOR_WIDTH, _round_up(min(c_in, _CH_MAX), _VECTOR_WIDTH))
    while (
        ch > _VECTOR_WIDTH
        and _general_bytes(ch, w_in, out_w, elem_bytes) * _UB_MULTIBUFFER_FACTOR
        > _UB_CAPACITY_BYTES
    ):
        ch = max(_VECTOR_WIDTH, _round_up(ch // 2, _VECTOR_WIDTH))
    return ch


def _degenerate_conditions(out_h, out_w, h_in, w_in, k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w):
    """Frozen sufficient conditions of the C4 / C4b specialisations."""
    identity = (
        k_h == 1
        and k_w == 1
        and d_h == 1
        and d_w == 1
        and p_h == 0
        and p_w == 0
        and s_h == 1
        and s_w == 1
    )
    global_plane = (
        out_h == 1
        and out_w == 1
        and p_h == 0
        and p_w == 0
        and d_h == 1
        and d_w == 1
        and (k_h - 1) * d_h + 1 >= h_in
        and (k_w - 1) * d_w + 1 >= w_in
    )
    return identity, global_plane


def _plan(
    n, c_in, h_in, w_in, k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w, ceil_mode, dtype, vector_cores=None
):
    """Derive the compile-time tiling plan for one shape/dtype combination.

    Returns a dict with the output extents, the channel block, the degenerate
    path tag, and the persistent core-split constants.  ``vector_cores``
    defaults to the device query (pure-Vector op -> AI cores x 2).
    """
    out_h = pool_output_dim(h_in, k_h, s_h, p_h, ceil_mode, d_h)
    out_w = pool_output_dim(w_in, k_w, s_w, p_w, ceil_mode, d_w)
    if out_h < 1 or out_w < 1:
        raise ValueError(f"output size must be greater than zero, got {(out_h, out_w)}")
    elem_bytes = _elem_bytes(dtype)
    ch = _choose_ch(c_in, w_in, out_w, elem_bytes)
    num_cblk = (c_in + ch - 1) // ch

    id_cond, gl_cond = _degenerate_conditions(
        out_h, out_w, h_in, w_in, k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w
    )
    is_identity = (
        id_cond
        and _identity_bytes(ch, h_in, w_in, elem_bytes) * _UB_MULTIBUFFER_FACTOR
        <= _UB_CAPACITY_BYTES
    )
    is_global = (
        gl_cond
        and not is_identity
        and _global_bytes(ch, h_in, w_in, elem_bytes) * _UB_MULTIBUFFER_FACTOR <= _UB_CAPACITY_BYTES
    )
    path = "identity" if is_identity else ("global" if is_global else "general")

    # Logical unit granularity is the single definition source of NUM_LOGICAL
    # (DESIGN.md 5.5 item 1): general unit = (n, o_h, c_block); C4 / C4b unit =
    # the whole (n, c_block) plane.
    num_logical = n * num_cblk if path in ("identity", "global") else n * out_h * num_cblk
    if vector_cores is None:
        vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_kernels = min(num_logical, vector_cores)
    num_local = (num_logical + num_kernels - 1) // num_kernels

    return {
        "out_h": out_h,
        "out_w": out_w,
        "ch": ch,
        "num_cblk": num_cblk,
        "path": path,
        "elem_bytes": elem_bytes,
        "num_logical": num_logical,
        "num_kernels": num_kernels,
        "num_local": num_local,
        "general_bytes": _general_bytes(ch, w_in, out_w, elem_bytes),
        "global_bytes": _global_bytes(ch, h_in, w_in, elem_bytes),
        "identity_bytes": _identity_bytes(ch, h_in, w_in, elem_bytes),
    }


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_max_pool2d(
    x, kernel_size, stride=None, padding=(0, 0), dilation=(1, 1), ceil_mode=False
):
    """PyTorch reference for max_pool2d, per DESIGN.md section 0.1.

    Directly uses the source-repo test benchmark (``MaxPool2dTest.ref_program``
    in examples/TileOPs/tests/ops/test_pool.py).  Exactly 4D NCHW; independent
    of the NPU algorithm (no transpose / no C-vectorisation / no C4-C4b
    specialisation).  Runs on CPU and does not require an NPU device.
    """
    return F.max_pool2d(
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
@functools.lru_cache(maxsize=32)
def _max_pool2d_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    """Build the windowed max pooling 2D kernel factory.

    Returns a build callable taking a single ``block_m`` (the CUDA ``threads``
    parameter is removed, K9); ``build(block_m)`` compiles and returns ``main``;
    ``main(x)`` returns ``out`` (``out_idx=[1]``).
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"MaxPool2dFwdOp supports float16, bfloat16, and float32, got {dtype}")
    if min(n, c_in, h_in, w_in) < 1:
        raise ValueError(f"input shape must be positive, got {(n, c_in, h_in, w_in)}")
    if min(kernel_h, kernel_w) < 1:
        raise ValueError(f"kernel_size must be greater than zero, got {(kernel_h, kernel_w)}")
    if min(stride_h, stride_w) < 1:
        raise ValueError(f"stride must be greater than zero, got {(stride_h, stride_w)}")
    if min(dilation_h, dilation_w) < 1:
        raise ValueError(f"dilation must be greater than zero, got {(dilation_h, dilation_w)}")
    if min(pad_h, pad_w) < 0:
        raise ValueError(f"padding must be non-negative, got {(pad_h, pad_w)}")
    if pad_h > kernel_h // 2 or pad_w > kernel_w // 2:
        raise ValueError(
            "padding must be at most half of kernel_size, got "
            f"padding={(pad_h, pad_w)} kernel_size={(kernel_h, kernel_w)}"
        )

    # The output extent is validated before the tiling plan is derived: a
    # non-positive extent makes the core-split arithmetic degenerate, so the
    # Op layer's "output size must be greater than zero" guard runs first.
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    if out_h < 1 or out_w < 1:
        raise ValueError(
            f"output size must be greater than zero, got {(out_h, out_w)} "
            f"for input {(h_in, w_in)} kernel {(kernel_h, kernel_w)}"
        )

    plan = _plan(
        n,
        c_in,
        h_in,
        w_in,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
        dtype,
    )
    assert plan["out_h"] == out_h and plan["out_w"] == out_w

    ch = plan["ch"]
    num_cblk = plan["num_cblk"]
    path = plan["path"]
    num_logical = plan["num_logical"]
    num_kernels = plan["num_kernels"]
    num_local = plan["num_local"]
    is_f32 = dtype == "float32"

    def _build(block_m=None):
        if block_m is not None and block_m < 1:
            raise ValueError(f"block_m must be positive, got block_m={block_m}")

        if path == "identity":
            # C4b: the window is a single element and out == in, so out == x
            # (pure UB transit, same dtype, NaN preserved).
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

        if path == "global":
            # C4: out_H == out_W == 1 -> the window is the whole plane; one
            # whole-plane max reduction plus an independent eq-mask NaN
            # reduction (never reduce_max's NaN behaviour).
            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_global(ch_):
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, h_in, w_in), dtype),
                    out: T.Tensor((n, c_in, out_h, out_w), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        flat = T.alloc_ub((ch, h_in, w_in), dtype)
                        flat32 = T.alloc_ub((ch, h_in, w_in), "float32")
                        eq_mask = T.alloc_ub((ch, h_in, w_in), "bool")
                        eq_f32 = T.alloc_ub((ch, h_in, w_in), "float32")
                        mid = T.alloc_ub((ch, h_in, 1), "float32")
                        gmax = T.alloc_ub((ch, 1, 1), "float32")
                        midn = T.alloc_ub((ch, h_in, 1), "float32")
                        nmin = T.alloc_ub((ch, 1, 1), "float32")
                        zeros = T.alloc_ub((ch, 1, 1), "float32")
                        infbuf = T.alloc_ub((ch, 1, 1), "float32")
                        nanbuf = T.alloc_ub((ch, 1, 1), "float32")
                        cond = T.alloc_ub((ch, 1, 1), "bool")
                        out32 = T.alloc_ub((ch, 1, 1), "float32")
                        gout = T.alloc_ub((ch, 1, 1), dtype)
                        pos_inf = T.infinity("float32")
                        for t in T.serial(num_local):
                            uid = t * num_kernels + cid
                            if uid < num_logical:
                                nb = uid // num_cblk
                                cb = uid % num_cblk
                                real_c = T.min(ch, c_in - cb * ch)
                                T.copy(
                                    x[nb, cb * ch : cb * ch + real_c, 0:h_in, 0:w_in],
                                    flat[0:real_c, 0:h_in, 0:w_in],
                                )
                                if is_f32:
                                    T.copy(flat, flat32)
                                else:
                                    T.vcast(flat, flat32, round_mode="rint")
                                # Whole-plane maximum (clear=True: the
                                # reduction target must not accumulate stale
                                # data).
                                T.reduce_max(flat32, mid, dim=2, clear=True)
                                T.reduce_max(mid, gmax, dim=1, clear=True)
                                # NaN detection: x == x is 1 for numbers and 0
                                # for NaN; the MIN of that mask is 0 iff the
                                # plane holds a NaN ("ne" is an ordered
                                # not-equal and can never detect NaN).
                                T.vcmp(flat32, flat32, eq_mask, "eq")
                                T.vcast(eq_mask, eq_f32, round_mode="rint")
                                T.reduce_min(eq_f32, midn, dim=2, clear=True)
                                T.reduce_min(midn, nmin, dim=1, clear=True)
                                T.clear(zeros)
                                T.vcmp(nmin, zeros, cond, "eq")
                                # NaN = inf - inf (IEEE, built in fp32 because
                                # vsub / reduce_max have no bf16 form).
                                T.vbrc(pos_inf, infbuf)
                                T.vsub(infbuf, infbuf, nanbuf)
                                T.vselect(cond, nanbuf, gmax, out32)
                                if is_f32:
                                    T.copy(out32, gout)
                                else:
                                    T.vcast(out32, gout, round_mode="rint")
                                T.copy(
                                    gout[0:real_c, 0:1, 0:1],
                                    out[nb, cb * ch : cb * ch + real_c, 0:1, 0:1],
                                )

                return main

            return _func_global(ch)

        # ---------------- general grid ----------------
        @tilelang.jit(out_idx=[1], target="npuir")
        def _func_general(ch_):
            @T.prim_func
            def main(
                x: T.Tensor((n, c_in, h_in, w_in), dtype),
                out: T.Tensor((n, c_in, out_h, out_w), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # Vector accumulators on the channel axis (vector axis = C).
                    acc = T.alloc_ub((out_w, ch), "float32")
                    flag = T.alloc_ub((out_w, ch), "float32")
                    zeros = T.alloc_ub((out_w, ch), "float32")
                    infbuf = T.alloc_ub((out_w, ch), "float32")
                    nanbuf = T.alloc_ub((out_w, ch), "float32")
                    out32 = T.alloc_ub((out_w, ch), "float32")
                    cfinal = T.alloc_ub((out_w, ch), "bool")
                    # Row staging: NCHW row slice -> transposed (C innermost).
                    row_src = T.alloc_ub((ch, w_in), dtype)
                    row_dst = T.alloc_ub((w_in, ch), dtype)
                    row_f32 = T.alloc_ub((w_in, ch), "float32")
                    eq_mask = T.alloc_ub((w_in, ch), "bool")
                    eq_f32 = T.alloc_ub((w_in, ch), "float32")
                    acc16 = T.alloc_ub((out_w, ch), dtype)
                    acc_t = T.alloc_ub((ch, out_w), dtype)

                    # max identity is -inf (not 0); the NaN-free flag
                    # accumulator starts at 1 (0 is the absorbing value).
                    neg_inf = -T.infinity("float32")
                    pos_inf = T.infinity("float32")
                    one_val = T.float32(1.0)

                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // (out_h * num_cblk)
                            rem = uid % (out_h * num_cblk)
                            oh = rem // num_cblk
                            cb = rem % num_cblk
                            real_c = T.min(ch, c_in - cb * ch)
                            T.vbrc(neg_inf, acc)
                            T.vbrc(one_val, flag)
                            T.clear(zeros)
                            for kh in T.serial(kernel_h):
                                # H guard hoisted to once per row (DESIGN.md
                                # 1.6.1 C1): ih does not depend on kw.
                                ih = oh * stride_h - pad_h + kh * dilation_h
                                if ih >= 0 and ih < h_in:
                                    T.copy(
                                        x[nb, cb * ch : cb * ch + real_c, ih, 0:w_in],
                                        row_src[0:real_c, 0:w_in],
                                    )
                                    T.transpose(row_src, row_dst, permutation=[1, 0])
                                    if is_f32:
                                        T.copy(row_dst, row_f32)
                                    else:
                                        T.vcast(row_dst, row_f32, round_mode="rint")
                                    # Row-level NaN probe materialised as an
                                    # fp32 {0, 1} mask: x == x is 1 for numbers
                                    # and 0 for NaN (the "ne" mode is an
                                    # ordered not-equal, always false for NaN).
                                    T.vcmp(row_f32, row_f32, eq_mask, "eq")
                                    T.vcast(eq_mask, eq_f32, round_mode="rint")
                                    for kw in T.serial(kernel_w):
                                        for ow in T.serial(out_w):
                                            iw = ow * stride_w - pad_w + kw * dilation_w
                                            if iw >= 0 and iw < w_in:
                                                T.vmax(
                                                    acc[ow : ow + 1, 0:ch],
                                                    row_f32[iw : iw + 1, 0:ch],
                                                    acc[ow : ow + 1, 0:ch],
                                                )
                                                T.vmul(
                                                    flag[ow : ow + 1, 0:ch],
                                                    eq_f32[iw : iw + 1, 0:ch],
                                                    flag[ow : ow + 1, 0:ch],
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
                            T.transpose(acc16, acc_t, permutation=[1, 0])
                            T.copy(
                                acc_t[0:real_c, 0:out_w],
                                out[nb, cb * ch : cb * ch + real_c, oh, 0:out_w],
                            )

            return main

        return _func_general(ch)

    return _build


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_KERNEL_CACHE = {}

# Per-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_kernel(
    n, c_in, h_in, w_in, k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w, ceil_mode, dtype_str, block_m=None
):
    """Cache compiled kernels (one per shape/dtype, mirroring the wrapper's
    per-shape kernel cache)."""
    key = (
        n,
        c_in,
        h_in,
        w_in,
        k_h,
        k_w,
        s_h,
        s_w,
        p_h,
        p_w,
        d_h,
        d_w,
        bool(ceil_mode),
        dtype_str,
        block_m,
    )
    if key not in _KERNEL_CACHE:
        factory = _max_pool2d_kernel(
            n,
            c_in,
            h_in,
            w_in,
            k_h,
            k_w,
            s_h,
            s_w,
            p_h,
            p_w,
            d_h,
            d_w,
            bool(ceil_mode),
            dtype_str,
        )
        _KERNEL_CACHE[key] = factory(block_m)
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


def _normalise(kernel_size, stride, padding, dilation):
    k_h, k_w = kernel_size
    s_h, s_w = (k_h, k_w) if stride is None else stride
    p_h, p_w = padding
    d_h, d_w = dilation
    return k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w


def _run_case(
    n,
    c_in,
    h_in,
    w_in,
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
    randn data is generated on CPU, cast, and moved to NPU
    (device-independent data generation).  ``expect_path`` asserts which
    compile-time branch the tiling plan selects.  Raises AssertionError on
    tolerance or path violation.
    """
    torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
    k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w = _normalise(kernel_size, stride, padding, dilation)
    out_h = pool_output_dim(h_in, k_h, s_h, p_h, ceil_mode, d_h)
    out_w = pool_output_dim(w_in, k_w, s_w, p_w, ceil_mode, d_w)

    plan = _plan(
        n, c_in, h_in, w_in, k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w, bool(ceil_mode), dtype_str
    )
    if expect_path is not None:
        assert plan["path"] == expect_path, (
            f"{tag}: expected the {expect_path} path, plan selected "
            f"{plan['path']} (CH={plan['ch']}, num_logical={plan['num_logical']})"
        )

    if x is None:
        x = torch.randn(n, c_in, h_in, w_in, dtype=torch.float32).to(torch_dtype)
    else:
        assert x.dtype == torch_dtype, f"input dtype {x.dtype} != expected {torch_dtype}"
        assert tuple(x.shape) == (n, c_in, h_in, w_in), (
            f"input shape {tuple(x.shape)} != {(n, c_in, h_in, w_in)}"
        )

    kernel = _get_kernel(
        n, c_in, h_in, w_in, k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w, bool(ceil_mode), dtype_str
    )
    out = kernel(x.npu()).cpu()

    ref = golden_max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    assert tuple(out.shape) == (n, c_in, out_h, out_w), (
        f"{tag}: output shape {tuple(out.shape)} != {(n, c_in, out_h, out_w)}"
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
            f"[{tag}] FAIL: shape=({n},{c_in},{h_in},{w_in}) "
            f"out=({out_h},{out_w}) k={kernel_size} s={stride} p={padding} "
            f"d={dilation} ceil={ceil_mode} dtype={dtype_str} "
            f"path={plan['path']} CH={plan['ch']} max_diff={max_diff:.3e} "
            f"violations={violations}/{out.numel()}"
        )
        raise
    print(
        f"[{tag}] PASS: shape=({n},{c_in},{h_in},{w_in}) out=({out_h},{out_w}) "
        f"dtype={dtype_str} path={plan['path']} CH={plan['ch']} "
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
        _max_pool2d_kernel(**kwargs)
    except ValueError as exc:
        print(f"[{case_id}] PASS: rejected ({exc})")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} WRONG-ERROR: {exc!r}")
    else:
        failures.append((case_id, AssertionError("invalid arguments accepted")))
        print(f"[collected] {case_id} FAILED: invalid arguments were accepted")


def _nan_window_x(n=1, c_in=8, h_in=11, w_in=13):
    """NaN injection for the *windowed* semantics: NaNs at corners, mid and a
    full column (every window touching them must yield NaN)."""
    x = torch.randn(n, c_in, h_in, w_in, dtype=torch.float32)
    x[0, 0, 0, 0] = float("nan")  # top-left corner
    x[0, 1, 5, 5] = float("nan")  # interior
    x[0, 2, :, 0] = float("nan")  # full left column
    x[0, 3, h_in - 1, w_in - 1] = float("nan")  # bottom-right corner
    return x


# ---------------------------------------------------------------------------
# L0 (gate suite -- DESIGN.md section 8.3)
# ---------------------------------------------------------------------------
def run_L0():
    """L0 gate: manifest workloads across all three dtypes, the frozen
    degenerate specialisations (including their dilation/ceil guards and the
    capacity fallbacks) and the max-specific NaN / +-inf semantics."""
    failures = []

    cases = [
        # n, c, h, w, kernel, stride, padding, dilation, ceil, dtype, tag, path
        (1, 8, 16, 16, (3, 3), (2, 2), (1, 1), (1, 1), False, "float16", "L0-1/smoke", "general"),
        (
            32,
            64,
            112,
            112,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            False,
            "float16",
            "L0-2/resnet-stem",
            "general",
        ),
        (
            16,
            128,
            56,
            56,
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            False,
            "float16",
            "L0-3/vgg-block",
            "general",
        ),
        (
            32,
            64,
            55,
            55,
            (3, 3),
            (2, 2),
            (0, 0),
            (1, 1),
            True,
            "float16",
            "L0-4/alexnet-ceil",
            "general",
        ),
        (
            1,
            8,
            16,
            16,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            False,
            "float32",
            "L0-5a/smoke-fp32",
            "general",
        ),
        (
            32,
            64,
            112,
            112,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            False,
            "float32",
            "L0-5b/resnet-stem-fp32",
            "general",
        ),
        (
            2,
            64,
            55,
            57,
            (3, 3),
            (2, 2),
            (0, 0),
            (1, 1),
            False,
            "bfloat16",
            "L0-6/nondiv-bf16",
            "general",
        ),
        (
            1,
            4,
            23,
            27,
            (3, 5),
            (2, 3),
            (1, 2),
            (1, 1),
            True,
            "float16",
            "L0-7/nonsquare-ceil",
            "general",
        ),
        (
            1,
            4,
            14,
            14,
            (3, 3),
            None,
            (1, 1),
            (2, 1),
            False,
            "float16",
            "L0-8/dilation-stride-none",
            "general",
        ),
        # L0-14 channel-block tail: C=96 -> CH=64 + real_c=32; C=4 -> CH=8 +
        # real_c=4 (padded lanes must never reach GM).
        (
            3,
            96,
            28,
            30,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            False,
            "float16",
            "L0-14a/cblk-tail-96",
            "general",
        ),
        (
            1,
            4,
            14,
            14,
            (3, 3),
            (3, 3),
            (1, 1),
            (1, 1),
            False,
            "float16",
            "L0-14b/cblk-tail-4",
            "general",
        ),
    ]
    for (
        n,
        c_in,
        h_in,
        w_in,
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
            h_in=h_in,
            w_in=w_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            dtype_str=dtype_str,
            tag=case_id.split("/")[0],
            expect_path=path,
        )

    # L0-9: NaN semantics -- a NaN anywhere in the window wins.
    case_id = "L0-9/nan-window"
    try:
        x = _nan_window_x().to(torch.float16)
        _run_case(
            n=1,
            c_in=8,
            h_in=11,
            w_in=13,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
            ceil_mode=False,
            dtype_str="float16",
            tag="L0-9",
            x=x,
            equal_nan=True,
            expect_path="general",
        )
        out = _get_kernel(1, 8, 11, 13, 3, 3, 2, 2, 1, 1, 1, 1, False, "float16")(x.npu()).cpu()
        # window(oh=0, ow=0) = rows {0,1} x cols {0,1} contains the corner NaN.
        assert bool(torch.isnan(out[0, 0, 0, 0])), (
            f"corner NaN did not win: out[0,0,0,0]={out[0, 0, 0, 0]}"
        )
        # window(oh=3, ow=3) = rows {5,6,7} x cols {5,6,7} contains (5, 5).
        assert bool(torch.isnan(out[0, 1, 3, 3])), (
            f"interior NaN did not win: out[0,1,3,3]={out[0, 1, 3, 3]}"
        )
        # window(oh=5, ow=5) = rows {9,10} x cols {9,10}: no NaN in range.
        assert not bool(torch.isnan(out[0, 0, 5, 5])), "NaN-free window produced NaN"
        print("[L0-9] PASS: NaN wins in-window; NaN-free window stays finite")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-10: +-inf semantics -- {+inf, -inf} in one window gives +inf (never
    # NaN); an all -inf window gives -inf.  equal_nan=False so a spurious NaN
    # output is caught.
    case_id = "L0-10/+-inf-semantics"
    try:
        x = torch.full((1, 8, 11, 13), float("-inf"), dtype=torch.float32)
        x[0, 1, 0, 0] = float("inf")  # ch1: one +inf among -inf
        x[0, 2, :, :] = float("inf")  # ch2: all +inf
        x = x.to(torch.float16)
        _run_case(
            n=1,
            c_in=8,
            h_in=11,
            w_in=13,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
            ceil_mode=False,
            dtype_str="float16",
            tag="L0-10",
            x=x,
            equal_nan=False,
            expect_path="general",
        )
        out = _get_kernel(1, 8, 11, 13, 3, 3, 2, 2, 1, 1, 1, 1, False, "float16")(x.npu()).cpu()
        assert out[0, 1, 0, 0].item() == float("inf"), (
            f"{{+inf,-inf}} window gave {out[0, 1, 0, 0]} (expected +inf)"
        )
        assert out[0, 1, 5, 6].item() == float("-inf"), (
            f"all -inf window gave {out[0, 1, 5, 6]} (expected -inf)"
        )
        assert out[0, 2, 4, 4].item() == float("inf"), (
            f"all +inf window gave {out[0, 2, 4, 4]} (expected +inf)"
        )
        print("[L0-10] PASS: {+inf,-inf} -> +inf; all -inf -> -inf; no spurious NaN")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-11: non-contiguous input -- the Op layer normalises with
    # ``.contiguous()`` before the kernel sees the tensor (DESIGN.md 0.1).
    case_id = "L0-11/non-contiguous"
    try:
        base = torch.randn(2, 8, 32, 16, dtype=torch.float32)
        xn = base[:, :, ::2, :]  # (2, 8, 16, 16), non-contiguous view
        assert tuple(xn.shape) == (2, 8, 16, 16)
        assert not xn.is_contiguous(), "test input unexpectedly contiguous"
        xc = xn.contiguous()  # Op-layer normalisation (metadata copy)
        _run_case(
            n=2,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
            ceil_mode=False,
            dtype_str="float16",
            tag="L0-11",
            x=xc.to(torch.float16),
            expect_path="general",
        )
        print("[L0-11] PASS: non-contiguous view -> .contiguous() -> kernel")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-12: C4 whole-plane specialisation, and the capacity fallback of an
    # oversized plane (pure optimisation -- the general path is correct).
    _try_case(
        failures,
        "L0-12a/c4-16x16",
        n=1,
        c_in=8,
        h_in=16,
        w_in=16,
        kernel_size=(16, 16),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        ceil_mode=False,
        dtype_str="float16",
        tag="L0-12a",
        expect_path="global",
    )
    _try_case(
        failures,
        "L0-12a2/c4-4x4-fp32",
        n=1,
        c_in=8,
        h_in=4,
        w_in=4,
        kernel_size=(4, 4),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        ceil_mode=False,
        dtype_str="float32",
        tag="L0-12a2",
        expect_path="global",
    )
    _try_case(
        failures,
        "L0-12a3/c4-fallback-64x64",
        n=1,
        c_in=8,
        h_in=64,
        w_in=64,
        kernel_size=(64, 64),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        ceil_mode=False,
        dtype_str="float16",
        tag="L0-12a3",
        expect_path="general",
    )

    # L0-12b: the C4 dilation guard -- out == 1x1 but d > 1 must NOT take the
    # whole-plane reduction (the window covers only {k_h*d_H} x {k_w*d_W}).
    _try_case(
        failures,
        "L0-12b/d4-guard",
        n=1,
        c_in=8,
        h_in=5,
        w_in=5,
        kernel_size=(2, 2),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(4, 4),
        ceil_mode=False,
        dtype_str="float16",
        tag="L0-12b",
        expect_path="general",
    )
    _try_case(
        failures,
        "L0-12b2/d3-guard",
        n=1,
        c_in=8,
        h_in=7,
        w_in=7,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(3, 3),
        ceil_mode=False,
        dtype_str="float16",
        tag="L0-12b2",
        expect_path="general",
    )
    case_id = "L0-12b3/d4-bit-exact"
    try:
        x = torch.zeros(1, 8, 5, 5, dtype=torch.float16)
        x[0, :, 2, 2] = 1.0  # the only nonzero lies outside the {0,4} window
        _run_case(
            n=1,
            c_in=8,
            h_in=5,
            w_in=5,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(4, 4),
            ceil_mode=False,
            dtype_str="float16",
            tag="L0-12b3",
            x=x,
            expect_path="general",
        )
        out = _get_kernel(1, 8, 5, 5, 2, 2, 1, 1, 0, 0, 4, 4, False, "float16")(x.npu()).cpu()
        assert bool((out == 0.0).all()), (
            f"d>1 out=1x1 config must return 0.0, got {out.flatten()[:4].tolist()}"
        )
        print("[L0-12b3] PASS: d=4 out=1x1 returns 0.0 (whole-plane reduction correctly excluded)")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-12c: ceil_mode step-back -- (8, k=3, s=4, p=0, ceil) -> out = 2
    # (pool_output_dim drops the last window that starts inside the padding).
    case_id = "L0-12c/ceil-step-back"
    try:
        assert pool_output_dim(8, 3, 4, 0, True, 1) == 2, "step-back not triggered"
        _run_case(
            n=1,
            c_in=8,
            h_in=8,
            w_in=8,
            kernel_size=(3, 3),
            stride=(4, 4),
            padding=(0, 0),
            dilation=(1, 1),
            ceil_mode=True,
            dtype_str="float16",
            tag="L0-12c",
            expect_path="general",
        )
        print("[L0-12c] PASS: ceil_mode step-back out=2")
    except AssertionError as exc:
        failures.append((case_id, exc))
        print(f"[collected] {case_id} FAILED: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # L0-13: C4b identity specialisation (k == 1, d == 1, p == 0, s == 1) and
    # the capacity fallback of an oversized plane.
    _try_case(
        failures,
        "L0-13a/identity-4x4",
        n=1,
        c_in=8,
        h_in=4,
        w_in=4,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        ceil_mode=False,
        dtype_str="float16",
        tag="L0-13a",
        expect_path="identity",
    )
    _try_case(
        failures,
        "L0-13b/identity-fallback-112",
        n=1,
        c_in=8,
        h_in=112,
        w_in=112,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        ceil_mode=False,
        dtype_str="float16",
        tag="L0-13b",
        expect_path="general",
    )

    # L0-15: parameter / input validation (Op-layer guards mirrored by the
    # factory so the rejection paths are exercised here too).
    _expect_value_error(
        failures,
        "L0-15/dtype-fp64",
        dict(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_h=3,
            kernel_w=3,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=1,
            dilation_h=1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float64",
        ),
    )
    _expect_value_error(
        failures,
        "L0-15/zero-batch",
        dict(
            n=0,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_h=3,
            kernel_w=3,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=1,
            dilation_h=1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-15/zero-stride",
        dict(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=0,
            pad_h=1,
            pad_w=1,
            dilation_h=1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-15/zero-dilation",
        dict(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_h=3,
            kernel_w=3,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=1,
            dilation_h=0,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-15/negative-padding",
        dict(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_h=3,
            kernel_w=3,
            stride_h=2,
            stride_w=2,
            pad_h=-1,
            pad_w=0,
            dilation_h=1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-15/padding-too-large",
        dict(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_h=3,
            kernel_w=3,
            stride_h=2,
            stride_w=2,
            pad_h=2,
            pad_w=1,
            dilation_h=1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-15/non-positive-output-size",
        dict(
            n=1,
            c_in=1,
            h_in=2,
            w_in=2,
            kernel_h=5,
            kernel_w=5,
            stride_h=1,
            stride_w=1,
            pad_h=0,
            pad_w=0,
            dilation_h=1,
            dilation_w=1,
            ceil_mode=False,
            dtype="float16",
        ),
    )
    _expect_value_error(
        failures,
        "L0-15/zero-kernel",
        dict(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_h=3,
            kernel_w=0,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=0,
            dilation_h=1,
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
    remaining degenerate specialisations and NaN on the C4 reduction path."""
    failures = []

    cases = [
        # n, c, h, w, kernel, stride, padding, dilation, ceil, dtype, tag, path
        (
            2,
            8,
            16,
            16,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            False,
            "bfloat16",
            "L1/bf16/smoke",
            "general",
        ),
        (
            16,
            128,
            56,
            56,
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            False,
            "bfloat16",
            "L1/bf16/vgg-block",
            "general",
        ),
        (
            32,
            64,
            55,
            55,
            (3, 3),
            (2, 2),
            (0, 0),
            (1, 1),
            True,
            "bfloat16",
            "L1/bf16/alexnet-ceil",
            "general",
        ),
        (
            16,
            128,
            56,
            56,
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            False,
            "float32",
            "L1/fp32/vgg-block",
            "general",
        ),
        (
            32,
            64,
            55,
            55,
            (3, 3),
            (2, 2),
            (0, 0),
            (1, 1),
            True,
            "float32",
            "L1/fp32/alexnet-ceil",
            "general",
        ),
        (1, 1, 7, 7, (2, 2), (2, 2), (1, 1), (1, 1), False, "float16", "L1/fp16/c-eq-1", "general"),
        (
            2,
            3,
            13,
            17,
            (2, 3),
            (1, 2),
            (0, 1),
            (1, 2),
            False,
            "bfloat16",
            "L1/bf16/asym-k-s-p-d",
            "general",
        ),
        (
            1,
            7,
            9,
            9,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            True,
            "float32",
            "L1/fp32/odd-ceil",
            "general",
        ),
        (
            1,
            33,
            6,
            8,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            False,
            "float16",
            "L1/fp16/identity-cblk-tail",
            "identity",
        ),
        (
            2,
            4,
            9,
            11,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            False,
            "bfloat16",
            "L1/bf16/identity",
            "identity",
        ),
        (
            1,
            8,
            5,
            7,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            False,
            "float32",
            "L1/fp32/identity-fp32",
            "identity",
        ),
        (
            2,
            16,
            7,
            7,
            (7, 7),
            (1, 1),
            (0, 0),
            (1, 1),
            False,
            "bfloat16",
            "L1/bf16/global",
            "global",
        ),
        (1, 8, 8, 8, (8, 8), (1, 1), (0, 0), (1, 1), False, "float32", "L1/fp32/global", "global"),
        (1, 16, 5, 5, (5, 5), (1, 1), (0, 0), (1, 1), False, "float16", "L1/fp16/global", "global"),
        (
            1,
            8,
            20,
            20,
            (20, 20),
            (1, 1),
            (0, 0),
            (1, 1),
            False,
            "float16",
            "L1/fp16/global-20x20",
            "global",
        ),
        # Disjoint vgg-style windows (s == k) and a tiny stride-1 overlap.
        (
            2,
            16,
            12,
            14,
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            False,
            "float16",
            "L1/fp16/disjoint",
            "general",
        ),
        (
            1,
            8,
            11,
            13,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            False,
            "float16",
            "L1/fp16/overlap-s1",
            "general",
        ),
        # Padding on H only, and dilation on W only.
        (
            1,
            6,
            15,
            15,
            (5, 3),
            (3, 2),
            (2, 1),
            (1, 1),
            False,
            "float16",
            "L1/fp16/hpad-only",
            "general",
        ),
        (
            1,
            8,
            17,
            19,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 3),
            True,
            "bfloat16",
            "L1/bf16/wdilation",
            "general",
        ),
    ]
    for (
        n,
        c_in,
        h_in,
        w_in,
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
            h_in=h_in,
            w_in=w_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            dtype_str=dtype_str,
            tag="L1",
            expect_path=path,
        )

    # NaN / +-inf on the C4 whole-plane reduction path (blocking: the C4 NaN
    # judgement is an independent eq-mask reduction, never reduce_max's NaN
    # behaviour).
    case_id = "L1/fp16/global-nan-inf"
    try:
        x = torch.randn(2, 16, 5, 5, dtype=torch.float32)
        x[0, 0, :, :] = float("inf")
        x[0, 1, :, :] = float("nan")
        x[1, 3, 0, 0] = float("-inf")
        x[1, 3, 2, 2] = float("nan")
        x[1, 5, :, :] = float("-inf")
        x = x.to(torch.float16)
        _run_case(
            n=2,
            c_in=16,
            h_in=5,
            w_in=5,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            ceil_mode=False,
            dtype_str="float16",
            tag="L1",
            x=x,
            equal_nan=True,
            expect_path="global",
        )
        out = _get_kernel(2, 16, 5, 5, 5, 5, 1, 1, 0, 0, 1, 1, False, "float16")(x.npu()).cpu()
        assert out[0, 0, 0, 0].item() == float("inf"), (
            f"+inf plane gave {out[0, 0, 0, 0]} (expected +inf)"
        )
        assert bool(torch.isnan(out[0, 1, 0, 0])), "all-NaN plane did not give NaN"
        assert bool(torch.isnan(out[1, 3, 0, 0])), "NaN-in-plane did not win"
        assert out[1, 5, 0, 0].item() == float("-inf"), (
            f"all -inf plane gave {out[1, 5, 0, 0]} (expected -inf)"
        )
        print("[L1] PASS: C4 NaN/Inf semantics (inf plane -> inf, NaN wins)")
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
                h_in=11,
                w_in=13,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                dilation=(1, 1),
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
        kernel = _max_pool2d_kernel(1, 8, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1, False, "float16")(256)
        xx = torch.randn(1, 8, 16, 16, dtype=torch.float16)
        out = kernel(xx.npu()).cpu()
        torch.testing.assert_close(
            out,
            golden_max_pool2d(xx, (3, 3), (2, 2), (1, 1), (1, 1), False),
            rtol=0.0,
            atol=0.0,
        )
        print("[L1] PASS: block_m=256 dispatch (wrapper contract)")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")

    # block_m=0 is rejected.
    try:
        _max_pool2d_kernel(1, 8, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1, False, "float16")(0)
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
        (1, 5, 7, 3, (2, 2), (2, 2), (0, 0), (1, 1), False, "float16"),
        (1, 1, 1, 1, (1, 1), (1, 1), (0, 0), (1, 1), False, "bfloat16"),
        (1, 2, 1, 37, (1, 5), (1, 5), (0, 2), (1, 1), False, "float16"),
        (2, 7, 13, 5, (3, 2), (2, 3), (1, 1), (1, 1), True, "bfloat16"),
        (3, 1024, 3, 3, (1, 1), (1, 1), (0, 0), (1, 1), False, "float16"),
        (1, 16, 64, 64, (3, 3), (2, 2), (1, 1), (1, 1), False, "float32"),
        (1, 8, 128, 128, (2, 2), (2, 2), (0, 0), (1, 1), False, "float16"),
    ]
    for n, c_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype_str in cases:
        try:
            _run_case(
                n=n,
                c_in=c_in,
                h_in=h_in,
                w_in=w_in,
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
                f"[L2] WARN (record only): shape=({n},{c_in},{h_in},{w_in}) "
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
        h_in=9,
        w_in=11,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        ceil_mode=False,
        dtype_str="float16",
    )

    # Zeros: exact zeros in, zeros out.
    try:
        _run_case(tag="Boundary-zeros", x=torch.zeros(2, 16, 9, 11, dtype=torch.float16), **common)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): zeros: {exc}")

    # Constant plane: every window max equals the constant.
    try:
        _run_case(
            tag="Boundary-const", x=torch.full((2, 16, 9, 11), 3.7, dtype=torch.float16), **common
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): const: {exc}")

    # Large magnitudes (max selects, never overflows).
    try:
        _run_case(
            tag="Boundary-large",
            x=(torch.randn(2, 16, 9, 11, dtype=torch.float32) * 1000.0).to(torch.float16),
            **common,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large: {exc}")

    # Subnormal fp16 magnitudes.
    try:
        _run_case(
            tag="Boundary-subnormal",
            x=(torch.randn(2, 16, 9, 11, dtype=torch.float32) * 1e-5).to(torch.float16),
            **common,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {exc}")

    # +-inf without NaN must never produce NaN (equal_nan=False).
    try:
        x = torch.full((2, 16, 9, 11), float("-inf"), dtype=torch.float32)
        x[0, 1, 0, 0] = float("inf")
        x[1, 2, :, :] = float("inf")
        _run_case(tag="Boundary-pm-inf-only", x=x.to(torch.float16), **common)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): pm-inf-only: {exc}")

    # All-NaN window on the general path.
    try:
        x = torch.randn(2, 16, 9, 11, dtype=torch.float32)
        x[0, 0, :, :] = float("nan")
        _run_case(tag="Boundary-all-nan-window", x=x.to(torch.float16), equal_nan=True, **common)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): all-nan-window: {exc}")

    # NaN / +-inf on the C4 whole-plane path.
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
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
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
        x = torch.randn(1, 8, 6, 6, dtype=torch.float32)
        x[0, 2, 3, 3] = float("nan")
        _run_case(
            n=1,
            c_in=8,
            h_in=6,
            w_in=6,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
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
