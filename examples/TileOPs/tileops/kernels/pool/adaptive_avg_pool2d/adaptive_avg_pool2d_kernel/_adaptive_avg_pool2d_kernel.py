# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Adaptive average pooling 2D forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``_adaptive_avg_pool2d_kernel`` (TileOPs
``AdaptiveAvgPool2dFwdOp``) to TileLang ``target="npuir"``.

Semantics (DESIGN.md section 0.1, frozen):
    out[n, c, oh, ow] = mean(x[n, c, ih0:ih1, iw0:iw1])
    (ih0, ih1) = (floor(oh * H_in / out_H), ceil((oh+1) * H_in / out_H))
    (iw0, iw1) = (floor(ow * W_in / out_W), ceil((ow+1) * W_in / out_W))
    fp32 accumulation, one final round-back to the input dtype
    (float16 / bfloat16 only; the Op layer rejects float32).

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel`` + ``(n, o_h, c_block)`` logical units with a
        persistent in-core ``T.serial`` grid-stride loop and an
        ``if uid < NUM_LOGICAL`` tail guard (static bounds).
    R2: UB fp32 vector accumulator on the channel axis; per-row
        ``T.vcast(rint)`` up to fp32, ``T.vadd`` slice accumulation
        (kh serial, o_w/kw compile-time expanded), reciprocal multiply
        for the normalisation (DESIGN.md section 1.6.1 C1).
    R3: per-row GM -> UB ``T.copy`` + in-core two-axis ``T.transpose``
        ([1, 0]) so the channel axis becomes the contiguous vector axis;
        the output is transposed back before the row-slice store.
    R4: degenerate-grid specialisation: C4 (out_H == out_W == 1)
        ``T.reduce_sum`` over the whole plane, C4b (out == in) identity
        UB transit copy (DESIGN.md section 1.6.1 C4 / C4b).

Implementation notes (Stage 3):
    * The frozen core-split unit granularity is kept: general grid
      ``NUM_LOGICAL = N * out_H * num_cblk``; C4 / C4b use the whole-plane
      unit ``NUM_LOGICAL = N * num_cblk`` (DESIGN.md sections 5.2 / 5.5).
    * Channel-block tails (``C % CH != 0``, e.g. C=96 with CH=64) use
      dynamic ``real_c = min(CH, C - cb*CH)`` copy extents; the padded
      ``[real_c:CH]`` lanes never reach GM.
    * The channel vector width is rounded up to a multiple of 8 (the fp16
      vector width) so the vector axis stays aligned for any C.
    * UB budget guard: the whole-plane buffers used by C4 / C4b are only
      used when they fit the UB budget (conservative x2 multi-buffer
      factor); otherwise the kernel falls back to the general path,
      which is always correct (the specialisation is a pure optimisation).
    * ``block_m`` is accepted by the returned callable for source
      interface compatibility (K9 kept the wrapper's ``block_m`` config
      knob); NPU tiling is shape-derived (``CH = min(C, 64)`` upward
      rounded to 8), so the value is validated and otherwise ignored.

Factory contract (source interface unchanged):
    _adaptive_avg_pool2d_kernel(n, c_in, h_in, w_in, out_h, out_w,
                                dtype="float16")
      -> build(block_m=None) -> main
    where ``main(x)`` returns ``out`` (``out_idx=[1]``).

Run the embedded hierarchical tests:
    python _adaptive_avg_pool2d_kernel.py --level L0
    python _adaptive_avg_pool2d_kernel.py --level all
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

# Programming mode actually verified in Stage 3 (Stage 5 syncs the wrapper's
# Kernel.ascend_mode from this literal).
ASCEND_MODE = "Developer"

__all__ = [
    "ASCEND_MODE",
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
# serial loop is active (measured 1.7x-2.5x on this toolchain, see
# tilelang-op-optimize pattern-library traps-*/UB-multibuffer-inflation).
_UB_MULTIBUFFER_FACTOR = 2

# Supported dtypes (the source op rejects float32).
_SUPPORTED_DTYPES = ("float16", "bfloat16")

# Precision tolerances (DESIGN.md section 8.2, identical to the source repo
# test_pool.py: (1e-3, 1e-3) fp16 / (1.6e-2, 1.6e-2) bf16).
_TOLERANCE = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "bfloat16": (torch.bfloat16, 1.6e-2, 1.6e-2),
}

# fp16/bf16 vector width in elements (alignment of the vector axis).
_VECTOR_WIDTH = 8


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_adaptive_avg_pool2d(x: torch.Tensor, output_size):
    """PyTorch reference for adaptive_avg_pool2d, per DESIGN.md section 0.1.

    Directly uses the source-repo test benchmark
    (examples/TileOPs/tests/ops/test_pool.py: ``ref_program``).  Accepts 3D
    CHW or 4D NCHW; ``None`` entries keep the input size.  Independent of the
    NPU algorithm (no transpose / no C-vectorisation / no C4-C4b
    specialisation).  Runs on CPU and does not require an NPU device.
    """
    size = (None, None) if output_size is None else output_size
    return F.adaptive_avg_pool2d(x.detach().float(), size).to(x.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
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

    # Channel vector block: C axis is the vector axis (DESIGN.md 1.6.3), so it
    # must be a multiple of the fp16 vector width; capped at 64 by the UB
    # budget (DESIGN.md 4.5 / 5.2, D-3 primary choice).
    ch = min(c_in, 64)
    ch = max(_VECTOR_WIDTH, ((ch + _VECTOR_WIDTH - 1) // _VECTOR_WIDTH) * _VECTOR_WIDTH)
    num_cblk = (c_in + ch - 1) // ch
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4

    # Degenerate-grid specialisations (DESIGN.md 0.6 R4 / 1.6.1 C4, C4b).
    # Only used when the whole-plane buffers fit the UB budget; the general
    # path is always correct, so an oversized plane simply falls back.
    is_identity = out_h == h_in and out_w == w_in
    plane_elems = ch * h_in * w_in
    identity_fits = plane_elems * elem_bytes * _UB_MULTIBUFFER_FACTOR <= _UB_CAPACITY_BYTES
    global_fits = plane_elems * (elem_bytes + 4) * _UB_MULTIBUFFER_FACTOR <= _UB_CAPACITY_BYTES
    is_global = (out_h == 1 and out_w == 1) and global_fits
    if is_identity and not identity_fits:
        is_identity = False

    # Logical unit granularity is the single definition source of NUM_LOGICAL
    # (DESIGN.md 5.5 item 1): general unit = (n, o_h, c_block); C4 / C4b unit =
    # the whole (n, c_block) plane.
    num_logical = n * num_cblk if is_identity or is_global else n * out_h * num_cblk

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

            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_identity(ch_: int):
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
            inv_plane = 1.0 / float(h_in * w_in)

            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_global(ch_: int):
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, h_in, w_in), dtype),
                    out: T.Tensor((n, c_in, out_h, out_w), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        flat = T.alloc_ub((ch, h_in, w_in), dtype)
                        flat32 = T.alloc_ub((ch, h_in, w_in), "float32")
                        mid = T.alloc_ub((ch, h_in, 1), "float32")
                        gacc = T.alloc_ub((ch, 1, 1), "float32")
                        gout = T.alloc_ub((ch, 1, 1), dtype)
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
                                T.vcast(flat, flat32, round_mode="rint")
                                T.reduce_sum(flat32, mid, dim=2)
                                T.reduce_sum(mid, gacc, dim=1)
                                T.vmul(gacc, inv_plane, gacc)
                                T.vcast(gacc, gout, round_mode="rint")
                                T.copy(
                                    gout[0:real_c, 0:1, 0:1],
                                    out[nb, cb * ch : cb * ch + real_c, 0:1, 0:1],
                                )

                return main

            return _func_global(ch)

        # ---------------- general grid ----------------
        @tilelang.jit(out_idx=[1], target="npuir")
        def _func_general(ch_: int):
            @T.prim_func
            def main(
                x: T.Tensor((n, c_in, h_in, w_in), dtype),
                out: T.Tensor((n, c_in, out_h, out_w), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    acc = T.alloc_ub((out_w, ch), "float32")
                    row_src = T.alloc_ub((ch, w_in), dtype)
                    row_dst = T.alloc_ub((w_in, ch), dtype)
                    row_f32 = T.alloc_ub((w_in, ch), "float32")
                    acc16 = T.alloc_ub((out_w, ch), dtype)
                    acc_t = T.alloc_ub((ch, out_w), dtype)

                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // (out_h * num_cblk)
                            rem = uid % (out_h * num_cblk)
                            oh = rem // num_cblk
                            cb = rem % num_cblk
                            real_c = T.min(ch, c_in - cb * ch)
                            ih0, ih1 = adaptive_bin(oh, h_in, out_h)
                            bh = ih1 - ih0

                            T.clear(acc)
                            for kh in T.serial(max_bin_h):
                                if kh < bh:
                                    T.copy(
                                        x[nb, cb * ch : cb * ch + real_c, ih0 + kh, 0:w_in],
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
                                                    acc[ow : ow + 1, 0:ch],
                                                    row_f32[iw0 + kw : iw0 + kw + 1, 0:ch],
                                                    acc[ow : ow + 1, 0:ch],
                                                )
                            for ow in T.serial(out_w):
                                iw0, iw1 = adaptive_bin(ow, w_in, out_w)
                                inv = 1.0 / T.cast(bh * (iw1 - iw0), "float32")
                                T.vmul(acc[ow : ow + 1, 0:ch], inv, acc[ow : ow + 1, 0:ch])
                            T.vcast(acc, acc16, round_mode="rint")
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
_FACTORY_CACHE = {}

# Per-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_factory(n, c_in, h_in, w_in, out_h, out_w, dtype_str):
    """Cache factory instances (one factory per shape/dtype, mirroring the
    wrapper's per-shape kernel cache)."""
    key = (n, c_in, h_in, w_in, out_h, out_w, dtype_str)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _adaptive_avg_pool2d_kernel(
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

    ref = golden_adaptive_avg_pool2d(x, output_size)
    if ndim3:
        assert out.shape == (c_in, out_h, out_w), f"3D rank not preserved: {out.shape}"
        ref = golden_adaptive_avg_pool2d(x[0], output_size)

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
    """L0 gate suite (DESIGN.md section 8.3, all 10 items).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    cases = [
        (1, 8, 4, 4, (2, 2), "float16", "L0-1/smoke"),  # manifest smoke
        (2, 128, 56, 56, (6, 6), "float16", "L0-2/spp"),  # manifest spp-6x6
        (8, 2048, 7, 7, (1, 1), "float16", "L0-3/resnet-global"),  # C4 + persistent
        (2, 64, 55, 57, (7, 7), "bfloat16", "L0-4/nondiv-bf16"),  # manifest nondiv
        (1, 8, 9, 11, None, "float16", "L0-5/identity"),  # C4b
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
    """L1: full functional coverage (shape x dtype x specialisation path).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    cases = [
        # Channel-block tail (C=96 -> CH=64 + real_c=32), bf16 asymmetric bins.
        (1, 96, 28, 30, (7, 5), "bfloat16", "L1/bf16/cblk-tail"),
        # Non-divisible fp16, W_in=57 (odd tail on the transposed row axis).
        (1, 128, 55, 57, (7, 7), "float16", "L1/fp16/nondiv"),
        # bf16 on the manifest spp shape.
        (2, 128, 56, 56, (6, 6), "bfloat16", "L1/bf16/spp"),
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

    # Factory contract guard: non-positive block_m is rejected (the value is
    # otherwise ignored, DESIGN.md section 3.5.2 K9).
    try:
        _adaptive_avg_pool2d_kernel(1, 8, 4, 4, 2, 2, "float16")(0)
    except ValueError as exc:
        print(f"[L1] PASS: block_m guard rejects 0 ({exc})")
    else:
        failures.append(("L1/block_m-guard", AssertionError("block_m=0 not rejected")))

    # Unsupported dtype / non-positive shapes are rejected.
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

    # Constant plane: bins of varying size average back to the constant.
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

    # Large magnitudes (fp16 near-max scale; overflow-free averaging).
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

    # inf / NaN natural IEEE propagation (DESIGN.md section 0.1: no special
    # handling; a bin containing +inf and -inf averages to NaN on both sides).
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

    # Global pool of an inf/NaN plane (C4 reduce path NaN propagation).
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
