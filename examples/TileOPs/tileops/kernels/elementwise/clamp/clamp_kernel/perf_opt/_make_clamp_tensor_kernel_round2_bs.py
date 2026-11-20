# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Tensor-bound clamp forward kernel (NPU, target="npuir", Developer mode).

Migrated from the GPU TileLang factory ``_make_clamp_tensor_kernel``
(Tensor-bound ``y = clamp(x, lo, hi)``; the non-fp8 branch computes in the
fp32 transit domain and adds explicit ``isnan`` NaN guards).

Semantics (DESIGN.md section 0.1, frozen by the integration contract):

    y[i] = min(max(x[i], lo[i]), hi[i]),  i in [0, N)

on three pre-broadcast, flattened, contiguous ``(N,)`` tensors of the same
dtype (float16 / bfloat16 / float32); output ``(N,)`` same dtype.  ``lo`` /
``hi`` are Tensors, present only when ``has_min`` / ``has_max`` is set.  Any
NaN operand propagates to the output; ``lo > hi`` yields ``hi`` (PyTorch
order-sensitive semantics).

NPU redesign (DESIGN.md section 0.6):

  * R1 -- 1-D ``T.Kernel(num_kernels, is_npu=True)`` + whole-buffer v-prefix
    vector chain replaces the CUDA ``block x thread`` two-level parallelism.
    ``threads * npt`` collapses into a single ``block_size`` (K11).
  * R2 -- computation domain is fp32 transit: the source non-fp8 branch is
    *already* fp32 transit (``cast -> min(max) -> single round back``), so
    fp16/fp32 are *preserved*; bfloat16 transit is *mandatory* because
    ``vclamp``/``vmin``/``vmax`` have no bf16 dtype in their matrix.  All
    half-precision dtypes go ``vcast(rint)`` up -> chain -> ``vcast(rint)``
    back (matches the contract golden's fp32 opmath + single round back).
  * R3 -- persistent core split: ``num_kernels = min(num_logical,
    vector_cores)`` with an in-core ``T.serial`` grid-stride over logical
    tiles and an if-guard masking out-of-range tiles (static bounds; ``N`` is
    a compile-time constant).
  * R5 -- no explicit NaN guards: device ``VMaxOp``/``VMinOp`` propagate NaN
    natively (device probe ``history_version/design_probe_clamp_nan.py``
    PROBE PASS + pattern-library TRAP-vmax-vmin-nan-propagation); the source
    CUDA guards are bit-redundant on NPU (8 ops -> 2 ops).

Hard constraints honoured (DESIGN.md sections 3.5.1 / 9.1):
  * R-7: every ``T.alloc_ub`` is allocated unconditionally at the
    ``T.Kernel`` top level (a conditional alloc triggers ``Undefined
    variable``, TRAP-conditional-alloc).
  * R-8: v-prefix ops act on the *whole* UB buffer; only ``T.copy`` uses the
    dynamic tail slice (``x_ub[0:tail]``, always zero-origin dst).
  * R-9: f16/bf16 <-> f32 ``vcast`` supports only ``round_mode="rint"``.
  * R-5: ``out_idx = [1 + n_in]`` (both -> ``[3]``, single bound -> ``[2]``);
    ``main`` has no runtime ``shape`` scalar so the input-argument count
    equals the frozen wrapper call sites ``_compiled_fn(x, lo, hi)`` /
    ``_compiled_fn(x, bound)``.
  * ``T.copy`` is same-dtype only (TRAP-C12-copy-dtype-cast): half-precision
    <-> fp32 always goes through explicit ``T.vcast``.

Factory contract (frozen wrapper
``examples/TileOPs/tileops/kernels/elementwise/clamp/clamp.py``):

    _make_clamp_tensor_kernel(N, dtype, has_min, has_max) -> kernel(block_size)

run: python _make_clamp_tensor_kernel.py --level all
"""

import os

# Developer mode for npuir: single-stage copy -> compute -> copy per tile,
# no manual sync points (DESIGN.md section 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

# Programming mode actually validated by this file (Stage 5 syncs this into
# the TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

__all__ = ["_make_clamp_tensor_kernel", "golden_clamp"]

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608

# ---------------------------------------------------------------------------
# Stage 4 (Round 1, r1_ubfree) UB budget -- sized to each path's LIVE set.
#
# The Stage 3 baseline allocated 8 buffers unconditionally (R-7) for every
# dtype: 4 input-dtype staging + 4 fp32 transit, i.e. 24 B/elem for fp16/bf16
# and 32 B/elem for fp32 -- even when the fp32 transit buffers are dead
# (float32 path) or only partly needed.  That capped the tile at 8192 (fp16) /
# 6144 (fp32) elements, which is the root cause of the small-burst DMA
# degradation measured in perf_opt/opt_log.md (baseline 16m-f16 = 70.8 us,
# 1.90 TB/s effective, aiv_mte2_ratio 0.96).
#
# r1_ubfree shrinks the live set per compile-time path:
#   * float32  -- the fp32 transit buffers are dead by construction (the data
#                 already is fp32); the chain runs in place on the staging
#                 buffer.  Live = 3 x 4 B = 12 B/elem.
#   * float16  -- `vclamp`/`vmax`/`vmin` have float16 in their dtype matrix,
#                 and for a *projection* operator round_fp16(clamp_fp32) ==
#                 clamp_fp16(round_fp16(...)) bit-for-bit (PL-1.17 4th proof),
#                 so the fp32 transit is dropped entirely.  Live = 3 x 2 B.
#   * bfloat16 -- no bf16 in the v-op dtype matrix, so the fp32 transit is
#                 mandatory; it is kept minimal (3 bf16 staging + 3 fp32
#                 transit, the clamp running in place in the transit buffer and
#                 the rounded result written back into the (dead) x staging
#                 buffer).  Live = 3 x 2 B + 3 x 4 B = 18 B/elem.
# ---------------------------------------------------------------------------
_LIVE_BYTES_PER_ELEM = {
    "float32": 12,
    "float16": 6,
    "bfloat16": 18,
}
_UB_MAX_BLOCK = {
    dtype: _UB_CAPACITY_BYTES // bpe for dtype, bpe in _LIVE_BYTES_PER_ELEM.items()
}

# Tile sizes (block_size) exposed to the wrapper.  Round 1 keeps the Stage 3
# K11-collapsed defaults so that r1 isolates the *structure* change; Round 2
# overwrites this table with the per-dtype tuned values (the D2 interface).
TUNED_BLOCK_SIZE = {
    "float32": 8192,
    "float16": 16384,
    "bfloat16": 10240,
}
# Alias kept for the L0/L1 harness and for the Stage-5 wrapper's perf_opt
# toggle block, which reads `TUNED_BLOCK_SIZE` first.
_DEFAULT_BLOCK_SIZE = TUNED_BLOCK_SIZE

_COMPUTE_DTYPE = "float32"


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_clamp(x, lo=None, hi=None):
    """Tensor-bound clamp reference (source semantics + contract benchmark).

    Contract (``examples/TileOPs/tests/ops/test_special_elementwise.py``
    ``ClampTensorTest.ref_program``):

        torch.clamp(x.float(), lo.float(), hi.float()).to(x.dtype)

    i.e. the clamp is evaluated in the fp32 opmath domain and rounded back to
    the input dtype exactly once.  ``lo`` / ``hi`` are optional (single-bound
    variants).  NaN semantics: any NaN operand -> NaN output; ``lo > hi`` ->
    ``hi`` (PyTorch order-sensitive).  This reference is independent from the
    NPU vector chain -- it never reuses vclamp / fp32 transit / the persistent
    structure, only the library implementation.
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    lo_f32 = None if lo is None else lo.detach().cpu().to(torch.float32)
    hi_f32 = None if hi is None else hi.detach().cpu().to(torch.float32)
    return torch.clamp(x_f32, lo_f32, hi_f32).to(x_cpu.dtype)


# ---------- Kernel ----------
def _make_clamp_tensor_kernel(N, dtype, has_min, has_max):
    """Build the Tensor-bound clamp kernel factory (E1: ``(N, dtype, has_min,
    has_max)``).

    Returns a JIT callable taking a single ``block_size`` (E2, the removed
    ``threads * npt`` product).  The bounded variant selection mirrors the
    source's three compile-time kernel specializations: both bounds ->
    ``main(x, lo, hi)`` with ``out_idx=[3]``; single bound -> ``main(x, bound)``
    with ``out_idx=[2]``.  ``dtype`` is a compile-time Python string selecting
    the computation domain (fp32 direct / fp16 native / bf16 fp32 transit).
    """
    if not (has_min or has_max):
        raise ValueError(
            "_make_clamp_tensor_kernel requires has_min or has_max to be True"
        )
    if dtype not in _UB_MAX_BLOCK:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_UB_MAX_BLOCK)}"
        )
    if N <= 0:
        # DESIGN.md section 9.3: N == 0 never reaches the kernel (the Op layer
        # intercepts); reject defensively with a clear message.
        raise ValueError(f"N must be a positive int, got N={N}")

    has_min = bool(has_min)
    has_max = bool(has_max)
    both = has_min and has_max

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op; measured 20 -> 40 on Ascend910_9362 / Ascend910B3).
    # Re-queried per factory call so deployments on other devices adapt.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    # Input argument count / out_idx (source contract): both -> [3], single
    # bound -> [2].
    n_in = (1 if has_min else 0) + (1 if has_max else 0)
    need_cast = dtype != _COMPUTE_DTYPE
    native16 = dtype == "float16"
    # bfloat16 is the only dtype that still needs the fp32 transit domain
    # (`vclamp`/`vmax`/`vmin` have no bf16 dtype -- PL-1.17).
    transit = need_cast and not native16
    max_block = _UB_MAX_BLOCK[dtype]
    default_block = _DEFAULT_BLOCK_SIZE[dtype]

    @tilelang.jit(out_idx=[1 + n_in], target="npuir")
    def kernel(block_size=None):
        # Host-side UB budget guard (DESIGN.md section 9.2).  ValueError is
        # used instead of a bare assert so the guard survives ``python -O``.
        if block_size is None:
            block_size = default_block
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if block_size > max_block:
            raise ValueError(
                f"block_size={block_size} exceeds the UB budget for {dtype} "
                f"(max {max_block} elements, {_UB_CAPACITY_BYTES} B UB capacity, "
                f"{_LIVE_BYTES_PER_ELEM[dtype]} B/elem live)"
            )

        # Persistent core split (DESIGN.md section 5.5).  All quantities are
        # compile-time Python ints: N and block_size are Python ints, so the
        # static loop bound is guaranteed (REVIEW.md dim-3 warning).
        num_logical = (N + block_size - 1) // block_size
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

        if both and transit:

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                lo: T.Tensor((N,), dtype),
                hi: T.Tensor((N,), dtype),
                y: T.Tensor((N,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # R-7: all UB buffers allocated unconditionally at the top
                    # level (a conditional T.alloc is illegal).
                    x_ub = T.alloc_ub((block_size,), dtype)
                    lo_ub = T.alloc_ub((block_size,), dtype)
                    hi_ub = T.alloc_ub((block_size,), dtype)
                    x32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)
                    lo32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)
                    hi32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off = block_id * block_size
                            tail = T.min(block_size, N - off)

                            # Load GM -> UB (same-dtype copies only, R-8).
                            T.copy(x[off : off + tail], x_ub[0:tail])
                            T.copy(lo[off : off + tail], lo_ub[0:tail])
                            T.copy(hi[off : off + tail], hi_ub[0:tail])

                            # R2: lossless up-cast to fp32 (rint) -- mandatory
                            # for bfloat16 (no native bf16 v-op dtype).
                            T.vcast(x_ub, x32, round_mode="rint")
                            T.vcast(lo_ub, lo32, round_mode="rint")
                            T.vcast(hi_ub, hi32, round_mode="rint")
                            # R5: no NaN guard; device max/min propagate NaN
                            # (TRAP-vmax-vmin-nan-propagation).  In-place
                            # min(max(x, lo), hi) -- order is order-sensitive
                            # and must not be swapped.
                            T.vmax(x32, lo32, x32)
                            T.vmin(x32, hi32, x32)
                            # Single round back to the input dtype (R2).  The
                            # bf16 staging buffer x_ub is dead after the up-cast
                            # (the vector ops run on x32), so the result lands
                            # there and no separate output buffer is needed.
                            T.vcast(x32, x_ub, round_mode="rint")

                            # Store UB -> GM.  Only [0:tail] is copied out (the
                            # v-ops ran on the whole buffer, so stale lanes
                            # follow).
                            T.copy(x_ub[0:tail], y[off : off + tail])

            return main

        if both:
            # float32 / float16-native: clamp in place in the input staging
            # buffer (PL-1.23 in-place v-op alias discipline).  The rounded
            # result is one of {x, lo, hi}, so the fp16 comparison domain is
            # bit-exact against the fp32-opmath golden (PL-1.17 4th proof).

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                lo: T.Tensor((N,), dtype),
                hi: T.Tensor((N,), dtype),
                y: T.Tensor((N,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((block_size,), dtype)
                    lo_ub = T.alloc_ub((block_size,), dtype)
                    hi_ub = T.alloc_ub((block_size,), dtype)

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off = block_id * block_size
                            tail = T.min(block_size, N - off)

                            T.copy(x[off : off + tail], x_ub[0:tail])
                            T.copy(lo[off : off + tail], lo_ub[0:tail])
                            T.copy(hi[off : off + tail], hi_ub[0:tail])

                            # y = min(max(x, lo), hi), in place on x_ub.
                            T.vmax(x_ub, lo_ub, x_ub)
                            T.vmin(x_ub, hi_ub, x_ub)

                            T.copy(x_ub[0:tail], y[off : off + tail])

            return main

        if has_min and transit:

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                lo: T.Tensor((N,), dtype),
                y: T.Tensor((N,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((block_size,), dtype)
                    lo_ub = T.alloc_ub((block_size,), dtype)
                    x32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)
                    lo32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off = block_id * block_size
                            tail = T.min(block_size, N - off)

                            T.copy(x[off : off + tail], x_ub[0:tail])
                            T.copy(lo[off : off + tail], lo_ub[0:tail])

                            T.vcast(x_ub, x32, round_mode="rint")
                            T.vcast(lo_ub, lo32, round_mode="rint")
                            # y = max(x, lo)  (both tensor operands)
                            T.vmax(x32, lo32, x32)
                            T.vcast(x32, x_ub, round_mode="rint")

                            T.copy(x_ub[0:tail], y[off : off + tail])

            return main

        if has_min:
            # float32 / float16-native, min only: y = max(x, lo), in place.

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                lo: T.Tensor((N,), dtype),
                y: T.Tensor((N,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((block_size,), dtype)
                    lo_ub = T.alloc_ub((block_size,), dtype)

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off = block_id * block_size
                            tail = T.min(block_size, N - off)

                            T.copy(x[off : off + tail], x_ub[0:tail])
                            T.copy(lo[off : off + tail], lo_ub[0:tail])

                            T.vmax(x_ub, lo_ub, x_ub)

                            T.copy(x_ub[0:tail], y[off : off + tail])

            return main

        if transit:
            # bfloat16 max-only: y = min(x, hi) via the mandatory fp32 transit.

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                hi: T.Tensor((N,), dtype),
                y: T.Tensor((N,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((block_size,), dtype)
                    hi_ub = T.alloc_ub((block_size,), dtype)
                    x32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)
                    hi32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off = block_id * block_size
                            tail = T.min(block_size, N - off)

                            T.copy(x[off : off + tail], x_ub[0:tail])
                            T.copy(hi[off : off + tail], hi_ub[0:tail])

                            T.vcast(x_ub, x32, round_mode="rint")
                            T.vcast(hi_ub, hi32, round_mode="rint")
                            # y = min(x, hi)
                            T.vmin(x32, hi32, x32)
                            T.vcast(x32, x_ub, round_mode="rint")

                            T.copy(x_ub[0:tail], y[off : off + tail])

            return main

        # max only, float32 / float16-native: y = min(x, hi), in place.

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            hi: T.Tensor((N,), dtype),
            y: T.Tensor((N,), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size,), dtype)
                hi_ub = T.alloc_ub((block_size,), dtype)

                for i in T.serial(num_local_tasks):
                    block_id = i * num_kernels + cid
                    if block_id < num_logical:
                        off = block_id * block_size
                        tail = T.min(block_size, N - off)

                        T.copy(x[off : off + tail], x_ub[0:tail])
                        T.copy(hi[off : off + tail], hi_ub[0:tail])

                        T.vmin(x_ub, hi_ub, x_ub)

                        T.copy(x_ub[0:tail], y[off : off + tail])

        return main

    return kernel


# ---------------------------------------------------------------------------
# Precision comparing helpers + hierarchical testing
# ---------------------------------------------------------------------------

_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Tolerances from examples/TileOPs/tests/ops/test_special_elementwise.py
# ``_get_tolerances`` (DESIGN.md section 8.2).  NaN cases use 0/0 with
# ``equal_nan=True`` (bit-exact assertion).
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
}

_FACTORY_CACHE = {}
_STATS_MAX_DIFF = {}
_CASE_COUNTS = {}
_FAILURES = []


def _get_factory(N, dtype_str, has_min, has_max):
    """Cache factory instances (mirrors the wrapper's one-factory-per-op
    usage; the jit wrapper then caches compiled mains per block_size)."""
    key = (N, dtype_str, has_min, has_max)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _make_clamp_tensor_kernel(N, dtype_str, has_min, has_max)
    return _FACTORY_CACHE[key]


def _layer_of(tag):
    return tag.split("-")[0]


def _bump(layer, ok):
    counts = _CASE_COUNTS.setdefault(layer, [0, 0])
    counts[0 if ok else 1] += 1


def _compare(out, ref, atol, rtol, equal_nan):
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
    max_diff = float(diff.max().item())
    violations = int((diff > (atol + rtol * ref.float().abs())).sum().item())
    return max_diff, violations


def _expand_flat(t, target_shape):
    """Expand ``t`` to ``target_shape`` and return a contiguous flat view
    (identical to the frozen Op layer ``ClampFwdOp._expand_flat``)."""
    if tuple(t.shape) != tuple(target_shape):
        t = t.expand(target_shape)
    return t.contiguous().view(-1)


def _rand_inputs(N, dtype_str, has_min, has_max, seed=0):
    """Generate (x, lo, hi) CPU inputs following the harness workload
    (``ClampTensorTest.gen_inputs``: randn / randn-0.5 / randn+0.5)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(N, dtype=torch.float32, generator=gen)
    lo = None
    hi = None
    if has_min:
        lo = torch.randn(N, dtype=torch.float32, generator=gen) - 0.5
    if has_max:
        hi = torch.randn(N, dtype=torch.float32, generator=gen) + 0.5
    return x, lo, hi


def _repeat_values(values, N, dtype):
    """Repeat a small value pattern up to N elements (edge / corner cases)."""
    base = torch.tensor(values, dtype=dtype)
    repeats = (N + len(values) - 1) // len(values)
    return base.repeat(repeats)[:N]


def _run_flat(
    dtype_str,
    tag,
    x,
    lo,
    hi,
    has_min,
    has_max,
    block_size=None,
    atol=None,
    rtol=None,
    equal_nan=False,
    display_shape=None,
):
    """Run one kernel case on flat CPU tensors and compare with the golden."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    t_atol, t_rtol = _TOLERANCE[dtype_str]
    if atol is None:
        atol = t_atol
    if rtol is None:
        rtol = t_rtol
    if block_size is None:
        block_size = _DEFAULT_BLOCK_SIZE[dtype_str]

    x = x.detach().cpu().reshape(-1).contiguous().to(torch_dtype)
    N = x.numel()
    assert N > 0, "N must be positive"

    args = [x.npu()]
    ref_lo = None
    ref_hi = None
    if has_min:
        lo = lo.detach().cpu().reshape(-1).contiguous().to(torch_dtype)
        assert lo.numel() == N, f"lo numel {lo.numel()} != N {N}"
        ref_lo = lo
        args.append(lo.npu())
    if has_max:
        hi = hi.detach().cpu().reshape(-1).contiguous().to(torch_dtype)
        assert hi.numel() == N, f"hi numel {hi.numel()} != N {N}"
        ref_hi = hi
        args.append(hi.npu())

    kernel = _get_factory(N, dtype_str, has_min, has_max)(block_size)
    out = kernel(*args).detach().cpu().reshape(-1)
    ref = golden_clamp(x, ref_lo, ref_hi)

    max_diff, violations = _compare(out, ref, atol, rtol, equal_nan)
    if max_diff == max_diff:  # skip NaN-only max_diff
        best = _STATS_MAX_DIFF.get(dtype_str)
        if best is None or max_diff > best:
            _STATS_MAX_DIFF[dtype_str] = max_diff

    shape = tuple(display_shape) if display_shape is not None else (N,)
    try:
        torch.testing.assert_close(
            out, ref, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    except AssertionError as exc:
        raise AssertionError(
            f"[{tag}] FAIL: shape={shape} dtype={dtype_str} "
            f"has_min={has_min} has_max={has_max} block_size={block_size} "
            f"max_diff={max_diff:.6e} violations={violations}/{N}\n{exc}"
        ) from exc
    print(
        f"[{tag}] PASS: shape={shape} dtype={dtype_str} "
        f"has_min={has_min} has_max={has_max} block_size={block_size} "
        f"max_diff={max_diff:.3e}"
    )
    return max_diff, violations


def _run_broadcast(
    dtype_str,
    tag,
    x,
    lo,
    hi,
    has_min,
    has_max,
    block_size=None,
    equal_nan=False,
):
    """Broadcast the three operands to the output shape, materialize them
    flat (Host-side materialization, DESIGN.md section 0.6 R4-A) and run."""
    shapes = [tuple(x.shape)]
    if has_min:
        shapes.append(tuple(lo.shape))
    if has_max:
        shapes.append(tuple(hi.shape))
    out_shape = tuple(torch.broadcast_shapes(*shapes))
    x_flat = _expand_flat(x, out_shape)
    lo_flat = _expand_flat(lo, out_shape) if has_min else None
    hi_flat = _expand_flat(hi, out_shape) if has_max else None
    return _run_flat(
        dtype_str,
        tag,
        x_flat,
        lo_flat,
        hi_flat,
        has_min,
        has_max,
        block_size=block_size,
        equal_nan=equal_nan,
        display_shape=out_shape,
    )


def _step(layer, case_id, blocking=True, **kwargs):
    """Run one case, collecting (not re-raising) per-case failures."""
    try:
        _run_flat(**kwargs)
    except Exception as exc:  # noqa: BLE001
        _bump(layer, False)
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        if blocking:
            _FAILURES.append((case_id, exc))
            print(f"[collected] {case_id} FAILED: {first_line}")
        else:
            print(f"[{layer}] WARN (Record without blocking) {case_id}: {first_line}")
        return False
    _bump(layer, True)
    return True


def _step_broadcast(layer, case_id, blocking=True, **kwargs):
    try:
        _run_broadcast(**kwargs)
    except Exception as exc:  # noqa: BLE001
        _bump(layer, False)
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        if blocking:
            _FAILURES.append((case_id, exc))
            print(f"[collected] {case_id} FAILED: {first_line}")
        else:
            print(f"[{layer}] WARN (Record without blocking) {case_id}: {first_line}")
        return False
    _bump(layer, True)
    return True


# ---------- L0 (blocking gate, DESIGN.md section 8.3) ----------
def run_L0():
    # L0-1: smoke, both bounds, all dtypes, N=2^20 (harness randn pattern).
    for dtype_str in ("float32", "float16", "bfloat16"):
        x, lo, hi = _rand_inputs(2**20, dtype_str, True, True, seed=1)
        _step(
            "L0",
            f"L0-1/{dtype_str}",
            dtype_str=dtype_str,
            tag="L0-1",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
        )

    # L0-2: large-scale spot check (elementwise-16m representative).
    for dtype_str in ("float32", "float16"):
        x, lo, hi = _rand_inputs(2**24, dtype_str, True, True, seed=2)
        _step(
            "L0",
            f"L0-2/{dtype_str}",
            dtype_str=dtype_str,
            tag="L0-2",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
        )

    # L0-3: NaN propagation (contract test form, bit-exact).
    for dtype_str in ("float32", "float16"):
        N = 256
        x, lo, hi = _rand_inputs(N, dtype_str, True, True, seed=3)
        x[0::8] = float("nan")
        lo[1::8] = float("nan")
        hi[2::8] = float("nan")
        _step(
            "L0",
            f"L0-3/{dtype_str}",
            dtype_str=dtype_str,
            tag="L0-3",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )

    # L0-4: lo > hi -> result is hi.
    for dtype_str in ("float32", "float16"):
        N = 4096
        x = torch.linspace(-3.0, 3.0, N)
        lo = torch.full((N,), 2.0)
        hi = torch.full((N,), 1.0)
        _step(
            "L0",
            f"L0-4/{dtype_str}/global",
            dtype_str=dtype_str,
            tag="L0-4",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
        )
    x, _, _ = _rand_inputs(4096, "float32", True, True, seed=4)
    mn = torch.randn(4096) - 0.5
    mx = torch.randn(4096) + 0.5
    _step(
        "L0",
        "L0-4/mixed",
        dtype_str="float32",
        tag="L0-4",
        x=x,
        lo=mn,
        hi=mx,
        has_min=True,
        has_max=True,
    )

    # L0-5: single-bound variants (min-only / max-only) + call-site arity.
    for dtype_str in ("float32", "float16", "bfloat16"):
        x, lo, hi = _rand_inputs(2**20, dtype_str, True, True, seed=5)
        _step(
            "L0",
            f"L0-5/{dtype_str}/min-only",
            dtype_str=dtype_str,
            tag="L0-5",
            x=x,
            lo=lo,
            hi=None,
            has_min=True,
            has_max=False,
        )
        _step(
            "L0",
            f"L0-5/{dtype_str}/max-only",
            dtype_str=dtype_str,
            tag="L0-5",
            x=x,
            lo=None,
            hi=hi,
            has_min=False,
            has_max=True,
        )

    # L0-6: broadcast / 0-dim bounds (host materialization path).
    x = torch.randn(4, 8)
    lo = torch.randn(1, 8) - 0.5
    hi = torch.randn(4, 1) + 0.5
    _step_broadcast(
        "L0",
        "L0-6/3-way",
        dtype_str="float32",
        tag="L0-6",
        x=x,
        lo=lo,
        hi=hi,
        has_min=True,
        has_max=True,
    )
    _step_broadcast(
        "L0",
        "L0-6/0-dim",
        dtype_str="float32",
        tag="L0-6",
        x=x,
        lo=torch.tensor(0.25),
        hi=torch.tensor(0.75),
        has_min=True,
        has_max=True,
    )
    _step_broadcast(
        "L0",
        "L0-6/min-only-brc",
        dtype_str="float16",
        tag="L0-6",
        x=x,
        lo=torch.randn(1, 8) - 0.5,
        hi=None,
        has_min=True,
        has_max=False,
    )
    _step_broadcast(
        "L0",
        "L0-6/max-only-0dim",
        dtype_str="float32",
        tag="L0-6",
        x=x,
        lo=None,
        hi=torch.tensor(0.5),
        has_min=False,
        has_max=True,
    )
    # lo > hi under broadcast.
    _step_broadcast(
        "L0",
        "L0-6/lo-gt-hi",
        dtype_str="float32",
        tag="L0-6",
        x=x,
        lo=torch.randn(4, 1) + 2.0,
        hi=torch.randn(1, 8) - 2.0,
        has_min=True,
        has_max=True,
    )

    # L0-7: non-divisible tails (single partial tile / non-32B-aligned tail).
    for dtype_str in ("float16", "float32"):
        for N in (1000, 4097, 5000):
            x, lo, hi = _rand_inputs(N, dtype_str, True, True, seed=7)
            _step(
                "L0",
                f"L0-7/{dtype_str}/N={N}",
                dtype_str=dtype_str,
                tag="L0-7",
                x=x,
                lo=lo,
                hi=hi,
                has_min=True,
                has_max=True,
            )

    # L0-8: tiny shapes (single non-full tile / num_logical=1 degenerate).
    for dtype_str, N in (
        ("float16", 1),
        ("float16", 7),
        ("float16", 32),
        ("float32", 32),
        ("bfloat16", 7),
    ):
        x, lo, hi = _rand_inputs(N, dtype_str, True, True, seed=8)
        _step(
            "L0",
            f"L0-8/{dtype_str}/N={N}",
            dtype_str=dtype_str,
            tag="L0-8",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
        )

    # L0-9: endpoints / extremes (+-0, +-inf, subnormal, 65504, +-88.8).
    x_vals = [0.0, -0.0, 1e-45, -1e-45, 88.8, -88.8, 65504.0, -65504.0, float("inf"), float("-inf")]
    lo_vals = [float("-inf"), -1.0, -0.0, 0.0, 1e-45, 1.0, 88.8, -65504.0, float("-inf"), float("inf")]
    hi_vals = [float("inf"), 1.0, 0.0, -0.0, -1e-45, -1.0, -88.8, 65504.0, float("inf"), float("-inf")]
    for dtype_str in ("float32", "float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        x = _repeat_values(x_vals, 4096, td)
        lo = _repeat_values(lo_vals, 4096, td)
        hi = _repeat_values(hi_vals, 4096, td)
        _step(
            "L0",
            f"L0-9/{dtype_str}",
            dtype_str=dtype_str,
            tag="L0-9",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
        )

    # L0-10: persistent core-count switch boundary (num_logical in {39,40,41}).
    bs = 4096
    for num_logical, N in (
        (39, 39 * bs),
        (40, 40 * bs),
        (41, 41 * bs),
        (41, 41 * bs - 1000),
        (40, 40 * bs - 1000),
    ):
        x, lo, hi = _rand_inputs(N, "float16", True, True, seed=10)
        _step(
            "L0",
            f"L0-10/fp16/num_logical={num_logical}/N={N}",
            dtype_str="float16",
            tag="L0-10",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
            block_size=bs,
        )


# ---------- L1 (blocking gate: full functional coverage) ----------
def run_L1():
    # dtype x {exact multiple, tail==1-ish, non-32B-aligned tail}.
    for dtype_str in ("float16", "bfloat16", "float32"):
        for N in (8192, 1000, 4097):
            x, lo, hi = _rand_inputs(N, dtype_str, True, True, seed=11)
            _step(
                "L1",
                f"L1/{dtype_str}/N={N}",
                dtype_str=dtype_str,
                tag="L1",
                x=x,
                lo=lo,
                hi=hi,
                has_min=True,
                has_max=True,
            )

    # Non-default block sizes (wrapper config sweep).
    for dtype_str, N, bsz in (
        ("float16", 2**20, 2048),
        ("float16", 2**20, 8192),
        ("float32", 2**20, 4096),
        ("float32", 2**20, 6144),
        ("bfloat16", 2**20, 8192),
    ):
        x, lo, hi = _rand_inputs(N, dtype_str, True, True, seed=12)
        _step(
            "L1",
            f"L1/{dtype_str}/bs={bsz}",
            dtype_str=dtype_str,
            tag="L1",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
            block_size=bsz,
        )

    # Single-bound variants across all dtypes.
    for dtype_str in ("float16", "bfloat16", "float32"):
        x, lo, hi = _rand_inputs(2**20, dtype_str, True, True, seed=13)
        _step(
            "L1",
            f"L1/{dtype_str}/min-only",
            dtype_str=dtype_str,
            tag="L1",
            x=x,
            lo=lo,
            hi=None,
            has_min=True,
            has_max=False,
        )
        _step(
            "L1",
            f"L1/{dtype_str}/max-only",
            dtype_str=dtype_str,
            tag="L1",
            x=x,
            lo=None,
            hi=hi,
            has_min=False,
            has_max=True,
        )

    # NaN propagation for bfloat16 (fp32 transit path).
    N = 512
    x, lo, hi = _rand_inputs(N, "bfloat16", True, True, seed=14)
    x[0::4] = float("nan")
    lo[1::4] = float("nan")
    hi[2::4] = float("nan")
    _step(
        "L1",
        "L1/bf16/nan",
        dtype_str="bfloat16",
        tag="L1",
        x=x,
        lo=lo,
        hi=hi,
        has_min=True,
        has_max=True,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )

    # Wrapper E2 contract: one factory callable serving two block_sizes
    # (the jit wrapper caches the compiled main per block_size).
    try:
        dtype_str = "float16"
        N = 8192
        x, lo, hi = _rand_inputs(N, dtype_str, True, True, seed=15)
        td = _TORCH_DTYPE[dtype_str]
        x_t = x.to(td).npu()
        lo_t = lo.to(td).npu()
        hi_t = hi.to(td).npu()
        ref = golden_clamp(x.to(td), lo.to(td), hi.to(td))
        fn = _get_factory(N, dtype_str, True, True)
        for bsz in (4096, 2048):
            out = fn(bsz)(x_t, lo_t, hi_t).cpu().reshape(-1)
            torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
        print("[L1] PASS: E2 factory(block_size) dual-config reuse")
        _bump("L1", True)
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append(("L1/E2-reuse", exc))
        _bump("L1", False)
        print(f"[collected] L1/E2-reuse FAILED: {exc!r}")

    # UB budget guards (DESIGN.md section 9.2).
    for dtype_str, bsz in (("bfloat16", 10923), ("float16", 32769), ("float32", 16385)):
        try:
            _make_clamp_tensor_kernel(4096, dtype_str, True, True)(bsz)
        except ValueError as exc:
            print(f"[L1] PASS: UB guard rejects {dtype_str} block_size={bsz} ({exc})")
            _bump("L1", True)
        except Exception as exc:  # noqa: BLE001
            _FAILURES.append((f"L1/UB-guard-{dtype_str}", exc))
            _bump("L1", False)
            print(f"[collected] L1/UB-guard-{dtype_str} FAILED: {exc!r}")
        else:
            _FAILURES.append(
                (f"L1/UB-guard-{dtype_str}", AssertionError(f"{bsz} not rejected"))
            )
            _bump("L1", False)
            print(f"[collected] L1/UB-guard-{dtype_str} FAILED: not rejected")

    # both-None rejection (source contract: has_min or has_max must hold).
    try:
        _make_clamp_tensor_kernel(1024, "float32", False, False)
    except ValueError as exc:
        print(f"[L1] PASS: both-None rejected ({exc})")
        _bump("L1", True)
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append(("L1/both-none-reject", exc))
        _bump("L1", False)
        print(f"[collected] L1/both-none-reject FAILED: {exc!r}")
    else:
        _FAILURES.append(("L1/both-none-reject", AssertionError("not rejected")))
        _bump("L1", False)
        print("[collected] L1/both-none-reject FAILED: not rejected")


# ---------- L2 (warn only, non-blocking) ----------
def run_L2():
    for dtype_str, N in (
        ("float16", 2),
        ("float32", 3),
        ("bfloat16", 2**22),
        ("float16", 2**25),
    ):
        x, lo, hi = _rand_inputs(N, dtype_str, True, True, seed=20)
        _step(
            "L2",
            f"L2/{dtype_str}/N={N}",
            blocking=False,
            dtype_str=dtype_str,
            tag="L2",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
        )


# ---------- Boundary (warn only, non-blocking) ----------
def run_boundary():
    N = 4096

    # Special values (+-0, +-inf, NaN, subnormal, large) per dtype.
    for dtype_str in ("float32", "float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        x, lo, hi = _rand_inputs(N, dtype_str, True, True, seed=21)
        x = _repeat_values(
            [0.0, -0.0, 1e-7, -1e-7, 1e-3, -1e-3, 1.0, -1.0, 88.8, -88.8], N, td
        )
        lo = _repeat_values([float("-inf"), -1.0, 0.0, -0.5, 0.5], N, td)
        hi = _repeat_values([float("inf"), 1.0, -1.0, 0.5, -0.5], N, td)
        _step(
            "Boundary",
            f"Boundary-special/{dtype_str}",
            blocking=False,
            dtype_str=dtype_str,
            tag="Boundary-special",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
        )

    # NaN placed in each operand slot, per dtype (bit-exact).
    for dtype_str in ("float32", "float16", "bfloat16"):
        N = 256
        x, lo, hi = _rand_inputs(N, dtype_str, True, True, seed=22)
        x[0::16] = float("nan")
        lo[1::16] = float("nan")
        hi[2::16] = float("nan")
        x[3::16] = float("inf")
        lo[4::16] = float("-inf")
        hi[5::16] = float("-inf")
        _step(
            "Boundary",
            f"Boundary-nan/{dtype_str}",
            blocking=False,
            dtype_str=dtype_str,
            tag="Boundary-nan",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )

    # Subnormal magnitudes (atol dominates).
    for dtype_str in ("float16", "bfloat16"):
        N = 4096
        x = (torch.randn(N) * 1e-6).to(_TORCH_DTYPE[dtype_str])
        lo = (torch.randn(N) * 1e-6 - 1e-7).to(_TORCH_DTYPE[dtype_str])
        hi = (torch.randn(N) * 1e-6 + 1e-7).to(_TORCH_DTYPE[dtype_str])
        _step(
            "Boundary",
            f"Boundary-subnormal/{dtype_str}",
            blocking=False,
            dtype_str=dtype_str,
            tag="Boundary-subnormal",
            x=x,
            lo=lo,
            hi=hi,
            has_min=True,
            has_max=True,
        )

    # Broadcast with special-valued bounds.
    x = torch.randn(8, 16)
    lo = torch.tensor([[-float("inf"), 0.0, 1e-3, -1e-3] * 4], dtype=torch.float32)
    hi = torch.tensor([[float("inf"), 1.0, -1.0, 0.5] * 4], dtype=torch.float32)
    _step_broadcast(
        "Boundary",
        "Boundary-brc-special",
        blocking=False,
        dtype_str="float32",
        tag="Boundary-brc-special",
        x=x,
        lo=lo,
        hi=hi,
        has_min=True,
        has_max=True,
    )


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()

    print("Case counts (layer: pass/fail):")
    for layer in ("L0", "L1", "L2", "Boundary"):
        counts = _CASE_COUNTS.get(layer, [0, 0])
        print(f"  {layer}: {counts[0]} pass / {counts[1]} fail")
    print(
        "Summary max_diff per dtype: "
        + (", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items())) or "n/a")
    )
    if _FAILURES:
        raise AssertionError(
            f"{len(_FAILURES)} case(s) failed: "
            + "; ".join(cid for cid, _ in _FAILURES)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
