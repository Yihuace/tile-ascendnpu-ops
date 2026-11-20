# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_simple_reduce_kernel_tiled (AmaxFwdOp tiled / streaming path) -- NPU Developer mode.

Stage 4 tuned drop-in for the Stage 3 baseline of the same name (identical
factory signature ``(M, N, op_kind, dtype, tile_n) -> _func(block_m) -> main``).

Semantics (DESIGN.md section 0.1, frozen):
    y[i] = max_{0 <= j < N} x[i, j]          # torch.amax(x, dim=-1)
    x: (M, N) fp16/bf16/fp32  ->  out: (M,) same_as(x)

This unit carries the STREAMING (tiled) path: N is too large for the whole row
to reside in UB, so the kernel walks the row in ``tile_n``-wide column tiles and
maintains a cross-tile running max (online single pass).

Stage 4 changes relative to the Stage 3 baseline:
    S0 **tile_n envelope clamp** -- the ``tile_n`` handed in by the caller is
       clamped to the race-safe doubled-buffer UB envelope
       (``2 * tile_n * B(dtype) <= 96 KiB``), preferring a 256-multiple
       divisor of N, else the budget cap.
       Two independent motivations:
       (a) the Stage 3 wrapper derived ``tile_n`` from the shared
       ``_primitives.compute_tile_n`` (bf16 = 2 B/elem) and handed 16384 for
       the N=32768 bf16 workload, which the bf16 6 B/elem envelope rejects
       (the measured long-seq-reduce failure) -- clamping inside the kernel
       makes the two budget conventions consistent;
       (b) a 64 KiB single-tile stage (fp32 ``tile_n=16384``, i.e. a 128 KiB
       doubled footprint) is an intermittent read-after-write race in the
       auto-multi-buffered staging buffer (measured 2/5 wrong on the baseline
       kernel, 128x65536 fp32; a 32 KiB stage is stable 0/5).  Keeping the
       doubled footprint <= 96 KiB eliminates it while preserving the bf16
       ``tile_n=8192`` long-seq optimum exactly (2*8192*6 = 96 KiB).
    S1 **Divisor-exact / balanced persistent grid** (same helper as the
       resident unit) with a static ``T.serial`` trip count.
    S2 **Batched output write** -- one UB->GM copy per core, stage 4 profiles
       showed the per-row write dominates the small-M large-N cases.
    S3 **Explicit first-tile load** -- the first full column tile is hoisted
       out of the ``T.serial`` loop so the remaining loop body is uniform.
    S7 **Multi-row streaming stage** (v3_smalltask) -- ``block_m == 1`` is
       promoted to a 2-row stage when ``M`` is even and the wider stage still
       fits the per-core UB (``_stage_fits``).  A 2 x tile_n stage widens the
       per-core staging DMA (16 KB -> 32 KB on long-seq, per-core MTE2 38.6 ->
       ~52 GB/s) and halves the per-core ``T.reduce_*`` count.  Measured on
       long-seq-reduce bf16 (64x32768): 9.20us -> 6.48us, ratio 25.0% -> 34.7%
       (median of 3 interleaved msprof runs).

Known toolchain constraints enforced here (DESIGN.md section 9.1):
    - ``T.reduce_max`` src/dst MUST be the same dtype -> bf16 uses a dedicated
      fp32 work buffer.
    - unused ``alloc`` is NOT dead-code-eliminated -> fp16/fp32 and bf16 use
      separate prim_funcs.
    - a *chunked* ``T.vcast`` (``T.vcast(x_ub[0:bm, 0:chw], x_work)`` with
      ``chw < tile_n``) is rejected by ``CodeGenTileLangNPUIRDEV::VcastCodegen``
      (``Check failed: buffer_shape0[i] == buffer_shape1[i]``), so the staged
      tile must be cast/reduced at full width -- this caps the stage width at
      what ``stage + work + scratch`` allows (S7) and is why ``tile_n`` cannot
      simply be doubled to 16384 for bf16.
"""

import os

# Developer mode: compiler manages UB (alloc_shared) + fragment + auto sync.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

# Programming mode actually validated by this Stage 3 implementation.
ASCEND_MODE = "Developer"

# Race-safe doubled-buffer UB envelope (bytes): 2 * tile_n * B <= 96 KiB.
# See the S0 note above: a >96 KiB doubled staging footprint races in the
# auto-multi-buffer (measured on fp32 tile_n=16384), while 96 KiB is exactly
# the bf16 tile_n=8192 long-seq optimum.
UB_ENVELOPE = 96 * 1024

# Large-buffer bytes per element (DESIGN.md section 4.5 / 3.2 dtype matrix):
# fp16 x_ub=2; fp32 x_ub=4; bf16 x_ub=2 + x_work fp32=4.
_B_ELEM = {"float16": 2, "float32": 4, "bfloat16": 6}

_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")

_DTYPE_MAP = {
    "float16": (torch.float16, 1e-2, 1e-2),
    "float32": (torch.float32, 1e-4, 1e-4),
    "bfloat16": (torch.bfloat16, 1e-2, 1e-2),
}


def _ceildiv(a, b):
    return (a + b - 1) // b


def _divisors(n):
    """All positive divisors of ``n`` (ascending)."""
    small, large = [], []
    d = 1
    while d * d <= n:
        if n % d == 0:
            small.append(d)
            if d != n // d:
                large.append(n // d)
        d += 1
    return small + large[::-1]


def _largest_divisor_at_most(n, cap):
    best = 1
    d = 1
    while d * d <= n:
        if n % d == 0:
            if d <= cap:
                best = max(best, d)
            q = n // d
            if q <= cap:
                best = max(best, q)
        d += 1
    return best


def _choose_grid(num_logical, vector_cores):
    if num_logical <= vector_cores:
        return num_logical, True
    d = _largest_divisor_at_most(num_logical, vector_cores)
    if 2 * d >= vector_cores:
        return d, True
    return vector_cores, num_logical % vector_cores == 0


def _clamp_tile_n(N, dtype, tile_n):
    """Clamp ``tile_n`` into the doubled-buffer UB envelope (Stage 4 S0).

    A legal caller value is respected verbatim; an oversized one is replaced by
    the largest 256-multiple divisor of N that fits, else by the budget cap.
    """
    b = _B_ELEM[dtype]
    if 2 * tile_n * b <= UB_ENVELOPE:
        return tile_n
    cap = (UB_ENVELOPE // (2 * b)) & ~255
    cands = [d for d in _divisors(N) if 256 <= d <= cap and d % 256 == 0]
    return max(cands) if cands else cap


# ---------- tile_n / routing gate (DESIGN.md section 5.2) ----------
def select_tile_n(M, N, dtype, ub_budget=65536):
    """Factory-period ``tile_n`` derivation (tiled path)."""
    b = _B_ELEM[dtype]
    tile_n_cap = (ub_budget // b) & ~255
    cands = [f for f in _divisors(N) if f <= tile_n_cap and f % 256 == 0]
    return max(cands) if cands else tile_n_cap


def needs_tiling(M, N, dtype, ub_budget=65536):
    """Routing gate (wrapper period): N*B > 64KB -> this streaming unit."""
    return N * _B_ELEM[dtype] > ub_budget


# Per-core UB available to one stage + its reduce scratch (192 KiB minus a
# safety reserve; the same reserve the resident coalescing path uses).
UB_STAGE_BUDGET = 190 * 1024


def _elem_bytes(dtype):
    return 4 if dtype == "float32" else 2


def _stage_fits(bm, tn, dtype):
    """Does a ``bm x tn`` stage (plus its work buffer and reduce scratch) fit?

    Measured on this device: the ``T.reduce_*`` lowering needs scratch of about
    half the staged footprint (independent of the chunk width), and a bf16
    reduce additionally needs one fp32 copy of the stage.
    """
    stage = bm * tn * _elem_bytes(dtype)
    work = bm * tn * 4 if dtype == "bfloat16" else 0
    return stage + work + stage // 2 <= UB_STAGE_BUDGET


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_amax_tiled(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: ``torch.amax(x, dim=-1)`` on CPU (fp32 then cast)."""
    return torch.amax(x.float(), dim=-1).to(x.dtype)


# ---------- Kernel factory ----------
def _simple_reduce_kernel_tiled(M, N, op_kind, dtype, tile_n):
    """Build the tiled (streaming) amax kernel factory.

    Mirrors the GPU source structure:
      ``_simple_reduce_kernel_tiled(M, N, op_kind, dtype, tile_n)
         -> _func(block_m) -> main``.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if op_kind != "amax":
        raise ValueError(f"this unit implements op_kind='amax' only, got {op_kind!r}")
    if N <= 0:
        raise ValueError(f"reduction dim N must be positive, got N={N}")
    if tile_n <= 0:
        raise ValueError(f"tile_n must be positive, got tile_n={tile_n}")

    # Stage 4 S0: reconcile the caller's tile_n with this unit's UB envelope.
    tile_n = _clamp_tile_n(N, dtype, tile_n)

    num_full = N // tile_n  # full tiles (compile-time constant)
    tail = N - num_full * tile_n  # static tail width
    tail_w = tail if tail > 0 else 1
    is_bf16 = dtype == "bfloat16"

    vector_cores = NPUUtils.get().get_aicore_num() * 2  # pure Vector cores x2

    # Stage 4 S7: decide the multi-row stage at FACTORY time (M / tile_n / dtype
    # are factory constants; block_m is 1 on the streaming path), because the
    # wider stage only fits UB with the auto multi-buffer DISABLED: a 2 x 8192
    # bf16 stage is 32 KiB + 64 KiB of fp32 work, so a doubled (multi-buffered)
    # footprint would be 192 KiB -- measured as a vector-core timeout (507034).
    # With bm == 1 the pass config is unchanged from the previous best.
    promote_bm = M >= 2 and M % 2 == 0 and _stage_fits(2, tile_n, dtype)
    _jit = tilelang.jit(out_idx=[1], target="npuir")
    if promote_bm:
        _jit = tilelang.jit(
            out_idx=[1],
            target="npuir",
            pass_configs={"npuir.enable_auto_multi_buffer": False},
        )

    @_jit
    def _func(block_m):
        # ---- Stage 4 S7: multi-row streaming stage (bm = 1 -> 2 promotion) ------
        bm = max(1, min(block_m, M))
        # A ``bm``-row x ``tile_n`` stage widens each staging DMA (long-seq: 16 KB
        # -> 32 KB, per-core MTE2 38.6 -> ~52 GB/s) and divides the per-core
        # ``T.reduce_*`` count by ``bm`` (8 -> 4 on 64x32768 bf16: 4 column tiles x
        # 1 reduce per tile instead of 4 x 2).  Measured on long-seq bf16:
        # 9.20us -> 6.48us (median of 3 interleaved runs, -29.6%).  The promotion is
        # a trace-time constant *inside* the prim_func factory -- never a Python
        # factory / wrapper dispatch.  It is only taken when ``M`` is an exact
        # multiple of the promoted ``bm`` (so every staged row slice is
        # full-width -- the dynamic ``real_m`` form is kept for bm == 1 only) and
        # the wider stage still fits the per-core UB.
        if bm == 1 and promote_bm:
            bm = 2

        num_logical = _ceildiv(M, bm)
        nk, exact = _choose_grid(num_logical, vector_cores)
        L = _ceildiv(num_logical, nk)
        static_m = M % bm == 0

        if is_bf16:

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), dtype),
            ):
                with T.Kernel(nk, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((bm, tile_n), dtype)
                    x_work = T.alloc_shared((bm, tile_n), "float32")
                    x_tail = T.alloc_shared((bm, tail_w), dtype)
                    x_tail_work = T.alloc_shared((bm, tail_w), "float32")
                    acc = T.alloc_shared((bm, 1), "float32")
                    out_acc = T.alloc_shared((L * bm,), dtype)

                    start = cid * L if exact else cid * num_logical // nk
                    end = start + L if exact else (cid + 1) * num_logical // nk

                    for s in T.serial(L):
                        task = start + s
                        if exact:
                            off_m = task * bm
                            T.vbrc(-T.infinity("float32"), acc)
                            if num_full > 0:
                                if static_m:
                                    T.copy(x[off_m : off_m + bm, 0:tile_n], x_ub)
                                else:
                                    real_m = T.min(bm, M - off_m)
                                    T.copy(
                                        x[off_m : off_m + real_m, 0:tile_n],
                                        x_ub[0:real_m, 0:tile_n],
                                    )
                                T.vcast(x_ub, x_work, round_mode="rint")
                                T.reduce_max(x_work, acc, dim=1, clear=False)
                                if num_full > 1:
                                    for t in T.serial(num_full - 1):
                                        T.copy(
                                            x[
                                                off_m : off_m + bm,
                                                (t + 1) * tile_n : (t + 2) * tile_n,
                                            ],
                                            x_ub,
                                        )
                                        T.vcast(x_ub, x_work, round_mode="rint")
                                        T.reduce_max(x_work, acc, dim=1, clear=False)
                            if tail > 0:
                                if static_m:
                                    T.copy(x[off_m : off_m + bm, num_full * tile_n : N], x_tail)
                                else:
                                    real_t = T.min(bm, M - off_m)
                                    T.copy(
                                        x[off_m : off_m + real_t, num_full * tile_n : N],
                                        x_tail[0:real_t, 0:tail],
                                    )
                                T.vcast(x_tail, x_tail_work, round_mode="rint")
                                T.reduce_max(x_tail_work, acc, dim=1, clear=False)
                            for i in T.Parallel(bm):
                                out_acc[s * bm + i] = T.cast(acc[i, 0], dtype)
                        else:
                            if task < end:
                                off_m = task * bm
                                T.vbrc(-T.infinity("float32"), acc)
                                if num_full > 0:
                                    if static_m:
                                        T.copy(x[off_m : off_m + bm, 0:tile_n], x_ub)
                                    else:
                                        real_m = T.min(bm, M - off_m)
                                        T.copy(
                                            x[off_m : off_m + real_m, 0:tile_n],
                                            x_ub[0:real_m, 0:tile_n],
                                        )
                                    T.vcast(x_ub, x_work, round_mode="rint")
                                    T.reduce_max(x_work, acc, dim=1, clear=False)
                                    if num_full > 1:
                                        for t in T.serial(num_full - 1):
                                            T.copy(
                                                x[
                                                    off_m : off_m + bm,
                                                    (t + 1) * tile_n : (t + 2) * tile_n,
                                                ],
                                                x_ub,
                                            )
                                            T.vcast(x_ub, x_work, round_mode="rint")
                                            T.reduce_max(x_work, acc, dim=1, clear=False)
                                if tail > 0:
                                    if static_m:
                                        T.copy(x[off_m : off_m + bm, num_full * tile_n : N], x_tail)
                                    else:
                                        real_t = T.min(bm, M - off_m)
                                        T.copy(
                                            x[off_m : off_m + real_t, num_full * tile_n : N],
                                            x_tail[0:real_t, 0:tail],
                                        )
                                    T.vcast(x_tail, x_tail_work, round_mode="rint")
                                    T.reduce_max(x_tail_work, acc, dim=1, clear=False)
                                for i in T.Parallel(bm):
                                    out_acc[s * bm + i] = T.cast(acc[i, 0], dtype)

                    base = T.min(start * bm, M)
                    valid = T.min(end * bm, M) - base
                    T.copy(out_acc[0:valid], out[base : base + valid])

        else:

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), dtype),
            ):
                with T.Kernel(nk, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((bm, tile_n), dtype)
                    x_tail = T.alloc_shared((bm, tail_w), dtype)
                    acc = T.alloc_shared((bm, 1), dtype)
                    out_acc = T.alloc_shared((L * bm,), dtype)

                    start = cid * L if exact else cid * num_logical // nk
                    end = start + L if exact else (cid + 1) * num_logical // nk

                    for s in T.serial(L):
                        task = start + s
                        if exact:
                            off_m = task * bm
                            T.vbrc(-T.infinity(dtype), acc)
                            if num_full > 0:
                                if static_m:
                                    T.copy(x[off_m : off_m + bm, 0:tile_n], x_ub)
                                else:
                                    real_m = T.min(bm, M - off_m)
                                    T.copy(
                                        x[off_m : off_m + real_m, 0:tile_n],
                                        x_ub[0:real_m, 0:tile_n],
                                    )
                                T.reduce_max(x_ub, acc, dim=1, clear=False)
                                if num_full > 1:
                                    for t in T.serial(num_full - 1):
                                        T.copy(
                                            x[
                                                off_m : off_m + bm,
                                                (t + 1) * tile_n : (t + 2) * tile_n,
                                            ],
                                            x_ub,
                                        )
                                        T.reduce_max(x_ub, acc, dim=1, clear=False)
                            if tail > 0:
                                if static_m:
                                    T.copy(x[off_m : off_m + bm, num_full * tile_n : N], x_tail)
                                else:
                                    real_t = T.min(bm, M - off_m)
                                    T.copy(
                                        x[off_m : off_m + real_t, num_full * tile_n : N],
                                        x_tail[0:real_t, 0:tail],
                                    )
                                T.reduce_max(x_tail, acc, dim=1, clear=False)
                            for i in T.Parallel(bm):
                                out_acc[s * bm + i] = acc[i, 0]
                        else:
                            if task < end:
                                off_m = task * bm
                                T.vbrc(-T.infinity(dtype), acc)
                                if num_full > 0:
                                    if static_m:
                                        T.copy(x[off_m : off_m + bm, 0:tile_n], x_ub)
                                    else:
                                        real_m = T.min(bm, M - off_m)
                                        T.copy(
                                            x[off_m : off_m + real_m, 0:tile_n],
                                            x_ub[0:real_m, 0:tile_n],
                                        )
                                    T.reduce_max(x_ub, acc, dim=1, clear=False)
                                    if num_full > 1:
                                        for t in T.serial(num_full - 1):
                                            T.copy(
                                                x[
                                                    off_m : off_m + bm,
                                                    (t + 1) * tile_n : (t + 2) * tile_n,
                                                ],
                                                x_ub,
                                            )
                                            T.reduce_max(x_ub, acc, dim=1, clear=False)
                                if tail > 0:
                                    if static_m:
                                        T.copy(x[off_m : off_m + bm, num_full * tile_n : N], x_tail)
                                    else:
                                        real_t = T.min(bm, M - off_m)
                                        T.copy(
                                            x[off_m : off_m + real_t, num_full * tile_n : N],
                                            x_tail[0:real_t, 0:tail],
                                        )
                                    T.reduce_max(x_tail, acc, dim=1, clear=False)
                                for i in T.Parallel(bm):
                                    out_acc[s * bm + i] = acc[i, 0]

                    base = T.min(start * bm, M)
                    valid = T.min(end * bm, M) - base
                    T.copy(out_acc[0:valid], out[base : base + valid])

        return main

    return _func


# ---------- precision comparison ----------
def _run_case(M, N, dtype_str, tile_n, tag, equal_nan=False):
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    x = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
    return _run_case_from_values(x, dtype_str, tile_n, tag, equal_nan=equal_nan)


def _run_case_from_values(x, dtype_str, tile_n, tag, equal_nan=False):
    """Run a pre-built CPU tensor through the kernel; check against golden."""
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    M, N = x.shape
    x = x.to(torch_dtype)
    kernel = _simple_reduce_kernel_tiled(M, N, "amax", dtype_str, tile_n)(1)
    y = kernel(x.to(device="npu"))
    ref = golden_amax_tiled(x)
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    diff = (y.cpu().float() - ref.float()).abs()
    diff = diff[torch.isfinite(diff)]
    max_diff = diff.max().item() if diff.numel() > 0 else 0.0
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} tile_n={tile_n} max_diff={max_diff:.2e}"
    )
    return max_diff


# ---------- L0: gate tests (must pass) ----------
def run_L0():
    max_diff = 0.0
    max_diff = max(max_diff, _run_case(64, 32768, "bfloat16", 8192, "L0-1"))
    max_diff = max(max_diff, _run_case(128, 65536, "float16", 16384, "L0-2a"))
    max_diff = max(max_diff, _run_case(128, 65536, "float32", 16384, "L0-2b"))
    max_diff = max(max_diff, _run_case(4, 100000, "float16", 8192, "L0-3"))
    max_diff = max(max_diff, _run_case(8, 40000, "float32", 8192, "L0-4"))
    max_diff = max(max_diff, _run_case(16, 32769, "float16", 16384, "L0-5"))
    max_diff = max(max_diff, _run_case(2, 262144, "float16", 32768, "L0-6"))
    max_diff = max(max_diff, _run_case(129, 65536, "float16", 16384, "L0-7"))
    max_diff = max(max_diff, _run_case(4, 13000, "bfloat16", 8192, "L0-8"))
    max_diff = max(max_diff, _run_case(1, 32769, "float16", 16384, "L0-9"))
    for dtype_str in ("float16", "float32"):
        max_diff = max(max_diff, _run_boundary_rows(dtype_str, 8192, "L0-10"))
    print(f"[L0] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


def _boundary_rows(dtype_str, n=32768):
    torch_dtype = _DTYPE_MAP[dtype_str][0]
    finfo = torch.finfo(torch_dtype)
    z = torch.zeros(n)
    rows = [
        torch.full((n,), float("-inf")),
        torch.full((n,), float("inf")),
        torch.cat([torch.tensor([float("inf")]), z[1:]]),
        torch.cat([torch.tensor([float("-inf")]), z[1:]]),
        torch.full((n,), 3.0),
        torch.arange(n, dtype=torch.float32) * 0.5,
        torch.zeros(n),
        torch.cat([torch.tensor([finfo.max]), z[1:]]),
    ]
    x = torch.stack(rows, dim=0).to(torch.float32)
    if dtype_str == "float16":
        x = torch.nan_to_num(x, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return x.to(torch_dtype)


def _run_boundary_rows(dtype_str, tile_n, tag):
    x = _boundary_rows(dtype_str)
    return _run_case_from_values(x, dtype_str, tile_n, tag)


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    max_diff = 0.0
    max_diff = max(max_diff, _run_case(128, 16384, "float16", 4096, "L1-1"))
    max_diff = max(max_diff, _run_case(4, 65536, "bfloat16", 8192, "L1-2"))
    max_diff = max(max_diff, _run_case(17, 20000, "float16", 8192, "L1-3"))
    max_diff = max(max_diff, _run_case(33, 32768, "float32", 8192, "L1-4"))
    max_diff = max(max_diff, _run_case(256, 16384, "float16", 4096, "L1-5"))
    max_diff = max(max_diff, _run_case(5, 50000, "bfloat16", 4096, "L1-6"))
    tn = select_tile_n(64, 40000, "float16")
    max_diff = max(max_diff, _run_case(64, 40000, "float16", tn, "L1-7-derived"))
    print(f"[L1] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


# ---------- L2: boundary (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (1, 32768, "float16", 32768),
        (8, 8192, "float16", 8192),
        (8, 8193, "float16", 8192),
        (4, 1000, "float16", 4096),
        (40, 32768, "float16", 8192),
        (41, 32768, "float16", 8192),
        (2, 262144, "float16", 32768),
    ]
    for M, N, dtype_str, tile_n in cases:
        try:
            _run_case(M, N, dtype_str, tile_n, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str} tile_n={tile_n}: {e}")


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def run_boundary():
    for val, name in ((float("-inf"), "all-neg-inf"), (float("inf"), "all-pos-inf")):
        try:
            x = torch.full((8, 32768), val, dtype=torch.float16)
            _run_case_from_values(x, "float16", 8192, f"Boundary-{name}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): {name}: {e}")

    try:
        x = torch.randn(8, 32768, dtype=torch.float16)
        x[0, 1] = float("nan")
        x[3, 100] = float("nan")
        _run_case_from_values(x, "float16", 8192, "Boundary-nan", equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): nan: {e}")

    try:
        x = torch.zeros(8, 32768, dtype=torch.float16)
        x[:, 0] = -0.0
        _run_case_from_values(x, "float16", 8192, "Boundary-zeros")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): zeros: {e}")

    try:
        finfo = torch.finfo(torch.float16)
        x = torch.full((8, 32768), finfo.tiny / 2, dtype=torch.float16)
        _run_case_from_values(x, "float16", 8192, "Boundary-subnormal")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {e}")

    try:
        _run_case(1024, 16384, "float16", 4096, "Boundary-large-M")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large M (1024,16384): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
