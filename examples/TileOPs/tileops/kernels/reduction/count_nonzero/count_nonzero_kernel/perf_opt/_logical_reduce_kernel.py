# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""``_logical_reduce_kernel`` -- count_nonzero forward kernel (NPU Developer mode).

Migrated from the GPU TileOPs ``LogicalReduceKernel`` baseline factory
(``tileops/kernels/reduction/logical_reduce.py``, ``_logical_reduce_kernel``).
The GPU factory is shared by the any / all / count_nonzero kinds; this
migration unit implements the **count_nonzero** kind only (``op_kind`` is kept
as a factory parameter to preserve the harness contract) and returns an
``(M,)`` **int64** count per row -- it must NOT follow the sibling kinds'
bool/int8 output contract.  The GPU N-tiled fallback
(``_logical_reduce_kernel_tiled``) is intentionally not migrated.

NPU algorithm (DESIGN.md sections 0.6 / 1.6.1):

    y[i] = N - sum_{j < N} [ x[i, j] == 0.0 ]

i.e. the **zero-count complement** with an IEEE **ordered EQ** judgement
(``T.vcmp(..., "eq")``).  ``eq`` is IEEE-correct, so NaN is never counted as
zero and therefore lands in the non-zero count, matching both
``torch.count_nonzero`` and the GPU source's unordered-NE semantics.
``"ne"`` is ordered-NE on this toolchain (NaN always false) and is forbidden
here.  The complement is applied on the tiny ``(block_m,)`` micro-buffer, so
no full-width ``1 - indicator`` pass is needed.

Chain: ``T.vcmp`` -> ``T.vcast(i1 -> f32)`` -> ``T.reduce_sum(dim=1)`` ->
``T.vsub(N, .)`` -> ``T.vcast(f32 -> i64)``.

Two capacity paths (factory-period dispatch, one dedicated ``prim_func`` per
variant -- a Python-level ``if`` that *defines buffers* inside a ``prim_func``
is not parseable by this toolchain: tail > 0 raises ``Undefined variable`` and
tail == 0 segfaults the compiler):

  * path R (resident, ``N <= E_CAP``): one tile per row block.
  * path C (column-tiled, ``N > E_CAP``): ``bm = 1``, ``num_full`` static-width
    column tiles with a no-alias running accumulator, plus an optional static
    width tail block (separate buffer group, generated at factory level).

Element budget: the indicator chain costs a measured 8.00 B/element of UB
(DESIGN.md section 4.5 probe), i.e. a hard limit of 24576 elements/block; the
race-safe design budget is ``E_CAP = 8192`` elements/block.

No host-side padding: the kernel declares the raw ``(M, N)`` window, matching
the Op layer contract ``_kernel_handles_padding = True``.  Counting is neutral
to zero padding, so the real-column domain needs no alignment.

Stage 4 performance tuning (perf_opt, measured with ``msprof op`` Task
Duration on Ascend910B3/B4-class hardware, 40 AIV):

  * Path R' (DESIGN.md section 1.6.4 alternative) is **adopted** for
    ``E_CAP < N <= UB element cap``: one tile per row (``block_m = 1``) with
    no tile loop, instead of column tiling.  Measured on the two manifest
    workloads that hit this range: ``sparsity-seq (32,16384)`` 4.41 -> 3.89 us
    and ``3d-multidim (128,16384)`` 10.23 -> 8.96 us (-11.8% / -12.5%).
    The drop of auto-multi-buffer did not produce any race (exact int64
    equality holds over repeated 30-launch runs).
  * The Stage 3 output structure (one tiny store per task) is kept: batched
    output staging measured neutral on the R' workloads and -10% on
    ``sparsity-hidden`` bfloat16 (extra UB copy on an already vector-bound
    pipe), so it was rolled back.
  * **Round 5: fp16 resident tier 2 -> 4 rows** (``_FP16_RESIDENT_ELEMS``).
    Round 4 had deferred this because the then-current toolchain priced the
    whole 4-row group at 12.03 B/element (197152 B > 196608 B UB).  On the
    round-5 toolchain the 4-row group compiles and the 8-row group is rejected
    at 331936 B (10.13 B/element), so 16384 elements is the fp16 ceiling.
    ``sparsity-hidden`` float16: 22.674 -> 16.335 us (-27.96%, interleaved
    A/B/A/B 3 pairs, 30 launches each, direction consistent) -- 13 tasks/core
    with one 32 KB contiguous GM->UB transfer per task instead of 26 x 16 KB.
    bf16 keeps the measured 3-row tier: its 4-row group is still rejected at
    196800 B, and the bisected (2-row compute group) variant is blocked by a
    codegen shape check in ``VcastCodegen`` for the bf16 dtype lift.
  * ``sparsity-hidden`` fp16 now runs at 16.79 MB / 16.34 us = 1.03 TB/s of
    effective read bandwidth, i.e. above the round-4 device-reported 0.8 TB/s
    GM roofline; the round-4 read-floor claim (23.8-25.3 us) is superseded by
    the round-5 measurement on this toolchain.

Run: ``python _logical_reduce_kernel.py --level all``
"""

import os

# Programming mode of this implementation (kept in sync with the standalone
# test entry point and with the TileOPs ``Kernel.ascend_mode`` used in Stage 5).
ASCEND_MODE = "Developer"
os.environ.setdefault("TILELANG_ASCEND_MODE", ASCEND_MODE)

import argparse  # noqa: E402

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402  (registers the "npu" device)
from tilelang.tvm import tir  # noqa: E402  (tilelang vendors tvm as tilelang.tvm)

from tileops.kernels._factory_cache import kernel_factory_cache  # noqa: E402

__all__ = ["ASCEND_MODE", "_logical_reduce_kernel", "_select_config", "golden_count_nonzero"]

# ---------------------------------------------------------------------------
# Factory-period constants (DESIGN.md sections 4.5 / 5.2)
# ---------------------------------------------------------------------------
E_CAP = 8192  # elements/block: race-safe budget at the measured 8.00 B/element
UB_HARD_ELEMS = 24576  # 192 KB / 8 B (measured compiler-side unit price)
# Stage 4: UB capacity is priced per element and per dtype by the reduction
# chain (CONST-reduce-ub-per-element): 8 B/elem for fp16/fp32, 12 B/elem for
# bf16 (vcmp operands are lifted to f32).  8 KB is kept as reserve, mirroring
# the design-period UB_SAFETY_RESERVE_BYTES.
UB_RESERVE_BYTES = 8192
_UB_PER_ELEM = {"float16": 8, "float32": 8, "bfloat16": 12}
# Stage 4 round 4: the resident row-block budget is dtype-priced.  Measured
# compiler UB requirement for this 2-D row-reduce chain is 12.01 B/element at
# W=4096 (16384 elems -> 196800 B, overflow by 192 B; 12288 elems compiles),
# identical for fp16 and bf16 -- i.e. the constraint at W=4096 is ~16.3 K
# elements, not the 8 B/element linear extrapolation.  The bf16 chain is
# vector-bound (vec_time 19.6 us vs 12.3 us for fp16 at identical traffic), so
# bf16 is allowed the 3-row tier (12288 elements) to cut the task count from
# 1024 to 683 logical blocks (18 vs 26 tasks per core).
_BF16_RESIDENT_ELEMS = 12288
# Stage 4 round 5 -- fp16 resident tier raised to 16384 elements (4 rows at
# W=4096).  Round-4 deferred this because the then-current toolchain priced the
# single-group chain at 12.03 B/element (4 rows = 197152 B > 196608 B).  On the
# round-5 toolchain (tilelang 0.1.2+84c6019, commit 06e1a1a) the same 4-row
# group COMPILES and the 8-row group is rejected at 331936 B = 10.13 B/element,
# i.e. the UB ceiling moved to ~19.4 K elements for fp16.  16384 is the largest
# ladder value below it.  bf16 is NOT raised: the 4-row group is still rejected
# at 196800 B (12.012 B/element), so bf16 keeps the measured 3-row tier.
_FP16_RESIDENT_ELEMS = 16384
MAX_EXACT_N = 2**24  # fp32 exact-integer domain: the count must stay exact
DEFAULT_ALIGNMENT = 256
LAUNCH_GATE = 80  # > 2 waves of 40 vector cores -> engage the extended ladder
_DEFAULT_VECTOR_CORES = 40  # queried on the device (DESIGN.md section 5.5)
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")
_UNSUPPORTED_STORAGE_DTYPES = (
    torch.bool,
    torch.complex64,
    torch.complex128,
    torch.int32,
    torch.int64,
)


# ---------------------------------------------------------------------------
# Factory-period helpers
# ---------------------------------------------------------------------------
def _ceildiv(a, b):
    return (a + b - 1) // b


def _vector_cores():
    """Physical Vector core count, queried on the device (DESIGN.md 5.5)."""
    try:
        from tilelang.utils.npu_arch import AscendArch

        return int(AscendArch().aivector_core_num)
    except Exception:  # noqa: BLE001  (fall back to AICore x 2)
        from tilelang.utils.npu_utils import NPUUtils

        return int(NPUUtils.get().get_aicore_num()) * 2


def _ub_elems(dtype):
    """Per-block element cap for a resident tile (dtype-aware UB pricing)."""
    per_elem = _UB_PER_ELEM[dtype]
    return min(UB_HARD_ELEMS, (192 * 1024 - UB_RESERVE_BYTES) // per_elem)


def _pick_tile_n(N, cap):
    """Column-tile width: largest 256-multiple <= cap that divides N.

    Falls back to the largest 256-multiple <= cap when no divisor exists, in
    which case the caller emits the static-width tail block (DESIGN.md 5.2).
    """
    limit = (cap // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    if limit <= 0:
        return max(1, min(N, cap))
    for candidate in range(limit, 0, -DEFAULT_ALIGNMENT):
        if N % candidate == 0:
            return candidate
    return limit


def _select_config(M, N, dtype):
    """Element-budget ladder + path switch (DESIGN.md sections 3.3 / 5.2).

    Three paths:

    * ``R``  -- ``N <= E_CAP``: whole row resident, ladder ``block_m``.
    * ``R'`` -- ``E_CAP < N <= _ub_elems(dtype)``: single-tile resident
      (``block_m = 1``), Stage 4 measured winner over column tiling.
    * ``C``  -- ``N > _ub_elems(dtype)``: column tiles + no-alias accumulator.

    Returns ``{"path", "block_m", "tile_n", "num_full", "tail"}`` with every
    value a factory-period Python int (static loop bounds).
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}. "
            f"bool / int32 / int64 / complex inputs must be pre-converted to float32 "
            f"by the Op layer (to_logical_float32)."
        )
    if M <= 0:
        raise ValueError(f"M must be positive, got M={M}")
    if N < 1:
        raise ValueError(f"reduction width N must be >= 1, got N={N}")
    if N > MAX_EXACT_N:
        raise ValueError(
            f"N={N} exceeds 2**24: the fp32 count domain is no longer exact "
            f"(DESIGN.md sections 1.6.1-E2 / 9.2-R7)"
        )

    if N <= E_CAP:
        # Path R: whole row resident in UB.
        path = "R"
        # bf16 only: allow the 3-row tier (see _BF16_RESIDENT_ELEMS).
        row_ladder = (1, 2, 3, 4, 8) if dtype == "bfloat16" else (1, 2, 4, 8)
        # Round 5: fp16 gets the measured 4-row tier (_FP16_RESIDENT_ELEMS);
        # bf16 keeps 12288; float32 keeps the race-safe E_CAP (never probed).
        if dtype == "bfloat16":
            elem_cap = _BF16_RESIDENT_ELEMS
        elif dtype == "float16":
            elem_cap = _FP16_RESIDENT_ELEMS
        else:
            elem_cap = E_CAP
        candidates = [p for p in row_ladder if p * N <= elem_cap]
        block_m = max(candidates) if candidates else 1
        if _ceildiv(M, block_m) > LAUNCH_GATE:
            extended = [
                p for p in (16, 32, 64, 128, 256, 512, 1024) if p * N <= elem_cap and p <= M
            ]
            block_m = max(extended + [block_m])
        tile_n, num_full, tail = N, 1, 0
    elif _ub_elems(dtype) >= N:
        # Path R' (DESIGN.md section 1.6.4): the row exceeds the race-safe
        # element budget but still fits UB as ONE tile, so the tile loop and
        # its micro-accumulate ops are dropped (block_m = 1, single buffer;
        # auto-multi-buffer is necessarily dropped as well).
        path = "R"
        block_m = 1
        tile_n, num_full, tail = N, 1, 0
    else:
        # Path C: column tiles + no-alias running accumulator (+ optional tail).
        path = "C"
        block_m = 1
        tile_n = _pick_tile_n(N, min(E_CAP, N))
        num_full = N // tile_n
        tail = N - num_full * tile_n

    block_m = min(block_m, M)  # never allocate more rows than exist
    return {
        "path": path,
        "block_m": block_m,
        "tile_n": tile_n,
        "num_full": num_full,
        "tail": tail,
    }


def _resolve_block_m(block_m, default_bm, tile_n, M):
    """Resolve the caller-supplied ``block_m`` against the element budget.

    ``default_bm`` is the ladder value from ``_select_config``.  A caller (the
    TileOPs wrapper's ``default_config``) may still carry the GPU-period
    SMEM heuristic until Stage 5 rewires it (DESIGN.md section 5.6); a value
    that overshoots ``E_CAP`` elements per tile would overflow UB, so it falls
    back to the ladder value.  ``block_m`` only selects the row split, so this
    can never change results.
    """
    if block_m is None:
        return default_bm
    block_m = min(block_m, M)
    if block_m * tile_n > E_CAP:
        return default_bm
    return block_m


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference -- independent of the NPU algorithm)
# ---------------------------------------------------------------------------
def golden_count_nonzero(x, dim=-1):
    """PyTorch reference: ``torch.count_nonzero(x, dim)`` as int64.

    Runs on CPU and does NOT reproduce the NPU algorithm (no complement, no
    indicator plane, no column tiling), guaranteeing validation independence
    (DESIGN.md section 8.1).  ``dim=None`` performs a full reduction, matching
    the Op layer's ``_empty_dim_policy = "full"``.
    """
    return torch.count_nonzero(x, dim=dim).to(torch.int64)


# ---------------------------------------------------------------------------
# Path R -- resident, whole row in UB (DESIGN.md sections 3.3 / 6.2)
# ---------------------------------------------------------------------------
def _build_resident(M, N, dtype, default_bm):
    """Single-tile-per-row-block kernel: the whole row (``N`` columns) is resident."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        block_m = _resolve_block_m(block_m, default_bm, N, M)
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, _vector_cores())
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                zero_ub = T.alloc_shared((block_m, N), dtype)
                mask_eq = T.alloc_shared((block_m, N), "bool")
                ind = T.alloc_shared((block_m, N), "float32")
                zc = T.alloc_fragment((block_m, 1), "float32")
                cnt = T.alloc_fragment((block_m,), "float32")
                n_v = T.alloc_fragment((block_m,), "float32")
                out_ub = T.alloc_shared((block_m,), "int64")

                # Task-invariant operands, hoisted out of the task loop.
                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(n_v, tir.const(N, "float32"))

                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:  # dynamic guard (static loop bound above)
                        real_m = T.min(block_m, M - start)  # M tail row count
                        T.copy(
                            x[start : start + real_m, 0:N],
                            x_ub[0:real_m, 0:N],
                        )
                        T.vcmp(x_ub, zero_ub, mask_eq, "eq")  # IEEE ordered EQ
                        T.vcast(mask_eq, ind, round_mode="rint")  # i1 -> f32
                        T.reduce_sum(ind, zc, dim=1, clear=True)
                        T.reshape(zc, cnt)  # (bm, 1) -> (bm,) bridge
                        T.vsub(n_v, cnt, cnt)  # complement, (bm,) micro-buffer
                        T.vcast(cnt, out_ub, round_mode="rint")  # f32 -> i64
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Path C -- column-tiled with no-alias running accumulator (DESIGN.md 3.3/6.2)
# ---------------------------------------------------------------------------
def _build_tiled(M, N, tile_n, num_full, dtype, default_bm):
    """Column-tiled kernel without a tail block (``tail == 0``)."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        block_m = _resolve_block_m(block_m, default_bm, tile_n, M)
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, _vector_cores())
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, tile_n), dtype)
                zero_ub = T.alloc_shared((block_m, tile_n), dtype)
                mask_eq = T.alloc_shared((block_m, tile_n), "bool")
                ind = T.alloc_shared((block_m, tile_n), "float32")
                zc = T.alloc_fragment((block_m, 1), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                acc_flat = T.alloc_fragment((block_m,), "float32")
                cnt = T.alloc_fragment((block_m,), "float32")
                n_v = T.alloc_fragment((block_m,), "float32")
                out_ub = T.alloc_shared((block_m,), "int64")

                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(n_v, tir.const(N, "float32"))

                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.fill(acc, tir.const(0, "float32"))
                        for t in T.serial(num_full):  # static bound
                            T.copy(
                                x[
                                    start : start + real_m,
                                    t * tile_n : (t + 1) * tile_n,
                                ],
                                x_ub[0:real_m, 0:tile_n],
                            )
                            T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                            T.vcast(mask_eq, ind, round_mode="rint")
                            T.reduce_sum(ind, zc, dim=1, clear=True)
                            # No-alias accumulate (never in-place on a
                            # loop-carried value).
                            T.vadd(acc, zc, new_acc)
                            T.copy(new_acc, acc)
                        T.reshape(acc, acc_flat)
                        T.vsub(n_v, acc_flat, cnt)  # minuend = full row width N
                        T.vcast(cnt, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func


def _build_tiled_tail(M, N, tile_n, num_full, tail, dtype, default_bm):
    """Column-tiled kernel with a static-width tail block (``tail > 0``).

    The tail uses its own buffer group at the static width ``tail`` (a
    factory-period constant), so no runtime condition or masked load is
    involved (DESIGN.md section 5.4).
    """

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        block_m = _resolve_block_m(block_m, default_bm, tile_n, M)
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, _vector_cores())
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, tile_n), dtype)
                zero_ub = T.alloc_shared((block_m, tile_n), dtype)
                mask_eq = T.alloc_shared((block_m, tile_n), "bool")
                ind = T.alloc_shared((block_m, tile_n), "float32")
                x_tail = T.alloc_shared((block_m, tail), dtype)
                zero_tail = T.alloc_shared((block_m, tail), dtype)
                mask_tail = T.alloc_shared((block_m, tail), "bool")
                ind_tail = T.alloc_shared((block_m, tail), "float32")
                zc = T.alloc_fragment((block_m, 1), "float32")
                zc_tail = T.alloc_fragment((block_m, 1), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc_t = T.alloc_fragment((block_m, 1), "float32")
                acc_flat = T.alloc_fragment((block_m,), "float32")
                cnt = T.alloc_fragment((block_m,), "float32")
                n_v = T.alloc_fragment((block_m,), "float32")
                out_ub = T.alloc_shared((block_m,), "int64")

                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(zero_tail, tir.const(0, dtype))
                T.fill(n_v, tir.const(N, "float32"))

                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.fill(acc, tir.const(0, "float32"))
                        for t in T.serial(num_full):
                            T.copy(
                                x[
                                    start : start + real_m,
                                    t * tile_n : (t + 1) * tile_n,
                                ],
                                x_ub[0:real_m, 0:tile_n],
                            )
                            T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                            T.vcast(mask_eq, ind, round_mode="rint")
                            T.reduce_sum(ind, zc, dim=1, clear=True)
                            T.vadd(acc, zc, new_acc)
                            T.copy(new_acc, acc)
                        T.copy(
                            x[
                                start : start + real_m,
                                num_full * tile_n : num_full * tile_n + tail,
                            ],
                            x_tail[0:real_m, 0:tail],
                        )
                        T.vcmp(x_tail, zero_tail, mask_tail, "eq")
                        T.vcast(mask_tail, ind_tail, round_mode="rint")
                        T.reduce_sum(ind_tail, zc_tail, dim=1, clear=True)
                        T.vadd(acc, zc_tail, new_acc_t)
                        T.copy(new_acc_t, acc)
                        T.reshape(acc, acc_flat)
                        T.vsub(n_v, acc_flat, cnt)
                        T.vcast(cnt, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Factory (harness contract: _logical_reduce_kernel(M, N, op_kind, dtype)
#                                                 -> _func(block_m) -> main)
# ---------------------------------------------------------------------------
@kernel_factory_cache
def _logical_reduce_kernel(M, N, op_kind, dtype):
    """Build the count_nonzero NPU kernel factory.

    ``op_kind`` is kept for the shared GPU factory contract; only
    ``"count_nonzero"`` is implemented by this migration unit (the sibling
    any/all kinds are separate units with a bool/int8 output contract).
    ``N`` is the raw reduction width -- no alignment padding is applied.
    ``threads`` does not exist on NPU (harness K9).
    """
    if op_kind != "count_nonzero":
        raise ValueError(
            f"unsupported op_kind {op_kind!r}: this migration unit implements "
            f"the 'count_nonzero' kind only (any/all are separate units)"
        )

    cfg = _select_config(M, N, dtype)

    if cfg["path"] == "R":
        _f = _build_resident(M, N, dtype, cfg["block_m"])
        _f.msprof_kernel_name = "count_nonzero_resident"
    elif cfg["tail"] == 0:
        _f = _build_tiled(M, N, cfg["tile_n"], cfg["num_full"], dtype, cfg["block_m"])
        _f.msprof_kernel_name = "count_nonzero_tiled"
    else:
        _f = _build_tiled_tail(
            M, N, cfg["tile_n"], cfg["num_full"], cfg["tail"], dtype, cfg["block_m"]
        )
        _f.msprof_kernel_name = "count_nonzero_tiled_tail"
    _f.design_config = cfg
    return _f


# ---------------------------------------------------------------------------
# Op-layer emulation helpers (host-side flow of CountNonzeroFwdOp)
# ---------------------------------------------------------------------------
def _to_logical_float32(x):
    """Mirror the wrapper's ``to_logical_float32`` (Op-layer pre-conversion).

    bool/int32/int64 -> direct float32 cast; complex -> 1.0 where
    ``real != 0 or imag != 0`` (nonzero-ness preserved in every case).
    """
    if x.dtype in (torch.bool, torch.int32, torch.int64):
        return x.to(torch.float32)
    return ((x.real != 0) | (x.imag != 0)).to(torch.float32)


def _normalize_dims(dim, ndim):
    """Op-layer dim normalisation; ``None`` / empty means a full reduction."""
    if dim is None:
        return []
    if isinstance(dim, int):
        dim = [dim]
    return sorted({d % ndim for d in dim})


def _prepare_2d(x, dim):
    """Emulate the Op layer: movedim reduced dims last, contiguous, reshape."""
    dims = _normalize_dims(dim, x.dim())
    keep = [d for d in range(x.dim()) if d not in dims]
    xd = x.permute(keep + dims).contiguous() if dims else x.reshape(-1).contiguous()
    M = 1
    for d in keep:
        M *= x.shape[d]
    N = xd.numel() // M
    return xd.reshape(M, N), tuple(x.shape[d] for d in keep)


def _declared_width(kernel):
    """Declared input width of a compiled kernel (prim_func buffer map)."""
    for buf in kernel.prim_func.buffer_map.values():
        if buf.dtype != "int64":
            return int(buf.shape[-1])
    return None


# ---------------------------------------------------------------------------
# Precision comparison helpers
# ---------------------------------------------------------------------------
_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _cfg_str(cfg):
    return (
        f"path={cfg['path']} block_m={cfg['block_m']} tile_n={cfg['tile_n']} "
        f"num_full={cfg['num_full']} tail={cfg['tail']}"
    )


def _exact_compare(y, ref, tag):
    """Exact int64 comparison (atol = rtol = 0, DESIGN.md section 8.2)."""
    if y.dtype != torch.int64:
        raise AssertionError(f"[{tag}] output dtype {y.dtype} != int64")
    yc = y.cpu()
    if tuple(yc.shape) != tuple(ref.shape):
        raise AssertionError(f"[{tag}] shape {tuple(yc.shape)} != ref {tuple(ref.shape)}")
    if not torch.equal(yc, ref):
        bad = (yc != ref).nonzero().flatten()
        n_bad = int(bad.numel())
        i = int(bad[0]) if n_bad else -1
        raise AssertionError(
            f"[{tag}] int64 count mismatch: {n_bad}/{ref.numel()} entries differ; "
            f"first at {i}: got={int(yc.flatten()[i])} want={int(ref.flatten()[i])}"
        )


def _make_input(M, N, dtype_str, seed=0, zero_frac=0.3):
    """Mixed-density input with forced zeros / +-0 / non-zero corners.

    ``randn`` alone would give an all-nonzero row (count == N), which cannot
    distinguish a correct counter from a constant.  ~30% of the entries are
    zeroed, and row 0 / row 1 get the harness-style forced corners.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(M, N, generator=gen)
    keep = (torch.rand(M, N, generator=gen) > zero_frac).to(torch.float32)
    x = x * keep
    if M > 0 and N > 1:
        x[0, 0] = 0.0
        x[0, 1] = 1.0
    if M > 1 and N > 2:
        x[1, 2] = -0.0
    return x.to(_DTYPE_MAP[dtype_str]).to("npu")


def _run_tensor(x, dtype_str, tag, expect_path=None, expect_block_m=None, expect_tile_n=None):
    """Run one 2D ``(M, N)`` NPU tensor case (raw width, no host padding)."""
    M, N = x.shape
    cfg = _select_config(M, N, dtype_str)
    if expect_path is not None and cfg["path"] != expect_path:
        raise AssertionError(f"[{tag}] path={cfg['path']} != expected {expect_path}")
    if expect_block_m is not None and cfg["block_m"] != expect_block_m:
        raise AssertionError(
            f"[{tag}] block_m={cfg['block_m']} != expected {expect_block_m} "
            f"(shape=({M},{N}) dtype={dtype_str})"
        )
    if expect_tile_n is not None and cfg["tile_n"] != expect_tile_n:
        raise AssertionError(
            f"[{tag}] tile_n={cfg['tile_n']} != expected {expect_tile_n} "
            f"(shape=({M},{N}) dtype={dtype_str})"
        )
    kernel = _logical_reduce_kernel(M, N, "count_nonzero", dtype_str)(cfg["block_m"])
    y = kernel(x)
    _exact_compare(y, golden_count_nonzero(x.cpu()), tag)
    return cfg, y


def _run_case(M, N, dtype_str, tag, **kw):
    return _run_tensor(_make_input(M, N, dtype_str), dtype_str, tag, **kw)


def _run_nd(x, dim, dtype_str, tag, **kw):
    """Run through the emulated Op-layer flow (movedim -> reshape -> back)."""
    x2, out_shape = _prepare_2d(x, dim)
    cfg, y = _run_tensor(x2, dtype_str, tag, **kw)
    ref = golden_count_nonzero(x.cpu(), dim=dim)
    _exact_compare(y.reshape(out_shape), ref, f"{tag}-nd")
    return cfg


# ---------------------------------------------------------------------------
# L0: gate tests (blocking) -- DESIGN.md section 8.2
# ---------------------------------------------------------------------------
def _run_constructed_rows(dtype_str, tag):
    """L0-10: rows exercising the complement's two extremes and its corners."""
    N = 512
    rows = []
    rows.append(torch.zeros(N))  # y = 0
    rows.append(torch.ones(N))  # y = N
    r = torch.zeros(N)
    r[0] = 1.0
    r[N - 1] = -1.0
    rows.append(r)  # first/last single non-zero -> y = 2
    r = torch.zeros(N)
    r[N // 2] = -0.0
    rows.append(r)  # mixed-sign zero only -> y = 0
    r = torch.ones(N)
    r[3] = 0.0
    rows.append(r)  # y = N - 1
    r = torch.zeros(N)
    r[0] = float("inf")
    rows.append(r)  # +inf counts -> y = 1
    r = torch.zeros(N)
    r[N - 1] = -float("inf")
    rows.append(r)  # -inf counts -> y = 1
    r = torch.zeros(N)
    r[1] = 1e-8
    rows.append(r)  # tiny (subnormal in fp16) counts -> y = 1
    x = torch.stack(rows).to(_DTYPE_MAP[dtype_str]).to("npu")
    cfg, _ = _run_tensor(x, dtype_str, tag)
    print(
        f"[{tag}] PASS: constructed rows M={x.shape[0]} N={N} dtype={dtype_str} ({_cfg_str(cfg)})"
    )


def _run_nan_rows(dtype_str, tag):
    """L0-12: the E1 equivalence gate -- NaN must count as non-zero."""
    M, N = 4, 512
    x = torch.zeros(M, N)
    x[0, 0] = float("nan")
    x[0, 1] = 1.0  # NaN + one -> y = 2
    x[1, :] = float("nan")  # all NaN -> y = N
    x[2, 0] = float("nan")
    x[2, 3] = 0.0  # NaN + zeros -> y = 1
    x[3, 7] = -0.0  # -0 only -> y = 0
    x = x.to(_DTYPE_MAP[dtype_str]).to("npu")
    cfg, y = _run_tensor(x, dtype_str, tag)
    print(
        f"[{tag}] PASS: NaN rows dtype={dtype_str} counts={y.cpu().tolist()} "
        f"({_cfg_str(cfg)}) -- vcmp('eq') is IEEE ordered EQ"
    )


def run_L0():
    n = 0
    # L0-1 smoke: three comparison-domain dtype paths.
    for dtype_str in ("float16", "bfloat16", "float32"):
        _run_case(32, 256, dtype_str, "L0-1", expect_path="R", expect_block_m=8)
        n += 1
    # L0-2 sparsity-hidden: persistent 40 cores x multi-wave race check.
    # Round 4 op2: the bf16 tier is 3 rows (683 logical blocks -> 18 tasks/core).
    # Round 5: fp16 resident tier 2 -> 4 rows (_FP16_RESIDENT_ELEMS = 16384).
    _run_case(2048, 4096, "float16", "L0-2", expect_path="R", expect_block_m=4)
    _run_case(2048, 4096, "bfloat16", "L0-2", expect_path="R", expect_block_m=3)
    n += 2
    # L0-3 / L0-4 path R' (single-tile resident over the race-safe budget):
    # the Stage 4 winner for 8192 < N <= UB element cap (no tile loop).
    _run_case(32, 16384, "float16", "L0-3", expect_path="R", expect_block_m=1)
    _run_case(128, 16384, "float16", "L0-4", expect_path="R", expect_block_m=1)
    n += 2
    # L0-5 raw non-aligned width (no padding anywhere).
    for dtype_str in ("float16", "float32"):
        _run_case(128, 300, dtype_str, "L0-5", expect_path="R", expect_block_m=8)
        n += 1
    # L0-6 M tail block (129 % 8 == 1) -- source-defect fix regression.
    _run_case(129, 512, "float16", "L0-6", expect_block_m=8)
    n += 1
    # L0-7 E_cap boundary configuration (block_m * N == 8192).
    # Round 5: 4 rows x 4096 = 16384 elements is the new fp16 resident cap.
    _run_case(256, 4096, "float16", "L0-7", expect_block_m=4)
    n += 1
    # L0-8 narrow raw widths + extended ladder (launch-item driven).
    _run_case(32768, 4, "float16", "L0-8a", expect_block_m=1024)
    # Round 5: the extended ladder follows the fp16 cap (16384 // 64 = 256).
    _run_case(2048, 64, "float16", "L0-8b", expect_block_m=256)
    n += 2
    # L0-9 single-row loads (1D flatten and explicit single row).
    x1 = _make_input(1, 512, "float16")
    _run_nd(x1.reshape(512), -1, "float16", "L0-9a", expect_block_m=1)
    _run_case(1, 256, "float16", "L0-9b", expect_block_m=1)
    n += 2
    # L0-10 / L0-12 constructed and NaN rows.
    _run_constructed_rows("float16", "L0-10a")
    _run_constructed_rows("float32", "L0-10b")
    n += 2
    for dtype_str in ("float16", "bfloat16", "float32"):
        _run_nan_rows(dtype_str, f"L0-12-{dtype_str}")
        n += 1
    # L0-11 integration width contract: the kernel must declare the raw width.
    M, N = 8, 300
    kernel = _logical_reduce_kernel(M, N, "count_nonzero", "float16")(8)
    width = _declared_width(kernel)
    if width != N:
        raise AssertionError(
            f"[L0-11] declared width {width} != raw N {N} "
            f"(host padding must not be applied: _kernel_handles_padding=True)"
        )
    x = _make_input(M, N, "float16")
    y = kernel(x)
    _exact_compare(y, golden_count_nonzero(x.cpu()), "L0-11")
    n += 1

    print(f"[L0] ALL PASS: {n} shape/edge cases")


# ---------------------------------------------------------------------------
# L1: functional coverage (blocking)
# ---------------------------------------------------------------------------
N_UB_FP16 = _ub_elems("float16")


def run_L1():
    n = 0
    # L1-1 basic + non-contiguous input.
    for dtype_str in ("float16", "bfloat16", "float32"):
        _run_case(128, 512, dtype_str, "L1-1")
        n += 1
    x_full = _make_input(128, 1024, "float16")
    x_nc = x_full[:, :512]
    x_c = x_nc.contiguous()
    cfg, y = _run_tensor(x_c, "float16", "L1-1-noncontig")
    _exact_compare(y, golden_count_nonzero(x_nc.cpu(), dim=-1), "L1-1-noncontig")
    n += 1

    # L1-2 / L1-3 higher-rank and non-last-axis dims (Op layer movedim+reshape).
    _run_nd(_make_input(128, 512, "float16").reshape(2, 64, 512), -1, "float16", "L1-2a")
    _run_nd(_make_input(64, 512, "float16").reshape(2, 4, 8, 512), -1, "float16", "L1-2b")
    _run_nd(_make_input(512, 64, "float16"), 0, "float16", "L1-3a")
    _run_nd(_make_input(64, 512, "float16").reshape(4, 16, 512), 0, "float16", "L1-3b")
    _run_nd(_make_input(2048, 64, "float16").reshape(4, 64, 512), 1, "float16", "L1-3c")
    _run_nd(_make_input(2048, 64, "float32").reshape(4, 64, 512), -2, "float32", "L1-3d")
    n += 6

    # L1-4 extreme / subnormal values (the E4 corner set re-checked on NPU).
    # Extremes are exact powers of two / dtype limits so the cast is lossless.
    for dtype_str, sub in (
        ("float16", 2.0**-24),  # smallest fp16 subnormal
        ("bfloat16", 2.0**-133),  # smallest bf16 subnormal
        ("float32", 2.0**-149),  # smallest fp32 subnormal
    ):
        big = float(torch.finfo(_DTYPE_MAP[dtype_str]).max)
        N = 256
        rows = [torch.zeros(N), torch.full((N,), big), torch.full((N,), -big)]
        r = torch.zeros(N)
        r[0] = sub
        r[N - 1] = -sub
        rows.append(r)
        r = torch.full((N,), sub)
        r[5] = 0.0
        rows.append(r)
        r = torch.zeros(N)
        r[0] = float("inf")
        r[1] = -float("inf")
        rows.append(r)
        x = torch.stack(rows).to(_DTYPE_MAP[dtype_str]).to("npu")
        cfg, y = _run_tensor(x, dtype_str, f"L1-4-{dtype_str}")
        print(f"[L1-4-{dtype_str}] PASS: extreme rows counts={y.cpu().tolist()}")
        n += 1

    # L1-5 unsupported-storage dtypes via the Op-layer float32 pre-conversion.
    for dtype in (torch.bool, torch.int32, torch.int64, torch.complex64, torch.complex128):
        M, N = 64, 512
        if dtype.is_complex:
            base = torch.randn(M, N)
            imag = torch.randn(M, N) * (torch.rand(M, N) > 0.3)
            x_cpu = torch.complex(base * (torch.rand(M, N) > 0.3), imag).to(dtype)
        elif dtype is torch.bool:
            x_cpu = torch.rand(M, N) > 0.4
        else:
            x_cpu = (torch.randn(M, N) * (torch.rand(M, N) > 0.3) * 100).to(dtype)
        ref = torch.count_nonzero(x_cpu, dim=-1).to(torch.int64)
        xf = _to_logical_float32(x_cpu).contiguous().to("npu")
        kernel = _logical_reduce_kernel(M, N, "count_nonzero", "float32")(8)
        y = kernel(xf)
        _exact_compare(y, ref, f"L1-5-{dtype}")
        n += 1

    # L1-6 M tail + non-aligned N together.
    _run_case(8, 300, "float16", "L1-6")
    n += 1

    # L1-7 path C details (N above the UB element cap): non-zero static tail
    # block and the divisor-preference branch of the tile-width rule
    # (DESIGN.md section 5.2).  25000/20000 have no 256-multiple divisor <= 8192
    # -> max width + static tail; 24576 = 3 * 8192 is exactly the UB cap
    # boundary and prefers the divisible width (zero tail).
    cfg, _ = _run_case(64, 25000, "float16", "L1-7a", expect_path="C", expect_tile_n=8192)
    if (cfg["num_full"], cfg["tail"]) != (3, 424):
        raise AssertionError(
            f"[L1-7a] (num_full, tail)={(cfg['num_full'], cfg['tail'])} != (3, 424)"
        )
    cfg, _ = _run_case(8, 24576, "float32", "L1-7b", expect_path="C", expect_tile_n=8192)
    if (cfg["num_full"], cfg["tail"]) != (3, 0):
        raise AssertionError(f"[L1-7b] (num_full, tail)={(cfg['num_full'], cfg['tail'])} != (3, 0)")
    cfg, _ = _run_case(8, 24000, "float16", "L1-7c", expect_path="C", expect_tile_n=8192)
    if (cfg["num_full"], cfg["tail"]) != (2, 7616):
        raise AssertionError(
            f"[L1-7c] (num_full, tail)={(cfg['num_full'], cfg['tail'])} != (2, 7616)"
        )
    # L1-7d: R' boundary -- the last width that still fits UB in one tile.
    cfg, _ = _run_case(8, 23552, "float16", "L1-7d", expect_path="R", expect_block_m=1)
    if cfg["tile_n"] != N_UB_FP16:
        raise AssertionError(f"[L1-7d] tile_n={cfg['tile_n']} != {N_UB_FP16}")
    n += 4

    # L1-8 caller-supplied block_m robustness.  The GPU-period wrapper
    # heuristic may still pass a block_m that overshoots the element budget
    # (e.g. 8 x 4096 = 32768 elements > the 24576 hard UB limit) until Stage 5
    # rewires ``default_config`` per DESIGN.md section 5.6: the kernel must fall
    # back to the ladder value rather than overflow UB, and a valid smaller
    # value must be honoured.
    M, N = 256, 4096
    x = _make_input(M, N, "float16")
    for bm_in in (1, 8):
        kernel = _logical_reduce_kernel(M, N, "count_nonzero", "float16")(bm_in)
        y = kernel(x)
        _exact_compare(y, golden_count_nonzero(x.cpu()), f"L1-8-bm{bm_in}")
    print(f"[L1-8] PASS: caller block_m 1 (honoured) and 8 (clamped) both exact on ({M},{N})")
    n += 1

    print(f"[L1] ALL PASS: {n} cases")


# ---------------------------------------------------------------------------
# L2: boundary / contract behaviour (record only, non-blocking)
# ---------------------------------------------------------------------------
def run_L2():
    # Factory-period raises.
    for label, fn in (
        ("M=0", lambda: _select_config(0, 256, "float16")),
        ("N=0", lambda: _select_config(1, 0, "float16")),
        ("N=2**24+1", lambda: _select_config(1, 2**24 + 1, "float16")),
        ("op_kind='any'", lambda: _logical_reduce_kernel(8, 256, "any", "float16")),
        ("dtype='int8'", lambda: _select_config(8, 256, "int8")),
    ):
        try:
            fn()
            print(f"[L2] WARN (record only): {label} did not raise as designed")
        except ValueError as exc:
            print(f"[L2] PASS: {label} raises ValueError ({str(exc)[:70]}...)")
        except Exception as exc:  # noqa: BLE001
            print(f"[L2] WARN (record only): {label} raised {type(exc).__name__}: {exc}")

    # Minimal shapes.
    for M, N in ((1, 1), (2, 1), (1, 2), (4, 2), (3, 1)):
        try:
            _run_case(M, N, "float16", f"L2-min-{M}x{N}")
            print(f"[L2] PASS: minimal shape ({M},{N})")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): minimal shape ({M},{N}): {e}")

    # Path R <-> path C consistency: the same data, with the extra columns of
    # the path-C shape forced to zero (counts are zero-neutral), must give
    # bit-identical counts.  N = 8192 stays resident; N = 16384 is column-tiled.
    try:
        M, N_half, N_full = 4, 8192, 16384
        data = _make_input(M, N_half, "float16")
        cfg_r, y_r = _run_tensor(data, "float16", "L2-R")
        x_full = torch.zeros(M, N_full, dtype=torch.float16, device="npu")
        x_full[:, :N_half] = data
        cfg_c, y_c = _run_tensor(x_full, "float16", "L2-C")
        if not torch.equal(y_r.cpu(), y_c.cpu()):
            raise AssertionError(f"path R {y_r.cpu().tolist()} != path C {y_c.cpu().tolist()}")
        print(
            f"[L2] PASS: resident R ({_cfg_str(cfg_r)}) == single-tile R' "
            f"({_cfg_str(cfg_c)}) "
            f"on the same effective data (extra columns zero-filled)"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): path R/C consistency: {e}")

    # N just above E_cap -> path C with the smallest possible tail.
    try:
        cfg, _ = _run_case(32, 8193, "float16", "L2-tn+1")
        print(f"[L2] PASS: N=E_cap+1 -> {_cfg_str(cfg)}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): N=E_cap+1: {e}")

    # M=0 would need a zero-sized grid; the factory raises (Op layer should
    # short-circuit instead -- DESIGN.md section 9.2-R6).
    print("[L2] NOTE: M=0 is a factory raise (Op-layer short-circuit recommended)")


# ---------------------------------------------------------------------------
# Boundary: extreme data values (record only, non-blocking)
# ---------------------------------------------------------------------------
def run_boundary():
    for dtype_str in ("float16", "bfloat16", "float32"):
        try:
            x = torch.zeros(32, 256, dtype=_DTYPE_MAP[dtype_str], device="npu")
            _run_tensor(x, dtype_str, f"Boundary-zeros-{dtype_str}")
            print(f"[Boundary] PASS: all zeros (32,256) {dtype_str} -> 0")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): all zeros {dtype_str}: {e}")
        try:
            x = torch.ones(32, 256, dtype=_DTYPE_MAP[dtype_str], device="npu")
            cfg, y = _run_tensor(x, dtype_str, f"Boundary-ones-{dtype_str}")
            if int(y.cpu().min()) != 256:
                raise AssertionError(f"all-ones count {int(y.cpu().min())} != 256")
            print(f"[Boundary] PASS: all ones (32,256) {dtype_str} -> 256")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): all ones {dtype_str}: {e}")

    # 1D loads across dtypes.
    for dtype_str in ("float16", "bfloat16", "float32"):
        try:
            x = _make_input(1, 512, dtype_str).reshape(512)
            _run_nd(x, -1, dtype_str, f"Boundary-1d-{dtype_str}")
            print(f"[Boundary] PASS: 1D (512,) {dtype_str}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): 1D {dtype_str}: {e}")

    # Larger random shape.
    try:
        _run_case(512, 2048, "float16", "Boundary-large")
        print("[Boundary] PASS: (512,2048) float16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): (512,2048): {e}")


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
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
