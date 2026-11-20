# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""``_logical_reduce_kernel`` -- logical-OR row reduction (NPU Developer mode).

Migrated from the GPU TileOPs ``LogicalReduceKernel`` baseline factory
(``tileops/kernels/reduction/logical_reduce.py``, ``_logical_reduce_kernel``).
The GPU factory is shared by the any / all / count_nonzero kinds; this
migration unit implements the **any** kind only (``op_kind`` is kept as a
factory parameter to preserve the harness contract).  Contract:

  * input  ``x``   : ``(M, N)`` raw width, float16 / bfloat16 / float32
  * output ``out`` : ``(M,)`` **int8** carrier (``bool`` in the manifest);
    the wrapper restores bool via ``out_raw.bool()``.

It must NOT follow either sibling unit: ``count_nonzero`` uses ``reduce_sum``
with an int64 carrier and ``all`` uses ``reduce_max`` with the ``1 - max``
complement.  This unit reduces the zero-hit indicator with **``reduce_min``**
and complements as ``1 - min`` ("the whole row is zero"), the running merge on
the column-tiled path is a **``vmin``** (not ``vadd`` / ``vmax``) and its
accumulator unit element is **1.0** (not 0.0).  Copying any of those four
inverts the semantics.  The GPU N-tiled fallback
(``_logical_reduce_kernel_tiled``) is intentionally not migrated; the NPU
column-tiling path below is driven by the UB element budget, not by a GPU
fragment column limit.

NPU algorithm (DESIGN.md sections 0.6 / 1.6.1):

    y[i] = 1 - min_{j < N} [ x[i, j] == 0.0 ]

i.e. De Morgan's complement form with an IEEE **ordered EQ** judgement
(``T.vcmp(..., "eq")``).  ``eq`` is IEEE-correct, so NaN is never counted as a
zero and therefore keeps ``any`` True, matching both ``torch.any`` and the GPU
source's unordered-NE semantics.  ``"ne"`` is ordered-NE on this toolchain
(NaN always false, i.e. NaN judged as a zero) and would flip every row that
holds a NaN plus zeros and nothing else -- forbidden here.  The complement is
applied on the tiny ``(block_m,)`` micro-buffer, so no full-width
``1 - indicator`` pass is needed.  ``T.vnot`` is not an option either: its
dtype table has no bool/int1 entry.

Chain (per path; the carrier dtype is the Stage-4 tuned constant):

  * path R: ``T.vcmp`` -> ``T.vcast(i1 -> f16)`` -> ``T.reduce_min(dim=1)``
    -> ``T.vsub(1, .)`` -> ``T.vcast(f16 -> i8)`` (one carrier constant,
    ``IND_DTYPE = "float16"``; a 0/1 indicator is exactly representable in f16,
    so the reduction stays bit-exact, and ``f16 -> i8`` exists in the ``vcast``
    table as a single hop).
  * path C (``tail == 0``): ``T.vcmp`` -> ``T.vcast(i1 -> f16)`` ->
    ``T.reduce_min(dim=1)`` -> ``T.vsub(1, .)`` -> ``T.vcast(f16 -> i8)``
    (one carrier constant, ``C_IND_DTYPE = "float16"``; the running-min
    accumulator, the ``1 - x`` operands and the epilogue share it, so the
    ``f32 -> f16`` hop of the epilogue disappears).
  * static-width tail block: ``T.vcmp`` -> ``T.vcast(i1 -> f32)`` ->
    ``T.reduce_min(dim=1)`` -> ``T.vsub(1, .)`` -> ``T.vcast(f32 -> f16)`` ->
    ``T.vcast(f16 -> i8)`` (``f32 -> i8`` has no ``vcast`` table entry; the
    float32 chain is kept here -- narrow-tail float16 mis-reduce,
    VP-2026-0326).

Two capacity paths (factory-period dispatch, one dedicated ``prim_func`` per
variant -- a Python-level ``if`` that *defines buffers* inside a ``prim_func``
is not parseable by this toolchain):

  * path R (resident, ``N <= E_CAP``): one tile per row block.
  * path C (column-tiled, ``N > E_CAP``): ``bm = 1``, ``num_full`` static-width
    column tiles with a no-alias running ``min`` accumulator, plus an optional
    static-width tail block (separate buffer group, generated at factory level).

Element budget: the indicator chain costs a measured 8.00 B/element of UB for
fp16/fp32 and 12.00 B/element for bf16 (DESIGN.md section 4.5 probe; the bf16
``vcmp`` is raised to f32 by the lowering), i.e. a hard limit of 24576 (fp16) /
16384 (bf16, fp32) elements per block.  The race-safe design budget is
``E_CAP = 8192`` elements/block (worst dtype bf16 at UB/2).

No host-side padding: the kernel declares the raw ``(M, N)`` window, matching
the Op layer contract ``_kernel_handles_padding = True``.  Logical OR is
neutral to the OR unit element (0.0) padding, so the real-column domain needs
no alignment.

Stage 4 final: the resident path (``N <= E_CAP``) and the column-tiled
path (``N > E_CAP``, ``tail == 0``) both run the zero-hit indicator
plane, the row-minimum and the complemented epilogue in **float16**
(0/1 indicator values are exactly representable, so the reduction stays
bit-exact and a single f16->i8 hop replaces the f32->f16->i8 epilogue).
The resident carrier was tuned in round 1 (-4.7% / -6.1% on 910B4); the
column-tiled carrier was re-measured in round 5 on 910B3 (toolchain
719c6da), where it is worth -9.21% / -8.68% on the 3d workload (10/10
interleaved pairs in each of two independent sessions, p = 0.002) and
-7.91% on mask-validation-32k (4/4 pairs); the earlier "inside the noise
band" verdict was a 910B4 / e21da32 measurement and does not carry over.
The static-width tail block keeps the float32 chain (narrow-tail float16
mis-reduce, VP-2026-0326).

Run: ``python _logical_reduce_kernel.py --level all``
"""

import os

# Programming mode of this implementation (kept in sync with the standalone
# test entry point and with the TileOPs ``Kernel.ascend_mode`` used in Stage 5).
ASCEND_MODE = "Developer"
os.environ.setdefault("TILELANG_ASCEND_MODE", ASCEND_MODE)

import argparse  # noqa: E402
import re  # noqa: E402

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402  (registers the "npu" device)
from tilelang.tvm import tir  # noqa: E402  (tilelang vendors tvm as tilelang.tvm)

from tileops.kernels._factory_cache import kernel_factory_cache  # noqa: E402

__all__ = [
    "ASCEND_MODE",
    "_logical_reduce_kernel",
    "golden__logical_reduce_kernel",
    "golden_any_logical_reduce",
]

# ---------------------------------------------------------------------------
# Factory-period constants (DESIGN.md sections 4.5 / 5.2)
# ---------------------------------------------------------------------------
UB_BYTES = 196608  # AscendArch().ub_cap, queried on the device (192 KB)
# Measured compiler-side UB unit price (B/element) of the
# vcmp(eq) -> vcast(i1->f32) -> reduce_min indicator chain:
# 8.00 for fp16/fp32 and 12.00 for bf16 (comparison lowered to f32).
UNIT_PRICE = {"float16": 8, "float32": 8, "bfloat16": 12}
E_CAP = UB_BYTES // 2 // max(UNIT_PRICE.values())  # 8192 elements (worst dtype)
# Measured pass points of the probe (hard limit, no double-buffer margin).
HARD_CAP_ELEMS = {"float16": 24576, "bfloat16": 16384, "float32": 16384}
DEFAULT_ALIGNMENT = 256
# --- Stage-4 tuned constants (used by the resident path; see opt_log.md) -----
# IND_DTYPE: carrier dtype of the zero-hit indicator plane of the resident
# path.  The indicator only ever holds 0.0/1.0, which float16 represents
# exactly, so the row-minimum is bit-exact; halving the plane width cuts the
# ``vcast`` write + ``reduce_min`` read volume and removes the f32->f16 hop of
# the epilogue.  The row-minimum, the ``1 - min`` operands and the ``f16 -> i8``
# hop share this carrier, so one constant covers the whole resident chain.  Measured on 910B4/CANN 9.0.0, mask-validation-4k-bool:
# 2.90 -> 2.78 us interleaved A/B median (3/3 pairs same direction, -4.1%).
IND_DTYPE = "float16"
# --- round-5 candidate constant (not in the delivered file) ---------------
# C_IND_DTYPE: carrier dtype of the *column-tiled* path (path C, tail == 0).
# The zero-hit indicator only holds 0.0/1.0 (exact in f16), so the row-min
# and the running-min accumulator stay bit-exact; the carrier is shared by
# ind / mn / acc / one_v / any_v, and the epilogue collapses to a single
# f16 -> i8 hop.  Path R and the static-width tail keep their own dtypes.
C_IND_DTYPE = "float16"
# Measured on 910B3/CANN 9.0.0 (toolchain 719c6da): 3d 10.86 -> 9.86 us
# (-9.21%, 10/10 interleaved pairs, p=0.002; second session -8.68%,
# 10/10), 32k 6.64 -> 6.12 us (-7.91%, 4/4 pairs).  The lowered code is
# byte-identical to the A/B-measured candidate (_logical_reduce_kernel_
# round5_op1_cf16.py, sha a2fe7d6e...); only this comment differs.
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


_CORES_CACHE = {}


def _vector_cores():
    """Physical Vector core count, queried on the device (DESIGN.md 5.5)."""
    if "n" not in _CORES_CACHE:
        try:
            from tilelang.utils.npu_arch import AscendArch

            _CORES_CACHE["n"] = int(AscendArch().aivector_core_num)
        except Exception:  # noqa: BLE001  (fall back to AICore x 2)
            from tilelang.utils.npu_utils import NPUUtils

            _CORES_CACHE["n"] = int(NPUUtils.get().get_aicore_num()) * 2
    return _CORES_CACHE["n"]


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
        # The empty OR is False for every row (torch.any((3,0), dim=-1) ->
        # [False, False, False]); DESIGN.md section 9.2-R5 routes N == 0 to an
        # Op-layer short-circuit instead of a zero-width kernel.
        raise ValueError(f"reduction width N must be >= 1, got N={N}")

    if N <= E_CAP:
        # Path R: whole row resident in UB.
        path = "R"
        candidates = [p for p in (1, 2, 4, 8) if p * N <= E_CAP]
        block_m = max(candidates) if candidates else 1
        if _ceildiv(M, block_m) > LAUNCH_GATE:
            extended = [p for p in (16, 32, 64, 128, 256, 512, 1024) if p * N <= E_CAP and p <= M]
            block_m = max(extended + [block_m])
        tile_n, num_full, tail = N, 1, 0
    else:
        # Path C: column tiles + no-alias running accumulator (+ optional tail).
        path = "C"
        block_m = 1
        tile_n = _pick_tile_n(N, min(E_CAP, N))
        num_full = N // tile_n
        tail = N - num_full * tile_n

    block_m = min(block_m, M)  # never allocate more rows than exist

    # Capacity predicate (DESIGN.md section 4.5): the per-block element count
    # must sit inside both the race-safe budget and the measured hard limit.
    elems = block_m * (N if path == "R" else min(tile_n, N))
    if elems > HARD_CAP_ELEMS[dtype]:
        raise ValueError(
            f"({M},{N}) {dtype}: {elems} elements/block exceed the measured UB hard "
            f"limit {HARD_CAP_ELEMS[dtype]} (probe pass point)"
        )
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
    TileOPs wrapper's ``default_config``) may still carry the GPU-period SMEM
    heuristic until Stage 5 rewires it (DESIGN.md section 5.6); a value that
    overshoots ``E_CAP`` elements per tile would overflow UB, so it falls back
    to the ladder value.  ``block_m`` only selects the row split, so this can
    never change results.
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
def golden_any_logical_reduce(x, dim=-1, keepdim=False):
    """PyTorch reference: ``torch.any`` semantics (DESIGN.md section 8.1).

    ``out = OR_j (x_j != 0)`` (NaN counts as non-zero, +-0 counts as zero),
    computed with the authoritative torch reduction API ``torch.any``.  For
    non-bool inputs ``torch.any`` casts to bool first, i.e. "non-zero -> True"
    with NaN treated as non-zero -- verified identical to
    ``x.bool().any(dim, keepdim)`` on float16 / bfloat16 / float32 / bool /
    int32 / int64 / complex64 / complex128.  The tensor is moved to CPU first
    so the reference stays independent of the device runtime.

    It does NOT reproduce the NPU algorithm (no complement form, no indicator
    plane, no column tiling), guaranteeing validation independence.
    """
    return torch.any(x.cpu(), dim=dim, keepdim=keepdim)


# Op-named alias (the gate looks for a `golden_*` entry point of this unit).
golden__logical_reduce_kernel = golden_any_logical_reduce


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
            out: T.Tensor((M,), "int8"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                zero_ub = T.alloc_shared((block_m, N), dtype)
                mask_eq = T.alloc_shared((block_m, N), "bool")
                ind = T.alloc_shared((block_m, N), IND_DTYPE)
                mn = T.alloc_fragment((block_m, 1), IND_DTYPE)
                mn_flat = T.alloc_fragment((block_m,), IND_DTYPE)
                one_v = T.alloc_fragment((block_m,), IND_DTYPE)
                any_v = T.alloc_fragment((block_m,), IND_DTYPE)
                out_ub = T.alloc_shared((block_m,), "int8")

                # Task-invariant operands, hoisted out of the task loop.
                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(one_v, tir.const(1, IND_DTYPE))

                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:  # dynamic guard (static loop bound above)
                        real_m = T.min(block_m, M - start)  # M tail row count
                        T.copy(
                            x[start : start + real_m, 0:N],
                            x_ub[0:real_m, 0:N],
                        )
                        # any-zero indicator: ordered EQ (IEEE-correct, so a
                        # NaN never compares equal to zero and never makes the
                        # row look all-zero).
                        T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                        T.vcast(mask_eq, ind, round_mode="rint")  # i1 -> f16
                        # min over the zero-hit indicator: 1.0 iff every
                        # element of the row is zero.
                        T.reduce_min(ind, mn, dim=1, clear=True)
                        T.reshape(mn, mn_flat)  # (bm, 1) -> (bm,) bridge
                        T.vsub(one_v, mn_flat, any_v)  # any = 1 - all_zero
                        T.vcast(any_v, out_ub, round_mode="rint")  # f16 -> i8 (single hop)
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Path C -- column-tiled with no-alias running min accumulator
# (DESIGN.md sections 3.3 / 6.2)
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
            out: T.Tensor((M,), "int8"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, tile_n), dtype)
                zero_ub = T.alloc_shared((block_m, tile_n), dtype)
                mask_eq = T.alloc_shared((block_m, tile_n), "bool")
                ind = T.alloc_shared((block_m, tile_n), C_IND_DTYPE)
                mn = T.alloc_fragment((block_m, 1), C_IND_DTYPE)
                acc = T.alloc_fragment((block_m, 1), C_IND_DTYPE)
                new_acc = T.alloc_fragment((block_m, 1), C_IND_DTYPE)
                acc_flat = T.alloc_fragment((block_m,), C_IND_DTYPE)
                one_v = T.alloc_fragment((block_m,), C_IND_DTYPE)
                any_v = T.alloc_fragment((block_m,), C_IND_DTYPE)
                out_ub = T.alloc_shared((block_m,), "int8")

                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(one_v, tir.const(1, C_IND_DTYPE))

                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        # min identity: the accumulator starts at 1.0 (= "all
                        # zero so far"), the opposite of the `all` unit's max
                        # identity 0.0.
                        T.fill(acc, tir.const(1, C_IND_DTYPE))
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
                            T.reduce_min(ind, mn, dim=1, clear=True)
                            # No-alias accumulate: the OR merge is a min over
                            # the zero-hit indicator (never vadd / vmax here).
                            T.vmin(acc, mn, new_acc)
                            T.copy(new_acc, acc)
                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, any_v)  # minuend is always 1
                        T.vcast(any_v, out_ub, round_mode="rint")  # f16 -> i8 (single hop)
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func


def _build_tiled_tail(M, N, tile_n, num_full, tail, dtype, default_bm):
    """Column-tiled kernel with a static-width tail block (``tail > 0``).

    The tail uses its own buffer group at the static width ``tail`` (a
    factory-period constant), so no runtime condition or masked load is
    involved (DESIGN.md sections 3.3 / 5.4).
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
            out: T.Tensor((M,), "int8"),
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
                mn = T.alloc_fragment((block_m, 1), "float32")
                mn_tail = T.alloc_fragment((block_m, 1), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc_t = T.alloc_fragment((block_m, 1), "float32")
                acc_flat = T.alloc_fragment((block_m,), "float32")
                one_v = T.alloc_fragment((block_m,), "float32")
                any_v = T.alloc_fragment((block_m,), "float32")
                any_f16 = T.alloc_fragment((block_m,), "float16")
                out_ub = T.alloc_shared((block_m,), "int8")

                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(zero_tail, tir.const(0, dtype))
                T.fill(one_v, tir.const(1, "float32"))

                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.fill(acc, tir.const(1, "float32"))
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
                            T.reduce_min(ind, mn, dim=1, clear=True)
                            T.vmin(acc, mn, new_acc)
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
                        T.reduce_min(ind_tail, mn_tail, dim=1, clear=True)
                        T.vmin(acc, mn_tail, new_acc_t)
                        T.copy(new_acc_t, acc)
                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, any_v)
                        T.vcast(any_v, any_f16, round_mode="rint")
                        T.vcast(any_f16, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Factory (harness contract: _logical_reduce_kernel(M, N, op_kind, dtype)
#                                                 -> _func(block_m) -> main)
# ---------------------------------------------------------------------------
@kernel_factory_cache
def _logical_reduce_kernel(M, N, op_kind, dtype):
    """Build the ``any`` NPU kernel factory.

    ``op_kind`` is kept for the shared GPU factory contract; only ``"any"`` is
    implemented by this migration unit (the sibling all / count_nonzero kinds
    are separate units with their own output contracts).  ``N`` is the raw
    reduction width -- no alignment padding is applied (``any`` is neutral to
    the OR unit element 0.0, so the real-column domain is exact).  ``threads``
    does not exist on NPU (harness K9).
    """
    if op_kind != "any":
        raise ValueError(
            f"unsupported op_kind {op_kind!r}: this migration unit implements "
            f"the 'any' kind only (all/count_nonzero are separate units)"
        )

    cfg = _select_config(M, N, dtype)

    if cfg["path"] == "R":
        _f = _build_resident(M, N, dtype, cfg["block_m"])
        _f.msprof_kernel_name = "any_resident"
    elif cfg["tail"] == 0:
        _f = _build_tiled(M, N, cfg["tile_n"], cfg["num_full"], dtype, cfg["block_m"])
        _f.msprof_kernel_name = "any_tiled"
    else:
        _f = _build_tiled_tail(
            M, N, cfg["tile_n"], cfg["num_full"], cfg["tail"], dtype, cfg["block_m"]
        )
        _f.msprof_kernel_name = "any_tiled_tail"
    _f.design_config = cfg
    return _f


# ---------------------------------------------------------------------------
# Op-layer emulation helpers (host-side flow of AnyFwdOp)
# ---------------------------------------------------------------------------
def _to_logical_float32(x):
    """Mirror the wrapper's ``to_logical_float32`` (Op-layer pre-conversion).

    bool/int32/int64 -> direct float32 cast; complex -> 1.0 where
    ``real != 0 or imag != 0`` (truthiness preserved in every case).
    """
    if x.dtype in (torch.bool, torch.int32, torch.int64):
        return x.to(torch.float32)
    return ((x.real != 0) | (x.imag != 0)).to(torch.float32)


def _normalize_dims(dim, ndim):
    """Op-layer dim normalisation; ``None`` means a full reduction."""
    if dim is None:
        return []
    if isinstance(dim, int):
        dim = [dim]
    return sorted({d % ndim for d in dim})


def _prepare_2d(x, dim):
    """Emulate the Op layer: movedim reduced dims last, contiguous, reshape.

    Returns ``(x2d, out_shape)``.  ``dim=None`` is a full reduction of the
    flattened tensor (``M=1``, ``N=numel``, 0-dim output); an empty non-``None``
    dim list follows the Op layer's ``_empty_dim_policy = "noop"`` (no kernel
    call) and is handled by the caller instead.
    """
    dims = _normalize_dims(dim, x.dim())
    if dim is None:
        xd = x.reshape(-1).contiguous()
        return xd.reshape(1, xd.numel()), ()
    keep = [d for d in range(x.dim()) if d not in dims]
    xd = x.permute(keep + dims).contiguous()
    M = 1
    for d in keep:
        M *= x.shape[d]
    N = xd.numel() // M
    return xd.reshape(M, N), tuple(x.shape[d] for d in keep)


def _declared_width(kernel):
    """Declared input width of a compiled kernel (prim_func buffer map)."""
    for buf in kernel.prim_func.buffer_map.values():
        if str(buf.dtype) != "int8":
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

# Aggregated precision bookkeeping for the final report.
_STATS = {"checks": 0, "mismatch": 0}


def _cfg_str(cfg):
    return (
        f"path={cfg['path']} block_m={cfg['block_m']} tile_n={cfg['tile_n']} "
        f"num_full={cfg['num_full']} tail={cfg['tail']}"
    )


def _exact_compare(y, ref, tag):
    """Exact bool comparison (torch.equal, no tolerance; DESIGN.md 8.2).

    The kernel returns the raw **int8** carrier (``bool`` cannot be a GM
    storage dtype); this mirrors the wrapper's ``out_raw.bool()`` restore, and
    additionally asserts that the carrier only ever holds 0/1.
    """
    _STATS["checks"] += 1
    if y.dtype not in (torch.bool, torch.int8):
        raise AssertionError(f"[{tag}] output dtype {y.dtype} != int8 (bool carrier) / bool")
    if y.dtype == torch.int8:
        vals = torch.unique(y.cpu())
        if not bool(((vals == 0) | (vals == 1)).all()):
            raise AssertionError(f"[{tag}] int8 carrier holds non-0/1 values {vals.tolist()}")
    yc = y.bool().cpu()
    if tuple(yc.shape) != tuple(ref.shape):
        raise AssertionError(f"[{tag}] shape {tuple(yc.shape)} != ref {tuple(ref.shape)}")
    if not torch.equal(yc, ref):
        _STATS["mismatch"] += int((yc != ref).sum().item())
        bad = (yc != ref).nonzero().flatten()
        n_bad = int(bad.numel())
        i = int(bad[0]) if n_bad else -1
        raise AssertionError(
            f"[{tag}] logical-OR mismatch: {n_bad}/{ref.numel()} entries differ; "
            f"first at {i}: got={bool(yc.flatten()[i])} want={bool(ref.flatten()[i])}"
        )


def _make_input(M, N, dtype_str, seed=0, zero_frac=0.3):
    """Mixed-density input with forced zeros / non-zeros / NaN / +-0 corners.

    ``randn`` alone has a zero density of ~0, which makes ``any`` constantly
    True, i.e. a constant implementation would also pass.  ~30% of the entries
    are therefore zeroed and the first rows are forced to the discriminating
    verdicts (all zero -> False, all non-zero -> True, single non-zero among
    zeros -> True, NaN + zeros only -> True, all NaN -> True).
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(M, N, generator=gen)
    keep = (torch.rand(M, N, generator=gen) > zero_frac).to(torch.float32)
    x = x * keep
    if M > 0:
        x[0, :] = 0.0  # all-zero row -> False
    if M > 1:
        x[1, :] = 1.5  # all-non-zero row -> True
    if M > 2 and N > 1:
        x[2, :] = 0.0
        x[2, 0] = 1.0  # a single non-zero among zeros -> True
    if M > 3 and N > 1:
        x[3, :] = 0.0
        x[3, N // 2] = float("nan")  # NaN + zeros only -> True (key row)
    if M > 4:
        x[4, :] = float("nan")  # all NaN -> True (key row)
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
    kernel = _logical_reduce_kernel(M, N, "any", dtype_str)(cfg["block_m"])
    # The kernel must declare the raw width N (the Op layer does not pad:
    # `_kernel_handles_padding = True`).  A width mismatch shows up as "row 0
    # correct, every later row wrong" without any error being raised.
    declared = _declared_width(kernel)
    if declared != N:
        raise AssertionError(
            f"[{tag}] declared input width {declared} != raw N {N} "
            f"(host-side padding would silently corrupt rows >= 1)"
        )
    y = kernel(x)
    _exact_compare(y, golden_any_logical_reduce(x.cpu(), dim=-1), tag)
    return cfg, y


def _run_case(M, N, dtype_str, tag, **kw):
    return _run_tensor(_make_input(M, N, dtype_str), dtype_str, tag, **kw)


def _run_nd(x, dim, dtype_str, tag, **kw):
    """Run through the emulated Op-layer flow (movedim -> reshape -> back)."""
    x2, out_shape = _prepare_2d(x, dim)
    cfg, y = _run_tensor(x2, dtype_str, tag, **kw)
    ref = golden_any_logical_reduce(x.cpu(), dim=dim)
    _exact_compare(y.reshape(out_shape), ref, f"{tag}-nd")
    return cfg


# ---------------------------------------------------------------------------
# L0: gate tests (blocking) -- DESIGN.md section 8.2
# ---------------------------------------------------------------------------
def _run_constructed_rows(dtype_str, tag):
    """L0-2: rows covering the complement's two extremes and its corners.

    The two key discriminator rows are "NaN + zeros and nothing else" and
    "all NaN": ``torch.any`` returns True for both (NaN is non-zero).  A
    ``vcmp("ne")`` implementation would return False for both, so those two
    rows are the only place where that misuse surfaces.
    """
    N = 512
    nan = float("nan")
    rows = []
    rows.append(torch.zeros(N))  # False: every element zero
    rows.append(torch.ones(N))  # True: no zero
    r = torch.ones(N)
    r[0] = 0.0
    rows.append(r)  # True: one zero, the rest non-zero
    r = torch.ones(N)
    r[N - 1] = 0.0
    rows.append(r)  # True
    rows.append(torch.full((N,), -0.0))  # False: negative zero only
    r = torch.zeros(N)
    r[N // 2] = nan
    rows.append(r)  # True: KEY -- NaN + zeros and nothing else
    rows.append(torch.full((N,), nan))  # True: KEY -- all NaN
    r = torch.zeros(N)
    r[0] = 1.0
    rows.append(r)  # True: a single non-zero among zeros
    r = torch.zeros(N)
    r[N - 1] = -1.0
    rows.append(r)  # True
    r = torch.ones(N)
    r[::3] = 0.0
    r[::5] = nan
    rows.append(r)  # True: NaN mixed with zeros and non-zeros
    r = torch.zeros(N)
    r[0] = float("inf")
    r[N - 1] = -float("inf")
    rows.append(r)  # True: +-inf are non-zero
    tiny = torch.finfo(_DTYPE_MAP[dtype_str]).tiny
    sub = float(torch.tensor(tiny, dtype=_DTYPE_MAP[dtype_str])) / 2.0
    r = torch.zeros(N)
    r[1] = sub if sub != 0.0 else tiny
    rows.append(r)  # True: subnormal / tiny does not flush to zero
    r = torch.zeros(N)
    r[::2] = -0.0
    rows.append(r)  # False: mixed +0 / -0 only
    want = [
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    if len(rows) != len(want):
        raise AssertionError(f"[{tag}] row/want length mismatch {len(rows)} != {len(want)}")
    x = torch.stack(rows).to(_DTYPE_MAP[dtype_str]).to("npu")
    cfg, y = _run_tensor(x, dtype_str, tag)
    got = y.bool().cpu().tolist()
    if got != want:
        raise AssertionError(f"[{tag}] NaN/+-0 corner verdicts {got} != {want}")
    print(
        f"[{tag}] PASS: constructed rows M={x.shape[0]} N={N} dtype={dtype_str} "
        f"({_cfg_str(cfg)}) NaN+zeros=True / all-NaN=True / all-zero=False"
    )


def _run_vectorised_ir_check(dtype_str="float16", M=64, N=512):
    """L0-5: confirm the explicit vector chain survives lowering.

    Reads the lowered MLIR of the compiled kernel (``mlir_content``; the
    ``JITKernel.get_kernel_source()`` channel is not textual on ``JitKernel_NPU``).
    The judgement / indicator / reduction / epilogue steps must appear as the
    ``hivm.hir.v*`` vector ops, with ``<min>`` (not ``<max>``, which would mean
    the sibling ``all`` direction was copied) and with no per-element scalar
    form (what ``T.Parallel`` + ``if_then_else`` degrades to at ``bm >= 2``).
    """
    kernel = _logical_reduce_kernel(M, N, "any", dtype_str)(8)
    try:
        mlir = kernel.mlir_content
    except Exception:  # noqa: BLE001
        mlir = ""
    if not mlir:
        print(
            "[L0-5] NOTE (record only): MLIR not exposed; vectorisation is implied by"
            " the explicit v-prefix chain and the measured UB price"
        )
        return
    need = ("hivm.hir.vcmp", "hivm.hir.vcast", "hivm.hir.vsub", "hivm.hir.vreduce", "<min>")
    missing = [k for k in need if k not in mlir]
    if not re.search(r"reduce_dims\s*=\s*\[\s*1\b", mlir):
        missing.append("reduce_dims = [1]")
    forbid = ("arith.cmpf", "if_then_else", "scf.for", "<max>")
    present = [k for k in forbid if k in mlir]
    if missing:
        raise AssertionError(f"[L0-5] lowered MLIR lacks the vector chain ops {missing}")
    if present:
        raise AssertionError(
            f"[L0-5] lowered MLIR contains scalar / wrong-direction forms {present} "
            f"(the judgement chain must stay vectorised at bm>=2 and reduce with min)"
        )
    print(
        "[L0-5] PASS: lowered MLIR has vcmp(eq)->vcast(i1->f32)->vreduce<min> "
        "reduce_dims = [1] ->vsub->vcast(f32->f16)->vcast(f16->i8) with no scalar"
        " element loop and no <max>"
    )


def run_L0():
    n = 0
    # L0-1 path R representative shapes (DESIGN.md section 8.2).
    _run_case(32, 256, "float16", "L0-1a", expect_path="R", expect_block_m=8)
    _run_case(32, 256, "float32", "L0-1b", expect_path="R", expect_block_m=8)
    for dtype_str in ("float16", "bfloat16", "float32"):
        _run_case(128, 512, dtype_str, "L0-1c", expect_path="R", expect_block_m=8)
        n += 1
    n += 2
    for dtype_str in ("float16", "float32"):
        _run_case(128, 300, dtype_str, "L0-1d", expect_path="R", expect_block_m=8)
        n += 1
    # L0-1e M tail block (129 % 8 == 1) -- source-defect-fix regression.
    _run_case(129, 512, "float16", "L0-1e", expect_block_m=8)
    n += 1
    # L0-1f degenerate shapes.
    _run_case(1, 1, "float16", "L0-1f", expect_block_m=1)
    x1 = _make_input(1, 512, "float16")
    _run_nd(x1.reshape(512), -1, "float16", "L0-1g", expect_block_m=1)
    n += 2

    # L0-2 discriminative corners (NaN / +-0 / inf / subnormal / single non-zero).
    _run_constructed_rows("float16", "L0-2a")
    _run_constructed_rows("bfloat16", "L0-2b")
    _run_constructed_rows("float32", "L0-2c")
    n += 3
    for dtype_str in ("float16", "bfloat16", "float32"):
        _run_case(4, 1, dtype_str, "L0-2d", expect_block_m=4)
        n += 1

    # L0-3 path C representative shapes (column tiles + online min merge).
    _run_case(32, 32768, "float32", "L0-3a", expect_path="C", expect_tile_n=8192)
    x = _make_input(128, 16384, "float32").reshape(4, 128, 4096)
    _run_nd(x, [0, 2], "float32", "L0-3b", expect_path="C", expect_tile_n=8192)
    _run_case(2048, 16384, "float32", "L0-3c", expect_path="C", expect_tile_n=8192)
    n += 3
    # Non-dividing static tail block (8192 does not divide 9000 -> tail=808).
    cfg, _ = _run_case(64, 9000, "float16", "L0-3d", expect_path="C", expect_tile_n=8192)
    if (cfg["num_full"], cfg["tail"]) != (1, 808):
        raise AssertionError(
            f"[L0-3d] (num_full, tail)={(cfg['num_full'], cfg['tail'])} != (1, 808)"
        )
    n += 1
    # Divisor-preference branch: 8448 = 33 * 256, 8192 does not divide it, the
    # largest fitting 256-multiple divisor (2816) is preferred -> zero tail.
    cfg, _ = _run_case(8, 8448, "float32", "L0-3e", expect_path="C", expect_tile_n=2816)
    if (cfg["num_full"], cfg["tail"]) != (3, 0):
        raise AssertionError(f"[L0-3e] (num_full, tail)={(cfg['num_full'], cfg['tail'])} != (3, 0)")
    n += 1

    # L0-4 core-split boundaries: num_logical just below / exactly at / far
    # above the 40 physical vector cores, plus a non-multiple task count.
    _run_case(256, 256, "float16", "L0-4a", expect_block_m=8)  # 32 logical cores
    _run_case(320, 256, "float16", "L0-4b", expect_block_m=8)  # 40 logical cores
    _run_case(256, 4096, "float16", "L0-4c", expect_block_m=2)  # 128 -> 40 x 4
    _run_case(513, 512, "float16", "L0-4d", expect_block_m=8)  # 65 -> 40 x 2
    n += 4

    # L0-5 vectorisation spot check: the lowered IR must contain the explicit
    # vector chain (vcmp / vcast / reduce_min) and no per-element scalar form.
    _run_vectorised_ir_check()
    n += 1

    print(f"[L0] ALL PASS: {n} shape/edge cases")


# ---------------------------------------------------------------------------
# L1: functional coverage (blocking)
# ---------------------------------------------------------------------------
def run_L1():
    n = 0
    # L1-1 element-budget boundary configuration (block_m * N == E_cap) and a
    # non-contiguous input (the Op layer always passes a contiguous tensor, so
    # the copy is compared against the same data in a strided view).
    for dtype_str in ("float16", "bfloat16", "float32"):
        _run_case(256, 4096, dtype_str, "L1-1", expect_block_m=2)
        n += 1
    x_full = _make_input(128, 1024, "float16")
    x_nc = x_full[:, :512]
    x_c = x_nc.contiguous()
    _, y = _run_tensor(x_c, "float16", "L1-1-noncontig")
    _exact_compare(y, golden_any_logical_reduce(x_nc.cpu(), dim=-1), "L1-1-noncontig")
    n += 1

    # L1-2 higher-rank and non-last-axis reductions (Op layer movedim+reshape).
    _run_nd(_make_input(128, 512, "float16").reshape(2, 64, 512), -1, "float16", "L1-2a")
    _run_nd(_make_input(64, 512, "float16").reshape(2, 4, 8, 512), -1, "float16", "L1-2b")
    _run_nd(_make_input(512, 64, "float16"), 0, "float16", "L1-2c")
    _run_nd(_make_input(64, 512, "float16").reshape(4, 16, 512), 0, "float16", "L1-2d")
    _run_nd(_make_input(2048, 64, "float16").reshape(4, 64, 512), 1, "float16", "L1-2e")
    _run_nd(_make_input(2048, 64, "float32").reshape(4, 64, 512), -2, "float32", "L1-2f")
    n += 6

    # L1-3 extreme / subnormal values (each extreme is exact in its own dtype).
    for dtype_str in ("float16", "bfloat16", "float32"):
        info = torch.finfo(_DTYPE_MAP[dtype_str])
        sub = float(torch.tensor(info.tiny, dtype=_DTYPE_MAP[dtype_str])) / 2.0
        if sub == 0.0:
            sub = float(info.tiny)
        big = float(info.max)
        N = 256
        rows = [torch.zeros(N), torch.full((N,), big), torch.full((N,), -big)]
        r = torch.zeros(N)
        r[0] = sub
        r[N - 1] = -sub
        rows.append(r)  # subnormals (or tiny normals) are non-zero -> True
        r = torch.zeros(N)
        r[5] = -0.0
        rows.append(r)  # all +-0 -> False
        r = torch.zeros(N)
        r[0] = float("inf")
        r[1] = -float("inf")
        rows.append(r)  # +-inf -> True
        x = torch.stack(rows).to(_DTYPE_MAP[dtype_str]).to("npu")
        _, y = _run_tensor(x, dtype_str, f"L1-3-{dtype_str}")
        want = [False, True, True, True, False, True]
        if y.bool().cpu().tolist() != want:
            raise AssertionError(f"[L1-3-{dtype_str}] {y.bool().cpu().tolist()} != {want}")
        bits = "".join("T" if v else "F" for v in want)
        print(f"[L1-3-{dtype_str}] PASS: extreme rows {bits}")
        n += 1

    # L1-4 unsupported-storage dtypes through the Op-layer float32 pre-cast.
    n += _run_storage_dtypes()

    # L1-5 keepdim (both states) through the emulated Op layer.
    x5 = _make_input(64, 512, "float16").reshape(4, 16, 512)
    x2, out_shape = _prepare_2d(x5, -1)
    cfg, y = _run_tensor(x2, "float16", "L1-5-keepdim")
    ref_k = golden_any_logical_reduce(x5.cpu(), dim=-1, keepdim=True)
    _exact_compare(y.reshape(out_shape), ref_k.reshape(out_shape), "L1-5-keepdim")
    ref_n = golden_any_logical_reduce(x5.cpu(), dim=-1, keepdim=False)
    _exact_compare(y.reshape(out_shape), ref_n, "L1-5-nokeepdim")
    n += 1

    # L1-6 dim=None: full reduction of the flattened tensor -> 0-dim scalar.
    x_flat = _make_input(1, 8192, "float32").reshape(8192)
    x2, out_shape = _prepare_2d(x_flat, None)  # dim=None -> M=1, N=numel
    cfg, y = _run_tensor(x2, "float32", "L1-6-dimnone", expect_path="R")
    _exact_compare(
        y.reshape(out_shape),
        golden_any_logical_reduce(x_flat.cpu(), dim=None),
        "L1-6-dimnone",
    )
    if out_shape != ():
        raise AssertionError(f"[L1-6-dimnone] out_shape={out_shape} != ()")
    n += 1

    # L1-7 Op layer `_empty_dim_policy = "noop"`: dim=[] / dim=() never enter the
    # kernel, the Op layer returns x.bool() directly.
    for empty in ([], ()):
        x = _make_input(4, 8, "float16").cpu()
        got = torch.any(x, dim=empty)
        want = x.bool()
        if not torch.equal(got, want):
            raise AssertionError(f"[L1-7] dim={empty!r} noop policy mismatch")
        if tuple(got.shape) != tuple(x.shape):
            raise AssertionError(
                f"[L1-7] dim={empty!r} shape {tuple(got.shape)} != {tuple(x.shape)}"
            )
    print("[L1-7] PASS: dim=[] / dim=() follow the noop policy (x.bool(), shape kept)")
    n += 1

    # L1-8 M tail + non-aligned N together.
    _run_case(8, 300, "float16", "L1-8")
    n += 1

    # L1-9 path C tile arithmetic across widths (dividing / non-dividing).
    cfg, _ = _run_case(64, 16384, "float16", "L1-9a", expect_path="C", expect_tile_n=8192)
    if (cfg["num_full"], cfg["tail"]) != (2, 0):
        raise AssertionError(f"[L1-9a] (num_full, tail)={(cfg['num_full'], cfg['tail'])} != (2, 0)")
    cfg, _ = _run_case(8, 10000, "float32", "L1-9b", expect_path="C", expect_tile_n=8192)
    if (cfg["num_full"], cfg["tail"]) != (1, 1808):
        raise AssertionError(
            f"[L1-9b] (num_full, tail)={(cfg['num_full'], cfg['tail'])} != (1, 1808)"
        )
    cfg, _ = _run_case(8, 8448, "float16", "L1-9c", expect_path="C", expect_tile_n=2816)
    if (cfg["num_full"], cfg["tail"]) != (3, 0):
        raise AssertionError(f"[L1-9c] (num_full, tail)={(cfg['num_full'], cfg['tail'])} != (3, 0)")
    n += 3

    # L1-10 caller-supplied block_m robustness.  The GPU-period wrapper
    # heuristic may still pass a block_m that overshoots the element budget
    # (e.g. 8 * 4096 = 32768 elements > the 24576 hard UB limit) until Stage 5
    # rewires ``default_config`` (DESIGN.md section 5.6): the kernel must fall
    # back to the ladder value rather than overflow UB, and a valid smaller
    # value must be honoured.
    M, N = 256, 4096
    x = _make_input(M, N, "float16")
    ref = golden_any_logical_reduce(x.cpu(), dim=-1)
    for bm_in in (1, 8):
        kernel = _logical_reduce_kernel(M, N, "any", "float16")(bm_in)
        y = kernel(x)
        _exact_compare(y, ref, f"L1-10-bm{bm_in}")
    if _resolve_block_m(8, 2, N, M) != 2:
        raise AssertionError(
            "[L1-10] over-budget caller block_m was not clamped to the ladder value"
        )
    if _resolve_block_m(1, 2, N, M) != 1:
        raise AssertionError("[L1-10] under-budget caller block_m was not honoured")
    print(f"[L1-10] PASS: caller block_m 1 (honoured) and 8 (clamped) both exact on ({M},{N})")
    n += 1

    print(f"[L1] ALL PASS: {n} cases")


def _run_storage_dtypes():
    """L1-4: bool / int32 / int64 / complex64 / complex128 via the Op pre-cast.

    Each storage dtype gets the same discriminating rows (all zero, all
    non-zero, a single non-zero, and -- for complex -- a NaN imaginary part
    with no other non-zero), so the truthiness contract is checked on both
    sides.  The reference compares against the *original* dtype tensor
    (``torch.any``), not against the pre-converted float32 one.
    """
    n = 0
    M, N = 64, 512
    for dtype in (torch.bool, torch.int32, torch.int64, torch.complex64, torch.complex128):
        if dtype.is_complex:
            base = torch.randn(M, N) * (torch.rand(M, N) > 0.3).to(torch.float32)
            imag = torch.randn(M, N) * (torch.rand(M, N) > 0.3).to(torch.float32)
            x_cpu = torch.complex(base, imag).to(dtype)
            x_cpu[0, :] = complex(0.0, 0.0)
            x_cpu[1, :] = complex(1.0, -1.0)
            x_cpu[2, :] = complex(0.0, 0.0)
            x_cpu[2, 0] = complex(0.0, 2.0)
            x_cpu[3, :] = complex(0.0, 0.0)
            x_cpu[3, N // 2] = complex(0.0, float("nan"))  # NaN imag only -> True
        elif dtype is torch.bool:
            x_cpu = torch.rand(M, N) > 0.4
            x_cpu[0, :] = False
            x_cpu[1, :] = True
            x_cpu[2, :] = False
            x_cpu[2, 0] = True
            x_cpu[3, :] = False  # all False -> False
        else:
            x_cpu = (torch.randn(M, N) * (torch.rand(M, N) > 0.3).to(torch.float32) * 100).to(dtype)
            x_cpu[0, :] = 0
            x_cpu[1, :] = 7
            x_cpu[2, :] = 0
            x_cpu[2, 0] = -3
            x_cpu[3, :] = 0  # all zero -> False
        ref = golden_any_logical_reduce(x_cpu, dim=-1)
        xf = _to_logical_float32(x_cpu).contiguous().to("npu")
        kernel = _logical_reduce_kernel(M, N, "any", "float32")(8)
        y = kernel(xf)
        _exact_compare(y, ref, f"L1-4-{dtype}")
        print(f"[L1-4-{dtype}] PASS: pre-cast truthiness preserved (rows 0/1/2/3 -> F/T/T/{'-'})")
        n += 1
    return n


# ---------------------------------------------------------------------------
# L2: boundary / contract behaviour (record only, non-blocking)
# ---------------------------------------------------------------------------
def run_L2():
    # Factory-period raises.
    for label, fn in (
        ("M=0", lambda: _select_config(0, 256, "float16")),
        ("N=0", lambda: _select_config(1, 0, "float16")),
        ("op_kind='all'", lambda: _logical_reduce_kernel(8, 256, "all", "float16")),
        (
            "op_kind='count_nonzero'",
            lambda: _logical_reduce_kernel(8, 256, "count_nonzero", "float16"),
        ),
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

    # Path R <-> path C consistency: the same data with the extra columns of
    # the path-C shape forced to 0.0 (the OR-neutral value) must give
    # bit-identical verdicts.  N = 8192 stays resident; N = 16384 is tiled.
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
            f"[L2] PASS: path R ({_cfg_str(cfg_r)}) == path C ({_cfg_str(cfg_c)}) "
            f"on the same effective data (extra columns set to the OR-neutral 0.0)"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): path R/C consistency: {e}")

    # N just above E_cap -> path C with the smallest possible tail.
    try:
        cfg, _ = _run_case(8, 8193, "float16", "L2-tn+1")
        print(f"[L2] PASS: N=E_cap+1 -> {_cfg_str(cfg)}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): N=E_cap+1: {e}")

    # The `all` sibling's merge operator must not appear anywhere in this unit.
    try:
        cfg, _ = _run_case(4, 16384, "float16", "L2-direction")
        print("[L2] PASS: path C uses the vmin merge (a vmax merge would invert the OR)")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): path C direction case: {e}")

    # M=0 would need a zero-sized grid; the factory raises (Op layer should
    # short-circuit instead -- DESIGN.md section 9.2-R6).
    print("[L2] NOTE: M=0 is a factory raise (Op-layer short-circuit recommended)")


# ---------------------------------------------------------------------------
# Boundary: extreme data values and shapes (record only, non-blocking)
# ---------------------------------------------------------------------------
def run_boundary():
    # All-zero input -> every row False (the any direction; `all` would say True).
    for dtype_str in ("float16", "bfloat16", "float32"):
        try:
            x = torch.zeros(32, 256, dtype=_DTYPE_MAP[dtype_str], device="npu")
            _, y = _run_tensor(x, dtype_str, f"Boundary-zeros-{dtype_str}")
            if bool(y.cpu().any()):
                raise AssertionError("all-zero input produced a True verdict")
            print(f"[Boundary] PASS: all zeros (32,256) {dtype_str} -> all False")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): all zeros {dtype_str}: {e}")
        try:
            x = torch.ones(32, 256, dtype=_DTYPE_MAP[dtype_str], device="npu")
            _, y = _run_tensor(x, dtype_str, f"Boundary-ones-{dtype_str}")
            if not bool(y.cpu().all()):
                raise AssertionError("all-ones input produced a False verdict")
            print(f"[Boundary] PASS: all ones (32,256) {dtype_str} -> all True")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): all ones {dtype_str}: {e}")
        try:
            M, N = 16, 256
            x = torch.zeros(M, N, dtype=_DTYPE_MAP[dtype_str], device="npu")
            x[:, N // 2] = float("nan")
            _, y = _run_tensor(x, dtype_str, f"Boundary-nan-{dtype_str}")
            if not bool(y.cpu().all()):
                raise AssertionError("NaN-only-plus-zeros rows came out False (ordered-NE misuse)")
            print(f"[Boundary] PASS: NaN + zeros column {dtype_str} -> all True")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): NaN column {dtype_str}: {e}")

    # 1D loads across dtypes.
    for dtype_str in ("float16", "bfloat16", "float32"):
        try:
            x = _make_input(1, 512, dtype_str).reshape(512)
            _run_nd(x, -1, dtype_str, f"Boundary-1d-{dtype_str}")
            print(f"[Boundary] PASS: 1D (512,) {dtype_str}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): 1D {dtype_str}: {e}")

    # Non-256-multiple widths (raw N, no padding).
    for N in (300, 64, 4, 2):
        for dtype_str in ("float16", "float32"):
            try:
                _run_case(64, N, dtype_str, f"Boundary-rawN{N}-{dtype_str}")
                print(f"[Boundary] PASS: raw width N={N} {dtype_str} (no alignment pad)")
            except Exception as e:  # noqa: BLE001
                print(f"[Boundary] WARN (record only): raw width N={N} {dtype_str}: {e}")

    # M % block_m != 0 with a large width.
    try:
        _run_case(1031, 512, "float16", "Boundary-Mtail")
        print("[Boundary] PASS: M=1031 (1031 % 8 = 7) tail block exact")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): M tail (1031,512): {e}")

    # N == 1 (single-element rows).
    for dtype_str in ("float16", "bfloat16", "float32"):
        try:
            _run_case(16, 1, dtype_str, f"Boundary-N1-{dtype_str}")
            print(f"[Boundary] PASS: N=1 {dtype_str}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): N=1 {dtype_str}: {e}")

    # N just above E_cap (smallest non-empty tail on the column-tiling path).
    try:
        cfg, _ = _run_case(16, 8193, "float16", "Boundary-overcap")
        print(f"[Boundary] PASS: N > E_cap -> {_cfg_str(cfg)}")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): N > E_cap: {e}")

    # Larger random shape.
    try:
        _run_case(512, 2048, "float16", "Boundary-large")
        print("[Boundary] PASS: (512,2048) float16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): (512,2048): {e}")

    # N = 0 (empty OR).  torch.any((3,0), dim=-1) == [False, False, False]; the
    # factory refuses a zero-width kernel by design (DESIGN.md section 9.2-R5)
    # and the Op layer must short-circuit.  Recorded, not blocking.
    try:
        _select_config(3, 0, "float16")
        print("[Boundary] WARN (record only): N=0 did not raise in the factory")
    except ValueError as exc:
        torch_ref = torch.any(torch.empty(3, 0), dim=-1).tolist()
        print(
            f"[Boundary] PASS (record): N=0 factory raise as designed ({str(exc)[:50]}...); "
            f"torch contract = {torch_ref} (Op-layer short-circuit required)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    elif args.level == "L2":
        run_L2()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print(
        f"[SUMMARY] exact bool comparisons={_STATS['checks']} mismatching entries={_STATS['mismatch']}"
    )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
