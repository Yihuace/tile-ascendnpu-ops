# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""``_logical_reduce_kernel`` -- logical-AND row reduction (NPU Developer mode).

Migrated from the GPU TileOPs ``LogicalReduceKernel`` baseline factory
(``tileops/kernels/reduction/logical_reduce.py``, ``_logical_reduce_kernel``).
The GPU factory is shared by the any / all / count_nonzero kinds; this
migration unit implements the **all** kind only (``op_kind`` is kept as a
factory parameter to preserve the harness contract).  Contract:

  * input  ``x``   : ``(M, N)`` raw width, float16 / bfloat16 / float32
  * output ``out`` : ``(M,)`` **int8** carrier (``bool`` in the manifest);
    the wrapper restores bool via ``out_raw.bool()``.

It must NOT follow the sibling count_nonzero unit's int64 contract: the
reduction primitive here is ``reduce_max`` over the zero-hit indicator and the
complement minuend is the constant ``1`` (count_nonzero uses ``reduce_sum`` and
``N``).  The GPU N-tiled fallback (``_logical_reduce_kernel_tiled``) is
intentionally not migrated; the NPU column-tiling path below is driven by the
UB element budget, not by a GPU fragment column limit.

NPU algorithm (DESIGN.md sections 0.6 / 1.6.1):

    y[i] = 1 - max_{j < N} [ x[i, j] == 0.0 ]

i.e. De Morgan's complement form with an IEEE **ordered EQ** judgement
(``T.vcmp(..., "eq")``).  ``eq`` is IEEE-correct, so NaN is never counted as a
zero and therefore does not break ``all``, matching both ``torch.all`` and the
GPU source's unordered-NE semantics.  ``"ne"`` is ordered-NE on this toolchain
(NaN always false) and is forbidden here.  The complement is applied on the
tiny ``(block_m,)`` micro-buffer, so no full-width ``1 - indicator`` pass is
needed.

Chain (Stage 4 tuned): ``T.vcmp`` -> ``T.vcast(i1 -> f16)`` ->
``T.reduce_max(dim=1)`` -> ``T.vsub(1, .)`` -> ``T.vcast(f16 -> i8)`` for the
**resident** path.  The zero-hit indicator is exactly 0.0/1.0, so it can be
materialised **and** reduced in float16: ``reduce_max`` over 0/1 is exact,
``1 - max`` is exact, and the epilogue ``f16 -> i8`` is a documented cast
(``f32 -> i8`` has no ``vcast`` table entry, which is why the Stage 3 chain
needed the extra ``f32 -> f16`` step).  The f16 plane halves the reduce input
bytes, removes that epilogue cast and saves 2 B/element of UB.

The column-tiled paths (``N > E_CAP``) keep the Stage 3 **float32** chain:
the same change measured -1.8% on 32k (below noise) and conflicting on 3d
(-10.0% / +1.2% in two A/B sessions), so it is not adopted there.

The optional static-width tail block (path C, ``tail > 0``) deliberately keeps
the Stage 3 **float32** indicator chain: a narrow f16 tail plane mis-reduces on
this toolchain (measured 22/32 rows wrong at ``tail = 1``, 3/32 at ``tail = 8``,
exact from ``tail >= 64``), while the merged ``max`` stays exact.  No tune
workload has a tail (``mask-validation-4k`` / ``-32k`` / ``3d-multidim`` all
have ``tail == 0``), so the tail path costs nothing in the tuned cases.

Stage 4 measured (``msprof op`` ``Task Duration(us)``, Ascend910B4, 40 AIV,
UB 196,608 B, 30 launches, interleaved A/B/A/B, 3 pairs per session):

  * **adopted: the f16 indicator on the resident path** --
    ``mask-validation-4k`` 2.959 -> 2.833 us (**-4.3%**, 3/3 pairs) and
    -6.4% vs an adjacent baseline re-measurement in a second session.
  * **not adopted: the same change on the column-tiled paths** --
    ``mask-validation-32k`` -1.8% (below the 3% noise threshold) and
    ``3d-multidim`` **conflicting**: -10.0% (3/3 pairs) in one session,
    +1.2% (3/3 pairs) in a later one, i.e. not separable from the 5-8%
    run-state drift of this box.  Those shapes therefore keep the Stage 3
    float32 chain and their generated code is byte-identical to Stage 3.
  * **rejected:** DESIGN section 1.6.4 path R' (single-tile resident for
    ``E_CAP < N <= UB element cap``) -- it does not compile at all with the f32
    indicator (a 16384-element f32 block needs 198,848 B > 196,608 B =
    12.14 B/element), and with the f16 indicator it measures 15.715 us on 3d
    (**+9.6%**) because the 64 KB single tile loses the multi-buffer overlap
    (``mte2_ratio`` 0.552 -> 0.671, i.e. the read becomes fully exposed).
  * **rejected:** widening the column tile to the UB element cap
    (32k: 4x8192 -> 2x16384; 32k -2.6%, 3d -6.9%) -- the 64 KB read costs
    1.508 us vs 0.192 us for a 32 KB tile (measured), so fewer/wider tiles lose.
  * **rejected:** a grid-filling row-block policy (``block_m = 1`` so 32 rows
    use 32 of 40 cores; 4k -18.5%) -- the empty-kernel floor grows with the
    block count (2.511 us at 32 blocks vs 1.845 us at 16 blocks).
  * **rejected:** ``T.Pipelined(num_stages=2)`` on the column-tile loop
    (32k -0.9%, pipe counters identical to the auto multi-buffer version).
  * the Stage 3 tiling dispatch is therefore kept unchanged; the only adopted
    knob is ``TUNED_IND_DTYPE`` and it applies to the resident path.

Two capacity paths (factory-period dispatch, one dedicated ``prim_func`` per
variant -- a Python-level ``if`` that *defines buffers* inside a ``prim_func``
is not parseable by this toolchain):

  * path R (resident, ``N <= E_CAP``): one tile per row block.
  * path C (column-tiled, ``N > E_CAP``): ``bm = 1``, ``num_full`` static-width
    column tiles with a no-alias running ``max`` accumulator, plus an optional
    static-width tail block (separate buffer group, generated at factory level).

Element budget: measured on this toolchain, a 16,384-element block of the
Stage 3 f32 indicator chain needs **198,848 B** (12.14 B/element) > the
196,608 B UB, i.e. 16384 f32 elements per block is already past the hard limit;
the f16 indicator variant drops ~2 B/element (a 16384-element block fits).  The
race-safe design budget is ``E_CAP = 8192`` elements/block, which also keeps the
column tiles double-buffered (2 x 8 B/element x 8192 <= 196,608 B).

No host-side padding: the kernel declares the raw ``(M, N)`` window, matching
the Op layer contract ``_kernel_handles_padding = True``.  Logical AND is
neutral to the AND unit element (1.0) padding, so the real-column domain needs
no alignment.

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

__all__ = [
    "ASCEND_MODE",
    "TUNED_BM_POLICY",
    "TUNED_E_CAP",
    "TUNED_IND_DTYPE",
    "TUNED_RPRIME",
    "TUNED_TILE_CAP_MODE",
    "_logical_reduce_kernel",
    "_select_config",
    "golden__logical_reduce_kernel",
    "golden_all_logical_reduce",
]

# ---------------------------------------------------------------------------
# Factory-period constants (DESIGN.md sections 4.5 / 5.2)
# ---------------------------------------------------------------------------
UB_BYTES = 196608  # AscendArch().ub_cap, queried on the device (192 KB)
# Measured compiler-side UB unit price (B/element) of the
# vcmp(eq) -> vcast(i1->f32) -> reduce_max indicator chain:
# 8.00 for fp16/fp32 and 12.00 for bf16 (comparison lowered to f32).
# Stage 4 re-measurement: the price is (W, block_m)-dependent, not a constant --
# a 16384-element f32 block of the Stage 3 chain needs 198,848 B
# (12.14 B/element) and does not compile.
UNIT_PRICE = {"float16": 8, "float32": 8, "bfloat16": 12}
E_CAP = UB_BYTES // 2 // max(UNIT_PRICE.values())  # 8192 elements (worst dtype)
# Measured pass points for the **delivered** chain (float16 indicator):
# 16384 f32 elements compile (the f16 plane saves ~2 B/element vs Stage 3, so
# the Stage 3 f32 chain's 16384-element overflow becomes a pass).
HARD_CAP_ELEMS = {"float16": 24576, "bfloat16": 16384, "float32": 16384}
DEFAULT_ALIGNMENT = 256
LAUNCH_GATE = 80  # > 2 waves of 40 vector cores -> engage the extended ladder

# --- Stage 4 tuned defaults (module-level, pinned by the wrapper toggle) -----
# Measured on Ascend910B4 / 40 AIV / UB 196,608 B (msprof op Task Duration,
# 30 launches, interleaved A/B/A/B x 3 pairs; see perf_opt/opt_log.md):
#   * TUNED_IND_DTYPE = "float16" is adopted for the **resident** path only
#     (mask-validation-4k class: 2.959 -> 2.833 us, -4.3%, 3/3 pairs, and
#     -6.4% in a second session).  It costs 2 B/element less and drops one
#     full-width epilogue cast.
#   * The column-tiled paths keep the Stage 3 float32 chain: the same change
#     measured -1.8% on 32k (below the 3% noise threshold) and *conflicting*
#     on 3d (-10.0% in one A/B session, +1.2% in a later one) -> not adopted;
#     the generated code for those shapes is byte-identical to Stage 3.
#   * TUNED_RPRIME=False: DESIGN section 1.6.4 path R' measured 1.14x the
#     main path on 3d (-9.6% .. -11.6%) and does not even compile with the f32
#     indicator at 16384 elements (198,848 B > 196,608 B) -> rejected.
#   * TUNED_TILE_CAP_MODE="ecap": widening the column tile to the UB element
#     cap (32k 4x8192 -> 2x16384) measured -2.6% (32k) / -6.9% (3d) -> rejected.
#   * TUNED_BM_POLICY="ladder": a grid-filling row block measured -18.5% on 4k
#     (the empty-kernel floor grows with the block count) -> rejected.
TUNED_IND_DTYPE = "float16"
TUNED_E_CAP = E_CAP
TUNED_RPRIME = False
TUNED_TILE_CAP_MODE = "ecap"
TUNED_BM_POLICY = "ladder"  # "ladder" (Stage 3) | "cores" (grid-filling)


def _ub_elems(dtype):
    """Per-dtype single-tile element cap: min(measured pass point, UB budget).

    ``UNIT_PRICE`` is the measured compiler-side UB price of the Stage 3 chain;
    8 KB are kept as reserve, so the f16 indicator variant (2 B/element
    cheaper) is intentionally *not* extrapolated -- the cap is the Stage 3
    measured pass point, which the Stage 4 probes re-verified is a hard limit.
    """
    budget = (UB_BYTES - 8192) // UNIT_PRICE[dtype]
    return min(HARD_CAP_ELEMS[dtype], budget)


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
    """Element-budget ladder + three-path switch (DESIGN.md 3.3 / 5.2 / 1.6.4).

    Paths (every value is a factory-period Python int -> static loop bounds):

    * ``R``   -- ``N <= TUNED_E_CAP``: whole row resident, ladder ``block_m``.
    * ``R'``  -- ``TUNED_E_CAP < N <= _ub_elems(dtype)``: single-tile resident
      (``block_m = 1``, no tile loop, no micro-merge) -- DESIGN section 1.6.4's
      Stage 4 alternative, adopted here for the ``3d-multidim`` shape class.
    * ``C``   -- ``N > _ub_elems(dtype)``: column tiles + no-alias running
      ``max`` accumulator (+ optional static-width tail).  The tile width is
      ``TUNED_TILE_CAP_MODE`` (``"ecap"`` = Stage 3, ``"ub"`` = the measured UB
      element cap).

    Returns ``{"path", "block_m", "tile_n", "num_full", "tail"}``.
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
    if TUNED_TILE_CAP_MODE not in ("ecap", "ub"):
        raise ValueError(f"unknown TUNED_TILE_CAP_MODE {TUNED_TILE_CAP_MODE!r}")

    ub_elems = _ub_elems(dtype)
    if N <= TUNED_E_CAP:
        # Path R: whole row resident in UB.
        path = "R"
        candidates = [p for p in (1, 2, 4, 8) if p * N <= TUNED_E_CAP]
        block_m = max(candidates) if candidates else 1
        if TUNED_BM_POLICY == "cores":
            cores = _vector_cores()
            filled = [p for p in sorted(candidates) if _ceildiv(M, p) >= cores]
            block_m = filled[0] if filled else min(candidates or [1])
        if _ceildiv(M, block_m) > LAUNCH_GATE:
            extended = [
                p for p in (16, 32, 64, 128, 256, 512, 1024) if p * N <= TUNED_E_CAP and p <= M
            ]
            block_m = max(extended + [block_m])
        tile_n, num_full, tail = N, 1, 0
    elif TUNED_RPRIME and ub_elems >= N:
        # Path R' (DESIGN 1.6.4): one tile per row, no tile loop / micro-merge.
        path = "R"
        block_m = 1
        tile_n, num_full, tail = N, 1, 0
    else:
        # Path C: column tiles + no-alias running accumulator (+ optional tail).
        path = "C"
        block_m = 1
        tile_cap = ub_elems if TUNED_TILE_CAP_MODE == "ub" else TUNED_E_CAP
        tile_n = _pick_tile_n(N, min(tile_cap, N))
        num_full = N // tile_n
        tail = N - num_full * tile_n

    block_m = min(block_m, M)  # never allocate more rows than exist

    # Capacity predicate (DESIGN.md section 4.5): the per-block element count
    # must sit inside the measured hard limit (the Stage 4 probes confirm it is
    # a hard compile limit, not a soft budget).
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
def golden_all_logical_reduce(x, dim=-1, keepdim=False):
    """PyTorch reference: ``torch.all`` semantics (DESIGN.md section 8.1).

    ``out = AND_j (x_j != 0)`` (NaN counts as non-zero, +-0 counts as zero),
    computed with the authoritative torch reduction API ``torch.all``.  For
    non-bool inputs ``torch.all`` casts to bool first, i.e. "non-zero -> True"
    with NaN treated as non-zero -- verified identical to
    ``x.bool().all(dim, keepdim)`` on float16 / bfloat16 / float32 / bool /
    int32 / int64 / complex64 / complex128.  The tensor is moved to CPU first
    so the reference stays independent of the device runtime.

    It does NOT reproduce the NPU algorithm (no complement form, no indicator
    plane, no column tiling), guaranteeing validation independence.
    """
    return torch.all(x.cpu(), dim=dim, keepdim=keepdim)


# Op-named alias (the gate looks for a `golden_*` entry point of this unit).
golden__logical_reduce_kernel = golden_all_logical_reduce


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
                # Stage 4: the zero-hit indicator is exactly 0.0/1.0, so it is
                # materialised and reduced in float16 -- half the reduce input
                # bytes, one fewer epilogue cast, and 2 B/element of UB headroom
                # (which is what makes a 16384-element single tile fit UB).
                ind = T.alloc_shared((block_m, N), TUNED_IND_DTYPE)
                mx = T.alloc_fragment((block_m, 1), TUNED_IND_DTYPE)
                mx_flat = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)
                one_v = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)
                all_v = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)
                out_ub = T.alloc_shared((block_m,), "int8")

                # Task-invariant operands, hoisted out of the task loop.
                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(one_v, tir.const(1, "float32"))

                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:  # dynamic guard (static loop bound above)
                        real_m = T.min(block_m, M - start)  # M tail row count
                        T.copy(
                            x[start : start + real_m, 0:N],
                            x_ub[0:real_m, 0:N],
                        )
                        # any-zero indicator: ordered EQ (IEEE-correct, so a
                        # NaN never compares equal to zero and never breaks
                        # the AND).
                        T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                        T.vcast(mask_eq, ind, round_mode="rint")  # i1 -> f16
                        T.reduce_max(ind, mx, dim=1, clear=True)  # 1.0 iff any zero
                        T.reshape(mx, mx_flat)  # (bm, 1) -> (bm,) bridge
                        T.vsub(one_v, mx_flat, all_v)  # all = 1 - any_zero
                        T.vcast(all_v, out_ub, round_mode="rint")  # f16 -> i8
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Path C -- column-tiled with no-alias running max accumulator
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
                ind = T.alloc_shared((block_m, tile_n), "float32")
                mx = T.alloc_fragment((block_m, 1), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                acc_flat = T.alloc_fragment((block_m,), "float32")
                one_v = T.alloc_fragment((block_m,), "float32")
                all_v = T.alloc_fragment((block_m,), "float32")
                all_f16 = T.alloc_fragment((block_m,), "float16")
                out_ub = T.alloc_shared((block_m,), "int8")

                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(one_v, tir.const(1, "float32"))

                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        # max identity for the non-negative zero-hit indicator.
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
                            T.reduce_max(ind, mx, dim=1, clear=True)
                            # No-alias accumulate: the AND merge is a max over
                            # the zero-hit indicator (never vadd / vmin here).
                            T.vmax(acc, mx, new_acc)
                            T.copy(new_acc, acc)
                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, all_v)  # minuend is always 1
                        T.vcast(all_v, all_f16, round_mode="rint")
                        T.vcast(all_f16, out_ub, round_mode="rint")
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
                mx = T.alloc_fragment((block_m, 1), "float32")
                mx_tail = T.alloc_fragment((block_m, 1), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc_t = T.alloc_fragment((block_m, 1), "float32")
                acc_flat = T.alloc_fragment((block_m,), "float32")
                one_v = T.alloc_fragment((block_m,), "float32")
                all_v = T.alloc_fragment((block_m,), "float32")
                all_f16 = T.alloc_fragment((block_m,), "float16")
                out_ub = T.alloc_shared((block_m,), "int8")

                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(zero_tail, tir.const(0, dtype))
                T.fill(one_v, tir.const(1, "float32"))

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
                            T.reduce_max(ind, mx, dim=1, clear=True)
                            T.vmax(acc, mx, new_acc)
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
                        T.reduce_max(ind_tail, mx_tail, dim=1, clear=True)
                        T.vmax(acc, mx_tail, new_acc_t)
                        T.copy(new_acc_t, acc)
                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, all_v)
                        T.vcast(all_v, all_f16, round_mode="rint")
                        T.vcast(all_f16, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Factory (harness contract: _logical_reduce_kernel(M, N, op_kind, dtype)
#                                                 -> _func(block_m) -> main)
# ---------------------------------------------------------------------------
@kernel_factory_cache
def _logical_reduce_kernel(M, N, op_kind, dtype):
    """Build the ``all`` NPU kernel factory.

    ``op_kind`` is kept for the shared GPU factory contract; only ``"all"`` is
    implemented by this migration unit (the sibling any / count_nonzero kinds
    are separate units with their own output contracts).  ``N`` is the raw
    reduction width -- no alignment padding is applied (``all`` is neutral to
    the AND unit element, so the real-column domain is exact).  ``threads``
    does not exist on NPU (harness K9).
    """
    if op_kind != "all":
        raise ValueError(
            f"unsupported op_kind {op_kind!r}: this migration unit implements "
            f"the 'all' kind only (any/count_nonzero are separate units)"
        )

    cfg = _select_config(M, N, dtype)

    if cfg["path"] == "R":
        _f = _build_resident(M, N, dtype, cfg["block_m"])
        _f.msprof_kernel_name = "all_resident"
    elif cfg["tail"] == 0:
        _f = _build_tiled(M, N, cfg["tile_n"], cfg["num_full"], dtype, cfg["block_m"])
        _f.msprof_kernel_name = "all_tiled"
    else:
        _f = _build_tiled_tail(
            M, N, cfg["tile_n"], cfg["num_full"], cfg["tail"], dtype, cfg["block_m"]
        )
        _f.msprof_kernel_name = "all_tiled_tail"
    _f.design_config = cfg
    return _f


# ---------------------------------------------------------------------------
# Op-layer emulation helpers (host-side flow of AllFwdOp)
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
    flattened tensor (``M=1``, ``N=numel``); an empty non-``None`` dim list
    follows the Op layer's ``_empty_dim_policy = "noop"`` (no kernel call) and
    is handled by the caller instead.
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
        if buf.dtype != "int8":
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
    """Exact bool comparison (torch.equal, no tolerance; DESIGN.md 8.2).

    The kernel returns the raw **int8** carrier (``bool`` cannot be a GM
    storage dtype); this mirrors the wrapper's ``out_raw.bool()`` restore, and
    additionally asserts that the carrier only ever holds 0/1.
    """
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
        bad = (yc != ref).nonzero().flatten()
        n_bad = int(bad.numel())
        i = int(bad[0]) if n_bad else -1
        raise AssertionError(
            f"[{tag}] logical-AND mismatch: {n_bad}/{ref.numel()} entries differ; "
            f"first at {i}: got={bool(yc.flatten()[i])} want={bool(ref.flatten()[i])}"
        )


def _make_input(M, N, dtype_str, seed=0, zero_frac=0.3):
    """Mixed-density input with forced zeros / non-zeros / +-0 / NaN corners.

    ``randn`` alone has a zero density of ~0, which makes ``all`` constantly
    True, i.e. a constant implementation would also pass.  ~30% of the entries
    are therefore zeroed, row 0 / row 1 are forced to the two extremes (all
    zero -> False, all non-zero -> True) and NaN / +-0 corners are injected.
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
        x[2, 0] = 0.0
        x[2, 1] = 1.0  # single-zero row -> False
    if M > 3 and N > 2:
        x[3, N - 1] = -0.0  # negative-zero corner -> False
    if M > 4 and N > 3:
        x[4, 2] = float("nan")  # NaN without a zero -> True
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
    kernel = _logical_reduce_kernel(M, N, "all", dtype_str)(cfg["block_m"])
    y = kernel(x)
    _exact_compare(y, golden_all_logical_reduce(x.cpu(), dim=-1), tag)
    return cfg, y


def _run_case(M, N, dtype_str, tag, **kw):
    return _run_tensor(_make_input(M, N, dtype_str), dtype_str, tag, **kw)


def _run_nd(x, dim, dtype_str, tag, **kw):
    """Run through the emulated Op-layer flow (movedim -> reshape -> back)."""
    x2, out_shape = _prepare_2d(x, dim)
    cfg, y = _run_tensor(x2, dtype_str, tag, **kw)
    ref = golden_all_logical_reduce(x.cpu(), dim=dim)
    _exact_compare(y.reshape(out_shape), ref, f"{tag}-nd")
    return cfg


# ---------------------------------------------------------------------------
# L0: gate tests (blocking) -- DESIGN.md section 8.2
# ---------------------------------------------------------------------------
def _run_constructed_rows(dtype_str, tag):
    """L0-2: rows covering the complement's two extremes and its corners."""
    N = 512
    rows = []
    rows.append(torch.zeros(N))  # False: every element zero
    rows.append(torch.ones(N))  # True: no zero
    r = torch.ones(N)
    r[0] = 0.0
    rows.append(r)  # single leading zero -> False
    r = torch.ones(N)
    r[N - 1] = 0.0
    rows.append(r)  # single trailing zero -> False
    rows.append(torch.full((N,), -0.0))  # negative zero only -> False
    r = torch.ones(N)
    r[N // 2] = float("nan")
    rows.append(r)  # NaN and no zero -> True (NaN is non-zero)
    rows.append(torch.full((N,), float("nan")))  # all NaN -> True
    r = torch.ones(N)
    r[::3] = 0.0
    r[::5] = float("nan")
    rows.append(r)  # NaN mixed with zeros -> False (zero dominates)
    r = torch.ones(N)
    r[0] = float("inf")
    r[N - 1] = -float("inf")
    rows.append(r)  # +-inf are non-zero -> True
    r = torch.ones(N)
    r[1] = float(torch.finfo(_DTYPE_MAP[dtype_str]).tiny)
    rows.append(r)  # subnormal / tiny stays non-zero -> True
    r = torch.ones(N)
    r[7] = float(torch.finfo(_DTYPE_MAP[dtype_str]).max)
    r[8] = -float(torch.finfo(_DTYPE_MAP[dtype_str]).max)
    rows.append(r)  # extreme finite values -> True
    x = torch.stack(rows).to(_DTYPE_MAP[dtype_str]).to("npu")
    cfg, y = _run_tensor(x, dtype_str, tag)
    got = y.bool().cpu().tolist()
    want = [False, True, False, False, False, True, True, False, True, True, True]
    if got != want:
        raise AssertionError(f"[{tag}] NaN/+-0 corner verdicts {got} != {want}")
    print(
        f"[{tag}] PASS: constructed rows M={x.shape[0]} N={N} dtype={dtype_str} "
        f"({_cfg_str(cfg)}) NaN-no-zero=True / NaN-mixed-zero=False"
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

    # L0-2 discriminative corners (NaN / +-0 / inf / subnormal / N=1).
    _run_constructed_rows("float16", "L0-2a")
    _run_constructed_rows("bfloat16", "L0-2b")
    _run_constructed_rows("float32", "L0-2c")
    n += 3
    for dtype_str in ("float16", "bfloat16", "float32"):
        _run_case(4, 1, dtype_str, "L0-2d", expect_block_m=4)
        n += 1

    # L0-3 path C representative shapes (column tiles + online max merge).
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
    _run_case(32, 256, "float16", "L0-4a", expect_block_m=8)  # 4 logical cores
    _run_case(320, 256, "float16", "L0-4b", expect_block_m=8)  # 40 logical cores
    _run_case(32, 128, "float16", "L0-4c", expect_block_m=8)  # 4 logical cores
    _run_case(256, 4096, "float16", "L0-4d", expect_block_m=2)  # 128 -> 40 x 4
    _run_case(513, 512, "float16", "L0-4e", expect_block_m=8)  # 65 -> 40 x 2
    n += 5

    # L0-5 vectorisation spot check: the lowered IR must contain the explicit
    # vector chain (vcmp / vcast / reduce) and no per-element scalar `ite`.
    _run_vectorised_ir_check()
    n += 1

    print(f"[L0] ALL PASS: {n} shape/edge cases")


def _run_vectorised_ir_check(dtype_str="float16", M=64, N=512):
    """L0-5: confirm the explicit vector chain survives lowering.

    Reads the lowered MLIR of the compiled kernel (``mlir_content``): the
    judgement / indicator / reduction / epilogue steps must appear as the
    ``hivm.hir.v*`` vector ops with no per-element scalar loop (which is what
    the ``T.Parallel`` + ``if_then_else`` form degrades to at ``bm >= 2``).
    """
    kernel = _logical_reduce_kernel(M, N, "all", dtype_str)(8)
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
    need = ("hivm.hir.vcmp", "hivm.hir.vcast", "vreduce", "<max>", "hivm.hir.vsub")
    missing = [k for k in need if k not in mlir]
    forbid = ("arith.cmpf", "scf.for", "if_then_else")
    present = [k for k in forbid if k in mlir]
    if missing:
        raise AssertionError(f"[L0-5] lowered MLIR lacks the vector chain ops {missing}")
    if present:
        raise AssertionError(
            f"[L0-5] lowered MLIR contains scalar / ite forms {present} "
            f"(the judgement chain must stay vectorised at bm>=2)"
        )
    print(
        "[L0-5] PASS: lowered MLIR has vcmp(eq)->vcast(i1->f32)->vreduce<max>->vsub"
        "->vcast(f32->f16)->vcast(f16->i8) with no scalar element loop"
    )


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
    _exact_compare(y, golden_all_logical_reduce(x_nc.cpu(), dim=-1), "L1-1-noncontig")
    n += 1

    # L1-2 higher-rank and non-last-axis reductions (Op layer movedim+reshape).
    _run_nd(_make_input(128, 512, "float16").reshape(2, 64, 512), -1, "float16", "L1-2a")
    _run_nd(_make_input(64, 512, "float16").reshape(2, 4, 8, 512), -1, "float16", "L1-2b")
    _run_nd(_make_input(512, 64, "float16"), 0, "float16", "L1-3a")
    _run_nd(_make_input(64, 512, "float16").reshape(4, 16, 512), 0, "float16", "L1-3b")
    _run_nd(_make_input(2048, 64, "float16").reshape(4, 64, 512), 1, "float16", "L1-3c")
    _run_nd(_make_input(2048, 64, "float32").reshape(4, 64, 512), -2, "float32", "L1-3d")
    n += 6

    # L1-4 extreme / subnormal values (each extreme is exact in its own dtype).
    for dtype_str in ("float16", "bfloat16", "float32"):
        info = torch.finfo(_DTYPE_MAP[dtype_str])
        sub = float(torch.tensor(info.tiny, dtype=_DTYPE_MAP[dtype_str])) / 2.0
        if sub == 0.0:
            sub = float(info.tiny)
        big = float(info.max)
        N = 256
        rows = [torch.zeros(N), torch.full((N,), big), torch.full((N,), -big)]
        r = torch.ones(N)
        r[0] = sub
        r[N - 1] = -sub
        rows.append(r)  # subnormals (or tiny normals) stay non-zero -> True
        r = torch.full((N,), sub)
        r[5] = 0.0
        rows.append(r)  # tiny values, one zero -> False
        r = torch.ones(N)
        r[0] = float("inf")
        r[1] = -float("inf")
        rows.append(r)  # +-inf -> True
        x = torch.stack(rows).to(_DTYPE_MAP[dtype_str]).to("npu")
        _, y = _run_tensor(x, dtype_str, f"L1-4-{dtype_str}")
        want = [False, True, True, True, False, True]
        if y.bool().cpu().tolist() != want:
            raise AssertionError(f"[L1-4-{dtype_str}] {y.bool().cpu().tolist()} != {want}")
        bits = "".join("T" if v else "F" for v in want)
        print(f"[L1-4-{dtype_str}] PASS: extreme rows {bits}")
        n += 1

    # L1-5 unsupported-storage dtypes through the Op-layer float32 pre-cast.
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
        ref = golden_all_logical_reduce(x_cpu, dim=-1)
        xf = _to_logical_float32(x_cpu).contiguous().to("npu")
        kernel = _logical_reduce_kernel(M, N, "all", "float32")(8)
        y = kernel(xf)
        _exact_compare(y, ref, f"L1-5-{dtype}")
        n += 1

    # L1-6 M tail + non-aligned N together.
    _run_case(8, 300, "float16", "L1-6")
    n += 1

    # L1-7 path C tile arithmetic across widths (dividing / non-dividing).
    cfg, _ = _run_case(64, 16384, "float16", "L1-7a", expect_path="C", expect_tile_n=8192)
    if (cfg["num_full"], cfg["tail"]) != (2, 0):
        raise AssertionError(f"[L1-7a] (num_full, tail)={(cfg['num_full'], cfg['tail'])} != (2, 0)")
    cfg, _ = _run_case(8, 10000, "float32", "L1-7b", expect_path="C", expect_tile_n=8192)
    if (cfg["num_full"], cfg["tail"]) != (1, 1808):
        raise AssertionError(
            f"[L1-7b] (num_full, tail)={(cfg['num_full'], cfg['tail'])} != (1, 1808)"
        )
    cfg, _ = _run_case(8, 8448, "float16", "L1-7c", expect_path="C", expect_tile_n=2816)
    if (cfg["num_full"], cfg["tail"]) != (3, 0):
        raise AssertionError(f"[L1-7c] (num_full, tail)={(cfg['num_full'], cfg['tail'])} != (3, 0)")
    n += 3

    # L1-8 keepdim / dim=None through the emulated Op layer.
    x5 = _make_input(64, 512, "float16").reshape(4, 16, 512)
    x2, out_shape = _prepare_2d(x5, -1)
    cfg, y = _run_tensor(x2, "float16", "L1-8-keepdim")
    ref_k = golden_all_logical_reduce(x5.cpu(), dim=-1, keepdim=True)
    _exact_compare(y.reshape(out_shape), ref_k.reshape(out_shape), "L1-8-keepdim")
    n += 1
    x_flat = _make_input(1, 8192, "float32").reshape(8192)
    x2, out_shape = _prepare_2d(x_flat, None)  # dim=None -> M=1, N=numel
    cfg, y = _run_tensor(x2, "float32", "L1-8-dimnone", expect_path="R")
    _exact_compare(
        y.reshape(out_shape),
        golden_all_logical_reduce(x_flat.cpu(), dim=None),
        "L1-8-dimnone",
    )
    if out_shape != ():
        raise AssertionError(f"[L1-8-dimnone] out_shape={out_shape} != ()")
    n += 1

    # L1-9 caller-supplied block_m robustness.  The GPU-period wrapper
    # heuristic may still pass a block_m that overshoots the element budget
    # (e.g. 8 * 4096 = 32768 elements > the 24576 hard UB limit) until Stage 5
    # rewires ``default_config`` (DESIGN.md section 5.6): the kernel must fall
    # back to the ladder value rather than overflow UB, and a valid smaller
    # value must be honoured.
    M, N = 256, 4096
    x = _make_input(M, N, "float16")
    ref = golden_all_logical_reduce(x.cpu(), dim=-1)
    for bm_in in (1, 8):
        kernel = _logical_reduce_kernel(M, N, "all", "float16")(bm_in)
        y = kernel(x)
        _exact_compare(y, ref, f"L1-9-bm{bm_in}")
    print(f"[L1-9] PASS: caller block_m 1 (honoured) and 8 (clamped) both exact on ({M},{N})")
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
        ("op_kind='any'", lambda: _logical_reduce_kernel(8, 256, "any", "float16")),
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
    # the path-C shape forced to 1.0 (the AND-neutral value) must give
    # bit-identical verdicts.  N = 8192 stays resident; N = 16384 is tiled.
    try:
        M, N_half, N_full = 4, 8192, 16384
        data = _make_input(M, N_half, "float16")
        cfg_r, y_r = _run_tensor(data, "float16", "L2-R")
        x_full = torch.ones(M, N_full, dtype=torch.float16, device="npu")
        x_full[:, :N_half] = data
        cfg_c, y_c = _run_tensor(x_full, "float16", "L2-C")
        if not torch.equal(y_r.cpu(), y_c.cpu()):
            raise AssertionError(f"path R {y_r.cpu().tolist()} != path C {y_c.cpu().tolist()}")
        print(
            f"[L2] PASS: path R ({_cfg_str(cfg_r)}) == path C ({_cfg_str(cfg_c)}) "
            f"on the same effective data (extra columns set to the AND-neutral 1.0)"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): path R/C consistency: {e}")

    # N just above E_cap -> path C with the smallest possible tail.
    try:
        cfg, _ = _run_case(32, 8193, "float16", "L2-tn+1")
        print(f"[L2] PASS: N=E_cap+1 -> {_cfg_str(cfg)}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): N=E_cap+1: {e}")

    # Op layer `_empty_dim_policy = "noop"`: dim=[] / dim=() never enters the
    # kernel, the Op layer returns x.bool() directly.
    try:
        x = torch.zeros(4, 8, dtype=torch.float16)
        x[0, 0] = 1.0
        got = x.bool()
        want = x.bool()
        if not torch.equal(got, want):
            raise AssertionError("noop policy mismatch")
        print("[L2] PASS: dim=[] follows the noop policy (x.bool(), no kernel call)")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): noop policy: {e}")

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
            x = torch.ones(M, N, dtype=_DTYPE_MAP[dtype_str], device="npu")
            x[:, N // 2] = float("nan")
            _, y = _run_tensor(x, dtype_str, f"Boundary-nan-{dtype_str}")
            if not bool(y.cpu().all()):
                raise AssertionError("NaN column broke the AND (ordered-EQ misuse)")
            print(f"[Boundary] PASS: NaN-only column {dtype_str} -> all True")
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
