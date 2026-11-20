# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Argmin first-occurrence reduction kernel (NPU, Developer mode).

Migrated from the GPU TileOPs ``_argreduce_kernel`` (``ArgminFwdOp``) to
``target="npuir"`` following the frozen ``DESIGN.md`` (v1) in this directory.

Semantics (DESIGN.md section 0.1/1.3, frozen):
    y[i] = min{ j in [0, N) : x[i, j] == min_k x[i, k] }
    First-occurrence tie-break, int64 output, matching ``torch.argmin``.

Input contract (DESIGN.md section 0.2 / 5.6, integration pre-check):
    ``ArgminFwdOp`` keeps ``_kernel_handles_padding = False`` and
    ``_pad_value() = +inf``, so the Op layer pads ``(M, N)`` to
    ``(M, N_padded)`` with ``+inf`` and passes the padded width.  The kernel
    factory derives ``N_padded = align_up(N, 256)`` and DECLARES
    ``(M, N_padded)`` -- declared width must equal the Op-passed width
    (width mismatch symptom: row 0 correct, rows >= 1 wrong, no error).
    ``+inf`` is the argmin unit element; the candidate is evaluated over the
    whole ``N_padded`` domain (E5: index-identical to the real-width scan).

NPU algorithm (DESIGN.md section 0.6 / 1.4, frozen):
    C1 resident two-step vectorized: ``reduce_min(x_frag)`` row extreme ->
    ``vbrc`` broadcast -> ``vcmp "eq"`` mask -> ``vselect(idx_j, BIG)``
    candidate -> ``reduce_min`` first index (explicit vector chain; the 2-D
    ``T.Parallel`` + ``if_then_else`` fused form is scalarized at bm >= 2).
    C2 online single-pass for N that does not fit UB: tile-0 direct write
    initialization + strict-less ``(min, first-idx)`` recurrence with
    no-alias ``vselect`` updates.
    E1 vectorized first index; E2 direct ``reduce_min`` (no ``-max(-x)``);
    E3 online recurrence; E4 fp16 native domain (bf16 pre-cast fp32 because
    ``T.reduce`` does not support bf16; fp32 same-dtype copy -- f32->f32
    ``T.vcast`` is non-identity); E5 ``+inf`` pad index-neutral.

Known toolchain constraints enforced (DESIGN.md section 9.1; entries bound to
910B2C / CANN 8.5.0 are flagged "to re-verify" in the design and are re-tested
by the L0 suite on this device):
    C-1 reduce dst dtype MUST equal src dtype (mixed dtype silently corrupts).
    C-2 bm >= 2 must use the explicit chain (fused 2-D Parallel scalarizes).
    C-3 unused fragment alloc is NOT DCE'd -> dedicated prim_func per
        (bf16 / non-bf16) path.
    C-4 compute reads fragments; shared is GM staging only.
    C-5 tiled path forces bm = 1.
    C-6 f32->f32 vcast is non-identity -> same-dtype copy on the fp32 path.
    C-7 in-place ``vselect(cond, A, B, B)`` loses loop-carried state ->
        no-alias scratch + copy back.
    C-8 same-shape ``vbrc`` emits an empty broadcast dim -> bm == 1 index
        path reshapes (N,) -> (1, N) directly.
"""

import os

# Developer mode: compiler manages UB (alloc_shared) + fragment + auto sync.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

from tileops.kernels._factory_cache import kernel_factory_cache

# Programming mode actually validated by this Stage 3 implementation.
# Must stay consistent with the TILELANG_ASCEND_MODE env above.
ASCEND_MODE = "Developer"

# fp32-exact sentinel (2**30 is a power of two, exactly representable in fp32,
# and larger than any valid index j < N_padded <= 2**24).
BIG = 2**30

# Column alignment boundary for the pad contract (DESIGN.md section 0.2).
DEFAULT_ALIGNMENT = 256
TILE_ALIGNMENT = 256

# Probe-calibrated race-safe manual UB budget per block (bytes); x2 for
# auto-multi-buffer double-buffering fits the 192KB UB (DESIGN.md section 4.5).
UB_MANUAL_BUDGET = 65536

# Launch-item gate above which the extended block_m ladder engages
# (DESIGN.md section 5.2; 2 x 40 vector cores).
_LAUNCH_GATE = 80

# fp32-exact index domain upper bound (DESIGN.md sections 5.2 / 9.2-R11).
_MAX_EXACT_N = 2**24

_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")
_KINDS = ("argmax", "argmin")

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def align_up(n: int, alignment: int) -> int:
    """Round *n* up to the nearest multiple of *alignment*."""
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((n + alignment - 1) // alignment) * alignment


def _ceildiv(a, b):
    return (a + b - 1) // b


def _B_elem(dtype, bm_class):
    """Fragment byte budget per element (DESIGN.md section 4.5 B/elem table).

    ``bm_class`` is "multi" for bm >= 2 or "bm1" for bm == 1 (which also
    carries the ``ext_brc`` broadcast buffer).
    """
    table = {
        "float16": {"multi": 8, "bm1": 10},
        "float32": {"multi": 12, "bm1": 16},
        "bfloat16": {"multi": 10, "bm1": 14},
    }
    return table[dtype][bm_class]


def _ub_slab_units(elem_bytes, dtype_slabs=1, fp32_slabs=0):
    """Large UB buffer inventory in elem-sized slab units (inlined auxiliary)."""
    total_bytes = dtype_slabs * elem_bytes + fp32_slabs * 4
    return (total_bytes + elem_bytes - 1) // elem_bytes


def _tiled_slab(dtype, elem_bytes):
    """Tiled-path slab count per DESIGN.md section 5.2 (fp16 5 / bf16 7 / fp32 4).

    Buffer inventory: x_ub (dtype) + x_work (work dtype) + ext_brc (work
    dtype) + cand (fp32).  The tiled path uses the bm == 1 fused candidate
    form (no idx_j buffer), so no index buffer enters the slab.
    """
    if dtype == "float16":
        # x_ub(2B) + x_work(2B) + ext_brc(2B) + cand(4B)
        return _ub_slab_units(elem_bytes, dtype_slabs=3, fp32_slabs=1)
    if dtype == "bfloat16":
        # x_ub(2B) + x_work(4B) + ext_brc(4B) + cand(4B)
        return _ub_slab_units(elem_bytes, dtype_slabs=1, fp32_slabs=3)
    # float32: x_ub(4B) + x_work(4B) + ext_brc(4B) + cand(4B)
    return _ub_slab_units(elem_bytes, dtype_slabs=4, fp32_slabs=0)


def _pick_tile_n(N_padded, dtype, budget=UB_MANUAL_BUDGET, margin=0.9):
    """Tiled-path tile_n: largest 256-aligned divisor of N_padded that fits
    the 0.9-margin budget; falls back to the budget cap + static tail tile
    (DESIGN.md section 5.2)."""
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    slab = _tiled_slab(dtype, elem_bytes)
    if slab * elem_bytes * N_padded <= budget:
        return N_padded
    max_cols = budget // (slab * elem_bytes)
    cap = (max_cols // TILE_ALIGNMENT) * TILE_ALIGNMENT
    best = 0
    for candidate in range(cap, 0, -TILE_ALIGNMENT):
        if N_padded % candidate == 0 and slab * elem_bytes * candidate <= int(budget * margin):
            best = candidate
            break
    if best > 0:
        return best
    return cap


# ---------------------------------------------------------------------------
# Stage 4 re-tuned ladder (measured 910B3 / CANN 9.0.0)
#
# The Stage 3 proxy budget (64 KB, 8/12/10 B-elem) under-estimated the real
# BishengIR footprint of the explicit chain: at N_padded=4096 the compiler
# requires ~16 B/elem (bm=4 -> "ub overflow, requires 2097408 bits while
# 1572864 bits available"), i.e. the chain's real footprint is ~2x the proxy.
# The Stage 4 ladder is therefore expressed in *measured* bytes per element
# against the real UB limit (192 KB - reserve):
#   fp16  16 B/elem  (x_ub 2 + ext_brc 2 + cmp 1 + idx_j 4 + sent 4 + cand 4,
#                     minus liveness reuse)
#   bf16  16 B/elem  (same, fp32 work copy instead of the fp16 x_ub read;
#                     measured: bm=2 fits, bm=4 overflows)
#   fp32  21 B/elem  (no cast buffers, 4-byte elements)
# Consequence: hidden-state bf16 moves from bm=1 to bm=2 (54.08us -> 36.79us,
# halving the per-core task count), while fp16 keeps bm=2 and fp32 keeps bm=1.
# ---------------------------------------------------------------------------
_B_CHAIN = {"float16": 16, "bfloat16": 16, "float32": 21}
UB_BUDGET_CHAIN = 131072
_LADDER = (1, 2, 4, 8, 16, 32)
_LADDER_EXT = (64, 128, 256, 512, 1024, 2048)
# Physical Vector core count of the deployment device (Ascend910B3 = 20 AIC x 2;
# DESIGN.md section 5.5 on-device query). Used only for the parallelism floor.
_PAR_CORES = 40


def _select_config(M, N, dtype):
    """Factory-period ladder (DESIGN.md section 5.2; Stage 4 recalibrated).

    Returns ``{"block_m", "path", "tile_n"}``; ``path`` is "resident" or
    "tiled".  ``block_m`` is the largest ladder entry whose measured UB
    footprint fits :data:`UB_BUDGET_CHAIN`, then reduced so that the logical
    block count stays near the physical core count.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if N <= 0:
        raise ValueError(f"reduction dim N must be positive, got N={N}")
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    if N_padded > _MAX_EXACT_N:
        raise ValueError(
            f"N_padded={N_padded} exceeds 2**24; fp32 cannot exactly represent "
            f"the index sentinel arithmetic (E1/E3 equivalence domain, "
            f"DESIGN.md section 5.2 / 9.2-R11)"
        )

    B = _B_CHAIN[dtype]
    fit = [p for p in _LADDER if p * N_padded * B <= UB_BUDGET_CHAIN]
    if not fit:
        # N_padded too wide for any resident block -> tiled online path.
        return {"block_m": 1, "path": "tiled", "tile_n": _pick_tile_n(N_padded, dtype)}
    block_m = max(fit)

    # Launch-item driven extension for small-N_padded / large-M shapes.
    if _ceildiv(M, block_m) > _LAUNCH_GATE:
        ext = [p for p in _LADDER_EXT if p * N_padded * B <= UB_BUDGET_CHAIN]
        block_m = max(ext + [block_m])

    # Parallelism floor: keep the logical block count near/above the physical
    # core count when M allows it (never below 8 rows per block for tiny M).
    par_floor = 8
    if M > _PAR_CORES:
        target = 1
        while target * 2 <= _ceildiv(M, _PAR_CORES) and target < block_m:
            target *= 2
        par_floor = max(par_floor, target)
    block_m = min(block_m, par_floor)

    # Never allocate more rows than exist (garbage trailing rows otherwise).
    block_m = max(1, min(block_m, M))
    return {"block_m": block_m, "path": "resident", "tile_n": None}


def _num_vector_cores():
    """Physical Vector core count, queried on the device (DESIGN.md 5.5)."""
    try:
        from tilelang.utils.npu_arch import AscendArch

        return int(AscendArch().aivector_core_num)
    except Exception:  # noqa: BLE001  (fall back to AICore x2)
        from tilelang.utils.npu_utils import NPUUtils

        return int(NPUUtils.get().get_aicore_num()) * 2


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference, independent of the NPU algorithm)
# ---------------------------------------------------------------------------
def golden_argreduce(x: torch.Tensor, op_kind: str = "argmin") -> torch.Tensor:
    """PyTorch reference: ``torch.argmin`` / ``torch.argmax`` first-occurrence.

    Runs on CPU.  Does NOT reproduce the NPU algorithm (no mask candidate /
    reduce_min / online recurrence), guaranteeing validation independence
    (DESIGN.md section 8.1).  The input is the logical ``(M, N)`` tensor
    (pre-pad); the pad is a host-side Op-layer implementation detail.
    """
    if op_kind == "argmin":
        return torch.argmin(x, dim=-1)
    return torch.argmax(x, dim=-1)


# ---------------------------------------------------------------------------
# Resident path (path S, whole row resident in UB) -- DESIGN.md section 3.3
# ---------------------------------------------------------------------------
def _build_resident(M, N_padded, op_kind, dtype, vector_cores, default_bm):
    """Resident path: reduce_min extreme -> vbrc -> vcmp -> vselect -> reduce_min.

    The explicit vector chain is used for both bm == 1 and bm >= 2 (the 2-D
    ``T.Parallel`` + ``if_then_else`` fused form scalarizes at bm >= 2).
    Two dedicated prim_funcs (bf16 / non-bf16) because unused fragment
    allocations are not DCE'd (C-3).
    """
    is_bf16 = dtype == "bfloat16"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        if block_m is None:
            block_m = default_bm
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        if is_bf16:

            @T.prim_func
            def main(
                x: T.Tensor((M, N_padded), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N_padded), dtype)
                    x_work = T.alloc_fragment((block_m, N_padded), "float32")
                    m = T.alloc_fragment((block_m, 1), "float32")
                    ext_brc = T.alloc_fragment((block_m, N_padded), "float32")
                    cmp_eq = T.alloc_fragment((block_m, N_padded), "bool")
                    idx_src = T.alloc_shared((N_padded,), "float32")
                    idx_row = T.alloc_shared((1, N_padded), "float32")
                    idx_j = T.alloc_shared((block_m, N_padded), "float32")
                    sent_v = T.alloc_fragment((block_m, N_padded), "float32")
                    cand = T.alloc_fragment((block_m, N_padded), "float32")
                    first = T.alloc_fragment((block_m, 1), "float32")
                    first_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    # Task-invariant operands hoisted out of the task loop.
                    # 1-D arange -> reshape -> vbrc (the 2-D stride-0 arange
                    # form races on padded-stride layouts; C-8 same-shape vbrc).
                    T.arange(idx_src, [1], 0)
                    if block_m >= 2:
                        T.reshape(idx_src, idx_row)
                        T.vbrc(idx_row, idx_j)
                    else:
                        T.reshape(idx_src, idx_j)
                    T.vbrc(T.float32(BIG), sent_v)

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(
                                x[off_m : off_m + real_m, 0:N_padded],
                                x_ub[0:real_m, 0:N_padded],
                            )
                            T.vcast(x_ub, x_work, round_mode="rint")
                            if op_kind == "argmax":
                                T.reduce_max(x_work, m, dim=1)
                            else:
                                T.reduce_min(x_work, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_work, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, first, dim=1)
                            # Vectorized int64 epilogue: reshape bridges the
                            # (bm, 1) -> (bm,) rank reduction so the cast
                            # auto-vectorizes (rint is identity on the
                            # integer-valued first-index domain).
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        else:

            @T.prim_func
            def main(
                x: T.Tensor((M, N_padded), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N_padded), dtype)
                    x_frag = T.alloc_fragment((block_m, N_padded), dtype)
                    m = T.alloc_fragment((block_m, 1), dtype)
                    ext_brc = T.alloc_fragment((block_m, N_padded), dtype)
                    cmp_eq = T.alloc_fragment((block_m, N_padded), "bool")
                    idx_src = T.alloc_shared((N_padded,), "float32")
                    idx_row = T.alloc_shared((1, N_padded), "float32")
                    idx_j = T.alloc_shared((block_m, N_padded), "float32")
                    sent_v = T.alloc_fragment((block_m, N_padded), "float32")
                    cand = T.alloc_fragment((block_m, N_padded), "float32")
                    first = T.alloc_fragment((block_m, 1), "float32")
                    first_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    # Same in-kernel idx chain as the bf16 variant.
                    T.arange(idx_src, [1], 0)
                    if block_m >= 2:
                        T.reshape(idx_src, idx_row)
                        T.vbrc(idx_row, idx_j)
                    else:
                        T.reshape(idx_src, idx_j)
                    T.vbrc(T.float32(BIG), sent_v)

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(
                                x[off_m : off_m + real_m, 0:N_padded],
                                x_ub[0:real_m, 0:N_padded],
                            )
                            # fp16/fp32 native compute domain (same-dtype copy;
                            # C-6: f32->f32 vcast is non-identity).
                            T.copy(x_ub, x_frag)
                            if op_kind == "argmax":
                                T.reduce_max(x_frag, m, dim=1)
                            else:
                                T.reduce_min(x_frag, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_frag, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, first, dim=1)
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Tiled path (path T, online running-(min, first-idx), bm == 1 forced)
# ---------------------------------------------------------------------------
def _build_tiled(M, N_padded, op_kind, dtype, work_dtype, vector_cores, tile_n):
    """Tiled online kernel (bm == 1 forced by the multitile + bm > 1 trap).

    Tile-0 initializes the running state directly (a BIG sentinel init would
    lose first-occurrence on all-``+inf`` rows under strict-less update);
    full tiles and the static tail tile apply the strict-less recurrence with
    no-alias ``vselect`` updates (in-place carried-state select miscompiles).
    The bm == 1 fused candidate form is used (vectorizes at bm == 1) to keep
    the frozen tile_n budget (no idx_j in the slab).
    """
    num_full = N_padded // tile_n
    tail = N_padded - num_full * tile_n
    # Static tail buffers are allocated unconditionally (the TVM tracer does
    # not track buffers allocated inside a conditional block); the tail
    # *processing* is guarded by `if tail > 0` below.
    tail_w = tail if tail > 0 else 1
    is_bf16 = dtype == "bfloat16"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        block_m = 1  # forced (C-5)
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        if is_bf16:

            @T.prim_func
            def main(
                x: T.Tensor((M, N_padded), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, tile_n), dtype)
                    x_work = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_ext = T.alloc_fragment((block_m, 1), "float32")
                    ext_brc = T.alloc_fragment((block_m, tile_n), "float32")
                    cand = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_first = T.alloc_fragment((block_m, 1), "float32")
                    chunk_global = T.alloc_fragment((block_m, 1), "float32")
                    cond = T.alloc_fragment((block_m, 1), "bool")
                    running_min = T.alloc_fragment((block_m, 1), "float32")
                    running_idx = T.alloc_fragment((block_m, 1), "float32")
                    new_min = T.alloc_fragment((block_m, 1), "float32")
                    new_idx = T.alloc_fragment((block_m, 1), "float32")
                    running_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    x_tail = T.alloc_shared((block_m, tail_w), dtype)
                    xw_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    ext_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    cand_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    tail_ext = T.alloc_fragment((block_m, 1), "float32")
                    tail_first = T.alloc_fragment((block_m, 1), "float32")
                    tail_global = T.alloc_fragment((block_m, 1), "float32")
                    cond_tail = T.alloc_fragment((block_m, 1), "bool")

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            # Tile 0 initializes the running state directly.
                            T.copy(x[block_id : block_id + block_m, 0:tile_n], x_ub)
                            T.vcast(x_ub, x_work, round_mode="rint")
                            if op_kind == "argmax":
                                T.reduce_max(x_work, running_min, dim=1)
                            else:
                                T.reduce_min(x_work, running_min, dim=1)
                            T.vbrc(running_min, ext_brc)
                            for i, j in T.Parallel(block_m, tile_n):
                                cand[i, j] = T.if_then_else(
                                    x_work[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, running_idx, dim=1)

                            # Full tiles 1 .. num_full-1.
                            if num_full > 1:
                                for t in T.serial(num_full - 1):
                                    T.copy(
                                        x[
                                            block_id : block_id + block_m,
                                            (t + 1) * tile_n : (t + 2) * tile_n,
                                        ],
                                        x_ub,
                                    )
                                    T.vcast(x_ub, x_work, round_mode="rint")
                                    if op_kind == "argmax":
                                        T.reduce_max(x_work, chunk_ext, dim=1)
                                    else:
                                        T.reduce_min(x_work, chunk_ext, dim=1)
                                    T.vbrc(chunk_ext, ext_brc)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        cand[i, j] = T.if_then_else(
                                            x_work[i, j] == ext_brc[i, j],
                                            T.cast(j, "float32"),
                                            T.float32(BIG),
                                        )
                                    T.reduce_min(cand, chunk_first, dim=1)
                                    for i in T.Parallel(block_m):
                                        chunk_global[i, 0] = (
                                            T.cast((t + 1) * tile_n, "float32") + chunk_first[i, 0]
                                        )
                                    if op_kind == "argmax":
                                        T.vcmp(chunk_ext, running_min, cond, "gt")
                                    else:
                                        T.vcmp(chunk_ext, running_min, cond, "lt")
                                    # No-alias update: in-place
                                    # T.vselect(cond, A, B, B) loses the
                                    # loop-carried running state on
                                    # multi-iteration serial loops.
                                    T.vselect(cond, chunk_ext, running_min, new_min)
                                    T.vselect(cond, chunk_global, running_idx, new_idx)
                                    T.copy(new_min, running_min)
                                    T.copy(new_idx, running_idx)

                            # Static tail tile.
                            if tail > 0:
                                T.copy(
                                    x[
                                        block_id : block_id + block_m,
                                        num_full * tile_n : N_padded,
                                    ],
                                    x_tail,
                                )
                                T.vcast(x_tail, xw_tail, round_mode="rint")
                                if op_kind == "argmax":
                                    T.reduce_max(xw_tail, tail_ext, dim=1)
                                else:
                                    T.reduce_min(xw_tail, tail_ext, dim=1)
                                T.vbrc(tail_ext, ext_tail)
                                for i, j in T.Parallel(block_m, tail):
                                    cand_tail[i, j] = T.if_then_else(
                                        xw_tail[i, j] == ext_tail[i, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(cand_tail, tail_first, dim=1)
                                for i in T.Parallel(block_m):
                                    tail_global[i, 0] = (
                                        T.cast(num_full * tile_n, "float32") + tail_first[i, 0]
                                    )
                                if op_kind == "argmax":
                                    T.vcmp(tail_ext, running_min, cond_tail, "gt")
                                else:
                                    T.vcmp(tail_ext, running_min, cond_tail, "lt")
                                T.vselect(cond_tail, tail_ext, running_min, new_min)
                                T.vselect(cond_tail, tail_global, running_idx, new_idx)
                                T.copy(new_min, running_min)
                                T.copy(new_idx, running_idx)

                            # Vectorized int64 epilogue.
                            T.reshape(running_idx, running_flat)
                            T.vcast(running_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[block_id : block_id + block_m])

        else:

            @T.prim_func
            def main(
                x: T.Tensor((M, N_padded), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, tile_n), dtype)
                    x_work = T.alloc_fragment((block_m, tile_n), work_dtype)
                    chunk_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    ext_brc = T.alloc_fragment((block_m, tile_n), work_dtype)
                    cand = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_first = T.alloc_fragment((block_m, 1), "float32")
                    chunk_global = T.alloc_fragment((block_m, 1), "float32")
                    cond = T.alloc_fragment((block_m, 1), "bool")
                    running_min = T.alloc_fragment((block_m, 1), work_dtype)
                    running_idx = T.alloc_fragment((block_m, 1), "float32")
                    new_min = T.alloc_fragment((block_m, 1), work_dtype)
                    new_idx = T.alloc_fragment((block_m, 1), "float32")
                    running_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    x_tail = T.alloc_shared((block_m, tail_w), dtype)
                    xw_tail = T.alloc_fragment((block_m, tail_w), work_dtype)
                    ext_tail = T.alloc_fragment((block_m, tail_w), work_dtype)
                    cand_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    tail_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    tail_first = T.alloc_fragment((block_m, 1), "float32")
                    tail_global = T.alloc_fragment((block_m, 1), "float32")
                    cond_tail = T.alloc_fragment((block_m, 1), "bool")

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            # Tile 0 initializes the running state directly.
                            T.copy(x[block_id : block_id + block_m, 0:tile_n], x_ub)
                            T.copy(x_ub, x_work)
                            if op_kind == "argmax":
                                T.reduce_max(x_work, running_min, dim=1)
                            else:
                                T.reduce_min(x_work, running_min, dim=1)
                            T.vbrc(running_min, ext_brc)
                            for i, j in T.Parallel(block_m, tile_n):
                                cand[i, j] = T.if_then_else(
                                    x_work[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, running_idx, dim=1)

                            # Full tiles 1 .. num_full-1.
                            if num_full > 1:
                                for t in T.serial(num_full - 1):
                                    T.copy(
                                        x[
                                            block_id : block_id + block_m,
                                            (t + 1) * tile_n : (t + 2) * tile_n,
                                        ],
                                        x_ub,
                                    )
                                    T.copy(x_ub, x_work)
                                    if op_kind == "argmax":
                                        T.reduce_max(x_work, chunk_ext, dim=1)
                                    else:
                                        T.reduce_min(x_work, chunk_ext, dim=1)
                                    T.vbrc(chunk_ext, ext_brc)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        cand[i, j] = T.if_then_else(
                                            x_work[i, j] == ext_brc[i, j],
                                            T.cast(j, "float32"),
                                            T.float32(BIG),
                                        )
                                    T.reduce_min(cand, chunk_first, dim=1)
                                    for i in T.Parallel(block_m):
                                        chunk_global[i, 0] = (
                                            T.cast((t + 1) * tile_n, "float32") + chunk_first[i, 0]
                                        )
                                    if op_kind == "argmax":
                                        T.vcmp(chunk_ext, running_min, cond, "gt")
                                    else:
                                        T.vcmp(chunk_ext, running_min, cond, "lt")
                                    T.vselect(cond, chunk_ext, running_min, new_min)
                                    T.vselect(cond, chunk_global, running_idx, new_idx)
                                    T.copy(new_min, running_min)
                                    T.copy(new_idx, running_idx)

                            # Static tail tile.
                            if tail > 0:
                                T.copy(
                                    x[
                                        block_id : block_id + block_m,
                                        num_full * tile_n : N_padded,
                                    ],
                                    x_tail,
                                )
                                T.copy(x_tail, xw_tail)
                                if op_kind == "argmax":
                                    T.reduce_max(xw_tail, tail_ext, dim=1)
                                else:
                                    T.reduce_min(xw_tail, tail_ext, dim=1)
                                T.vbrc(tail_ext, ext_tail)
                                for i, j in T.Parallel(block_m, tail):
                                    cand_tail[i, j] = T.if_then_else(
                                        xw_tail[i, j] == ext_tail[i, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(cand_tail, tail_first, dim=1)
                                for i in T.Parallel(block_m):
                                    tail_global[i, 0] = (
                                        T.cast(num_full * tile_n, "float32") + tail_first[i, 0]
                                    )
                                if op_kind == "argmax":
                                    T.vcmp(tail_ext, running_min, cond_tail, "gt")
                                else:
                                    T.vcmp(tail_ext, running_min, cond_tail, "lt")
                                T.vselect(cond_tail, tail_ext, running_min, new_min)
                                T.vselect(cond_tail, tail_global, running_idx, new_idx)
                                T.copy(new_min, running_min)
                                T.copy(new_idx, running_idx)

                            # Vectorized int64 epilogue.
                            T.reshape(running_idx, running_flat)
                            T.vcast(running_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[block_id : block_id + block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Factory (harness contract: _argreduce_kernel(M, N, op_kind, dtype) -> _func)
# ---------------------------------------------------------------------------
@kernel_factory_cache
def _argreduce_kernel(M, N, op_kind, dtype):
    """Build the argmax/argmin NPU kernel factory.

    Mirrors the GPU source structure ``_argreduce_kernel(M, N, op_kind,
    dtype) -> _func(block_m) -> main``.  ``N`` is the original reduction
    length; the kernel declares ``N_padded = align_up(N, 256)`` to match the
    Op-layer pad width.  ``threads`` is removed (NPU has no CUDA threads).
    """
    if op_kind not in _KINDS:
        raise ValueError(f"unsupported op_kind {op_kind!r}; expected one of {sorted(_KINDS)}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if M <= 0:
        raise ValueError(f"M must be positive, got M={M}")

    cfg = _select_config(M, N, dtype)
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    work_dtype = "float32" if dtype == "bfloat16" else dtype
    vector_cores = _num_vector_cores()

    # C6 N-split dispatch (DESIGN.md section 1.6.4 A/B): small-M wide-N shapes
    # otherwise leave the single-kernel tiled path at M/vector_cores occupancy.
    ncfg = _nsplit_config(M, N_padded, dtype, vector_cores)
    if ncfg is not None:
        _f = _build_nsplit_dispatch(
            M, N, op_kind, dtype, work_dtype, vector_cores, ncfg["tn"], ncfg["nchunk"]
        )
        _f.msprof_kernel_name = "argreduce_partial"
        _f.design_config = {"path": "nsplit", **ncfg, "block_m": 1}
        return _f

    if cfg["path"] == "resident":
        _f = _build_resident(M, N_padded, op_kind, dtype, vector_cores, cfg["block_m"])
        _f.msprof_kernel_name = "main"
        _f.design_config = cfg
        return _f

    _f = _build_tiled(M, N_padded, op_kind, dtype, work_dtype, vector_cores, cfg["tile_n"])
    _f.msprof_kernel_name = "argreduce_tiled"
    _f.design_config = cfg
    return _f


# ---------------------------------------------------------------------------
# C6 N-split partial + merge (DESIGN.md section 1.6.4 A/B, Stage 4)
# ---------------------------------------------------------------------------
# Dispatch predicate: M <= 16 rows, fewer row blocks than vector cores, and a
# wide reduction (N_padded >= 32768).  For that class the single-kernel tiled
# path is forced to bm=1 (CG-2026-0014) and leaves only M/40 cores busy.
_NSPLIT_MAX_M = 16
_NSPLIT_MIN_N = 32768
_NSPLIT_MERGE_ELEMS = 6144
# Per-row working-set bytes (x_r + w_r + e_r + c_r) for the manual-budget cap.
_NSPLIT_ROW_B = {"float16": 10, "bfloat16": 14, "float32": 16}
_NSPLIT_MANUAL_CAP = 139264


def _nsplit_config(M, N, dtype, vector_cores):
    """C6 dispatch predicate + (tn, nchunk) derivation; None when ineligible."""
    B_row = _NSPLIT_ROW_B.get(dtype)
    if B_row is None:
        return None
    if M > _NSPLIT_MAX_M or N < _NSPLIT_MIN_N or vector_cores <= M:
        return None
    if N % 256 != 0:
        return None
    max_nc = min(vector_cores, N // 256, _NSPLIT_MERGE_ELEMS // M)
    if max_nc < 2:
        return None
    tn = 256
    while tn * 2 <= N:
        if N % tn == 0:
            nc = N // tn
            if 2 <= nc <= max_nc and tn * B_row <= _NSPLIT_MANUAL_CAP:
                return {"tn": tn, "nchunk": nc}
        tn += 256
    return None


def _build_nsplit_partial(M, N, op_kind, dtype, vector_cores, tn, nchunk):
    """Column-chunk partial kernel: one block per chunk, per-row (1, tn) work."""
    num_kernels = min(nchunk, vector_cores)
    num_local_tasks = _ceildiv(nchunk, num_kernels)
    is_bf16 = dtype == "bfloat16"

    @tilelang.jit(target="npuir")  # side-effect kernel (writes the workspace)
    def _partial(_block_m=None):
        if is_bf16:

            @T.prim_func
            def argreduce_partial(
                x: T.Tensor((M, N), dtype),
                ws_val: T.Tensor((nchunk * M,), "float32"),
                ws_idx: T.Tensor((nchunk * M,), "float32"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_r = T.alloc_shared((1, tn), dtype)
                    w_r = T.alloc_fragment((1, tn), "float32")
                    m_r = T.alloc_fragment((1, 1), "float32")
                    e_r = T.alloc_fragment((1, tn), "float32")
                    c_r = T.alloc_fragment((1, tn), "float32")
                    f_r = T.alloc_fragment((1, 1), "float32")
                    val_out = T.alloc_shared((M,), "float32")
                    gidx_out = T.alloc_shared((M,), "float32")

                    for s in T.serial(num_local_tasks):
                        c = cid + s * num_kernels
                        if c < nchunk:
                            for r in range(M):
                                T.copy(x[r : r + 1, c * tn : (c + 1) * tn], x_r)
                                T.vcast(x_r, w_r, round_mode="rint")
                                if op_kind == "argmax":
                                    T.reduce_max(w_r, m_r, dim=1)
                                else:
                                    T.reduce_min(w_r, m_r, dim=1)
                                T.vbrc(m_r, e_r)
                                for i, j in T.Parallel(1, tn):
                                    c_r[i, j] = T.if_then_else(
                                        w_r[i, j] == e_r[i, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(c_r, f_r, dim=1)
                                val_out[r] = m_r[0, 0]
                                gidx_out[r] = T.cast(c * tn, "float32") + f_r[0, 0]
                            T.copy(val_out[0:M], ws_val[c * M : (c + 1) * M])
                            T.copy(gidx_out[0:M], ws_idx[c * M : (c + 1) * M])

        else:

            @T.prim_func
            def argreduce_partial(
                x: T.Tensor((M, N), dtype),
                ws_val: T.Tensor((nchunk * M,), dtype),
                ws_idx: T.Tensor((nchunk * M,), "float32"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_r = T.alloc_shared((1, tn), dtype)
                    w_r = T.alloc_fragment((1, tn), dtype)
                    m_r = T.alloc_fragment((1, 1), dtype)
                    e_r = T.alloc_fragment((1, tn), dtype)
                    c_r = T.alloc_fragment((1, tn), "float32")
                    f_r = T.alloc_fragment((1, 1), "float32")
                    val_out = T.alloc_shared((M,), dtype)
                    gidx_out = T.alloc_shared((M,), "float32")

                    for s in T.serial(num_local_tasks):
                        c = cid + s * num_kernels
                        if c < nchunk:
                            for r in range(M):
                                T.copy(x[r : r + 1, c * tn : (c + 1) * tn], x_r)
                                T.copy(x_r, w_r)
                                if op_kind == "argmax":
                                    T.reduce_max(w_r, m_r, dim=1)
                                else:
                                    T.reduce_min(w_r, m_r, dim=1)
                                T.vbrc(m_r, e_r)
                                for i, j in T.Parallel(1, tn):
                                    c_r[i, j] = T.if_then_else(
                                        w_r[i, j] == e_r[i, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(c_r, f_r, dim=1)
                                val_out[r] = m_r[0, 0]
                                gidx_out[r] = T.cast(c * tn, "float32") + f_r[0, 0]
                            T.copy(val_out[0:M], ws_val[c * M : (c + 1) * M])
                            T.copy(gidx_out[0:M], ws_idx[c * M : (c + 1) * M])

        return argreduce_partial

    return _partial


def _build_nsplit_merge(M, N, op_kind, dtype, work_dtype, tn, nchunk):
    """Merge kernel: (nchunk, M) -> transpose -> (M, nchunk) tie-break chain."""

    @tilelang.jit(out_idx=[2], target="npuir")
    def _merge():
        @T.prim_func
        def argreduce_merge(
            ws_val: T.Tensor((nchunk, M), work_dtype),
            ws_idx: T.Tensor((nchunk, M), "float32"),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                val_t = T.alloc_shared((nchunk, M), work_dtype)
                val_s = T.alloc_shared((M, nchunk), work_dtype)
                idx_t = T.alloc_shared((nchunk, M), "float32")
                idx_s = T.alloc_shared((M, nchunk), "float32")
                val_f = T.alloc_fragment((M, nchunk), work_dtype)
                idx_f = T.alloc_fragment((M, nchunk), "float32")
                g = T.alloc_fragment((M, 1), work_dtype)
                g_brc = T.alloc_fragment((M, nchunk), work_dtype)
                cmp_eq = T.alloc_fragment((M, nchunk), "bool")
                sent_v = T.alloc_fragment((M, nchunk), "float32")
                cand = T.alloc_fragment((M, nchunk), "float32")
                first = T.alloc_fragment((M, 1), "float32")
                first_flat = T.alloc_fragment((M,), "float32")
                out_ub = T.alloc_shared((M,), "int64")

                T.vbrc(T.float32(BIG), sent_v)
                T.copy(ws_val, val_t)
                T.transpose(val_t, val_s, permutation=[1, 0])
                T.copy(ws_idx, idx_t)
                T.transpose(idx_t, idx_s, permutation=[1, 0])
                T.copy(val_s, val_f)
                T.copy(idx_s, idx_f)
                if op_kind == "argmax":
                    T.reduce_max(val_f, g, dim=1)
                else:
                    T.reduce_min(val_f, g, dim=1)
                T.vbrc(g, g_brc)
                T.vcmp(val_f, g_brc, cmp_eq, "eq")
                T.vselect(cmp_eq, idx_f, sent_v, cand)
                T.reduce_min(cand, first, dim=1)
                T.reshape(first, first_flat)
                T.vcast(first_flat, out_ub, round_mode="rint")
                T.copy(out_ub[0:M], out[0:M])

        return argreduce_merge

    return _merge


def _build_nsplit_dispatch(M, N, op_kind, dtype, work_dtype, vector_cores, tn, nchunk):
    """C6 composition: harness contract preserved, dual launch inside.

    Returns ``_func(block_m) -> main(x) -> out``; ``main`` allocates the flat
    chunk-major workspace per call and launches partial then merge, so the
    wrapper still sees a single callable (argmax Stage 4 form).
    """
    partial = _build_nsplit_partial(M, N, op_kind, dtype, vector_cores, tn, nchunk)(M)
    merge = _build_nsplit_merge(M, N, op_kind, dtype, work_dtype, tn, nchunk)()
    ws_elems = nchunk * M
    val_torch_dtype = _DTYPE_MAP[work_dtype]

    def _func(block_m=None):
        def main(x):
            ws_val = torch.empty(ws_elems, dtype=val_torch_dtype, device=x.device)
            ws_idx = torch.empty(ws_elems, dtype=torch.float32, device=x.device)
            partial(x, ws_val, ws_idx)
            return merge(ws_val, ws_idx)

        return main

    return _func


# ---------------------------------------------------------------------------
# Precision comparison helpers
# ---------------------------------------------------------------------------
def _pad_input(x, op_kind):
    """Simulate the Op-layer pad contract (``_kernel_handles_padding=False``).

    ``+inf`` for argmin / ``-inf`` for argmax (the unit element, DESIGN.md
    section 0.2).  The kernel is declared and called at ``N_padded``.
    """
    M, N = x.shape
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    if N_padded == N:
        return x
    fill = float("inf") if op_kind == "argmin" else float("-inf")
    xp = torch.full((M, N_padded), fill, dtype=x.dtype, device=x.device)
    xp[:, 0:N] = x
    return xp


def _declared_width(kernel):
    """Declared input width of a compiled kernel (prim_func buffer_map)."""
    for buf in kernel.prim_func.buffer_map.values():
        if buf.dtype != "int64":
            return int(buf.shape[-1])
    return None


def _run_tensor(x, op_kind, dtype_str, tag, expect_block_m=None, expect_tile_n=None):
    """Run one case on a logical ``(M, N)`` NPU tensor; exact ``torch.equal``."""
    M, N = x.shape
    cfg = _select_config(M, N, dtype_str)
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
    kernel = _argreduce_kernel(M, N, op_kind, dtype_str)(cfg["block_m"])
    y = kernel(_pad_input(x, op_kind))
    ref = golden_argreduce(x.cpu(), op_kind)
    if y.dtype != torch.int64:
        raise AssertionError(f"[{tag}] output dtype {y.dtype} != int64")
    if not torch.equal(y.cpu(), ref):
        got, want = y.cpu(), ref
        bad = (got != want).nonzero().flatten()
        n_bad = int(bad.numel())
        first = int(bad[0]) if n_bad else -1
        raise AssertionError(
            f"[{tag}] mismatch shape=({M},{N}) dtype={dtype_str} op={op_kind}: "
            f"{n_bad}/{M} rows differ; row {first} got={int(got[first])} "
            f"want={int(want[first])}"
        )
    return cfg


def _run_case(M, N, op_kind, dtype_str, tag, expect_block_m=None, expect_tile_n=None):
    x = torch.randn(M, N, dtype=_DTYPE_MAP[dtype_str], device="npu")
    return _run_tensor(
        x, op_kind, dtype_str, tag, expect_block_m=expect_block_m, expect_tile_n=expect_tile_n
    )


def _run_edge_cases(dtype_str, op_kind, tag):
    """Constructed first-occurrence / extreme-value rows (DESIGN.md L0-9)."""
    N = 300  # non-aligned -> N_padded = 512 (pad-contract coverage)
    big = float("inf") if op_kind == "argmin" else float("-inf")
    small = float("-inf") if op_kind == "argmin" else float("inf")
    rows = []

    r = torch.randn(N) * 0.1
    r[5] = -7.0 if op_kind == "argmin" else 7.0
    r[200] = -7.0 if op_kind == "argmin" else 7.0
    rows.append(r)  # 0: duplicate extreme -> 5

    rows.append(torch.full((N,), big))  # 1: all unit element -> 0
    rows.append(torch.full((N,), small))  # 2: all opposite-inf -> 0

    r = torch.full((N,), big)
    r[10] = small
    rows.append(r)  # 3: single opposite-inf at col 10 -> 10

    r = torch.ones(N)
    r[50] = 1e-40  # fp32 subnormal (rounded to 0 in fp16 -> first index shifts)
    rows.append(r)  # 4: subnormal / underflow row

    rows.append(torch.zeros(N))  # 5: all zeros (mixed-sign zero on the fp32 run)

    r = torch.ones(N)
    r[3] = -5.0 if op_kind == "argmin" else 5.0
    r[259] = r[3]
    rows.append(r)  # 6: cross-tile tie -> 3

    x = torch.stack(rows).to(dtype=_DTYPE_MAP[dtype_str], device="npu")
    _run_tensor(x, op_kind, dtype_str, tag)
    print(f"[{tag}] PASS: edge rows M={x.shape[0]} N={N} dtype={dtype_str} op={op_kind}")


def _run_cross_tile_tie(dtype_str, op_kind, tag):
    """Cross-tile first-occurrence tie on the tiled path (DESIGN.md L0-9)."""
    N = 16384
    dt = _DTYPE_MAP[dtype_str]
    v = -9.0 if op_kind == "argmin" else 9.0
    x = torch.zeros(2, N, dtype=dt, device="npu")
    x[0, 7] = v
    x[0, 4096 + 7] = v  # duplicate extreme in tile 0 and tile 1 -> 7
    x[1, 4096 + 3] = v  # extreme only in a later tile -> 4096 + 3
    cfg = _run_tensor(x, op_kind, dtype_str, tag)
    print(
        f"[{tag}] PASS: cross-tile tie shape=(2,{N}) dtype={dtype_str} "
        f"op={op_kind} path={cfg['path']} tile_n={cfg['tile_n']}"
    )


# ---------------------------------------------------------------------------
# L0: gate tests (blocking) -- DESIGN.md section 8.2
# ---------------------------------------------------------------------------
def run_L0():
    n = 0
    for dtype_str in ("float16", "bfloat16", "float32"):  # L0-1 smoke
        _run_case(32, 256, "argmin", dtype_str, "L0-1", expect_block_m=8)
        n += 1
    _run_case(4, 102400, "argmin", "float16", "L0-2", expect_block_m=1, expect_tile_n=5120)
    _run_case(4, 102400, "argmin", "bfloat16", "L0-3", expect_block_m=1, expect_tile_n=4096)
    _run_case(2048, 4096, "argmin", "float16", "L0-4", expect_block_m=2)
    _run_case(2048, 4096, "argmin", "bfloat16", "L0-5", expect_block_m=2)
    n += 4
    for dtype_str in ("float16", "bfloat16"):  # L0-6 non-aligned + pad width
        _run_case(256, 300, "argmin", dtype_str, "L0-6", expect_block_m=8)
        n += 1
    _run_case(129, 512, "argmin", "float16", "L0-7", expect_block_m=8)  # L0-7 M tail
    n += 1
    _run_case(1, 512, "argmin", "float16", "L0-8a", expect_block_m=1)  # L0-8 single row
    _run_case(1, 102400, "argmin", "float16", "L0-8b", expect_block_m=1, expect_tile_n=5120)
    n += 2
    _run_edge_cases("float32", "argmin", "L0-9a")  # L0-9 constructed rows
    _run_edge_cases("float16", "argmin", "L0-9b")
    _run_cross_tile_tie("float16", "argmin", "L0-9c")
    n += 3
    for dtype_str in ("float16", "bfloat16", "float32"):  # L0-10 argmax mirror
        _run_case(32, 256, "argmax", dtype_str, "L0-10", expect_block_m=8)
        n += 1
    _run_case(4, 102400, "argmax", "float16", "L0-10", expect_block_m=1, expect_tile_n=5120)
    n += 1

    # L0-11: integration width consistency on a non-aligned shape.  A declared
    # width mismatched against the Op-passed width gives row 0 correct / rows
    # >= 1 wrong with no error (VP-2026-0128).
    M, N = 8, 300
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    cfg = _select_config(M, N, "float16")
    kernel = _argreduce_kernel(M, N, "argmin", "float16")(cfg["block_m"])
    width = _declared_width(kernel)
    if width != 512:
        raise AssertionError(f"[L0-11] declared width {width} != N_padded 512 (N=300)")
    y = kernel(_pad_input(x, "argmin"))
    if not torch.equal(y.cpu(), golden_argreduce(x.cpu(), "argmin")):
        raise AssertionError("[L0-11] output mismatch on non-aligned (8,300)")
    n += 1

    print(f"[L0] ALL PASS: {n} shape/edge cases")


# ---------------------------------------------------------------------------
# L1: functional coverage (blocking)
# ---------------------------------------------------------------------------
def run_L1():
    cases = [
        (128, 512, "float32", "argmin", 8),
        (128, 512, "bfloat16", "argmin", 8),
        (130, 256, "float16", "argmin", 8),  # M tail: 130 % 8 = 2
        (64, 1024, "float16", "argmin", 8),
        (256, 4096, "float16", "argmin", 2),  # persistent multi-wave
        (3, 300, "bfloat16", "argmin", 3),  # bm capped to M
        (8, 32768, "float16", "argmin", 1),  # tiled fp16 multi-tile
        (128, 16384, "float32", "argmin", 1),  # tiled fp32 multi-tile
        (4, 8192, "bfloat16", "argmin", 1),  # tiled bf16
    ]
    for M, N, dtype_str, op_kind, bm in cases:
        cfg = _run_case(M, N, op_kind, dtype_str, "L1", expect_block_m=bm)
        print(
            f"[L1] PASS: shape=({M},{N}) dtype={dtype_str} path={cfg['path']} "
            f"block_m={cfg['block_m']} tile_n={cfg['tile_n']}"
        )

    # Non-last-axis dim sentinel: 3-D [4,128,4096] dim=0 -> the Op layer
    # movedim()s the reduction axis to the last position and reshapes to
    # (M, N) = (128*4096, 4).  N=4 -> N_padded=256 (pad contract).
    t = torch.randn(4, 128, 4096, dtype=torch.float16, device="npu")
    x = t.movedim(0, -1).contiguous().reshape(128 * 4096, 4)
    cfg = _run_tensor(x, "argmin", "float16", "L1-dim0")
    print(
        f"[L1] PASS: dim=0 3-D [4,128,4096] -> (M={x.shape[0]}, N=4) "
        f"path={cfg['path']} block_m={cfg['block_m']}"
    )
    del t, x

    # dim=None flatten semantics: Op layer flattens to (M=1, N=numel).
    t = torch.randn(3, 4, 8, dtype=torch.float16, device="npu")
    flat = t.reshape(-1)
    x = flat.reshape(1, flat.numel())
    _run_tensor(x, "argmin", "float16", "L1-none")
    print(f"[L1] PASS: dim=None flatten -> (M=1, N={flat.numel()})")
    print(f"[L1] ALL PASS: {len(cases) + 2} cases")


# ---------------------------------------------------------------------------
# L2: boundary / known-divergence (warn only, non-blocking) -- DESIGN.md 8.2
# ---------------------------------------------------------------------------
def run_L2():
    for M, N, dtype_str in [(1, 1, "float16"), (2, 1, "float16"), (9, 256, "float16")]:
        try:
            _run_case(M, N, "argmin", dtype_str, "L2")
            print(f"[L2] PASS: shape=({M},{N}) dtype={dtype_str}")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str}: {e}")

    # NaN row: NPU reduce_min propagates NaN -> mask match fails -> sentinel
    # 2**30; torch returns the first NaN index.  Documented divergence
    # (DESIGN.md section 9.2-R1) -> record only.
    try:
        x = torch.randn(4, 256, dtype=torch.float16, device="npu")
        x[0, 100] = float("nan")
        cfg = _select_config(4, 256, "float16")
        kernel = _argreduce_kernel(4, 256, "argmin", "float16")(cfg["block_m"])
        y = kernel(_pad_input(x, "argmin"))
        ref = golden_argreduce(x.cpu(), "argmin")
        print(
            f"[L2] NaN record: kernel={y.cpu().tolist()} torch={ref.tolist()} (sentinel BIG={BIG})"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): NaN row: {e}")

    # Factory assertions (raise rather than silently lose precision).
    for label, fn in [
        ("M=0", lambda: _argreduce_kernel(0, 256, "argmin", "float16")),
        ("N=0", lambda: _select_config(1, 0, "float16")),
        ("N>2**24", lambda: _select_config(1, _MAX_EXACT_N + 256, "float16")),
    ]:
        try:
            fn()
            print(f"[L2] WARN: {label} did not raise (unexpected)")
        except ValueError as e:
            print(f"[L2] {label} raise OK: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): {label}: {e}")


# ---------------------------------------------------------------------------
# Boundary: extreme values (warn only, non-blocking)
# ---------------------------------------------------------------------------
def run_boundary():
    # Mixed +-0.0 rows: IEEE-equal match agrees with the torch-CPU golden
    # (torch-NPU device argmin treats signed zero as an ordered pair; the
    # kernel-level golden is CPU, DESIGN.md section 9.2-R2).
    try:
        x = torch.zeros(4, 256, dtype=torch.float32, device="npu")
        x[1, 3] = 1.0
        x[2, 7] = -0.0
        x[3, 5] = -0.0
        x[3, 6] = 0.0
        cfg = _select_config(4, 256, "float32")
        kernel = _argreduce_kernel(4, 256, "argmin", "float32")(cfg["block_m"])
        y = kernel(_pad_input(x, "argmin"))
        ref = golden_argreduce(x.cpu(), "argmin")
        if not torch.equal(y.cpu(), ref):
            print(
                f"[Boundary] WARN (record only): +-0.0 kernel={y.cpu().tolist()} "
                f"torch={ref.tolist()}"
            )
        else:
            print(f"[Boundary] PASS: +-0.0 mixed-sign zero rows -> {y.cpu().tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): +-0.0: {e}")

    # fp32 subnormal as the row minimum: first-occurrence must stay exact.
    try:
        x = torch.ones(8, 256, dtype=torch.float32, device="npu")
        x[0, 50] = 1e-40  # fp32 subnormal
        x[1, 60] = 1e-40
        x[1, 200] = 1e-40  # tie across a wide span -> 60
        cfg = _select_config(8, 256, "float32")
        kernel = _argreduce_kernel(8, 256, "argmin", "float32")(cfg["block_m"])
        y = kernel(_pad_input(x, "argmin"))
        ref = golden_argreduce(x.cpu(), "argmin")
        if not torch.equal(y.cpu(), ref):
            raise AssertionError(f"subnormal mismatch {y.cpu().tolist()} vs {ref.tolist()}")
        print("[Boundary] PASS: fp32 subnormal extreme + wide tie")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): fp32 subnormal: {e}")

    # All-+inf tiled row: the pad unit element in the whole row -> index 0.
    try:
        x = torch.full((2, 32768), float("inf"), dtype=torch.float16, device="npu")
        cfg = _select_config(2, 32768, "float16")
        kernel = _argreduce_kernel(2, 32768, "argmin", "float16")(cfg["block_m"])
        y = kernel(_pad_input(x, "argmin"))
        ref = golden_argreduce(x.cpu(), "argmin")
        if not torch.equal(y.cpu(), ref):
            raise AssertionError(f"all-+inf mismatch {y.cpu().tolist()} vs {ref.tolist()}")
        print(f"[Boundary] PASS: all-+inf tiled rows -> {y.cpu().tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): all-+inf tiled: {e}")


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
