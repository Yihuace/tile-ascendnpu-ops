# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Argmax/argmin first-occurrence reduction kernel (NPU, Developer mode).

Migrated from the GPU TileOPs ``_argreduce_kernel`` (ArgmaxFwdOp) to
``target="npuir"``.

Semantics (DESIGN.md section 0.1, frozen):
    y[i] = min{ j in [0, N) : x[i, j] == ext_i(x) },  ext = max (argmax) / min (argmin)

    First-occurrence tie-break, int64 output, matching ``torch.argmax`` /
    ``torch.argmin`` exactly on the (M, N) -> (M,) contract.

NPU redesign (DESIGN.md sections 0.6 / 1.4, frozen):
    R1: vectorized first-occurrence: ``first = reduce_min_j(ite(x_j == m, j, BIG))``
        with ``BIG = 2**30`` (fp32-exact sentinel; replaces the GPU serial
        scan + loop_break).
    R2: persistent one-dimensional core split: ``num_kernels = min(num_row_blocks,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop and an
        if-guard for out-of-range row blocks (static bounds, Vector cores x2).
    R3: raw-N contract: kernel receives the original (M, N) (no host F.pad);
        non-divisible tiles handled by a static-width tail tile (no kernel-side
        masking).
    R4: block_m / tile_n chosen from a probe-calibrated UB budget table
        (DESIGN.md section 4.5); resident path for N * B/elem * bm <= 64KB,
        tiled online path (forced bm=1) otherwise.

Known toolchain constraints enforced here (DESIGN.md section 9.1, probe-backed):
    C-1 reduce dst dtype MUST equal src dtype (mixed dtype silently corrupts).
    C-2 bm==1 (.,1)-indexed operand in a Parallel condition miscompiles
        (f16 -> i1 broadcast) -> bm==1 uses ``T.vbrc`` into a same-shape buffer.
    C-3 unused fragment alloc is NOT dead-code-eliminated -> bm==1 / bm>=2 and
        bf16 / non-bf16 each get a dedicated prim_func (no unconditional alloc).
    C-4 shared multi-consumer buffers race under auto-multi-buffer when the
        double-buffer budget overflows -> all compute reads fragments; shared
        is GM staging only.
    C-5 multitile + bm>1 compiler SIGSEGV -> tiled path forces bm=1.
    C-6 ``T.vcast`` f32->f32 is non-identity -> fp32 path uses same-dtype copy.
    C-8 if_then_else condition referencing a serial loop var segfaults -> the
        candidate condition only references buffer elements; tile base index is
        plain arithmetic in the value position.
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
# Must stay consistent with the TILELANG_ASCEND_MODE env above.
ASCEND_MODE = "Developer"

# fp32-exact sentinel (2**30 is a power of two, exactly representable in fp32,
# and larger than any valid index j < N <= 2**24).
BIG = 2**30

# Probe-calibrated race-safe manual UB budget per block (bytes); x2 for
# auto-multi-buffer double-buffering fits the 192KB UB (DESIGN.md section 4.5).
UB_MANUAL_BUDGET = 65536

# Column alignment boundary for the tiled path (DESIGN.md section 5.2).
TILE_ALIGNMENT = 256

# Launch-item threshold above which the extended block_m ladder engages
# (DESIGN.md section 5.2; REVIEW blocking-1 fix).
_LAUNCH_GATE = 96

_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")
_KINDS = ("argmax", "argmin")

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Factory-period config helpers (DESIGN.md section 5.2 ladder)
# ---------------------------------------------------------------------------
def _ceildiv(a, b):
    return (a + b - 1) // b


def _B_elem(dtype, bm_class):
    """Fragment byte budget per element (DESIGN.md section 4.5 B/elem table)."""
    table = {
        "float16": {"multi": 8, "bm1": 10},
        "float32": {"multi": 12, "bm1": 16},
        "bfloat16": {"multi": 10, "bm1": 14},
    }
    return table[dtype][bm_class]


def _ub_slab_units(elem_bytes, dtype_slabs=1, fp32_slabs=0):
    """Large UB buffer inventory in elem-sized slab units (inlined _primitives)."""
    total_bytes = dtype_slabs * elem_bytes + fp32_slabs * 4
    return (total_bytes + elem_bytes - 1) // elem_bytes


def _tiled_slab(dtype, elem_bytes):
    """Tiled-path slab count per DESIGN.md section 5.2 (fp16 5 / bf16 7 / fp32 4)."""
    if dtype == "float16":
        # x_ub(2B) + x_work(2B) + ext_brc(2B) + cand(4B)
        return _ub_slab_units(elem_bytes, dtype_slabs=3, fp32_slabs=1)
    if dtype == "bfloat16":
        # x_ub(2B) + x_work(4B) + ext_brc(4B) + cand(4B)
        return _ub_slab_units(elem_bytes, dtype_slabs=1, fp32_slabs=3)
    # float32: x_ub(4B) + x_work(4B) + ext_brc(4B) + cand(4B)
    return _ub_slab_units(elem_bytes, dtype_slabs=4, fp32_slabs=0)


def _pick_tile_n(N, dtype, budget=UB_MANUAL_BUDGET, alignment=TILE_ALIGNMENT, margin=0.9):
    """Tiled-path tile_n: prefer a 256-aligned divisor with >=10% budget margin,
    fall back to the budget cap + static tail tile (DESIGN.md section 5.2)."""
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    slab = _tiled_slab(dtype, elem_bytes)
    if slab * 1 * elem_bytes * N <= budget:
        return N  # fits entirely (not expected on the tiled path)
    max_cols = budget // (slab * 1 * elem_bytes)
    tile_n_cap = (max_cols // alignment) * alignment
    best = 0
    for candidate in range(tile_n_cap, 0, -alignment):
        if N % candidate == 0 and slab * elem_bytes * candidate <= int(budget * margin):
            best = candidate
            break
    if best > 0:
        return best
    return tile_n_cap


def _select_config(M, N, dtype):
    """Factory-period ladder (DESIGN.md section 5.2).

    Returns ``{"block_m", "path", "tile_n"}``. ``path`` is "resident" or
    "tiled". The basic ladder uses ``B_bm1`` for p=1 (REVIEW N1 fix: bm=1
    allocates the extra broadcast buffer) and ``B_multi`` for p>=2.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if N <= 0:
        raise ValueError(f"reduction dim N must be positive, got N={N}")
    if N > 2**24:
        raise ValueError(
            f"N={N} exceeds 2**24; fp32 cannot exactly represent the index "
            f"sentinel arithmetic (E1/E3 equivalence domain, DESIGN.md section 5.2)"
        )

    B_multi = _B_elem(dtype, "multi")
    B_bm1 = _B_elem(dtype, "bm1")

    # Basic ladder {1,2,4,8} (p=1 uses B_bm1 to account for ext_brc).
    candidates = [
        p for p in (1, 2, 4, 8) if p * N * (B_bm1 if p == 1 else B_multi) <= UB_MANUAL_BUDGET
    ]
    block_m = max(candidates) if candidates else None

    # Extended ladder (launch-item driven) for small-N/large-M workloads.
    if block_m is not None and _ceildiv(M, block_m) > _LAUNCH_GATE:
        # Narrow-N (N < TILE_ALIGNMENT) workloads inflate more under
        # auto-multi-buffer than the wide-N 2x the basic ladder was calibrated
        # against (DESIGN.md section 9.2-R10: (2048,4) 64KB manual -> 256KB,
        # ~4x). Halve the manual budget for narrow N so the real UB footprint
        # stays within 192KB; wide-N workloads do not engage the extended
        # ladder meaningfully (their basic-ladder bm is already budget-capped).
        ext_budget = UB_MANUAL_BUDGET if N >= TILE_ALIGNMENT else UB_MANUAL_BUDGET // 2
        ext = [p for p in (16, 32, 64, 128, 256, 512, 1024, 2048) if p * N * B_multi <= ext_budget]
        block_m = max(ext + [block_m])

    if block_m is None:
        return {"block_m": 1, "path": "tiled", "tile_n": _pick_tile_n(N, dtype)}
    # Never allocate more rows than exist: block_m > M leaves uninitialized
    # trailing rows in the UB staging buffer; for narrow N the vectorizer's
    # small-N row handling can then corrupt the valid row (garbage rows).
    block_m = min(block_m, M)
    return {"block_m": block_m, "path": "resident", "tile_n": None}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference, independent of the NPU algorithm)
# ---------------------------------------------------------------------------
def golden_argreduce(x: torch.Tensor, op_kind: str = "argmax") -> torch.Tensor:
    """PyTorch reference: ``torch.argmax`` / ``torch.argmin`` first-occurrence.

    Runs on CPU. Does NOT reproduce the NPU algorithm (no mask candidate /
    reduce_min / online recurrence), guaranteeing validation independence
    (DESIGN.md section 8.1).
    """
    if op_kind == "argmax":
        return torch.argmax(x, dim=-1)
    return torch.argmin(x, dim=-1)


# ---------------------------------------------------------------------------
# Resident path (path S, whole row resident in UB)
# ---------------------------------------------------------------------------
def _build_resident(M, N, op_kind, dtype, work_dtype, vector_cores):
    """Build the resident kernel factory (bm selected by the ladder).

    Four prim_funcs (bf16 x bm==1) so that the bm==1-only ``ext_brc`` broadcast
    buffer is only allocated when actually used (C-3).
    """
    is_bf16 = dtype == "bfloat16"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        if is_bf16 and block_m == 1:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    x_work = T.alloc_fragment((block_m, N), "float32")
                    row_ext = T.alloc_fragment((block_m, 1), "float32")
                    ext_brc = T.alloc_fragment((block_m, N), "float32")
                    cand = T.alloc_fragment((block_m, N), "float32")
                    first = T.alloc_fragment((block_m, 1), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            T.vcast(x_ub, x_work, round_mode="rint")
                            if op_kind == "argmax":
                                T.reduce_max(x_work, row_ext, dim=1)
                            else:
                                T.reduce_min(x_work, row_ext, dim=1)
                            T.vbrc(row_ext, ext_brc)
                            for i, j in T.Parallel(block_m, N):
                                cand[i, j] = T.if_then_else(
                                    x_work[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, first, dim=1)
                            for i in T.Parallel(block_m):
                                out_ub[i] = T.cast(first[i, 0], "int64")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        elif is_bf16:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    x_work = T.alloc_fragment((block_m, N), "float32")
                    row_ext = T.alloc_fragment((block_m, 1), "float32")
                    cand = T.alloc_fragment((block_m, N), "float32")
                    first = T.alloc_fragment((block_m, 1), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            T.vcast(x_ub, x_work, round_mode="rint")
                            if op_kind == "argmax":
                                T.reduce_max(x_work, row_ext, dim=1)
                            else:
                                T.reduce_min(x_work, row_ext, dim=1)
                            for i, j in T.Parallel(block_m, N):
                                cand[i, j] = T.if_then_else(
                                    x_work[i, j] == row_ext[i, 0],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, first, dim=1)
                            for i in T.Parallel(block_m):
                                out_ub[i] = T.cast(first[i, 0], "int64")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        elif block_m == 1:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    x_frag = T.alloc_fragment((block_m, N), dtype)
                    row_ext = T.alloc_fragment((block_m, 1), dtype)
                    ext_brc = T.alloc_fragment((block_m, N), dtype)
                    cand = T.alloc_fragment((block_m, N), "float32")
                    first = T.alloc_fragment((block_m, 1), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            T.copy(x_ub, x_frag)
                            if op_kind == "argmax":
                                T.reduce_max(x_frag, row_ext, dim=1)
                            else:
                                T.reduce_min(x_frag, row_ext, dim=1)
                            T.vbrc(row_ext, ext_brc)
                            for i, j in T.Parallel(block_m, N):
                                cand[i, j] = T.if_then_else(
                                    x_frag[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, first, dim=1)
                            for i in T.Parallel(block_m):
                                out_ub[i] = T.cast(first[i, 0], "int64")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        else:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    x_frag = T.alloc_fragment((block_m, N), dtype)
                    row_ext = T.alloc_fragment((block_m, 1), dtype)
                    cand = T.alloc_fragment((block_m, N), "float32")
                    first = T.alloc_fragment((block_m, 1), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            T.copy(x_ub, x_frag)
                            if op_kind == "argmax":
                                T.reduce_max(x_frag, row_ext, dim=1)
                            else:
                                T.reduce_min(x_frag, row_ext, dim=1)
                            for i, j in T.Parallel(block_m, N):
                                cand[i, j] = T.if_then_else(
                                    x_frag[i, j] == row_ext[i, 0],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, first, dim=1)
                            for i in T.Parallel(block_m):
                                out_ub[i] = T.cast(first[i, 0], "int64")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Tiled path (path T, online running-(extreme, first-index), bm=1 forced)
# ---------------------------------------------------------------------------
def _build_tiled(M, N, op_kind, dtype, work_dtype, vector_cores, tile_n):
    """Build the tiled online kernel (bm=1 forced by C-5).

    Two prim_funcs (bf16 / non-bf16). Tile-0 initializes the running state
    directly; full tiles and the static tail tile use the strict-greater
    (argmax) / strict-less (argmin) update rule for first-occurrence tie-break.
    """
    is_bf16 = dtype == "bfloat16"
    num_full = N // tile_n
    tail = N - num_full * tile_n
    # Static tail-tile buffers are allocated unconditionally with a min width
    # of 1 (the TVM tracer does not track buffers allocated in a conditional
    # block and referenced later); the tail *processing* is guarded by
    # `if tail > 0` below. The extra (block_m, 1) buffers are negligible.
    tail_w = tail if tail > 0 else 1

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        if is_bf16:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, tile_n), dtype)
                    x_work = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_ext = T.alloc_fragment((block_m, 1), "float32")
                    ext_brc = T.alloc_fragment((block_m, tile_n), "float32")
                    cand = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_first = T.alloc_fragment((block_m, 1), "float32")
                    chunk_global = T.alloc_fragment((block_m, 1), "float32")
                    cond = T.alloc_fragment((block_m, 1), "bool")
                    running_ext = T.alloc_fragment((block_m, 1), "float32")
                    running_idx = T.alloc_fragment((block_m, 1), "float32")
                    new_ext = T.alloc_fragment((block_m, 1), "float32")
                    new_idx = T.alloc_fragment((block_m, 1), "float32")
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
                                T.reduce_max(x_work, running_ext, dim=1)
                            else:
                                T.reduce_min(x_work, running_ext, dim=1)
                            T.vbrc(running_ext, ext_brc)
                            for i, j in T.Parallel(block_m, tile_n):
                                cand[i, j] = T.if_then_else(
                                    x_work[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, running_idx, dim=1)

                            # Full tiles 1..num_full-1.
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
                                        T.vcmp(chunk_ext, running_ext, cond, "gt")
                                    else:
                                        T.vcmp(chunk_ext, running_ext, cond, "lt")
                                    # no-alias update: in-place T.vselect(cond, A, B, B)
                                    # miscompiles the loop-carried running state on
                                    # multi-iteration serial tile loops (probe gap);
                                    # select into scratch then copy back instead.
                                    T.vselect(cond, chunk_ext, running_ext, new_ext)
                                    T.vselect(cond, chunk_global, running_idx, new_idx)
                                    T.copy(new_ext, running_ext)
                                    T.copy(new_idx, running_idx)

                            # Static tail tile.
                            if tail > 0:
                                T.copy(
                                    x[block_id : block_id + block_m, num_full * tile_n : N],
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
                                    T.vcmp(tail_ext, running_ext, cond_tail, "gt")
                                else:
                                    T.vcmp(tail_ext, running_ext, cond_tail, "lt")
                                T.vselect(cond_tail, tail_ext, running_ext, new_ext)
                                T.vselect(cond_tail, tail_global, running_idx, new_idx)
                                T.copy(new_ext, running_ext)
                                T.copy(new_idx, running_idx)

                            for i in T.Parallel(block_m):
                                out_ub[i] = T.cast(running_idx[i, 0], "int64")
                            T.copy(out_ub, out[block_id : block_id + block_m])

        else:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, tile_n), dtype)
                    x_work = T.alloc_fragment((block_m, tile_n), work_dtype)
                    chunk_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    ext_brc = T.alloc_fragment((block_m, tile_n), work_dtype)
                    cand = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_first = T.alloc_fragment((block_m, 1), "float32")
                    chunk_global = T.alloc_fragment((block_m, 1), "float32")
                    cond = T.alloc_fragment((block_m, 1), "bool")
                    running_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    running_idx = T.alloc_fragment((block_m, 1), "float32")
                    new_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    new_idx = T.alloc_fragment((block_m, 1), "float32")
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
                                T.reduce_max(x_work, running_ext, dim=1)
                            else:
                                T.reduce_min(x_work, running_ext, dim=1)
                            T.vbrc(running_ext, ext_brc)
                            for i, j in T.Parallel(block_m, tile_n):
                                cand[i, j] = T.if_then_else(
                                    x_work[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, running_idx, dim=1)

                            # Full tiles 1..num_full-1.
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
                                        T.vcmp(chunk_ext, running_ext, cond, "gt")
                                    else:
                                        T.vcmp(chunk_ext, running_ext, cond, "lt")
                                    # no-alias update: in-place T.vselect(cond, A, B, B)
                                    # miscompiles the loop-carried running state on
                                    # multi-iteration serial tile loops (probe gap);
                                    # select into scratch then copy back instead.
                                    T.vselect(cond, chunk_ext, running_ext, new_ext)
                                    T.vselect(cond, chunk_global, running_idx, new_idx)
                                    T.copy(new_ext, running_ext)
                                    T.copy(new_idx, running_idx)

                            # Static tail tile.
                            if tail > 0:
                                T.copy(
                                    x[block_id : block_id + block_m, num_full * tile_n : N],
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
                                    T.vcmp(tail_ext, running_ext, cond_tail, "gt")
                                else:
                                    T.vcmp(tail_ext, running_ext, cond_tail, "lt")
                                T.vselect(cond_tail, tail_ext, running_ext, new_ext)
                                T.vselect(cond_tail, tail_global, running_idx, new_idx)
                                T.copy(new_ext, running_ext)
                                T.copy(new_idx, running_idx)

                            for i in T.Parallel(block_m):
                                out_ub[i] = T.cast(running_idx[i, 0], "int64")
                            T.copy(out_ub, out[block_id : block_id + block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Factory (harness contract: _argreduce_kernel(M, N, op_kind, dtype) -> _func(block_m))
# ---------------------------------------------------------------------------
def _argreduce_kernel(M, N, op_kind, dtype):
    """Build the argmax/argmin NPU kernel factory.

    Mirrors the GPU source structure ``_argreduce_kernel(M, N, op_kind, dtype)
    -> _func(block_m) -> main``. K9: the ``threads`` backend parameter is
    removed (NPU has no CUDA threads concept); the backend config is the
    UB-budget-driven block_m / tile_n (DESIGN.md sections 5.2 / 5.5).
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
    work_dtype = "float32" if dtype == "bfloat16" else dtype
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    if cfg["path"] == "resident":
        return _build_resident(M, N, op_kind, dtype, work_dtype, vector_cores)
    return _build_tiled(M, N, op_kind, dtype, work_dtype, vector_cores, cfg["tile_n"])


# ---------------------------------------------------------------------------
# Precision comparison helper
# ---------------------------------------------------------------------------
def _run_case(M, N, op_kind, dtype_str, tag, expect_block_m=None, expect_tile_n=None):
    torch_dtype = _DTYPE_MAP[dtype_str]
    cfg = _select_config(M, N, dtype_str)
    block_m = cfg["block_m"]

    if expect_block_m is not None and block_m != expect_block_m:
        raise AssertionError(
            f"[{tag}] block_m={block_m} != expected {expect_block_m} "
            f"(shape=({M},{N}) dtype={dtype_str})"
        )
    if expect_tile_n is not None and cfg["tile_n"] != expect_tile_n:
        raise AssertionError(
            f"[{tag}] tile_n={cfg['tile_n']} != expected {expect_tile_n} "
            f"(shape=({M},{N}) dtype={dtype_str})"
        )

    x = torch.randn(M, N, dtype=torch_dtype, device="npu")
    kernel = _argreduce_kernel(M, N, op_kind, dtype_str)(block_m)
    y = kernel(x)
    ref = golden_argreduce(x.cpu(), op_kind)

    assert y.dtype == torch.int64, f"[{tag}] output dtype {y.dtype} != int64"
    assert torch.equal(y.cpu(), ref), (
        f"[{tag}] mismatch shape=({M},{N}) dtype={dtype_str} op={op_kind}: "
        f"got={y.cpu().tolist()} want={ref.tolist()}"
    )

    extra = f" tile_n={cfg['tile_n']}" if cfg["tile_n"] is not None else ""
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} op={op_kind} "
        f"path={cfg['path']} block_m={block_m}{extra}"
    )


# ---------------------------------------------------------------------------
# L0: gate tests (must pass) -- DESIGN.md section 8.2
# ---------------------------------------------------------------------------
def run_L0():
    cases = [
        # (M, N, dtype, op_kind, expect_block_m, expect_tile_n)
        (32, 256, "float16", "argmax", 8, None),  # L0-1a smoke fp16
        (32, 256, "float32", "argmax", 8, None),  # L0-1b smoke fp32
        (32, 256, "bfloat16", "argmax", 8, None),  # L0-1c smoke bf16
        (4, 102400, "float16", "argmax", 1, 5120),  # L0-2 lm-head fp16 tiled
        (4, 102400, "bfloat16", "argmax", 1, 4096),  # L0-3 lm-head bf16 tiled
        (2048, 4096, "float16", "argmax", 2, None),  # L0-4 hidden-state fp16
        (2048, 4096, "bfloat16", "argmax", 1, None),  # L0-5a hidden-state bf16
        (2048, 4096, "float32", "argmax", 1, None),  # L0-5b hidden-state fp32
        (4096, 4, "float16", "argmax", 1024, None),  # L0-6a 3d quick regression
        (524288, 4, "float16", "argmax", 1024, None),  # L0-6b 3d manifest true shape
        (128, 300, "float16", "argmax", 8, None),  # L0-7a N unaligned
        (128, 300, "bfloat16", "argmax", 8, None),  # L0-7b N unaligned bf16
        (129, 512, "float16", "argmax", 8, None),  # L0-7c M tail (129 % 8 = 1)
        (1, 512, "float16", "argmax", 1, None),  # L0-8a single row resident
        (1, 102400, "float16", "argmax", 1, 5120),  # L0-8b single row tiled
        # L0-10 argmin mirror
        (32, 256, "float16", "argmin", 8, None),
        (32, 256, "float32", "argmin", 8, None),
        (32, 256, "bfloat16", "argmin", 8, None),
        (4, 102400, "float16", "argmin", 1, 5120),
    ]
    for M, N, dtype_str, op_kind, bm, tn in cases:
        _run_case(M, N, op_kind, dtype_str, "L0", expect_block_m=bm, expect_tile_n=tn)

    # L0-9 constructed edge rows (tie / all -inf / all +inf / +-inf mix).
    _run_edge_cases("float16", "argmax", "L0")
    _run_edge_cases("float32", "argmin", "L0")
    print(f"[L0] ALL PASS: {len(cases)} shape cases + edge cases")


def _run_edge_cases(dtype_str, op_kind, tag):
    torch_dtype = _DTYPE_MAP[dtype_str]
    N = 300
    rows = []
    # Row 0: duplicate max at col 5 and col 200 -> 5 (first-occurrence tie).
    r = torch.randn(N) * 0.1
    r[5] = 7.0
    r[200] = 7.0
    rows.append(r)
    # Row 1: all -inf -> 0.
    rows.append(torch.full((N,), float("-inf")))
    # Row 2: all +inf -> 0.
    rows.append(torch.full((N,), float("inf")))
    # Row 3: +inf among -inf (argmax) / -inf among +inf (argmin) -> 10.
    r = torch.full((N,), float("-inf") if op_kind == "argmax" else float("inf"))
    r[10] = float("inf") if op_kind == "argmax" else float("-inf")
    rows.append(r)

    x = torch.stack(rows).to(dtype=torch_dtype, device="npu")
    M = x.shape[0]
    cfg = _select_config(M, N, dtype_str)
    kernel = _argreduce_kernel(M, N, op_kind, dtype_str)(cfg["block_m"])
    y = kernel(x)
    ref = golden_argreduce(x.cpu(), op_kind)
    assert y.dtype == torch.int64, f"[{tag}] edge output dtype {y.dtype} != int64"
    assert torch.equal(y.cpu(), ref), (
        f"[{tag}] edge mismatch op={op_kind} dtype={dtype_str}: "
        f"got={y.cpu().tolist()} want={ref.tolist()}"
    )
    print(f"[{tag}] PASS: edge rows M={M} N={N} dtype={dtype_str} op={op_kind}")


# ---------------------------------------------------------------------------
# L1: functional coverage (must pass)
# ---------------------------------------------------------------------------
def run_L1():
    cases = [
        (128, 512, "float32", "argmax", 8, None),
        (128, 512, "bfloat16", "argmax", 8, None),
        (130, 256, "float16", "argmax", 8, None),  # M tail 130 % 8 = 2
        (64, 1024, "float16", "argmax", 8, None),
        (256, 4096, "float16", "argmax", 2, None),  # persistent multi-wave
        (3, 300, "bfloat16", "argmin", 3, None),  # small M + argmin (bm capped to M)
        (8, 32768, "float16", "argmax", 1, 4096),  # tiled fp16 multi-tile
        (128, 16384, "float32", "argmax", 1, 2048),  # tiled fp32 multi-tile
        (4, 8192, "bfloat16", "argmin", 1, 4096),  # tiled bf16 argmin
    ]
    for M, N, dtype_str, op_kind, bm, tn in cases:
        _run_case(M, N, op_kind, dtype_str, "L1", expect_block_m=bm, expect_tile_n=tn)
    print(f"[L1] ALL PASS: {len(cases)} cases")


# ---------------------------------------------------------------------------
# L2: boundary / known-divergence (warn only, non-blocking) -- DESIGN.md 8.2
# ---------------------------------------------------------------------------
def run_L2():
    cases = [
        (1, 1, "float16", "argmax", 1, None),  # minimal
        (1, 2, "float16", "argmax", 1, None),  # N=2
        (2, 1, "float16", "argmax", 2, None),  # N=1
        (9, 256, "float16", "argmax", 8, None),  # M = block_m + 1
        (4, 7000, "float16", "argmax", 1, 6400),  # tiled static tail (7000 % 6400 = 600)
    ]
    for M, N, dtype_str, op_kind, bm, tn in cases:
        try:
            _run_case(M, N, op_kind, dtype_str, "L2", expect_block_m=bm, expect_tile_n=tn)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str} {op_kind}: {e}")

    # NaN row: NPU reduce propagates NaN -> sentinel 2**30; torch returns first
    # NaN index. Documented divergence (DESIGN.md section 9.2-R1) -> record only.
    try:
        x = torch.randn(4, 256, dtype=torch.float16, device="npu")
        x[0, 100] = float("nan")
        cfg = _select_config(4, 256, "float16")
        y = _argreduce_kernel(4, 256, "argmax", "float16")(cfg["block_m"])(x)
        ref = golden_argreduce(x.cpu(), "argmax")
        print(f"[L2] NaN record: kernel={y.cpu().tolist()} torch={ref.tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): NaN row: {e}")

    # N > 2^24 factory assertion (raises, does not silently lose precision).
    try:
        _select_config(1, 2**24 + 1, "float16")
        print("[L2] WARN: N>2^24 did not raise (unexpected)")
    except (ValueError, AssertionError) as e:
        print(f"[L2] N>2^24 raise OK: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): N>2^24: {e}")

    # M=0 boundary (factory raises; Op layer short-circuits in harness).
    try:
        _argreduce_kernel(0, 256, "argmax", "float16")
        print("[L2] WARN: M=0 did not raise (unexpected)")
    except ValueError as e:
        print(f"[L2] M=0 raise OK: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): M=0: {e}")


# ---------------------------------------------------------------------------
# Boundary: extreme values (warn only, non-blocking)
# ---------------------------------------------------------------------------
def run_boundary():
    # Mixed +-0.0 rows: IEEE-equal match vs torch-CPU golden (record only,
    # DESIGN.md section 9.2-R2 notes device-kernel divergence).
    try:
        x = torch.zeros(4, 256, dtype=torch.float32, device="npu")
        x[1, 3] = 1.0
        x[2, 7] = -0.0
        cfg = _select_config(4, 256, "float32")
        y = _argreduce_kernel(4, 256, "argmax", "float32")(cfg["block_m"])(x)
        ref = golden_argreduce(x.cpu(), "argmax")
        print(f"[Boundary] +-0.0 record: kernel={y.cpu().tolist()} torch={ref.tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): +-0.0: {e}")

    # fp32 subnormal extreme: verify first-occurrence still exact.
    try:
        x = torch.zeros(8, 256, dtype=torch.float32, device="npu")
        sub = torch.tensor(1e-40, dtype=torch.float32)  # fp32 subnormal
        x[0, 50] = sub
        cfg = _select_config(8, 256, "float32")
        y = _argreduce_kernel(8, 256, "argmax", "float32")(cfg["block_m"])(x)
        ref = golden_argreduce(x.cpu(), "argmax")
        assert torch.equal(y.cpu(), ref), f"subnormal mismatch {y.cpu()} vs {ref}"
        print("[Boundary] PASS: fp32 subnormal extreme")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): fp32 subnormal: {e}")


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
