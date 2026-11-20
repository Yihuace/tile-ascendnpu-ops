# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Bitwise-XOR forward kernel (NPU, Developer mode).

Migrated from the GPU TileOPs operator ``bitwise_xor_fwd_kernel``
(``_make_binary_explicit`` factory specialised with ``op_func(a, b) = a ^ b``)
to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y[i] = a[off_a(i)] ^ b[off_b(i)]
on two already-flattened contiguous ``(a_numel,)`` / ``(b_numel,)`` tensors
of the same integer dtype; the output is the flat ``(N_total,)`` view of
``broadcast_shapes(a_shape, b_shape)``.  ``off_a`` / ``off_b`` are the
broadcast index maps; for same-shape inputs they are the identity.

NPU redesign (DESIGN.md section 0.6):
    R1: the source's per-element unrolled divmod chain (used only on the
        broadcast path) is replaced by an outer-row scalar base (one
        unrolled mixed-radix divmod per task) plus innermost-dimension
        vectorisation.
    R2: one-dimensional ``T.Kernel(num_kernels, is_npu=True)`` with a
        persistent in-core ``T.serial`` grid-stride loop and an if-guard,
        replacing the CUDA grid x block two-level parallel structure.
        Element parallelism is handled by the Vector unit via ``T.vxor``.
    R3: ``bool`` / ``uint8`` storage is carried as ``int8`` (bit-identical,
        E1) because npuir cannot carry ``uint8`` GM tensors (CG-2026-0018).
        The Op layer views the public 1-byte tensor as ``int8`` before
        calling the kernel.
    R4: ``T.vxor`` accepts **no** automatic broadcast and **no** scalar /
        size-1 operand (CG-2026-0020 / CG-2026-0019; probes B/C/Q1/E/Q2 in
        ``history_version/``), strictly weaker than its ``T.vand`` /
        ``T.vor`` siblings.  Therefore BOTH operands are explicitly
        materialised to ``(block_size,)`` before the same-shape ``T.vxor``:
        contiguous ``T.copy`` when the innermost stride is 1, ``T.vbrc``
        when the innermost stride is 0 (row-internal scalar broadcast).
        This is the key divergence from the sibling ``bitwise_and`` design,
        which can lean on ``T.vand``'s rank>=2 broadcast.

Implementation notes (toolchain traps found during Stage 3):
    - Direct ``T.vbrc`` with an int8 destination followed by ``T.vxor`` is
      bit-exact (unlike ``T.vand``, whose int8 vbrc hangs the Vector core --
      TRAP-vbrc-int8-vector-core-timeout is *op-specific*, see
      ``repro/TRAP-vbrc-int8-vector-core-timeout-op-specific.py``).  No
      float16 transit is needed on the xor path.
    - The kernel body's ``block_size`` must be the same value used to derive
      the task decomposition (``chunks`` / ``num_logical`` / ``num_kernels``
      / ``num_local_tasks``).  Those are computed *inside* the jit function
      (not at factory-build time) so the returned callable stays consistent
      for every block_size it is invoked with (TRAP-jit-blocksize-
      decomposition).

Factory contract (frozen; see examples/TileOPs/tileops/kernels/elementwise/
bitwise_xor/bitwise_xor.py):
    bitwise_xor_fwd_kernel(N_total, dtype, coalesced_shape, a_strides,
                           b_strides, a_numel, b_numel,
                           output_dtype=None, block_size=4096)
        -> kernel(block_size) -> main(a, b) -> y   (``out_idx=[2]``)

Run the embedded hierarchical tests:
    python bitwise_xor_fwd_kernel.py --level L0
    python bitwise_xor_fwd_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per task, no manual
# sync points (DESIGN.md section 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import math
import time

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = [
    "ASCEND_MODE",
    "bitwise_xor_fwd_kernel",
    "coalesce_broadcast_dims",
    "golden_bitwise_xor",
]

# Programming mode actually validated by this file (TileOPs harness
# migration requires this module-level literal).
ASCEND_MODE = "Developer"

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608
# Auto-multi-buffer inflation factor (CONST-capacity-910B2C).  Round 4 kept
# it only for the 1-D path documentation; the 2-D tile budget and the
# block_size guard now use the measured x2 accounting (see below).
_UB_INFLATION = 1.7

# Element sizes for the supported integer dtypes.
_ELEM_BYTES = {"int8": 1, "int16": 2, "int32": 4, "int64": 8}

# dtypes handled natively by the kernel.  uint8 is deliberately rejected
# (CG-2026-0018) and bool/uint8 must be viewed to int8 by the Op layer (R3).
_ALLOWED_DTYPES = ("int8", "int16", "int32", "int64")

_MAX_2D_L = 8192
_MAX_2D_ELEMS = 16384

# Stage 4 round 8 (v4): upper bound on the packed-int64 pair tile width
# UB footprint coefficient of the pair tile, reverse-engineered from BiShengIR's
# own ub-overflow message: r_pos=3136, pack_words=2, int32 ->
# "requires 1,605,888 bits" = 200,736 B = 8 * 3136 * 2 * 4 + 32 B.  The 8 is
# 4 live tiles (a2, y2, the materialised broadcast row and one temp) times the
# default auto-multi-buffer factor 2.  With auto-MB off the coefficient is 4.
_PAIR_UB_COEF_MB = 8
_PAIR_UB_COEF_NO_MB = 4
# Auto-multi-buffer for the pair tile (measured in round 8; see TUNED comment).
_PAIR_MULTI_BUFFER = False

_DEFAULT_BLOCK = 4096

# Stage 4 tuned execution parameters (per-dtype block_size).  Exposed as
# module-level constants so the Op-layer wrapper source-selection block can
# adopt the SAME values it compiles against (tilelang-op-optimize Phase 3).
# Stage 4 round 7 (v3 int32-pack): block_size of the *host-side int32-packed*
# 1-byte path.  The packed operand is a real int32 kernel input, so the int32
# UB budget applies and the bool-carrier 16384 would be rejected by the factory
# guard.  4096 = `TUNED_BLOCK_SIZE`, the tuned int32 value.  XOR needs no r_blk
# change: its 2-D tile is sized from the measured UB budget (2*3*r_blk*L*E), so
# the packed shape (L = 784 int32) already admits r_blk = 8 automatically.
TUNED_PACK_BLOCK_SIZE = 4096

TUNED_BLOCK_SIZE = 4096  # non-1-byte dtypes (int16/int32/int64)
TUNED_BOOL_BLOCK_SIZE = 16384  # int8 storage carrier (bool); Round 4 sweep
# Round 4 measured sweep on hidden-state-prefill bool (1-D path, N=8.39e6):
#   bs=8192 -> 17.74 us / Ratio 52.54%   (round-3 value)
#   bs=16384 -> 14.94 us / Ratio 62.39%
#   bs=32768 -> 14.86 us / Ratio 62.73%  (tie with 16384, but 32768 sits exactly
#              at the UB capacity: 3 x 2 x 32768 = 196,608 B)
# 16384 is adopted: same latency within noise, half the UB footprint.

# Full-size UB bytes per element, worst case: three block-sized buffers
# (a_ub / b_ub / y_ub) of the operand dtype.  The two 1-element scalar
# sources are negligible.  Unlike the bitwise_and sibling there is no float16
# transit on the xor path (direct int8 T.vbrc works, R4), so the budget is
# the tighter, more honest 3 x E.
_UB_BYTES_PER_ELEM = {
    "int8": 3 * 1,
    "int16": 3 * 2,
    "int32": 3 * 4,
    "int64": 3 * 8,
}

# Upper bound on block_size from the UB budget.  Round 4 replaced the
# conservative x1.7 inflation estimate (DESIGN.md section 4.5) with the
# measured compiler accounting: BiShengIR double-buffers the three loop-live
# block buffers, so the demand is exactly 2 x 3 x E x block_size bytes
# (verified by the "ub overflow, requires ... while 1572864 bits available"
# probes: 2 x 3 x 2 x 3136 x 4 = 150,528 B passes, 2 x 3 x 4 x 3136 x 4 =
# 301,056 B fails).  Hence the exact guard below:
#   196608 / (3 x E x 2) -> int8 32768 / int16 16384 / int32 8192 / int64 4096.
_UB_MAX_BLOCK = {
    dt: int(_UB_CAPACITY_BYTES // (_UB_BYTES_PER_ELEM[dt] * 2)) for dt in _ALLOWED_DTYPES
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_bitwise_xor(a, b):
    """Bitwise-XOR reference (source-operator semantics, direct broadcast).

    Matches the source-repo test benchmark
    (examples/TileOPs/tests/ops/test_bitwise_xor.py: ``ref =
    torch.bitwise_xor(a, b)``; integer/bool compared exactly).  Inputs keep
    their original broadcast shapes; the output shape is
    ``broadcast_shapes(a.shape, b.shape)``.  Independent of the NPU
    algorithm (no row/inner indexing, no vbrc, no persistent structure, no
    int8 carrier) and runs on CPU without an NPU device.
    """
    return torch.bitwise_xor(a.detach().cpu(), b.detach().cpu())


# Alias kept so the generic ``golden_{op}`` gate name resolves for
# op_name == "bitwise_xor_fwd_kernel" as well.
golden_bitwise_xor_fwd_kernel = golden_bitwise_xor


# ---------------------------------------------------------------------------
# Broadcast lowering (verbatim from the source operator's helper)
# ---------------------------------------------------------------------------
def coalesce_broadcast_dims(a_shape, b_shape):
    """Coalesce N-dim broadcast into minimal effective dimensions.

    Merges adjacent dimensions that have the same broadcast behaviour
    (both real or both broadcast) to minimise the number of divmod
    operations inside the kernel body.  This is a lowering decision, not
    part of the op contract: the caller hands the factory two operand
    shapes and the kernel picks its own index representation.

    Returns ``(out_shape, coalesced_shape, a_strides, b_strides)`` where
    strides use 0 for broadcast dimensions.
    """
    if len(a_shape) == 0:
        a_shape = (1,)
    if len(b_shape) == 0:
        b_shape = (1,)

    out_shape = torch.broadcast_shapes(a_shape, b_shape)
    ndim = len(out_shape)
    a_pad = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
    b_pad = (1,) * (ndim - len(b_shape)) + tuple(b_shape)

    def _make_strides(padded_shape):
        strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * padded_shape[i + 1]
        return [0 if padded_shape[i] == 1 and out_shape[i] > 1 else strides[i] for i in range(ndim)]

    a_raw = _make_strides(a_pad)
    b_raw = _make_strides(b_pad)

    groups = [(out_shape[0], a_raw[0], b_raw[0])]
    for i in range(1, ndim):
        prev_out, prev_as, prev_bs = groups[-1]
        a_can = (a_raw[i] == 0 and prev_as == 0) or (
            a_raw[i] != 0 and prev_as == a_raw[i] * out_shape[i]
        )
        b_can = (b_raw[i] == 0 and prev_bs == 0) or (
            b_raw[i] != 0 and prev_bs == b_raw[i] * out_shape[i]
        )
        if a_can and b_can:
            groups[-1] = (prev_out * out_shape[i], a_raw[i], b_raw[i])
        else:
            groups.append((out_shape[i], a_raw[i], b_raw[i]))

    groups = [g for g in groups if g[0] > 1] or [(1, 0, 0)]
    coalesced_shape = tuple(g[0] for g in groups)
    a_strides = tuple(g[1] for g in groups)
    b_strides = tuple(g[2] for g in groups)
    return out_shape, coalesced_shape, a_strides, b_strides


# ---------------------------------------------------------------------------
# Outer-row scalar base (compile-time unrolled mixed-radix divmod)
# ---------------------------------------------------------------------------
def _row_base(row, outer_shape, strides):
    """Build the outer-row base offset expression for one operand.

    ``row`` is the runtime row index; ``outer_shape`` and ``strides`` are
    Python constants, so the loop unrolls at trace time into a chain of
    ``T.floordiv`` / ``T.floormod`` ops (one divmod per outer dimension,
    executed once per task rather than once per element -- DESIGN.md R1,
    ~L times cheaper than the source per-element chain).
    Returns the Python int 0 when there is no outer dimension (k == 1).
    """
    base = 0
    rem = row
    n_outer = len(outer_shape)
    for j in range(n_outer):
        div = 1
        for t in range(j + 1, n_outer):
            div *= outer_shape[t]
        coord = T.floordiv(rem, div)
        rem = T.floormod(rem, div)
        base = base + coord * strides[j]
    return base


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------
def _normalize_dtype(dtype):
    """Return the TileLang dtype string, rejecting unsupported dtypes."""
    name = str(dtype).replace("torch.", "") if isinstance(dtype, torch.dtype) else str(dtype)
    if name == "uint8":
        raise ValueError(
            "uint8 GM tensors are not supported on npuir (CG-2026-0018); "
            "load the bool/uint8-storage path through int8 instead"
        )
    if name == "bool":
        raise ValueError(
            "bool tensors must be viewed to int8 by the Op layer before "
            "reaching the kernel (DESIGN.md R3)"
        )
    if name not in _ALLOWED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {_ALLOWED_DTYPES}")
    return name


def bitwise_xor_fwd_kernel(
    N_total,
    dtype,
    coalesced_shape,
    a_strides,
    b_strides,
    a_numel,
    b_numel,
    output_dtype=None,
    block_size=_DEFAULT_BLOCK,
    pack_words=1,
):
    """Build the bitwise-XOR kernel factory.

    Returns a JIT callable taking a single ``block_size`` which compiles and
    returns ``main``; ``main(a, b)`` returns ``y`` (``out_idx=[2]``,
    source-operator contract).
    """
    dtype = _normalize_dtype(dtype)
    out_dtype = _normalize_dtype(output_dtype) if output_dtype is not None else dtype
    coalesced_shape = tuple(int(x) for x in coalesced_shape)
    a_strides = tuple(int(x) for x in a_strides)
    b_strides = tuple(int(x) for x in b_strides)
    N_total = int(N_total)
    a_numel = int(a_numel)
    b_numel = int(b_numel)

    if N_total <= 0:
        raise ValueError(f"N_total must be a positive int, got {N_total}")
    if (
        len(coalesced_shape) == 0
        or len(a_strides) != len(coalesced_shape)
        or len(b_strides) != len(coalesced_shape)
    ):
        raise ValueError("coalesced_shape / a_strides / b_strides rank mismatch")

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op).  Re-queried per factory call so other deployments
    # adapt automatically (this machine: 20 AI cores -> 40 Vector cores).
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    if pack_words not in (1, 2):
        raise ValueError(f"pack_words must be 1 or 2, got {pack_words}")

    # ---- Stage 4 round 8 (v4): packed-int64 pair tile --------------------
    # A host-side int64 -> int32 split turns every element into two words
    # [low, high].  When the broadcast operand is *unit* on the last dim
    # (channel / scalar broadcast) its 64-bit value has to be repeated along
    # that whole row, i.e. the [low, high] pair must be broadcast across the
    # pair lane -- a 2-D [r_blk, 1] tile can only broadcast a single word.
    #
    # Geometry (R_lead, run, npos, W) with W = 2*K:  a/y are word rows of W
    # words (= K element positions, kept contiguous in memory) and b is
    # (run, W) -- one per-channel row holding the [low, high] pair repeated K
    # times.  The op is then the ordinary rank-2 "[M, N] op [1, N]" broadcast
    # the row tile already uses, and the DMAs are K*8-byte contiguous rows.
    # K = 1 (W = 2) degenerates to 8-byte rows and is DMA-hostile (measured
    # 13.9 ms on cnn-feat-4d vs 0.196 ms for the int64 path), which is why
    # the host folds K positions into every tile row.
    if pack_words == 2:
        if len(coalesced_shape) != 4 or coalesced_shape[-1] % pack_words != 0:
            raise ValueError(
                "packed pair geometry must be (R_lead, run, npos, W) with "
                f"W = 2*K, got {coalesced_shape}"
            )
        R_lead, run_p, npos, W = (int(x) for x in coalesced_shape)
        if min(R_lead, run_p, npos, W) <= 0:
            raise ValueError(f"packed pair geometry must be positive, got {coalesced_shape}")
        K = W // pack_words
        if npos % K != 0:
            raise ValueError(f"npos={npos} must be a multiple of K={K}")
        if R_lead * run_p * npos * pack_words != N_total:
            raise ValueError(
                f"N_total={N_total} != R_lead*run*npos*{pack_words}"
                f"={R_lead * run_p * npos * pack_words}"
            )
        if b_numel != run_p * W:
            raise ValueError(f"b_numel={b_numel} != run*W={run_p * W}")
        nrows_total = npos // K  # tile rows per (lead, channel)
        # Tile rows per task: as many as the UB allows.  Footprint =
        # coef * G * W * E (+32 B), see ``_PAIR_UB_COEF_MB``.
        # b2 is materialised (vxor has no broadcast), so the tile needs
        # a2 + b1 + b2 + y2 live buffers and doubles: measured demand is
        # 8 * tile + one b row (r=8, W=784 int32 -> "requires 1,630,720 bits"
        # = 203,840 B = 8*25,088 + 3,136), i.e. coef = 8 either way.
        coef = 8
        cap = (_UB_CAPACITY_BYTES - 1024) // (coef * W * _ELEM_BYTES[dtype])
        rows_per_task = 1
        for cand in range(1, min(nrows_total, cap) + 1):
            if nrows_total % cand == 0:
                rows_per_task = cand
        nchunks = nrows_total // rows_per_task
        num_logical_pair = R_lead * run_p * nchunks
        num_kernels_pair = min(num_logical_pair, vector_cores)
        num_local_pair = T.ceildiv(num_logical_pair, num_kernels_pair)

        @tilelang.jit(
            out_idx=[2],
            target="npuir",
            pass_configs={"npuir.enable_auto_multi_buffer": _PAIR_MULTI_BUFFER},
        )
        def kernel_pair(block_size=block_size):
            # block_size is unused by the pair tile (its width is fixed by the
            # fold); the argument only keeps the K11 call contract identical
            # to the element-wise form.
            @T.prim_func
            def main_pair(
                a: T.Tensor((R_lead * run_p * nrows_total, W), dtype),
                b: T.Tensor((run_p, W), dtype),
                y: T.Tensor((R_lead * run_p * nrows_total, W), out_dtype),
            ):
                with T.Kernel(num_kernels_pair, is_npu=True) as (cid, _):
                    a2 = T.alloc_ub((rows_per_task, W), dtype)
                    b1 = T.alloc_ub((1, W), dtype)
                    b2 = T.alloc_ub((rows_per_task, W), dtype)
                    y2 = T.alloc_ub((rows_per_task, W), out_dtype)
                    for i in T.serial(num_local_pair):
                        task = i * num_kernels_pair + cid
                        if task < num_logical_pair:
                            blk = task // nchunks
                            sub = task % nchunks
                            row0 = blk * nrows_total + sub * rows_per_task
                            # per-channel broadcast row ([low, high] x K)
                            # T.vxor has no automatic broadcast (CG-2026-0019/
                            # 0020), so the [low, high] row is materialised to
                            # the full tile shape with T.vbrc first.
                            T.copy(b[blk % run_p, 0], b1)
                            T.vbrc(b1, b2)
                            T.copy(a[row0, 0], a2)
                            T.vxor(a2, b2, y2)
                            T.copy(y2, y[row0, 0])

            return main_pair

        return kernel_pair

    L = coalesced_shape[-1]
    outer = list(coalesced_shape[:-1])
    R = math.prod(outer) if outer else 1
    sa, sb = a_strides[-1], b_strides[-1]
    if sa not in (0, 1) or sb not in (0, 1):
        raise ValueError(f"innermost strides must be 0 or 1, got sa={sa}, sb={sb}")

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(block_size=block_size):
        # Host-side UB budget guard (DESIGN.md section 4.5).
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        max_block = _UB_MAX_BLOCK[dtype]
        if block_size > max_block:
            raise ValueError(
                f"block_size={block_size} exceeds the UB budget for {dtype} "
                f"(max {max_block} elements, {_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # block_size-dependent task decomposition, evaluated here (not at
        # factory time) so the compiled kernel always matches the block_size
        # it is invoked with -- the returned callable is cached per
        # block_size (wrapper K11 contract; see TRAP-jit-blocksize-
        # decomposition).  block_size folds to a compile-time constant.
        chunks = T.ceildiv(L, block_size)
        # Static chunk extent when the decomposition is exact (the row divides
        # the block, or the whole row is one chunk) -- a trace-time constant
        # makes every GM<->UB burst a static-extent DMA (+4.7 %-points on
        # hidden-state-prefill-bool).  It is only taken when the 1-D path's
        # worst-case UB footprint still fits: the measured demand is 7 tiles
        # (3 operand buffers x auto-multi-buffer + 1 temp), so int64 at
        # block_size 4096 (7 * 32 KB = 229,376 B) keeps the dynamic extent.
        # The buffers stay block_size-wide (a degenerate tiny L must not shrink
        # them: a 1-element tile makes the emitted MLIR fail verification).
        _static_tail = 0
        if 7 * block_size * _ELEM_BYTES[dtype] <= _UB_CAPACITY_BYTES:
            _static_tail = block_size if L % block_size == 0 else (L if block_size > L else 0)
        num_logical = R * chunks
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = T.ceildiv(num_logical, num_kernels)

        # Stage 4: an operand broadcast across *every* outer dim
        # (all outer strides 0) that fits in a single chunk is the SAME
        # row for every task -- hoist its GM->UB load out of the serial
        # loop instead of re-reading it once per task (row-broadcast).
        hoist_a = sa == 1 and chunks == 1 and all(s == 0 for s in a_strides[:-1])
        hoist_b = sb == 1 and chunks == 1 and all(s == 0 for s in b_strides[:-1])

        # --- [R_blk, L] 2-D tile -------------------------------------------
        # T.vxor 无自动广播（CG-2026-0019/0020），因此 b 先用 T.vbrc 物化成
        # 整块 (r_blk, L)（与 ge/gt/le 对 T.vcmp 的做法一致），再做同形 vxor。
        run = outer[-1] if outer else 1
        b_2d_col = (
            sb == 0
            and len(outer) >= 1
            and len(b_strides) >= 2
            and b_strides[-2] == 1
            and b_numel == run
        )
        b_2d_row = sb == 1 and all(x == 0 for x in b_strides[:-1]) and b_numel == L
        # Round 4 (H2, branch op1): open the 2-D path to the 1-byte (int8
        # carrier = torch.bool storage) dtype, and size the tile from the
        # MEASURED compiler budget instead of the literal 2.
        # Measured accounting (round-3 op1 probe, BiShengIR "ub overflow,
        # requires 2408704 bits while 1572864 bits available" at r_blk=4 int32):
        #   demand = 2 x (a2 + b2 + y2)  = 2 x 3 x r_blk x L x E  bytes
        # because the loop-carried tile buffers are double-buffered -- this,
        # not the r_blk literal, is what caps int32 at r_blk=2 today.
        # For int8 the same budget admits r_blk = 8 on cnn-feat (L=3136:
        # 2x3x8x3136 = 150,528 B <= 196,608 B), which cuts the per-task
        # emission count (run/r_blk -> 32 tiles instead of 4096 rows) -- the
        # dominant cost of the 10.6%-Ratio int8 case.
        # int64 stays excluded: its vbrc-materialised 2-D tile timed out the
        # Vector core (historical, unchanged) and 2x3x2xLx8 does not fit for
        # L>=4096 anyway.
        _e_bytes = _ELEM_BYTES[dtype]
        # Stage 4 round 8: r_blk stays at the 3-buffer budget below -- the
        # hoisted form was probed at r_blk=4 (2 x 2 tiles) and is *slower*
        # (48.8 us vs 47.8 us at r_blk=2 on cnn-feat-broadcast-int32), so the
        # smaller tile with more tasks wins.
        max_2d_elems = int(_UB_CAPACITY_BYTES // (2 * 3 * _e_bytes))
        max_2d_r = max_2d_elems // max(L, 1)
        r_blk = 2
        if b_2d_col or b_2d_row:
            r_blk = 0
            for cand in range(max_2d_r, 1, -1):
                if run % cand == 0:
                    r_blk = cand
                    break
        R_lead = R // run if run else 1
        # 实测：int64 的 vbrc 物化 + 2-D 仍会 Vector 核超时 -> 仅 int8/int32
        use_2d = bool(
            dtype in ("int8", "int32")
            and (b_2d_col or b_2d_row)
            and len(outer) >= 1
            and sa == 1
            and a_strides[-2] == L
            and a_numel == R * L
            and L <= _MAX_2D_L
            and L % 16 == 0
            and (L * _e_bytes) % 32 == 0
            and 2 <= r_blk <= run
            and run % r_blk == 0
            and r_blk * L <= max_2d_elems
        )

        if use_2d and b_2d_col:
            # Stage 4 round 8: b is *materialised* per task in this form
            # (vxor has no broadcast), so keep the channel group fixed in the
            # outer loop and iterate the leads inside it: T.vbrc then runs
            # once per (core, group) instead of once per task, i.e. R_lead = 16
            # times fewer vector passes on cnn-feat (measured vec ratio 0.61
            # before, vs 0.19 for the broadcast-capable vand).
            groups_per_lead = run // r_blk
            num_subs = groups_per_lead
            num_kernels_c = min(num_subs, vector_cores)
            subs_per_core = T.ceildiv(num_subs, num_kernels_c)

            @T.prim_func
            def main2c(
                a: T.Tensor((R, L), dtype),
                b: T.Tensor((run, 1), dtype),
                y: T.Tensor((R, L), out_dtype),
            ):
                with T.Kernel(num_kernels_c, is_npu=True) as (cid, _):
                    a2 = T.alloc_ub((r_blk, L), dtype)
                    b2 = T.alloc_ub((r_blk, L), dtype)  # 物化后的整块
                    b_c = T.alloc_ub((r_blk, 1), dtype)
                    y2 = T.alloc_ub((r_blk, L), out_dtype)
                    for s in T.serial(subs_per_core):
                        sub = cid + s * num_kernels_c
                        if sub < num_subs:
                            T.copy(b[sub * r_blk, 0], b_c)
                            T.vbrc(b_c, b2)
                            for j in T.serial(R_lead):
                                row0 = j * run + sub * r_blk
                                T.copy(a[row0, 0], a2)
                                T.vxor(a2, b2, y2)
                                T.copy(y2, y[row0, 0])

            return main2c

        if use_2d:
            groups_per_lead = run // r_blk
            num_logical_2d = R_lead * groups_per_lead
            num_kernels_2d = min(num_logical_2d, vector_cores)
            num_local_tasks_2d = T.ceildiv(num_logical_2d, num_kernels_2d)

            @T.prim_func
            def main2(
                a: T.Tensor((R, L), dtype),
                b: T.Tensor((run, 1) if b_2d_col else (1, L), dtype),
                y: T.Tensor((R, L), out_dtype),
            ):
                with T.Kernel(num_kernels_2d, is_npu=True) as (cid, _):
                    a2 = T.alloc_ub((r_blk, L), dtype)
                    b2 = T.alloc_ub((r_blk, L), dtype)  # 物化后的整块
                    b_c = T.alloc_ub((r_blk, 1) if b_2d_col else (1, L), dtype)
                    y2 = T.alloc_ub((r_blk, L), out_dtype)
                    if not b_2d_col:
                        T.copy(b[0, 0], b_c)
                        T.vbrc(b_c, b2)  # [1,L] -> [r_blk,L]
                    for i in T.serial(num_local_tasks_2d):
                        task = i * num_kernels_2d + cid
                        if task < num_logical_2d:
                            lead = task // groups_per_lead
                            sub = task % groups_per_lead
                            row0 = lead * run + sub * r_blk
                            T.copy(a[row0, 0], a2)
                            if b_2d_col:
                                T.copy(b[sub * r_blk, 0], b_c)
                                T.vbrc(b_c, b2)  # [r_blk,1] -> [r_blk,L]
                            T.vxor(a2, b2, y2)
                            T.copy(y2, y[row0, 0])

            return main2

        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                # a_ub / b_ub / y_ub 恒为 (block_size,) 同形——T.vxor 要求
                # src0/src1/dst 形状与 dtype 完全一致（CG-2026-0020），不能像
                # 支持广播的 vand/vor 那样留一个小形状 operand。
                a_ub = T.alloc_ub((block_size,), dtype)
                b_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                # 标量广播源：仅当某 operand 的末维 stride==0（该 operand 在
                # 行内是标量）时使用，必须先经 T.vbrc 物化成 (block_size,)。
                a1_ub = T.alloc_ub((1,), dtype)
                b1_ub = T.alloc_ub((1,), dtype)

                # Stage 4 hoisted broadcast operand (loaded once per core).
                if hoist_a:
                    T.copy(a[0:L], a_ub[0:L])
                if hoist_b:
                    T.copy(b[0:L], b_ub[0:L])

                # persistent 核内串行循环：边界为编译期静态值（num_local_tasks
                # 由 num_logical/num_kernels 常量除得），grid-stride 映射
                # task = i*num_kernels + cid，if 屏蔽越界 task。
                for i in T.serial(num_local_tasks):
                    task = i * num_kernels + cid
                    if task < num_logical:
                        row = task // chunks
                        inner_off = (task % chunks) * block_size
                        # 行 base 标量：每 task 仅一次 unrolled divmod（非每元素），
                        # 末维仅在 stride==1 时贡献 inner_off。
                        a_base = _row_base(row, outer, a_strides) + (inner_off if sa == 1 else 0)
                        b_base = _row_base(row, outer, b_strides) + (inner_off if sb == 1 else 0)
                        # Stage 4 round 8: when the row divides the block
                        # exactly there is no partial chunk, so the tail is a
                        # trace-time constant and the copies below get a static
                        # extent (`_exact_chunks` is a Python bool, so the
                        # ternary folds at trace time).  A runtime `min`
                        # extent makes every GM<->UB burst a dynamic-length
                        # DMA.
                        tail = _static_tail if _static_tail else T.min(block_size, L - inner_off)

                        # --- 物化 a 为 (block_size,)（R4：vxor 无广播）---
                        if sa == 1:
                            # a 末维连续：直接连续段搬入（Stage 4 hoist 时跳过）。
                            if not hoist_a:
                                T.copy(a[a_base : a_base + tail], a_ub[0:tail])
                        else:
                            # a 行内标量：单元素搬入 + vbrc 广播。int8 直连
                            # vbrc 对 vxor 实测 bit-exact（vand 会挂死，本算子
                            # 不需要 f16 transit）。
                            T.copy(a[a_base : a_base + 1], a1_ub[0:1])
                            T.vbrc(a1_ub, a_ub)

                        # --- 物化 b 为 (block_size,)（同上）---
                        if sb == 1:
                            if not hoist_b:
                                T.copy(b[b_base : b_base + tail], b_ub[0:tail])
                        else:
                            T.copy(b[b_base : b_base + 1], b1_ub[0:1])
                            T.vbrc(b1_ub, b_ub)

                        # 两 operand 同形后才可 vxor；尾块 [tail:] 的陈旧数据
                        # 一并参与向量计算但不写回 GM。
                        T.vxor(a_ub, b_ub, y_ub)
                        flat0 = row * L + inner_off
                        T.copy(y_ub[0:tail], y[flat0 : flat0 + tail])

        return main

    return kernel


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_TORCH_DTYPE = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

# Random-data bounds (inclusive low, exclusive high) per dtype; the full
# two's-complement range is sampled for int8/int16 and a wide range for the
# wider dtypes, with explicit extremes covered by L0-9.
_RAND_BOUNDS = {
    "int8": (-128, 128),
    "int16": (-32768, 32768),
    "int32": (-2147483648, 2147483648),
    "int64": (-4611686018427387904, 4611686018427387904),
    "uint8": (0, 256),
}

_EXTREMES = {
    "int8": (-128, 127),
    "int16": (-32768, 32767),
    "int32": (-2147483648, 2147483647),
    "int64": (-9223372036854775808, 9223372036854775807),
}

_FACTORY_CACHE = {}

# Per-dtype worst integer max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_factory(N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel):
    key = (N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = bitwise_xor_fwd_kernel(
            N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel
        )
    return _FACTORY_CACHE[key]


def _rand_cpu(shape, dtype_str):
    if dtype_str == "bool":
        return torch.randint(0, 2, tuple(shape), dtype=torch.int64).to(torch.bool)
    lo, hi = _RAND_BOUNDS[dtype_str]
    return torch.randint(lo, hi, tuple(shape), dtype=torch.int64).to(_TORCH_DTYPE[dtype_str])


def _exact_compare(out, ref):
    """Bit-exact integer/bool comparison; returns (max_diff, violations)."""
    if tuple(out.shape) != tuple(ref.shape):
        raise AssertionError(f"shape mismatch: out {tuple(out.shape)} != ref {tuple(ref.shape)}")
    if out.dtype != ref.dtype:
        raise AssertionError(f"dtype mismatch: out {out.dtype} != ref {ref.dtype}")
    if torch.equal(out, ref):
        return 0, 0
    o = out.to(torch.int64)
    r = ref.to(torch.int64)
    viol = int((o != r).sum().item())
    max_diff = int((o - r).abs().max().item())
    raise AssertionError(f"exact compare failed: {viol} violation(s), max_diff={max_diff}")


def _run_case(
    a_shape,
    b_shape,
    dtype_str,
    tag,
    block_size=None,
    storage_view=None,
    a_cpu=None,
    b_cpu=None,
):
    """Run one kernel case and compare against the golden (bit-exact).

    ``storage_view`` is None for native integer dtypes, or "bool"/"uint8"
    to exercise the R3 int8-carrier path: the public 1-byte tensors are
    viewed as int8 before the kernel and the output is viewed back before
    comparing against the golden computed in the public dtype.  When
    ``a_cpu``/``b_cpu`` are given they are used verbatim (special-value
    cases); otherwise random data is generated.
    """
    kernel_dtype = "int8" if storage_view else dtype_str
    if block_size is None:
        block_size = _DEFAULT_BLOCK

    out_shape, cshape, a_strides, b_strides = coalesce_broadcast_dims(a_shape, b_shape)
    N_total = math.prod(out_shape)
    a_numel = math.prod(a_shape) if len(a_shape) else 1
    b_numel = math.prod(b_shape) if len(b_shape) else 1

    if a_cpu is None:
        a_cpu = _rand_cpu(a_shape, dtype_str)
        b_cpu = _rand_cpu(b_shape, dtype_str)
    assert b_cpu is not None, "a_cpu and b_cpu must be supplied together"

    ref = golden_bitwise_xor(a_cpu, b_cpu)
    assert tuple(ref.shape) == tuple(out_shape), (
        f"golden shape {tuple(ref.shape)} != broadcast {tuple(out_shape)}"
    )

    if storage_view:
        # uint8/bool 走 1 字节 int8 承载（npuir 不支持 uint8 GM 张量，
        # CG-2026-0018；位模式不变，E1 已验证位级恒等）。
        a_npu = a_cpu.reshape(-1).view(torch.int8).npu()
        b_npu = b_cpu.reshape(-1).view(torch.int8).npu()
    else:
        a_npu = a_cpu.reshape(-1).npu()
        b_npu = b_cpu.reshape(-1).npu()

    fn = _get_factory(N_total, kernel_dtype, cshape, a_strides, b_strides, a_numel, b_numel)
    y = fn(block_size)(a_npu, b_npu)
    y_cpu = y.cpu()

    if storage_view:
        target = torch.bool if storage_view == "bool" else torch.uint8
        out = y_cpu.view(target).reshape(out_shape)
        assert out.dtype == target, f"{storage_view} path produced dtype {out.dtype}"
    else:
        out = y_cpu.reshape(out_shape)

    max_diff, viol = _exact_compare(out, ref)
    stat_key = storage_view if storage_view else dtype_str
    if max_diff > _STATS_MAX_DIFF.get(stat_key, -1):
        _STATS_MAX_DIFF[stat_key] = max_diff

    print(
        f"[{tag}] PASS: a={tuple(a_shape)} b={tuple(b_shape)} "
        f"dtype={('->'.join((storage_view, 'int8')) if storage_view else dtype_str)} "
        f"block={block_size} max_diff={max_diff}"
    )
    return max_diff, viol


def _try_case(failures, case_id, **kwargs):
    """Run ``_run_case``, collecting (not re-raising) per-case failures."""
    try:
        _run_case(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, ten items)."""
    failures = []

    # L0-1: smoke, same-shape, all dtypes (manifest smoke-1k).
    for dtype_str in ("int8", "int16", "int32", "int64"):
        _try_case(
            failures,
            f"L0-1/{dtype_str}",
            a_shape=(1024,),
            b_shape=(1024,),
            dtype_str=dtype_str,
            tag="L0-1",
        )

    # L0-2: large same-shape int32/int64 (manifest hidden-state-prefill).
    for dtype_str in ("int32", "int64"):
        _try_case(
            failures,
            f"L0-2/{dtype_str}",
            a_shape=(2048, 4096),
            b_shape=(2048, 4096),
            dtype_str=dtype_str,
            tag="L0-2",
        )

    # L0-3: broadcast three modes int32 (manifest cnn-feat + row-broadcast
    # + bias-add).
    for a_shape, b_shape, label in (
        ((16, 256, 56, 56), (256, 1, 1), "cnn-feat"),
        ((4096, 4096), (1, 4096), "row-broadcast"),
        ((2, 64, 128), (1, 1, 128), "bias-add"),
    ):
        _try_case(
            failures,
            f"L0-3/{label}",
            a_shape=a_shape,
            b_shape=b_shape,
            dtype_str="int32",
            tag="L0-3",
        )

    # L0-4: a-side broadcast (roles swapped, exercises sa == 0).
    for a_shape, b_shape in (((1, 4096), (4096, 4096)), ((2, 1, 3), (2, 3, 3))):
        _try_case(
            failures,
            f"L0-4/{a_shape}",
            a_shape=a_shape,
            b_shape=b_shape,
            dtype_str="int32",
            tag="L0-4",
        )

    # L0-5: scalar broadcast / full scalar (L == 1 both-vbrc degenerate,
    # avoids TRAP-vbrc-same-shape-empty-broadcast since dst > 1).
    _try_case(
        failures,
        "L0-5/scalar-broadcast",
        a_shape=(64, 128),
        b_shape=(1, 1),
        dtype_str="int32",
        tag="L0-5",
    )
    _try_case(
        failures, "L0-5/full-scalar", a_shape=(1,), b_shape=(1,), dtype_str="int32", tag="L0-5"
    )

    # L0-6: non-divisible tails (single partial tile / multi-chunk / tail
    # length not 32B aligned).
    for N in (1000, 3136, 4097):
        _try_case(
            failures, f"L0-6/N={N}", a_shape=(N,), b_shape=(N,), dtype_str="int32", tag="L0-6"
        )

    # L0-7: bool storage path (bool -> int8 view): same-shape, and the
    # manifest cnn-feat per-channel broadcast (sb == 0) plus the full-scalar
    # case -- both force the int8-dst T.vbrc materialisation (R4), the path
    # REVIEW.md asked to cover explicitly.
    for a_shape, b_shape in (((2048, 4096), (2048, 4096)), ((2, 512, 768), (1, 1, 768))):
        a = torch.randint(0, 2, a_shape).bool()
        b = torch.randint(0, 2, b_shape).bool()
        _try_case(
            failures,
            f"L0-7/bool/{a_shape}",
            a_shape=a_shape,
            b_shape=b_shape,
            dtype_str="bool",
            tag="L0-7",
            storage_view="bool",
            a_cpu=a,
            b_cpu=b,
        )
    for a_shape, b_shape in (((16, 256, 56, 56), (256, 1, 1)), ((64, 128), (1, 1))):
        a = torch.randint(0, 2, a_shape).bool()
        b = torch.randint(0, 2, b_shape).bool()
        _try_case(
            failures,
            f"L0-7/bool-vbrc/{a_shape}",
            a_shape=a_shape,
            b_shape=b_shape,
            dtype_str="bool",
            tag="L0-7",
            storage_view="bool",
            a_cpu=a,
            b_cpu=b,
        )

    # L0-7b: plain uint8 direct input through the int8 carrier (REVIEW.md
    # dimension 6 gap; manifest smoke contains a torch.uint8 param).
    for a_shape, b_shape in (((1024,), (1024,)), ((16, 256, 56, 56), (256, 1, 1))):
        _try_case(
            failures,
            f"L0-7b/uint8/{a_shape}",
            a_shape=a_shape,
            b_shape=b_shape,
            dtype_str="uint8",
            tag="L0-7b",
            storage_view="uint8",
        )

    # L0-8: persistent core-count switch boundary (num_logical in 39/40/41).
    bs = _DEFAULT_BLOCK
    for num_logical, N in ((39, 39 * bs), (40, 40 * bs), (41, 41 * bs)):
        _try_case(
            failures,
            f"L0-8/num_logical={num_logical}",
            a_shape=(N,),
            b_shape=(N,),
            dtype_str="int32",
            tag="L0-8",
        )

    # L0-9: two's-complement extremes (sign bit must not affect the XOR).
    for dtype_str in ("int8", "int16", "int32", "int64"):
        lo, hi = _EXTREMES[dtype_str]
        td = _TORCH_DTYPE[dtype_str]
        pattern = [lo, hi, -1, 0, 1, lo, hi, -1]
        reps = 32
        a = torch.tensor(pattern * reps, dtype=td)
        b = torch.tensor(pattern[::-1] * reps, dtype=td)
        _try_case(
            failures,
            f"L0-9/{dtype_str}",
            a_shape=(256,),
            b_shape=(256,),
            dtype_str=dtype_str,
            tag="L0-9",
            a_cpu=a,
            b_cpu=b,
        )

    # L0-10: vxor same-shape-constraint regression -- `[M,N]x[1,N]` and
    # `[M,N]x[M,1]` are directly expressible on GPU but require explicit
    # same-shape materialisation here (CG-2026-0020).
    for a_shape, b_shape, label in (
        ((64, 512), (1, 512), "row-brc"),
        ((64, 512), (64, 1), "col-brc"),
    ):
        _try_case(
            failures,
            f"L0-10/{label}",
            a_shape=a_shape,
            b_shape=b_shape,
            dtype_str="int32",
            tag="L0-10",
        )

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (ten items)")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x shape x block_size)."""
    failures = []

    # dtype x {exact multiple, tail==1-ish, non-32B-aligned tail}.
    for dtype_str in ("int8", "int16", "int32", "int64"):
        for N in (8192, 1000, 4097):
            _try_case(
                failures,
                f"L1/{dtype_str}/N={N}",
                a_shape=(N,),
                b_shape=(N,),
                dtype_str=dtype_str,
                tag="L1",
            )

    # Broadcast variants across dtypes and shapes.
    for dtype_str in ("int8", "int32"):
        for a_shape, b_shape in (
            ((4, 8, 16), (1, 8, 1)),
            ((3, 5), (1, 1)),
            ((2, 3, 4), (2, 3, 4)),
        ):
            _try_case(
                failures,
                f"L1/{dtype_str}/{a_shape}",
                a_shape=a_shape,
                b_shape=b_shape,
                dtype_str=dtype_str,
                tag="L1",
            )

    # int8 carrier + innermost stride-0 broadcast across dtypes (the vbrc
    # path of R4).
    for a_shape, b_shape in (((16, 256, 56, 56), (256, 1, 1)), ((32, 128), (1, 1))):
        _try_case(
            failures,
            f"L1/int8-vbrc/{a_shape}",
            a_shape=a_shape,
            b_shape=b_shape,
            dtype_str="int8",
            tag="L1",
        )

    # Non-default block sizes within the UB budget.
    _try_case(
        failures,
        "L1/int8/N=2^20/bs=8192",
        a_shape=(2**20,),
        b_shape=(2**20,),
        dtype_str="int8",
        tag="L1",
        block_size=8192,
    )
    _try_case(
        failures,
        "L1/int32/N=2^20/bs=2048",
        a_shape=(2**20,),
        b_shape=(2**20,),
        dtype_str="int32",
        tag="L1",
        block_size=2048,
    )
    _try_case(
        failures,
        "L1/int64/N=2^18/bs=1024",
        a_shape=(2**18,),
        b_shape=(2**18,),
        dtype_str="int64",
        tag="L1",
        block_size=1024,
    )

    # E2 factory contract: one factory callable serving two block_sizes.
    try:
        dtype_str = "int32"
        a_shape = (8192,)
        out_shape, cshape, a_strides, b_strides = coalesce_broadcast_dims(a_shape, a_shape)
        a = _rand_cpu(a_shape, dtype_str).npu()
        b = _rand_cpu(a_shape, dtype_str).npu()
        ref = golden_bitwise_xor(a.cpu(), b.cpu())
        fn = _get_factory(8192, dtype_str, cshape, a_strides, b_strides, 8192, 8192)
        for bs in (4096, 2048):
            out = fn(bs)(a, b).cpu()
            _exact_compare(out, ref)
        print("[L1] PASS: E2 factory(block_size) dual-config reuse")
    except AssertionError as exc:
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse RUNTIME-ERROR: {exc!r}")

    # UB budget guard + dtype rejection.
    try:
        bitwise_xor_fwd_kernel(4096, "int64", (4096,), (1,), (1,), 4096, 4096)(8192)
    except ValueError as exc:
        print(f"[L1] PASS: UB guard rejects int64 block_size=8192 ({exc})")
    else:
        failures.append(("L1/UB-guard-int64", AssertionError("int64 8192 not rejected")))
    for bad in ("uint8", "bool", "float32"):
        try:
            _normalize_dtype(bad)
        except ValueError:
            print(f"[L1] PASS: dtype {bad!r} rejected")
        else:
            failures.append((f"L1/dtype-reject-{bad}", AssertionError(f"{bad} not rejected")))

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes (warn only, non-blocking)."""
    for N, dtype_str in (
        (1, "int32"),
        (2, "int8"),
        (3, "int64"),
        (2**22, "int8"),
        (2**24, "int32"),
    ):
        try:
            _run_case(a_shape=(N,), b_shape=(N,), dtype_str=dtype_str, tag="L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    N = 4096

    # a ^ a == 0 (self-inverse identity).
    try:
        a = _rand_cpu((N,), "int32")
        _run_case(
            a_shape=(N,),
            b_shape=(N,),
            dtype_str="int32",
            tag="Boundary-a-xor-a",
            a_cpu=a,
            b_cpu=a.clone(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): a^a: {e}")

    # all -1 (every bit set) ^ x == ~x.
    try:
        a = torch.full((N,), -1, dtype=torch.int32)
        b = _rand_cpu((N,), "int32")
        _run_case(
            a_shape=(N,), b_shape=(N,), dtype_str="int32", tag="Boundary-all-ones", a_cpu=a, b_cpu=b
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): all-ones: {e}")

    # all zeros ^ x == x.
    try:
        a = torch.zeros(N, dtype=torch.int32)
        b = _rand_cpu((N,), "int32")
        _run_case(
            a_shape=(N,), b_shape=(N,), dtype_str="int32", tag="Boundary-zeros", a_cpu=a, b_cpu=b
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros: {e}")

    # scalar broadcast: x ^ 0 == x.
    try:
        a = _rand_cpu((N,), "int32")
        b = torch.zeros((1,), dtype=torch.int32)
        _run_case(
            a_shape=(N,),
            b_shape=(1,),
            dtype_str="int32",
            tag="Boundary-scalar-zero",
            a_cpu=a,
            b_cpu=b,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): scalar-zero: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    if args.level in ("L0", "all"):
        t0 = time.time()
        failures += run_L0()
        print(f"[L0] elapsed {time.time() - t0:.1f}s")
    if args.level in ("L1", "all"):
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
    if args.level in ("L2", "all"):
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
    if args.level in ("boundary", "all"):
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    print(
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}={v}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
