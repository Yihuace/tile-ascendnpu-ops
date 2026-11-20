# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Bitwise-AND forward kernel (NPU, Developer mode).

Migrated from the GPU TileOPs operator ``bitwise_and_fwd_kernel``
(``_make_binary_explicit`` factory specialised with ``op_func(a, b) = a & b``)
to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y[i] = a[off_a(i)] & b[off_b(i)]
on two already-flattened contiguous ``(a_numel,)`` / ``(b_numel,)`` tensors
of the same integer dtype; the output is the flat ``(N_total,)`` view of
``broadcast_shapes(a_shape, b_shape)``.  ``off_a`` / ``off_b`` are the
broadcast index maps; for same-shape inputs they are the identity.

NPU redesign (DESIGN.md section 0.6):
    R1: the source's per-element unrolled divmod chain (used only on the
        broadcast path) is replaced by an outer-row scalar base (one
        unrolled mixed-radix divmod per task) plus innermost-dimension
        vectorisation.  Rows with last-dim stride 0 (a scalar within the
        row) are materialised with ``T.vbrc`` because ``T.vand`` rejects
        scalar / size-1 operands (CG-2026-0019).
    R2: one-dimensional ``T.Kernel(num_kernels, is_npu=True)`` with a
        persistent in-core ``T.serial`` grid-stride loop and an if-guard,
        replacing the CUDA grid x block two-level parallel structure.
        Element parallelism is handled by the Vector unit via ``T.vand``.
    R3: ``bool`` storage is carried as ``int8`` (bit-identical, E1) because
        npuir cannot carry ``uint8`` GM tensors (CG-2026-0018).  The Op
        layer views the public ``torch.bool`` tensor as ``int8`` before
        calling the kernel.

Stage 4 round 6 (closeout) changes, both merged here:
    - r_blk for the 2-D tile is now derived from the compiler-reported footprint
      (row form `2*r_blk*L*E + L*E`, col form `2*r_blk*L*E + 288 B`) instead of a
      hardcoded 2/4, which lifts the row-broadcast int32 tile from 32 KB (r_blk=2)
      to 64 KB (r_blk=4): A/B −3.2 % on row-broadcast-int32.
    - int64 is allowed on the 2-D path for the *row* broadcast form only
      (leading-dim broadcast); the col form times the Vector core out for int64,
      and both int8 forms hang (runtime timeout for the broadcast form, compiler
      hang for the explicit `T.vbrc` materialisation) -- see
      perf_opt/probe_2d_dtype.py + logs/r6/.

Stage 4 round 7 (v3, int32-pack) changes: the 1-byte (bool / int8) path no
longer enters this module as `int8`.  The Op wrapper views the two contiguous
1-byte operands as `int32` on the *host* (4 bytes per word; a broadcast-along-
last-dim operand is byte-replicated, value ``b | b<<8 | b<<16 | b<<24``) and
calls this factory with the packed shapes and ``dtype="int32"``.  A bitwise op
is bit-parallel, so one int32 word does the work of four int8 elements with a
bit-identical result.  Nothing in this module changes behaviour for the packed
case -- it is the existing, already-tuned int32 path -- except that
``_TUNED_RBLK_COL`` was raised from 4 to 8 (see its comment).  The int8 code
path is still present and is still what runs when packing is not applicable
(last participating dim not a multiple of 4, non-contiguous operand, other
dtype); the int8 2-D forms remain forbidden (CG-2026-0043), and the packed
route never touches them: it hands the kernel an int32 tensor.

Implementation notes (toolchain traps found during Stage 3):
    - ``T.vbrc`` with an int8 destination hangs the Vector core at runtime
      ("vector core timeout", error 507034/507035) on this toolchain, even
      though docs/Tilelang.language/shape操作/T.vbrc.md lists int8 as
      supported.  int16/int32/int64 are unaffected.  The int8 scalar-broadcast
      path therefore detours through float16
      (``int8 -> vcast(rint) -> float16 -> vbrc -> float16 -> vcast(rint) ->
      int8``): every int8 value is exactly representable in float16, so the
      round trip is bit-exact.
    - The kernel body's ``block_size`` must be the same value used to derive
      the task decomposition (``chunks`` / ``num_logical`` / ``num_kernels``
      / ``num_local_tasks``).  Those are computed *inside* the jit function
      (not at factory-build time) so the returned callable stays consistent
      for every block_size it is invoked with.

Factory contract (frozen; see examples/TileOPs/tileops/kernels/elementwise/
bitwise_and/bitwise_and.py):
    bitwise_and_fwd_kernel(N_total, dtype, coalesced_shape, a_strides,
                           b_strides, a_numel, b_numel,
                           output_dtype=None, block_size=4096)
        -> kernel(block_size) -> main(a, b) -> y   (``out_idx=[2]``)

Run the embedded hierarchical tests:
    python bitwise_and_fwd_kernel.py --level L0
    python bitwise_and_fwd_kernel.py --level all
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
    "bitwise_and_fwd_kernel",
    "coalesce_broadcast_dims",
    "golden_bitwise_and",
]

# Programming mode actually validated by this file (TileOPs harness
# migration requires this module-level literal).
ASCEND_MODE = "Developer"

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608
# Conservative auto-multi-buffer inflation factor (CONST-capacity-910B2C).
_UB_INFLATION = 1.7

# Element sizes for the supported integer dtypes.
_ELEM_BYTES = {"int8": 1, "int16": 2, "int32": 4, "int64": 8}

# dtypes handled natively by the kernel.  uint8 is deliberately rejected
# (CG-2026-0018) and bool must be viewed to int8 by the Op layer (R3).
_ALLOWED_DTYPES = ("int8", "int16", "int32", "int64")

_DEFAULT_BLOCK = 4096

# Stage 4 round-3: 2-D [r_blk, L] row tile for the *channel-broadcast* shapes
# (cnn-feat: a=[N,C,H,W], b=[C,1,1]).  Proven on FloorDivide (its cnn fp32 went
# 17.7% -> 72.3%): a whole row-block is loaded once and the per-row scalar is
# broadcast natively by T.vand (no per-row T.vbrc), turning "many tiny row
# bursts" into fewer large bursts.
#
# Stage 4 round 7 (v3 int32-pack): the cap was 4, which for the *packed* 1-byte
# path (bool viewed as int32 on the host -> L = 56/4 = 14 per (C,H,W) row, i.e.
# coalesced L = 784 int32 = 3136 B) restricted the per-task tile to
# r_blk*L*E = 12.5 KB.  The UB-derived cap for that shape is 31, so the ladder
# 8/16 was opened up by measurement: r_blk 4->8 takes the cnn-feat bool case
# from 24.0 us to 13.1 us (tile 12.5 -> 25 KB, i.e. 8 contiguous rows per DMA
# task), r_blk=16 is a tie (13.1 vs 13.2 us) and 32 exceeds UB.  Every other
# tune workload keeps its previous r_blk because the UB cap (not this constant)
# binds there: cnn int32 caps at 7 -> 4, cnn int64 at 3 -> 2, row int32 at
# 5 -> 4, row int64 at 2 -> 2.
_TUNED_RBLK_COL = 8
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

# Stage 4 tuned execution parameters (per-dtype block_size).  Exposed as
# module-level constants so the Op-layer wrapper source-selection block can
# adopt the SAME values it compiles against (tilelang-op-optimize Phase 3).
TUNED_BLOCK_SIZE = 4096  # non-1-byte dtypes (int16/int32/int64)
TUNED_BOOL_BLOCK_SIZE = 8192  # int8 storage carrier (bool)

# Stage 4 round 7: block_size used by the *host-side int32-packed* 1-byte path.
# The packed tensor is a genuine int32 kernel input, so its UB budget is the
# int32 one (`_UB_MAX_BLOCK["int32"]` = 7228) and the bool-carrier 8192 would be
# rejected by the factory guard.  4096 is the tuned int32 value
# (`TUNED_BLOCK_SIZE`) and is what the packed cnn/hidden measurements used.
TUNED_PACK_BLOCK_SIZE = 4096

# Bytes of full-size UB buffer per element, worst case.  Three buffers
# (a/b/y) of the operand dtype plus the two float16 transit buffers used for
# the int8 scalar-broadcast materialisation (see the int8 trap note in the
# module docstring).  The transit buffers are allocated for every dtype but are
# only live for int8; budgeting them for all dtypes is the conservative
# bound.
_UB_BYTES_PER_ELEM = {
    "int8": 3 * 1 + 2 * 2,  # 3 x int8 + 2 x float16 transit
    "int16": 3 * 2 + 2 * 2,
    "int32": 3 * 4 + 2 * 2,
    "int64": 3 * 8 + 2 * 2,
}

# Upper bound on block_size from the UB budget (DESIGN.md section 4.5,
# conservative x1.7 inflation).
_UB_MAX_BLOCK = {
    dt: int(_UB_CAPACITY_BYTES // (_UB_BYTES_PER_ELEM[dt] * _UB_INFLATION))
    for dt in _ALLOWED_DTYPES
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_bitwise_and(a, b):
    """Bitwise-AND reference (source-operator semantics, direct broadcast).

    Matches the source-repo test benchmark
    (examples/TileOPs/tests/ops/test_bitwise_and.py: ``ref =
    torch.bitwise_and(a, b)``; integer/bool compared exactly).  Inputs keep
    their original broadcast shapes; the output shape is
    ``broadcast_shapes(a.shape, b.shape)``.  Independent of the NPU
    algorithm (no row/inner indexing, no vbrc, no persistent structure) and
    runs on CPU without an NPU device.
    """
    return torch.bitwise_and(a.detach().cpu(), b.detach().cpu())


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
    executed once per task rather than once per element -- DESIGN.md R1).
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
            "load the bool-storage path through int8 instead"
        )
    if name == "bool":
        raise ValueError(
            "bool tensors must be viewed to int8 by the Op layer before "
            "reaching the kernel (DESIGN.md R3)"
        )
    if name not in _ALLOWED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {_ALLOWED_DTYPES}")
    return name


def bitwise_and_fwd_kernel(
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
    """Build the bitwise-AND kernel factory.

    Returns a JIT callable taking a single ``block_size`` which compiles and
    returns ``main``; ``main(a, b)`` returns ``y`` (``out_idx=[2]``,
    source-operator contract).

    ``pack_words=2`` selects the Stage-4 round-8 packed-int64 pair form (see
    the branch below); the default ``1`` is the ordinary element-wise form
    and its signature/semantics are unchanged.
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
    # adapt automatically.
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
        coef = _PAIR_UB_COEF_MB if _PAIR_MULTI_BUFFER else _PAIR_UB_COEF_NO_MB
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
                    b2 = T.alloc_ub((1, W), dtype)
                    y2 = T.alloc_ub((rows_per_task, W), out_dtype)
                    for i in T.serial(num_local_pair):
                        task = i * num_kernels_pair + cid
                        if task < num_logical_pair:
                            blk = task // nchunks
                            sub = task % nchunks
                            row0 = blk * nrows_total + sub * rows_per_task
                            # per-channel broadcast row ([low, high] x K)
                            T.copy(b[blk % run_p, 0], b2)
                            T.copy(a[row0, 0], a2)
                            T.vand(a2, b2, y2)
                            T.copy(y2, y[row0, 0])

            return main_pair

        return kernel_pair

    L = coalesced_shape[-1]
    outer = list(coalesced_shape[:-1])
    R = math.prod(outer) if outer else 1
    sa, sb = a_strides[-1], b_strides[-1]
    if sa not in (0, 1) or sb not in (0, 1):
        raise ValueError(f"innermost strides must be 0 or 1, got sa={sa}, sb={sb}")

    # ---- 2-D tile eligibility (channel-broadcast shapes) ----------------
    # 与 FloorDivide 已验证的判定一致：列模式要求 innermost stride==0（在 L 上广播）
    # 且次内层 stride==1（逐行/逐通道变化）、b 元素数 == run（= outer[-1]）。
    run = outer[-1] if outer else 1
    b_2d_col = sb == 0 and len(outer) >= 1 and b_strides[-2] == 1 and b_numel == run
    b_2d_row = sb == 1 and all(st == 0 for st in b_strides[:-1]) and b_numel == L
    # Stage 4 round 6 (r6_op1): r_blk 按**编译器报文的真实足迹**取上限，而不是
    # 一个硬编码的 2/4。BiShengIR 自己的 ub-overflow 报文反解（见 logs/r6/
    # probe_*_row.log，探针 probe_2d_dtype.py）：
    #   row 形态（b2 为 (1,L)，逐行共享同一 b）: 2*r_blk*L*E + L*E
    #     int32 L=4096 r_blk=8 -> 2,228,224 bit = 278,528 B（= 2*8*4096*4 + 4096*4）✓
    #   col 形态（b2 为 (r_blk,1)）: 2*r_blk*L*E + 288 B
    #     int32 L=3136 r_blk=8 -> 1,607,936 bit = 200,992 B（= 2*8*3136*4 + 288）✓
    # 即真实需求是 "2 个 rtile + 卷积核自身的小常数量"，与 §8/§10 的 vcmp 族同构；
    # 之前 col/row 都按 1.7 膨胀拍脑袋，row 因此被压到 r_blk=2（tile 32 KB）。
    _E = _ELEM_BYTES[dtype]
    _2D_RESERVE = 1024

    def _max_rblk_2d(form, run_n):
        if form == "col":
            caps = (_UB_CAPACITY_BYTES - _2D_RESERVE) // (2 * L * _E)
        else:
            caps = ((_UB_CAPACITY_BYTES - _2D_RESERVE) // (L * _E) - 1) // 2
        caps = min(caps, _TUNED_RBLK_COL)
        best = 1
        for cand in range(1, caps + 1):
            if run_n % cand == 0:
                best = cand
        return best

    r_blk = _max_rblk_2d("col" if b_2d_col else "row", run) if (b_2d_col or b_2d_row) else 1
    R_lead = R // run if run else 1
    # int8(bool) 的 T.vand + rank-2 广播会让 Vector 核超时（同族已知 int8 陷阱），
    # 因此 int8 一律走 1-D 路径。
    # Stage 4 round 6 (r6_op2, merged): int64 的 2-D 封锁是 **col 形态** 专属的——
    # 实测（probe_2d_dtype.py，logs/r6/probe_int64_r2_col_bcast.log）int64 的 col
    # 形态（b2 为 (r_blk,1)，沿末维广播）在运行期 Vector 核超时；而 **row 形态**
    # （b2 为 (1,L)，只沿前导维广播）编译通过且 bit-exact
    # （logs/r6/probe_int64_r2_row.log）。row-broadcast 类因此可以进 2-D。
    _2d_dtype_ok = dtype == "int32" or (dtype == "int64" and b_2d_row)
    use_2d = bool(
        _2d_dtype_ok
        and (b_2d_col or b_2d_row)
        and len(outer) >= 1
        and sa == 1
        and a_strides[-2] == L
        and a_numel == R * L
        and L <= _MAX_2D_L
        and L % 16 == 0
        and 2 <= r_blk <= run
        and run % r_blk == 0
        and r_blk * L <= _MAX_2D_ELEMS
    )

    # int8 scalar broadcast must detour through float16 (see the module
    # docstring); the flag is resolved at trace time.
    use_f16_transit = dtype == "int8"

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(block_size=block_size):
        if use_2d:
            # ---- alternative: 2-D [r_blk, L] row tile per task ----------
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
                    b2 = T.alloc_ub((r_blk, 1) if b_2d_col else (1, L), dtype)
                    y2 = T.alloc_ub((r_blk, L), out_dtype)
                    # row mode: b 对所有行相同 -> 每核只搬一次
                    if not b_2d_col:
                        T.copy(b[0, 0], b2)
                    for i in T.serial(num_local_tasks_2d):
                        task = i * num_kernels_2d + cid
                        if task < num_logical_2d:
                            lead = task // groups_per_lead
                            sub = task % groups_per_lead
                            row0 = lead * run + sub * r_blk
                            T.copy(a[row0, 0], a2)
                            if b_2d_col:
                                T.copy(b[sub * r_blk, 0], b2)
                            # T.vand 原生 rank>=2 广播（b2 为 [r_blk,1]），
                            # 不需要逐行 T.vbrc。
                            T.vand(a2, b2, y2)
                            T.copy(y2, y[row0, 0])

            return main2

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
        # block_size (wrapper K11 contract; same pattern as the lerp
        # precedent).  block_size folds to a compile-time constant.
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

        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((block_size,), dtype)
                b_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                # Scalar-broadcast sources (used only when the operand's
                # innermost stride is 0).
                a1_ub = T.alloc_ub((1,), dtype)
                b1_ub = T.alloc_ub((1,), dtype)
                # float16 transit buffers for the int8 scalar-broadcast path
                # (see the int8 trap note in the module docstring).  Allocated
                # unconditionally; dead-code eliminated when unused.
                a1_f16 = T.alloc_ub((1,), "float16")
                b1_f16 = T.alloc_ub((1,), "float16")
                a_f16 = T.alloc_ub((block_size,), "float16")
                b_f16 = T.alloc_ub((block_size,), "float16")

                # Stage 4 hoisted broadcast operand (loaded once per core).
                if hoist_a:
                    T.copy(a[0:L], a_ub[0:L])
                if hoist_b:
                    T.copy(b[0:L], b_ub[0:L])

                # Persistent in-core serial task loop (static bound,
                # grid-stride mapping, if-guard masks out-of-range tasks).
                for i in T.serial(num_local_tasks):
                    task = i * num_kernels + cid
                    if task < num_logical:
                        row = task // chunks
                        inner_off = (task % chunks) * block_size
                        # Outer-row scalar base (once per task).  The
                        # innermost dim contributes only when its stride is 1.
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

                        # Load a: contiguous segment or materialised scalar
                        # broadcast (T.vand rejects size-1 operands).
                        if sa == 1:
                            if not hoist_a:
                                T.copy(a[a_base : a_base + tail], a_ub[0:tail])
                        else:
                            T.copy(a[a_base : a_base + 1], a1_ub[0:1])
                            if use_f16_transit:
                                T.vcast(a1_ub, a1_f16, round_mode="rint")
                                T.vbrc(a1_f16, a_f16)
                                T.vcast(a_f16, a_ub, round_mode="rint")
                            else:
                                T.vbrc(a1_ub, a_ub)

                        # Load b: same treatment.
                        if sb == 1:
                            if not hoist_b:
                                T.copy(b[b_base : b_base + tail], b_ub[0:tail])
                        else:
                            T.copy(b[b_base : b_base + 1], b1_ub[0:1])
                            if use_f16_transit:
                                T.vcast(b1_ub, b1_f16, round_mode="rint")
                                T.vbrc(b1_f16, b_f16)
                                T.vcast(b_f16, b_ub, round_mode="rint")
                            else:
                                T.vbrc(b1_ub, b_ub)

                        # Vectorised bitwise AND, then store the valid lanes.
                        T.vand(a_ub, b_ub, y_ub)
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
}

# Random-data bounds (inclusive low, exclusive high) per dtype; the full
# two's-complement range is sampled for int8/int16 and a wide range for the
# wider dtypes, with explicit extremes covered by L0-9.
_RAND_BOUNDS = {
    "int8": (-128, 128),
    "int16": (-32768, 32768),
    "int32": (-2147483648, 2147483648),
    "int64": (-4611686018427387904, 4611686018427387904),
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
        _FACTORY_CACHE[key] = bitwise_and_fwd_kernel(
            N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel
        )
    return _FACTORY_CACHE[key]


def _rand_cpu(shape, dtype_str):
    lo, hi = _RAND_BOUNDS[dtype_str]
    return torch.randint(lo, hi, shape, dtype=torch.int64).to(_TORCH_DTYPE[dtype_str])


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
    bool_storage=False,
    a_cpu=None,
    b_cpu=None,
):
    """Run one kernel case and compare against the golden (bit-exact).

    ``bool_storage`` views the public bool tensors as int8 (R3); the
    kernel is built with dtype ``int8`` and the output is viewed back to
    bool.  When ``a_cpu``/``b_cpu`` are given they are used verbatim
    (special-value cases); otherwise random data is generated.
    """
    kernel_dtype = "int8" if bool_storage else dtype_str
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

    ref = golden_bitwise_and(a_cpu, b_cpu)
    assert tuple(ref.shape) == tuple(out_shape), (
        f"golden shape {tuple(ref.shape)} != broadcast {tuple(out_shape)}"
    )

    if bool_storage:
        a_npu = a_cpu.reshape(-1).view(torch.int8).npu()
        b_npu = b_cpu.reshape(-1).view(torch.int8).npu()
    else:
        a_npu = a_cpu.reshape(-1).npu()
        b_npu = b_cpu.reshape(-1).npu()

    fn = _get_factory(N_total, kernel_dtype, cshape, a_strides, b_strides, a_numel, b_numel)
    y = fn(block_size)(a_npu, b_npu)
    y_cpu = y.cpu()

    if bool_storage:
        out = y_cpu.view(torch.bool).reshape(out_shape)
        assert out.dtype == torch.bool, f"bool path produced dtype {out.dtype}"
    else:
        out = y_cpu.reshape(out_shape)

    max_diff, viol = _exact_compare(out, ref)
    stat_key = "bool" if bool_storage else dtype_str
    if max_diff > _STATS_MAX_DIFF.get(stat_key, -1):
        _STATS_MAX_DIFF[stat_key] = max_diff

    print(
        f"[{tag}] PASS: a={tuple(a_shape)} b={tuple(b_shape)} "
        f"dtype={'bool->int8' if bool_storage else dtype_str} "
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
    """L0 gate suite (DESIGN.md section 8.3, nine items)."""
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

    # L0-5: scalar broadcast / full scalar (L == 1 both-vbrc degenerate).
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

    # L0-7: bool storage path (bool -> int8 view): same-shape, row-broadcast
    # (sb == 1) and the manifest cnn-feat per-channel broadcast (sb == 0,
    # exercises the int8 scalar-broadcast float16 transit).
    for a_shape, b_shape in (
        ((2048, 4096), (2048, 4096)),
        ((2, 512, 768), (1, 1, 768)),
        ((16, 256, 56, 56), (256, 1, 1)),
    ):
        a = torch.randint(0, 2, a_shape).bool()
        b = torch.randint(0, 2, b_shape).bool()
        _try_case(
            failures,
            f"L0-7/{a_shape}",
            a_shape=a_shape,
            b_shape=b_shape,
            dtype_str="bool",
            tag="L0-7",
            bool_storage=True,
            a_cpu=a,
            b_cpu=b,
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

    # L0-9: two's-complement extremes (sign bit must not affect the AND).
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

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (nine items)")
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
        ref = golden_bitwise_and(a.cpu(), b.cpu())
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
        bitwise_and_fwd_kernel(4096, "int64", (4096,), (1,), (1,), 4096, 4096)(8192)
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

    # all zeros -> all zeros.
    try:
        a = torch.zeros(N, dtype=torch.int32)
        b = torch.zeros(N, dtype=torch.int32)
        _run_case(
            a_shape=(N,), b_shape=(N,), dtype_str="int32", tag="Boundary-zeros", a_cpu=a, b_cpu=b
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros: {e}")

    # a == b -> out == a.
    try:
        a = _rand_cpu((N,), "int32")
        _run_case(
            a_shape=(N,),
            b_shape=(N,),
            dtype_str="int32",
            tag="Boundary-a-eq-b",
            a_cpu=a,
            b_cpu=a.clone(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): a==b: {e}")

    # all -1 (every bit set) AND x == x.
    try:
        a = torch.full((N,), -1, dtype=torch.int32)
        b = _rand_cpu((N,), "int32")
        _run_case(
            a_shape=(N,), b_shape=(N,), dtype_str="int32", tag="Boundary-all-ones", a_cpu=a, b_cpu=b
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): all-ones: {e}")

    # all zeros AND x == 0 (broadcast scalar).
    try:
        a = torch.zeros(N, dtype=torch.int32)
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
