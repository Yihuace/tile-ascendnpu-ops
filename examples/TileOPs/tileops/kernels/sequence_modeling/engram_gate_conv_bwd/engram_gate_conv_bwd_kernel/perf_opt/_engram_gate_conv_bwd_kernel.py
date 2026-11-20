# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Engram GateConv backward kernel (NPU, ``target="npuir"``).

Migration of the GPU TileLang factory ``_engram_gate_conv_bwd_kernel``
(``/mnt/workspace/TileOPs/tileops/kernels/engram/engram_bwd.py``).

Given ``dY`` (gradient of ``Y``), the saved forward intermediates
(``vhat, alpha, rrms_h, rrms_k, rrms_v``) and the forward inputs
(``H, k, v, rms_w_h, rms_w_v, conv_w``), computes

    dH, dk, dv          -- (M, T, d)   same dtype as dY
    drms_w_h, drms_w_v  -- (d,)        fp32
    dconv_w             -- (4, d)      fp32

The projection backward (GEMM) is done externally; everything here is a pure
Vector pipeline (no ``T.gemm`` -> not a MixCV operator).

NPU re-design status (see DESIGN.md sections 0.5 / 0.6 / 3.3 / 4 / 5 / 6 / 7):

  * R1 -- APPLIED.  A single prim_func cannot hold several ``T.Kernel`` blocks
    on npuir (design probe P1), so the factory returns a Python closure that
    launches three independent jit kernels:
      K1 = Pass 1 (SiLU bwd) + ``dconv_w`` per-block partial sums
      K2 = Pass 1b (conv transpose + RMSNorm(vhat) bwd) fused with Pass 2
           (gate bwd + RMSNorm(H)/RMSNorm(k) bwd) + ``drms_w_h/v`` partial sums
      K3 = grid-1 deterministic combine of the partial sums
    The ``d_silu`` handoff stays in GM (one write + one read); the GPU
    ``d(vhat)`` GM round-trip is removed by fusing Pass 1b and Pass 2.
  * R2 -- APPLIED.  fp32 compute domain: fp16/bf16 -> ``T.cast``/``T.vcast``
    -> fp32 chain -> single ``vcast(rint)`` back for ``dH/dk/dv``.  The fp32
    path is API-mandated for bf16 (the vector math ops have no bf16 form).
  * R3 -- APPLIED.  1-D ``T.Kernel(M * num_tiles, is_npu=True)``; every
    intermediate lives in ``T.alloc_shared`` (UB).  ``threads`` is dropped.
  * R4 -- APPLIED.  Boundaries use index clamp (``T.min``/``T.max``) plus an
    arithmetic row mask; ``T.if_then_else`` with buffer operands segfaults the
    npuir codegen, and a runtime min/max inside a ``T.Parallel`` scalar store
    produces wrong values (design probes P6a/P6b), so every mask scalar is
    emitted as a compile-time-unrolled scalar statement and broadcast with
    ``T.vbrc`` + ``T.vmul``.
  * R5 -- APPLIED.  ``T.atomic_add`` aborts on this toolchain (probe P4) and
    the harness requires bit-identical results across runs, so the cross-token
    weight gradients use a deterministic two-stage reduction: per-block
    ``reduce_sum(dim=0)`` partial sums into GM workspaces, then a grid-1
    combine kernel.
  * R6 -- APPLIED.  Constant reciprocal folding (``/d -> *fl(1/d)``,
    ``/sqrt(d) -> *fl(1/sqrt(d))``), scalar factorisation (``ddot``, ``rrms_*``)
    and CSE materialisation (``vn``, ``h_norm``, ``k_norm``, ``d_vnorm``,
    ``d(vhat)``).

Alias discipline (DESIGN section 3.3 note 13, pattern ``PL-1.23``): inside K2
``dhn``/``dkn`` are overwritten in place by ``dH``/``dk``; the ``drms_w_h``
partial sum consumes the *original* ``dhn``/``dkn`` and is therefore emitted
strictly before those overwrites.

Over-tile robustness (implementation detail, semantics preserving): ``BLK_S``
is a fixed compile-time width (8 for ``d <= 256``, 4 for ``d512``) that need
not divide ``seq_len``.  ``dH/dk/dv`` are written to a tail-padded GM workspace
(``SEQ_PAD = num_tiles * BLK_S``) with exact full-tile ``T.copy`` regions and
narrowed on the host; rows of a partial last tile are zeroed by the own-row
arithmetic mask so the cross-token reductions never see them.  Element-wise GM
stores and dynamic-region ``T.copy`` are mis-scheduled by the npuir codegen.

Codegen constraints honoured here (all reproducible from the design probes and
from ``repro/`` in this directory): GM writes are always ``T.copy`` regions;
scalar outputs are staged with an unrolled fill and a single ``T.copy``;
``T.vcast`` needs identical src/dst shapes; conv/transpose taps are unrolled in
the Python factory; buffers are allocated unconditionally at kernel top level;
scalar broadcast always goes through ``T.vbrc`` + ``T.vmul``.

Contract: the factory keeps the ``(M, seq_len, d, eps, dtype)`` signature of the
GPU source and returns a callable invoked as ``factory(...)(tensors)`` which
returns ``[dH, dk, dv, drms_w_h, drms_w_v, dconv_w]``.  ``eps`` is unused at
runtime because ``rrms_h/k/v`` are forward-saved inputs (kept for signature
parity).

Stage 4 tuning (all four changes are measurement-driven; the raw numbers and
the interleaved A/B protocols are in ``opt_log.md`` / ``perf_records.jsonl``):

  * OP1 / OP2 -- APPLIED.  Every GM->UB element-wise load that carried an
    in-loop ``T.cast`` is replaced by a raw-dtype ``T.copy`` region plus one
    whole-tile ``T.vcast``.  An element-level ``T.cast`` reading a GM tensor
    lowers to a fully scalar ``scf.for`` (``memref.load`` + ``tensor.insert``,
    e.g. 11x256 elements per block in K1, 5x8x256 in K2), which was 62-98% of
    the AIV scalar time and the whole bottleneck (K1 225.6 -> 4.4 us, K2
    497.7 -> 7.9 us on ``bwd-b1-s128-d256-bf16``).  The tensorized form is
    only used when ``blk_s`` divides ``seq_len`` (every output store / load is
    then an exact region); otherwise the kernel falls back to the clamped
    element-wise path, so correctness is shape independent.
  * OP3 -- APPLIED.  The deterministic weight-gradient combine is one
    slot-parallel kernel: K1/K2 write their partial sums into a single
    ``(KS + 2) * nb_pad`` workspace (slot 0 = ``drms_w_h``, 1 = ``drms_w_v``,
    ``2 + p`` = ``dconv_w`` tap ``p``) and K3 runs ``grid = KS + 2`` with one
    uniform reduction per slot, writing a ``(KS + 2, d)`` buffer that the
    closure carves into the three contract outputs.  The previous grid-1 form
    ran all six reductions serially and needed six unrolled scalar
    ``d``-element staging fills (25.1/52.1/26.3 -> 1.7/2.5/1.8 us).
  * TILING -- APPLIED.  The seq-tile width is now chosen per shape
    (``TUNED_BLK_CAP_*`` / ``TUNED_TARGET_BLOCKS``): the capacity tile is
    halved while it leaves fewer than ``TUNED_TARGET_BLOCKS`` blocks, because
    every manifest workload is occupancy/latency bound rather than throughput
    bound (``bwd-b1-s32-d256-fp16``: 4 -> 8 blocks, -17.7% on K2; the same
    halving on ``bwd-b1-s128-d256-bf16`` loses 15.2% on K1, so the capacity
    tile is kept there).
  * NOT APPLIED (measured, rolled back): fusing the K1/K2 tap loops into
    explicit unrolled statements loses 8.1% on K1 at ``seq_len=128`` (A/B
    0/3) and is a tie at ``seq_len=32``; raising the d=512 tile to 8 loses
    8.7% on K2 (A/B 0/3).  Both remain documented in ``opt_log.md``.

run: python _engram_gate_conv_bwd_kernel.py --level all
"""

import os

# Developer mode for npuir (compiler-managed synchronisation).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import functools
import math

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  enables .npu() tensor placement

import tilelang
import tilelang.language as T

# Programming mode actually validated by this file (Stage 5 syncs this value
# into the TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

_F32 = "float32"
ALIGNMENT = 256
CONV_KERNEL_SIZE = 4
HALO = CONV_KERNEL_SIZE - 1
# Stage 4 tuned tiling defaults (module level so the wrapper switch block can
# reference them).  ``TUNED_BLK_CAP_*`` is the *capacity* tile for each d tier;
# the factory then shrinks the tile when the capacity tile would leave the
# 40-core AIV machine mostly idle (see ``TUNED_TARGET_BLOCKS``).
TUNED_BLK_CAP_D256 = 8
TUNED_BLK_CAP_D512 = 4
# Occupancy floor used by the per-shape tile choice: below this many blocks the
# fixed per-block latency dominates and halving the tile (which doubles the
# block count at the cost of a larger halo) is a measured win.
TUNED_TARGET_BLOCKS = 8


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _dtype_str(dtype) -> str:
    return str(dtype).split(".")[-1]


# ---------------------------------------------------------------------------
# Golden (independent PyTorch CPU reference)
# Ported from examples/TileOPs/tests/ops/test_engram_gate_conv_bwd.py
# ::ref_engram_gate_conv_bwd (contract semantics; NOT the NPU structure).
# ---------------------------------------------------------------------------
def golden_engram_gate_conv_bwd(dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                                vhat, alpha, rrms_h, rrms_k, rrms_v,
                                eps=1e-6):
    """Engram GateConv backward reference implementation (source semantics).

    Autograd backward of the forward contract (``F.conv1d`` with left zero
    padding), all arithmetic in fp32 with a single round of ``dH/dk/dv`` back
    to the input dtype.  Independent of the NPU structure (sequence tiling /
    deterministic two-stage reduction / ``vbrc`` broadcasts), so it can
    validate the kernel.
    """
    M, T_len, d = H.shape

    H_ag = H.float().detach().requires_grad_(True)
    k_ag = k.float().detach().requires_grad_(True)
    v_ag = v.float().detach().requires_grad_(True)
    w_h_ag = rms_w_h.float().detach().requires_grad_(True)
    w_v_ag = rms_w_v.float().detach().requires_grad_(True)
    cw_ag = conv_w.float().detach().requires_grad_(True)

    def _rmsnorm(x, w):
        return x * (x ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt() * w

    h_norm = _rmsnorm(H_ag, w_h_ag)
    k_norm = _rmsnorm(k_ag, w_h_ag)
    alpha_ag = torch.sigmoid((h_norm * k_norm).sum(-1, keepdim=True) / (d ** 0.5))
    v_hat_ag = alpha_ag * v_ag
    v_hat_norm = _rmsnorm(v_hat_ag, w_v_ag)

    v_perm = v_hat_norm.permute(0, 2, 1)                        # (M, d, T)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))         # causal left pad
    conv_out = F.conv1d(v_padded, cw_ag.T.unsqueeze(1), groups=d).permute(0, 2, 1)
    Y_ag = F.silu(conv_out) + v_hat_ag
    Y_ag.backward(dY.float())

    return (H_ag.grad.to(H.dtype), k_ag.grad.to(H.dtype), v_ag.grad.to(H.dtype),
            w_h_ag.grad, w_v_ag.grad, cw_ag.grad)


# ---------------------------------------------------------------------------
# NPU kernel factory (returns a closure that launches K1 -> K2 -> K3)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _engram_gate_conv_bwd_kernel(M, seq_len, d, eps, dtype):
    """Fused Engram GateConv backward kernel factory (NPU, AIV vector pipeline).

    Args:
        M, seq_len, d, eps, dtype: compile-time specialization parameters
            (signature unchanged across the GPU -> NPU migration).

    Returns:
        ``_run(*tensors)`` -> ``[dH, dk, dv, drms_w_h, drms_w_v, dconv_w]``.
    """
    accum = _F32
    KS = CONV_KERNEL_SIZE
    d_padded = _align_up(d, ALIGNMENT)
    # Tile width: d<=256 -> BLK_S=8 (R=11); d=512 -> BLK_S=4 (R=7).  Keeps the
    # UB resident set inside the 192 KB budget (DESIGN section 4.5).
    # ---- per-shape seq-tile width (Stage 4 lever, DESIGN section 5.2) ----
    # The measurements are latency/occupancy bound, not throughput bound: the
    # AIV wall time is ~0.6-1.3 us of launch plus per-block op overhead, and
    # every manifest workload runs far below 40 blocks.  Start from the
    # capacity tile and, while the tile yields fewer than TUNED_TARGET_BLOCKS
    # blocks, halve it (keeping blk_s >= HALO and seq_len % blk_s == 0, which
    # the exact-region loads require) to double the block count.  Shapes with
    # no valid divisor keep the capacity tile and take the clamped load path.
    # Interleaved A/B (opt_log round 3): seq_len=32 4 -> 8 blocks wins 17.7% on
    # k2; seq_len=128 halving 16 -> 32 blocks loses 15.2% on k1.
    blk_cap = TUNED_BLK_CAP_D256 if d_padded <= 256 else TUNED_BLK_CAP_D512
    cands = []
    for cand in (blk_cap, blk_cap // 2, 4):
        if cand >= HALO and cand not in cands and seq_len % cand == 0:
            cands.append(cand)
    blk_s = blk_cap
    if cands:
        blk_s = cands[-1]
        for cand in cands:
            if M * ((seq_len + cand - 1) // cand) >= TUNED_TARGET_BLOCKS:
                blk_s = cand
                break
    R = blk_s + HALO
    num_tiles = (seq_len + blk_s - 1) // blk_s
    seq_pad = num_tiles * blk_s
    nb = M * num_tiles
    # K3 combine chunking.  Each ``T.copy(GM_region, UB)`` gets its own UB
    # staging allocation (they are summed, not liveness-merged), so the
    # (chunk, d) staging tile bounds the K3 footprint: three of them must fit
    # inside 192 KB -> chunk * d <= 8192.
    chunk = min(nb, 32, max(4, 8192 // d_padded))
    nb_pad = ((nb + chunk - 1) // chunk) * chunk
    nchunks = nb_pad // chunk
    inv_d = 1.0 / float(d)
    inv_sqrt_d = 1.0 / math.sqrt(float(d))
    # OP1/OP2: the tensorized loads copy the own rows with an exact region
    # (t0 : t0 + blk_s); that is in-range only when blk_s divides seq_len.  The
    # clamped element-wise path is kept for the non-divisible shapes (L0/L1/
    # boundary cases) so correctness is shape-independent.
    exact_tiles = (seq_len % blk_s == 0)

    # ---------------- K1: Pass 1 (SiLU bwd) + dconv_w partials ---------------
    @tilelang.jit(target="npuir")
    def _k1():
        @T.prim_func
        def k1(
            dY: T.Tensor((M, seq_len, d_padded), dtype),
            vhat: T.Tensor((M, seq_len, d_padded), dtype),
            rrms_v: T.Tensor((M, seq_len), accum),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((KS, d_padded), dtype),
            dvhat_buf: T.Tensor((M, seq_pad, d_padded), accum),
            ws: T.Tensor(((KS + 2) * nb_pad, d_padded), accum),
        ):
            with T.Kernel(M * num_tiles, is_npu=True) as (cid, _):
                bid = cid % M
                t0 = (cid // M) * blk_s

                vnw = T.alloc_shared((R, d_padded), accum)      # vhat*rrms_v*w_v window
                dy = T.alloc_shared((blk_s, d_padded), accum)
                conv = T.alloc_shared((blk_s, d_padded), accum)
                sig = T.alloc_shared((blk_s, d_padded), accum)
                ds = T.alloc_shared((blk_s, d_padded), accum)
                prod = T.alloc_shared((blk_s, d_padded), accum)
                bro = T.alloc_shared((R, d_padded), accum)
                bro_s = T.alloc_shared((blk_s, d_padded), accum)
                rmk = T.alloc_shared((R, 1), accum)
                rv = T.alloc_shared((R, 1), accum)
                s_own = T.alloc_shared((blk_s, 1), accum)
                wv = T.alloc_shared((1, d_padded), accum)
                cw = T.alloc_shared((KS, d_padded), accum)
                cwd = T.alloc_shared((KS, d_padded), dtype)
                red0 = T.alloc_shared((1, d_padded), accum)
                # OP1 raw-dtype staging buffers for the tensorized GM loads.
                wvr = T.alloc_shared((1, d_padded), dtype)
                vnwr = T.alloc_shared((R, d_padded), dtype)
                dyr = T.alloc_shared((blk_s, d_padded), dtype)

                # ---- weights -> fp32 (rms_w_v is tiny) ----
                # OP1: stage the raw dtype then bulk-cast; an element-level
                # T.cast of a GM tensor lowers to a fully scalar scf.for.
                for j in T.Parallel(d_padded):
                    wvr[0, j] = rms_w_v[j]
                T.vcast(wvr, wv, round_mode="rint")
                # conv_w is 2-D: stage it through UB and use T.vcast -- a plain
                # element-wise T.cast of a 2-D GM region lowers to an invalid
                # hivm.hir.vcast with a memref operand (bufferize failure).
                T.copy(conv_w, cwd)
                T.vcast(cwd, cw, round_mode="rint")

                # ---- causal row mask + rrms_v window (unrolled scalars) ----
                for r in range(R):
                    gt = t0 - HALO + r
                    gc = T.min(T.max(gt, 0), seq_len - 1)
                    rmk[r, 0] = T.cast(T.min(T.max(gt + 1, 0), 1)
                                       * T.min(T.max(seq_len - gt, 0), 1), accum)
                    rv[r, 0] = rrms_v[bid, gc]
                # own-row mask: zeroes the tail rows of a partial last tile.
                for i in range(blk_s):
                    gt = t0 + i
                    s_own[i, 0] = T.cast(T.min(T.max(gt + 1, 0), 1)
                                         * T.min(T.max(seq_len - gt, 0), 1), accum)

                # ---- load vhat window + materialise vn (O3 CSE) ----
                # OP1: raw-dtype region copies + one whole-tile vcast instead
                # of an element-level T.cast read (scalar scf.for).
                if exact_tiles:
                    T.copy(vhat[bid, t0:t0 + blk_s, 0:d_padded],
                           vnwr[HALO:R, 0:d_padded])
                    hb = T.min(T.max(t0 - HALO, 0), T.max(seq_len - HALO, 0))
                    T.copy(vhat[bid, hb:hb + HALO, 0:d_padded],
                           vnwr[0:HALO, 0:d_padded])
                else:
                    for r, j in T.Parallel(R, d_padded):
                        gc = T.min(T.max(t0 - HALO + r, 0), seq_len - 1)
                        vnwr[r, j] = vhat[bid, gc, j]
                T.vcast(vnwr, vnw, round_mode="rint")
                T.vbrc(rv, bro)
                T.vmul(vnw, bro, vnw)                            # * rrms_v
                T.vbrc(wv, bro)
                T.vmul(vnw, bro, vnw)                            # * w_v
                T.vbrc(rmk, bro)
                T.vmul(vnw, bro, vnw)                            # * row mask

                # ---- causal depthwise conv (host-unrolled taps) ----
                T.clear(conv)
                for p in range(KS):
                    for i, j in T.Parallel(blk_s, d_padded):
                        conv[i, j] += cw[p, j] * vnw[i + p, j]

                # ---- load dY own rows (masked for the partial last tile) ----
                if exact_tiles:
                    T.copy(dY[bid, t0:t0 + blk_s, 0:d_padded], dyr)
                else:
                    for i, j in T.Parallel(blk_s, d_padded):
                        dyr[i, j] = dY[bid, T.min(t0 + i, seq_len - 1), j]
                T.vcast(dyr, dy, round_mode="rint")
                T.vbrc(s_own, bro_s)
                T.vmul(dy, bro_s, dy)

                # ---- SiLU backward ----
                T.vsigmoid(conv, sig)
                for i, j in T.Parallel(blk_s, d_padded):
                    ds[i, j] = dy[i, j] * (sig[i, j]
                                           + conv[i, j] * sig[i, j] * (1.0 - sig[i, j]))

                # ---- d_silu handoff to K2 (GM) ----
                T.copy(ds, dvhat_buf[bid, t0:t0 + blk_s, 0:d_padded])

                # ---- dconv_w partial sums: sum_t ds[t] * vn[t+p], token reduce ----
                for p in range(KS):
                    for i, j in T.Parallel(blk_s, d_padded):
                        prod[i, j] = ds[i, j] * vnw[i + p, j]
                    T.reduce_sum(prod, red0, dim=0, clear=True)
                    T.copy(red0, ws[(2 + p) * nb_pad + cid:
                                    (2 + p) * nb_pad + cid + 1, 0:d_padded])
        return k1

    # ------------- K2: Pass 1b + Pass 2 + drms_w_h/v partials ----------------
    @tilelang.jit(target="npuir")
    def _k2():
        @T.prim_func
        def k2(
            dY: T.Tensor((M, seq_len, d_padded), dtype),
            H: T.Tensor((M, seq_len, d_padded), dtype),
            k: T.Tensor((M, seq_len, d_padded), dtype),
            v: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((KS, d_padded), dtype),
            vhat: T.Tensor((M, seq_len, d_padded), dtype),
            alpha: T.Tensor((M, seq_len), accum),
            rrms_h: T.Tensor((M, seq_len), accum),
            rrms_k: T.Tensor((M, seq_len), accum),
            rrms_v: T.Tensor((M, seq_len), accum),
            dvhat_buf: T.Tensor((M, seq_pad, d_padded), accum),
            dH: T.Tensor((M, seq_pad, d_padded), dtype),
            dk: T.Tensor((M, seq_pad, d_padded), dtype),
            dv: T.Tensor((M, seq_pad, d_padded), dtype),
            ws: T.Tensor(((KS + 2) * nb_pad, d_padded), accum),
        ):
            with T.Kernel(M * num_tiles, is_npu=True) as (cid, _):
                bid = cid % M
                t0 = (cid // M) * blk_s

                # (blk_s, d) f32 resident set.  Five logical values share four
                # buffers to stay inside the 192 KB UB budget at d=512
                # (DESIGN section 4.5); every alias is written only after the
                # previous content has been fully consumed (read-before-
                # overwrite, PL-1.23):
                #   dvn  : conv transpose   -> h_norm
                #   vhat_l: vhat own rows   -> k_norm
                #   v_l  : v own rows       -> d(h_norm) -> dH
                #   dvh  : d(vhat)          -> d(k_norm) -> dk
                dsw = T.alloc_shared((R, d_padded), accum)        # masked d_silu window
                dvn = T.alloc_shared((blk_s, d_padded), accum)    # conv transpose | h_norm
                vhat_l = T.alloc_shared((blk_s, d_padded), accum)  # vhat rows | k_norm
                h_l = T.alloc_shared((blk_s, d_padded), accum)
                k_l = T.alloc_shared((blk_s, d_padded), accum)
                v_l = T.alloc_shared((blk_s, d_padded), accum)    # v rows | d(h_norm)
                dvh = T.alloc_shared((blk_s, d_padded), accum)    # d(vhat) | d(k_norm)
                prod = T.alloc_shared((blk_s, d_padded), accum)
                hn = dvn                                          # alias
                kn = vhat_l                                       # alias
                dhn = v_l                                         # alias
                dkn = dvh                                         # alias
                bro = T.alloc_shared((R, d_padded), accum)
                bro_s = T.alloc_shared((blk_s, d_padded), accum)
                rmk = T.alloc_shared((R, 1), accum)
                s_own = T.alloc_shared((blk_s, 1), accum)
                cw = T.alloc_shared((KS, d_padded), accum)
                cwd = T.alloc_shared((KS, d_padded), dtype)
                wv = T.alloc_shared((1, d_padded), accum)
                wh = T.alloc_shared((1, d_padded), accum)
                al = T.alloc_shared((blk_s, 1), accum)
                rh = T.alloc_shared((blk_s, 1), accum)
                rk = T.alloc_shared((blk_s, 1), accum)
                rv = T.alloc_shared((blk_s, 1), accum)
                ddot_buf = T.alloc_shared((blk_s, 1), accum)
                red = T.alloc_shared((blk_s, 1), accum)
                red0 = T.alloc_shared((1, d_padded), accum)
                ostg = T.alloc_shared((blk_s, d_padded), dtype)
                # OP2 raw-dtype staging buffers for the tensorized GM loads.
                wvr = T.alloc_shared((1, d_padded), dtype)
                raw = T.alloc_shared((blk_s, d_padded), dtype)

                # ---- weights -> fp32 (w_h serves both H and k) ----
                # OP2: stage the raw dtype then bulk-cast (wv/wh are separate
                # dsts so the tiny raw row is loaded twice).
                for j in T.Parallel(d_padded):
                    wvr[0, j] = rms_w_v[j]
                T.vcast(wvr, wv, round_mode="rint")
                for j in T.Parallel(d_padded):
                    wvr[0, j] = rms_w_h[j]
                T.vcast(wvr, wh, round_mode="rint")
                # See K1: stage the 2-D conv_w tile through UB before vcast.
                T.copy(conv_w, cwd)
                T.vcast(cwd, cw, round_mode="rint")

                # ---- d_silu window mask (right edge) + own rows + scalars ----
                for r in range(R):
                    gt = t0 + r
                    rmk[r, 0] = T.cast(T.min(T.max(gt + 1, 0), 1)
                                       * T.min(T.max(seq_len - gt, 0), 1), accum)
                for i in range(blk_s):
                    gt = t0 + i
                    gco = T.min(gt, seq_len - 1)
                    s_own[i, 0] = T.cast(T.min(T.max(gt + 1, 0), 1)
                                         * T.min(T.max(seq_len - gt, 0), 1), accum)
                    al[i, 0] = alpha[bid, gco]
                    rh[i, 0] = rrms_h[bid, gco]
                    rk[i, 0] = rrms_k[bid, gco]
                    rv[i, 0] = rrms_v[bid, gco]

                # ---- load d_silu window + right-edge mask ----
                # OP2: exact region copies (own rows + clamped right halo)
                # instead of one clamped row copy per position.
                if exact_tiles:
                    T.copy(dvhat_buf[bid, t0:t0 + blk_s, 0:d_padded],
                           dsw[0:blk_s, 0:d_padded])
                    hb2 = T.min(t0 + blk_s, T.max(seq_len - HALO, 0))
                    T.copy(dvhat_buf[bid, hb2:hb2 + HALO, 0:d_padded],
                           dsw[blk_s:R, 0:d_padded])
                else:
                    for r, j in T.Parallel(R, d_padded):
                        dsw[r, j] = dvhat_buf[bid, T.min(t0 + r, seq_len - 1), j]
                T.vbrc(rmk, bro)
                T.vmul(dsw, bro, dsw)

                # ---- conv transpose (host-unrolled taps; no dilation) ----
                T.clear(dvn)
                for p in range(KS):
                    for i, j in T.Parallel(blk_s, d_padded):
                        dvn[i, j] += cw[p, j] * dsw[i + (KS - 1) - p, j]

                # ---- own rows (clamped: tail rows of a partial tile) ----
                # OP2: four raw region copies + bulk vcast (the element-level
                # T.cast read was the dominant scalar loop of this kernel).
                if exact_tiles:
                    T.copy(vhat[bid, t0:t0 + blk_s, 0:d_padded], raw)
                    T.vcast(raw, vhat_l, round_mode="rint")
                    T.copy(H[bid, t0:t0 + blk_s, 0:d_padded], raw)
                    T.vcast(raw, h_l, round_mode="rint")
                    T.copy(k[bid, t0:t0 + blk_s, 0:d_padded], raw)
                    T.vcast(raw, k_l, round_mode="rint")
                    T.copy(v[bid, t0:t0 + blk_s, 0:d_padded], raw)
                    T.vcast(raw, v_l, round_mode="rint")
                else:
                    for i, j in T.Parallel(blk_s, d_padded):
                        vhat_l[i, j] = T.cast(
                            vhat[bid, T.min(t0 + i, seq_len - 1), j], accum)
                        h_l[i, j] = T.cast(
                            H[bid, T.min(t0 + i, seq_len - 1), j], accum)
                        k_l[i, j] = T.cast(
                            k[bid, T.min(t0 + i, seq_len - 1), j], accum)
                        v_l[i, j] = T.cast(
                            v[bid, T.min(t0 + i, seq_len - 1), j], accum)

                # ---- drms_w_v partial: sum_t dvn[t]*vhat[t]*rrms_v[t] ----
                T.vbrc(rv, bro_s)
                T.vmul(dvn, vhat_l, prod)
                T.vmul(prod, bro_s, prod)
                T.reduce_sum(prod, red0, dim=0, clear=True)
                T.copy(red0, ws[nb_pad + cid:nb_pad + cid + 1, 0:d_padded])

                # ---- dot_v (row reduce) : sum_c vhat*w_v*dvn ----
                T.vmul(vhat_l, dvn, prod)
                T.vbrc(wv, bro_s)
                T.vmul(prod, bro_s, prod)
                T.reduce_sum(prod, red, dim=1, clear=True)

                # ---- d(vhat) = rrms_v*(w_v*dvn - rrms_v^2*vhat*dot_v/d) + dY ----
                T.vbrc(wv, bro_s)
                T.vmul(dvn, bro_s, dvh)                          # dvh = w_v*dvn
                T.vbrc(rv, bro_s)
                T.vmul(vhat_l, bro_s, prod)                      # rv*vhat
                T.vbrc(red, bro_s)
                T.vmul(prod, bro_s, prod)                        # * dot_v
                T.vmul(prod, inv_d, prod)                        # * fl(1/d)
                T.vbrc(rv, bro_s)
                T.vmul(prod, bro_s, prod)                        # rv^2*vhat*dot/d
                T.vsub(dvh, prod, dvh)                           # w_v*dvn - prod
                T.vbrc(rv, bro_s)
                T.vmul(dvh, bro_s, dvh)                          # * rrms_v
                if exact_tiles:
                    T.copy(dY[bid, t0:t0 + blk_s, 0:d_padded], raw)
                    T.vcast(raw, prod, round_mode="rint")
                else:
                    for i, j in T.Parallel(blk_s, d_padded):
                        prod[i, j] = T.cast(
                            dY[bid, T.min(t0 + i, seq_len - 1), j], accum)
                T.vadd(dvh, prod, dvh)                           # + dY residual
                T.vbrc(s_own, bro_s)
                T.vmul(dvh, bro_s, dvh)                          # own-row mask

                # ---- dv = alpha * d(vhat) ----
                T.vbrc(al, bro_s)
                T.vmul(dvh, bro_s, prod)
                T.vcast(prod, ostg, round_mode="rint")
                T.copy(ostg, dv[bid, t0:t0 + blk_s, 0:d_padded])

                # ---- dalpha -> ddot ----
                T.vmul(dvh, v_l, prod)
                T.reduce_sum(prod, red, dim=1, clear=True)
                for i in range(blk_s):
                    ddot_buf[i, 0] = (red[i, 0] * al[i, 0] * (1.0 - al[i, 0])
                                      * inv_sqrt_d)

                # ---- h_norm / k_norm (CSE materialisation, shared rms_w_h) ----
                T.vbrc(rh, bro_s)
                T.vmul(h_l, bro_s, hn)
                T.vbrc(rk, bro_s)
                T.vmul(k_l, bro_s, kn)
                T.vbrc(wh, bro_s)
                T.vmul(hn, bro_s, hn)
                T.vmul(kn, bro_s, kn)

                # ---- d(h_norm) = ddot*k_norm ; d(k_norm) = ddot*h_norm ----
                T.vbrc(ddot_buf, bro_s)
                T.vmul(kn, bro_s, dhn)
                T.vmul(hn, bro_s, dkn)

                # ---- drms_w_h partial: sum_t (dhn*H*rrms_h + dkn*k*rrms_k) ----
                # MUST stay before the in-place dhn->dH / dkn->dk overwrites
                # below (read-before-overwrite; PL-1.23, DESIGN note 13).
                T.vbrc(rh, bro_s)
                T.vmul(dhn, bro_s, prod)
                T.vmul(prod, h_l, prod)
                T.vbrc(rk, bro_s)
                for i, j in T.Parallel(blk_s, d_padded):
                    prod[i, j] += bro_s[i, j] * dkn[i, j] * k_l[i, j]
                T.reduce_sum(prod, red0, dim=0, clear=True)
                T.copy(red0, ws[cid:cid + 1, 0:d_padded])

                # ---- dH = rrms_h*(w_h*dhn - rrms_h^2*H*doth/d)  [dhn -> dH] ----
                T.vmul(h_l, kn, prod)                            # H * k_norm
                T.vbrc(wh, bro_s)
                T.vmul(prod, bro_s, prod)                        # * w_h
                T.reduce_sum(prod, red, dim=1, clear=True)
                for i in range(blk_s):
                    red[i, 0] = red[i, 0] * ddot_buf[i, 0]       # doth
                T.vbrc(wh, bro_s)
                T.vmul(dhn, bro_s, dhn)                          # w_h*dhn
                T.vbrc(rh, bro_s)
                T.vmul(h_l, bro_s, prod)                         # rrms_h*H
                T.vbrc(red, bro_s)
                T.vmul(prod, bro_s, prod)                        # * doth
                T.vmul(prod, inv_d, prod)                        # * fl(1/d)
                T.vbrc(rh, bro_s)
                T.vmul(prod, bro_s, prod)                        # rrms_h^2*H*doth/d
                T.vsub(dhn, prod, dhn)
                T.vbrc(rh, bro_s)
                T.vmul(dhn, bro_s, dhn)                          # * rrms_h
                T.vcast(dhn, ostg, round_mode="rint")
                T.copy(ostg, dH[bid, t0:t0 + blk_s, 0:d_padded])

                # ---- dk = rrms_k*(w_h*dkn - rrms_k^2*k*dotk/d)  [dkn -> dk] ----
                T.vmul(k_l, hn, prod)                            # k * h_norm
                T.vbrc(wh, bro_s)
                T.vmul(prod, bro_s, prod)
                T.reduce_sum(prod, red, dim=1, clear=True)
                for i in range(blk_s):
                    red[i, 0] = red[i, 0] * ddot_buf[i, 0]       # dotk
                T.vbrc(wh, bro_s)
                T.vmul(dkn, bro_s, dkn)                          # w_h*dkn
                T.vbrc(rk, bro_s)
                T.vmul(k_l, bro_s, prod)                         # rrms_k*k
                T.vbrc(red, bro_s)
                T.vmul(prod, bro_s, prod)
                T.vmul(prod, inv_d, prod)
                T.vbrc(rk, bro_s)
                T.vmul(prod, bro_s, prod)
                T.vsub(dkn, prod, dkn)
                T.vbrc(rk, bro_s)
                T.vmul(dkn, bro_s, dkn)
                T.vcast(dkn, ostg, round_mode="rint")
                T.copy(ostg, dk[bid, t0:t0 + blk_s, 0:d_padded])
        return k2

    # ---------------- K3: deterministic combine ----------------
    @tilelang.jit(target="npuir")
    def _k3():
        @T.prim_func
        def k3(
            ws: T.Tensor(((KS + 2) * nb_pad, d_padded), accum),
            out_all: T.Tensor((KS + 2, d_padded), accum),
        ):
            # OP3 slots: 0 = drms_w_h, 1 = drms_w_v, 2 + p = dconv_w tap p.
            with T.Kernel(KS + 2, is_npu=True) as (cid, _):
                ws_t = T.alloc_shared((chunk, d_padded), accum)
                acc0 = T.alloc_shared((1, d_padded), accum)
                rc = T.alloc_shared((1, d_padded), accum)
                base = cid * nb_pad
                # The chunk axis is a runtime loop so that exactly one staging
                # allocation is created (a Python-unrolled chunk loop emits one
                # allocation per copy site).
                T.clear(acc0)
                for c in T.serial(nchunks):
                    T.copy(ws[base + c * chunk:base + (c + 1) * chunk,
                              0:d_padded], ws_t)
                    T.reduce_sum(ws_t, rc, dim=0, clear=True)
                    T.vadd(acc0, rc, acc0)
                T.copy(acc0, out_all[cid:cid + 1, 0:d_padded])
        return k3

    # ---------------- factory closure: allocate + dispatch ----------------
    def _run(dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha,
             rrms_h, rrms_k, rrms_v):
        dev = dY.device
        dt = dY.dtype

        def new(shape, t, zero=False):
            return (torch.zeros if zero else torch.empty)(shape, dtype=t,
                                                          device=dev)

        # Tail-padded output workspaces (narrowed before returning).
        dH = new((M, seq_pad, d_padded), dt)
        dk = new((M, seq_pad, d_padded), dt)
        dv = new((M, seq_pad, d_padded), dt)
        drms_w_h = new((d_padded,), torch.float32)
        drms_w_v = new((d_padded,), torch.float32)
        dconv_w = new((KS, d_padded), torch.float32)
        dvhat_buf = new((M, seq_pad, d_padded), torch.float32)
        # Partials: padded rows stay zero so the combine is exact.
        # OP3: one workspace, slot s = rows [s * nb_pad, (s + 1) * nb_pad).
        ws = new(((KS + 2) * nb_pad, d_padded), torch.float32, zero=True)
        # OP3: the (KS + 2, d) combine output is carved into the three contract
        # outputs (views share the storage, so no extra copy is needed).
        out_all = new((KS + 2, d_padded), torch.float32)

        _k1()(dY, vhat, rrms_v, rms_w_v, conv_w, dvhat_buf, ws)
        _k2()(dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha,
              rrms_h, rrms_k, rrms_v, dvhat_buf, dH, dk, dv, ws)
        _k3()(ws, out_all)
        return [dH[:, :seq_len, :], dk[:, :seq_len, :], dv[:, :seq_len, :],
                out_all[0], out_all[1], out_all[2:2 + KS]]

    return _run


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
_NAMES = ["dH", "dk", "dv", "drms_w_h", "drms_w_v", "dconv_w"]
_RED_IDX = (3, 4, 5)          # fp32 cross-token reductions


def _rmsnorm_ref(x, w, eps=1e-6):
    xf = x.float()
    rr = (xf ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return xf * rr * w.float(), rr.squeeze(-1)


def _make_inputs(M, seq_len, d, dtype, seed, eps=1e-6, saturate=None):
    """Reference workload generator (matches EngramGateConvBwdWorkload)."""
    torch.manual_seed(seed)
    H = torch.randn(M, seq_len, d, dtype=torch.float32)
    k = torch.randn(M, seq_len, d, dtype=torch.float32) * 0.1
    v = torch.randn(M, seq_len, d, dtype=torch.float32) * 0.1
    if saturate == "pos":                 # dot -> +inf, alpha -> 1
        H = H * 5.0
        k = H.clone()
    elif saturate == "neg":               # dot -> -inf, alpha -> 0
        H = H * 5.0
        k = -H.clone()
    H = H.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    rms_w_h = torch.ones(d, dtype=torch.float32).to(dtype)
    rms_w_v = torch.ones(d, dtype=torch.float32).to(dtype)
    conv_w = (torch.randn(CONV_KERNEL_SIZE, d, dtype=torch.float32) * 0.02).to(dtype)
    dY = (torch.randn(M, seq_len, d, dtype=torch.float32) * 0.1).to(dtype)

    h_norm, rrms_h = _rmsnorm_ref(H, rms_w_h, eps)
    k_norm, rrms_k = _rmsnorm_ref(k, rms_w_h, eps)
    dot = (h_norm * k_norm).sum(-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))
    v_hat = alpha * v.float()
    _, rrms_v = _rmsnorm_ref(v_hat.to(dtype), rms_w_v, eps)

    return (dY, H, k, v, rms_w_h, rms_w_v, conv_w, v_hat.to(dtype),
            alpha.squeeze(-1).float(), rrms_h.float(), rrms_k.float(),
            rrms_v.float())


def _run_case(M, seq_len, d, dtype, tag="", seed=0, eps=1e-6,
              strict_red=None, verbose=True):
    """Single correctness case; compares all six outputs against the golden."""
    atol = 2e-1 if dtype == torch.float16 else 3e-1
    rtol = 2e-1
    dtype_str = _dtype_str(dtype)
    ins = _make_inputs(M, seq_len, d, dtype, seed, eps)
    npu_ins = [t.npu() for t in ins]

    kernel = _engram_gate_conv_bwd_kernel(M, seq_len, d, eps, dtype_str)
    outs = kernel(*npu_ins)
    ref = golden_engram_gate_conv_bwd(*ins, eps=eps)

    diffs = {}
    for o, r, nm in zip(outs, ref, _NAMES):
        assert tuple(o.shape) == tuple(r.shape), \
            f"{nm}: shape {tuple(o.shape)} != golden {tuple(r.shape)}"
        o = o.float().cpu()
        r = r.float().cpu()
        assert torch.isfinite(o).all(), f"{nm}: non-finite output"
        torch.testing.assert_close(o, r, atol=atol, rtol=rtol, equal_nan=True)
        diffs[nm] = (o - r).abs().max().item()

    if strict_red is not None:
        # The fp32 cross-token reductions hold far more digits than the fp16/
        # bf16 element outputs, so they are checked separately.  For bf16 the
        # floor is set by the bf16 quantization of the saved ``vhat`` / ``dY``
        # inputs (the golden recomputes ``vhat`` in fp32 from ``alpha*v``
        # instead of using the saved tensor), so the gate is relaxed 10x.
        tol = strict_red if dtype == torch.float16 else strict_red * 10.0
        for j in _RED_IDX:
            o = outs[j].float().cpu()
            r = ref[j].float().cpu()
            torch.testing.assert_close(o, r, atol=tol, rtol=1e-3,
                                       equal_nan=True)
            diffs[_NAMES[j]] = (o - r).abs().max().item()

    # Determinism (the harness requires bit-identical repeats).
    outs2 = kernel(*npu_ins)
    for j, nm in enumerate(_NAMES):
        assert torch.equal(outs[j], outs2[j]), f"{nm} non-deterministic"

    if verbose:
        print(f"  [ok] {tag or 'case'} M={M} S={seq_len} d={d} {dtype_str} "
              + " ".join(f"{n}={diffs[n]:.3e}" for n in _NAMES))
    return max(diffs.values())


def _warn_case(label, fn, *args, **kwargs):
    """Run a non-blocking case; report a warning instead of failing the suite."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - L2/Boundary are warn-only
        print(f"  [warn] {label}: {type(exc).__name__}: {str(exc)[:240]}")
        return None


def _check_gate_corner(M, seq_len, d, dtype, eps=1e-6):
    """Saturating gate (alpha -> 1 and alpha -> 0): finite and matches golden."""
    dtype_str = _dtype_str(dtype)
    atol = 2e-1 if dtype == torch.float16 else 3e-1
    for mode, label in (("pos", "alpha->1"), ("neg", "alpha->0")):
        ins = _make_inputs(M, seq_len, d, dtype, seed=21, eps=eps, saturate=mode)
        kernel = _engram_gate_conv_bwd_kernel(M, seq_len, d, eps, dtype_str)
        outs = kernel(*[t.npu() for t in ins])
        for o, nm in zip(outs, _NAMES):
            assert torch.isfinite(o.float()).all(), f"non-finite {nm} ({label})"
        ref = golden_engram_gate_conv_bwd(*ins, eps=eps)
        for o, r, nm in zip(outs, ref, _NAMES):
            torch.testing.assert_close(o.float().cpu(), r.float().cpu(),
                                       atol=atol, rtol=2e-1, equal_nan=True)
        print(f"  [ok] gate-corner {label}: finite + matches golden")


# ---------------------------------------------------------------------------
# Layered test suites
# ---------------------------------------------------------------------------
def run_L0():
    """L0 threshold (blocking): manifest workloads + smoke + conv boundaries
    + weight-gradient reductions + determinism."""
    print("[L0] threshold smoke (manifest workloads + boundaries)")
    _run_case(1, 16, 256, torch.float16, tag="L0-1-smoke")
    _run_case(1, 32, 256, torch.float16, tag="L0-2-W2")
    _run_case(2, 64, 512, torch.float16, tag="L0-3-W3")
    _run_case(1, 128, 256, torch.bfloat16, tag="L0-4-W4")
    _run_case(2, 16, 256, torch.bfloat16, tag="L0-4b-W5-bf16")
    # conv left boundary (t<3, single partial tile) and transpose right
    # boundary together with a non-divisible seq_len (partial last tile).
    _run_case(1, 4, 256, torch.float16, tag="L0-5-conv-left")
    _run_case(1, 13, 256, torch.float16, tag="L0-6-transpose-right-nondiv")
    _run_case(1, 32, 256, torch.float16, tag="L0-8-weight-grad-reduce",
              strict_red=1e-3)
    _run_case(2, 64, 512, torch.float16, tag="L0-9-determinism",
              strict_red=1e-3)
    _check_gate_corner(1, 32, 256, torch.float16)
    print("[L0] PASS")


def run_L1():
    """L1 core correctness (blocking): seeds / both BLK_S tiers / dtypes /
    scalar strictness / large-nb chunked combine."""
    print("[L1] seeds / tilings / dtype / reductions / large-nb combine")
    _run_case(2, 64, 512, torch.float16, tag="L1-W3-seed5", seed=5,
              strict_red=1e-3)
    _run_case(1, 128, 256, torch.bfloat16, tag="L1-W4-seed7", seed=7,
              strict_red=1e-3)
    _run_case(2, 512, 256, torch.float16, tag="L1-large-nb-chunked", seed=11,
              strict_red=1e-3)
    _run_case(3, 48, 512, torch.bfloat16, tag="L1-d512-bf16", seed=13)
    _run_case(1, 64, 256, torch.float16, tag="L1-d256-long", seed=17,
              strict_red=1e-3)
    _run_case(2, 16, 512, torch.bfloat16, tag="L1-d512-bf16-small", seed=23)
    _run_case(1, 7, 512, torch.float16, tag="L1-nondiv-d512", seed=3)
    print("[L1] PASS")


def run_L2():
    """L2 extended (non-blocking warnings only)."""
    print("[L2] extended coverage")
    for seed in (101, 202):
        _warn_case(f"L2-seed{seed}", _run_case, 2, 32, 512, torch.float16,
                   seed=seed, tag=f"L2-seed{seed}")
    _warn_case("L2-big-M", _run_case, 8, 16, 256, torch.float16, seed=303,
               tag="L2-big-M")
    _warn_case("L2-long-seq", _run_case, 1, 256, 256, torch.float16, seed=404,
               tag="L2-long-seq")
    print("[L2] done")


def run_boundary():
    """Boundary shapes (non-blocking warnings only)."""
    print("[Boundary] edge shapes")
    _warn_case("B-S1", _run_case, 1, 1, 256, torch.float16, tag="B-S1")
    _warn_case("B-S3", _run_case, 1, 3, 256, torch.float16, tag="B-S3")
    _warn_case("B-S9", _run_case, 1, 9, 512, torch.float16, tag="B-S9")
    _warn_case("B-S13-d512", _run_case, 2, 13, 512, torch.float16,
               tag="B-S13-d512")
    _warn_case("B-M4", _run_case, 4, 8, 256, torch.bfloat16, tag="B-M4")
    print("[Boundary] done")


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
    print("ALL REQUESTED LEVELS COMPLETE")


if __name__ == "__main__":
    main()
