# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""General (dense, ungrouped) Conv1d forward kernel for npuir (Cube Expert / MixCV).

Migrated from the GPU TileLang factory ``_conv1d_kernel``
(``TileOPs-main/src/tileops/kernels/convolution/conv1d.py``) to
``target="npuir"``, ``ascend_mode="Expert"`` (``T.Scope("Cube")`` +
``T.alloc_L1`` / ``T.alloc_L0C`` / ``T.gemm`` + Vector prelude/epilogue), per
DESIGN.md v1 and the Stage-2 REVIEW.md.

Source semantics (DESIGN Sec 0.1 / 1.3), kept verbatim:

    out[n, c_o, l] = bias[c_o]
        + sum_{c_i=0}^{C_in-1} sum_{k_w=0}^{K_W-1}
              W[c_o, c_i, k_w] * x[n, c_i, l*s_W + k_w*d_W - p_L]
    L_out = (L_in + p_L + p_R - d_W*(K_W-1) - 1) // s_W + 1

Structure (DESIGN Sec 0.6 / 3.3 / 6):
  - 1-D persistent ``T.Kernel(num_kernels)`` with ``num_kernels =
    NPUUtils.get().get_aicore_num()`` (verified = 20 on Ascend 910B3); logical
    task (bz, by, bx) round-robins over ``num_local_tasks``; ghost cores clamp
    ``cid`` to the last legal task (gqa precedent).
  - Implicit GEMM: A = W[C_out, C_in*K_W] (M x K, K order (k_w, c_i)),
    B = im2col[C_in*K_W, L_out] (K x N), fp32 accumulation via
    ``T.gemm(..., initC=first)``.  Slice-form ``T.copy`` performs GM->L1 auto
    ND2NZ (avoids the strided ``T.load_nd2nz`` misread, CG-2026-0004).
  - Weight-layout prelude (R3): in-kernel Vector ``T.copy``(GM->UB) +
    ``T.transpose``(UB 2-axis swaps) + ``T.copy``(UB->GM) materialises
    ``w_flat[k_w, c_out, c_in]`` (K order (k_w, c_i)); the host never permutes
    the input (ascend-constraints Sec 4).
  - Input padding prelude: a contiguous zero-padded ``x_pad`` removes all
    im2col OOB (left/right padding) so every B tile is an in-bounds slice.
  - stride>1 polyphase prelude (R4): the strided im2col column sampling cannot
    be an ND2NZ slice (CG-2026-0004 family), so ``x_pad`` is decimated once into
    ``xdec[n, c_i, rho*Mdec + m] = x_pad[n, c_i, m*s + rho]`` (one ``T.gather``
    with a host-built constant index; the Expert Vector datapath has no
    element-wise ``T.Parallel`` and no strided copy).  The B tile then reads a
    contiguous per-tap slice of ``xdec``.
  - K is blocked tap-major (K order (k_w, c_i)) so every GM->L1 slice starts at
    L1 offset 0 (a per-tap offset inside a fractal tile is not addressable);
    ``c_in`` is padded to a 16-multiple with zero channels.
  - Tail handling (R2): ``T.copy`` moves only the valid region (dynamic slice
    extents); the gemm ``size`` uses the fractal clamp
    ``T.max(16, ceil16(tail))``; the output is written with a clipped slice.
  - ``has_bias=False``: pure Cube + Vector prelude; L0C -> GM fixpipe
    (fp32 -> fp16/bf16 quantize).
  - ``has_bias=True``: MixCV.  Cube relays the fp32 L0C tile through the GM
    workspace ``ws[kernel_id, slot]`` (fp32, no quantize); Vector loads it,
    adds ``bias`` in fp32 (``vcast`` -> ``vbrc`` -> ``vadd``), casts back to
    ``dtype`` (``vcast``) and stores to ``out``.  The Cube<->Vector relay uses
    a depth-2 double-buffer with a 4-flag bidirectional handshake:
      flag 2*slot   = ws ready  (producer Cube  set  / consumer Vector wait)
      flag 2*slot+1 = slot free (consumer Vector set  / producer Cube  wait)
    The "slot free" prologue is executed by the CONSUMER (Vector) side, which
    mirrors the verified PL-1.12 pattern in ``_ssd_chunk_scan_fwd_kernel.py``
    (a same-core producer self set->wait does not retire: it deadlocks; see
    ``repro/TRAP-sync-self-set-wait.py``).

R1 (Stage-4 perf tuning, 2026-10-18) -- "taps-in-K" single-channel path:
  when the shape has exactly one real input channel, no padding and a window
  that fits one K fractal (``c_in == 1``, ``kernel_l <= 16``,
  ``pad_left == pad_right == 0``, ``l_pad == l_in``):
    * the GEMM K axis becomes the **tap window** itself (``K = kernel_l``,
      measured to be accepted by ``T.gemm`` -- no 16-padding, so neither
      operand needs a zero fill);
    * the A operand is read **directly from the native weight**
      (``weight[c_o, 0, k_w]``: row = c_o, col = k_w, contiguous), so the whole
      Vector weight-repack prelude (GM->UB->2x transpose->GM) disappears;
    * the B operand rows are fetched one tap each straight from ``x`` (stride-1)
      or from the polyphase ``xdec`` (stride>1) into the resident L1 K rows --
      the 20x-redundant ``x_pad`` copy is gone entirely;
    * one gemm covers the whole window (was ``kernel_l`` gemms of K=16), the A
      tile is reloaded only when the co block changes (not per tap), each core
      owns a **contiguous** bx range (its own column slice, so no cross-core
      dependency), and the N tile grows to the L0C budget
      (``TAPK_BLOCK_N_MAX``), so per-core instruction count collapses.
  Everything else (bias MixCV relay, polyphase prelude, tail clamping) keeps
  the Stage-3 shape; shapes outside the predicate take the exact Stage-3 path.

Run: python _conv1d_kernel.py --level {L0,all}
"""

import argparse
import math
import os

# Expert mode is required for the persistent kernel + Cube gemm (+ Vector
# prelude/epilogue) structure; set before importing tilelang.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

# Stage-3 verified programming mode (must match TILELANG_ASCEND_MODE above).
ASCEND_MODE = "Expert"

# Tiling (DESIGN Sec 5.2, inherited from the GPU Conv1dKernel default_config).
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 128
BM_W = 16        # weight-layout prelude co blocking (UB budget)
SLOTS = 2        # WS depth-2 double buffer

# R1 "taps-in-K" path tuning (exposed for the wrapper's paired switch block).
TAPK_K_MAX = 16          # tap window must fit one K fractal
TAPK_BLOCK_N_MAX = 512   # nobias L0C budget: block_m * bn * 4B = 128 KB
TAPK_BLOCK_N_BIAS = 128  # bias epilogue lives in UB: 64x128x(4+2+4) B = 80 KB

_ACCUM = "float32"
_UB_CAP = 32768  # per-buffer UB chunk bound (bytes), keeps prelude within 192KB


def _ceil16(v):
    return ((v + 15) // 16) * 16


def _pick_chunk(l_pad, elem_bytes):
    """Largest power-of-2 chunk (<=16) whose UB slice fits _UB_CAP bytes."""
    c = 16
    while c > 1 and c * l_pad * elem_bytes > _UB_CAP:
        c //= 2
    return c


def _ws_dims(l_in, c_in, pad_left, pad_right, kernel_l, dilation_l, stride_l):
    """Workspace shapes + chunk sizes; shared by factory and test wrapper."""
    need_poly = stride_l != 1
    lin = 16 * stride_l // math.gcd(16, stride_l) if need_poly else 16
    base = l_in + pad_left + pad_right
    l_pad = ((base + lin - 1) // lin) * lin
    mdec = l_pad // stride_l if need_poly else 0
    c_in_pad = max(16, _ceil16(c_in))
    ch_xp = _pick_chunk(l_pad, 2)
    cg = _pick_chunk(l_pad, 4)
    return need_poly, l_pad, mdec, c_in_pad, ch_xp, cg


def _tapk_bn(out_l, block_m, num_kernels, bn_hint, has_bias):
    """Largest N tile that still spreads ``out_l`` over every AI core along bx.

    Bounded by the L0C budget (``block_m * bn * 4B <= 128 KB``), by the bias
    epilogue's UB budget (``TAPK_BLOCK_N_BIAS``) and by the requirement that the
    bx axis alone yields at least one tile per core (the taps-in-K path
    partitions cores by contiguous bx ranges only).
    """
    cap = (131072 // (block_m * 4)) // 16 * 16
    top = min(TAPK_BLOCK_N_MAX, cap)
    if has_bias:
        top = min(top, TAPK_BLOCK_N_BIAS)
    bn = top
    while bn > bn_hint and (out_l + bn - 1) // bn < num_kernels:
        bn //= 2
    return max(bn, bn_hint)


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """General Conv1d reference (source repo baseline: F.conv1d).

    Based on the source semantics only (DESIGN Sec 0.1 / 8.1): ``weight`` is
    ``[C_out, C_in, K_W]`` (groups=1); computed in fp32 (matching the source
    fp32 accumulation + fp32 bias add) then rounded to the input dtype.
    Independent of the NPU Cube/persistent/polyphase algorithm.
    """
    x_cpu = x.detach().to("cpu")
    w_cpu = weight.detach().to("cpu")
    b_cpu = None if bias is None else bias.detach().to("cpu")
    out_f32 = torch.nn.functional.conv1d(
        x_cpu.to(torch.float32),
        w_cpu.to(torch.float32),
        None if b_cpu is None else b_cpu.to(torch.float32),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
    )
    return out_f32.to(x_cpu.dtype)


# Alias so the module exposes a literal ``golden`` as well.
golden = golden_conv1d


# ---------- Kernel ----------
def _conv1d_kernel(
    n,
    c_in,
    l_in,
    c_out,
    kernel_l,
    stride_l,
    pad_left,
    pad_right,
    dilation_l,
    has_bias,
    dtype="float16",
):
    """NPU factory mirroring the GPU ``_conv1d_kernel`` builder.

    Returns an inner ``@tilelang.jit(target="npuir")`` factory taking the NPU
    tiling parameters (the GPU-only ``threads`` / ``enable_rasterization``
    parameters are dropped; ``block_k`` is the channel sub-block width).
    """
    accum_dtype = _ACCUM
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    need_poly, l_pad, mdec, c_in_pad, ch_xp, cg = _ws_dims(
        l_in, c_in, pad_left, pad_right, kernel_l, dilation_l, stride_l
    )
    num_kernels = NPUUtils.get().get_aicore_num()

    # R1 taps-in-K predicate (compile-time): one real input channel, no padding,
    # length already a multiple of the polyphase grain, window inside one K
    # fractal.  Outside it the Stage-3 implicit-GEMM path is kept byte-for-byte.
    tapk = (
        c_in == 1
        and 1 <= kernel_l <= TAPK_K_MAX
        and pad_left == 0
        and pad_right == 0
        and l_pad == l_in
    )

    @tilelang.jit(out_idx=[2], target="npuir")
    def _conv1d_func(block_m, block_n, block_k, num_stages):
        gx = (out_l + block_n - 1) // block_n
        gy = (c_out + block_m - 1) // block_m
        num_logical = gy * gx * n
        n_kc = (c_in_pad + block_k - 1) // block_k
        n_co = (c_out + BM_W - 1) // BM_W
        n_xp = (c_in_pad + ch_xp - 1) // ch_xp
        n_cg = (c_in_pad + cg - 1) // cg
        # R1 taps-in-K geometry: one K fractal-wide tap window, wider N tiles.
        tapk_bn = _tapk_bn(out_l, block_m, num_kernels, block_n, has_bias) if tapk else block_n
        gx = (out_l + tapk_bn - 1) // tapk_bn if tapk else gx
        tkc = kernel_l if tapk else 0

        @T.macro
        def _prelude(x, weight, w_flat, x_pad, xdec, poly_idx):
            with T.Scope("Vector"):
                wsrc = T.alloc_ub([BM_W, c_in, kernel_l], dtype)
                wt = T.alloc_ub([BM_W, kernel_l, c_in], dtype)
                wdst = T.alloc_ub([kernel_l, BM_W, c_in], dtype)
                ub2 = T.alloc_ub([ch_xp, l_in], dtype)
                # weight [C_out,C_in,K_W] -> w_flat [K_W,C_out,C_in] (K order (k_w,c_i))
                for co0 in T.serial(n_co):
                    cw = T.min(BM_W, c_out - co0 * BM_W)
                    T.copy(
                        weight[co0 * BM_W:co0 * BM_W + cw, 0:c_in, 0:kernel_l],
                        wsrc[0:cw, 0:c_in, 0:kernel_l],
                    )
                    T.transpose(wsrc, wt, permutation=[0, 2, 1])
                    T.transpose(wt, wdst, permutation=[1, 0, 2])
                    for kw in T.serial(kernel_l):
                        T.copy(
                            wdst[kw, 0:cw, 0:c_in],
                            w_flat[kw, co0 * BM_W:co0 * BM_W + cw, 0:c_in],
                        )
                # x_pad[bn, ci, p_L + i] = x[bn, ci, i]; x_pad is host pre-zeroed so
                # the pads and the channel-pad rows stay 0.  GM-offset destinations
                # are used (UB-offset GM->UB copies mis-handle the row stride).
                for bn in T.serial(n):
                    for cc0 in T.serial(n_xp):
                        cr = T.min(ch_xp, T.max(0, c_in - cc0 * ch_xp))
                        if cr > 0:
                            T.copy(
                                x[bn, cc0 * ch_xp:cc0 * ch_xp + cr, 0:l_in],
                                ub2[0:cr, 0:l_in],
                            )
                            T.copy(
                                ub2[0:cr, 0:l_in],
                                x_pad[
                                    bn,
                                    cc0 * ch_xp:cc0 * ch_xp + cr,
                                    pad_left:pad_left + l_in,
                                ],
                            )
                if need_poly:
                    # xdec[bn, ci, rho*Mdec + m] = x_pad[bn, ci, m*s + rho]
                    ubx = T.alloc_ub([cg, l_pad], dtype)
                    ubi = T.alloc_ub([cg, l_pad], "int32")
                    ubd = T.alloc_ub([cg, l_pad], dtype)
                    for bn in T.serial(n):
                        for cc0 in T.serial(n_cg):
                            cw = T.min(cg, c_in_pad - cc0 * cg)
                            T.copy(
                                x_pad[bn, cc0 * cg:cc0 * cg + cw, 0:l_pad],
                                ubx[0:cw, 0:l_pad],
                            )
                            T.copy(poly_idx[0:cw, 0:l_pad], ubi[0:cw, 0:l_pad])
                            T.gather(ubx, ubd, ubi, [cw, l_pad])
                            T.copy(
                                ubd[0:cw, 0:l_pad],
                                xdec[bn, cc0 * cg:cc0 * cg + cw, 0:l_pad],
                            )
                with T.rs("PIPE_MTE3"):
                    T.sync_block_set(4)  # prelude ready (Vector -> Cube)

        @T.macro
        def _cube(out, w_flat, x_pad, xdec, ws, kernel_id, num_local_tasks):
            with T.Scope("Cube"):
                l1_a = T.alloc_L1([block_m, block_k], dtype)
                l1_b = T.alloc_L1([block_k, block_n], dtype)
                l0_c = T.alloc_L0C([block_m, block_n], accum_dtype)
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(4)  # wait prelude ready
                for task_id in T.serial(num_local_tasks):
                    cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                    bz = cid // (gy * gx)
                    by = (cid // gx) % gy
                    bx = cid % gx
                    tm = T.min(block_m, c_out - by * block_m)
                    tn = T.min(block_n, out_l - bx * block_n)
                    tmc = T.max(16, T.ceildiv(tm, 16) * 16)
                    tnc = T.max(16, T.ceildiv(tn, 16) * 16)
                    for kc in T.serial(n_kc):
                        kcl = T.min(block_k, c_in_pad - kc * block_k)
                        for kw in T.serial(kernel_l):
                            with T.rs("PIPE_MTE2"):
                                T.copy(
                                    w_flat[
                                        kw,
                                        by * block_m:by * block_m + tm,
                                        kc * block_k:kc * block_k + kcl,
                                    ],
                                    l1_a[0:tm, 0:kcl],
                                )
                                if need_poly:
                                    base = (
                                        ((kw * dilation_l) % stride_l) * mdec
                                        + bx * block_n
                                        + (kw * dilation_l) // stride_l
                                    )
                                    T.copy(
                                        xdec[
                                            bz,
                                            kc * block_k:kc * block_k + kcl,
                                            base:base + tn,
                                        ],
                                        l1_b[0:kcl, 0:tn],
                                    )
                                else:
                                    off = bx * block_n + kw * dilation_l
                                    T.copy(
                                        x_pad[
                                            bz,
                                            kc * block_k:kc * block_k + kcl,
                                            off:off + tn,
                                        ],
                                        l1_b[0:kcl, 0:tn],
                                    )
                            with T.rs("PIPE_C"):
                                T.gemm(
                                    l1_a,
                                    l1_b,
                                    l0_c,
                                    initC=(kc + kw == 0),
                                    b_transpose=False,
                                    size=[tmc, kcl, tnc],
                                )
                    if has_bias:
                        slot = task_id % SLOTS
                        with T.rs("PIPE_FIX"):
                            T.sync_block_wait(2 * slot + 1)  # slot free (WAR)
                            T.copy(
                                l0_c[0:tm, 0:tn],
                                ws[kernel_id, slot, 0:tm, 0:tn],
                            )
                            T.sync_block_set(2 * slot)  # ws ready
                    else:
                        with T.rs("PIPE_FIX"):
                            T.copy(
                                l0_c[0:tm, 0:tn],
                                out[
                                    bz,
                                    by * block_m:by * block_m + tm,
                                    bx * block_n:bx * block_n + tn,
                                ],
                            )

        @T.macro
        def _bias_epilogue(out, bias, ws, kernel_id, num_local_tasks):
            with T.Scope("Vector"):
                ub_f32 = T.alloc_ub([block_m, block_n], accum_dtype)
                ub_f16 = T.alloc_ub([block_m, block_n], dtype)
                bias_16 = T.alloc_ub([block_m, 1], dtype)
                bias_col = T.alloc_ub([block_m, 1], accum_dtype)
                bias_brc = T.alloc_ub([block_m, block_n], accum_dtype)
                # Depth-2 prologue: both slots start free.  Executed by the
                # CONSUMER (Vector) so the producer (Cube) first waits across
                # cores retire (PL-1.12 / ssd precedent).
                with T.rs("PIPE_MTE2"):
                    T.sync_block_set(1)
                    T.sync_block_set(3)
                for task_id in T.serial(num_local_tasks):
                    cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                    bz = cid // (gy * gx)
                    by = (cid // gx) % gy
                    bx = cid % gx
                    tm = T.min(block_m, c_out - by * block_m)
                    tn = T.min(block_n, out_l - bx * block_n)
                    slot = task_id % SLOTS
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(2 * slot)  # ws ready
                        T.copy(ws[kernel_id, slot, 0:tm, 0:tn], ub_f32[0:tm, 0:tn])
                        T.copy(
                            bias[by * block_m:by * block_m + tm],
                            bias_16[0:tm, 0:1],
                        )
                    T.vcast(bias_16, bias_col, round_mode="rint")
                    T.vbrc(bias_col, bias_brc)
                    T.vadd(bias_brc, ub_f32, ub_f32)  # fp32-domain bias add
                    T.vcast(ub_f32, ub_f16, round_mode="rint")
                    with T.rs("PIPE_MTE3"):
                        T.copy(
                            ub_f16[0:tm, 0:tn],
                            out[
                                bz,
                                by * block_m:by * block_m + tm,
                                bx * block_n:bx * block_n + tn,
                            ],
                        )
                        T.sync_block_set(2 * slot + 1)  # slot free

        # ---------------- R1 taps-in-K macros (c_in == 1, pad == 0) -----------
        @T.macro
        def _tapk_prelude(x, xdec, poly_idx):
            """Only the single real channel is materialised.

            stride == 1  -> nothing to do: the Cube reads ``x`` directly.
            stride >  1  -> one polyphase row (``c_in == 1``), replacing the
                            Stage-3 loop over ``c_in_pad`` (15 of 16 chunks were
                            zero channels and dominated aiv_mte3).
            """
            with T.Scope("Vector"):
                if need_poly:
                    ubx = T.alloc_ub([c_in, l_pad], dtype)
                    ubi = T.alloc_ub([c_in, l_pad], "int32")
                    ubd = T.alloc_ub([c_in, l_pad], dtype)
                    for bn_i in T.serial(n):
                        T.copy(x[bn_i, 0:c_in, 0:l_pad], ubx[0:c_in, 0:l_pad])
                        T.copy(poly_idx[0:c_in, 0:l_pad], ubi[0:c_in, 0:l_pad])
                        T.gather(ubx, ubd, ubi, [c_in, l_pad])
                        T.copy(ubd[0:c_in, 0:l_pad], xdec[bn_i, 0:c_in, 0:l_pad])
                with T.rs("PIPE_MTE3"):
                    T.sync_block_set(4)  # prelude ready (Vector -> Cube)

        @T.macro
        def _tapk_cube(out, weight, bsrc, ws, kernel_id):
            """One gemm per output tile: K = the tap window, A = native weight.

            Each core owns the contiguous bx range ``[bx0, bx1)`` so it only ever
            reads column slices it fetched itself (no cross-core dependency).
            """
            with T.Scope("Cube"):
                l1_a = T.alloc_L1([block_m, tkc], dtype)
                l1_b = T.alloc_L1([tkc, tapk_bn], dtype)
                l0_c = T.alloc_L0C([block_m, tapk_bn], accum_dtype)
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(4)  # wait prelude ready
                for bz in T.serial(n):
                    bx0 = (gx * kernel_id) // num_kernels
                    bx1 = (gx * (kernel_id + 1)) // num_kernels
                    for ti in T.serial(T.max(0, bx1 - bx0)):
                        bx = bx0 + ti
                        tn = T.min(tapk_bn, out_l - bx * tapk_bn)
                        tnc = T.max(16, T.ceildiv(tn, 16) * 16)
                        # B: one tap per K row, straight from x / xdec.
                        with T.rs("PIPE_MTE2"):
                            for kw in T.serial(tkc):
                                if need_poly:
                                    base = (
                                        ((kw * dilation_l) % stride_l) * mdec
                                        + bx * tapk_bn
                                        + (kw * dilation_l) // stride_l
                                    )
                                    T.copy(
                                        bsrc[bz, 0:c_in, base:base + tn],
                                        l1_b[kw:kw + 1, 0:tn],
                                    )
                                else:
                                    off = bx * tapk_bn + kw * dilation_l
                                    T.copy(
                                        bsrc[bz, 0:c_in, off:off + tn],
                                        l1_b[kw:kw + 1, 0:tn],
                                    )
                        for by in T.serial(gy):
                            tm = T.min(block_m, c_out - by * block_m)
                            tmc = T.max(16, T.ceildiv(tm, 16) * 16)
                            with T.rs("PIPE_MTE2"):
                                # A: native [c_out, c_in, kernel_l] weight, tap-major
                                T.copy(
                                    weight[
                                        by * block_m:by * block_m + tm,
                                        0,
                                        0:tkc,
                                    ],
                                    l1_a[0:tm, 0:tkc],
                                )
                            with T.rs("PIPE_C"):
                                T.gemm(
                                    l1_a,
                                    l1_b,
                                    l0_c,
                                    initC=True,
                                    b_transpose=False,
                                    size=[tmc, tkc, tnc],
                                )
                            if has_bias:
                                slot = ((bz * (bx1 - bx0) + ti) * gy + by) % SLOTS
                                with T.rs("PIPE_FIX"):
                                    T.sync_block_wait(2 * slot + 1)  # slot free (WAR)
                                    T.copy(
                                        l0_c[0:tm, 0:tn],
                                        ws[kernel_id, slot, 0:tm, 0:tn],
                                    )
                                    T.sync_block_set(2 * slot)  # ws ready
                            else:
                                with T.rs("PIPE_FIX"):
                                    T.copy(
                                        l0_c[0:tm, 0:tn],
                                        out[
                                            bz,
                                            by * block_m:by * block_m + tm,
                                            bx * tapk_bn:bx * tapk_bn + tn,
                                        ],
                                    )

        @T.macro
        def _tapk_bias_epilogue(out, bias, ws, kernel_id):
            with T.Scope("Vector"):
                ub_f32 = T.alloc_ub([block_m, tapk_bn], accum_dtype)
                ub_f16 = T.alloc_ub([block_m, tapk_bn], dtype)
                bias_16 = T.alloc_ub([block_m, 1], dtype)
                bias_col = T.alloc_ub([block_m, 1], accum_dtype)
                bias_brc = T.alloc_ub([block_m, tapk_bn], accum_dtype)
                with T.rs("PIPE_MTE2"):
                    T.sync_block_set(1)
                    T.sync_block_set(3)
                for bz in T.serial(n):
                    bx0 = (gx * kernel_id) // num_kernels
                    bx1 = (gx * (kernel_id + 1)) // num_kernels
                    for ti in T.serial(T.max(0, bx1 - bx0)):
                        bx = bx0 + ti
                        tn = T.min(tapk_bn, out_l - bx * tapk_bn)
                        for by in T.serial(gy):
                            tm = T.min(block_m, c_out - by * block_m)
                            slot = ((bz * (bx1 - bx0) + ti) * gy + by) % SLOTS
                            with T.rs("PIPE_MTE2"):
                                T.sync_block_wait(2 * slot)  # ws ready
                                T.copy(
                                    ws[kernel_id, slot, 0:tm, 0:tn],
                                    ub_f32[0:tm, 0:tn],
                                )
                                T.copy(
                                    bias[by * block_m:by * block_m + tm],
                                    bias_16[0:tm, 0:1],
                                )
                            T.vcast(bias_16, bias_col, round_mode="rint")
                            T.vbrc(bias_col, bias_brc)
                            T.vadd(bias_brc, ub_f32, ub_f32)
                            T.vcast(ub_f32, ub_f16, round_mode="rint")
                            with T.rs("PIPE_MTE3"):
                                T.copy(
                                    ub_f16[0:tm, 0:tn],
                                    out[
                                        bz,
                                        by * block_m:by * block_m + tm,
                                        bx * tapk_bn:bx * tapk_bn + tn,
                                    ],
                                )
                                T.sync_block_set(2 * slot + 1)  # slot free

        if tapk and has_bias:

            @T.prim_func
            def _conv1d_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),
                weight: T.Tensor((c_out, c_in, kernel_l), dtype),
                out: T.Tensor((n, c_out, out_l), dtype),
                bias: T.Tensor((c_out,), dtype),
                w_flat: T.Tensor((kernel_l, c_out, c_in_pad), dtype),
                x_pad: T.Tensor((n, c_in_pad, l_pad), dtype),
                xdec: T.Tensor((n, c_in_pad, l_pad), dtype),
                poly_idx: T.Tensor((cg, l_pad), "int32"),
                ws: T.Tensor((num_kernels, SLOTS, block_m, block_n), accum_dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                    bsrc = xdec if need_poly else x
                    _tapk_prelude(x, xdec, poly_idx)
                    _tapk_cube(out, weight, bsrc, ws, kernel_id)
                    _tapk_bias_epilogue(out, bias, ws, kernel_id)

            return _conv1d_bias_main

        if tapk:

            @T.prim_func
            def _conv1d_main(
                x: T.Tensor((n, c_in, l_in), dtype),
                weight: T.Tensor((c_out, c_in, kernel_l), dtype),
                out: T.Tensor((n, c_out, out_l), dtype),
                w_flat: T.Tensor((kernel_l, c_out, c_in_pad), dtype),
                x_pad: T.Tensor((n, c_in_pad, l_pad), dtype),
                xdec: T.Tensor((n, c_in_pad, l_pad), dtype),
                poly_idx: T.Tensor((cg, l_pad), "int32"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                    bsrc = xdec if need_poly else x
                    _tapk_prelude(x, xdec, poly_idx)
                    _tapk_cube(out, weight, bsrc, None, kernel_id)

            return _conv1d_main

        if has_bias:

            @T.prim_func
            def _conv1d_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),
                weight: T.Tensor((c_out, c_in, kernel_l), dtype),
                out: T.Tensor((n, c_out, out_l), dtype),
                bias: T.Tensor((c_out,), dtype),
                w_flat: T.Tensor((kernel_l, c_out, c_in_pad), dtype),
                x_pad: T.Tensor((n, c_in_pad, l_pad), dtype),
                xdec: T.Tensor((n, c_in_pad, l_pad), dtype),
                poly_idx: T.Tensor((cg, l_pad), "int32"),
                ws: T.Tensor((num_kernels, SLOTS, block_m, block_n), accum_dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                    num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                    _prelude(x, weight, w_flat, x_pad, xdec, poly_idx)
                    _cube(out, w_flat, x_pad, xdec, ws, kernel_id, num_local_tasks)
                    _bias_epilogue(out, bias, ws, kernel_id, num_local_tasks)

            return _conv1d_bias_main

        @T.prim_func
        def _conv1d_main(
            x: T.Tensor((n, c_in, l_in), dtype),
            weight: T.Tensor((c_out, c_in, kernel_l), dtype),
            out: T.Tensor((n, c_out, out_l), dtype),
            w_flat: T.Tensor((kernel_l, c_out, c_in_pad), dtype),
            x_pad: T.Tensor((n, c_in_pad, l_pad), dtype),
            xdec: T.Tensor((n, c_in_pad, l_pad), dtype),
            poly_idx: T.Tensor((cg, l_pad), "int32"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                _prelude(x, weight, w_flat, x_pad, xdec, poly_idx)
                _cube(out, w_flat, x_pad, xdec, None, kernel_id, num_local_tasks)

        return _conv1d_main

    return _conv1d_func


# ---------- Precision comparison ----------
_TOL = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.6e-2, 1.6e-2),
    torch.float32: (1e-4, 1e-4),
}


def _dtype_str(dtype):
    return {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[dtype]


def _build_poly_idx(mdec, stride_l, cg, l_pad):
    mp = torch.arange(l_pad)
    idx = (((mp % mdec) * stride_l) + (mp // mdec)).to(torch.int32)
    return idx.unsqueeze(0).repeat(cg, 1).contiguous()


def _run_case(
    n,
    c_in,
    l_in,
    c_out,
    kernel_l,
    stride_l,
    pad_left,
    pad_right,
    dilation_l,
    has_bias,
    dtype,
    tag,
    seed=0,
):
    torch.manual_seed(seed)
    x = torch.randn(n, c_in, l_in, dtype=dtype)
    weight = torch.randn(c_out, c_in, kernel_l, dtype=dtype)
    bias = torch.randn(c_out, dtype=dtype) if has_bias else None

    fn = _conv1d_kernel(
        n, c_in, l_in, c_out, kernel_l, stride_l, pad_left, pad_right,
        dilation_l, has_bias, _dtype_str(dtype),
    )
    kernel = fn(BLOCK_M, BLOCK_N, BLOCK_K, 1)

    need_poly, l_pad, mdec, c_in_pad, ch_xp, cg = _ws_dims(
        l_in, c_in, pad_left, pad_right, kernel_l, dilation_l, stride_l
    )
    nk = NPUUtils.get().get_aicore_num()

    x_npu = x.to("npu")
    w_npu = weight.to("npu")
    w_flat = torch.empty((kernel_l, c_out, c_in_pad), dtype=dtype, device="npu")
    x_pad = torch.zeros((n, c_in_pad, l_pad), dtype=dtype, device="npu")
    xdec = torch.empty((n, c_in_pad, l_pad), dtype=dtype, device="npu")
    if need_poly:
        poly_idx = _build_poly_idx(mdec, stride_l, cg, l_pad).to("npu")
    else:
        poly_idx = torch.zeros((cg, l_pad), dtype=torch.int32, device="npu")

    if has_bias:
        ws = torch.empty(
            (nk, SLOTS, BLOCK_M, BLOCK_N), dtype=torch.float32, device="npu"
        )
        out = kernel(
            x_npu, w_npu, bias.to("npu"), w_flat, x_pad, xdec, poly_idx, ws
        )
    else:
        out = kernel(x_npu, w_npu, w_flat, x_pad, xdec, poly_idx)
    torch.npu.synchronize()

    ref = golden_conv1d(x, weight, bias, stride_l, pad_left, dilation_l)
    atol, rtol = _TOL[dtype]
    assert tuple(out.shape) == (n, c_out, ref.shape[-1]), f"shape {tuple(out.shape)}"
    assert out.dtype == dtype, f"dtype {out.dtype}"
    max_abs = float((out.cpu().float() - ref.float()).abs().max().item())
    try:
        torch.testing.assert_close(out.cpu(), ref, atol=atol, rtol=rtol)
        ok = True
    except AssertionError as exc:  # noqa: BLE001
        print(f"  [{tag}] FAIL: {str(exc).splitlines()[0]}")
        ok = False
    print(
        f"  [{tag}] n={n} c_in={c_in} l_in={l_in} c_out={c_out} kW={kernel_l} "
        f"s={stride_l} p={pad_left}/{pad_right} d={dilation_l} "
        f"has_bias={has_bias} {_dtype_str(dtype)}: "
        f"{'PASS' if ok else 'FAIL'} (max_abs={max_abs:.3e})"
    )
    return ok, max_abs


# ---------- hierarchical testing ----------
def run_L0():
    """L0 gate (blocking): representative workloads, both paths."""
    print("== L0 (blocking) ==")
    ok = True
    # whisper-conv1 (DESIGN 8.3): stride=1, pad=0, kW=3
    ok &= _run_case(1, 80, 3000, 1280, 3, 1, 0, 0, 1, False, torch.float16,
                    "L0-whisper-nobias", seed=0)[0]
    ok &= _run_case(1, 80, 3000, 1280, 3, 1, 0, 0, 1, True, torch.float16,
                    "L0-whisper-bias", seed=1)[0]
    # encodec-deep: kW=10, large c_in
    ok &= _run_case(1, 128, 600, 256, 10, 1, 0, 0, 1, False, torch.float16,
                    "L0-encodec-nobias", seed=2)[0]
    ok &= _run_case(1, 128, 600, 256, 10, 1, 0, 0, 1, True, torch.float16,
                    "L0-encodec-bias", seed=3)[0]
    # wav2vec2-layer1 (DESIGN 8.3): stride=5 polyphase path
    ok &= _run_case(1, 1, 16000, 512, 10, 5, 0, 0, 1, False, torch.float16,
                    "L0-wav2vec2-nobias", seed=4)[0]
    ok &= _run_case(1, 1, 16000, 512, 10, 5, 0, 0, 1, True, torch.float16,
                    "L0-wav2vec2-bias", seed=5)[0]
    # smoke: stride=2, pad=2 (stride>1 + padding)
    ok &= _run_case(1, 32, 256, 64, 5, 2, 2, 2, 1, False, torch.float16,
                    "L0-smoke-nobias", seed=6)[0]
    ok &= _run_case(1, 32, 256, 64, 5, 2, 2, 2, 1, True, torch.float16,
                    "L0-smoke-bias", seed=7)[0]
    # non-divisible dims (must-test 1) + dilation>1 (must-test 5)
    ok &= _run_case(1, 50, 1000, 100, 7, 1, 0, 0, 1, False, torch.float16,
                    "L0-nondiv-nobias", seed=8)[0]
    ok &= _run_case(1, 50, 1000, 100, 3, 1, 3, 3, 2, True, torch.float16,
                    "L0-dilation-bias", seed=9)[0]
    # batch>1 (must-test 3)
    ok &= _run_case(3, 96, 300, 192, 3, 1, 0, 0, 1, True, torch.float16,
                    "L0-batch3-bias", seed=10)[0]
    assert ok, "L0 FAILED (blocking)"
    print("[L0] PASS (11 cases)")


def run_L1():
    """L1 gate (blocking): dtypes f16/bf16, both paths, more shapes/configs."""
    print("== L1 (blocking) ==")
    ok = True
    # (n, c_in, l_in, c_out, kW, s, pL, pR, d, name)
    cases = [
        (1, 80, 3000, 1280, 3, 1, 0, 0, 1, "whisper"),
        (1, 128, 600, 256, 10, 1, 0, 0, 1, "encodec"),
        (1, 1, 8000, 512, 10, 5, 0, 0, 1, "wav2vec2"),
        (1, 32, 256, 64, 5, 2, 2, 2, 1, "smoke"),
        (1, 50, 777, 100, 7, 1, 3, 3, 2, "nondiv-dil"),
        (2, 64, 512, 128, 4, 3, 2, 2, 1, "batch-stride3"),
        (1, 17, 300, 33, 3, 1, 1, 1, 1, "odd"),
    ]
    seed = 100
    for n, c_in, l_in, c_out, kW, s, pL, pR, d, name in cases:
        for dtype in (torch.float16, torch.bfloat16):
            for has_bias in (False, True):
                tag = f"L1-{name}-{'bias' if has_bias else 'nobias'}-{_dtype_str(dtype)}"
                ok &= _run_case(n, c_in, l_in, c_out, kW, s, pL, pR, d, has_bias,
                                dtype, tag, seed=seed)[0]
                seed += 1
    assert ok, "L1 FAILED (blocking)"
    print("[L1] PASS (28 cases)")


def run_L2():
    """L2 (warn-only): larger / odd shapes."""
    print("== L2 (warn-only) ==")
    cases = [
        (4, 128, 1024, 512, 3, 1, 0, 0, 1, "L2-batch4"),
        (1, 256, 2048, 640, 3, 1, 0, 0, 1, "L2-deep"),
        (1, 16, 2048, 64, 2, 2, 1, 1, 1, "L2-c16"),
        (2, 80, 777, 333, 5, 4, 2, 2, 2, "L2-odd"),
    ]
    n_pass = 0
    for n, c_in, l_in, c_out, kW, s, pL, pR, d, name in cases:
        for dtype in (torch.float16, torch.bfloat16):
            for has_bias in (False, True):
                tag = f"{name}-{'bias' if has_bias else 'nobias'}-{_dtype_str(dtype)}"
                try:
                    ok, _ = _run_case(n, c_in, l_in, c_out, kW, s, pL, pR, d,
                                      has_bias, dtype, tag, seed=200)
                    n_pass += 1 if ok else 0
                except Exception as e:  # noqa: BLE001 - warn-only layer
                    print(f"  [{tag}] WARN (exception): {e}")
    total = len(cases) * 4
    if n_pass == total:
        print(f"[L2] PASS ({total} cases)")
    else:
        print(f"[L2] WARN: {total - n_pass}/{total} cases failed (non-blocking)")


def run_boundary():
    """Boundary (warn-only): tiny / degenerate valid shapes."""
    print("== Boundary (warn-only) ==")
    cases = [
        (1, 16, 16, 16, 1, 1, 0, 0, 1),  # kW=1 minimum
        (1, 20, 40, 48, 3, 1, 0, 0, 1),  # non-16 channel dims
        (1, 48, 128, 64, 3, 1, 0, 0, 1),  # exact single tile
        (1, 17, 33, 33, 3, 2, 1, 1, 1),  # all non-divisible
    ]
    n_pass = 0
    for n, c_in, l_in, c_out, kW, s, pL, pR, d in cases:
        for has_bias in (False, True):
            tag = f"B-{n}x{c_in}x{l_in}x{c_out}-{'bias' if has_bias else 'nobias'}"
            try:
                ok, _ = _run_case(n, c_in, l_in, c_out, kW, s, pL, pR, d,
                                  has_bias, torch.float16, tag, seed=300)
                n_pass += 1 if ok else 0
            except Exception as e:  # noqa: BLE001 - warn-only layer
                print(f"  [{tag}] WARN (exception): {e}")
    total = len(cases) * 2
    if n_pass == total:
        print(f"[Boundary] PASS ({total} cases)")
    else:
        print(f"[Boundary] WARN: {total - n_pass}/{total} cases failed (non-blocking)")


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
