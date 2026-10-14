# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Grouped (groups>1) Conv1d forward kernel for npuir (Cube Expert / MixCV).

Migrated from the GPU TileLang factory ``_conv1d_group_kernel``
(``TileOPs-main/src/tileops/kernels/convolution/conv1d.py``; builder L405-539,
``GroupConv1dKernel`` L929-1071) to ``target="npuir"``, ``ascend_mode="Expert"``
(``T.Scope("Cube")`` + ``T.alloc_L1`` / ``T.alloc_L0C`` / ``T.gemm`` + Vector
prelude/epilogue), per DESIGN.md v1 and the Stage-2 REVIEW.md.

Source semantics (DESIGN Sec 0.1 / 1.3), kept verbatim:

    out[n, c_o, l] = bias[c_o]
        + sum_{c_i^g=0}^{C_in^g-1} sum_{k_w=0}^{K_W-1}
              W[c_o, c_i^g, k_w] * x[n, g(c_o)*C_in^g + c_i^g, l*s_W + k_w*d_W - p_L]
    g(c_o) = c_o // C_out^g ; C_in^g = C_in/groups ; C_out^g = C_out/groups
    L_out = (L_in + p_L + p_R - d_W*(K_W-1) - 1)//s_W + 1

This is the GENERAL ``_conv1d_kernel`` plus a GROUP axis: every group is an
independent implicit GEMM over K = C_in^g * K_W, and the groups partition the
output channels.  Structurally it reuses the function-2 kernel (persistent 1-D
``T.Kernel`` + ``cid`` decode, tap-major K, ``w_flat`` / ``x_gpad`` / polyphase
preludes, depth-2 MixCV bias handshake) and ADDS: ``n*groups`` folded into the
logical-task count, ``bz -> (batch_id, group_id)`` decode, and per-group indexing
of A / B / bias / out (``oc_base = group_id*C_out^g + by*block_M``); the output
slice extent ``tm = min(block_M, C_out^g - by*block_M)`` is measured against
C_out^g so a tail tile never spills into the next group (source
``c_out_g % block_m`` branch), so the GROUP-OVERFLOW protection is structural.

Source-key difference from function 2: the source group builder has NO host
``weight.permute`` (``GroupConv1dKernel.forward`` passes ``weight
(C_out, C_in_g, K_W)`` straight to ``launch``), so the A-operand K-order reorder
is done entirely in-kernel by the R3 prelude (per-group ``T.transpose`` chain).

Structure (DESIGN Sec 0.6 / 3.3 / 6):
  - 1-D persistent ``T.Kernel(num_kernels)`` with ``num_kernels =
    NPUUtils.get().get_aicore_num()`` (verified = 20 on Ascend 910B3); logical
    task (bz, by, bx) with ``bz = batch_id*groups + group_id`` round-robins over
    ``num_local_tasks``; ghost cores clamp ``cid`` to the last legal task.
  - Implicit GEMM: A = w_flat[group_id, k_w, C_out^g, C_in_g_pad] (K order
    (k_w, c_i^g)), B = per-group im2col (contiguous per-tap rectangle), fp32
    accumulation via ``T.gemm(..., initC=(kc+kw==0))``.  Slice-form ``T.copy``
    performs GM->L1 auto ND2NZ (avoids the strided ``T.load_nd2nz`` misread,
    CG-2026-0004).
  - Weight-layout prelude (R3): in-kernel Vector ``T.copy``(GM->UB) +
    ``T.transpose``(UB 2-axis swaps) + ``T.copy``(UB->GM), once per group,
    materialises ``w_flat[g, k_w, c_o^g, c_i^g]`` (K order (k_w, c_i^g)); the
    host never permutes the input (ascend-constraints Sec 4).
  - Input padding prelude (R4): a contiguous zero-padded ``x_gpad`` holds the
    per-group channel segment (``x_gpad[bn,g,ci,p_L+i]=x[bn,g*C_in^g+ci,i]``),
    removing all im2col OOB so every B tile is an in-bounds slice.
  - stride>1 polyphase prelude (R4): the strided im2col column sampling cannot
    be an ND2NZ slice (CG-2026-0004 family), so ``x_gpad`` is decimated once per
    (bn, g) into ``xdec_g[bn,g,ci,rho*Mdec+m] = x_gpad[bn,g,ci,m*s+rho]`` (one
    ``T.gather`` with a host-built constant index).
  - K is blocked tap-major (K order (k_w, c_i^g)) so every GM->L1 slice starts
    at L1 offset 0; ``c_in_g`` is padded to a 16-multiple with zero channels.
  - Tail handling (R2): ``T.copy`` moves only the valid region; the gemm
    ``size`` uses the fractal clamp ``T.max(16, ceil16(tail))``; the output is
    written with a clipped slice; ``tm`` is measured against ``c_out_g`` so the
    output slice stops exactly at the group's last channel.
  - ``has_bias=False``: pure Cube + Vector prelude; L0C -> GM fixpipe.
  - ``has_bias=True``: MixCV.  Cube relays the fp32 L0C tile through the GM
    workspace ``ws[kernel_id, slot]`` (fp32, no quantize); Vector loads it, adds
    ``bias[oc_base:+tm]`` in fp32 (``vcast`` -> ``vbrc`` -> ``vadd``), casts
    back to ``dtype`` (``vcast``) and stores to ``out``.  Depth-2 double-buffer
    with a 4-flag bidirectional handshake:
      flag 2*slot   = ws ready  (producer Cube  set  / consumer Vector wait)
      flag 2*slot+1 = slot free (consumer Vector set  / producer Cube  wait)
    The "slot free" prologue runs on the CONSUMER (Vector) side (mirrors the
    verified PL-1.12 pattern; a same-core producer self set->wait deadlocks, see
    ``_conv1d_pointwise_kernel/repro/TRAP-sync-self-set-wait.py``).

Run: python _conv1d_group_kernel.py --level {L0,all}
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

# Tiling (DESIGN Sec 5.2, inherited from the GPU GroupConv1dKernel default_config).
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 128
BM_W = 16        # weight-layout prelude co blocking (UB budget)
SLOTS = 2        # WS depth-2 double buffer

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


def _plan(l_in, c_in_g, pad_left, pad_right, stride_l):
    """Workspace shapes + chunk sizes; shared by the factory and test wrapper."""
    need_poly = stride_l != 1
    lin = 16 * stride_l // math.gcd(16, stride_l) if need_poly else 16
    base = l_in + pad_left + pad_right
    l_pad = ((base + lin - 1) // lin) * lin
    mdec = l_pad // stride_l if need_poly else 0
    c_in_g_pad = max(16, _ceil16(c_in_g))
    ch_xp = _pick_chunk(l_pad, 2)
    cg = _pick_chunk(l_pad, 4)
    return need_poly, l_pad, mdec, c_in_g_pad, ch_xp, cg


def _build_poly_idx(mdec, stride_l, cg, l_pad):
    """Constant gather index: xdec[rho*Mdec+m] = x_gpad[m*s + rho]."""
    mp = torch.arange(l_pad)
    idx = (((mp % mdec) * stride_l) + (mp // mdec)).to(torch.int32)
    return idx.unsqueeze(0).repeat(cg, 1).contiguous()


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_conv1d_group(x, weight, bias=None, stride=1, padding=0, dilation=1,
                        groups=1):
    """Grouped Conv1d reference (source repo baseline: F.conv1d(groups=groups)).

    Based on the source semantics only (DESIGN Sec 0.1 / 8.1): ``weight`` is
    ``[C_out, C_in/groups, K_W]``; computed in fp32 (matching the source fp32
    accumulation + fp32 bias add) then rounded to the input dtype.  Independent
    of the NPU per-group Cube/persistent/polyphase algorithm.
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
        groups=groups,
    )
    return out_f32.to(x_cpu.dtype)


# Alias so the module exposes a literal ``golden`` as well.
golden = golden_conv1d_group


# ---------- Kernel ----------
def _conv1d_group_kernel(
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
    groups=1,
    c_in_g=0,
    c_out_g=0,
):
    """NPU factory mirroring the GPU ``_conv1d_group_kernel`` builder.

    Returns an inner ``@tilelang.jit(target="npuir")`` factory taking the NPU
    tiling parameters (the GPU-only ``threads`` / ``enable_rasterization``
    parameters are dropped; ``block_k`` is the channel sub-block width).
    """
    accum_dtype = _ACCUM
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    c_in_g = c_in_g if c_in_g > 0 else c_in // groups
    c_out_g = c_out_g if c_out_g > 0 else c_out // groups
    need_poly, l_pad, mdec, c_in_g_pad, ch_xp, cg = _plan(
        l_in, c_in_g, pad_left, pad_right, stride_l
    )
    num_kernels = NPUUtils.get().get_aicore_num()

    @tilelang.jit(out_idx=[2], target="npuir")
    def _conv1d_group_func(block_m, block_n, block_k, num_stages):
        gx = (out_l + block_n - 1) // block_n
        gy = (c_out_g + block_m - 1) // block_m
        num_logical = gy * gx * n * groups              # group axis folded in
        n_kc = (c_in_g_pad + block_k - 1) // block_k
        n_co = (c_out_g + BM_W - 1) // BM_W
        n_xp = (c_in_g_pad + ch_xp - 1) // ch_xp
        n_cg = (c_in_g_pad + cg - 1) // cg

        @T.macro
        def _prelude(x, weight, w_flat, x_gpad, xdec_g, poly_idx):
            with T.Scope("Vector"):
                wsrc = T.alloc_ub([BM_W, c_in_g, kernel_l], dtype)
                wt = T.alloc_ub([BM_W, kernel_l, c_in_g], dtype)
                wdst = T.alloc_ub([kernel_l, BM_W, c_in_g], dtype)
                ub2 = T.alloc_ub([ch_xp, l_in], dtype)
                # weight [C_out,C_in_g,K_W] -> w_flat [groups,K_W,C_out^g,C_in^g]
                # (K order (k_w,c_i^g)), once per group.
                for g in T.serial(groups):
                    for co0 in T.serial(n_co):
                        cw = T.min(BM_W, c_out_g - co0 * BM_W)
                        T.copy(
                            weight[
                                g * c_out_g + co0 * BM_W:g * c_out_g + co0 * BM_W + cw,
                                0:c_in_g,
                                0:kernel_l,
                            ],
                            wsrc[0:cw, 0:c_in_g, 0:kernel_l],
                        )
                        T.transpose(wsrc, wt, permutation=[0, 2, 1])
                        T.transpose(wt, wdst, permutation=[1, 0, 2])
                        for kw in T.serial(kernel_l):
                            T.copy(
                                wdst[kw, 0:cw, 0:c_in_g],
                                w_flat[g, kw, co0 * BM_W:co0 * BM_W + cw, 0:c_in_g],
                            )
                # x_gpad[bn, g, ci, p_L + i] = x[bn, g*C_in^g + ci, i]; host
                # pre-zeroed so pads / channel-pad rows stay 0.  GM-side offset
                # (UB-offset GM->UB copies mis-handle the row stride).
                for bn in T.serial(n):
                    for g in T.serial(groups):
                        for cc0 in T.serial(n_xp):
                            cr = T.min(ch_xp, T.max(0, c_in_g - cc0 * ch_xp))
                            if cr > 0:
                                T.copy(
                                    x[
                                        bn,
                                        g * c_in_g + cc0 * ch_xp:
                                        g * c_in_g + cc0 * ch_xp + cr,
                                        0:l_in,
                                    ],
                                    ub2[0:cr, 0:l_in],
                                )
                                T.copy(
                                    ub2[0:cr, 0:l_in],
                                    x_gpad[
                                        bn,
                                        g,
                                        cc0 * ch_xp:cc0 * ch_xp + cr,
                                        pad_left:pad_left + l_in,
                                    ],
                                )
                if need_poly:
                    # xdec_g[bn,g,ci,rho*Mdec+m] = x_gpad[bn,g,ci,m*s+rho]
                    ubx = T.alloc_ub([cg, l_pad], dtype)
                    ubi = T.alloc_ub([cg, l_pad], "int32")
                    ubd = T.alloc_ub([cg, l_pad], dtype)
                    for bn in T.serial(n):
                        for g in T.serial(groups):
                            for cc0 in T.serial(n_cg):
                                cw = T.min(cg, c_in_g_pad - cc0 * cg)
                                T.copy(
                                    x_gpad[bn, g, cc0 * cg:cc0 * cg + cw, 0:l_pad],
                                    ubx[0:cw, 0:l_pad],
                                )
                                T.copy(poly_idx[0:cw, 0:l_pad], ubi[0:cw, 0:l_pad])
                                T.gather(ubx, ubd, ubi, [cw, l_pad])
                                T.copy(
                                    ubd[0:cw, 0:l_pad],
                                    xdec_g[bn, g, cc0 * cg:cc0 * cg + cw, 0:l_pad],
                                )
                with T.rs("PIPE_MTE3"):
                    T.sync_block_set(4)  # prelude ready (Vector -> Cube)

        @T.macro
        def _cube(out, w_flat, x_gpad, xdec_g, ws, kernel_id, num_local_tasks):
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
                    batch_id = bz // groups                 # group-axis decode
                    group_id = bz % groups
                    oc_base = group_id * c_out_g + by * block_m
                    tm = T.min(block_m, c_out_g - by * block_m)
                    tn = T.min(block_n, out_l - bx * block_n)
                    tmc = T.max(16, T.ceildiv(tm, 16) * 16)
                    tnc = T.max(16, T.ceildiv(tn, 16) * 16)
                    for kc in T.serial(n_kc):
                        kcl = T.min(block_k, c_in_g_pad - kc * block_k)
                        for kw in T.serial(kernel_l):
                            with T.rs("PIPE_MTE2"):
                                # A: this group + tap channel block (L1 offset 0).
                                T.copy(
                                    w_flat[
                                        group_id,
                                        kw,
                                        by * block_m:by * block_m + tm,
                                        kc * block_k:kc * block_k + kcl,
                                    ],
                                    l1_a[0:tm, 0:kcl],
                                )
                                # B: per-group channel-segment per-tap rectangle.
                                if need_poly:
                                    base = (
                                        ((kw * dilation_l) % stride_l) * mdec
                                        + bx * block_n
                                        + (kw * dilation_l) // stride_l
                                    )
                                    T.copy(
                                        xdec_g[
                                            batch_id,
                                            group_id,
                                            kc * block_k:kc * block_k + kcl,
                                            base:base + tn,
                                        ],
                                        l1_b[0:kcl, 0:tn],
                                    )
                                else:
                                    off = bx * block_n + kw * dilation_l
                                    T.copy(
                                        x_gpad[
                                            batch_id,
                                            group_id,
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
                                    batch_id,
                                    oc_base:oc_base + tm,
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
                    batch_id = bz // groups
                    group_id = bz % groups
                    oc_base = group_id * c_out_g + by * block_m
                    tm = T.min(block_m, c_out_g - by * block_m)
                    tn = T.min(block_n, out_l - bx * block_n)
                    slot = task_id % SLOTS
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(2 * slot)  # ws ready
                        T.copy(ws[kernel_id, slot, 0:tm, 0:tn], ub_f32[0:tm, 0:tn])
                        T.copy(
                            bias[oc_base:oc_base + tm],
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
                                batch_id,
                                oc_base:oc_base + tm,
                                bx * block_n:bx * block_n + tn,
                            ],
                        )
                        T.sync_block_set(2 * slot + 1)  # slot free

        if has_bias:

            @T.prim_func
            def _conv1d_group_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),
                weight: T.Tensor((c_out, c_in_g, kernel_l), dtype),
                out: T.Tensor((n, c_out, out_l), dtype),
                bias: T.Tensor((c_out,), dtype),
                w_flat: T.Tensor(
                    (groups, kernel_l, c_out_g, c_in_g_pad), dtype
                ),
                x_gpad: T.Tensor((n, groups, c_in_g_pad, l_pad), dtype),
                xdec_g: T.Tensor((n, groups, c_in_g_pad, l_pad), dtype),
                poly_idx: T.Tensor((cg, l_pad), "int32"),
                ws: T.Tensor(
                    (num_kernels, SLOTS, block_m, block_n), accum_dtype
                ),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                    num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                    _prelude(x, weight, w_flat, x_gpad, xdec_g, poly_idx)
                    _cube(out, w_flat, x_gpad, xdec_g, ws, kernel_id, num_local_tasks)
                    _bias_epilogue(out, bias, ws, kernel_id, num_local_tasks)

            return _conv1d_group_bias_main

        @T.prim_func
        def _conv1d_group_main(
            x: T.Tensor((n, c_in, l_in), dtype),
            weight: T.Tensor((c_out, c_in_g, kernel_l), dtype),
            out: T.Tensor((n, c_out, out_l), dtype),
            w_flat: T.Tensor((groups, kernel_l, c_out_g, c_in_g_pad), dtype),
            x_gpad: T.Tensor((n, groups, c_in_g_pad, l_pad), dtype),
            xdec_g: T.Tensor((n, groups, c_in_g_pad, l_pad), dtype),
            poly_idx: T.Tensor((cg, l_pad), "int32"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                _prelude(x, weight, w_flat, x_gpad, xdec_g, poly_idx)
                _cube(out, w_flat, x_gpad, xdec_g, None, kernel_id, num_local_tasks)

        return _conv1d_group_main

    return _conv1d_group_func


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
    groups,
    tag,
    seed=0,
):
    assert c_in % groups == 0 and c_out % groups == 0, "groups must divide channels"
    torch.manual_seed(seed)
    x = torch.randn(n, c_in, l_in, dtype=dtype)
    c_in_g = c_in // groups
    c_out_g = c_out // groups
    weight = torch.randn(c_out, c_in_g, kernel_l, dtype=dtype)
    bias = torch.randn(c_out, dtype=dtype) if has_bias else None

    fn = _conv1d_group_kernel(
        n, c_in, l_in, c_out, kernel_l, stride_l, pad_left, pad_right,
        dilation_l, has_bias, _dtype_str(dtype), groups, c_in_g, c_out_g,
    )
    kernel = fn(BLOCK_M, BLOCK_N, BLOCK_K, 1)

    need_poly, l_pad, mdec, c_in_g_pad, ch_xp, cg = _plan(
        l_in, c_in_g, pad_left, pad_right, stride_l
    )
    nk = NPUUtils.get().get_aicore_num()

    x_npu = x.to("npu")
    w_npu = weight.to("npu")
    # w_flat / x_gpad are host pre-zeroed so pad columns/channels stay 0
    # (0 * 0 = 0 independent of the other operand's bit pattern).
    w_flat = torch.zeros((groups, kernel_l, c_out_g, c_in_g_pad), dtype=dtype,
                         device="npu")
    x_gpad = torch.zeros((n, groups, c_in_g_pad, l_pad), dtype=dtype, device="npu")
    xdec_g = torch.empty((n, groups, c_in_g_pad, l_pad), dtype=dtype, device="npu")
    if need_poly:
        poly_idx = _build_poly_idx(mdec, stride_l, cg, l_pad).to("npu")
    else:
        poly_idx = torch.zeros((cg, l_pad), dtype=torch.int32, device="npu")

    if has_bias:
        ws = torch.empty(
            (nk, SLOTS, BLOCK_M, BLOCK_N), dtype=torch.float32, device="npu"
        )
        out = kernel(
            x_npu, w_npu, bias.to("npu"), w_flat, x_gpad, xdec_g, poly_idx, ws
        )
    else:
        out = kernel(x_npu, w_npu, w_flat, x_gpad, xdec_g, poly_idx)
    torch.npu.synchronize()

    ref = golden_conv1d_group(x, weight, bias, stride_l, pad_left, dilation_l, groups)
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
        f"s={stride_l} p={pad_left}/{pad_right} d={dilation_l} groups={groups} "
        f"(c_in_g={c_in_g},c_out_g={c_out_g}) has_bias={has_bias} "
        f"{_dtype_str(dtype)}: {'PASS' if ok else 'FAIL'} (max_abs={max_abs:.3e})"
    )
    return ok, max_abs


# ---------- hierarchical testing ----------
def run_L0():
    """L0 gate (blocking): representative grouped workloads, both paths."""
    print("== L0 (blocking) ==")
    ok = True
    # grouped smoke (DESIGN 8.3): groups=2, stride=1, pad=1, kW=3, f16
    ok &= _run_case(1, 32, 128, 64, 3, 1, 1, 1, 1, False, torch.float16, 2,
                    "L0-smoke-nobias", seed=0)[0]
    ok &= _run_case(1, 32, 128, 64, 3, 1, 1, 1, 1, True, torch.float16, 2,
                    "L0-smoke-bias", seed=1)[0]
    # full-groups2-k3-fp16
    ok &= _run_case(1, 32, 128, 64, 3, 1, 1, 1, 1, True, torch.float16, 2,
                    "L0-groups2-k3-bias", seed=2)[0]
    # full-groups3-coutg24-fp16 (c_out_g=24 < block_M -> group tail)
    ok &= _run_case(1, 48, 128, 72, 3, 1, 1, 1, 1, True, torch.float16, 3,
                    "L0-groups3-coutg24-bias", seed=3)[0]
    # group overflow: c_out_g=96 = block_M + 32 (M tail inside a group)
    ok &= _run_case(1, 32, 128, 192, 3, 1, 1, 1, 1, True, torch.float16, 2,
                    "L0-group-overflow-bias", seed=4)[0]
    # groups=4, kW=7
    ok &= _run_case(1, 32, 200, 128, 7, 1, 3, 3, 1, False, torch.float16, 4,
                    "L0-groups4-kw7-nobias", seed=5)[0]
    # stride>1 + padding (polyphase), groups=2, kW=5
    ok &= _run_case(1, 32, 256, 64, 5, 2, 2, 2, 1, False, torch.float16, 2,
                    "L0-stride2-nobias", seed=6)[0]
    ok &= _run_case(1, 32, 256, 64, 5, 2, 2, 2, 1, True, torch.bfloat16, 2,
                    "L0-stride2-bias-bf16", seed=7)[0]
    # dilation>1, groups>1
    ok &= _run_case(1, 32, 256, 64, 3, 1, 3, 3, 2, True, torch.float16, 2,
                    "L0-dilation-bias", seed=8)[0]
    # batch>1 & groups>1
    ok &= _run_case(3, 32, 128, 64, 3, 1, 1, 1, 1, True, torch.float16, 2,
                    "L0-batch3-bias", seed=9)[0]
    # groups=8, small c_in_g
    ok &= _run_case(1, 32, 128, 64, 3, 1, 1, 1, 1, False, torch.float16, 8,
                    "L0-groups8-nobias", seed=10)[0]
    # non-divisible dims (c_in_g=25, c_out_g=50), kW=10, groups=2
    ok &= _run_case(1, 50, 300, 100, 10, 1, 0, 0, 1, True, torch.float16, 2,
                    "L0-nondiv-kw10-bias", seed=11)[0]
    assert ok, "L0 FAILED (blocking)"
    print("[L0] PASS (12 cases)")


def run_L1():
    """L1 gate (blocking): dtypes f16/bf16, group counts, stride/dil/pad."""
    print("== L1 (blocking) ==")
    ok = True
    # (n, c_in, l_in, c_out, kW, s, pL, pR, d, groups, name)
    cases = [
        (1, 32, 128, 64, 3, 1, 1, 1, 1, 2, "g2-smoke"),
        (1, 48, 200, 96, 7, 1, 3, 3, 1, 2, "g2-kw7"),
        (1, 64, 128, 128, 10, 1, 0, 0, 1, 2, "g2-kw10"),
        (1, 32, 300, 64, 5, 2, 2, 2, 1, 2, "g2-stride2"),
        (1, 32, 128, 64, 3, 1, 1, 1, 1, 4, "g4-smoke"),
        (1, 64, 256, 128, 7, 2, 2, 2, 1, 4, "g4-stride2-kw7"),
        (1, 32, 128, 64, 3, 1, 1, 1, 1, 8, "g8-smoke"),
        (2, 64, 128, 128, 3, 2, 1, 1, 1, 8, "g8-batch2-stride2"),
        (1, 32, 128, 64, 7, 1, 1, 1, 1, 1, "g1-kw7"),
        (1, 48, 200, 96, 10, 1, 5, 5, 2, 2, "g2-kw10-dil2"),
    ]
    seed = 100
    for n, c_in, l_in, c_out, kW, s, pL, pR, d, groups, name in cases:
        for dtype in (torch.float16, torch.bfloat16):
            for has_bias in (False, True):
                tag = (f"L1-{name}-{'bias' if has_bias else 'nobias'}-"
                       f"{_dtype_str(dtype)}")
                ok &= _run_case(n, c_in, l_in, c_out, kW, s, pL, pR, d,
                                has_bias, dtype, groups, tag, seed=seed)[0]
                seed += 1
    assert ok, "L1 FAILED (blocking)"
    print("[L1] PASS (40 cases)")


def run_L2():
    """L2 (warn-only): larger / odd shapes."""
    print("== L2 (warn-only) ==")
    cases = [
        (2, 128, 1024, 512, 3, 1, 0, 0, 1, 2, "L2-batch2"),
        (1, 256, 2048, 640, 3, 1, 0, 0, 1, 2, "L2-deep"),
        (1, 64, 2048, 128, 2, 2, 1, 1, 1, 4, "L2-c64"),
        (2, 80, 777, 336, 5, 4, 2, 2, 2, 2, "L2-odd"),
    ]
    n_pass = 0
    for n, c_in, l_in, c_out, kW, s, pL, pR, d, groups, name in cases:
        for dtype in (torch.float16, torch.bfloat16):
            for has_bias in (False, True):
                tag = f"{name}-{'bias' if has_bias else 'nobias'}-{_dtype_str(dtype)}"
                try:
                    ok, _ = _run_case(n, c_in, l_in, c_out, kW, s, pL, pR, d,
                                      has_bias, dtype, groups, tag, seed=200)
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
        (1, 16, 16, 32, 1, 1, 0, 0, 1, 2),   # kW=1 min, c_in_g=8, c_out_g=16
        (1, 20, 40, 48, 3, 1, 1, 1, 1, 2),   # non-16 channels
        (1, 48, 128, 64, 3, 1, 0, 0, 1, 2),  # exact single tile
        (1, 18, 33, 34, 3, 2, 1, 1, 1, 2),   # all non-divisible
    ]
    n_pass = 0
    for n, c_in, l_in, c_out, kW, s, pL, pR, d, groups in cases:
        for has_bias in (False, True):
            tag = f"B-{n}x{c_in}x{l_in}x{c_out}-g{groups}-{'bias' if has_bias else 'nobias'}"
            try:
                ok, _ = _run_case(n, c_in, l_in, c_out, kW, s, pL, pR, d,
                                  has_bias, torch.float16, groups, tag, seed=300)
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
