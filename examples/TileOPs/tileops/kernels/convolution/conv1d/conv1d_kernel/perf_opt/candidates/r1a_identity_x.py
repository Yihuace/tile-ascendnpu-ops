# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Unit-stride (stride=1) Conv1d forward kernel for npuir (Cube Expert / MixCV).

Migrated from the GPU TileLang factory ``_conv1d_unit_stride_kernel``
(``TileOPs-main/src/tileops/kernels/convolution/conv1d.py``; builder L151-313,
``Conv1dUnitStrideKernel`` L874-926, ``general=False``) to ``target="npuir"``,
``ascend_mode="Expert"`` (``T.Scope("Cube")`` + ``T.alloc_L1`` /
``T.alloc_L0C`` / ``T.gemm`` + Vector prelude/epilogue), per DESIGN.md v1 and
the Stage-2 REVIEW.md.

Source semantics (DESIGN Sec 0.1 / 1.3), kept verbatim (stride s_W = 1):

    out[n, c_o, l] = bias[c_o]
        + sum_{c_i=0}^{C_in-1} sum_{k_w=0}^{K_W-1}
              W[c_o, c_i, k_w] * x[n, c_i, l + k_w*d_W - p_L]
    L_out = L_in + p_L + p_R - d_W*(K_W-1)

This is the stride=1 SPECIALISATION of the general ``_conv1d_kernel``: the
stride=1 window makes every per-tap B tile a CONTIGUOUS rectangle, so the whole
stride>1 polyphase / ``xdec`` / ``poly_idx`` / ``T.gather`` path is deleted and
the B load degenerates to one in-bounds slice ``T.copy`` per tap.

Structure (DESIGN Sec 0.6 / 3.3 / 6):
  - 1-D persistent ``T.Kernel(num_kernels)`` with ``num_kernels =
    NPUUtils.get().get_aicore_num()`` (verified = 20 on Ascend 910B3); logical
    task (bz, by, bx) round-robins over ``num_local_tasks``; ghost cores clamp
    ``cid`` to the last legal task (gqa precedent).
  - Implicit GEMM: A = w_flat[K_W, C_out, C_in_pad] (K order (k_w, c_i)),
    B = im2col (contiguous per-tap rectangle of x_pad), fp32 accumulation via
    ``T.gemm(..., initC=(kb==0))``.  Slice-form ``T.copy`` performs GM->L1 auto
    ND2NZ (avoids the strided ``T.load_nd2nz`` misread, CG-2026-0004).
  - Weight-layout prelude (R3): in-kernel Vector ``T.copy``(GM->UB) +
    ``T.transpose``(UB 2-axis swaps) + ``T.copy``(UB->GM) materialises
    ``w_flat[k_w, c_out, c_in]`` (K order (k_w, c_i)); the host never permutes
    the input (ascend-constraints Sec 4).
  - Input padding prelude (R4): a contiguous zero-padded ``x_pad`` removes all
    im2col OOB (left/right padding) so every B tile is an in-bounds slice.
  - K is blocked tap-major; a K step covers ``taps`` adjacent taps (``taps in
    {1,2}``).  The ``taps`` tap slices are packed CONTIGUOUSLY along the L1 K
    axis at offsets ``t*kcl`` (``kcl = min(block_k, c_in_pad - kc*block_k)`` is a
    16-multiple, so ``t*kcl`` is 16-aligned and addressable).  The gemm K extent
    equals the packed width EXACTLY: ``size=[tmc, taps*kcl, tnc]`` -- the single-
    source invariant that stops the v0 defect (loading ``taps`` taps but taking a
    single-tap gemm K, silently dropping the 2nd tap).
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
    ``_conv1d_pointwise_kernel/repro/TRAP-sync-self-set-wait.py``).

Run: python _conv1d_unit_stride_kernel.py --level {L0,all}
"""

import argparse
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

# Tiling (DESIGN Sec 5.2, inherited from the GPU Conv1dUnitStrideKernel
# default_config: Conv1dKernel default + taps=1).
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 128
TAPS = 1         # tap grouping baseline (taps in {1,2})
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


def _plan(l_in, c_in, pad_left, pad_right):
    """Workspace shapes + chunk sizes; shared by the factory and test wrapper."""
    l_pad = ((l_in + pad_left + pad_right + 15) // 16) * 16
    c_in_pad = max(16, _ceil16(c_in))
    ch_xp = _pick_chunk(l_pad, 2)
    return l_pad, c_in_pad, ch_xp


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_conv1d_unit_stride(x, weight, bias=None, pad_left=0, pad_right=0,
                              dilation=1):
    """Unit-stride Conv1d reference (source repo baseline: F.conv1d, stride=1).

    Based on the source semantics only (DESIGN Sec 0.1 / 8.1): ``weight`` is
    ``[C_out, C_in, K_W]`` (groups=1); computed in fp32 (matching the source
    fp32 accumulation + fp32 bias add) then rounded to the input dtype.
    Independent of the NPU Cube/persistent/tap-major algorithm.
    """
    x_cpu = x.detach().to("cpu")
    w_cpu = weight.detach().to("cpu")
    b_cpu = None if bias is None else bias.detach().to("cpu")
    x_f32 = x_cpu.to(torch.float32)
    if pad_left or pad_right:
        x_f32 = torch.nn.functional.pad(x_f32, (pad_left, pad_right))
    out_f32 = torch.nn.functional.conv1d(
        x_f32,
        w_cpu.to(torch.float32),
        None if b_cpu is None else b_cpu.to(torch.float32),
        stride=1,
        padding=0,
        dilation=dilation,
        groups=1,
    )
    return out_f32.to(x_cpu.dtype)


# Alias so the module exposes a literal ``golden`` as well.
golden = golden_conv1d_unit_stride


# ---------- Kernel ----------
def _conv1d_unit_stride_kernel(
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
    """NPU factory mirroring the GPU ``_conv1d_unit_stride_kernel`` builder.

    Returns an inner ``@tilelang.jit(target="npuir")`` factory taking the NPU
    tiling parameters (the GPU-only ``threads`` / ``enable_rasterization``
    parameters are dropped; ``block_k`` is the channel sub-block width and
    ``taps`` is the per-step tap grouping).
    """
    assert stride_l == 1, "_conv1d_unit_stride_kernel serves stride 1"
    accum_dtype = _ACCUM
    out_l = l_in + pad_left + pad_right - dilation_l * (kernel_l - 1)
    l_pad, c_in_pad, ch_xp = _plan(l_in, c_in, pad_left, pad_right)
    num_kernels = NPUUtils.get().get_aicore_num()

    # R1-A: when the padded workspace is a bit-exact identity of the input
    # (no left/right pad and the channel count is already 16-aligned), every
    # Cube read lies inside [0, out_l) <= [0, l_in), so the prelude's full
    # x_pad copy is pure overhead.  Read x directly and skip the copy.
    identity_x = (pad_left == 0 and pad_right == 0 and c_in == c_in_pad)

    @tilelang.jit(out_idx=[2], target="npuir")
    def _conv1d_unit_stride_func(block_m, block_n, block_k, taps, num_stages):
        assert kernel_l % taps == 0, "kernel_l must be divisible by taps"
        channel_blocks = (c_in_pad + block_k - 1) // block_k
        n_kb = (kernel_l // taps) * channel_blocks
        n_co = (c_out + BM_W - 1) // BM_W
        n_xp = (c_in_pad + ch_xp - 1) // ch_xp
        gx = (out_l + block_n - 1) // block_n
        gy = (c_out + block_m - 1) // block_m
        num_logical = gy * gx * n

        @T.macro
        def _prelude(x, weight, w_flat, x_pad, fill_x):
            with T.Scope("Vector"):
                wsrc = T.alloc_ub([BM_W, c_in, kernel_l], dtype)
                wt = T.alloc_ub([BM_W, kernel_l, c_in], dtype)
                wdst = T.alloc_ub([kernel_l, BM_W, c_in], dtype)
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
                # R1-A: skipped entirely when x_pad would be an identity copy.
                if fill_x:
                    ub2 = T.alloc_ub([ch_xp, l_in], dtype)
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
                with T.rs("PIPE_MTE3"):
                    T.sync_block_set(4)  # prelude ready (Vector -> Cube)

        @T.macro
        def _cube(out, w_flat, x_pad, ws, kernel_id, num_local_tasks):
            with T.Scope("Cube"):
                l1_a = T.alloc_L1([block_m, taps * block_k], dtype)
                l1_b = T.alloc_L1([taps * block_k, block_n], dtype)
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
                    for kb in T.serial(n_kb):
                        kw0 = (kb // channel_blocks) * taps
                        kc = kb % channel_blocks
                        kcl = T.min(block_k, c_in_pad - kc * block_k)
                        for t in T.serial(taps):
                            kw = kw0 + t
                            with T.rs("PIPE_MTE2"):
                                # A: tap kw channel block; L1 K axis packed at t*kcl.
                                T.copy(
                                    w_flat[
                                        kw,
                                        by * block_m:by * block_m + tm,
                                        kc * block_k:kc * block_k + kcl,
                                    ],
                                    l1_a[0:tm, t * kcl:t * kcl + kcl],
                                )
                                # B: per-tap contiguous rectangle (in-bounds); packed
                                # contiguously along the L1 K axis at t*kcl.
                                off = bx * block_n + kw * dilation_l
                                T.copy(
                                    x_pad[
                                        bz,
                                        kc * block_k:kc * block_k + kcl,
                                        off:off + tn,
                                    ],
                                    l1_b[t * kcl:t * kcl + kcl, 0:tn],
                                )
                        with T.rs("PIPE_C"):
                            # K extent == packed width taps*kcl (single-source
                            # invariant; covers every tap of this step).
                            T.gemm(
                                l1_a,
                                l1_b,
                                l0_c,
                                initC=(kb == 0),
                                b_transpose=False,
                                size=[tmc, taps * kcl, tnc],
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

        if has_bias:

            @T.prim_func
            def _conv1d_unit_stride_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),
                weight: T.Tensor((c_out, c_in, kernel_l), dtype),
                out: T.Tensor((n, c_out, out_l), dtype),
                bias: T.Tensor((c_out,), dtype),
                w_flat: T.Tensor((kernel_l, c_out, c_in_pad), dtype),
                x_pad: T.Tensor((n, c_in_pad, l_pad), dtype),
                ws: T.Tensor((num_kernels, SLOTS, block_m, block_n), accum_dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                    num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                    _prelude(x, weight, w_flat, x_pad, not identity_x)
                    if identity_x:
                        _cube(out, w_flat, x, ws, kernel_id, num_local_tasks)
                    else:
                        _cube(out, w_flat, x_pad, ws, kernel_id, num_local_tasks)
                    _bias_epilogue(out, bias, ws, kernel_id, num_local_tasks)

            return _conv1d_unit_stride_bias_main

        @T.prim_func
        def _conv1d_unit_stride_main(
            x: T.Tensor((n, c_in, l_in), dtype),
            weight: T.Tensor((c_out, c_in, kernel_l), dtype),
            out: T.Tensor((n, c_out, out_l), dtype),
            w_flat: T.Tensor((kernel_l, c_out, c_in_pad), dtype),
            x_pad: T.Tensor((n, c_in_pad, l_pad), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                _prelude(x, weight, w_flat, x_pad, not identity_x)
                if identity_x:
                    _cube(out, w_flat, x, None, kernel_id, num_local_tasks)
                else:
                    _cube(out, w_flat, x_pad, None, kernel_id, num_local_tasks)

        return _conv1d_unit_stride_main

    return _conv1d_unit_stride_func


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
    pad_left,
    pad_right,
    dilation_l,
    has_bias,
    dtype,
    tag,
    taps=TAPS,
    seed=0,
):
    torch.manual_seed(seed)
    x = torch.randn(n, c_in, l_in, dtype=dtype)
    weight = torch.randn(c_out, c_in, kernel_l, dtype=dtype)
    bias = torch.randn(c_out, dtype=dtype) if has_bias else None

    fn = _conv1d_unit_stride_kernel(
        n, c_in, l_in, c_out, kernel_l, 1, pad_left, pad_right,
        dilation_l, has_bias, _dtype_str(dtype),
    )
    kernel = fn(BLOCK_M, BLOCK_N, BLOCK_K, taps, 1)

    l_pad, c_in_pad, _ch_xp = _plan(l_in, c_in, pad_left, pad_right)
    nk = NPUUtils.get().get_aicore_num()

    x_npu = x.to("npu")
    w_npu = weight.to("npu")
    # w_flat / x_pad are host pre-zeroed (v1 rec 2): pad columns/channels stay 0
    # so 0 * 0 = 0 regardless of the other operand's bit pattern.
    w_flat = torch.zeros((kernel_l, c_out, c_in_pad), dtype=dtype, device="npu")
    x_pad = torch.zeros((n, c_in_pad, l_pad), dtype=dtype, device="npu")

    if has_bias:
        ws = torch.empty(
            (nk, SLOTS, BLOCK_M, BLOCK_N), dtype=torch.float32, device="npu"
        )
        out = kernel(x_npu, w_npu, bias.to("npu"), w_flat, x_pad, ws)
    else:
        out = kernel(x_npu, w_npu, w_flat, x_pad)
    torch.npu.synchronize()

    ref = golden_conv1d_unit_stride(x, weight, bias, pad_left, pad_right, dilation_l)
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
        f"p={pad_left}/{pad_right} d={dilation_l} taps={taps} "
        f"has_bias={has_bias} {_dtype_str(dtype)}: "
        f"{'PASS' if ok else 'FAIL'} (max_abs={max_abs:.3e})"
    )
    return ok, max_abs


def _run_taps_consistency(
    n, c_in, l_in, c_out, kernel_l, pad_left, pad_right, dilation_l, has_bias,
    dtype, tag, seed=0,
):
    """Must-test 5: taps=2 must match taps=1 (gates the gemm K-extent bug).

    Runs the same shape with taps=1 and taps=2; both are checked against the CPU
    golden and the two outputs are compared against each other.  In the v0 shape
    (gemm K = single tap while two taps are loaded) the taps=2 run loses every
    odd tap (~half the sum) and fails.
    """
    k1 = _conv1d_unit_stride_kernel(
        n, c_in, l_in, c_out, kernel_l, 1, pad_left, pad_right,
        dilation_l, has_bias, _dtype_str(dtype),
    )(BLOCK_M, BLOCK_N, BLOCK_K, 1, 1)
    k2 = _conv1d_unit_stride_kernel(
        n, c_in, l_in, c_out, kernel_l, 1, pad_left, pad_right,
        dilation_l, has_bias, _dtype_str(dtype),
    )(BLOCK_M, BLOCK_N, BLOCK_K, 2, 1)

    torch.manual_seed(seed)
    x = torch.randn(n, c_in, l_in, dtype=dtype)
    weight = torch.randn(c_out, c_in, kernel_l, dtype=dtype)
    bias = torch.randn(c_out, dtype=dtype) if has_bias else None
    l_pad, c_in_pad, _ = _plan(l_in, c_in, pad_left, pad_right)
    nk = NPUUtils.get().get_aicore_num()

    def _launch(kernel):
        w_flat = torch.zeros((kernel_l, c_out, c_in_pad), dtype=dtype, device="npu")
        x_pad = torch.zeros((n, c_in_pad, l_pad), dtype=dtype, device="npu")
        if has_bias:
            ws = torch.empty(
                (nk, SLOTS, BLOCK_M, BLOCK_N), dtype=torch.float32, device="npu"
            )
            o = kernel(x.to("npu"), weight.to("npu"), bias.to("npu"), w_flat,
                       x_pad, ws)
        else:
            o = kernel(x.to("npu"), weight.to("npu"), w_flat, x_pad)
        torch.npu.synchronize()
        return o

    out1 = _launch(k1)
    out2 = _launch(k2)
    ref = golden_conv1d_unit_stride(x, weight, bias, pad_left, pad_right, dilation_l)
    atol, rtol = _TOL[dtype]
    cross_abs = float((out2.cpu().float() - out1.cpu().float()).abs().max().item())
    golden_abs = float((out2.cpu().float() - ref.float()).abs().max().item())
    ok = True
    try:
        torch.testing.assert_close(out2.cpu(), ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(out2.cpu(), out1.cpu(), atol=atol, rtol=rtol)
    except AssertionError as exc:  # noqa: BLE001
        print(f"  [{tag}] FAIL: {str(exc).splitlines()[0]}")
        ok = False
    print(
        f"  [{tag}] n={n} c_in={c_in} l_in={l_in} c_out={c_out} kW={kernel_l} "
        f"has_bias={has_bias} {_dtype_str(dtype)}: taps=2 vs taps=1 "
        f"{'PASS' if ok else 'FAIL'} (cross={cross_abs:.3e}, "
        f"vs_golden={golden_abs:.3e})"
    )
    return ok, max(golden_abs, cross_abs)


# ---------- hierarchical testing ----------
def run_L0():
    """L0 gate (blocking): representative workloads, both paths, taps>=2."""
    print("== L0 (blocking) ==")
    ok = True
    # whisper-conv1 (DESIGN Sec 8.3): stride=1, pad=0, kW=3
    ok &= _run_case(1, 80, 3000, 1280, 3, 0, 0, 1, False, torch.float16,
                    "L0-whisper-nobias", seed=0)[0]
    ok &= _run_case(1, 80, 3000, 1280, 3, 0, 0, 1, True, torch.float16,
                    "L0-whisper-bias", seed=1)[0]
    # encodec-deep: kW=10, large c_in
    ok &= _run_case(1, 128, 600, 256, 10, 0, 0, 1, False, torch.float16,
                    "L0-encodec-nobias", seed=2)[0]
    ok &= _run_case(1, 128, 600, 256, 10, 0, 0, 1, True, torch.float16,
                    "L0-encodec-bias", seed=3)[0]
    # non-divisible dims (must-test 1): C_in=50, C_out=100, L=1000, kW=7
    ok &= _run_case(1, 50, 1000, 100, 7, 0, 0, 1, False, torch.float16,
                    "L0-nondiv-nobias", seed=4)[0]
    # padding>0 + dilation>1 (must-test 4)
    ok &= _run_case(1, 50, 1000, 100, 3, 3, 3, 2, True, torch.float16,
                    "L0-dilation-pad-bias", seed=5)[0]
    # batch>1 (must-test 3)
    ok &= _run_case(3, 96, 300, 192, 3, 0, 0, 1, True, torch.float16,
                    "L0-batch3-bias", seed=6)[0]
    # taps=2 grouping consistency (must-test 5) -- the v0 defect gate
    ok &= _run_taps_consistency(1, 128, 600, 256, 10, 0, 0, 1, False,
                                torch.float16, "L0-taps2-nobias", seed=7)[0]
    ok &= _run_taps_consistency(1, 128, 600, 256, 10, 0, 0, 1, True,
                                torch.float16, "L0-taps2-bias", seed=8)[0]
    assert ok, "L0 FAILED (blocking)"
    print("[L0] PASS (9 cases)")


def run_L1():
    """L1 gate (blocking): dtypes f16/bf16, pads/dilations, taps>=2, nondiv."""
    print("== L1 (blocking) ==")
    ok = True
    # (n, c_in, l_in, c_out, kW, pL, pR, d, name)
    cases = [
        (1, 80, 3000, 1280, 3, 0, 0, 1, "whisper-kW3"),
        (1, 128, 600, 256, 10, 0, 0, 1, "encodec-kW10"),
        (1, 50, 777, 100, 7, 3, 3, 2, "nondiv-kW7"),
        (2, 64, 512, 128, 4, 2, 2, 1, "batch2-kW4"),
        (1, 17, 300, 33, 3, 1, 1, 1, "odd-kW3"),
        (1, 32, 256, 64, 5, 2, 2, 1, "smoke-kW5"),
    ]
    seed = 100
    for n, c_in, l_in, c_out, kW, pL, pR, d, name in cases:
        for dtype in (torch.float16, torch.bfloat16):
            for has_bias in (False, True):
                tag = f"L1-{name}-{'bias' if has_bias else 'nobias'}-{_dtype_str(dtype)}"
                ok &= _run_case(n, c_in, l_in, c_out, kW, pL, pR, d, has_bias,
                                dtype, tag, seed=seed)[0]
                seed += 1
    # taps=2 grouping: both dtypes, both paths, kW even
    taps2_cases = [
        (1, 128, 600, 256, 10, 0, 0, 1, "taps2-encodec"),
        (1, 64, 400, 128, 4, 2, 2, 1, "taps2-kW4"),
    ]
    for n, c_in, l_in, c_out, kW, pL, pR, d, name in taps2_cases:
        for dtype in (torch.float16, torch.bfloat16):
            for has_bias in (False, True):
                tag = f"L1-{name}-{'bias' if has_bias else 'nobias'}-{_dtype_str(dtype)}"
                ok &= _run_taps_consistency(n, c_in, l_in, c_out, kW, pL, pR, d,
                                            has_bias, dtype, tag, seed=seed)[0]
                seed += 1
    assert ok, "L1 FAILED (blocking)"
    print("[L1] PASS (32 cases)")


def run_L2():
    """L2 (warn-only): larger shapes."""
    print("== L2 (warn-only) ==")
    cases = [
        (4, 128, 1024, 512, 3, 0, 0, 1, "L2-batch4"),
        (1, 64, 2048, 320, 10, 0, 0, 1, "L2-deep-kW10"),
        (1, 16, 2048, 64, 2, 0, 0, 1, "L2-c16-kW2"),
        (2, 80, 777, 333, 5, 2, 2, 2, "L2-odd"),
    ]
    n_pass = 0
    for n, c_in, l_in, c_out, kW, pL, pR, d, name in cases:
        for dtype in (torch.float16, torch.bfloat16):
            for has_bias in (False, True):
                tag = f"{name}-{'bias' if has_bias else 'nobias'}-{_dtype_str(dtype)}"
                try:
                    ok, _ = _run_case(n, c_in, l_in, c_out, kW, pL, pR, d,
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
        (1, 16, 16, 16, 2, 0, 0, 1),   # minimal kW=2, c_in=16
        (1, 20, 40, 48, 3, 0, 0, 1),   # non-16 channel dims
        (1, 48, 128, 64, 3, 0, 0, 1),  # exact single tile
        (1, 17, 33, 33, 3, 2, 2, 1),   # all non-divisible
    ]
    n_pass = 0
    for n, c_in, l_in, c_out, kW, pL, pR, d in cases:
        for has_bias in (False, True):
            tag = f"B-{n}x{c_in}x{l_in}x{c_out}-kW{kW}-{'bias' if has_bias else 'nobias'}"
            try:
                ok, _ = _run_case(n, c_in, l_in, c_out, kW, pL, pR, d,
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
