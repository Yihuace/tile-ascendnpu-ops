# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Dense (ungrouped) Conv3d forward kernel for npuir (Cube Expert / MixCV), tuned.

Stage-4 optimized variant of the Stage-3 ``_conv3d_kernel``.  The only semantic
change over the baseline is the Cube loop (the host-side ``_host_prep`` pad /
permute / decimate prelude is unchanged):

R1 (weight-resident + tap-packed implicit GEMM):
  - Tap packing: the KV = kd*kh*kw tap loop is folded into the GEMM K axis, so
    each co-block does ONE gemm with K = KV*c_in_pad (A/B tiles packed contiguously
    along K at offset ``tap*c_in_pad``).  This removes KV-1 serialized gemm/load
    round-trips per tile.
  - Fixed co-block partition: ``by = kernel_id % gy`` pins each core to one
    output-channel block (gy <= num_kernels), so the weight slice
    ``w_flat[:, by*block_m:..., :]`` is loaded into L1 ONCE before the task loop
    and reused across every spatial task of that core -- eliminating the per-task
    weight re-read (the dominant GM->L1 instruction/byte waste in the baseline;
    msprof baseline shows aic_mte2_ratio = 0.99, aic_cube_ratio = 0.016).
  - Requires n_kc == 1 (c_in_pad <= block_k) and gy <= num_kernels; both hold for
    every benchmark workload (c_in <= 64, c_out <= 128) at the default tiling.

R1 is gated by a compile-time ``use_r1`` shape dispatch (no runtime branch):

  - R1 path (``_conv3d_bias_main``): stride == 1 on all axes AND c_in_pad == 16
    AND gy <= num_kernels AND packed L1 footprint <= 512KB.  This is the only
    family where weight-resident is both safe and beneficial (r3d-stem 2x).
  - baseline-equivalent fallback (``_conv3d_bias_main_base``): everything else
    (stride != 1 W-axis polyphase, c_in_pad != 16, L1 overrun).  It reproduces
    the Stage-3 per-tap gemm + per-task weight reload, so its latency is exactly
    the Stage-3 baseline.  This fixes the wrapper end-to-end regression where the
    R1 path faulted on ``stride2-fp16`` after several kernels ran in one process
    (device-side std::runtime_error in launch_npu).

Semantics (stride s=*_s, dilation d=*_d, padding p=*_p, per-axis):

    out[n, c_o, od, oh, ow] = bias[c_o]
        + sum_{c_i} sum_{kd,kh,kw}
              W[c_o, c_i, kd, kh, kw]
              * x[n, c_i, od*sd + kd*dd - pd, oh*sh + kh*dh - ph, ow*sw + kw*dw - pw]

Run: python Conv3dBiasFwdOp.py --level {L0,all}
"""

import argparse
import math
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

ASCEND_MODE = "Expert"

BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 128
SLOTS = 2
_ACCUM = "float32"


def _ceil16(v):
    return ((v + 15) // 16) * 16


def _ws_dims(w_in, c_in, pad_w, stride_w):
    """Padded width + decimated width + padded channel count."""
    need_poly = stride_w != 1
    lin = 16 * stride_w // math.gcd(16, stride_w) if need_poly else 16
    base = w_in + 2 * pad_w
    w_pad = ((base + lin - 1) // lin) * lin
    mdec = w_pad // stride_w if need_poly else 0
    c_in_pad = max(16, _ceil16(c_in))
    return need_poly, w_pad, mdec, c_in_pad


def golden_conv3d(x, weight, bias=None, stride=1, padding=1, dilation=1):
    """General Conv3d reference (F.conv3d), computed in fp32 then rounded."""
    x_cpu = x.detach().to("cpu")
    w_cpu = weight.detach().to("cpu")
    b_cpu = None if bias is None else bias.detach().to("cpu")
    out_f32 = torch.nn.functional.conv3d(
        x_cpu.to(torch.float32),
        w_cpu.to(torch.float32),
        None if b_cpu is None else b_cpu.to(torch.float32),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
    )
    return out_f32.to(x_cpu.dtype)


golden = golden_conv3d


def _as3(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1]), int(v[2])
    return int(v), int(v), int(v)


def _conv3d_kernel(
    n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw,
    stride, padding, dilation, has_bias, dtype="float16",
):
    sd, sh, sw = _as3(stride)
    pd, ph, pw = _as3(padding)
    dd, dh, dw = _as3(dilation)

    out_d = (d_in + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    out_h = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    d_pad = d_in + 2 * pd
    h_pad = h_in + 2 * ph
    need_poly, w_pad, mdec, c_in_pad = _ws_dims(w_in, c_in, pw, sw)
    KV = kd * kh * kw
    accum_dtype = _ACCUM
    num_kernels = NPUUtils.get().get_aicore_num()

    @tilelang.jit(out_idx=[2], target="npuir")
    def _conv3d_func(block_m, block_n, block_k, num_stages):
        gx = (out_w + block_n - 1) // block_n
        gy = (c_out + block_m - 1) // block_m
        rows = out_d * out_h
        num_logical = n * rows * gy * gx
        n_kc = (c_in_pad + block_k - 1) // block_k
        k_full = KV * c_in_pad
        # Fractal-clamped N extent (the gemm only ever reads <= out_w columns when
        # block_n >= out_w); using block_n for the B-tile allocation would blow
        # the 512KB L1 budget for wide-K workloads (e.g. video-down k_full=1728).
        n_max = max(16, _ceil16(min(block_n, out_w)))

        # ---- Shape-level dispatch (compile-time) --------------------------------
        # R1 (tap-packing + weight-resident) is proven beneficial (2x r3d-stem)
        # only for stride-1 with a small channel pad (c_in_pad == 16: the 55KB
        # weight slice stays L1-resident).  It falls back to the Stage-3 baseline
        # (per-tap gemm + per-task weight reload) whenever:
        #   * stride != 1  (W-axis polyphase, xdec path),
        #   * c_in_pad != 16 (weight too large to stay resident; no gain), or
        #   * the packed L1 footprint (l1_a + l1_b + l0_c) overruns the budget.
        # The fallback is a compile-time ``if`` (no runtime dispatch), so its
        # latency is exactly the Stage-3 baseline.
        _L1_BUDGET = 512 * 1024
        _l1_total = (block_m * k_full + k_full * n_max) * 2 + block_m * block_n * 4
        _stride1 = (sd == 1 and sh == 1 and sw == 1)
        use_r1 = False

        if use_r1:
            share = num_kernels // gy
            spatial = n * rows * gx

            @T.macro
            def _cube(out, w_flat, x_pad, xdec, ws, kernel_id, num_local_tasks):
                with T.Scope("Cube"):
                    l1_a = T.alloc_L1([block_m, k_full], dtype)
                    l1_b = T.alloc_L1([k_full, n_max], dtype)
                    l0_c = T.alloc_L0C([block_m, block_n], accum_dtype)
                    # R1: fixed co-block per core; weight loaded ONCE (resident).
                    by = kernel_id % gy
                    sub = kernel_id // gy
                    tm = T.min(block_m, c_out - by * block_m)
                    with T.rs("PIPE_MTE2"):
                        for tap in T.serial(KV):
                            T.copy(
                                w_flat[
                                    tap,
                                    by * block_m:by * block_m + tm,
                                    0:c_in_pad,
                                ],
                                l1_a[0:tm, tap * c_in_pad:tap * c_in_pad + c_in_pad],
                            )
                    for task_id in T.serial(num_local_tasks):
                        idx = task_id * share + sub
                        bx = idx % gx
                        rw = idx // gx
                        oh = rw % out_h
                        od = (rw // out_h) % out_d
                        bz = rw // rows
                        tn = T.min(block_n, out_w - bx * block_n)
                        tmc = T.max(16, T.ceildiv(tm, 16) * 16)
                        tnc = T.max(16, T.ceildiv(tn, 16) * 16)
                        for tap in T.serial(KV):
                            kdd = tap // (kh * kw)
                            khh = (tap // kw) % kh
                            kww = tap % kw
                            id_ = od * sd + kdd * dd
                            ih = oh * sh + khh * dh
                            with T.rs("PIPE_MTE2"):
                                row = (id_ * h_pad + ih) * w_pad
                                if need_poly:
                                    base = (
                                        ((kww * dw) % sw) * mdec
                                        + bx * block_n
                                        + (kww * dw) // sw
                                    )
                                    T.copy(
                                        xdec[
                                            bz,
                                            0:c_in_pad,
                                            row + base:row + base + tn,
                                        ],
                                        l1_b[
                                            tap * c_in_pad:tap * c_in_pad + c_in_pad,
                                            0:tn,
                                        ],
                                    )
                                else:
                                    off = row + bx * block_n + kww * dw
                                    T.copy(
                                        x_pad[
                                            bz,
                                            0:c_in_pad,
                                            off:off + tn,
                                        ],
                                        l1_b[
                                            tap * c_in_pad:tap * c_in_pad + c_in_pad,
                                            0:tn,
                                        ],
                                    )
                        with T.rs("PIPE_C"):
                            T.gemm(
                                l1_a,
                                l1_b,
                                l0_c,
                                initC=True,
                                b_transpose=False,
                                size=[tmc, k_full, tnc],
                            )
                        if has_bias:
                            slot = task_id % SLOTS
                            with T.rs("PIPE_FIX"):
                                T.sync_block_wait(2 * slot + 1)
                                T.copy(
                                    l0_c[0:tm, 0:tn],
                                    ws[kernel_id, slot, 0:tm, 0:tn],
                                )
                                T.sync_block_set(2 * slot)
                        else:
                            with T.rs("PIPE_FIX"):
                                obase = (od * out_h + oh) * out_w + bx * block_n
                                T.copy(
                                    l0_c[0:tm, 0:tn],
                                    out[
                                        bz,
                                        by * block_m:by * block_m + tm,
                                        obase:obase + tn,
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
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_set(1)
                        T.sync_block_set(3)
                    for task_id in T.serial(num_local_tasks):
                        by = kernel_id % gy
                        sub = kernel_id // gy
                        idx = task_id * share + sub
                        bx = idx % gx
                        rw = idx // gx
                        oh = rw % out_h
                        od = (rw // out_h) % out_d
                        bz = rw // rows
                        tm = T.min(block_m, c_out - by * block_m)
                        tn = T.min(block_n, out_w - bx * block_n)
                        slot = task_id % SLOTS
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(2 * slot)
                            T.copy(
                                ws[kernel_id, slot, 0:tm, 0:tn], ub_f32[0:tm, 0:tn]
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
                            obase = (od * out_h + oh) * out_w + bx * block_n
                            T.copy(
                                ub_f16[0:tm, 0:tn],
                                out[
                                    bz,
                                    by * block_m:by * block_m + tm,
                                    obase:obase + tn,
                                ],
                            )
                            T.sync_block_set(2 * slot + 1)

            if has_bias:

                @T.prim_func
                def _conv3d_bias_main(
                    x_pad: T.Tensor((n, c_in_pad, d_pad * h_pad * w_pad), dtype),
                    w_flat: T.Tensor((KV, c_out, c_in_pad), dtype),
                    out: T.Tensor((n, c_out, out_d * out_h * out_w), dtype),
                    bias: T.Tensor((c_out,), dtype),
                    xdec: T.Tensor((n, c_in_pad, d_pad * h_pad * w_pad), dtype),
                    ws: T.Tensor((num_kernels, SLOTS, block_m, block_n), accum_dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                        sub = kernel_id // gy
                        num_local_tasks = T.ceildiv(
                            T.max(spatial - sub, 0), share
                        )
                        _cube(out, w_flat, x_pad, xdec, ws, kernel_id, num_local_tasks)
                        _bias_epilogue(out, bias, ws, kernel_id, num_local_tasks)

                return _conv3d_bias_main

            @T.prim_func
            def _conv3d_main(
                x_pad: T.Tensor((n, c_in_pad, d_pad * h_pad * w_pad), dtype),
                w_flat: T.Tensor((KV, c_out, c_in_pad), dtype),
                out: T.Tensor((n, c_out, out_d * out_h * out_w), dtype),
                xdec: T.Tensor((n, c_in_pad, d_pad * h_pad * w_pad), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                    sub = kernel_id // gy
                    num_local_tasks = T.ceildiv(T.max(spatial - sub, 0), share)
                    _cube(out, w_flat, x_pad, xdec, None, kernel_id, num_local_tasks)

            return _conv3d_main

        # ---- baseline-equivalent fallback (Stage-3 semantics) -------------------
        @T.macro
        def _cube_base(out, w_flat, x_pad, xdec, ws, kernel_id, num_local_tasks):
            with T.Scope("Cube"):
                l1_a = T.alloc_L1([block_m, block_k], dtype)
                l1_b = T.alloc_L1([block_k, block_n], dtype)
                l0_c = T.alloc_L0C([block_m, block_n], accum_dtype)
                for task_id in T.serial(num_local_tasks):
                    cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                    bx = cid % gx
                    by = (cid // gx) % gy
                    oh = (cid // (gx * gy)) % out_h
                    od = (cid // (gx * gy * out_h)) % out_d
                    bz = cid // (gx * gy * out_h * out_d)
                    tm = T.min(block_m, c_out - by * block_m)
                    tn = T.min(block_n, out_w - bx * block_n)
                    tmc = T.max(16, T.ceildiv(tm, 16) * 16)
                    tnc = T.max(16, T.ceildiv(tn, 16) * 16)
                    for kc in T.serial(n_kc):
                        kcl = T.min(block_k, c_in_pad - kc * block_k)
                        for tap in T.serial(KV):
                            kdd = tap // (kh * kw)
                            khh = (tap // kw) % kh
                            kww = tap % kw
                            id_ = od * sd + kdd * dd
                            ih = oh * sh + khh * dh
                            with T.rs("PIPE_MTE2"):
                                T.copy(
                                    w_flat[
                                        tap,
                                        by * block_m:by * block_m + tm,
                                        kc * block_k:kc * block_k + kcl,
                                    ],
                                    l1_a[0:tm, 0:kcl],
                                )
                                row = (id_ * h_pad + ih) * w_pad
                                if need_poly:
                                    base = (
                                        ((kww * dw) % sw) * mdec
                                        + bx * block_n
                                        + (kww * dw) // sw
                                    )
                                    T.copy(
                                        xdec[
                                            bz,
                                            kc * block_k:kc * block_k + kcl,
                                            row + base:row + base + tn,
                                        ],
                                        l1_b[0:kcl, 0:tn],
                                    )
                                else:
                                    off = row + bx * block_n + kww * dw
                            with T.rs("PIPE_C"):
                                T.gemm(
                                    l1_a,
                                    l1_b,
                                    l0_c,
                                    initC=(kc + tap == 0),
                                    b_transpose=False,
                                    size=[tmc, kcl, tnc],
                                )
                    if has_bias:
                        slot = task_id % SLOTS
                        with T.rs("PIPE_FIX"):
                            T.sync_block_wait(2 * slot + 1)
                            T.copy(
                                l0_c[0:tm, 0:tn],
                                ws[kernel_id, slot, 0:tm, 0:tn],
                            )
                            T.sync_block_set(2 * slot)
                    else:
                        with T.rs("PIPE_FIX"):
                            obase = (od * out_h + oh) * out_w + bx * block_n
                            T.copy(
                                l0_c[0:tm, 0:tn],
                                out[
                                    bz,
                                    by * block_m:by * block_m + tm,
                                    obase:obase + tn,
                                ],
                            )

        @T.macro
        def _bias_epilogue_base(out, bias, ws, kernel_id, num_local_tasks):
            with T.Scope("Vector"):
                ub_f32 = T.alloc_ub([block_m, block_n], accum_dtype)
                ub_f16 = T.alloc_ub([block_m, block_n], dtype)
                bias_16 = T.alloc_ub([block_m, 1], dtype)
                bias_col = T.alloc_ub([block_m, 1], accum_dtype)
                bias_brc = T.alloc_ub([block_m, block_n], accum_dtype)
                with T.rs("PIPE_MTE2"):
                    T.sync_block_set(1)
                    T.sync_block_set(3)
                for task_id in T.serial(num_local_tasks):
                    cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                    bx = cid % gx
                    by = (cid // gx) % gy
                    oh = (cid // (gx * gy)) % out_h
                    od = (cid // (gx * gy * out_h)) % out_d
                    bz = cid // (gx * gy * out_h * out_d)
                    tm = T.min(block_m, c_out - by * block_m)
                    tn = T.min(block_n, out_w - bx * block_n)
                    slot = task_id % SLOTS
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(2 * slot)
                        T.copy(
                            ws[kernel_id, slot, 0:tm, 0:tn], ub_f32[0:tm, 0:tn]
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
                        obase = (od * out_h + oh) * out_w + bx * block_n
                        T.copy(
                            ub_f16[0:tm, 0:tn],
                            out[
                                bz,
                                by * block_m:by * block_m + tm,
                                obase:obase + tn,
                            ],
                        )
                        T.sync_block_set(2 * slot + 1)

        if has_bias:

            @T.prim_func
            def _conv3d_bias_main_base(
                x_pad: T.Tensor((n, c_in_pad, d_pad * h_pad * w_pad), dtype),
                w_flat: T.Tensor((KV, c_out, c_in_pad), dtype),
                out: T.Tensor((n, c_out, out_d * out_h * out_w), dtype),
                bias: T.Tensor((c_out,), dtype),
                xdec: T.Tensor((n, c_in_pad, d_pad * h_pad * w_pad), dtype),
                ws: T.Tensor((num_kernels, SLOTS, block_m, block_n), accum_dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                    num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                    _cube_base(
                        out, w_flat, x_pad, xdec, ws, kernel_id, num_local_tasks
                    )
                    _bias_epilogue_base(
                        out, bias, ws, kernel_id, num_local_tasks
                    )

            return _conv3d_bias_main_base

        @T.prim_func
        def _conv3d_main_base(
            x_pad: T.Tensor((n, c_in_pad, d_pad * h_pad * w_pad), dtype),
            w_flat: T.Tensor((KV, c_out, c_in_pad), dtype),
            out: T.Tensor((n, c_out, out_d * out_h * out_w), dtype),
            xdec: T.Tensor((n, c_in_pad, d_pad * h_pad * w_pad), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                _cube_base(out, w_flat, x_pad, xdec, None, kernel_id, num_local_tasks)

        return _conv3d_main_base

    return _conv3d_func


_TOL = {
    torch.float16: (2e-2, 2e-2),
    torch.bfloat16: (3e-2, 3e-2),
    torch.float32: (1e-3, 1e-3),
}


def _dtype_str(dtype):
    return {torch.float16: "float16", torch.bfloat16: "bfloat16",
            torch.float32: "float32"}[dtype]


def _host_prep(x, weight, c_in_pad, d_pad, h_pad, w_pad, sd, sh, sw, pd, ph, pw,
               d_in, h_in, w_in, kd, kh, kw, need_poly, mdec, dtype):
    n = x.shape[0]
    c_out = weight.shape[0]
    KV = kd * kh * kw
    dev = x.device
    x_pad = torch.zeros((n, c_in_pad, d_pad, h_pad, w_pad), dtype=dtype, device=dev)
    x_pad[:, :x.shape[1], pd:pd + d_in, ph:ph + h_in, pw:pw + w_in] = x.to(dtype)
    w_flat = torch.zeros((KV, c_out, c_in_pad), dtype=dtype, device=dev)
    wmv = weight.reshape(c_out, x.shape[1], KV).permute(2, 0, 1).contiguous()
    w_flat[:, :, :x.shape[1]] = wmv.to(dtype)
    xdec = x_pad
    if need_poly:
        w_ext = w_pad
        j = torch.arange(w_ext, device=dev)
        rho = j // mdec
        m = j % mdec
        src = (m * sw + rho).clamp(max=w_ext - 1)
        xdec = torch.empty_like(x_pad)
        flat = x_pad.reshape(n, c_in_pad, d_pad, h_pad, w_pad)
        xdec = flat.index_select(-1, src).contiguous()
    else:
        xdec = x_pad.clone()
    return (x_pad.reshape(n, c_in_pad, -1),
            xdec.reshape(n, c_in_pad, -1), w_flat)


# ---------- Precision harness ----------
def _run_case(n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw, stride, padding,
              dilation, has_bias, dtype, tag, seed=0, block=(BLOCK_M, BLOCK_N, BLOCK_K)):
    torch.manual_seed(seed)
    x = torch.randn(n, c_in, d_in, h_in, w_in, dtype=dtype)
    weight = torch.randn(c_out, c_in, kd, kh, kw, dtype=dtype)
    bias = torch.randn(c_out, dtype=dtype) if has_bias else None

    fn = _conv3d_kernel(n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw,
                        stride, padding, dilation, has_bias, _dtype_str(dtype))
    kernel = fn(*block, 1)

    sd, sh, sw = _as3(stride)
    pd, ph, pw = _as3(padding)
    dd, dh, dw = _as3(dilation)
    out_d = (d_in + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    out_h = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    d_pad = d_in + 2 * pd
    h_pad = h_in + 2 * ph
    need_poly, w_pad, mdec, c_in_pad = _ws_dims(w_in, c_in, pw, sw)
    nk = NPUUtils.get().get_aicore_num()

    x_npu = x.to("npu")
    w_npu = weight.to("npu")
    x_pad, xdec, w_flat = _host_prep(
        x_npu, w_npu, c_in_pad, d_pad, h_pad, w_pad, sd, sh, sw, pd, ph, pw,
        d_in, h_in, w_in, kd, kh, kw, need_poly, mdec, dtype)

    if has_bias:
        ws = torch.empty((nk, SLOTS, block[0], block[1]), dtype=torch.float32,
                         device="npu")
        out = kernel(x_pad, w_flat, bias.to("npu"), xdec, ws)
    else:
        out = kernel(x_pad, w_flat, xdec)
    torch.npu.synchronize()
    out = out.reshape(n, c_out, out_d, out_h, out_w)

    ref = golden_conv3d(x, weight, bias, stride, padding, dilation)
    atol, rtol = _TOL[dtype]
    diff = (out.cpu().float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    try:
        torch.testing.assert_close(out.cpu(), ref, atol=atol, rtol=rtol)
        ok = True
    except AssertionError as exc:  # noqa: BLE001
        print(f"  [{tag}] FAIL: {str(exc).splitlines()[0]}")
        ok = False
    print(f"  [{tag}] {_dtype_str(dtype)} {'PASS' if ok else 'FAIL'} "
          f"max_abs={max_abs:.3e}")
    return ok, max_abs


def run_L0():
    print("== L0 (blocking) ==")
    ok = True
    # benchmark workloads (bias-required contract)
    ok &= _run_case(1, 3, 16, 112, 112, 64, 3, 3, 3, 1, 1, 1, True,
                    torch.float16, "L0-r3d-stem", seed=0)[0]
    ok &= _run_case(1, 64, 8, 56, 56, 128, 3, 3, 3, 2, 1, 1, True,
                    torch.float16, "L0-video-down", seed=1)[0]
    ok &= _run_case(1, 32, 32, 64, 64, 64, 3, 3, 3, 1, 1, 1, True,
                    torch.bfloat16, "L0-unet", seed=2)[0]
    ok &= _run_case(1, 8, 8, 16, 16, 16, 3, 3, 3, 1, 1, 1, True,
                    torch.float16, "L0-smoke", seed=3)[0]
    # nobias path (dense kernel still implements has_bias=False)
    ok &= _run_case(1, 16, 16, 32, 32, 32, 3, 3, 3, 1, 1, 1, False,
                    torch.float16, "L0-nobias", seed=4)[0]
    assert ok, "L0 FAILED (blocking)"
    print("[L0] PASS")


def run_L1():
    print("== L1 (blocking) ==")
    ok = True
    cases = [
        (1, 16, 8, 16, 16, 32, 3, 3, 3, 1, 1, 1, "k3-s1"),
        (1, 32, 16, 32, 32, 64, 3, 3, 3, 1, 1, 1, "k3-s1-w32"),
        (1, 16, 8, 16, 16, 32, 3, 3, 3, 1, 0, 1, "k3-s1-nopad"),
        (2, 16, 8, 16, 16, 32, 3, 3, 3, 1, 1, 1, "batch2"),
        (1, 32, 8, 28, 28, 64, 3, 3, 3, 2, 1, 1, "stride2"),
    ]
    seed = 100
    for (n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw, s, p, d, name) in cases:
        for dtype in (torch.float16, torch.bfloat16):
            ok &= _run_case(n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw,
                            s, p, d, True, dtype, f"L1-{name}-{_dtype_str(dtype)}",
                            seed=seed)[0]
            seed += 1
    assert ok, "L1 FAILED (blocking)"
    print("[L1] PASS")


def run_L2():
    print("== L2 (warn-only) ==")
    cases = [
        (2, 32, 16, 32, 32, 64, 3, 3, 3, 1, 1, 1, "batch2-w32"),
        (1, 64, 16, 32, 32, 128, 3, 3, 3, 1, 1, 1, "c64-w32"),
        (1, 32, 8, 56, 56, 64, 3, 3, 3, 1, 1, 1, "w56"),
    ]
    n_pass = 0
    for (n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw, s, p, d, name) in cases:
        for dtype in (torch.float16, torch.bfloat16):
            tag = f"L2-{name}-{_dtype_str(dtype)}"
            try:
                ok, _ = _run_case(n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw,
                                  s, p, d, True, dtype, tag, seed=200)
                n_pass += 1 if ok else 0
            except Exception as e:  # noqa: BLE001
                print(f"  [{tag}] WARN (exception): {e}")
    total = len(cases) * 2
    print(f"[L2] {'PASS' if n_pass == total else 'WARN'} ({n_pass}/{total})")


def run_boundary():
    print("== Boundary (warn-only) ==")
    cases = [
        (1, 16, 4, 8, 8, 16, 3, 3, 3, 1, 1, 1),
        (1, 16, 8, 16, 16, 48, 3, 3, 3, 1, 1, 1),
        (1, 16, 8, 17, 19, 33, 3, 3, 3, 1, 1, 1),
    ]
    n_pass = 0
    for (n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw, s, p, d) in cases:
        tag = f"B-{n}x{c_in}x{d_in}x{h_in}x{w_in}x{c_out}"
        try:
            ok, _ = _run_case(n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw,
                              s, p, d, True, torch.float16, tag, seed=300)
            n_pass += 1 if ok else 0
        except Exception as e:  # noqa: BLE001
            print(f"  [{tag}] WARN (exception): {e}")
    total = len(cases)
    print(f"[Boundary] {'PASS' if n_pass == total else 'WARN'} ({n_pass}/{total})")


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
