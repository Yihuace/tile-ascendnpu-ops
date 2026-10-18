# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Round-1 candidate: tap-packed + weight-resident conv3d kernel.

Structural change over the Stage-3 baseline (single main optimization point:
eliminate the per-task weight re-read by making the weight L1-resident):

  - Tap packing: the KV=27 (kd*kh*kw) tap loop is folded into the GEMM K axis,
    so each co-block/channel-block does ONE gemm with K = KV*c_in_pad instead of
    KV gemms with K = c_in_pad.  A/B tiles are packed contiguously along K at
    offset ``tap*c_in_pad`` (the single-source K-extent invariant mirrors the
    verified Conv1d ``taps`` grouping).
  - Fixed co-block partition: ``by = kernel_id % gy`` fixes each core to one
    output-channel block (gy <= num_kernels asserted), so the weight for that
    block ``w_flat[:, by*block_m:..., :]`` is loaded into L1 ONCE (before the
    task loop) and reused across every spatial task of that core.  This removes
    the ~1792x per-task weight re-read (36-50% of the MTE2 traffic).
  - Requires n_kc == 1 (c_in_pad <= block_k): asserted; every benchmark workload
    (c_in <= 64) satisfies it at block_k=128.

Baseline diagnosis: msprof shows aic_mte2_ratio = 0.99 (6050us of 6052us) and
aic_cube_ratio = 0.016 -- the Cube side is GM->L1 instruction-bound, not
compute-bound.  The tap-packed + weight-resident restructure removes the
redundant weight traffic.
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
        n_kc = (c_in_pad + block_k - 1) // block_k
        assert n_kc == 1, (
            f"R1 tap-packing requires c_in_pad <= block_k (c_in_pad={c_in_pad}, "
            f"block_k={block_k})"
        )
        assert gy <= num_kernels, (
            f"R1 weight-resident partition requires gy <= num_kernels "
            f"(got {gy} > {num_kernels})"
        )
        share = num_kernels // gy
        spatial = n * rows * gx
        k_full = KV * c_in_pad
        # Fractal-clamped N extent (the gemm only ever reads <= out_w columns when
        # block_n >= out_w); using block_n for the B-tile allocation would blow
        # the 512KB L1 budget for wide-K workloads (e.g. video-down k_full=1728).
        n_max = max(16, _ceil16(min(block_n, out_w)))

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
                        T.copy(ws[kernel_id, slot, 0:tm, 0:tn], ub_f32[0:tm, 0:tn])
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
