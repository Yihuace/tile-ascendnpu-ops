# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""General (dense, ungrouped) Conv3d forward kernel for npuir (Cube Expert / MixCV).

NPU adaptation of the GPU ``_conv3d_kernel`` (TileOPs conv3d.py), following the
verified Conv1d general kernel (``_conv1d_kernel.py``) and the Conv3d feasibility
probe (``examples/conv3d/probe/conv3d_probe.py``).

Semantics (stride s=*_s, dilation d=*_d, padding p=*_p, per-axis):

    out[n, c_o, od, oh, ow] = bias[c_o]
        + sum_{c_i} sum_{kd,kh,kw}
              W[c_o, c_i, kd, kh, kw]
              * x[n, c_i, od*sd + kd*dd - pd, oh*sh + kh*dh - ph, ow*sw + kw*dw - pw]

Structure:
  - persistent 1-D ``T.Kernel(num_kernels)``; logical task =
    (n, od, oh, c_out-block, w-block) round-robins over ``num_local_tasks``.
  - implicit GEMM: A = w_flat[KV, C_out, c_in_pad] (K order (tap, c_i)), B = the
    per-tap im2col slice, fp32 accumulation via ``T.gemm(..., initC=first)``.
    Slice-form ``T.copy`` performs GM->L1 auto ND2NZ (CG-2026-0004).
  - weight layout (KV, C_out, c_in) and the zero-padded input ``x_pad`` are
    prepared on the host (per-call ``torch`` prep): ``w_flat`` is a pure
    permutation, ``x_pad`` a zero pad; both are constant-folded prep, not device
    compute.
  - stride_w>1 polyphase: consecutive output positions map to strided input
    columns, which cannot be an ND2NZ slice, so the padded width is decimated
    once on the host into ``xdec[..., j] = x_pad[..., (j % mdec)*sw + j // mdec]``;
    the per-tap B tile then reads a contiguous slice of ``xdec``.  (D/H strides
    need no decimation: they are absorbed by the per-task row index.)
  - K blocked channel-major, tap-minor: every GM->L1 slice starts at L1 offset 0.
  - tail handling: dynamic slice extents; gemm size clamps to the 16-fractal.
  - has_bias: MixCV, Cube relays fp32 L0C through the GM ws to Vector, which adds
    bias in fp32 (vcast->vbrc->vadd), casts back and stores (depth-2 handshake).

Run: python _conv3d_kernel.py --level {L0,all}
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

        @T.macro
        def _cube(out, w_flat, x_pad, xdec, ws, kernel_id, num_local_tasks):
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
                    num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
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
                num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
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
    # padded input  x_pad[n, c_in_pad, d_pad, h_pad, w_pad]
    x_pad = torch.zeros((n, c_in_pad, d_pad, h_pad, w_pad), dtype=dtype, device=dev)
    x_pad[:, :x.shape[1], pd:pd + d_in, ph:ph + h_in, pw:pw + w_in] = x.to(dtype)
    # weight permutation -> w_flat[KV, c_out, c_in_pad]
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
