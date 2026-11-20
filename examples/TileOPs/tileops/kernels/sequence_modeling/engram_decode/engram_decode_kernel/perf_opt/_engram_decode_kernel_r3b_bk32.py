# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Engram fused decode kernel (NPU, ``target="npuir"``) -- Stage 4 optimized.

Stage 4 (2026-10-16) performance work on top of the Stage 3 baseline
(``../../_engram_decode_kernel.py``).  Single structural change:

  * The O(d_mem) serial GEMV ``k_local[j] += e_local[i] * W[i, j]`` was
    lowered to a **fully scalar** ``scf.for`` (``tensor.extract`` /
    ``arith.mulf`` / ``arith.addf`` / ``tensor.insert``): the loop-carried
    accumulator blocks the npuir vectorizer.  `msprof op` on the baseline
    showed ``aiv_scalar_ratio = 0.979`` (97.9% of the kernel stalled) with
    ``aiv_vec_ratio = 0.0016``.
  * It is replaced by a K-blocked outer product + hardware reduction:
    stage a ``(BK, d_padded)`` fp32 weight tile, multiply it by a row-wise
    broadcast of the ``BK``-element ``e`` slice, and accumulate with
    ``T.reduce_sum(..., dim=0, clear=False)`` (``hivm.hir.vreduce``).
    No `T.Parallel` loop-carried elementwise accumulate remains.
  * ``T.vcast`` on a 2-D buffer is rejected by the frontend
    (``load->indices.size() == ndim`` InternalError), so the fp32 staging
    uses the silent dtype-converting ``T.copy`` (same mechanism the baseline
    already relied on for its 1-D ``W_K_local``).

Stage 4 round 2 (2026-10-30) adds one more structural change on top of the
round-1 winner:

  * ``new_conv_state`` shift was copied **row by row** (``max_conv_len - 1``
    iterations x 2 tiny GM<->UB copies; 11/19/17 rows per workload).  Each
    small DMA instruction costs ~0.16 us on this device (measured, see
    ``perf_opt/opt_log.md`` H1 probes), so the shift alone cost 1.6-5 us.
    It is replaced by two bulk 2-D region copies of the whole
    ``(max_conv_len-1, d_padded)`` block (bit-exact: same dtype in and out).

Measured (`msprof op`, median of 20, Ascend910B3 / tilelang 0.1.2+76e4550 /
CANN 9.0.0): round-1 winner decode-b1 30.18 -> 28.55 us, decode-b4 78.40 ->
73.56 us, decode-b8 31.31 -> 28.63 us.  Round-1 GEMV work: decode-b1 5988.8 ->
30.1 us, decode-b4 23668.2 -> 78.6 us, decode-b8 5988.6 -> 31.4 us.



Migration of the GPU TileLang factory ``_engram_decode_kernel``
(``/mnt/workspace/TileOPs/tileops/kernels/engram/engram_decode.py``).

Single-token decode pipeline fully fused into one kernel per batch:

    k = e @ W_K                                          # GEMV: (d_mem,) -> (d,)
    v = e @ W_V                                          # GEMV: (d_mem,) -> (d,)
    h_n = RMSNorm(h, w_h)
    k_n = RMSNorm(k, w_h)
    alpha = sigmoid(dot(h_n, k_n) / sqrt(d))
    v_hat = alpha * v
    v_n = RMSNorm(v_hat, w_v)
    conv_out = sum_p conv_w[p] * window[p]               # dilated causal conv
    new_conv_state = shift_left(conv_state) | v_n
    y = SiLU(conv_out) + v_hat

NPU re-design status (see DESIGN.md sections 0.5 / 0.6 / 4 and RETROSPECTIVE):

  * R1 -- NOT APPLIED.  The Cube ``T.gemm`` GEMV (M=d_padded, K=d_mem tiled by
    ``BK``, N=1, fp32 L0C accumulator) was implemented and exercised, but the
    npuir Developer auto CV-split rejects every full-kernel MIX combination
    with ``hivm.hir.load`` / ``hivm.hir.copy`` memory-hierarchy verifier
    errors ("Unsupported copy from cbuf to cc", "load only support src == gm
    and dst != gm", "expected result type with stride ... in dim = 0").  The
    GEMV is therefore the fp32 vector FMA chain used by the reference NPU
    implementation ``examples/engram/engram_decode.py``.
  * R2 -- APPLIED.  fp32 compute domain; RMSNorm's reciprocal is ``sqrt(s)/s``
    (``vsqrt`` + scalar divide) instead of the approximate ``vrsqrt`` device
    instruction (DESIGN O1).
  * R3 -- PARTIALLY APPLIED.  1-D ``T.Kernel(batch, is_npu=True)`` (one block
    per batch); vector quantities live in ``T.alloc_shared`` (UB-backed).
    ``T.alloc_ub`` + scalar broadcasts segfault the codegen, so the
    reference's shared scope is used.
  * R4 -- APPLIED.  ``v_hat = alpha * v`` is materialized (CSE) and reused by
    the SiLU residual instead of being recomputed.
  * R5 -- NOT APPLIED (kept as the reference's per-row ``T.serial`` shift):
    the 2-D region ``T.copy`` variant compiles but the row loop is the proven
    form and the shift is bit-exact (L1 probe).

Buffer discipline: all buffers are declared unconditionally at the top of the
``T.Kernel`` scope; the conv taps are read through a serial loop whose row
index is a runtime expression accepted by ``T.copy``.

run: python _engram_decode_kernel.py --level all
"""

import os

# Developer mode for npuir (auto CV-split; no manual sync needed).
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

tilelang.cache.clear_cache()

_F32 = "float32"
ALIGNMENT = 256

# Tuned GEMV K-block width (Stage 4 round 3 / R3-B re-check under the
# 01805d7a toolchain).  BK=32 halves the number of DMA instructions
# (d_mem/BK weight tiles + e_n slices per block).
TUNED_BK = 32


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# Golden (independent PyTorch CPU reference)
# Ported from examples/TileOPs/tests/ops/test_engram_decode.py
# ::engram_decode_step_torch  (contract semantics; NOT the NPU structure).
# ---------------------------------------------------------------------------
def _rmsnorm_decode(x, w, eps=1e-6):
    # x: (B, 1, d), w: (d,); returns (normalized, rrms)
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f * rrms * w.float()), rrms


def golden_engram_decode(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v,
                         conv_w, max_conv_len, dilation, eps=1e-6):
    """Single decode step reference implementation (source semantics).

    All arithmetic in fp32 (``.float()``), a single round back to the input
    dtype at the end.  Independent of the NPU structure (Cube GEMV /
    vsqrt+vdiv / 2-D region copies) so it can validate the kernel.
    """
    B, d = h_t.shape
    w = conv_w.shape[0]
    L = conv_state.shape[1]

    k = e_t.float() @ W_K.float()
    v = e_t.float() @ W_V.float()

    h_norm, _ = _rmsnorm_decode(h_t.unsqueeze(1), rms_w_h)
    k_norm, _ = _rmsnorm_decode(k.unsqueeze(1).to(h_t.dtype), rms_w_h)
    h_norm = h_norm.squeeze(1)
    k_norm = k_norm.squeeze(1)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))
    v_hat = alpha * v

    v_hat_norm, _ = _rmsnorm_decode(v_hat.unsqueeze(1).to(h_t.dtype), rms_w_v)
    v_hat_norm = v_hat_norm.squeeze(1)

    if max_conv_len > L:
        padded_state = F.pad(conv_state.float(), (0, 0, max_conv_len - L, 0))
    else:
        padded_state = conv_state.float()

    conv_out = torch.zeros(B, d, device=h_t.device)
    for p in range(w - 1):
        idx = max_conv_len - (w - 1 - p) * dilation
        if 0 <= idx < max_conv_len:
            conv_out += conv_w[p].float().unsqueeze(0) * padded_state[:, idx, :]
    conv_out += conv_w[w - 1].float().unsqueeze(0) * v_hat_norm

    if max_conv_len > L:
        new_conv_state = torch.cat(
            [conv_state, v_hat_norm.unsqueeze(1).to(conv_state.dtype)], dim=1)
    else:
        new_conv_state = torch.cat(
            [conv_state[:, 1:, :],
             v_hat_norm.unsqueeze(1).to(conv_state.dtype)], dim=1)

    y_t = F.silu(conv_out) + v_hat
    return y_t.to(h_t.dtype), new_conv_state


# ---------------------------------------------------------------------------
# NPU kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[8, 9], target="npuir")
def _engram_decode_kernel(batch, d_mem, d, max_conv_len, conv_kernel_size,
                          dilation, eps, dtype, BK=TUNED_BK):
    """Fused single-token decode kernel factory (NPU, AIV vector pipeline).

    Args:
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, eps, dtype:
            compile-time specialization parameters (unchanged across migration).
        BK: GEMV K-block width (tuned Stage 4 constant, see ``TUNED_BK``).

    Design notes (DESIGN.md sections 0.6 / 4):

      * GEMV: K-blocked outer product + ``T.reduce_sum(dim=0)`` hardware
        reduction (Stage 4).  See the module docstring for the measured
        rationale; the toolchain re-check of the Cube ``T.gemm`` MIX route is
        recorded in ``perf_opt/opt_log.md``.
      * The reciprocal RMS uses ``vsqrt`` + scalar divide (``sqrt(s)/s``)
        instead of the approximate ``vrsqrt`` device instruction (DESIGN R2 / O1).
      * ``v_hat = alpha * v`` is materialized once and reused by the SiLU
        residual (DESIGN R4 CSE).
    """
    d_padded = _align_up(d, ALIGNMENT)
    w = conv_kernel_size
    accum_dtype = "float32"

    @T.macro
    def _decode_fused(
        e_t: T.Tensor((batch, d_mem), dtype),
        h_t: T.Tensor((batch, d_padded), dtype),
        conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
        W_K: T.Tensor((d_mem, d_padded), dtype),
        W_V: T.Tensor((d_mem, d_padded), dtype),
        rms_w_h: T.Tensor((d_padded,), dtype),
        rms_w_v: T.Tensor((d_padded,), dtype),
        conv_w: T.Tensor((w, d_padded), dtype),
        y_t: T.Tensor((batch, d_padded), dtype),
        new_conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
    ):
        with T.Kernel(batch, is_npu=True) as (bid, _):
            e_local = T.alloc_shared((d_mem,), accum_dtype)
            T.copy(e_t[bid, :], e_local)

            # ---- K-blocked GEMV: row-broadcast product + vreduce accumulate ----
            nk = d_mem // BK
            w_f = T.alloc_shared((BK, d_padded), accum_dtype)
            e_n = T.alloc_shared((BK,), accum_dtype)
            prod = T.alloc_shared((BK, d_padded), accum_dtype)
            acc_k = T.alloc_shared((1, d_padded), accum_dtype)
            acc_v = T.alloc_shared((1, d_padded), accum_dtype)
            T.clear(acc_k)
            T.clear(acc_v)
            for k0 in T.serial(nk):
                T.copy(e_t[bid, k0 * BK : (k0 + 1) * BK], e_n)
                T.copy(W_K[k0 * BK : (k0 + 1) * BK, :], w_f)
                for kk, j in T.Parallel(BK, d_padded):
                    prod[kk, j] = w_f[kk, j] * e_n[kk]
                T.reduce_sum(prod, acc_k, dim=0, clear=False)

                T.copy(W_V[k0 * BK : (k0 + 1) * BK, :], w_f)
                for kk, j in T.Parallel(BK, d_padded):
                    prod[kk, j] = w_f[kk, j] * e_n[kk]
                T.reduce_sum(prod, acc_v, dim=0, clear=False)

            k_local = T.alloc_shared((d_padded,), accum_dtype)
            v_local = T.alloc_shared((d_padded,), accum_dtype)
            for j in T.Parallel(d_padded):
                k_local[j] = acc_k[0, j]
                v_local[j] = acc_v[0, j]

            sqrt_tmp = T.alloc_shared((1, 1), accum_dtype)
            h_local = T.alloc_shared((d_padded,), accum_dtype)
            T.copy(h_t[bid, :], h_local)

            hsq_2d = T.alloc_shared((1, d_padded), accum_dtype)
            for j in T.Parallel(d_padded):
                hsq_2d[0, j] = h_local[j] * h_local[j]

            sumsq_h = T.alloc_shared((1, 1), accum_dtype)
            T.reduce_sum(hsq_2d, sumsq_h, dim=1)

            sumsq_h[0, 0] = sumsq_h[0, 0] / d + eps
            T.vsqrt(sumsq_h, sqrt_tmp); sumsq_h[0, 0] = sqrt_tmp[0, 0] / sumsq_h[0, 0]

            rms_w_h_local = T.alloc_shared((d_padded,), accum_dtype)
            T.copy(rms_w_h, rms_w_h_local)
            for j in T.Parallel(d_padded):
                h_local[j] = h_local[j] * sumsq_h[0, 0] * rms_w_h_local[j]

            ksq_2d = T.alloc_shared((1, d_padded), accum_dtype)
            for j in T.Parallel(d_padded):
                ksq_2d[0, j] = k_local[j] * k_local[j]
            sumsq_k = T.alloc_shared((1, 1), accum_dtype)
            T.reduce_sum(ksq_2d, sumsq_k, dim=1)
            T.vdiv(sumsq_k, d, sumsq_k)
            T.vadd(sumsq_k, eps, sumsq_k)
            T.vsqrt(sumsq_k, sqrt_tmp); sumsq_k[0, 0] = sqrt_tmp[0, 0] / sumsq_k[0, 0]
            for j in T.Parallel(d_padded):
                k_local[j] = k_local[j] * sumsq_k[0, 0] * rms_w_h_local[j]

            hk_2d = T.alloc_shared((1, d_padded), accum_dtype)
            for j in T.Parallel(d_padded):
                hk_2d[0, j] = h_local[j] * k_local[j]
            dot_hk = T.alloc_shared((1, 1), accum_dtype)
            T.reduce_sum(hk_2d, dot_hk, dim=1)
            sqrt_d = T.alloc_shared((1, 1), accum_dtype)
            sqrt_d[0, 0] = d
            T.vsqrt(sqrt_d, sqrt_d)
            alpha = T.alloc_shared((1, 1), accum_dtype)
            alpha[0, 0] = dot_hk[0, 0] / sqrt_d[0, 0]
            T.vsigmoid(alpha, alpha)
            val = alpha[0, 0]
            vhat_local = T.alloc_shared((d_padded,), accum_dtype)
            vn_local = T.alloc_shared((d_padded,), accum_dtype)
            for j in T.Parallel(d_padded):
                vhat_local[j] = val * v_local[j]

            vsq_2d = T.alloc_shared((1, d_padded), accum_dtype)
            for j in T.Parallel(d_padded):
                vsq_2d[0, j] = vhat_local[j] * vhat_local[j]
            sumsq_v = T.alloc_shared((1, 1), accum_dtype)
            T.reduce_sum(vsq_2d, sumsq_v, dim=1)
            T.vdiv(sumsq_v, d, sumsq_v)
            T.vadd(sumsq_v, eps, sumsq_v)
            T.vsqrt(sumsq_v, sqrt_tmp); sumsq_v[0, 0] = sqrt_tmp[0, 0] / sumsq_v[0, 0]
            rms_w_v_local = T.alloc_shared((d_padded,), accum_dtype)
            T.copy(rms_w_v, rms_w_v_local)
            for j in T.Parallel(d_padded):
                vn_local[j] = vhat_local[j] * sumsq_v[0, 0] * rms_w_v_local[j]

            conv_out = T.alloc_shared((d_padded,), accum_dtype)
            T.clear(conv_out)

            for p in T.serial(w - 1):
                conv_w_local = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(conv_w[p, :], conv_w_local)
                conv_state_local = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(
                    conv_state[bid, max_conv_len - (w - 1 - p) * dilation, :],
                    conv_state_local,
                )
                for j in T.Parallel(d_padded):
                    conv_out[j] += conv_w_local[j] * conv_state_local[j]

            conv_w_l = T.alloc_shared((d_padded,), accum_dtype)
            T.copy(conv_w[w - 1, :], conv_w_l)
            for j in T.Parallel(d_padded):
                conv_out[j] += conv_w_l[j] * vn_local[j]

            # Unsupported memref to memref copy yet.
            # for s in T.serial(max_conv_len - 1):
            #     for j in T.Parallel(d_padded):
            #         new_conv_state[bid, s, j] = conv_state[bid, s + 1, j]
            shift_buf = T.alloc_shared((max_conv_len - 1, d_padded), dtype)
            T.copy(conv_state[bid, 1:, :], shift_buf)
            T.copy(shift_buf, new_conv_state[bid, 0:max_conv_len - 1, :])

            T.copy(vn_local, new_conv_state[bid, max_conv_len - 1, :])

            sig = T.alloc_shared((d_padded,), accum_dtype)
            T.vsigmoid(conv_out, sig)
            alpha_val = alpha[0, 0]
            for j in T.Parallel(d_padded):
                conv_out[j] = conv_out[j] * sig[j] + vhat_local[j]
            T.copy(conv_out, y_t[bid, :])

    @T.prim_func
    def main(
        e_t: T.Tensor((batch, d_mem), dtype),
        h_t: T.Tensor((batch, d_padded), dtype),
        conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
        W_K: T.Tensor((d_mem, d_padded), dtype),
        W_V: T.Tensor((d_mem, d_padded), dtype),
        rms_w_h: T.Tensor((d_padded,), dtype),
        rms_w_v: T.Tensor((d_padded,), dtype),
        conv_w: T.Tensor((w, d_padded), dtype),
        y_t: T.Tensor((batch, d_padded), dtype),
        new_conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
    ):
        _decode_fused(
            e_t,
            h_t,
            conv_state,
            W_K,
            W_V,
            rms_w_h,
            rms_w_v,
            conv_w,
            y_t,
            new_conv_state,
        )

    return main


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _dtype_str(dtype) -> str:
    return str(dtype).split(".")[-1]


def _pad_last(x, target):
    if x.shape[-1] == target:
        return x
    assert x.shape[-1] < target, (x.shape, target)
    return F.pad(x, (0, target - x.shape[-1]))


def _pad_state(cs, max_conv_len):
    L = cs.shape[1]
    if L == max_conv_len:
        return cs
    assert L < max_conv_len, (L, max_conv_len)
    return F.pad(cs, (0, 0, max_conv_len - L, 0))


def _make_inputs(batch, d_mem, d, L, w, dtype, seed):
    torch.manual_seed(seed)
    dt = dtype
    e_t = (torch.randn(batch, d_mem) * 0.5).to(dt)
    h_t = torch.randn(batch, d).to(dt)
    cs = (torch.randn(batch, L, d) * 0.5).to(dt)
    W_K = (torch.randn(d_mem, d) * 0.05).to(dt)
    W_V = (torch.randn(d_mem, d) * 0.05).to(dt)
    rms_w_h = torch.randn(d).to(dt)
    rms_w_v = torch.randn(d).to(dt)
    conv_w = (torch.randn(w, d) * 0.1).to(dt)
    return e_t, h_t, cs, W_K, W_V, rms_w_h, rms_w_v, conv_w


def _run_op(kernel, batch, d_mem, d, max_conv_len, w, dilation, eps,
            e_t, h_t, cs, W_K, W_V, rms_w_h, rms_w_v, conv_w):
    """Pad host-side to the compiled contract, run the kernel, crop back to d."""
    d_pad = _align_up(d, ALIGNMENT)
    e_n = e_t.npu()
    h_n = h_t.npu()
    cs_n = _pad_state(cs, max_conv_len).npu()
    WK_n = W_K.npu()
    WV_n = W_V.npu()
    wh_n = rms_w_h.npu()
    wv_n = rms_w_v.npu()
    cw_n = conv_w.npu()
    pad = d_pad - d
    if pad:
        h_n = _pad_last(h_n, d_pad)
        cs_n = _pad_last(cs_n, d_pad)
        WK_n = _pad_last(WK_n, d_pad)
        WV_n = _pad_last(WV_n, d_pad)
        wh_n = _pad_last(wh_n, d_pad)
        wv_n = _pad_last(wv_n, d_pad)
        cw_n = _pad_last(cw_n, d_pad)
    y, ns = kernel(e_n, h_n, cs_n, WK_n, WV_n, wh_n, wv_n, cw_n)
    y = y.float().cpu()
    ns = ns.float().cpu()
    if pad:
        y = y[:, :d]
        ns = ns[:, :, :d]
    return y, ns


def run_case(batch, d_mem, d, max_conv_len, w, dilation, dtype, eps=1e-6,
             seed=0, atol=None, rtol=5e-2, verbose=True, tag=""):
    """Single-step correctness case; returns max abs diff over y and state."""
    if atol is None:
        atol = 5e-2 if dtype == torch.float16 else 1e-1
    assert max_conv_len >= dilation * (w - 1), (max_conv_len, dilation, w)

    kernel = _engram_decode_kernel(batch, d_mem, d, max_conv_len, w,
                                   dilation, eps, _dtype_str(dtype))
    ins = _make_inputs(batch, d_mem, d, max_conv_len, w, dtype, seed)
    e_t, h_t, cs, W_K, W_V, rms_w_h, rms_w_v, conv_w = ins

    y_op, ns_op = _run_op(kernel, batch, d_mem, d, max_conv_len, w, dilation,
                          eps, *ins)
    y_ref, ns_ref = golden_engram_decode(
        e_t, h_t, cs, W_K, W_V, rms_w_h, rms_w_v, conv_w,
        max_conv_len, dilation, eps)

    y_ref = y_ref.float()
    ns_ref = ns_ref.float()
    ydiff = (y_op - y_ref).abs().max().item()
    sdiff = (ns_op - ns_ref).abs().max().item()
    torch.testing.assert_close(y_op, y_ref, atol=atol, rtol=rtol,
                               equal_nan=True)
    torch.testing.assert_close(ns_op, ns_ref, atol=atol, rtol=rtol,
                               equal_nan=True)
    if verbose:
        print(f"  [ok] {tag or 'case'} B={batch} d_mem={d_mem} d={d} "
              f"L={max_conv_len} w={w} dil={dilation} {_dtype_str(dtype)} "
              f"y_maxdiff={ydiff:.3e} st_maxdiff={sdiff:.3e}")
    return max(ydiff, sdiff)


def run_multistep(batch, d_mem, d, max_conv_len, w, dilation, dtype, eps=1e-6,
                  seed=123, steps=None, tag=""):
    """Multi-step decode: start from an empty conv state and iterate."""
    if steps is None:
        steps = max_conv_len + 8
    kernel = _engram_decode_kernel(batch, d_mem, d, max_conv_len, w,
                                   dilation, eps, _dtype_str(dtype))
    torch.manual_seed(seed)
    dt = dtype
    W_K = (torch.randn(d_mem, d) * 0.02).to(dt)
    W_V = (torch.randn(d_mem, d) * 0.02).to(dt)
    rms_w_h = torch.ones(d).to(dt)
    rms_w_v = torch.ones(d).to(dt)
    conv_w = (torch.randn(w, d) * 0.02).to(dt)

    cs = torch.zeros(batch, 0, d, dtype=dt)
    max_y, max_s = 0.0, 0.0
    for step in range(steps):
        e_t = (torch.randn(batch, d_mem) * 0.1).to(dt)
        h_t = torch.randn(batch, d).to(dt)
        y_op, ns_op = _run_op(kernel, batch, d_mem, d, max_conv_len, w,
                              dilation, eps, e_t, h_t, cs, W_K, W_V,
                              rms_w_h, rms_w_v, conv_w)
        y_ref, ns_ref = golden_engram_decode(
            e_t, h_t, cs, W_K, W_V, rms_w_h, rms_w_v, conv_w,
            max_conv_len, dilation, eps)
        y_ref = y_ref.float()
        ns_ref = ns_ref.float()
        # The kernel always emits the full ``max_conv_len``-row sliding window;
        # the golden returns only the logically valid rows for a short incoming
        # state.  Front-pad the golden state with zeros to the same window so
        # the two are compared element-wise.
        if ns_ref.shape[1] < max_conv_len:
            ns_ref = F.pad(ns_ref, (0, 0, max_conv_len - ns_ref.shape[1], 0))
        y_err = (y_op - y_ref).abs().max().item()
        s_err = (ns_op - ns_ref).abs().max().item()
        assert y_err < 1e-1, f"step {step}: y max_err={y_err}"
        assert s_err < 5e-2, f"step {step}: state max_err={s_err}"
        max_y, max_s = max(max_y, y_err), max(max_s, s_err)
        cs = ns_op.to(dt)
    print(f"  [ok] {tag or 'multi-step'} steps={steps} "
          f"y_maxerr={max_y:.3e} st_maxerr={max_s:.3e}")
    return max(max_y, max_s)


def _check_state_shift(batch, d_mem, d, max_conv_len, w, dilation, dtype,
                       seed=7):
    """new_conv_state[0:L-1] must equal conv_state[1:L] bit-exactly."""
    kernel = _engram_decode_kernel(batch, d_mem, d, max_conv_len, w,
                                   dilation, 1e-6, _dtype_str(dtype))
    ins = _make_inputs(batch, d_mem, d, max_conv_len, w, dtype, seed)
    e_t, h_t, cs, W_K, W_V, rms_w_h, rms_w_v, conv_w = ins
    _, ns_op = _run_op(kernel, batch, d_mem, d, max_conv_len, w, dilation,
                       1e-6, *ins)
    shifted = cs[:, 1:, :].float()
    got = ns_op[:, :max_conv_len - 1, :]
    d_err = (got - shifted).abs().max().item()
    assert d_err == 0.0, f"state shift mismatch: {d_err}"
    print(f"  [ok] state-shift bit-exact (max_err={d_err:.1e})")


def _check_gate_corner(batch, d_mem, d, max_conv_len, w, dilation, dtype,
                       eps=1e-6):
    """Saturating gate: no NaN/Inf and still matches golden."""
    kernel = _engram_decode_kernel(batch, d_mem, d, max_conv_len, w,
                                   dilation, eps, _dtype_str(dtype))
    torch.manual_seed(11)
    dt = dtype
    e_t = (torch.randn(batch, d_mem) * 10.0).to(dt)
    h_t = (torch.randn(batch, d) * 50.0).to(dt)
    cs = (torch.randn(batch, max_conv_len, d) * 0.5).to(dt)
    W_K = (torch.randn(d_mem, d) * 0.5).to(dt)
    W_V = (torch.randn(d_mem, d) * 0.5).to(dt)
    rms_w_h = torch.full((d,), 5.0).to(dt)
    rms_w_v = torch.full((d,), 5.0).to(dt)
    conv_w = (torch.randn(w, d) * 0.5).to(dt)
    ins = (e_t, h_t, cs, W_K, W_V, rms_w_h, rms_w_v, conv_w)
    y_op, ns_op = _run_op(kernel, batch, d_mem, d, max_conv_len, w, dilation,
                          eps, *ins)
    assert torch.isfinite(y_op).all(), "non-finite y at gate corner"
    assert torch.isfinite(ns_op).all(), "non-finite state at gate corner"
    y_ref, ns_ref = golden_engram_decode(*ins, max_conv_len, dilation, eps)
    atol = 5e-2 if dtype == torch.float16 else 1e-1
    torch.testing.assert_close(y_op, y_ref.float(), atol=atol, rtol=5e-2,
                               equal_nan=True)
    print("  [ok] gate-corner finite + matches golden")


# ---------------------------------------------------------------------------
# Layered test suites
# ---------------------------------------------------------------------------
def run_L0():
    """L0 threshold: manifest workloads + smoke (blocking)."""
    print("[L0] threshold smoke (manifest workloads)")
    run_case(1, 256, 256, 9, 4, 3, torch.float16, tag="smoke")
    run_case(1, 512, 256, 12, 4, 3, torch.float16, tag="L0-2")
    run_case(4, 1024, 512, 20, 4, 5, torch.float16, tag="L0-3")
    run_case(8, 512, 256, 18, 4, 3, torch.bfloat16, tag="L0-4")
    print("[L0] PASS")


def run_L1():
    """L1 core correctness: multi-step recurrence, state consistency, corners
    (blocking)."""
    print("[L1] recurrence / consistency / corners")
    run_multistep(2, 256, 256, 9, 4, 3, torch.float16, tag="L1-multistep")
    run_case(1, 512, 256, 12, 4, 3, torch.bfloat16, tag="L1-bf16")
    _check_state_shift(2, 256, 256, 12, 4, 3, torch.float16)
    _check_gate_corner(2, 512, 256, 12, 4, 3, torch.float16)
    # dilation positions: delta=3 (rows L-9/L-6/L-3/L-0) and delta=5
    run_case(2, 512, 256, 12, 4, 3, torch.float16, tag="L1-dil3")
    run_case(1, 1024, 512, 20, 4, 5, torch.float16, tag="L1-dil5")
    # extra seed
    run_case(2, 256, 256, 9, 4, 3, torch.float16, tag="L1-seed5", seed=5)
    print("[L1] PASS")


def _warn_case(label, fn, *args, **kwargs):
    """Run a non-blocking case; report a warning instead of failing the suite."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - L2/Boundary are warn-only
        print(f"  [warn] {label}: {type(exc).__name__}: {str(exc)[:160]}")
        return None


def run_L2():
    """L2 extended (non-blocking warnings only)."""
    print("[L2] extended coverage")
    # fp32 is not a manifest workload; the npuir compiler currently rejects the
    # fp32-input lowering ("expected result type with stride ... in dim = 0"),
    # so it is exercised here as a warn-only probe.
    _warn_case("L2-fp32", run_case, 2, 256, 256, 9, 4, 3, torch.float32,
               atol=1e-4, rtol=1e-4, tag="L2-fp32")
    # extra seeds / larger batch
    for seed in (101, 202):
        _warn_case(f"L2-seed{seed}", run_case, 4, 512, 256, 16, 4, 3,
                   torch.float16, seed=seed, tag=f"L2-seed{seed}")
    print("[L2] done")


def run_boundary():
    """Boundary shapes (non-blocking warnings only)."""
    print("[Boundary] edge shapes")
    # smallest legal cache: max_conv_len == dilation * (w - 1)
    _warn_case("B-minL", run_case, 1, 256, 256, 9, 4, 3, torch.float16,
               tag="B-minL")
    # w = 2 (one history tap)
    _warn_case("B-w2", run_case, 1, 256, 256, 6, 2, 3, torch.float16,
               tag="B-w2")
    # w = 1 (current tap only; keep one state row so the shift region is non-empty)
    _warn_case("B-w1", run_case, 1, 256, 256, 2, 1, 1, torch.float16,
               tag="B-w1")
    # batch = 1, large d_mem
    _warn_case("B-b1", run_case, 1, 1024, 256, 12, 4, 3, torch.float16,
               tag="B-b1")
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
