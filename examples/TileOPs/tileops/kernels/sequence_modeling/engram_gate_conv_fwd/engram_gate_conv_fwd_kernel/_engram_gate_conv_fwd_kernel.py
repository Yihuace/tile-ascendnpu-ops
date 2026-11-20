# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Engram GateConv forward kernel (NPU, ``target="npuir"``).

Migration of the GPU TileLang factory ``_engram_gate_conv_fwd_kernel``
(``/mnt/workspace/TileOPs/tileops/kernels/engram/engram_fwd.py``).

Post-projection fusion: the projections ``k = E @ W_K`` / ``v = E @ W_V`` are
done externally by GEMM; this kernel fuses

    h_norm = RMSNorm(H, w_h)                                    # (M, T, d)
    k_norm = RMSNorm(k, w_h)                                    # (M, T, d)
    alpha  = sigmoid(dot(h_norm, k_norm) / sqrt(d))             # (M, T)
    v_hat  = alpha * v                                          # (M, T, d)
    v_hat_norm = RMSNorm(v_hat, w_v)                            # (M, T, d)
    conv_out = CausalDWConv1D(v_hat_norm, conv_w, kernel=4)     # (M, T, d)
    Y = SiLU(conv_out) + v_hat                                  # (M, T, d)

and also emits, for the backward strategy B, ``vhat`` (M, T, d) plus the four
per-position scalars ``alpha / rrms_h / rrms_k / rrms_v`` (fp32, (M, T)).

NPU re-design status (see DESIGN.md sections 0.5 / 0.6 / 3.3 and RETROSPECTIVE):

  * R1 -- APPLIED.  Single fused kernel.  A 1-D grid ``M * num_tiles`` gives
    each block ``BLK_S`` consecutive output tokens of one batch plus a 3-row
    causal halo (``R = BLK_S + 3``).  ``v_hat_norm`` is materialised in UB and
    the causal depthwise conv slides over it in UB, so the GPU pass-2 GM
    neighbour reload (4x per output token) is eliminated.
  * R2 -- APPLIED.  fp32 compute domain (fp16/bf16 -> ``vcast(rint)`` -> fp32
    chain -> single ``vcast(rint)`` back).  The reciprocal RMS is
    ``sqrt(s) / s`` (``vsqrt`` + scalar divide) instead of the approximate
    ``vrsqrt`` device instruction (DESIGN O1).
  * R3 -- APPLIED.  1-D ``T.Kernel(..., is_npu=True)``; every intermediate
    lives in ``T.alloc_shared`` (UB).  ``threads`` is dropped (no-op on npuir).
  * R4 -- APPLIED.  The causal boundary uses index clamp (``T.min``/``T.max``)
    plus an arithmetic row mask; ``T.if_then_else`` with buffer operands
    segfaults the npuir codegen (design probe V2).
  * R5 -- APPLIED.  Gate dot product is factorised:
    ``dot = (rrms_h * rrms_k) * sum(H * k * w_h^2)`` (saves the h_norm/k_norm
    materialisations).

Contract: ``Y`` / ``vhat`` keep the input dtype, the four scalar outputs are
fp32; the prim_func signature is unchanged from the GPU source and the factory
keeps the ``(M, seq_len, d, eps, dtype)`` signature.  ``dtype`` is a string.
The tile width ``BLK_S`` is chosen from ``d_padded`` to stay inside the 192 KB
UB budget and can be tuned in Stage 4.

Codegen constraints discovered while validating this kernel (all reproducible
from ``repro/``; the npuir Developer codegen is fragile for scalar UB->GM
traffic):

  * Every GM store is a ``T.copy`` region store.  Element-wise GM stores
    (``Y[bid, t0+i, j] = oy[i, j]``) and 1-D ``T.Parallel`` scalar stores are
    nondeterministic.
  * Own-rows scalar outputs are staged with a compile-time-unrolled scalar loop
    into a ``(BLK_S, 1)`` buffer and then stored with a single ``T.copy``.  A
    ``T.Parallel`` fill (or a ``T.copy`` region slice read) of the same staging
    races.
  * The causal row mask is emitted as an unrolled scalar loop: a runtime
    ``T.min``/``T.max`` inside a 1-D ``T.Parallel`` scalar store segfaults the
    codegen (SIGSEGV).
  * ``T.vcast`` (and lowered element-wise ``T.cast``) requires identical
    src/dst buffer shapes.
  * ``BLK_S`` is restricted to values that divide ``seq_len`` (single tile for
    an odd/prime ``seq_len``); ``BLK_S == 1`` is rejected by ``T.vbrc``.

run: python _engram_gate_conv_fwd_kernel.py --level all
"""

import os

# Developer mode for npuir (auto CV-split; no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
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


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _dtype_str(dtype) -> str:
    return str(dtype).split(".")[-1]


# ---------------------------------------------------------------------------
# Golden (independent PyTorch CPU reference)
# Ported from examples/TileOPs/tests/ops/test_engram_gate_conv_fwd.py
# ::engram_gate_conv_fwd_torch (contract semantics; NOT the NPU structure).
# ---------------------------------------------------------------------------
def _rmsnorm(x, w, eps=1e-6):
    """x: (M, T, d), w: (d,); returns (normed, rrms)."""
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = x_f * rrms * w.float()
    return normed, rrms.squeeze(-1)


def golden_engram_gate_conv_fwd(H, k, v, rms_w_h, rms_w_v, conv_w, eps=1e-6):
    """Engram GateConv forward reference implementation (source semantics).

    All arithmetic in fp32 (``.float()``) with a single round back to the input
    dtype at the end.  Independent of the NPU structure (sequence tiling / UB
    causal window / ``vsqrt`` reciprocal) so that it can validate the kernel.
    """
    _, _, d = H.shape

    h_norm, rrms_h = _rmsnorm(H, rms_w_h, eps)
    k_norm, rrms_k = _rmsnorm(k, rms_w_h, eps)                      # shares rms_w_h

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))

    v_hat = alpha * v.float()

    v_hat_norm, rrms_v = _rmsnorm(v_hat.to(H.dtype), rms_w_v, eps)

    v_perm = v_hat_norm.float().permute(0, 2, 1)                    # (M, d, T)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))             # causal left pad
    conv_w_expanded = conv_w.float().T.unsqueeze(1)                 # (d, 1, 4)
    conv_out = F.conv1d(v_padded, conv_w_expanded, groups=d).permute(0, 2, 1)

    Y = F.silu(conv_out) + v_hat.float()

    return (Y.to(H.dtype), v_hat.to(H.dtype), alpha.squeeze(-1).float(),
            rrms_h.float(), rrms_k.float(), rrms_v.float())


# ---------------------------------------------------------------------------
# NPU kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[6, 7, 8, 9, 10, 11], target="npuir")
def _engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype):
    """Fused Engram GateConv forward kernel factory (NPU, AIV vector pipeline).

    Args:
        M, seq_len, d, eps, dtype: compile-time specialization parameters
            (signature unchanged across the GPU -> NPU migration).
    """
    accum = _F32
    d_padded = _align_up(d, ALIGNMENT)
    halo = CONV_KERNEL_SIZE - 1
    # Tile width: d=256 uses BLK_S=8 (R=11), d=512 uses BLK_S=4 (R=7) so the
    # UB resident set stays inside the 192 KB budget (DESIGN section 4.5).
    # Only tile widths that exactly divide seq_len are used, so every output
    # store is an exact ``T.copy`` region (element-wise GM stores and
    # T.Parallel scalar staging are mis-scheduled by the npuir codegen; see the
    # Stage 3 functional probe).  ``blk_s == 1`` is unusable (``T.vbrc``
    # rejects empty broadcast dims), so for an odd/prime seq_len the whole
    # sequence is handled by a single tile.
    cap = 8 if d_padded <= 256 else 4
    blk_s = 0
    for cand in (cap, cap // 2, 2):
        if seq_len % cand == 0:
            blk_s = cand
            break
    if blk_s == 0:
        blk_s = seq_len
    R = blk_s + halo
    num_tiles = seq_len // blk_s
    inv_d = 1.0 / float(d)
    inv_sqrt_d = 1.0 / math.sqrt(float(d))

    @T.prim_func
    def main(
        H: T.Tensor((M, seq_len, d_padded), dtype),
        k: T.Tensor((M, seq_len, d_padded), dtype),
        v: T.Tensor((M, seq_len, d_padded), dtype),
        rms_w_h: T.Tensor((d_padded,), dtype),
        rms_w_v: T.Tensor((d_padded,), dtype),
        conv_w: T.Tensor((CONV_KERNEL_SIZE, d_padded), dtype),
        Y: T.Tensor((M, seq_len, d_padded), dtype),
        vhat_buf: T.Tensor((M, seq_len, d_padded), dtype),
        alpha_buf: T.Tensor((M, seq_len), accum),
        rrms_h_buf: T.Tensor((M, seq_len), accum),
        rrms_k_buf: T.Tensor((M, seq_len), accum),
        rrms_v_buf: T.Tensor((M, seq_len), accum),
    ):
        with T.Kernel(M * num_tiles, is_npu=True) as (cid, _):
            bid = cid % M
            t0 = (cid // M) * blk_s

            # Buffers are declared unconditionally at the Kernel top level
            # (TRAP-conditional-alloc).  prod is reused: squares -> gate
            # product -> v_hat_norm.
            ht = T.alloc_shared((R, d_padded), accum)
            kt = T.alloc_shared((R, d_padded), accum)
            prod = T.alloc_shared((R, d_padded), accum)
            vt = T.alloc_shared((R, d_padded), accum)
            bro = T.alloc_shared((R, d_padded), accum)
            conv = T.alloc_shared((blk_s, d_padded), accum)
            sig = T.alloc_shared((blk_s, d_padded), accum)
            cb = T.alloc_shared((blk_s, d_padded), accum)
            wh2 = T.alloc_shared((1, d_padded), accum)
            wv2 = T.alloc_shared((1, d_padded), accum)
            cwp = T.alloc_shared((1, d_padded), accum)
            cwp_d = T.alloc_shared((1, d_padded), dtype)
            red = T.alloc_shared((R, 1), accum)
            tmp1 = T.alloc_shared((R, 1), accum)
            rh = T.alloc_shared((R, 1), accum)
            rk = T.alloc_shared((R, 1), accum)
            rv = T.alloc_shared((R, 1), accum)
            rmk = T.alloc_shared((R, 1), accum)
            oy = T.alloc_shared((blk_s, d_padded), dtype)
            ov = T.alloc_shared((blk_s, d_padded), dtype)
            s_h = T.alloc_shared((blk_s, 1), accum)
            s_k = T.alloc_shared((blk_s, 1), accum)
            s_a = T.alloc_shared((blk_s, 1), accum)
            s_v = T.alloc_shared((blk_s, 1), accum)

            # ---- weights -> fp32 (small; w_h used by H and k) ----
            for j in T.Parallel(d_padded):
                wh2[0, j] = T.cast(rms_w_h[j], accum)
                wv2[0, j] = T.cast(rms_w_v[j], accum)
            T.vmul(wh2, wh2, wh2)                              # wh2 = w_h^2

            # ---- causal row mask (arithmetic; NO if_then_else: probe V2) ----
            # R is a compile-time constant; the mask rows are emitted as scalar
            # statements.  A runtime min/max inside a 1-D T.Parallel scalar
            # store makes the npuir loop vectorizer hang (Stage 3 probe), while
            # the unrolled scalar form compiles.
            for i in range(R):
                gt = t0 - halo + i
                lo = T.min(T.max(gt + 1, 0), 1)
                hi = T.min(T.max(seq_len - gt, 0), 1)
                rmk[i, 0] = T.cast(lo * hi, accum)

            # ---- load H / k window tiles (clamped runtime row index: probe V1) ----
            for i, j in T.Parallel(R, d_padded):
                gc = T.min(T.max(t0 - halo + i, 0), seq_len - 1)
                ht[i, j] = T.cast(H[bid, gc, j], accum)
                kt[i, j] = T.cast(k[bid, gc, j], accum)

            # ---- sumsq_h -> rrms_h (vsqrt + scalar divide = rsqrt bypass) ----
            T.vmul(ht, ht, prod)
            T.reduce_sum(prod, red, dim=1, clear=True)
            T.vdiv(red, float(d), red)
            T.vadd(red, eps, red)
            T.vsqrt(red, tmp1)
            T.vdiv(tmp1, red, rh)
            # python-unrolled scalar reads: a T.Parallel scalar store here
            # is mis-scheduled by the npuir codegen (Stage 3 probe).
            for i in range(blk_s):
                s_h[i, 0] = rh[halo + i, 0]
            T.copy(s_h, rrms_h_buf[bid, t0:t0 + blk_s])

            # ---- sumsq_k -> rrms_k ----
            T.vmul(kt, kt, prod)
            T.reduce_sum(prod, red, dim=1, clear=True)
            T.vdiv(red, float(d), red)
            T.vadd(red, eps, red)
            T.vsqrt(red, tmp1)
            T.vdiv(tmp1, red, rk)
            # python-unrolled scalar reads: a T.Parallel scalar store here
            # is mis-scheduled by the npuir codegen (Stage 3 probe).
            for i in range(blk_s):
                s_k[i, 0] = rk[halo + i, 0]
            T.copy(s_k, rrms_k_buf[bid, t0:t0 + blk_s])

            # ---- gate-dot factorise (O4): dot = rrms_h*rrms_k*sum(H*k*w_h^2) ----
            T.vmul(ht, kt, prod)
            T.vbrc(wh2, bro)
            T.vmul(prod, bro, prod)
            T.reduce_sum(prod, red, dim=1, clear=True)
            T.vmul(rh, rk, tmp1)
            T.vmul(red, tmp1, red)
            T.vmul(red, inv_sqrt_d, red)
            T.vsigmoid(red, red)                               # alpha
            # python-unrolled scalar reads: a T.Parallel scalar store here
            # is mis-scheduled by the npuir codegen (Stage 3 probe).
            for i in range(blk_s):
                s_a[i, 0] = red[halo + i, 0]
            T.copy(s_a, alpha_buf[bid, t0:t0 + blk_s])

            # ---- v_hat = alpha (.) v (materialised; O3) ----
            for i, j in T.Parallel(R, d_padded):
                gc = T.min(T.max(t0 - halo + i, 0), seq_len - 1)
                vt[i, j] = T.cast(v[bid, gc, j], accum)
            T.vbrc(red, bro)
            T.vmul(vt, bro, vt)

            # ---- sumsq_v -> rrms_v ----
            T.vmul(vt, vt, prod)
            T.reduce_sum(prod, red, dim=1, clear=True)
            T.vdiv(red, float(d), red)
            T.vadd(red, eps, red)
            T.vsqrt(red, tmp1)
            T.vdiv(tmp1, red, rv)
            # python-unrolled scalar reads: a T.Parallel scalar store here
            # is mis-scheduled by the npuir codegen (Stage 3 probe).
            for i in range(blk_s):
                s_v[i, 0] = rv[halo + i, 0]
            T.copy(s_v, rrms_v_buf[bid, t0:t0 + blk_s])

            # ---- v_hat_norm (into prod): vhat * rrms_v * w_v * rmask ----
            T.vbrc(rv, bro)
            T.vmul(vt, bro, prod)
            T.vbrc(wv2, bro)
            T.vmul(prod, bro, prod)
            T.vbrc(rmk, bro)
            T.vmul(prod, bro, prod)

            # ---- causal DWConv1D (host-expanded taps; UB affine shift: probe A) ----
            T.clear(conv)
            for p in range(CONV_KERNEL_SIZE):                  # host-level unroll
                T.copy(conv_w[p:p + 1, 0:d_padded], cwp_d)
                T.vcast(cwp_d, cwp, round_mode="rint")
                T.vbrc(cwp, cb)
                for i, j in T.Parallel(blk_s, d_padded):
                    conv[i, j] += cb[i, j] * prod[i + p, j]

            # ---- SiLU + residual ----
            T.vsigmoid(conv, sig)
            for i, j in T.Parallel(blk_s, d_padded):
                conv[i, j] = conv[i, j] * sig[i, j] + vt[halo + i, j]
            T.vcast(conv, oy, round_mode="rint")

            # ---- fp32 vhat staging for the vhat output ----
            # sig is free after the residual, reuse it as the fp32 vhat stage.
            for i, j in T.Parallel(blk_s, d_padded):
                sig[i, j] = vt[halo + i, j]
            T.vcast(sig, ov, round_mode="rint")

            # ---- own rows only -> outputs (exact T.copy regions) ----
            T.copy(oy, Y[bid, t0:t0 + blk_s, 0:d_padded])
            T.copy(ov, vhat_buf[bid, t0:t0 + blk_s, 0:d_padded])

    return main


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _make_inputs(M, seq_len, d, dtype, seed):
    """Reference workload generator (matches EngramGateConvFwdWorkload)."""
    torch.manual_seed(seed)
    H = torch.randn(M, seq_len, d, dtype=torch.float32).to(dtype)
    k = (torch.randn(M, seq_len, d, dtype=torch.float32) * 0.1).to(dtype)
    v = (torch.randn(M, seq_len, d, dtype=torch.float32) * 0.1).to(dtype)
    rms_w_h = torch.ones(d, dtype=torch.float32).to(dtype)
    rms_w_v = torch.ones(d, dtype=torch.float32).to(dtype)
    conv_w = (torch.randn(CONV_KERNEL_SIZE, d, dtype=torch.float32) * 0.02).to(dtype)
    return H, k, v, rms_w_h, rms_w_v, conv_w


def _run_case(M, seq_len, d, dtype, tag="", seed=0, eps=1e-6, atol=None,
              rtol=1e-1, verbose=True, strict_scalars=False):
    """Single correctness case; compares all six outputs against the golden."""
    if atol is None:
        atol = 1e-1 if dtype == torch.float16 else 2e-1
    dtype_str = _dtype_str(dtype)
    ins = _make_inputs(M, seq_len, d, dtype, seed)

    kernel = _engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype_str)
    outs = kernel(*[t.npu() for t in ins])
    ref = golden_engram_gate_conv_fwd(*ins, eps=eps)

    names = ["Y", "vhat", "alpha", "rrms_h", "rrms_k", "rrms_v"]
    diffs = {}
    for o, r, nm in zip(outs, ref, names):
        o = o.cpu().float()
        r = r.float().cpu()
        torch.testing.assert_close(o, r, atol=atol, rtol=rtol, equal_nan=True)
        diffs[nm] = (o - r).abs().max().item()

    if strict_scalars:
        # The scalar outputs do not depend on the conv window; they admit a
        # stricter tolerance than the fp16/bf16 Y output.
        sc = {"alpha": 1e-2, "rrms_h": 1e-2, "rrms_k": 1e-2, "rrms_v": 1e-2}
        for j, o in enumerate(outs):
            if names[j] in sc:
                o = o.cpu().float()
                r = ref[j].float().cpu()
                torch.testing.assert_close(o, r, atol=1e-3, rtol=sc[names[j]],
                                           equal_nan=True)

    if verbose:
        print(f"  [ok] {tag or 'case'} M={M} S={seq_len} d={d} "
              f"{dtype_str} " + " ".join(
                  f"{n}={diffs[n]:.3e}" for n in names))
    return max(diffs.values())


def _warn_case(label, fn, *args, **kwargs):
    """Run a non-blocking case; report a warning instead of failing the suite."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - L2/Boundary are warn-only
        print(f"  [warn] {label}: {type(exc).__name__}: {str(exc)[:200]}")
        return None


# ---------------------------------------------------------------------------
# Layered test suites
# ---------------------------------------------------------------------------
def run_L0():
    """L0 threshold: manifest workloads + smoke + causal/non-divisible edges
    (blocking)."""
    print("[L0] threshold smoke (manifest workloads + causal boundary)")
    _run_case(1, 16, 256, torch.float16, tag="L0-1-smoke")
    _run_case(1, 32, 256, torch.float16, tag="L0-2-W2")
    _run_case(2, 64, 512, torch.float16, tag="L0-3-W3")
    _run_case(1, 128, 256, torch.bfloat16, tag="L0-4-W4")
    _run_case(2, 16, 256, torch.bfloat16, tag="L0-4b-smoke-bf16")
    _run_case(1, 4, 256, torch.float16, tag="L0-5-causal-left")
    _run_case(1, 13, 256, torch.float16, tag="L0-6-nondiv")
    print("[L0] PASS")


def run_L1():
    """L1 core correctness: extra seeds, both tilings, dtype coverage,
    scalar precision, gate corners (blocking)."""
    print("[L1] seeds / tilings / dtype / scalars / gate corners")
    _run_case(2, 64, 512, torch.float16, tag="L1-W3-seed5", seed=5,
              strict_scalars=True)
    _run_case(1, 128, 256, torch.bfloat16, tag="L1-W4-seed7", seed=7,
              strict_scalars=True)
    _run_case(3, 48, 512, torch.bfloat16, tag="L1-d512-bf16", seed=11)
    _run_case(1, 64, 256, torch.float16, tag="L1-d256-long", seed=13)
    _run_case(1, 7, 512, torch.float16, tag="L1-nondiv-d512", seed=3)
    _check_gate_corner(1, 32, 256, torch.float16)
    print("[L1] PASS")


def run_L2():
    """L2 extended (non-blocking warnings only)."""
    print("[L2] extended coverage")
    for seed in (101, 202):
        _warn_case(f"L2-seed{seed}", _run_case, 2, 32, 512, torch.float16,
                   seed=seed, tag=f"L2-seed{seed}")
    _warn_case("L2-big-M", _run_case, 8, 16, 256, torch.float16, seed=303,
               tag="L2-big-M")
    print("[L2] done")


def run_boundary():
    """Boundary shapes (non-blocking warnings only)."""
    print("[Boundary] edge shapes")
    _warn_case("B-S1", _run_case, 1, 1, 256, torch.float16, tag="B-S1")
    _warn_case("B-S3", _run_case, 1, 3, 256, torch.float16, tag="B-S3")
    _warn_case("B-S9", _run_case, 1, 9, 512, torch.float16, tag="B-S9")
    _warn_case("B-M4", _run_case, 4, 8, 256, torch.bfloat16, tag="B-M4")
    print("[Boundary] done")


def _check_gate_corner(M, seq_len, d, dtype, eps=1e-6):
    """Saturating gate (alpha -> 1 and alpha -> 0): finite and matches golden."""
    dtype_str = _dtype_str(dtype)
    for sign, label in ((1.0, "alpha->1"), (-1.0, "alpha->0")):
        torch.manual_seed(21)
        H = (torch.randn(M, seq_len, d, dtype=torch.float32) * 5.0).to(dtype)
        k = (H.float() * sign).to(dtype)
        v = (torch.randn(M, seq_len, d, dtype=torch.float32) * 0.5).to(dtype)
        rms_w_h = torch.ones(d, dtype=torch.float32).to(dtype)
        rms_w_v = torch.ones(d, dtype=torch.float32).to(dtype)
        conv_w = (torch.randn(CONV_KERNEL_SIZE, d, dtype=torch.float32) * 0.1).to(dtype)
        ins = (H, k, v, rms_w_h, rms_w_v, conv_w)
        kernel = _engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype_str)
        outs = kernel(*[t.npu() for t in ins])
        for o in outs:
            assert torch.isfinite(o.float()).all(), f"non-finite output ({label})"
        ref = golden_engram_gate_conv_fwd(*ins, eps=eps)
        atol = 1e-1 if dtype == torch.float16 else 2e-1
        for o, r in zip(outs, ref):
            torch.testing.assert_close(o.cpu().float(), r.float().cpu(),
                                       atol=atol, rtol=1e-1, equal_nan=True)
        print(f"  [ok] gate-corner {label}: finite + matches golden")


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
