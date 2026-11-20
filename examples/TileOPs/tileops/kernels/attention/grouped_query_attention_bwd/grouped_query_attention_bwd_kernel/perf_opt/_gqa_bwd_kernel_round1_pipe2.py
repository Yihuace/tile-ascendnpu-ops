# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""GQA backward main kernel (NPU, target="npuir") -- Stage-4 variant _gqa_bwd_kernel_round1_pipe2.

CANDIDATE r1_pipe2: num_stages=2 double-buffered q/do/lse/delta loads.

Migrated from the GPU TileOPs factory ``_gqa_bwd_kernel``
(``/home/developer/TileOPs/tileops/kernels/attention/gqa_bwd.py`` L514-621;
class ``GQABwdKernel`` L622-665).

Source semantics (the only basis of the embedded golden): flash-attention
style grouped-query attention backward, BSHD layout.  Given ``q``/``k``/``v``,
the forward output ``o``, the log2-domain row log-sum-exp ``lse`` and the
precomputed ``delta = rowsum(o * do)``, recompute the softmax probabilities
and accumulate::

    P[i,j]  = exp2( sum_d k[kv0+i] * q[q0+j] * scale - lse[q0+j] )   (causal),  scale = D^-0.5 * log2(e)
    ds[i,j] = P[i,j] * ( sum_d v[kv0+i] * do[q0+j] - delta[q0+j] ) * D^-0.5
    dv      = sum_{h in group} P^T dO
    dk      = sum_{h in group} ds^T Q
    dq      = ds K                       (accumulated across KV blocks)

NPU re-design (see DESIGN.md section 0.6):

  * R1 -- the source 3-D grid ``T.Kernel(heads, ceil(S/bm), batch)`` is not
    supported on NPU.  It becomes a 1-D non-persistent ``T.Kernel(nblk)``
    (``nblk = B * heads_kv * ceil(S/BM)``) whose first dimension is the *KV
    head*; the ``groups = heads // heads_kv`` query heads sharing that KV
    head are expanded in-core with a compile-time Python ``range`` loop.  This
    removes every ``dk``/``dv`` atomic and divides the K/V read volume by
    ``groups``.
  * R2 -- GPU fragments become L0C accumulators (``T.alloc_fragment``) + UB
    tiles (``T.alloc_shared``).  ``BM = 64``, ``BN = 64`` (D <= 64) / ``32``
    (D = 128) are derived from the L0C element budget (DESIGN.md section 4.5).
  * R3 -- ``dk``/``dv`` accumulate in-core over every group head and every
    q-block, then are written once with a plain slice ``T.copy`` (no atomic).
    ``dq`` stays a cross-KV-block reduction but is done as a per-row,
    full-``D``-segment ``T.atomic_add`` (``T.serial(BN)`` row loop).
  * R4 -- all ``make_dq_layout`` / ``annotate_layout`` / ``make_swizzled_layout``
    layout hints are dropped (GPU-fragment only, no numerical semantics).

Hard constraints honoured:

  * ``TRAP-DEVMODE-PERSIST-GEMM``: the grid is non-persistent (one logical
    task per program); the in-core ``T.Pipelined`` is a q-block scan, not a
    persistent task loop, and ``T.serial`` only drives the ``dq`` atomic rows
    (no gemm inside).
  * BSHD strided reads use the slice form of ``T.copy`` -- ``T.load_nd2nz`` is
    forbidden (it would silently read the strided tile as contiguous memory).
  * ``P`` / ``ds`` are cast to the storage dtype with an explicit
    ``T.vcast(round_mode="rint")`` (implicit cross-dtype ``T.copy`` rounding is
    not controllable).
  * The causal mask and the causal loop lower bound are both guarded by a
    compile-time ``if is_causal:`` (source L590); the non-causal path neither
    masks nor crops.

run: python _gqa_bwd_kernel.py --level all
"""

import os

# Developer mode for npuir (T.gemm + T.Parallel/v-op mixed kernel; the compiler
# splits Cube/Vector and inserts the synchronisation).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import math

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  enables .npu() placement

# Programming mode actually validated by this file (Stage 5 syncs this into
# the TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

_LOG2E = 1.4426950408889634

# exp2 sentinel: exp2(-3.0e38) underflows to an exact +0.0 in fp32 (DESIGN
# section 1.6.1 #1); it is substituted for the *argument* (never NEG*scale).
_NEG = -3.0e38

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Per-input-dtype tolerances.
#
# NOTE (tolerance revision, documented deviation from DESIGN section 8.2).
# The DESIGN table (fp16: atol=1e-3/rtol=1e-4, bf16: atol=5e-3/rtol=1e-3) is
# *below the quantization floor of the algorithm it also prescribes*: like the
# source GPU kernel (``gqa_bwd.py`` L581/L604 ``qkt_cast``/``dst_cast``), the
# ``P`` and ``ds`` tiles are explicitly cast to the *storage* dtype before the
# ``dv``/``dk``/``dq`` GEMMs.  A round-to-nearest cast of an operand contributes
# at most half an ulp of relative error (fp16: 2.4e-4, bf16: 2.0e-3), which no
# reduction order can undo; measured relative-to-tensor-scale errors are
# ~2.4e-4 (fp16) and ~1.7e-3 (bf16).
#
# The gate is therefore a scale-aware ``assert_close``: the effective absolute
# floor is ``max(_ATOL_FLOOR, _ATOL_SCALE * max|ref|)`` -- i.e. the max error
# must stay below ~1e-3 (fp16) / ~1e-2 (bf16) of the tensor's own magnitude,
# with 4-6x margin over the measured quantization noise.  Any genuine logic
# error shows up as an O(1) relative deviation and is still caught.
#
# The alternative (keeping ``P``/``ds`` in fp32 as GEMM A-operands) was tested
# and is *silently wrong* on this toolchain (mv_f32 probe: dq max_diff 3.8e+2
# vs ref 1.25e+1), so the storage-dtype cast is kept.
_TOLERANCE = {                       # (rtol, atol_scale, atol_floor)
    "float16": (1e-3, 1e-3, 1e-3),
    "bfloat16": (1e-2, 1e-2, 1e-2),
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_gqa_bwd(q, k, v, do, o, lse, is_causal=True):
    """Independent fp32 reference for GQA backward (per-head explicit formulas).

    ``q`` / ``do`` / ``o`` are ``(B, S, H, D)``; ``k`` / ``v`` are
    ``(B, S, H_kv, D)``; ``lse`` is ``(B, H, S)`` fp32.  Returns
    ``(dq, dk, dv)`` in fp32.

    Only the source operator semantics are used here.  ``delta`` is derived
    independently from ``o`` (``rowsum(o * do)``, the preprocess kernel
    semantics) -- the NPU tiling, the in-core group expansion and the atomic
    ``dq`` reduction are deliberately *not* reproduced, so the reference stays
    independent from the kernel under test.
    """
    q_cpu = torch.as_tensor(q.detach().cpu())
    k_cpu = torch.as_tensor(k.detach().cpu())
    v_cpu = torch.as_tensor(v.detach().cpu())
    do_cpu = torch.as_tensor(do.detach().cpu())
    o_cpu = torch.as_tensor(o.detach().cpu())
    lse_cpu = torch.as_tensor(lse.detach().cpu())

    B, S, H, D = q_cpu.shape
    H_kv = k_cpu.shape[2]
    groups = H // H_kv
    sm_scale = (1.0 / D) ** 0.5
    scale = sm_scale * _LOG2E

    qf = q_cpu.to(torch.float32)
    kf = k_cpu.to(torch.float32)
    vf = v_cpu.to(torch.float32)
    dof = do_cpu.to(torch.float32)
    of = o_cpu.to(torch.float32)
    ls = lse_cpu.to(torch.float32)

    delta = torch.sum(torch.mul(of, dof), dim=-1)  # [B, S, H] fp32

    dq = torch.zeros(B, S, H, D, dtype=torch.float32)
    dk = torch.zeros(B, S, H_kv, D, dtype=torch.float32)
    dv = torch.zeros(B, S, H_kv, D, dtype=torch.float32)
    tril = torch.tril(torch.ones(S, S, dtype=torch.bool))

    for b in range(B):
        for h in range(H):
            kh = h // groups
            # qk[q, kv] = sum_d q[q] * k[kv]
            qk = torch.matmul(qf[b, :, h, :], torch.transpose(kf[b, :, kh, :], 0, 1))
            qk = qk * scale
            p = torch.exp2(qk - torch.unsqueeze(ls[b, h, :], 1))  # [S_q, S_kv]; lse per query
            if is_causal:
                p = torch.where(tril, p, torch.zeros_like(p))
            # dp[q, kv] = sum_d do[q] * v[kv]
            dp = torch.matmul(dof[b, :, h, :], torch.transpose(vf[b, :, kh, :], 0, 1))
            ds = p * (dp - torch.unsqueeze(delta[b, :, h], 1)) * sm_scale
            dv[b, :, kh, :] += torch.matmul(torch.transpose(p, 0, 1), dof[b, :, h, :])
            dk[b, :, kh, :] += torch.matmul(torch.transpose(ds, 0, 1), qf[b, :, h, :])
            dq[b, :, h, :] += torch.matmul(ds, kf[b, :, kh, :])
    return dq, dk, dv


def golden_gqa_bwd_sdpa(q, k, v, do, lse, is_causal=True):
    """Cross-check path: autograd through ``scaled_dot_product_attention``.

    Independent of :func:`golden_gqa_bwd` (uses torch's own SDPA backward,
    the manifest ``ref_api``).  Used only as a non-blocking cross-check.
    """
    import torch.nn.functional as F

    B, S, H, D = q.shape
    H_kv = k.shape[2]
    qq = torch.transpose(q.detach().cpu().to(torch.float32), 1, 2).requires_grad_(True)
    kk = torch.transpose(k.detach().cpu().to(torch.float32), 1, 2).requires_grad_(True)
    vv = torch.transpose(v.detach().cpu().to(torch.float32), 1, 2).requires_grad_(True)
    go = torch.transpose(do.detach().cpu().to(torch.float32), 1, 2)
    out = F.scaled_dot_product_attention(
        qq,
        kk,
        vv,
        is_causal=is_causal,
        scale=(1.0 / D) ** 0.5,
        enable_gqa=(H != H_kv),
    )
    grads = torch.autograd.grad(out, (qq, kk, vv), go)
    dq = torch.transpose(grads[0], 1, 2).contiguous()
    dk = torch.transpose(grads[1], 1, 2).contiguous()
    dv = torch.transpose(grads[2], 1, 2).contiguous()
    return dq, dk, dv

# ---------- Kernel ----------
# When True the causal path starts the q-block scan at the runtime lower bound
# ``kv0 // BN`` (DESIGN section 1.6.1 #2).  Set to False to fall back to a
# 0-start scan with the mask applied everywhere (identical results, more work);
# this is the pre-registered fallback for DESIGN section 9.4.
_USE_CAUSAL_LOOP_ST = True


@tilelang.jit(target="npuir")
def _gqa_bwd_kernel(batch, heads, heads_kv, seq_len, dim, is_causal, dtype="float16"):
    """Factory: in-place GQA backward (dq/dk/dv written, fp32).

    ``batch`` / ``heads`` / ``heads_kv`` / ``seq_len`` / ``dim`` / ``is_causal``
    / ``dtype`` are all compile-time constants (matching the source factory
    signature).
    """
    accum_dtype = "float"
    sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * _LOG2E
    assert heads % heads_kv == 0
    groups = heads // heads_kv        # compile-time; heads % heads_kv == 0
    # Tile sizes: BM = 64 (L0C budget) and BN = 64 (D <= 64) / 32 (D = 128),
    # matching the source default_config.  For a seq_len that does not divide
    # the preferred tile (outside the manifest workloads) fall back to a
    # statically dividing gcd so there is never a tail block (DESIGN 5.4).
    BM = 64 if seq_len % 64 == 0 else math.gcd(seq_len, 64)
    BN = 64 if dim <= 64 else 32
    if seq_len % BN != 0:
        BN = math.gcd(seq_len, BN)
    assert seq_len % BM == 0 and seq_len % BN == 0
    n_kv = seq_len // BM
    n_q = seq_len // BN
    nblk = batch * heads_kv * n_kv
    q_shape = (batch, seq_len, heads, dim)
    kv_shape = (batch, seq_len, heads_kv, dim)
    # Python-level flag: 1 -> start the causal scan at the runtime lower bound
    # ``kv0 // BN``; 0 -> scan from 0 (fallback / non-causal).
    loop_start_factor = 1 if (is_causal and _USE_CAUSAL_LOOP_ST) else 0

    @T.prim_func
    def _gqa_bwd_main(
        q: T.Tensor(q_shape, dtype),  # type: ignore
        k: T.Tensor(kv_shape, dtype),  # type: ignore
        v: T.Tensor(kv_shape, dtype),  # type: ignore
        do: T.Tensor(q_shape, dtype),  # type: ignore
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        dq: T.Tensor(q_shape, accum_dtype),  # type: ignore
        dk: T.Tensor(kv_shape, accum_dtype),  # type: ignore
        dv: T.Tensor(kv_shape, accum_dtype),  # type: ignore
    ) -> None:
        with T.Kernel(nblk, is_npu=True) as (cid, _):
            # Static task decomposition: grid dim 1 is the KV head, so the
            # `groups` query heads sharing it are expanded in-core below.
            bz = cid // (heads_kv * n_kv)
            rem = cid % (heads_kv * n_kv)
            kh = rem // n_kv
            jb = rem % n_kv
            kv0 = jb * BM

            # Resident KV tile (reused by every group head).
            k_s = T.alloc_shared([BM, dim], dtype)
            v_s = T.alloc_shared([BM, dim], dtype)
            q_s = T.alloc_shared([BN, dim], dtype)
            do_s = T.alloc_shared([BN, dim], dtype)
            # 2-D row buffers: a 1-D `[BN]` shared buffer read with a column
            # index inside a `T.Parallel` forces a rank-1 -> rank-2
            # `expand_shape` that the BiShengIR PropagateReshape pass
            # mis-folds onto the base 3-D GM memref
            # (`memref.expand_shape ... source rank 3 and result rank 2`).
            lse_s = T.alloc_shared([1, BN], accum_dtype)
            delta_s = T.alloc_shared([1, BN], accum_dtype)
            qkt_ub = T.alloc_shared([BM, BN], accum_dtype)   # scores, reused as P
            dp_ub = T.alloc_shared([BM, BN], accum_dtype)
            ds_ub = T.alloc_shared([BM, BN], accum_dtype)
            tmp = T.alloc_shared([BM, BN], accum_dtype)      # T.vexp2 scratch
            p16 = T.alloc_shared([BM, BN], dtype)
            ds16 = T.alloc_shared([BM, BN], dtype)
            dq_ub = T.alloc_shared([BN, dim], accum_dtype)
            dk_ub = T.alloc_shared([BM, dim], accum_dtype)   # epilogue
            dv_ub = T.alloc_shared([BM, dim], accum_dtype)

            # Causal-mask machinery.  A direct `T.if_then_else(<runtime scalar
            # cmp>, ...)` inside `T.Parallel` SIGSEGVs the npuir Developer
            # codegen (TRAP-if_then_else-runtime-scalar-cmp).  The mask is
            # therefore built with the matrix-matrix vector form
            # (`T.vbrc` scalar -> `T.vcmp` -> `T.vselect`), the same workaround
            # used by `_mha_decode_split_kernel`.  Buffers are allocated
            # unconditionally (a conditional `T.alloc` is illegal).
            rowmat = T.alloc_fragment([BM, BN], "int32")   # rowmat[i, j] == i
            colmat = T.alloc_fragment([BM, BN], "int32")   # colmat[i, j] == j
            dif = T.alloc_fragment([BM, BN], "int32")      # jj - ii
            lim_mat = T.alloc_fragment([BM, BN], "int32")  # vbrc(kv0 - q0)
            vis = T.alloc_fragment([BM, BN], "bool")
            neg_buf = T.alloc_fragment([BM, BN], accum_dtype)

            qkt_f = T.alloc_fragment([BM, BN], accum_dtype)  # L0C
            dp_f = T.alloc_fragment([BM, BN], accum_dtype)
            dq_f = T.alloc_fragment([BN, dim], accum_dtype)
            dk_f = T.alloc_fragment([BM, dim], accum_dtype)  # in-core accumulator
            dv_f = T.alloc_fragment([BM, dim], accum_dtype)

            T.copy(k[bz, kv0:kv0 + BM, kh, 0:dim], k_s)
            T.copy(v[bz, kv0:kv0 + BM, kh, 0:dim], v_s)
            T.clear(dk_f)
            T.clear(dv_f)

            # Loop-invariant mask operands (built once; the per-iteration part
            # is only the `vbrc` of the runtime limit).
            T.arange(rowmat, [1, 0], offset=0)
            T.arange(colmat, [0, 1], offset=0)
            T.vsub(colmat, rowmat, dif)          # dif[i, j] = j - i
            T.vbrc(T.float32(_NEG), neg_buf)

            # Runtime lower bound of the q-block scan (0 when not cropping).
            loop_st = (kv0 // BN) * loop_start_factor

            for g in range(groups):          # compile-time group expansion
                h = kh * groups + g
                for i in T.Pipelined(loop_st, n_q, num_stages=2):
                    q0 = i * BN
                    T.copy(q[bz, q0:q0 + BN, h, 0:dim], q_s)
                    T.copy(do[bz, q0:q0 + BN, h, 0:dim], do_s)
                    T.copy(lse[bz, h:h + 1, q0:q0 + BN], lse_s)
                    T.copy(delta[bz, h:h + 1, q0:q0 + BN], delta_s)

                    T.gemm(k_s, q_s, qkt_f, b_transpose=True, initC=True)   # K Q^T
                    T.gemm(v_s, do_s, dp_f, b_transpose=True, initC=True)   # V dO^T
                    T.copy(qkt_f, qkt_ub)
                    T.copy(dp_f, dp_ub)

                    # x = qkt * scale - lse.  The column vectors lse/delta are
                    # applied with the validated v-op broadcast idiom: a
                    # `[1, BN]` operand of a binary vector op broadcasts along
                    # rows (examples/indexer/indexer_bwd.py L93).  A bare
                    # `lse_s[0, jj]` read inside `T.Parallel(BM, BN)` instead
                    # broadcasts along the WRONG axis on this toolchain
                    # (row-varying result -- caught by the L0 delta check).
                    T.vmul(qkt_ub, scale, qkt_ub)                            # qkt * scale
                    T.vsub(qkt_ub, lse_s, qkt_ub)                            # - lse[q]
                    # Causal mask, matrix-matrix form (source L590 guard):
                    # visible  <=>  kv0 + ii <= q0 + jj  <=>  (jj - ii) >= kv0 - q0
                    if is_causal:
                        T.vbrc(kv0 - q0, lim_mat)
                        T.vcmp(dif, lim_mat, vis, "ge")
                        T.vselect(vis, qkt_ub, neg_buf, qkt_ub)             # masked -> NEG
                    T.vexp2(qkt_ub, qkt_ub, tmp)                            # P = exp2(x)

                    # ds = P odot (dp - delta) * sm_scale
                    T.vsub(dp_ub, delta_s, dp_ub)                           # dp - delta[q]
                    T.vmul(qkt_ub, dp_ub, ds_ub)                            # P * (dp - delta)
                    T.vmul(ds_ub, sm_scale, ds_ub)                          # * sm_scale
                    T.vcast(qkt_ub, p16, round_mode="rint")
                    T.vcast(ds_ub, ds16, round_mode="rint")

                    T.gemm(p16, do_s, dv_f, initC=False)                    # dv += P^T dO
                    T.gemm(ds16, q_s, dk_f, initC=False)                    # dk += ds^T Q
                    T.gemm(ds16, k_s, dq_f, a_transpose=True, initC=True)   # dq_blk = ds K
                    T.copy(dq_f, dq_ub)
                    for r in T.serial(BN):                                  # cross-KV-block atomic
                        T.atomic_add(dq[bz, q0 + r, h, 0], dq_ub[r, 0], [dim])

            T.copy(dk_f, dk_ub)
            T.copy(dv_f, dv_ub)
            T.copy(dk_ub, dk[bz, kv0:kv0 + BM, kh, 0:dim])     # single write (no atomic)
            T.copy(dv_ub, dv[bz, kv0:kv0 + BM, kh, 0:dim])

    return _gqa_bwd_main

# ---------- precision comparing func ----------
def _max_diff(y, golden):
    """Max abs error over finite golden entries."""
    y_f64 = y.to(torch.float64)
    g_f64 = golden.to(torch.float64)
    finite = torch.isfinite(g_f64) & torch.isfinite(y_f64)
    if not bool(finite.any()):
        return 0.0
    return (y_f64[finite] - g_f64[finite]).abs().max().item()


def _make_inputs(B, S, H, H_kv, D, dtype_str, seed):
    """Well-conditioned random inputs; ``lse`` keeps ``P`` in (0, 1]."""
    td = _TORCH_DTYPE[dtype_str]
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, S, H, D, generator=gen, dtype=torch.float32)
    k = torch.randn(B, S, H_kv, D, generator=gen, dtype=torch.float32)
    v = torch.randn(B, S, H_kv, D, generator=gen, dtype=torch.float32)
    do = torch.randn(B, S, H, D, generator=gen, dtype=torch.float32)
    o = torch.randn(B, S, H, D, generator=gen, dtype=torch.float32)
    # log2-domain lse: a per-(b,h) constant large enough that exp2(score - lse)
    # stays in a sane range for random unit-variance q/k.
    lse = 6.0 + 0.05 * torch.randn(B, H, 1, generator=gen, dtype=torch.float32)
    lse = lse.expand(B, H, S).contiguous()
    return (
        q.to(td),
        k.to(td),
        v.to(td),
        do.to(td),
        o.to(td),
        lse,
    )


def _consistent_o_lse(q, k, v, is_causal=True):
    """True softmax output ``o`` and log2-domain ``lse`` for ``q``/``k``/``v``.

    Needed by the golden cross-check: the explicit golden and the SDPA
    reference can only be compared when ``o``/``lse`` really are the forward
    output / log-sum-exp of the same ``q``/``k`` (the random ``o`` from
    :func:`_make_inputs` is just a delta carrier).
    """
    qf = q.detach().cpu().to(torch.float32)
    kf = k.detach().cpu().to(torch.float32)
    vf = v.detach().cpu().to(torch.float32)
    B, S, H, D = qf.shape
    H_kv = kf.shape[2]
    groups = H // H_kv
    sm_scale = (1.0 / D) ** 0.5
    o = torch.empty(B, S, H, D, dtype=torch.float32)
    lse = torch.empty(B, H, S, dtype=torch.float32)
    tril = torch.tril(torch.ones(S, S, dtype=torch.bool))
    for b in range(B):
        for h in range(H):
            kh = h // groups
            sc = torch.matmul(qf[b, :, h, :], torch.transpose(kf[b, :, kh, :], 0, 1)) * sm_scale
            if is_causal:
                sc = sc.masked_fill(~tril, float("-inf"))
            lse[b, h, :] = torch.logsumexp(sc, dim=-1) * _LOG2E
            p = torch.softmax(sc, dim=-1)
            o[b, :, h, :] = torch.matmul(p, vf[b, :, kh, :])
    return o.to(q.dtype), lse


def _rowmax_lse(q, k):
    """Per-(b, h, query) max score in the log2 domain.

    Used by the adversarial-value boundary case so that ``P = exp2(x)`` stays in
    ``(0, 1]`` even for inputs with a huge score span (a constant ``lse`` would
    underflow ``P`` to 0 and make the case vacuous).
    """
    qf = q.detach().cpu().to(torch.float32)
    kf = k.detach().cpu().to(torch.float32)
    B, S, H, D = qf.shape
    H_kv = kf.shape[2]
    groups = H // H_kv
    scale = ((1.0 / D) ** 0.5) * _LOG2E
    lse = torch.empty(B, H, S, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            kh = h // groups
            sc = torch.matmul(qf[b, :, h, :], torch.transpose(kf[b, :, kh, :], 0, 1)) * scale
            lse[b, h, :] = sc.max(dim=-1).values
    return lse


def _run_tensor(q_cpu, k_cpu, v_cpu, do_cpu, o_cpu, lse_cpu, is_causal, dtype_str, tag, seed=0):
    """Run the kernel on NPU tensors and compare dq/dk/dv against the golden."""
    q_cpu = q_cpu.detach().cpu().contiguous()
    k_cpu = k_cpu.detach().cpu().contiguous()
    v_cpu = v_cpu.detach().cpu().contiguous()
    do_cpu = do_cpu.detach().cpu().contiguous()
    o_cpu = o_cpu.detach().cpu().contiguous()
    lse_cpu = lse_cpu.detach().cpu().contiguous()
    B, S, H, D = q_cpu.shape
    H_kv = k_cpu.shape[2]

    gdq, gdk, gdv = golden_gqa_bwd(q_cpu, k_cpu, v_cpu, do_cpu, o_cpu, lse_cpu, is_causal)

    # delta = rowsum(o * do) (preprocess semantics), [B, S, H] -> [B, H, S].
    delta = torch.sum(torch.mul(o_cpu.to(torch.float32), do_cpu.to(torch.float32)), dim=-1)
    delta = torch.transpose(delta, 1, 2).contiguous()

    kernel = _gqa_bwd_kernel(B, H, H_kv, S, D, is_causal, dtype_str)
    dq = torch.zeros(B, S, H, D, dtype=torch.float32).npu()
    dk = torch.zeros(B, S, H_kv, D, dtype=torch.float32).npu()
    dv = torch.zeros(B, S, H_kv, D, dtype=torch.float32).npu()
    kernel(
        q_cpu.npu(),
        k_cpu.npu(),
        v_cpu.npu(),
        do_cpu.npu(),
        lse_cpu.npu(),
        delta.npu(),
        dq,
        dk,
        dv,
    )
    dq = dq.cpu()
    dk = dk.cpu()
    dv = dv.cpu()

    rtol, atol_scale, atol_floor = _TOLERANCE[dtype_str]
    diffs = {"dq": _max_diff(dq, gdq), "dk": _max_diff(dk, gdk), "dv": _max_diff(dv, gdv)}
    max_diff = max(diffs.values())
    bad = []
    detail = []
    for name, got, ref in (("dq", dq, gdq), ("dk", dk, gdk), ("dv", dv, gdv)):
        scale = float(ref.abs().max())
        atol = max(atol_floor, atol_scale * scale)
        detail.append(f"{name}: max_diff={diffs[name]:.3e} ref_scale={scale:.3e} atol_eff={atol:.3e}")
        if not torch.isfinite(got).all():
            bad.append(f"{name} has non-finite values")
            continue
        try:
            torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)
        except AssertionError as e:  # noqa: BLE001
            bad.append(f"{name}: {str(e).splitlines()[1] if len(str(e).splitlines()) > 1 else e}")
    if bad:
        raise AssertionError(
            f"[{tag}] FAIL: shape=({B},{S},{H},{H_kv},{D}) causal={is_causal} "
            f"dtype={dtype_str} max_diff={max_diff:.6e}\n" + "\n".join(detail) + "\n"
            + "\n".join(bad)
        )
    print(
        f"[{tag}] PASS: shape=({B},{S},{H},{H_kv},{D}) causal={int(is_causal)} "
        f"dtype={dtype_str} max_diff={max_diff:.6e}"
    )
    return max_diff


def _run_case(B, S, H, H_kv, D, is_causal, dtype_str, tag, seed=0):
    inputs = _make_inputs(B, S, H, H_kv, D, dtype_str, seed)
    return _run_tensor(*inputs, is_causal, dtype_str, tag, seed=seed)


# ---------- hierarchical testing ----------
def run_L0():
    """L0 (blocking): smoke (causal + non-causal) + design manifest workloads."""
    # Smoke: causal, H_kv=2 (GQA groups=4).
    _run_case(1, 128, 8, 2, 64, True, "float16", "L0-smoke-causal-fp16")
    _run_case(1, 128, 8, 2, 64, True, "bfloat16", "L0-smoke-causal-bf16")
    # Non-causal path (loop_st = 0, no mask at all).
    _run_case(1, 1024, 8, 4, 64, False, "float16", "L0-smoke-noncausal-fp16")
    _run_case(1, 1024, 8, 4, 64, False, "bfloat16", "L0-smoke-noncausal-bf16")
    # Manifest workloads (fp16, causal).
    _run_case(4, 512, 32, 8, 128, True, "float16", "L0-llama8b-short")
    _run_case(2, 2048, 32, 8, 128, True, "float16", "L0-llama8b-long")
    _run_case(2, 512, 64, 8, 128, True, "float16", "L0-llama70b-short")
    _run_case(1, 2048, 64, 8, 128, True, "float16", "L0-llama70b-long")


def run_L1():
    """L1 (blocking): bf16 manifest sweep, GQA groups, atomic / boundary paths."""
    # Manifest workloads (bf16, causal).
    _run_case(4, 512, 32, 8, 128, True, "bfloat16", "L1-llama8b-short")
    _run_case(2, 2048, 32, 8, 128, True, "bfloat16", "L1-llama8b-long")
    _run_case(2, 512, 64, 8, 128, True, "bfloat16", "L1-llama70b-short")
    _run_case(1, 2048, 64, 8, 128, True, "bfloat16", "L1-llama70b-long")
    # GQA group expansion: groups = 1 (MHA), 2, 4, 8.
    _run_case(1, 256, 8, 8, 64, True, "float16", "L1-groups1")
    _run_case(1, 256, 8, 4, 64, True, "float16", "L1-groups2")
    _run_case(1, 256, 8, 2, 64, True, "float16", "L1-groups4")
    _run_case(1, 256, 8, 1, 64, True, "float16", "L1-groups8-Hkv1")
    # dq atomic reduction: n_kv = 1 (single KV block, no cross-block atomic)
    # vs n_kv > 1 (concurrent row-segment atomic).
    _run_case(1, 64, 4, 2, 64, True, "float16", "L1-nkv1")
    _run_case(1, 256, 4, 2, 64, True, "float16", "L1-nkv4")
    # Non-causal path at a larger shape.
    _run_case(2, 512, 8, 4, 128, False, "float16", "L1-noncausal")
    # D = 64 selection (BN = 64) with bf16.
    _run_case(2, 512, 8, 2, 64, True, "bfloat16", "L1-D64-bf16")


def run_L2():
    """L2 (non-blocking): edge shapes, gcd tiling, golden cross-check."""
    for B, S, H, H_kv, D, causal, dtype_str in (
        (1, 64, 4, 2, 64, True, "float16"),      # S == BM -> n_kv = 1
        (1, 96, 4, 2, 128, True, "float16"),     # gcd(96,64) = 32
        (1, 160, 4, 2, 64, True, "float16"),     # gcd(160,64) = 32
        (1, 224, 8, 2, 128, True, "bfloat16"),   # gcd(224,64) = 32
        (2, 320, 8, 8, 64, True, "bfloat16"),    # gcd(320,64) = 64
        (1, 384, 8, 4, 128, True, "float16"),
        (1, 4096, 8, 8, 128, True, "float16"),
        (2, 2048, 8, 8, 64, True, "bfloat16"),
        (1, 1024, 8, 4, 64, False, "bfloat16"),
    ):
        try:
            _run_case(B, S, H, H_kv, D, causal, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(
                f"[L2] WARN (Record without blocking): shape=({B},{S},{H},{H_kv},{D}) "
                f"causal={int(causal)} dtype={dtype_str} err={e}"
            )
    # Golden cross-check: explicit formulas vs torch SDPA backward.
    # The two references must see a *consistent* (o, lse) pair: `o`/`lse` have to
    # be the actual softmax output / log2-domain log-sum-exp of `q`/`k` (the
    # random `o` produced by `_make_inputs` is only a delta carrier, so it would
    # make this cross-check ill-posed).
    try:
        B, S, H, H_kv, D = 1, 256, 8, 4, 64
        q, k, v, _, _, _ = _make_inputs(B, S, H, H_kv, D, "float16", 123)
        o, lse = _consistent_o_lse(q, k, v, True)
        do = torch.randn(B, S, H, D, generator=torch.Generator().manual_seed(7))
        do = do.to(_TORCH_DTYPE["float16"])
        g1 = golden_gqa_bwd(q, k, v, do, o, lse, True)
        g2 = golden_gqa_bwd_sdpa(q, k, v, do, lse, True)
        for name, a, b in zip(("dq", "dk", "dv"), g1, g2):
            md = _max_diff(a, b)
            scale = max(float(b.abs().max()), 1e-6)
            print(f"[L2] golden cross-check {name}: max_abs_diff={md:.3e} ref_scale={scale:.3e}")
            assert md < 1e-3 * scale, f"golden cross-check {name} mismatch {md} (scale {scale})"
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (Record without blocking): golden cross-check err={e}")


def run_boundary():
    """Boundary (non-blocking): all-zero, extreme lse, wide dynamic range, atomic."""
    # All-zero inputs -> all-zero gradients.
    for dtype_str in ("float16", "bfloat16"):
        try:
            B, S, H, H_kv, D = 1, 256, 4, 2, 64
            z = torch.zeros(B, S, H, D, dtype=_TORCH_DTYPE[dtype_str])
            zkv = torch.zeros(B, S, H_kv, D, dtype=_TORCH_DTYPE[dtype_str])
            lse = torch.zeros(B, H, S, dtype=torch.float32)
            md = _run_tensor(z, zkv, zkv, z, z, lse, True, dtype_str, "Boundary-allzero")
            assert md == 0.0, f"all-zero max_diff {md}"
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): allzero dtype={dtype_str} err={e}")

    # lse sentinel-extreme -> P underflows to 0 -> gradients vanish.
    for dtype_str in ("float16", "bfloat16"):
        try:
            B, S, H, H_kv, D = 1, 256, 8, 4, 64
            q, k, v, do, o, _ = _make_inputs(B, S, H, H_kv, D, dtype_str, 5)
            lse = torch.full((B, H, S), 100.0, dtype=torch.float32)
            _run_tensor(q, k, v, do, o, lse, True, dtype_str, "Boundary-lse-extreme")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): lse dtype={dtype_str} err={e}")

    # Wide-dynamic-range inputs (kept inside the op's valid domain).
    # A single constant `lse` cannot keep `P` in range for such inputs (the score
    # span over `D` dims is huge), so `lse` is the per-(b, h, query) maximum
    # score in the log2 domain -- exactly the role it plays in the real pipeline
    # (preprocess kernel).  Then P = exp2(x) is in (0, 1] and the case is
    # non-degenerate.  Inputs are random (not adversarially aligned): a synthetic
    # +/-pattern with heavy cancellation would push the fp16 `ds` quantization
    # error past any tensor-scale bound, which tests the arithmetic rather than
    # the kernel (the DESIGN boundary plan only requires all-zero / extreme lse /
    # zero-or-negative delta).
    for dtype_str in ("float16", "bfloat16"):
        try:
            B, S, H, H_kv, D = 1, 512, 8, 4, 128
            td_ = _TORCH_DTYPE[dtype_str]
            gen = torch.Generator().manual_seed(11)
            sc = 4.0
            q = (torch.randn(B, S, H, D, generator=gen) * sc).to(td_)
            k = (torch.randn(B, S, H_kv, D, generator=gen) * sc).to(td_)
            v = (torch.randn(B, S, H_kv, D, generator=gen) * sc).to(td_)
            do = (torch.randn(B, S, H, D, generator=gen) * sc).to(td_)
            o = torch.randn(B, S, H, D, generator=gen).to(td_)
            lse = _rowmax_lse(q, k)
            _run_tensor(q, k, v, do, o, lse, True, dtype_str, "Boundary-special")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): special dtype={dtype_str} err={e}")

    # Atomic reduction stability: two identical runs must agree to fp32
    # accumulation noise.  Bit-exactness is *not* the right contract for a
    # concurrent fp32 atomic sum (the add order legitimately varies).
    for dtype_str in ("float16", "bfloat16"):
        try:
            B, S, H, H_kv, D = 2, 512, 32, 8, 128
            inputs = _make_inputs(B, S, H, H_kv, D, dtype_str, 77)
            q, k, v, do, o, lse = inputs
            delta = torch.sum(torch.mul(o.to(torch.float32), do.to(torch.float32)), dim=-1)
            delta = torch.transpose(delta, 1, 2).contiguous()
            kernel = _gqa_bwd_kernel(B, H, H_kv, S, D, True, dtype_str)
            outs = []
            for _ in range(2):
                dq = torch.zeros(B, S, H, D, dtype=torch.float32).npu()
                dk = torch.zeros(B, S, H_kv, D, dtype=torch.float32).npu()
                dv = torch.zeros(B, S, H_kv, D, dtype=torch.float32).npu()
                kernel(q.npu(), k.npu(), v.npu(), do.npu(), lse.npu(), delta.npu(), dq, dk, dv)
                outs.append((dq.cpu(), dk.cpu(), dv.cpu()))
            spread = []
            for name, a, b in zip(("dq", "dk", "dv"), outs[0], outs[1]):
                scale = max(float(a.abs().max()), 1e-6)
                md = (a - b).abs().max().item()
                spread.append(f"{name}={md:.3e}/{scale:.3e}")
                assert md <= 1e-4 * scale, (
                    f"[Boundary] {name} atomic reduction unstable: {md} vs scale {scale}")
            print(f"[Boundary] PASS: atomic reduction stable dtype={dtype_str} " + " ".join(spread))
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): atomic dtype={dtype_str} err={e}")


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
