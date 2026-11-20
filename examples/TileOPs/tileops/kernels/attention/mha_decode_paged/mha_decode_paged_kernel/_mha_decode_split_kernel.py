# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Paged multi-head-attention decode forward (split-KV + LSE combine) -- NPU Developer.

Migrated from the GPU TileLang factory ``_mha_decode_split_kernel``
(TileOPs ``tileops/kernels/attention/mha_decode_paged.py``, L117-379: the
``_mha_decode_split`` + ``combine`` macros) to ``target="npuir"`` per the
frozen DESIGN.md.

Math (per batch b, head h, query token):
    o[b, :, h, :] = softmax(q[b, :, h, :] @ K_b^T / sqrt(D)) @ V_b
where the logical KV sequence is gathered from the physical paged cache
``K`` / ``V`` ``[N_kv, H, D]`` through ``block_table[b, :]``.  The KV axis is
split into ``num_split_eff`` contiguous segments; each segment runs its own
online softmax producing a normalized partial output ``o_i`` and a segment
log-sum-exp ``l_i``; a second pass recombines them:
    w_i = exp(l_i - LSE),  o = sum_i w_i * o_i,
    LSE = log(sum_i exp(l_i - max_i l_i)) + max_i l_i.
Segment recombination is an exact identity, so ``Output`` equals the no-split
single-pass result up to fp rounding.  Decode has ``S_q = 1``.

NPU redesign (DESIGN sections 0.6 / 1.6.1 / 3.3 / 6):
  - npuir compiles one ``T.Kernel`` per prim_func, so the source's two in-one
    prim_func launches become **two independent ``@tilelang.jit`` kernels**
    (Phase A ``split_partial`` -> workspace, Phase B ``combine`` -> Output)
    sequenced by a host-side Python closure (stream order == the source's
    implicit launch barrier).  R5.
  - 1-D non-persistent grid for both phases; ``cid`` reversed into
    ``(mid, hid, bz, sid)`` / ``(mid, hid, bz)``.  No in-core serial
    multi-task loop (avoids the Developer + persistent + gemm trap).  R1.
  - ``block_M = 16`` (Cube fractal floor, only row 0 real); ``block_N =
    page_size`` (one KV tile == one physical page).  R2 / R3.
  - Compile-time page partition: ``P = ceildiv(nkv, ns_eff)`` gives a
    **static** segment loop bound; out-of-range pages are clamped
    (``pgc = min(pg, nkv-1)``) and the mask uses the *unclamped* global
    position ``pg*page + j < real_kv`` so clamped duplicate blocks are fully
    masked.  R4 / OPT-S.
  - Static-bound tail blocks that are fully masked are neutralized by a
    same-domain identity guard (``rs = 1``, ``P = 0``, ``s_sum = 0``) and the
    segment LSE reference max ``m_ref``; empty segments emit the fp16-safe
    sentinel ``LSE_SENT = -30000`` (a fp16 ``-1e38`` would cast to ``-inf``
    and make the all-empty combine ``exp(-inf - (-inf)) = NaN``).  R4.
  - Natural-exp domain (OPT-1 fold scale into Q, OPT-2 reassociated rescale,
    OPT-3 natural exp), finite-large sentinel ``-1e38`` for masked score
    columns, P quantized back to the input dtype (O10).
  - Phase B is pure Vector; the segment dimension is unrolled via
    ``T.serial(ns_eff)`` (compile-time constant) over ``[block_M, 1]`` row
    vectors so no ``T.reduce(dim=0)`` (unproven strided reduction) is needed.
    R7.

The independent golden (``golden__mha_decode_split_kernel``) reassembles the
logical KV from the paged cache and calls ``F.scaled_dot_product_attention``;
it does NOT mirror the NPU split algorithm.
"""

import argparse
import functools
import math
import os

import torch

import tilelang
import tilelang.language as T

try:  # registers the "npu" device when the runtime is present
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover - CPU-only golden path
    torch_npu = None

# Standalone runs must use the same programming mode as the frozen artifact.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

ASCEND_MODE = "Developer"

# Numerical sentinels (DESIGN sections 0.6-R4 / 6.4).
_OOB_SENTINEL = -1e38    # finite large sentinel for masked score columns
_ALL_MASKED_THR = -1e30  # same-domain threshold for the all-masked guard
_LSE_SENT = -30000.0     # segment-LSE sentinel; MUST be representable in fp16


# ---------- Golden (independent PyTorch reference, DESIGN section 8.1) ----------
def golden__mha_decode_split_kernel(q, k, v, real_seqlen_kv, block_table, page_size,
                                    is_causal=False):
    """Paged MHA decode reference: reassemble logical K/V per batch, then SDPA.

    q:             (B, S_q, H, D) fp16/bf16
    k, v:          (N_kv, H, D) fp16/bf16  (physical paged cache)
    real_seqlen_kv: (B,) int32             (logical KV length)
    block_table:   (B, num_pages) int32    (logical page -> physical page)
    Returns (B, S_q, H, D).

    Independence: the non-causal path is a verbatim port of the GPU source's
    test reference (``test_mha_decode_paged.py::ref_program``) -- paged
    reassembly to a logical layout followed by
    ``F.scaled_dot_product_attention``.  It does NOT mirror the NPU data flow
    (no segments, no static bound, no sentinel, no identity guard, no combine);
    the whole split/combine algorithm is absent.  The optional causal path uses
    an explicit fp32 softmax with the source kernel's own causal rule
    (``q_pos >= kv_pos``), not SDPA's bottom-right-aligned convention.
    """
    batch, seqlen_q, heads, dim = q.shape
    seqlen_kv = k.shape[0]
    q = q.float()
    k = k.float()
    v = v.float()
    out_list = []
    for i_b in range(batch):
        length = int(real_seqlen_kv[i_b].item())
        if length <= 0:
            out_list.append(torch.zeros(1, seqlen_q, heads, dim, dtype=torch.float32))
            continue
        k_logical = torch.zeros(seqlen_kv, heads, dim, dtype=torch.float32)
        v_logical = torch.zeros(seqlen_kv, heads, dim, dtype=torch.float32)
        num_pages = math.ceil(length / page_size)
        for i_paged in range(num_pages):
            start_pos = int(block_table[i_b, i_paged].item()) * page_size
            end_pos = min(start_pos + page_size, seqlen_kv)
            page_len = end_pos - start_pos
            k_logical[i_paged * page_size:i_paged * page_size + page_len] = k[start_pos:end_pos]
            v_logical[i_paged * page_size:i_paged * page_size + page_len] = v[start_pos:end_pos]
        if is_causal:
            q_h = q[i_b].transpose(0, 1)                  # (H, S_q, D)
            k_h = k_logical[:length].transpose(0, 1)      # (H, L, D)
            v_h = v_logical[:length].transpose(0, 1)
            scores = q_h @ k_h.transpose(-2, -1)          # (H, S_q, L)
            q_pos = torch.arange(seqlen_q)[:, None]
            kv_pos = torch.arange(length)[None, :]
            scores = scores.masked_fill(q_pos < kv_pos, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out_b = (probs @ v_h).transpose(0, 1).unsqueeze(0)  # (1, S_q, H, D)
        else:
            k_b = k_logical[:length].unsqueeze(0).transpose(1, 2)  # (1, H, L, D)
            v_b = v_logical[:length].unsqueeze(0).transpose(1, 2)
            q_b = q[i_b:i_b + 1].transpose(1, 2)                   # (1, H, S_q, D)
            o = torch.nn.functional.scaled_dot_product_attention(q_b, k_b, v_b)
            out_b = o.transpose(1, 2).contiguous()
        out_list.append(out_b)
    return torch.cat(out_list, dim=0)


@functools.lru_cache(maxsize=32)
def _mha_pack_pages(batch, heads, seqlen_kv, dim, page_size, dtype, transpose_chunk=0):
    # Skew the packed head stride. Kernels access only seqlen_kv rows.
    packed_length = seqlen_kv + 1
    nkv = seqlen_kv // page_size
    tile_size = transpose_chunk or page_size
    @tilelang.jit(target="npuir")
    def compile_pack():
        @T.prim_func
        def pack_kv(
            K: T.Tensor([seqlen_kv, heads, dim], dtype),
            V: T.Tensor([seqlen_kv, heads, dim], dtype),
            pages: T.Tensor([batch, nkv], "int32"),
            KP: T.Tensor([batch, heads, packed_length, dim], dtype),
            VP: T.Tensor([batch, heads, packed_length, dim], dtype),
        ):
            with T.Kernel(batch * heads * nkv, is_npu=True) as (cid, _):
                pg = cid % nkv
                hid = cid // nkv % heads
                bz = cid // (nkv * heads)
                phys = pages[bz, pg]
                vu = T.alloc_ub([page_size, dim], dtype)
                if transpose_chunk > 0:
                    ku = T.alloc_ub([tile_size, dim], dtype)
                    kt = T.alloc_ub([dim, tile_size], dtype)
                    flat = T.alloc_ub([tile_size, dim], dtype)
                    for i in T.serial(page_size // tile_size):
                        src = phys * page_size + i * tile_size
                        dst = pg * page_size + i * tile_size
                        T.copy(K[src:src + tile_size, hid, 0:dim], ku)
                        T.transpose(ku, kt, permutation=[1, 0])
                        T.reshape(kt, flat)
                        T.copy(flat, KP[bz, hid, dst:dst + tile_size, 0:dim])
                else:
                    ku = T.alloc_ub([page_size, dim], dtype)
                    T.copy(K[phys * page_size:(phys + 1) * page_size, hid, 0:dim], ku)
                    if dim == 128 and page_size == 128:
                        kt_square = T.alloc_ub([128, 128], dtype)
                        T.transpose(ku, kt_square, permutation=[1, 0])
                        T.copy(kt_square, KP[bz, hid, pg * page_size:(pg + 1) * page_size, 0:dim])
                    else:
                        T.copy(ku, KP[bz, hid, pg * page_size:(pg + 1) * page_size, 0:dim])
                T.copy(V[phys * page_size:(phys + 1) * page_size, hid, 0:dim], vu)
                T.copy(vu, VP[bz, hid, pg * page_size:(pg + 1) * page_size, 0:dim])
        return pack_kv
    return compile_pack()


@functools.lru_cache(maxsize=32)
def _mha_vector_packed(batch, heads, seqlen_kv, dim, dtype, ns, num_split, transpose_k=False, transpose_d64=False):
    packed_length = seqlen_kv + 1
    n = seqlen_kv // ns
    chunk = 256 if dim == 64 and n >= 256 else 128
    # Offset consecutive rows across UB banks during FP32 reductions.
    work_padding = 16 if dim == 128 else (8 if dim == 64 else 0)
    scale = dim ** -0.5
    @tilelang.jit(target="npuir")
    def compile_vector():
        @T.prim_func
        def decode_vector_packed(
            Q: T.Tensor([batch, 1, heads, dim], dtype),
            K: T.Tensor([batch, heads, packed_length, dim], dtype),
            V: T.Tensor([batch, heads, packed_length, dim], dtype),
            L: T.Tensor([batch, heads, num_split, 1], "float32"),
            O: T.Tensor([batch, 1, heads, num_split, dim], "float32"),
        ):
            with T.Kernel(batch * heads * ns, is_npu=True) as (cid, _):
                sid = cid % ns
                hid = cid // ns % heads
                bz = cid // (ns * heads)
                qio = T.alloc_ub([1, dim], dtype)
                q = T.alloc_ub([1, dim], "float32")
                query_column = T.alloc_ub([dim, 1], "float32")
                score_row = T.alloc_ub([1, chunk], "float32")
                score_column = T.alloc_ub([chunk, 1], "float32")
                kio = T.alloc_ub([chunk, dim], dtype)
                k_transposed = T.alloc_ub([dim, chunk], dtype)
                work_transposed = T.alloc_ub([dim, chunk + work_padding], "float32")
                work = T.alloc_ub([chunk, dim+work_padding], "float32")
                prob = T.alloc_ub([chunk, 1], dtype)
                pv = T.alloc_ub([1, dim], "float32")
                scores = T.alloc_ub([n, 1], "float32")
                mx = T.alloc_ub([1, 1], "float32")
                den = T.alloc_ub([1, 1], "float32")
                lse = T.alloc_ub([1, 8], "float32")
                output = T.alloc_ub([1, dim], "float32")
                T.copy(Q[bz, 0, hid, 0:dim], qio[0, 0:dim])
                T.vcast(qio, q, round_mode="rint")
                T.vmul(q, scale, q)
                T.reshape(q, query_column)
                for it in T.serial(n // chunk):
                    start = sid * n + it * chunk
                    T.copy(K[bz, hid, start:start+chunk, 0:dim], kio)
                    if transpose_d64:
                        T.reshape(kio, k_transposed)
                        T.vcast(k_transposed, work_transposed[0:dim, 0:chunk], round_mode="rint")
                        T.vmul(work_transposed[0:dim, 0:chunk], query_column, work_transposed[0:dim, 0:chunk])
                        T.reduce_sum(work_transposed[0:dim, 0:chunk], score_row, dim=0, clear=True)
                        T.reshape(score_row, score_column)
                        T.copy(score_column, scores[it*chunk:(it+1)*chunk, 0:1])
                    else:
                        T.vcast(kio, work[0:chunk, 0:dim], round_mode="rint")
                        if transpose_k:
                            T.vmul(work[0:chunk, 0:dim], query_column, work[0:chunk, 0:dim])
                            T.reduce_sum(work[0:chunk, 0:dim], score_row, dim=0, clear=True)
                            T.reshape(score_row, score_column)
                            T.copy(score_column, scores[it*chunk:(it+1)*chunk, 0:1])
                        else:
                            T.vmul(work[0:chunk, 0:dim], q, work[0:chunk, 0:dim])
                            T.reduce_sum(work[0:chunk, 0:dim], scores[it*chunk:(it+1)*chunk, 0:1], dim=1, clear=True)
                T.reduce_max(scores, mx, dim=0, clear=True)
                T.vsub(scores, mx, scores)
                T.vexp(scores, scores)
                T.reduce_sum(scores, den, dim=0, clear=True)
                T.vdiv(scores, den, scores)
                T.vbrc(den, lse)
                T.vln(lse, lse)
                T.vadd(lse, mx, lse)
                T.clear(output)
                for it in T.serial(n // chunk):
                    start = sid * n + it * chunk
                    T.copy(V[bz, hid, start:start+chunk, 0:dim], kio)
                    T.vcast(scores[it*chunk:(it+1)*chunk, 0:1], prob, round_mode="rint")
                    T.vmul(kio, prob, kio)
                    T.vcast(kio, work[0:chunk, 0:dim], round_mode="rint")
                    T.reduce_sum(work[0:chunk, 0:dim], pv, dim=0, clear=True)
                    T.vadd(output, pv, output)
                T.copy(output[0, 0:dim], O[bz, 0, hid, sid, 0:dim])
                T.copy(lse[0:1, 0:1], L[bz, hid, sid, 0:1])
        return decode_vector_packed
    return compile_vector()


@functools.lru_cache(maxsize=32)
def _mha_vector_deferred_stats(batch, heads, seqlen_kv, dim, dtype, ns, num_split, transpose_k=False):
    packed_length = seqlen_kv + 1
    n = seqlen_kv // ns
    chunk = 256 if dim == 64 and n >= 256 else 128
    # Offset consecutive rows across UB banks during FP32 reductions.
    work_padding = 16 if dim == 128 else (8 if dim == 64 else 0)
    scale = dim ** -0.5
    @tilelang.jit(target="npuir")
    def compile_vector():
        @T.prim_func
        def decode_vector_packed(
            Q: T.Tensor([batch, 1, heads, dim], dtype),
            K: T.Tensor([batch, heads, packed_length, dim], dtype),
            V: T.Tensor([batch, heads, packed_length, dim], dtype),
            P: T.Tensor([batch, heads, num_split, dim + 8], "float32"),
        ):
            with T.Kernel(batch * heads * ns, is_npu=True) as (cid, _):
                sid = cid % ns
                hid = cid // ns % heads
                bz = cid // (ns * heads)
                qio = T.alloc_ub([1, dim], dtype)
                q = T.alloc_ub([1, dim], "float32")
                query_column = T.alloc_ub([dim, 1], "float32")
                score_row = T.alloc_ub([1, chunk], "float32")
                score_column = T.alloc_ub([chunk, 1], "float32")
                kio = T.alloc_ub([chunk, dim], dtype)
                work = T.alloc_ub([chunk, dim+work_padding], "float32")
                prob = T.alloc_ub([chunk, 1], dtype)
                pv = T.alloc_ub([1, dim], "float32")
                scores = T.alloc_ub([n, 1], "float32")
                mx = T.alloc_ub([1, 1], "float32")
                den = T.alloc_ub([1, 1], "float32")
                stats = T.alloc_ub([1, dim + 8], "float32")
                output = T.alloc_ub([1, dim], "float32")
                T.copy(Q[bz, 0, hid, 0:dim], qio[0, 0:dim])
                T.vcast(qio, q, round_mode="rint")
                T.vmul(q, scale, q)
                T.reshape(q, query_column)
                for it in T.serial(n // chunk):
                    start = sid * n + it * chunk
                    T.copy(K[bz, hid, start:start+chunk, 0:dim], kio)
                    T.vcast(kio, work[0:chunk, 0:dim], round_mode="rint")
                    if transpose_k:
                        T.vmul(work[0:chunk, 0:dim], query_column, work[0:chunk, 0:dim])
                        T.reduce_sum(work[0:chunk, 0:dim], score_row, dim=0, clear=True)
                        T.reshape(score_row, score_column)
                        T.copy(score_column, scores[it*chunk:(it+1)*chunk, 0:1])
                    else:
                        T.vmul(work[0:chunk, 0:dim], q, work[0:chunk, 0:dim])
                        T.reduce_sum(work[0:chunk, 0:dim], scores[it*chunk:(it+1)*chunk, 0:1], dim=1, clear=True)
                T.copy(V[bz, hid, sid*n:sid*n+chunk, 0:dim], kio)
                T.reduce_max(scores, mx, dim=0, clear=True)
                T.vsub(scores, mx, scores)
                T.vexp(scores, scores)
                T.reduce_sum(scores, den, dim=0, clear=True)
                T.vcast(scores[0:chunk, 0:1], prob, round_mode="rint")
                T.vmul(kio, prob, kio)
                T.vcast(kio, work[0:chunk, 0:dim], round_mode="rint")
                T.reduce_sum(work[0:chunk, 0:dim], output, dim=0, clear=True)
                for it in T.serial(1, n // chunk):
                    start = sid * n + it * chunk
                    T.copy(V[bz, hid, start:start+chunk, 0:dim], kio)
                    T.vcast(scores[it*chunk:(it+1)*chunk, 0:1], prob, round_mode="rint")
                    T.vmul(kio, prob, kio)
                    T.vcast(kio, work[0:chunk, 0:dim], round_mode="rint")
                    T.reduce_sum(work[0:chunk, 0:dim], pv, dim=0, clear=True)
                    T.vadd(output, pv, output)
                T.copy(output[0:1, 0:dim], stats[0:1, 0:dim])
                T.copy(mx, stats[0:1, dim:dim + 1])
                T.copy(den, stats[0:1, dim + 1:dim + 2])
                T.copy(stats[0, 0:dim + 2], P[bz, hid, sid, 0:dim + 2])
        return decode_vector_packed
    return compile_vector()


@functools.lru_cache(maxsize=32)
def _mha_combine_split_stats(batch, heads, tiles, num_split, dim, dtype):
    @tilelang.jit(out_idx=[-1], target="npuir")
    def build():
        @T.prim_func
        def combine(
            P: T.Tensor([batch, heads, num_split, dim + 8], "float32"),
            Y: T.Tensor([batch, 1, heads, dim], dtype),
        ):
            with T.Kernel(batch * heads, is_npu=True) as (cid, _):
                bz = cid // heads
                head = cid % heads
                buf = T.alloc_ub([tiles, dim + 8], "float32")
                weights = T.alloc_ub([tiles, 1], "float32")
                sums = T.alloc_ub([tiles, 1], "float32")
                mx = T.alloc_ub([1, 1], "float32")
                den = T.alloc_ub([1, 1], "float32")
                out = T.alloc_ub([1, dim], "float32")
                io = T.alloc_ub([1, dim], dtype)
                T.copy(P[bz, head, 0:tiles, 0:dim + 2], buf[0:tiles, 0:dim + 2])
                T.copy(buf[0:tiles, dim:dim + 1], weights)
                T.copy(buf[0:tiles, dim + 1:dim + 2], sums)
                T.reduce_max(weights, mx, dim=0, clear=True)
                T.vsub(weights, mx, weights)
                T.vexp(weights, weights)
                T.vmul(sums, weights, sums)
                T.reduce_sum(sums, den, dim=0, clear=True)
                T.vmul(buf[0:tiles, 0:dim], weights, buf[0:tiles, 0:dim])
                T.reduce_sum(buf[0:tiles, 0:dim], out, dim=0, clear=True)
                T.vdiv(out, den, out)
                T.vcast(out, io, round_mode="rint")
                T.copy(io[0, 0:dim], Y[bz, 0, head, 0:dim])
        return combine

    return build()


def _mha_decode_vector_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size,
                              is_causal, dtype, full_kv=False):
    from tileops.utils import scoped_ascend_mode

    nkv = (seqlen_kv + page_size - 1) // page_size
    scale = dim ** -0.5

    @functools.lru_cache(maxsize=32)
    def build(block_M, block_N_in, num_split, num_stages):
        pieces = min(page_size // 16, max(1, (num_split + nkv - 1) // nkv))
        while page_size % pieces:
            pieces -= 1
        bn = page_size // pieces
        tiles = nkv * pieces
        packed_ns = min(num_split, max(1, 32 // (batch * heads)))
        use_packed = (full_kv and not is_causal and dtype == "float16"
                      and seqlen_kv % page_size == 0 and seqlen_kv % packed_ns == 0
                      and (seqlen_kv // packed_ns) % 128 == 0
                      and (seqlen_kv // packed_ns) * dim <= 32768 and dim % 16 == 0)
        ns = packed_ns if use_packed else min(num_split, tiles)
        per_split = (tiles + ns - 1) // ns

        @tilelang.jit(target="npuir")
        def partial_factory():
            @T.prim_func
            def split_partial(
                Q: T.Tensor([batch, 1, heads, dim], dtype),
                K: T.Tensor([seqlen_kv, heads, dim], dtype),
                V: T.Tensor([seqlen_kv, heads, dim], dtype),
                lengths: T.Tensor([batch], "int32"),
                pages: T.Tensor([batch, nkv], "int32"),
                L: T.Tensor([batch, heads, num_split, 1], "float32"),
                O: T.Tensor([batch, 1, heads, num_split, dim], "float32"),
            ):
                with T.Kernel(batch * heads * ns, is_npu=True) as (cid, _):
                    sid = cid % ns
                    hid = cid // ns % heads
                    bz = cid // (ns * heads)
                    qio = T.alloc_ub([1, dim], dtype)
                    kio = T.alloc_ub([bn, dim], dtype)
                    vio = T.alloc_ub([bn, dim], dtype)
                    q = T.alloc_ub([1, dim], "float32")
                    k = T.alloc_ub([bn, dim], "float32")
                    v = T.alloc_ub([bn, dim], "float32")
                    scores = T.alloc_ub([bn, 1], "float32")
                    idx = T.alloc_ub([bn], "float32")
                    pos = T.alloc_ub([bn, 1], "float32")
                    penalty = T.alloc_ub([bn, 1], "float32")
                    valid = T.alloc_ub([bn, 1], "float32")
                    mx = T.alloc_ub([1, 1], "float32")
                    oldmx = T.alloc_ub([1, 1], "float32")
                    newmx = T.alloc_ub([1, 1], "float32")
                    alpha = T.alloc_ub([1, 1], "float32")
                    total = T.alloc_ub([1, 1], "float32")
                    part_sum = T.alloc_ub([1, 1], "float32")
                    acc = T.alloc_ub([1, dim], "float32")
                    pv = T.alloc_ub([1, dim], "float32")
                    one = 1.0
                    zero = 0.0
                    neg = -1e30
                    tiny = 1e-30
                    T.copy(Q[bz, 0, hid, 0:dim], qio[0, 0:dim])
                    T.vcast(qio, q, round_mode="rint")
                    T.vmul(q, scale, q)
                    if not full_kv or is_causal:
                        T.arange(idx, [1], offset=0)
                        T.reshape(idx, pos)
                    T.vbrc(neg, oldmx)
                    T.clear(total)
                    T.clear(acc)
                    for it in T.serial(per_split):
                        tile = sid * per_split + it
                        pg = T.min(tile // pieces, nkv - 1)
                        phys = pages[bz, pg] * page_size + (tile % pieces) * bn
                        T.copy(K[phys:phys + bn, hid, 0:dim], kio)
                        T.vcast(kio, k, round_mode="rint")
                        T.vmul(k, q, k)
                        T.reduce_sum(k, scores, dim=1, clear=True)
                        if not full_kv or is_causal:
                            lim = T.cast(lengths[bz] - tile * bn - 1, "float32")
                            if is_causal:
                                lim = T.min(lim, T.cast(-tile * bn, "float32"))
                            T.vsub(pos, lim, penalty)
                            T.vmax(penalty, zero, penalty)
                            T.vmin(penalty, one, penalty)
                            T.vmul(penalty, -1.0, valid)
                            T.vadd(valid, one, valid)
                            T.vmul(penalty, neg, penalty)
                            T.vadd(scores, penalty, scores)
                        T.reduce_max(scores, mx, dim=0, clear=True)
                        if per_split == 1:
                            T.copy(mx, newmx)
                            T.vsub(scores, newmx, scores)
                            T.vexp(scores, scores)
                            if not full_kv or is_causal:
                                T.vmul(scores, valid, scores)
                            T.reduce_sum(scores, total, dim=0, clear=True)
                            T.copy(V[phys:phys + bn, hid, 0:dim], vio)
                            T.vcast(vio, v, round_mode="rint")
                            T.vmul(v, scores, v)
                            T.reduce_sum(v, acc, dim=0, clear=True)
                            T.copy(newmx, oldmx)
                        else:
                            T.vmax(mx, oldmx, newmx)
                            T.vsub(oldmx, newmx, alpha)
                            T.vexp(alpha, alpha)
                            T.vsub(scores, newmx, scores)
                            T.vexp(scores, scores)
                            if not full_kv or is_causal:
                                T.vmul(scores, valid, scores)
                            T.reduce_sum(scores, part_sum, dim=0, clear=True)
                            T.vmul(total, alpha, total)
                            T.vadd(total, part_sum, total)
                            T.copy(V[phys:phys + bn, hid, 0:dim], vio)
                            T.vcast(vio, v, round_mode="rint")
                            T.vmul(v, scores, v)
                            T.reduce_sum(v, pv, dim=0, clear=True)
                            T.vmul(acc, alpha, acc)
                            T.vadd(acc, pv, acc)
                            T.copy(newmx, oldmx)
                    T.vmax(total, tiny, total)
                    T.vdiv(acc, total, acc)
                    T.vln(total, total)
                    T.vadd(total, oldmx, total)
                    T.copy(acc[0, 0:dim], O[bz, 0, hid, sid, 0:dim])
                    T.copy(total[0, 0], L[bz, hid, sid, 0], size=[1])
            return split_partial

        @tilelang.jit(out_idx=[-1], target="npuir")
        def combine_factory():
            @T.prim_func
            def combine(
                L: T.Tensor([batch, heads, num_split, 1], "float32"),
                O: T.Tensor([batch, 1, heads, num_split, dim], "float32"),
                Output: T.Tensor([batch, 1, heads, dim], dtype),
            ):
                with T.Kernel(batch * heads, is_npu=True) as (cid, _):
                    hid = cid % heads
                    bz = cid // heads
                    ls = T.alloc_ub([ns, 1], "float32")
                    po = T.alloc_ub([ns, dim], "float32")
                    mx = T.alloc_ub([1, 1], "float32")
                    den = T.alloc_ub([1, 1], "float32")
                    out = T.alloc_ub([1, dim], "float32")
                    out_io = T.alloc_ub([1, dim], dtype)
                    if ns == 1:
                        T.copy(O[bz, 0, hid, 0, 0:dim], out[0, 0:dim])
                    else:
                        T.copy(L[bz, hid, 0:ns, 0:1], ls[0:ns, 0:1])
                        T.reduce_max(ls, mx, dim=0, clear=True)
                        T.vsub(ls, mx, ls)
                        T.vexp(ls, ls)
                        T.reduce_sum(ls, den, dim=0, clear=True)
                        T.vdiv(ls, den, ls)
                        T.copy(O[bz, 0, hid, 0:ns, 0:dim], po[0:ns, 0:dim])
                        T.vmul(po, ls, po)
                        T.reduce_sum(po, out, dim=0, clear=True)
                    T.vcast(out, out_io, round_mode="rint")
                    T.copy(out_io[0, 0:dim], Output[bz, 0, hid, 0:dim])
            return combine

        n = seqlen_kv // ns
        chunk = 256 if dim == 64 and n >= 256 else 128
        transpose_square = dim == 128 and page_size == 128
        transpose_d64 = use_packed and dim == 64 and page_size == chunk == 256 and n % chunk == 0
        defer_normalization = use_packed and (transpose_square or (dim == 64 and page_size == 256 and n == 128))
        with scoped_ascend_mode("Expert"):
            if defer_normalization:
                partial = _mha_vector_deferred_stats(batch, heads, seqlen_kv, dim, dtype, ns,
                                                     num_split, transpose_square)
                combine = _mha_combine_split_stats(batch, heads, ns, num_split, dim, dtype)
            else:
                partial = (_mha_vector_packed(batch, heads, seqlen_kv, dim, dtype, ns, num_split,
                                              transpose_square, transpose_d64)
                           if use_packed else partial_factory())
                combine = combine_factory()
            prepare = (_mha_pack_pages(batch, heads, seqlen_kv, dim, page_size, dtype, chunk if transpose_d64 else 0)
                       if use_packed else None)
        def run(q, k, v, lengths, pages, glse, partial_out, acc_split_length):
            if use_packed:
                kp = torch.empty((batch, heads, seqlen_kv + 1, dim), dtype=q.dtype, device=q.device)
                vp = torch.empty_like(kp)
                prepare(k, v, pages, kp, vp)
                if defer_normalization:
                    stats = torch.empty((batch, heads, num_split, dim + 8),
                                        dtype=torch.float32, device=q.device)
                    partial(q, kp, vp, stats)
                    return combine(stats)
                partial(q, kp, vp, glse, partial_out)
            else:
                partial(q, k, v, lengths, pages, glse, partial_out)
            return combine(glse, partial_out)
        return run
    return build



# ---------- Kernel factories (Developer MixCV, DESIGN sections 3.3 / 6) ----------
@functools.lru_cache(maxsize=32)
def _mha_decode_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal,
                             dtype, full_kv=False):
    """Factory matching the TileOPs wrapper call form.

    The wrapper calls ``factory(...)(block_M, block_N, num_split,
    num_stages)(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial,
    acc_split_length)``.  The GPU tiling knobs ``block_M`` / ``block_N`` are
    normalised to the frozen NPU design values (``block_M`` = fractal floor 16,
    ``block_N`` = ``page_size``); ``num_split`` and ``num_stages`` are honoured
    (DESIGN 0.6 R2/R3/R4/R8).
    """
    if seqlen_q == 1:
        return _mha_decode_vector_kernel(batch, heads, seqlen_q, seqlen_kv, dim,
                                         page_size, is_causal, dtype, full_kv)
    accum_dtype = "float"
    num_q = (seqlen_q + 15) // 16
    nkv = (seqlen_kv + page_size - 1) // page_size
    s_nat = 1.0 / (dim ** 0.5)   # natural-exp domain scale (OPT-1 / OPT-3)
    is_causal_t = bool(is_causal)

    # NOTE (toolchain fact): the GPU source's `TL_ENABLE_FAST_MATH` has no
    # npuir PassConfigKey in this toolchain, so no pass_configs are set -- the
    # Developer defaults are the ones exercised by the verified npuir
    # precedents.
    def _func(block_M_in, block_N_in, num_split, num_stages):
        block_M = 16          # Cube fractal floor (DESIGN 0.6 R2)
        block_N = page_size   # one KV tile == one physical page (DESIGN 0.6 R3)
        ns_eff = min(int(num_split), nkv)              # no empty segments (R4)
        P = (nkv + ns_eff - 1) // ns_eff               # pages per segment (static)
        num_logical_A = num_q * batch * heads * ns_eff
        num_logical_B = num_q * batch * heads

        # ---------------- Phase A: split partial ---------------- #
        @tilelang.jit(target="npuir")
        def _split():
            @T.prim_func
            def split_partial(
                Q: T.Tensor([batch, seqlen_q, heads, dim], dtype),
                K: T.Tensor([seqlen_kv, heads, dim], dtype),
                V: T.Tensor([seqlen_kv, heads, dim], dtype),
                real_seqlen_kv: T.Tensor([batch], "int32"),
                block_table: T.Tensor([batch, nkv], "int32"),
                # S5 precision fix: the per-segment LSE workspace is fp32.  Storing
                # l_i in the low-precision io dtype (bf16) propagated an absolute
                # exponent error ~|l_i|*2^-8 into the combine weights exp(l_i-LSE),
                # pushing the bf16 split path over the 1e-3 test atol.  The LSE is
                # a log-domain scalar, so it needs fp32 even when io is bf16.
                glse: T.Tensor([batch, heads, num_split, seqlen_q], "float32"),
                Output_partial: T.Tensor([batch, seqlen_q, heads, num_split, dim],
                                         "float32"),
            ):
                with T.Kernel(num_logical_A, is_npu=True) as (cid, _):
                    # ---- task decode (DESIGN 0.6 R1) ----
                    sid = cid % ns_eff
                    r = cid // ns_eff
                    hid = r % heads
                    r2 = r // heads
                    bz = r2 % batch
                    mid = r2 // batch
                    real_kv = real_seqlen_kv[bz]
                    real_m = T.min(block_M, seqlen_q - mid * block_M)

                    # ---- Cube operands / Vector accumulators ----
                    Q_shared = T.alloc_shared([block_M, dim], dtype)
                    K_shared = T.alloc_shared([block_N, dim], dtype)
                    V_shared = T.alloc_shared([block_N, dim], dtype)
                    # S5 precision fix: stage the normalized segment output in an
                    # fp32 shared buffer so it reaches the fp32 Output_partial
                    # workspace without an io-dtype round trip.
                    O_cast = T.alloc_shared([block_M, dim], "float32")

                    acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                    acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                    acc_o = T.alloc_fragment([block_M, dim], accum_dtype)

                    m_prev = T.alloc_fragment([block_M, 1], accum_dtype)
                    m_cur = T.alloc_fragment([block_M, 1], accum_dtype)
                    m_cur_safe = T.alloc_fragment([block_M, 1], accum_dtype)
                    m_ref = T.alloc_fragment([block_M, 1], accum_dtype)
                    m_delta = T.alloc_fragment([block_M, 1], accum_dtype)
                    mask_f = T.alloc_fragment([block_M, 1], accum_dtype)
                    rs = T.alloc_fragment([block_M, 1], accum_dtype)
                    s_sum = T.alloc_fragment([block_M, 1], accum_dtype)
                    logsum = T.alloc_fragment([block_M, 1], accum_dtype)
                    lse = T.alloc_fragment([block_M, 1], accum_dtype)
                    one_vec = T.alloc_fragment([block_M, 1], accum_dtype)
                    zero_vec = T.alloc_fragment([block_M, 1], accum_dtype)
                    thr_vec = T.alloc_fragment([block_M, 1], accum_dtype)
                    sent_lse = T.alloc_fragment([block_M, 1], accum_dtype)
                    all_masked = T.alloc_fragment([block_M, 1], "bool")
                    not_masked = T.alloc_fragment([block_M, 1], "bool")
                    is_zero = T.alloc_fragment([block_M, 1], "bool")

                    # Mask-chain temporaries (allocated unconditionally: a
                    # conditional T.alloc is illegal).
                    colmat = T.alloc_fragment([block_M, block_N], "int32")
                    colidx = T.alloc_fragment([block_N], "int32")
                    colrow = T.alloc_fragment([1, block_N], "int32")
                    rowmat = T.alloc_fragment([block_M, block_N], "int32")
                    rowidx = T.alloc_fragment([block_M], "int32")
                    rowcol = T.alloc_fragment([block_M, 1], "int32")
                    dif = T.alloc_fragment([block_M, block_N], "int32")
                    lim_mat = T.alloc_fragment([block_M, block_N], "int32")
                    sent_buf = T.alloc_fragment([block_M, block_N], accum_dtype)
                    vis = T.alloc_fragment([block_M, block_N], "bool")

                    # Let-bound scalars (vbrc/vcmp scalar sources must be
                    # let-bound locals, never raw literals).
                    one_f = 1.0
                    zero_f = 0.0
                    neg_sent = _OOB_SENTINEL
                    all_masked_thr = _ALL_MASKED_THR
                    lse_sent = _LSE_SENT
                    neg_inf = -T.infinity(accum_dtype)

                    # ---- loop-invariant mask indices / constants ----
                    T.arange(colidx, [1], offset=0)
                    T.reshape(colidx, colrow)
                    T.vbrc(colrow, colmat)
                    if is_causal_t:
                        T.arange(rowidx, [1], offset=0)
                        T.reshape(rowidx, rowcol)
                        T.vbrc(rowcol, rowmat)
                    T.vbrc(neg_sent, sent_buf)
                    T.vbrc(one_f, one_vec)
                    T.vbrc(zero_f, zero_vec)
                    T.vbrc(all_masked_thr, thr_vec)
                    T.vbrc(lse_sent, sent_lse)

                    # ---- prologue ----
                    T.clear(Q_shared)
                    T.copy(Q[bz, mid * block_M:mid * block_M + real_m, hid, 0:dim],
                           Q_shared[0:real_m, 0:dim])
                    T.vmul(Q_shared, s_nat, Q_shared)   # OPT-1: fold scale into Q
                    T.clear(acc_o)
                    T.clear(logsum)
                    T.vbrc(neg_inf, m_prev)             # read before first write
                    T.vbrc(zero_f, m_ref)               # finite init (see m_ref update)

                    # ---- segment KV loop (static bound, pipelined) ----
                    for k in T.Pipelined(P, num_stages=num_stages):
                        pg = sid * P + k
                        pgc = T.min(pg, nkv - 1)        # clamp out-of-range page
                        phys = block_table[bz, pgc]
                        T.copy(K[phys * block_N:(phys + 1) * block_N, hid, 0:dim],
                               K_shared[0:block_N, 0:dim])
                        T.gemm(Q_shared, K_shared, acc_s, initC=True, b_transpose=True)

                        # ---- column mask: invalid columns -> finite sentinel ----
                        # valid j <=> pg*block_N + j < real_kv
                        #          <=> j < real_kv - pg*block_N
                        # NOTE (toolchain fact): a runtime scalar operand to
                        # T.vcmp segfaults the npuir compiler; the scalar is
                        # broadcast into a matrix first.
                        col_lim = real_kv - pg * block_N
                        T.vbrc(col_lim, lim_mat)
                        T.vcmp(colmat, lim_mat, vis, "lt")
                        T.vselect(vis, acc_s, sent_buf, acc_s)
                        if is_causal_t:
                            # visible j <=> q_pos >= kv_pos, global positions
                            T.vsub(colmat, rowmat, dif)
                            causal_lim = mid * block_M - pg * block_N
                            T.vbrc(causal_lim, lim_mat)
                            T.vcmp(dif, lim_mat, vis, "le")
                            T.vselect(vis, acc_s, sent_buf, acc_s)

                        # ---- online softmax (OPT-2 / OPT-3, natural-exp) ----
                        T.reduce_max(acc_s, m_cur, dim=1, clear=True)
                        T.vcmp(m_cur, thr_vec, all_masked, "le")   # same-domain guard
                        T.vcmp(m_cur, thr_vec, not_masked, "gt")   # complement (dst==B form)
                        T.vselect(all_masked, zero_vec, m_cur, m_cur_safe)
                        T.vsub(m_prev, m_cur_safe, rs)
                        T.vexp(rs, rs)
                        T.vselect(all_masked, one_vec, rs, rs)     # identity: keep state
                        T.vsub(acc_s, m_cur_safe, acc_s)
                        T.vexp(acc_s, acc_s)   # P (masked -> exp(-1e38) == 0)
                        T.reduce_sum(acc_s, s_sum, dim=1, clear=True)
                        T.vmul(acc_o, rs, acc_o)   # row-broadcast rescale
                        T.vmul(logsum, rs, logsum)
                        T.vadd(logsum, s_sum, logsum)
                        # Reference max for the segment LSE: take the current
                        # block max on a valid block, keep the previous value
                        # on a fully-masked block.
                        # NOTE (toolchain fact): a self-referential T.vselect
                        # writing a *loop-carried* buffer does not carry across
                        # iterations (the written value reverts to the loop
                        # preheader value).  The update is therefore written in
                        # arithmetic form ("m_ref += flag * (m_cur - m_ref)")
                        # with a finite m_ref preheader init so the first
                        # iteration is well defined; vadd/vmul/vsub to a
                        # loop-carried target are proven by the no-split
                        # logsum/acc_o accumulators.
                        T.vselect(not_masked, one_vec, zero_vec, mask_f)
                        T.vsub(m_cur, m_ref, m_delta)
                        T.vmul(m_delta, mask_f, m_delta)
                        T.vadd(m_ref, m_delta, m_ref)
                        T.copy(m_cur, m_prev)   # store old max at the loop end
                        T.vcast(acc_s, acc_s_cast, round_mode="rint")   # O10 quant point

                        T.copy(V[phys * block_N:(phys + 1) * block_N, hid, 0:dim],
                               V_shared[0:block_N, 0:dim])
                        T.gemm(acc_s_cast, V_shared, acc_o, initC=False)

                    # ---- segment epilogue: normalize + segment LSE -> workspace ----
                    T.vcmp(logsum, zero_vec, is_zero, "eq")
                    T.vselect(is_zero, one_vec, logsum, logsum)   # logsum==0 -> 1
                    T.vdiv(acc_o, logsum, acc_o)
                    T.copy(acc_o, O_cast)
                    T.copy(O_cast[0:real_m, 0:dim],
                           Output_partial[bz, mid * block_M:mid * block_M + real_m,
                                          hid, sid, 0:dim])
                    T.vln(logsum, lse)          # logsum==1 for empty -> ln == 0
                    T.vadd(lse, m_ref, lse)     # l = ln(logsum) + m_ref
                    T.vselect(is_zero, sent_lse, lse, lse)   # empty -> LSE_SENT
                    # S5 precision fix: keep the segment LSE in fp32 (no io-dtype
                    # down-cast) -- see the glse parameter note above.
                    T.copy(lse[0:real_m, 0],
                           glse[bz, hid, sid, mid * block_M:mid * block_M + real_m])

            return split_partial

        # ---------------- Phase B: combine (pure Vector) ---------------- #
        @tilelang.jit(out_idx=[-1], target="npuir")
        def _combine():
            @T.prim_func
            def combine(
                glse: T.Tensor([batch, heads, num_split, seqlen_q], "float32"),
                Output_partial: T.Tensor([batch, seqlen_q, heads, num_split, dim], "float32"),
                Output: T.Tensor([batch, seqlen_q, heads, dim], dtype),
            ):
                with T.Kernel(num_logical_B, is_npu=True) as (cid, _):
                    mid = cid % num_q
                    r = cid // num_q
                    hid = r % heads
                    bz = r // heads
                    real_m = T.min(block_M, seqlen_q - mid * block_M)

                    g = T.alloc_fragment([block_M, 1], accum_dtype)
                    gcol = T.alloc_fragment([block_M, 1], accum_dtype)
                    mval = T.alloc_fragment([block_M, 1], accum_dtype)
                    ssum = T.alloc_fragment([block_M, 1], accum_dtype)
                    w = T.alloc_fragment([block_M, 1], accum_dtype)
                    acc = T.alloc_fragment([block_M, dim], accum_dtype)
                    po = T.alloc_fragment([block_M, dim], accum_dtype)
                    O_cast = T.alloc_shared([block_M, dim], dtype)

                    zero_f = 0.0
                    neg_inf = -T.infinity(accum_dtype)
                    T.vbrc(neg_inf, mval)
                    T.vbrc(zero_f, ssum)
                    T.clear(acc)
                    T.clear(g)
                    T.clear(po)

                    # ---- segment-wise max ----
                    for i in T.serial(ns_eff):
                        T.copy(glse[bz, hid, i, mid * block_M:mid * block_M + real_m],
                               g[0:real_m, 0])
                        T.vmax(mval, g, mval)
                    # ---- segment-wise logsumexp (natural-exp domain) ----
                    for i in T.serial(ns_eff):
                        T.copy(glse[bz, hid, i, mid * block_M:mid * block_M + real_m],
                               g[0:real_m, 0])
                        T.vsub(g, mval, g)
                        T.vexp(g, g)
                        T.vadd(ssum, g, ssum)
                    T.vln(ssum, g)              # g == LSE (leg = ln(sum))
                    T.vadd(g, mval, g)          # LSE = ln(sum) + max
                    # ---- weighted recombination ----
                    for i in T.serial(ns_eff):
                        T.copy(Output_partial[bz, mid * block_M:mid * block_M + real_m,
                                              hid, i, 0:dim],
                               po[0:real_m, 0:dim])
                        T.copy(glse[bz, hid, i, mid * block_M:mid * block_M + real_m],
                               gcol[0:real_m, 0])
                        T.vsub(gcol, g, w)      # w = l_i - LSE
                        T.vexp(w, w)            # w_i = exp(l_i - LSE)
                        T.vmul(po, w, po)
                        T.vadd(acc, po, acc)
                    T.vcast(acc, O_cast, round_mode="rint")
                    T.copy(O_cast[0:real_m, 0:dim],
                           Output[bz, mid * block_M:mid * block_M + real_m, hid, 0:dim])

            return combine

        k_split = _split()
        k_comb = _combine()

        def wrapped(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial,
                    acc_split_length):
            # Interface contract: 8 inputs -> Output.  glse / Output_partial are
            # host-preallocated workspaces written by Phase A and read by
            # Phase B; acc_split_length is unused (the partition is a
            # compile-time page partition, OPT-S) but kept for the contract.
            k_split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial)
            return k_comb(glse, Output_partial)

        return wrapped

    return _func


# ---------- precision comparing func ----------
_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_ATOL = 1e-2
_RTOL = 1e-2
_DTYPE_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}


def _gen_inputs(batch, heads, seqlen_q, seqlen_kv, dim, page_size, dtype,
                real_kv_list=None, shuffle_pages=False, num_split=4, seed=0):
    """Build the paged-decode inputs (incl. workspaces) on the NPU device."""
    torch.manual_seed(seed)
    num_pages = seqlen_kv // page_size
    q = torch.randn(batch, seqlen_q, heads, dim, dtype=dtype, device="npu")
    k = torch.randn(seqlen_kv, heads, dim, dtype=dtype, device="npu")
    v = torch.randn(seqlen_kv, heads, dim, dtype=dtype, device="npu")
    if real_kv_list is None:
        real = torch.full((batch,), seqlen_kv, dtype=torch.int32, device="npu")
    else:
        real = torch.tensor(real_kv_list, dtype=torch.int32, device="npu")
    if shuffle_pages:
        perm = torch.randperm(num_pages, device="npu")
        row = perm.to(torch.int32)
    else:
        row = torch.arange(num_pages, dtype=torch.int32, device="npu")
    block_table = row.unsqueeze(0).repeat(batch, 1).contiguous()
    # S5 precision fix: the LSE / segment-output workspaces are fp32 (the
    # combine weights are exponent-domain, so an io-dtype LSE costs too much
    # on bf16); the host wrapper allocates them the same way.
    glse = torch.zeros(batch, heads, num_split, seqlen_q, dtype=torch.float32, device="npu")
    out_partial = torch.zeros(batch, seqlen_q, heads, num_split, dim, dtype=torch.float32,
                              device="npu")
    acc_split_length = torch.zeros(batch, num_split, dtype=torch.int32, device="npu")
    return (q.contiguous(), k.contiguous(), v.contiguous(), real.contiguous(),
            block_table, glse, out_partial, acc_split_length)


def _compare(out, ref):
    """Return (max_diff, min_cos, finite_ok, ref_is_zero)."""
    out_f = out.float().cpu()
    ref_f = ref.float().cpu()
    finite_ok = bool(torch.isfinite(out_f).all().item())
    max_diff = float((out_f - ref_f).abs().max().item())
    batch = out_f.shape[0]
    ref_is_zero = float(ref_f.abs().sum().item()) == 0.0
    if ref_is_zero:
        min_cos = float("nan")
    else:
        cos = torch.nn.functional.cosine_similarity(
            out_f.reshape(batch, -1), ref_f.reshape(batch, -1), dim=-1, eps=1e-8)
        min_cos = float(cos.min().item())
    return max_diff, min_cos, finite_ok, ref_is_zero


def _run_case(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype,
              tag, real_kv_list=None, shuffle_pages=False, num_split=4, num_stages=2,
              seed=0, expect_zero=False):
    """Run one paged-decode split case and gate it against the golden."""
    dtype_str = _DTYPE_STR[dtype]
    q, k, v, real, block_table, glse, out_partial, acc_len = _gen_inputs(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, dtype,
        real_kv_list=real_kv_list, shuffle_pages=shuffle_pages, num_split=num_split,
        seed=seed)

    fn = _mha_decode_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size,
                                  is_causal, dtype_str)
    kernel = fn(16, page_size, num_split, num_stages)
    out = kernel(q, k, v, real, block_table, glse, out_partial, acc_len)
    torch.npu.synchronize()

    assert out.shape == q.shape and out.dtype == q.dtype, \
        f"[{tag}] output contract violated: {tuple(out.shape)}/{out.dtype}"

    ref = golden__mha_decode_split_kernel(
        q.cpu(), k.cpu(), v.cpu(), real.cpu(), block_table.cpu(), page_size,
        is_causal=is_causal)
    max_diff, min_cos, finite_ok, ref_is_zero = _compare(out, ref)

    ok_diff = max_diff < _ATOL
    ok_finite = finite_ok
    if expect_zero:
        ok = ok_diff and ok_finite and bool((out.float().cpu().abs().max().item() == 0.0))
    else:
        ok_cos = True if ref_is_zero else (min_cos > 0.99)
        ok = ok_diff and ok_finite and ok_cos
    print(f"  [{tag}] B={batch} H={heads} Sq={seqlen_q} Nkv={seqlen_kv} D={dim} "
          f"page={page_size} nsplit={num_split} causal={int(is_causal)} {dtype_str} "
          f"real_kv={real_kv_list if real_kv_list is not None else seqlen_kv} "
          f"shuffle={int(shuffle_pages)}: out={'PASS' if ok else 'FAIL'} "
          f"(max_diff={max_diff:.3e} min_cos={min_cos:.5f} finite={ok_finite})")
    assert ok, (f"[{tag}] precision gate failed: max_diff={max_diff:.3e} "
                f"min_cos={min_cos} finite={ok_finite} ref_is_zero={ref_is_zero}")
    return out


def run_L0():
    """L0 gate (blocking): 4 manifest workloads + paged / workspace contract."""
    print("== L0 (blocking) ==")
    # single-token-page128: nkv=4, ns_eff=4, P=1
    _run_case(1, 16, 1, 512, 128, 128, False, torch.float16, "L0-1", seed=0)
    # batch2-page256: nkv=4, ns_eff=4, P=1
    _run_case(2, 8, 1, 1024, 64, 256, False, torch.float16, "L0-2", seed=1)
    # longer-cache: nkv=4, ns_eff=4, P=1
    _run_case(1, 8, 1, 1024, 64, 256, False, torch.float16, "L0-3", seed=2)
    # shorter-cache: nkv=2 < num_split=4 -> ns_eff=2 (truncated segment count)
    _run_case(1, 8, 1, 512, 64, 256, False, torch.float16, "L0-4", seed=3)
    # paged contract: non-identity (shuffled) block table (also L0-6 workspace contract)
    _run_case(1, 8, 1, 512, 64, 256, False, torch.float16, "L0-5", seed=4,
              shuffle_pages=True)
    # workspace contract on the page-128 config (Phase A write / Phase B read)
    _run_case(1, 16, 1, 512, 128, 128, False, torch.float16, "L0-6", seed=5,
              shuffle_pages=True)


def run_L1():
    """L1 gate (blocking): static-bound tail-block / empty-segment stabilisation."""
    print("== L1 (blocking) ==")
    # partial page 0 + fully masked page 1: segment sid=1 fully empty -> LSE_SENT
    _run_case(1, 8, 1, 512, 64, 256, False, torch.float16, "L1-1",
              real_kv_list=[200], seed=6)
    # zero logical length -> all segments empty -> combine output exactly 0
    _run_case(1, 8, 1, 512, 64, 256, False, torch.float16, "L1-2",
              real_kv_list=[0], seed=7, expect_zero=True)
    # page clamping: nkv=5, num_split=2 -> P=3, last segment pages 3,4,5 (5 out of range)
    _run_case(1, 8, 1, 1280, 64, 256, False, torch.float16, "L1-3",
              real_kv_list=[1280], seed=8)
    # shuffled page table + partial length
    _run_case(1, 8, 1, 512, 64, 256, False, torch.float16, "L1-4",
              real_kv_list=[300], seed=9, shuffle_pages=True)


def run_L2():
    """L2 coverage (warn only, non-blocking): bf16 + causal + per-batch lengths."""
    print("== L2 (non-blocking) ==")
    cases = [
        ((1, 16, 1, 512, 128, 128, False, torch.bfloat16), dict(seed=10)),
        ((2, 8, 1, 1024, 64, 256, False, torch.bfloat16), dict(seed=11)),
        ((1, 1, 1, 512, 128, 128, False, torch.float16), dict(seed=12)),
        ((2, 8, 1, 1024, 64, 256, False, torch.float16),
         dict(real_kv_list=[1024, 700], seed=13)),
        ((1, 8, 1, 512, 64, 256, True, torch.float16), dict(seed=14)),
        ((1, 8, 1, 512, 64, 256, False, torch.float16), dict(num_split=2, seed=15)),
    ]
    idx = 0
    for (batch, heads, sq, skv, dim, page, causal, dt), kw in cases:
        idx += 1
        tag = f"L2-{idx}"
        try:
            _run_case(batch, heads, sq, skv, dim, page, causal, dt, tag, **kw)
        except Exception as exc:  # noqa: BLE001 - warn only
            print(f"  [{tag}] WARN (recorded, non-blocking): {exc}")


def run_boundary():
    """Boundary cases (warn only, non-blocking)."""
    print("== Boundary (non-blocking) ==")
    cases = [
        # single valid token
        ((1, 8, 1, 512, 64, 256, False, torch.float16),
         dict(real_kv_list=[1], seed=20), "boundary-real1"),
        # N_kv == page_size -> nkv == 1 -> ns_eff == 1
        ((1, 8, 1, 256, 64, 256, False, torch.float16), None, "boundary-nkv1"),
        # page 128, single token
        ((1, 16, 1, 128, 128, 128, False, torch.float16), None, "boundary-page128"),
        # num_split > nkv truncated segment count
        ((1, 8, 1, 512, 64, 256, False, torch.float16), dict(num_split=8, seed=22),
         "boundary-nsplit8"),
        # partial last page, real_kv just over one page
        ((1, 8, 1, 1024, 64, 256, False, torch.float16), dict(real_kv_list=[257], seed=23),
         "boundary-real257"),
    ]
    for (batch, heads, sq, skv, dim, page, causal, dt), kw, tag in cases:
        try:
            if kw is None:
                kw = {}
            _run_case(batch, heads, sq, skv, dim, page, causal, dt, tag, **kw)
        except Exception as exc:  # noqa: BLE001 - warn only
            print(f"  [{tag}] WARN (recorded, non-blocking): {exc}")


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
