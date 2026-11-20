# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Paged GQA prefill kernels for the npuir backend.

The fused Vector kernel handles supported uniform, aligned FP16 requests.
The decomposed fallback runs QK, softmax, and PV as separate kernels.
See ../TERMINAL_STATE.md for the original NPU port.
"""

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

__all__ = [
    "gqa_prefill_paged_fwd",
    "golden_gqa_prefill_paged",
]


# ---------------------------------------------------------------------------
# Kernel 1: S = Q @ K^T   (Cube only)
# ---------------------------------------------------------------------------
def kernel_qk(M: int, N: int, D: int):
    """Q: (M, D), Kt: (D, N) -> S: (M, N).  Pure Cube, no cross-engine sync."""
    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(Q: T.Tensor((M, D), "float16"),
                 Kt: T.Tensor((D, N), "float16"),
                 S: T.Tensor((M, N), "float16")):
            with T.Kernel(1, is_npu=True) as (cid, subid):
                with T.Scope("Cube"):
                    Q_L1 = T.alloc_L1((M, D), "float16")
                    K_L1 = T.alloc_L1((D, N), "float16")
                    S_L0 = T.alloc_L0C((M, N), "float32")
                    T.copy(Q, Q_L1)
                    T.copy(Kt, K_L1)
                    # initC=True is REQUIRED: without it the GEMM ACCUMULATES into
                    # the uninitialised L0C buffer, so the result is "stale + A@B"
                    # and the algebra is wrong (measured: I @ Kt came out as
                    # garbage, max_err 2.05).  The repo's own MHA kernel does the
                    # same -- `with T.rs("PIPE_C"): T.gemm(..., initC=True, ...)`.
                    with T.rs("PIPE_C"):
                        T.gemm(Q_L1, K_L1, S_L0, initC=True)
                    with T.rs("PIPE_FIX"):
                        T.npuir_store_fixpipe(S_L0, S, [M, N], enable_nz2nd=True)
        return main
    return _f()


# ---------------------------------------------------------------------------
# Kernel 2: softmax(S + mask)   (Vector only)
# ---------------------------------------------------------------------------
def kernel_softmax(M: int, N: int):
    """P = softmax(S + mask) row-wise.  Pure Vector.

    The additive mask carries the causal structure (0 on the kept side, a large
    negative on the masked side), so the kernel itself needs no branching.
    """
    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(S: T.Tensor((M, N), "float16"),
                 mask: T.Tensor((M, N), "float32"),
                 P: T.Tensor((M, N), "float16")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                s16 = T.alloc_ub((1, N), "float16")
                row = T.alloc_ub((1, N), "float32")
                mrow = T.alloc_ub((1, N), "float32")
                o16 = T.alloc_ub((1, N), "float16")
                mx = T.alloc_ub((1, 1), "float32")
                sm = T.alloc_ub((1, 1), "float32")
                for i in T.serial(M):
                    # T.copy does not cast, so load fp16 then widen explicitly.
                    T.copy(S[i:i + 1, 0:N], s16[0:1, 0:N])
                    T.vcast(s16, row, round_mode="rint")
                    T.copy(mask[i:i + 1, 0:N], mrow[0:1, 0:N])
                    T.vadd(row, mrow, row)                # apply the causal mask
                    T.reduce(row, mx, dims=1, reduce_mode="max")
                    T.vsub(row, mx[0, 0], row)            # - max (stability)
                    T.vexp(row, row)
                    T.reduce(row, sm, dims=1, reduce_mode="sum")
                    T.vdiv(row, sm[0, 0], row)            # / sum
                    T.vcast(row, o16, round_mode="rint")
                    T.copy(o16[0:1, 0:N], P[i:i + 1, 0:N])
        return main
    return _f()


# ---------------------------------------------------------------------------
# Kernel 3: O = P @ V   (Cube only)
# ---------------------------------------------------------------------------
def kernel_pv(M: int, N: int, D: int):
    """P: (M, N), V: (N, D) -> O: (M, D).  Pure Cube."""
    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(P: T.Tensor((M, N), "float16"),
                 V: T.Tensor((N, D), "float16"),
                 O: T.Tensor((M, D), "float16")):
            with T.Kernel(1, is_npu=True) as (cid, subid):
                with T.Scope("Cube"):
                    P_L1 = T.alloc_L1((M, N), "float16")
                    V_L1 = T.alloc_L1((N, D), "float16")
                    O_L0 = T.alloc_L0C((M, D), "float32")
                    T.copy(P, P_L1)
                    T.copy(V, V_L1)
                    with T.rs("PIPE_C"):
                        T.gemm(P_L1, V_L1, O_L0, initC=True)   # initC: see kernel_qk
                    with T.rs("PIPE_FIX"):
                        T.npuir_store_fixpipe(O_L0, O, [M, D], enable_nz2nd=True)
        return main
    return _f()


# ---------------------------------------------------------------------------
# Host assembly: paged gather + dequant + head repeat + causal mask
# ---------------------------------------------------------------------------
# Kernel factory cache.  Building a factory is NOT free: measured on this
# toolchain ``kernel_qk(M, N, D)`` costs ~11 ms even when the compiled kernel is
# already in tilelang's cache, while an actual kernel call costs 0.12 ms.  The
# first version of this file rebuilt the factories inside the row-block loop, so
# for gqa-mid (2 batches x 4 blocks x 3 kernels = 24 rebuilds) that alone burned
# ~264 ms of the 346 ms total.  Keyed by the exact shape, the factories are now
# built once per process.
_KERNEL_CACHE: dict = {}


def _get_kernels(m: int, n: int, d: int):
    """Return (qk, softmax, pv) kernels for an (m, n, d) score block, cached."""
    key = (m, n, d)
    hit = _KERNEL_CACHE.get(key)
    if hit is None:
        hit = (kernel_qk(m, n, d), kernel_softmax(m, n), kernel_pv(m, n, d))
        _KERNEL_CACHE[key] = hit
    return hit

def _page_positions(block_table_row, n_page, page_size):
    """Token indices for the pages of ONE request.

    ``k_pages`` is shaped (physical_tokens, H_kv, D) -- flattened over tokens --
    so a page id maps to the ``page_size`` CONSECUTIVE tokens starting at
    ``page_id * page_size``.  Indexing the token axis with raw page ids (the first
    version of this code did that) gathers the wrong rows and the later reshape
    fails with a size mismatch.
    """
    pages = block_table_row[:n_page].long()                     # (n_page,)
    # The offsets must live on the SAME device as the page table, otherwise
    # index_select below raises "Expected all tensors to be on the same device".
    off = torch.arange(page_size, dtype=torch.long, device=pages.device)
    return (pages.unsqueeze(1) * page_size + off.unsqueeze(0)).reshape(-1)


def _gather_kv(k_pages, v_pages, k_new, v_new, k_scale, v_scale,
               cu_seqlens_q, cache_seqlens, block_table, page_size, b):
    """Dense K/V for batch b: cache pages (dequantized) then the new tokens.

    Returns (K, V) each of shape (S_kv, H_kv, D) in fp16.
    """
    q0, q1 = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
    n_q = q1 - q0
    n_cache = int(cache_seqlens[b])
    n_page = (n_cache + page_size - 1) // page_size

    # 1. gather the cache pages listed in block_table for this request
    pos = _page_positions(block_table[b], n_page, page_size)
    kc = k_pages.index_select(0, pos)[:n_cache]               # (n_cache, H_kv, D)
    vc = v_pages.index_select(0, pos)[:n_cache]

    # 2. online dequant (scalar scales, cast back to fp16)
    if k_scale is not None:
        kc = (kc.to(torch.float32) * k_scale.reshape(()).to(torch.float32)).to(torch.float16)
    if v_scale is not None:
        vc = (vc.to(torch.float32) * v_scale.reshape(()).to(torch.float32)).to(torch.float16)

    # 3. append the new tokens for this request
    kn = k_new[q0:q1]
    vn = v_new[q0:q1]
    K = torch.cat([kc, kn], dim=0).contiguous()
    V = torch.cat([vc, vn], dim=0).contiguous()
    return K, V, n_q, n_cache


def _causal_mask(n_q: int, n_kv: int, n_cache: int):
    """Additive mask: query i may attend to keys up to (n_cache + i).

    The masked value is a LARGE FINITE NEGATIVE, not ``-inf`` / ``finfo.min``:
    the kernel adds it to S and then exponentiates, and ``vexp`` on a value near
    -1.7e38 produced inf/NaN and poisoned the whole row (measured: the softmax
    stage's error jumped from ~3e-4 with an all-zero mask to 1.2e-1 with the -inf
    mask).  -1e4 is far below any real score here, so exp() underflows to exactly
    0 as intended while staying inside fp32 range.
    """
    MASK_NEG = -1.0e4
    qi = torch.arange(n_q).unsqueeze(1)
    ki = torch.arange(n_kv).unsqueeze(0)
    allowed = ki <= (n_cache + qi)
    mask = torch.zeros(n_q, n_kv, dtype=torch.float32)
    return mask.masked_fill(~allowed, MASK_NEG).contiguous()


# ---------------------------------------------------------------------------
# Public entry: decomposed GQA prefill with a paged KV cache
# ---------------------------------------------------------------------------
def gqa_prefill_paged_fwd(q, k_new, v_new, k_pages, v_pages, k_scale, v_scale,
                          cu_seqlens_q, cache_seqlens, block_table,
                          page_size, is_causal=True, sm_scale=None):
    """Returns o of shape (total_q, H, D).

    The decomposition is described in the module docstring.  All indexing work
    (page gather, dequant, head repeat, causal mask) happens on the host; the NPU
    sees three single-engine kernels.
    """
    total_q, H, D = q.shape
    H_kv = k_new.shape[1]
    if H % H_kv:
        raise ValueError(f"H={H} must be a multiple of H_kv={H_kv}")
    rep = H // H_kv
    B = cu_seqlens_q.numel() - 1

    o = torch.empty_like(q)
    for b in range(B):
        K, V, n_q, n_cache = _gather_kv(
            k_pages, v_pages, k_new, v_new, k_scale, v_scale,
            cu_seqlens_q, cache_seqlens, block_table, page_size, b)
        n_kv = K.shape[0]

        # GQA head mapping is applied by INDEX (`K[:, h // rep, :]`) below rather
        # than by materialising `repeat_interleave(rep)` copies of K/V: the repeats
        # were only ever read one head at a time, so building them cost two large
        # tensor copies per request for nothing.

        q0 = int(cu_seqlens_q[b])
        _dev = q.device
        mask = (_causal_mask(n_q, n_kv, n_cache).to(_dev)
                if is_causal else torch.zeros(n_q, n_kv, dtype=torch.float32,
                                              device=_dev))
        scale = (1.0 / (D ** 0.5)) if sm_scale is None else float(sm_scale)

        # L0C holds n_q * n_kv fp32 scores; 128 KB / 4 B = 32768 elements.  Beyond
        # that the compiler refuses the kernel:
        #   error: cc overflow, requires 1572864 bits while 1048576 bits available!
        # Chunking over ROWS is mathematically EXACT here -- softmax is row-wise and
        # P@V is row-independent -- so a smaller block changes nothing but the
        # buffer size.  (Chunking over n_kv would need an online softmax and is not
        # done; a single row wider than the budget is rejected below.)
        _L0C_ELEMS = 16384            # half the hardware limit, for headroom
        if n_kv > _L0C_ELEMS:
            raise ValueError(
                f"n_kv={n_kv} exceeds the L0C budget ({_L0C_ELEMS} fp32 scores); "
                f"this decomposed form does not implement an online softmax over "
                f"column blocks")
        bq = max(1, min(n_q, _L0C_ELEMS // n_kv))

        # Hoist every per-head tensor transform out of the loops.  Kt/Vh depend
        # ONLY on the head, yet the first version rebuilt them inside the row-block
        # loop as well, so they were recomputed blocks x heads times -- and each
        # rebuild is a fresh NPU allocation + copy.  Building them once per request
        # turns 3 * blocks * heads tensor ops into 3 * heads, and the row slices
        # taken below are then contiguous VIEWS (a row range of a contiguous 2-D
        # tensor is contiguous), so no further copies happen per kernel call.
        Kt_heads = [K[:, h // rep, :].t().contiguous().to(torch.float16)
                    for h in range(H)]                                   # H x (D, n_kv)
        V_heads = [V[:, h // rep, :].contiguous().to(torch.float16)
                   for h in range(H)]                                    # H x (n_kv, D)
        Q_heads = [(q[q0:q0 + n_q, h, :].to(torch.float16) * scale).contiguous()
                   for h in range(H)]                                    # H x (n_q, D)

        for qb in range(0, n_q, bq):
            qlen_b = min(bq, n_q - qb)
            maskb = mask[qb:qb + qlen_b]
            kq, ks, kp = _get_kernels(qlen_b, n_kv, D)
            for h in range(H):
                Qh = Q_heads[h][qb:qb + qlen_b]          # contiguous view
                S = kq(Qh, Kt_heads[h])                  # fp16 (qlen_b, n_kv)
                P = ks(S, maskb)
                O = kp(P, V_heads[h])
                # `o` lives on the same device as `q`, so the kernel result must
                # NOT be moved to CPU (that raises a device-mismatch error).
                o[q0 + qb:q0 + qb + qlen_b, h, :] = O.to(q.dtype)
    return o


# ---------------------------------------------------------------------------
# Golden (independent of the NPU algorithm)
# ---------------------------------------------------------------------------
def golden_gqa_prefill_paged(q, k_new, v_new, k_pages, v_pages, k_scale, v_scale,
                             cu_seqlens_q, cache_seqlens, block_table,
                             page_size, is_causal=True, sm_scale=None):
    """Straightforward per-request torch reference (no paging tricks)."""
    total_q, H, D = q.shape
    H_kv = k_new.shape[1]
    rep = H // H_kv
    B = cu_seqlens_q.numel() - 1
    o = torch.empty_like(q, dtype=torch.float32)
    scale = (1.0 / (D ** 0.5)) if sm_scale is None else float(sm_scale)

    for b in range(B):
        q0, q1 = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        n_q = q1 - q0
        n_cache = int(cache_seqlens[b])
        n_page = (n_cache + page_size - 1) // page_size
        pos = _page_positions(block_table[b], n_page, page_size)
        kc = k_pages.index_select(0, pos)[:n_cache]
        vc = v_pages.index_select(0, pos)[:n_cache]
        kc = (kc.float() * float(k_scale.reshape(()))).half()
        vc = (vc.float() * float(v_scale.reshape(()))).half()
        K = torch.cat([kc, k_new[q0:q1]], dim=0).float()
        V = torch.cat([vc, v_new[q0:q1]], dim=0).float()
        n_kv = K.shape[0]

        K_rep = K.repeat_interleave(rep, dim=1)               # (n_kv, H, D)
        V_rep = V.repeat_interleave(rep, dim=1)
        for h in range(H):
            Qh = q[q0:q1, h, :].float() * scale               # (n_q, D)
            Kh = K_rep[:, h, :]                               # (n_kv, D)
            Vh = V_rep[:, h, :]
            S = Qh @ Kh.t()                                   # (n_q, n_kv)
            if is_causal:
                qi = torch.arange(n_q).unsqueeze(1)
                ki = torch.arange(n_kv).unsqueeze(0)
                S = S.masked_fill(ki > (n_cache + qi), float("-inf"))
            P = torch.softmax(S, dim=-1)
            o[q0:q1, h, :] = P @ Vh
    return o


# ---------------------------------------------------------------------------
# Fused Vector kernel
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def fused_paged_gqa(batch, qlen, cache, heads, kv_heads, dim, physical, pages, page,
                    causal, scale, dtype, m, contiguous_cache):
    """Build the fused Vector kernel for uniform, aligned FP16 requests."""
    n = qlen + cache
    short = n == 192
    mask_shape = (8, 2 * qlen) if short else (1, qlen)
    wide = n == 384
    qt = (qlen + m - 1) // m
    groups = heads // kv_heads
    total_q = batch * qlen

    @tilelang.jit(target="npuir", out_idx=[8],
                  pass_configs={"npuir.enable_auto_multi_buffer": False})
    def compile_kernel():
        @T.prim_func
        def paged_gqa(
            Q: T.Tensor((total_q, heads, dim), dtype),
            Knew: T.Tensor((total_q, kv_heads, dim), dtype),
            Vnew: T.Tensor((total_q, kv_heads, dim), dtype),
            Kpages: T.Tensor((physical, kv_heads, dim), dtype),
            Vpages: T.Tensor((physical, kv_heads, dim), dtype),
            Kscale: T.Tensor((1,), "float32"),
            Vscale: T.Tensor((1,), "float32"),
            Table: T.Tensor((batch, pages), "int32"),
            O: T.Tensor((total_q, heads, dim), dtype),
        ):
            with T.Kernel(batch * heads * qt, is_npu=True) as (cid, _):
                qtile = cid % qt
                head = cid // qt % heads
                kv_head = head // groups
                batch_id = cid // (qt * heads)
                qstart = batch_id * qlen

                keys_row = T.alloc_ub((n, 4, 16), dtype)
                keys_pack = T.alloc_ub((4, n, 16), dtype)
                key = T.alloc_ub((4 * n, 16), dtype)
                value = T.alloc_ub((n, dim), dtype)
                work = T.alloc_ub((n, dim), dtype)
                query = T.alloc_ub((1, dim), dtype)
                dot = T.alloc_ub((n, 1), dtype)
                q_lanes = T.alloc_ub((4, 16), dtype)
                qk_work = T.alloc_ub((4 * n, 16), dtype)
                dot_row = T.alloc_ub((1, n), dtype)
                score = T.alloc_ub((1, n), "float32")
                mask = T.alloc_ub(mask_shape, "float32")
                mx = T.alloc_ub((1, 1), "float32")
                den = T.alloc_ub((1, 1), "float32")
                out = T.alloc_ub((1, dim), dtype)

                if contiguous_cache:
                    first = Table[batch_id, 0] * page
                    T.copy(Kpages[first:first + cache, kv_head, 0:dim], work[0:cache, 0:dim])
                    T.copy(Vpages[first:first + cache, kv_head, 0:dim], value[0:cache, 0:dim])
                else:
                    for p in T.serial(cache // page):
                        phys = Table[batch_id, p] * page
                        T.copy(Kpages[phys:phys + page, kv_head, 0:dim],
                               work[p * page:(p + 1) * page, 0:dim])
                        T.copy(Vpages[phys:phys + page, kv_head, 0:dim],
                               value[p * page:(p + 1) * page, 0:dim])
                T.copy(Knew[qstart:qstart + qlen, kv_head, 0:dim], work[cache:n, 0:dim])
                T.copy(Vnew[qstart:qstart + qlen, kv_head, 0:dim], value[cache:n, 0:dim])
                # Pack K in groups of 16 features for reuse across query rows.
                T.reshape(work, keys_row)
                T.transpose(keys_row, keys_pack, permutation=[1, 0, 2])
                T.reshape(keys_pack, key)
                T.reshape(work, qk_work)
                if causal and short:
                    T.arange(mask, [-1, 1], -qlen)
                    T.vmax(mask, 0.0, mask)
                    T.vmin(mask, 1.0, mask)
                    T.vmul(mask, -1e4, mask)
                for i in T.serial(T.min(m, qlen - qtile * m)):
                    row = qtile * m + i
                    T.copy(Q[qstart + row, head, 0:dim], query[0, 0:dim])
                    T.reshape(query, q_lanes)
                    for lane in T.unroll(4):
                        begin = lane * n
                        T.vmul(key[begin:begin + n, 0:16], q_lanes[lane:lane + 1, 0:16],
                               qk_work[begin:begin + n, 0:16])
                    T.vadd(qk_work[0:2 * n, 0:16], qk_work[2 * n:4 * n, 0:16], qk_work[0:2 * n, 0:16])
                    T.vadd(qk_work[0:n, 0:16], qk_work[n:2 * n, 0:16], qk_work[0:n, 0:16])
                    T.reduce_sum(qk_work[0:n, 0:16], dot, dim=1, clear=True)
                    T.reshape(dot, dot_row)
                    T.vcast(dot_row, score, round_mode="rint")
                    T.vmul(score, scale, score)
                    T.vmul(score[0:1, 0:cache], Kscale[0], score[0:1, 0:cache])
                    if causal:
                        if short:
                            # Align the mask window to a 32-byte UB boundary.
                            lane = row % 8
                            start = qlen - row // 8 * 8
                            T.vadd(score[0:1, cache:n], mask[lane:lane + 1, start:start + qlen],
                                   score[0:1, cache:n])
                        else:
                            T.arange(mask, [0, 1], -row)
                            T.vmax(mask, 0.0, mask)
                            T.vmin(mask, 1.0, mask)
                            T.vmul(mask, -1e4, mask)
                            T.vadd(score[0:1, cache:n], mask, score[0:1, cache:n])
                    T.reduce_max(score, mx, dim=1, clear=True)
                    T.vsub(score, mx, score)
                    T.vexp(score, score)
                    T.reduce_sum(score, den, dim=1, clear=True)
                    T.vrec(den, den)
                    T.vmul(score, den, score)
                    T.vmul(score[0:1, 0:cache], Vscale[0], score[0:1, 0:cache])
                    T.vcast(score, dot_row, round_mode="rint")
                    T.reshape(dot_row, dot)
                    T.vmul(value, dot, work)
                    if wide:
                        T.vadd(work[0:192, 0:dim], work[192:384, 0:dim], work[0:192, 0:dim])
                    T.vadd(work[0:96, 0:dim], work[96:192, 0:dim], work[0:96, 0:dim])
                    T.vadd(work[0:48, 0:dim], work[48:96, 0:dim], work[0:48, 0:dim])
                    T.vadd(work[0:24, 0:dim], work[24:48, 0:dim], work[0:24, 0:dim])
                    T.vadd(work[0:12, 0:dim], work[12:24, 0:dim], work[0:12, 0:dim])
                    T.vadd(work[0:6, 0:dim], work[6:12, 0:dim], work[0:6, 0:dim])
                    T.vadd(work[0:3, 0:dim], work[3:6, 0:dim], work[0:3, 0:dim])
                    T.vadd(work[0:1, 0:dim], work[1:2, 0:dim], out)
                    T.vadd(out, work[2:3, 0:dim], out)
                    T.copy(out[0, 0:dim], O[qstart + row, head, 0:dim])
        return paged_gqa
    return compile_kernel()
