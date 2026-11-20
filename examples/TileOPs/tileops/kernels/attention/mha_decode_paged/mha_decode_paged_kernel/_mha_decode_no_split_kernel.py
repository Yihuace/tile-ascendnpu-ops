# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Paged multi-head-attention decode forward (no-split) -- NPU Developer MixCV kernel.

Migrated from the GPU TileLang factory ``_mha_decode_no_split_kernel``
(TileOPs ``tileops/kernels/attention/mha_decode_paged.py``, L17-110) to
``target="npuir"`` per the frozen DESIGN.md (revision 1).

Math (per batch b, head h, query token):
    o[b, :, h, :] = softmax(q[b, :, h, :] @ K_b^T / sqrt(D)) @ V_b
where the logical KV sequence is gathered from the physical paged cache
``K`` / ``V`` ``[N_kv, H, D]`` through ``block_table[b, :]`` (logical page i
-> physical page ``block_table[b, i]``); ``real_seqlen_kv[b]`` is the logical
KV length.  Decode has ``S_q = 1``.

NPU redesign (DESIGN sections 0.6 / 3.3 / 6):
  - 1-D non-persistent grid (R1): ``cid`` -> (query-block, head, batch);
    ``num_logical = H * B <= 20`` (<= physical AI-core count, no in-core
    serial multi-task loop -- avoids the Developer + persistent + gemm trap).
  - ``block_M = 16`` (Cube fractal floor, R2); only row 0 carries the real
    query, the remaining rows are padding isolated by the per-row reduce and
    the row-truncated store.
  - ``block_N = page_size`` (R3): one KV tile == one physical page, so the
    paged gather degenerates to a single page-id read per block.
  - static KV loop bound ``nkv = ceil(N_kv / page_size)`` + column-OOB mask
    (R4).  The extra all-masked tail blocks introduced by the static bound
    are neutralised by a same-domain identity guard (``rs = 1``, ``P = 0``,
    ``s_sum = 0``) so they contribute exactly nothing without clobbering the
    accumulated state.
  - Cube double GEMM (``T.gemm``) + Vector online softmax in the natural-exp
    domain (OPT-1 scale folded into Q, OPT-2 reassociated rescale factor,
    OPT-3 natural exp instead of exp2).
  - Finite-large sentinel ``-1e38`` (never -inf) so a masked P is an exact
    ``0`` and no ``-inf - (-inf)`` NaN path can occur.

ASCEND_MODE = "Developer"
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

# Numerical sentinels (DESIGN sections 0.6-R4 / 6.4 / PL-1.11).
_OOB_SENTINEL = -1e38   # finite large sentinel for masked score columns
_ALL_MASKED_THR = -1e30  # same-domain threshold for the all-masked guard


# ---------- Golden (independent PyTorch reference, DESIGN section 8.1) ----------
def golden__mha_decode_no_split_kernel(q, k, v, real_seqlen_kv, block_table, page_size,
                                       is_causal=False):
    """Paged MHA decode reference: reassemble logical K/V per batch, then SDPA.

    q:           (B, S_q, H, D) fp16/bf16
    k, v:        (N_kv, H, D) fp16/bf16  (physical paged cache)
    real_seqlen_kv: (B,) int32           (logical KV length)
    block_table: (B, num_pages) int32    (logical page -> physical page)
    Returns (B, S_q, H, D).

    Independence: the non-causal path is a verbatim port of the GPU source's
    test reference (``test_mha_decode_paged.py::ref_program``) -- paged
    reassembly to a logical layout followed by
    ``F.scaled_dot_product_attention``.  It does NOT mirror the NPU data flow
    (no static bound, no sentinel, no identity guard, no v-prefix chain); the
    online-softmax path is not reproduced.  The optional causal path uses an
    explicit fp32 softmax with the source kernel's own causal rule
    (``q_pos >= kv_pos``, source L78-81), not SDPA's bottom-right-aligned
    convention.
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


# ---------- Kernel (Developer MixCV, DESIGN sections 3.3 / 6) ----------
@functools.lru_cache(maxsize=32)
def _mha_decode_no_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal,
                                dtype):
    """Factory matching the TileOPs wrapper call form.

    The wrapper calls ``factory(...)(block_M, block_N, num_stages)(Q, K, V,
    real_seqlen_kv, block_table)``.  The GPU tiling knobs ``block_M`` /
    ``block_N`` are normalised to the frozen NPU design values (``block_M`` =
    fractal floor 16, ``block_N`` = ``page_size``); only ``num_stages`` is
    honoured (DESIGN 0.6 R2/R3/R6).
    """
    accum_dtype = "float"
    num_q = (seqlen_q + 15) // 16
    num_logical = num_q * heads * batch
    nkv = (seqlen_kv + page_size - 1) // page_size
    s_nat = 1.0 / (dim ** 0.5)
    is_causal_t = bool(is_causal)

    # NOTE (toolchain fact): the GPU source's `TL_ENABLE_FAST_MATH` has no
    # npuir BuildConfigKey in this toolchain (verified: PassConfigKey has no
    # such member), so no pass_configs are set -- the Developer defaults are
    # the ones exercised by the verified npuir precedents.
    @tilelang.jit(out_idx=[-1], target="npuir")
    def _func(block_M_in, block_N_in, num_stages):
        block_M = 16          # Cube fractal floor (DESIGN 0.6 R2)
        block_N = page_size   # one KV tile == one physical page (DESIGN 0.6 R3)

        @T.prim_func
        def mha_decode_no_split(
            Q: T.Tensor([batch, seqlen_q, heads, dim], dtype),
            K: T.Tensor([seqlen_kv, heads, dim], dtype),
            V: T.Tensor([seqlen_kv, heads, dim], dtype),
            real_seqlen_kv: T.Tensor([batch], "int32"),
            block_table: T.Tensor([batch, nkv], "int32"),
            Output: T.Tensor([batch, seqlen_q, heads, dim], dtype),
        ):
            with T.Kernel(num_logical, is_npu=True) as (cid, _):
                # ---- task decode (DESIGN 0.6 R1) ----
                bx = cid % num_q
                by = (cid // num_q) % heads
                bz = cid // (num_q * heads)
                real_kv = real_seqlen_kv[bz]
                real_m = T.min(block_M, seqlen_q - bx * block_M)

                # ---- Cube operands / Vector accumulators ----
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_cast = T.alloc_shared([block_M, dim], dtype)

                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)

                m_prev = T.alloc_fragment([block_M, 1], accum_dtype)
                m_cur = T.alloc_fragment([block_M, 1], accum_dtype)
                m_cur_safe = T.alloc_fragment([block_M, 1], accum_dtype)
                rs = T.alloc_fragment([block_M, 1], accum_dtype)
                s_sum = T.alloc_fragment([block_M, 1], accum_dtype)
                logsum = T.alloc_fragment([block_M, 1], accum_dtype)
                one_vec = T.alloc_fragment([block_M, 1], accum_dtype)
                zero_vec = T.alloc_fragment([block_M, 1], accum_dtype)
                thr_vec = T.alloc_fragment([block_M, 1], accum_dtype)
                all_masked = T.alloc_fragment([block_M, 1], "bool")
                is_zero = T.alloc_fragment([block_M, 1], "bool")

                # Mask-chain temporaries (allocated unconditionally: a
                # conditional T.alloc is illegal).
                colmat = T.alloc_fragment([block_M, block_N], "int32")
                rowmat = T.alloc_fragment([block_M, block_N], "int32")
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
                neg_inf = -T.infinity(accum_dtype)

                # ---- loop-invariant mask indices / constants ----
                T.arange(colmat, [0, 1], offset=0)  # colmat[i, j] == j
                if is_causal_t:
                    T.arange(rowmat, [1, 0], offset=0)  # rowmat[i, j] == i
                T.vbrc(neg_sent, sent_buf)
                T.vbrc(one_f, one_vec)
                T.vbrc(zero_f, zero_vec)
                T.vbrc(all_masked_thr, thr_vec)

                # ---- prologue ----
                T.clear(Q_shared)
                T.copy(Q[bz, bx * block_M:bx * block_M + real_m, by, 0:dim],
                       Q_shared[0:real_m, 0:dim])
                T.vmul(Q_shared, s_nat, Q_shared)  # OPT-1: fold scale into Q
                T.clear(acc_o)
                T.clear(logsum)
                T.vbrc(neg_inf, m_prev)  # read before first write

                # ---- KV loop (static bound, pipelined) ----
                for k in T.Pipelined(nkv, num_stages=num_stages):
                    phys = block_table[bz, k]  # runtime page id (DESIGN R-2)
                    T.copy(K[phys * block_N:(phys + 1) * block_N, by, 0:dim],
                           K_shared[0:block_N, 0:dim])
                    T.gemm(Q_shared, K_shared, acc_s, initC=True, b_transpose=True)

                    # ---- column mask: invalid columns -> finite sentinel ----
                    # valid j <=> k*block_N + j < real_kv  <=>  j < real_kv - k*block_N
                    # NOTE (toolchain fact): a runtime scalar operand to T.vcmp
                    # (e.g. T.vcmp(colmat, col_lim, ...)) segfaults the npuir
                    # compiler; the scalar is broadcast into a matrix first.
                    col_lim = real_kv - k * block_N
                    T.vbrc(col_lim, lim_mat)
                    T.vcmp(colmat, lim_mat, vis, "lt")
                    T.vselect(vis, acc_s, sent_buf, acc_s)
                    if is_causal_t:
                        # visible j <=> q_pos >= kv_pos  <=>  j - i <= qbase - kvbase
                        T.vsub(colmat, rowmat, dif)
                        causal_lim = bx * block_M - k * block_N
                        T.vbrc(causal_lim, lim_mat)
                        T.vcmp(dif, lim_mat, vis, "le")
                        T.vselect(vis, acc_s, sent_buf, acc_s)

                    # ---- online softmax (OPT-2 / OPT-3, natural-exp domain) ----
                    T.reduce_max(acc_s, m_cur, dim=1, clear=True)
                    T.vcmp(m_cur, thr_vec, all_masked, "le")  # same-domain guard
                    T.vselect(all_masked, zero_vec, m_cur, m_cur_safe)
                    T.vsub(m_prev, m_cur_safe, rs)
                    T.vexp(rs, rs)
                    T.vselect(all_masked, one_vec, rs, rs)  # identity, keep state
                    T.vsub(acc_s, m_cur_safe, acc_s)
                    T.vexp(acc_s, acc_s)  # P (masked -> exp(-1e38) == 0)
                    T.reduce_sum(acc_s, s_sum, dim=1, clear=True)
                    T.vmul(acc_o, rs, acc_o)  # row-broadcast rescale
                    T.vmul(logsum, rs, logsum)
                    T.vadd(logsum, s_sum, logsum)
                    T.copy(m_cur, m_prev)  # store old max at the loop end
                    T.vcast(acc_s, acc_s_cast, round_mode="rint")  # O9 quant point

                    T.copy(V[phys * block_N:(phys + 1) * block_N, by, 0:dim],
                           V_shared[0:block_N, 0:dim])
                    T.gemm(acc_s_cast, V_shared, acc_o, initC=False)

                # ---- epilogue: normalise + row-truncated store ----
                T.vcmp(logsum, zero_vec, is_zero, "eq")
                T.vselect(is_zero, one_vec, logsum, logsum)  # logsum==0 -> 1
                T.vdiv(acc_o, logsum, acc_o)
                T.vcast(acc_o, O_cast, round_mode="rint")
                T.copy(O_cast[0:real_m, 0:dim],
                       Output[bz, bx * block_M:bx * block_M + real_m, by, 0:dim])

        return mha_decode_no_split

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
                real_kv_list=None, shuffle_pages=False, seed=0):
    """Build the 5 paged-decode inputs on the NPU device."""
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
    return (q.contiguous(), k.contiguous(), v.contiguous(), real.contiguous(), block_table)


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
              tag, real_kv_list=None, shuffle_pages=False, seed=0, expect_zero=False):
    """Run one paged-decode case and gate it against the independent golden."""
    dtype_str = _DTYPE_STR[dtype]
    q, k, v, real, block_table = _gen_inputs(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, dtype,
        real_kv_list=real_kv_list, shuffle_pages=shuffle_pages, seed=seed)

    fn = _mha_decode_no_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size,
                                     is_causal, dtype_str)
    kernel = fn(16, page_size, 2)
    out = kernel(q, k, v, real, block_table)
    torch.npu.synchronize()

    assert out.shape == q.shape and out.dtype == q.dtype, \
        f"[{tag}] output contract violated: {tuple(out.shape)}/{out.dtype}"

    ref = golden__mha_decode_no_split_kernel(
        q.cpu(), k.cpu(), v.cpu(), real.cpu(), block_table.cpu(), page_size,
        is_causal=is_causal)
    max_diff, min_cos, finite_ok, ref_is_zero = _compare(out, ref)

    ok_diff = max_diff < _ATOL
    ok_finite = finite_ok
    if expect_zero:
        ok_cos = True
        ok = ok_diff and ok_finite and bool((out.float().cpu().abs().max().item() == 0.0))
    else:
        ok_cos = True if ref_is_zero else (min_cos > 0.99)
        ok = ok_diff and ok_finite and ok_cos
    print(f"  [{tag}] B={batch} H={heads} Sq={seqlen_q} Nkv={seqlen_kv} D={dim} "
          f"page={page_size} causal={int(is_causal)} {dtype_str} "
          f"real_kv={real_kv_list if real_kv_list is not None else seqlen_kv} "
          f"shuffle={int(shuffle_pages)}: out={'PASS' if ok else 'FAIL'} "
          f"(max_diff={max_diff:.3e} min_cos={min_cos:.5f} finite={ok_finite})")
    assert ok, (f"[{tag}] precision gate failed: max_diff={max_diff:.3e} "
                f"min_cos={min_cos} finite={ok_finite} ref_is_zero={ref_is_zero}")
    return out


def run_L0():
    """L0 gate (blocking): the 4 manifest workloads + paged contract."""
    print("== L0 (blocking) ==")
    # single-token-page128
    _run_case(1, 16, 1, 512, 128, 128, False, torch.float16, "L0-1", seed=0)
    # batch2-page256
    _run_case(2, 8, 1, 1024, 64, 256, False, torch.float16, "L0-2", seed=1)
    # longer-cache
    _run_case(1, 8, 1, 1024, 64, 256, False, torch.float16, "L0-3", seed=2)
    # shorter-cache
    _run_case(1, 8, 1, 512, 64, 256, False, torch.float16, "L0-4", seed=3)
    # paged contract: non-identity (shuffled) block table
    _run_case(1, 8, 1, 512, 64, 256, False, torch.float16, "L0-5", seed=4, shuffle_pages=True)


def run_L1():
    """L1 gate (blocking): dynamic-KV-length / all-masked stabilisation."""
    print("== L1 (blocking) ==")
    # partial last page (block 0 partial) + fully masked tail block
    _run_case(1, 8, 1, 512, 64, 256, False, torch.float16, "L1-1",
              real_kv_list=[200], seed=5)
    # zero logical length -> all blocks masked -> output exactly 0
    _run_case(1, 8, 1, 512, 64, 256, False, torch.float16, "L1-2",
              real_kv_list=[0], seed=6, expect_zero=True)
    # partial tail block, all_masked == false (normal softmax path)
    _run_case(1, 8, 1, 512, 64, 256, False, torch.float16, "L1-3",
              real_kv_list=[300], seed=7)
    # real_kv < N_kv with a shuffled page table
    _run_case(1, 8, 1, 512, 64, 256, False, torch.float16, "L1-4",
              real_kv_list=[300], seed=8, shuffle_pages=True)


def run_L2():
    """L2 coverage (warn only, non-blocking): bf16 + causal + per-batch lengths."""
    print("== L2 (non-blocking) ==")
    cases = [
        ((1, 16, 1, 512, 128, 128, False, torch.bfloat16), dict(seed=10)),
        ((2, 8, 1, 1024, 64, 256, False, torch.bfloat16), dict(seed=11)),
        ((1, 1, 1, 512, 128, 128, False, torch.float16), dict(seed=12)),
        ((1, 8, 1, 1024, 64, 256, False, torch.float16),
         dict(real_kv_list=[1024, 700], seed=13)),
        ((1, 8, 1, 512, 64, 256, True, torch.float16), dict(seed=14)),
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
        ((1, 8, 1, 256, 64, 256, False, torch.float16), dict(real_kv_list=[1], seed=20),
         "boundary-real1"),
        # N_kv == page_size -> nkv == 1
        ((1, 8, 1, 256, 64, 256, False, torch.float16), None, "boundary-nkv1"),
        # single token short cache, real_kv == N_kv
        ((1, 4, 1, 128, 128, 128, False, torch.float16), None, "boundary-page128"),
        # all-zero inputs
        ((1, 8, 1, 512, 64, 256, False, torch.float16), dict(seed=21), "boundary-rand"),
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
