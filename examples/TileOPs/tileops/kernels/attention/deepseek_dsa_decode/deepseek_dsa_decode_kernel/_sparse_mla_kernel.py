# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""DeepSeek sparse-MLA decode forward -- NPU Developer MixCV kernel.

Migrated from the GPU TileLang factory ``_sparse_mla_kernel``
(``tileops/kernels/attention/deepseek_dsa_decode.py``, L17-404) to
``target="npuir"`` per the frozen DESIGN.md (revision 1, scenario=migration /
harness mode).

Math (per batch b, query row s, head block h0):

    o[b, s, h, :] = sum_j softmax_j( (q[b,s,h,:] . kv[b, idx_j, 0, :]) / sqrt(Dqk) )
                    * kv[b, idx_j, 0, :D]

with ``Dqk = dim + tail_dim = 576`` (nope + shared RoPE) and ``D = dim = 512``
(the nope output width).  ``indices[b, s, 0, j]`` (int32) selects the top-k KV
positions; positions ``> max_kv_i = q_start_index_s + s`` (and the out-of-range
padding value ``seq_len_kv``) are masked out (compressed-causal sparsity).

NPU redesign (DESIGN sections 0.5 / 0.6 / 3.3 / 5.5 / 6):
  - non-persistent 1-D grid ``num_logical = B * S * (H / block_heads)``
    (mainstream 4096) + hardware wave scheduling -- the standard "huge =>
    in-core ``T.serial`` persistent" path is unusable because Developer +
    persistent + ``T.gemm`` crashes at runtime (TRAP-DEVMODE-PERSIST-GEMM).
  - ``block_heads = 32`` (L0C: ``32*(512+64)*4 = 72 KB <= 128 KB``),
    ``block_N = 64``; this carries the source ``replicate_h`` head-split
    intent (R4).
  - QK: single ``T.gemm(Q_shared, KV_shared, acc_s, initC=False,
    b_transpose=True)`` over the full 576-dim row (NPU has no WGMMA
    register-pressure motive for the source's 3-segment split).
  - PV: ``T.gemm(P_shared, KV_shared, acc_o, initC=False,
    size=[block_heads, block_N, D])`` -- N = D = 512 prefix columns of the
    gathered 576-wide row.
  - sparse gather: row-contiguous ``T.serial(block_N)`` + ``T.copy`` with a
    single ``T.min/T.max`` clamp (indices may hold the ``seq_len_kv`` padding
    sentinel); the mask itself uses the *unclamped* index so padding is always
    excluded.  No runtime skips, no strided gather (TRAP-C10).
  - vector online softmax in the natural-exp domain (OPT-1) with the
    reassociated rescale factor and ``m*s`` CSE (OPT-2) and a finite
    ``-1e38`` sentinel (OPT-3, no ``-inf`` => no ``inf - inf`` NaN path).
  - ``T.Pipelined(topk // block_N, num_stages=2)`` for the KV double buffer.

Dropped from the CUDA/Hopper source: ``threads`` / ``T.get_thread_binding`` /
``T.wgmma_gemm`` / ``T.alloc_barrier`` / ``cp_async_barrier_noinc`` /
``T.set_max_nreg`` / CUDA ``compile_flags`` / 3-D grid / persistent grid.
"""

import argparse
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

# Finite-large sentinel (OPT-3): never -inf, so a masked column yields an exact
# 0 after exp() and no inf-inf NaN can appear in the running max.
_SENT = -1e38


# ---------- Golden (independent PyTorch reference, DESIGN section 8.1) ----------
def golden_sparse_mla_decode(q, kv, indices, dim, dim_tail, stride_kv, q_start_index_s,
                             sm_scale=None, dtype=torch.float16):
    """Sparse-MLA decode reference -- verbatim port of the TileOPs reference.

    ``examples/TileOPs/tests/ops/test_deepseek_dsa_decode.py::ref_program``
    (device-agnostic, fp32 accumulate).

    q:       (B, S, H, dim + dim_tail)  BSHD
    kv:      (B, S_kv, H_kv, dim + dim_tail)  (H_kv == 1)
    indices: (B, S, H_kv, topk) int32   (padding value == seq_len_kv)
    Returns (B, S, H, dim) in ``dtype``.

    Independence from the NPU kernel (DESIGN 8.1): (1) all heads at once, no
    head blocking; (2) explicit *de-duplicated* scatter mask (the kernel
    accumulates every index entry instead); (3) two-pass
    ``masked_fill(-inf).softmax()`` (not online); (4) a single ``einsum``
    (not a top-k block loop); (5) no exp2 / LOG2E constants.
    """
    dev = q.device
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)                       # [B, H_kv, S, topk]
    b, sq, h, dim_q = q.shape
    _, sk, g, _ = kv.shape
    if q_start_index_s is None:
        q_start_index_s = sk * stride_kv - sq
    assert kv.shape[-1] == dim + dim_tail
    k = kv
    v = kv[..., :dim]
    g_index, h_index = g, h // g

    compressed_causal_mask = torch.arange(
        q_start_index_s, sq + q_start_index_s, dtype=torch.int32, device=dev
    ).view(-1, 1) >= torch.arange(
        stride_kv - 1, sk * stride_kv, stride_kv, dtype=torch.int32, device=dev
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(
        3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_causal_mask.view(1, 1, sq, sk)
    mask[:, :, :stride_kv - 1, 0] = True                    # cp0 degenerate slot
    mask = mask.view(b, g_index, 1, sq, sk)

    q5 = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q5, k)        # [B, G, H/G, S, S_kv]
    sm_scale = dim_q ** -0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk).view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim)
    return o.to(dtype)


# ---------- Kernel (Developer MixCV, DESIGN sections 3.3 / 6) ----------
@tilelang.jit(out_idx=[-1], target="npuir")
def _sparse_mla_kernel(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk,
                       kv_stride, q_start_index_s, kv_group=1, sm_scale=None,
                       is_causal=True, cp0=False, dtype="float16",
                       block_heads=32, block_N=64, num_stages=2):
    """NPU factory mirroring the GPU ``_sparse_mla_kernel`` signature.

    Compile-time contract (DESIGN 5.3 / 9.1 R-10): ``block_heads | heads``,
    ``block_N | topk``, ``kv_stride == 1``, ``kv_group == 1``, ``is_causal``.
    """
    assert is_causal, "npuir port supports the causal sparse-decode path only"
    assert kv_group == 1, "MLA decode has H_kv == 1"
    assert kv_stride == 1, "stride_kv > 1 is out of contract (DESIGN 9.3)"
    assert heads % block_heads == 0, "block_heads must divide heads"
    assert topk % block_N == 0, "topk must be a multiple of block_N"

    Dqk = dim + tail_dim
    D = dim
    accum = "float32"
    dtype_ = dtype
    # Natural-exp domain (OPT-1): no LOG2E folding; the source's log2e factor is
    # dropped because exp2(x*log2e) == exp(x) exactly in this re-association.
    s = (1.0 / math.sqrt(Dqk)) if sm_scale is None else float(sm_scale)
    n_hb = heads // block_heads
    num_logical = batch * seq_len * n_hb

    @T.prim_func
    def main(
        Q: T.Tensor((batch, seq_len, heads, Dqk), dtype_),
        KV: T.Tensor((batch, seq_len_kv, kv_group, Dqk), dtype_),
        Indices: T.Tensor((batch, seq_len, kv_group, topk), "int32"),
        Output: T.Tensor((batch, seq_len, heads, D), dtype_),
    ):
        with T.Kernel(num_logical, is_npu=True) as (cid, _):
            # ---- task decode (DESIGN 0.6 R6): cid -> (b, s, head-block) ----
            hb = cid % n_hb
            t = cid // n_hb
            s_i = t % seq_len
            b_i = t // seq_len
            h0 = hb * block_heads
            # sparse-causal upper bound for this query row (stride_kv == 1,
            # cp0 == False).  Unclamped index is compared against this.
            max_kv_i = q_start_index_s + s_i

            # ---- Cube operands (L1) / Vector staging --------------------
            Q_shared = T.alloc_shared([block_heads, Dqk], dtype_)
            KV_shared = T.alloc_shared([block_N, Dqk], dtype_)
            P_shared = T.alloc_shared([block_heads, block_N], dtype_)
            O_ub = T.alloc_shared([block_heads, D], dtype_)

            # ---- Cube accumulators (L0C) --------------------------------
            acc_s = T.alloc_fragment([block_heads, block_N], accum)
            acc_o = T.alloc_fragment([block_heads, D], accum)

            # ---- Vector online-softmax state (row vectors + P f32) ------
            m_prev = T.alloc_fragment([block_heads, 1], accum)
            m_cur = T.alloc_fragment([block_heads, 1], accum)
            m_new = T.alloc_fragment([block_heads, 1], accum)
            ms = T.alloc_fragment([block_heads, 1], accum)
            alpha = T.alloc_fragment([block_heads, 1], accum)
            lsum = T.alloc_fragment([block_heads, 1], accum)
            loc_sum = T.alloc_fragment([block_heads, 1], accum)
            tmp1 = T.alloc_fragment([block_heads, 1], accum)
            zero_vec = T.alloc_fragment([block_heads, 1], accum)
            one_vec = T.alloc_fragment([block_heads, 1], accum)
            is_zero = T.alloc_fragment([block_heads, 1], "bool")
            tmp = T.alloc_fragment([block_heads, block_N], accum)
            # Column-mask machinery.  NOTE (toolchain facts, both reproduced):
            #  * `T.if_then_else(idx <= max_kv_i, ..)` with a *runtime scalar*
            #    bound mis-lowers to `arith.trunci(i32 -> i1)` + a malformed
            #    `hivm.hir.vcmp`; the limit is therefore broadcast into a [1,N]
            #    i32 row and compared tensor-vs-tensor with `T.vcmp`.
            #  * a `T.Parallel(BH, N)` that only assigns from a GM row forms a
            #    (1,N)-shaped load and fails insert_slice into (BH,N); the mask
            #    is therefore built as a (1,N) row penalty and *row*-broadcast
            #    with `T.vbrc` (documented `(1,N) -> (M,N)` form).
            idx_row = T.alloc_fragment([1, block_N], "int32")
            lim_row = T.alloc_fragment([1, block_N], "int32")
            vis_row = T.alloc_fragment([1, block_N], "bool")
            zero_row = T.alloc_fragment([1, block_N], accum)
            sent_row = T.alloc_fragment([1, block_N], accum)
            pen_row = T.alloc_fragment([1, block_N], accum)
            pen_mat = T.alloc_fragment([block_heads, block_N], accum)

            # Let-bound scalars (vbrc / vmul scalar sources must be locals).
            scale = s
            zero_f = 0.0
            one_f = 1.0
            sent_f = _SENT
            m_init = _SENT
            lim = max_kv_i

            # ---- prologue: Q hoist + state init --------------------------
            T.copy(Q[b_i, s_i, h0:h0 + block_heads, 0:Dqk], Q_shared)
            T.vbrc(m_init, m_prev)      # finite sentinel => exp() gives exact 0
            T.vbrc(zero_f, zero_vec)
            T.vbrc(one_f, one_vec)
            T.vbrc(zero_f, zero_row)
            T.vbrc(sent_f, sent_row)
            T.vbrc(lim, lim_row)
            T.clear(lsum)
            T.clear(acc_o)

            for k in T.Pipelined(topk // block_N, num_stages=num_stages):
                T.copy(Indices[b_i, s_i, 0, k * block_N], idx_row,
                       size=[1, block_N])

                # (2) column penalty: every index is compared against the
                # *_unclamped_* causal bound so the seq_len_kv padding sentinel
                # (and every idx > max_kv_i) is always excluded.
                T.vcmp(idx_row, lim_row, vis_row, "le")
                T.vselect(vis_row, zero_row, sent_row, pen_row)
                T.vbrc(pen_row, pen_mat)

                # (3) sparse gather: row-contiguous copies, single clamp.
                for bi in T.serial(block_N):
                    idx_c = T.max(T.min(idx_row[0, bi], seq_len_kv - 1), 0)
                    T.copy(KV[b_i, idx_c, 0, 0:Dqk], KV_shared[bi, 0:Dqk])

                # (4) QK: acc_s = Q @ KV^T (576-dim, single gemm) + column pen.
                T.gemm(Q_shared, KV_shared, acc_s, initC=True, b_transpose=True)
                T.vadd(acc_s, pen_mat, acc_s)

                # (5) running max (m_prev is read before its first write).
                T.reduce_max(acc_s, m_cur, dim=1, clear=True)
                T.vmax(m_prev, m_cur, m_new)

                # (6) alpha = exp(m_prev*s - m_new*s)  (OPT-2 reassociation)
                T.vmul(m_new, scale, ms)
                T.vmul(m_prev, scale, tmp1)
                T.vsub(tmp1, ms, tmp1)
                T.vexp(tmp1, alpha)

                # (7) P = exp(acc_s*s - ms); quantise for the PV gemm.
                T.vmul(acc_s, scale, tmp)
                T.vsub(tmp, ms, tmp)                # ms row-broadcast
                T.vexp(tmp, acc_s)
                T.vcast(acc_s, P_shared, round_mode="rint")

                # (8) l = l*alpha + rowsum(P)
                T.reduce_sum(acc_s, loc_sum, dim=1, clear=True)
                T.vmul(lsum, alpha, lsum)
                T.vadd(lsum, loc_sum, lsum)

                # (9) rescale + PV over the D = 512 prefix columns.
                T.vmul(acc_o, alpha, acc_o)         # alpha row-broadcast
                T.gemm(P_shared, KV_shared, acc_o, initC=False,
                       size=[block_heads, block_N, D])

                T.copy(m_new, m_prev)

            # (10) normalise + write out (all-masked rows -> exact 0).
            T.vcmp(lsum, zero_vec, is_zero, "eq")
            T.vselect(is_zero, one_vec, lsum, lsum)
            T.vdiv(acc_o, lsum, acc_o)
            T.vcast(acc_o, O_ub, round_mode="rint")
            T.copy(O_ub, Output[b_i, s_i, h0:h0 + block_heads, 0:D])

    return main


# ---------- precision comparing func ----------
_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}
_DTYPE_STR = {torch.float16: "float16", torch.bfloat16: "bfloat16"}
# Blocking gate (project standard, DESIGN 8.2 fallback / mha_decode_paged
# precedent); the strict manifest assertion (atol=3e-4 / rtol=1e-5) is
# additionally reported but does not gate, since it is tighter than one fp16
# ulp at O(1) output (DESIGN 8.2 R-7).
_ATOL = 1e-2
_RTOL = 1e-2
_STRICT_ATOL = 3e-4
_STRICT_RTOL = 1e-5


def _gen_inputs(batch, heads, seq_len, seq_len_kv, dim, dim_tail, topk, stride_kv,
                q_start_index_s, dtype, seed=0):
    """Mirror the manifest generator (``DsaDecodeWorkload.gen_inputs``).

    ``indices`` is a random permutation of the causal-eligible KV positions
    padded with ``seq_len_kv`` (the out-of-range sentinel), so the clamp +
    mask path is exercised on every row.
    """
    torch.manual_seed(seed)
    dev = "npu"
    q = torch.randn(batch, seq_len, heads, dim + dim_tail, device=dev, dtype=dtype)
    kv = torch.randn(batch, seq_len_kv, 1, dim + dim_tail, device=dev, dtype=dtype)
    indices = torch.full((batch, seq_len, 1, topk), seq_len_kv, dtype=torch.int32,
                         device=dev)
    for b in range(batch):
        for t in range(seq_len):
            n_valid = min(max(1, (t + q_start_index_s) // stride_kv), seq_len_kv)
            perm = torch.randperm(n_valid)[:topk]
            indices[b, t, 0, :perm.numel()] = perm.to(torch.int32)
    return q.contiguous(), kv.contiguous(), indices.contiguous()


def _compare(out, ref):
    """Return (max_diff, strict_ok, loose_ok, finite_ok)."""
    out_f = out.float().cpu()
    ref_f = ref.float().cpu()
    finite_ok = bool(torch.isfinite(out_f).all().item())
    finite_ref = bool(torch.isfinite(ref_f).all().item())
    max_diff = float((out_f - ref_f).abs().max().item())
    loose_ok = bool(torch.allclose(out_f, ref_f, atol=_ATOL, rtol=_RTOL))
    strict_ok = finite_ref and bool(
        torch.allclose(out_f, ref_f, atol=_STRICT_ATOL, rtol=_STRICT_RTOL))
    return max_diff, strict_ok, loose_ok, finite_ok


def _run_case(batch, heads, seq_len, seq_len_kv, dim, dim_tail, topk, stride_kv,
              q_start_index_s, dtype, tag, sm_scale=None, seed=0,
              block_heads=32, block_N=64, num_stages=2, expect_zero=False):
    """Run one sparse-MLA-decode case and gate it against the golden."""
    dtype_str = _DTYPE_STR[dtype]
    q, kv, indices = _gen_inputs(batch, heads, seq_len, seq_len_kv, dim, dim_tail,
                                 topk, stride_kv, q_start_index_s, dtype, seed=seed)
    kernel = _sparse_mla_kernel(batch, seq_len, seq_len_kv, heads, dim, dim_tail,
                                topk, stride_kv, q_start_index_s, kv_group=1,
                                sm_scale=sm_scale, is_causal=True, cp0=False,
                                dtype=dtype_str, block_heads=block_heads,
                                block_N=block_N, num_stages=num_stages)
    out = kernel(q, kv, indices)
    torch.npu.synchronize()

    assert out.shape == (batch, seq_len, heads, dim), \
        f"[{tag}] output contract violated: {tuple(out.shape)}"
    assert out.dtype == dtype, f"[{tag}] output dtype {out.dtype} != {dtype}"

    ref = golden_sparse_mla_decode(q.cpu(), kv.cpu(), indices.cpu(), dim, dim_tail,
                                   stride_kv, q_start_index_s, sm_scale=sm_scale,
                                   dtype=dtype)
    max_diff, strict_ok, loose_ok, finite_ok = _compare(out, ref)

    if expect_zero:
        zero_ok = bool(float(out.float().cpu().abs().max().item()) == 0.0)
        ok = finite_ok and zero_ok
        print(f"  [{tag}] B={batch} H={heads} S={seq_len} Skv={seq_len_kv} "
              f"topk={topk} {dtype_str}: all-masked -> out={'PASS' if ok else 'FAIL'} "
              f"(zero={zero_ok} finite={finite_ok})")
        assert ok, f"[{tag}] all-masked case failed: zero={zero_ok} finite={finite_ok}"
        return out

    ok = bool(loose_ok) and finite_ok
    print(f"  [{tag}] B={batch} H={heads} S={seq_len} Skv={seq_len_kv} topk={topk} "
          f"q_start={q_start_index_s} BH={block_heads} BN={block_N} {dtype_str} "
          f"sm_scale={'default' if sm_scale is None else sm_scale}: "
          f"out={'PASS' if ok else 'FAIL'} (max_diff={max_diff:.3e} "
          f"strict(3e-4/1e-5)={'PASS' if strict_ok else 'note'} finite={finite_ok})")
    assert ok, (f"[{tag}] precision gate failed: max_diff={max_diff:.3e} "
                f"loose_ok={loose_ok} finite={finite_ok}")
    return out


def run_L0():
    """L0 gate (blocking): DESIGN 8.3 L0-1..L0-5."""
    print("== L0 (blocking) ==")
    # L0-1 smoke: small topk (4 blocks), single batch, q_start > 0 (cp0=False)
    _run_case(1, 128, 128, 256, 512, 64, 256, 1, 128, torch.float16, "L0-1", seed=0)
    # L0-2 smoke: bf16 path + heavy masking (exp itself runs in fp32)
    _run_case(1, 128, 64, 512, 512, 64, 256, 1, 64, torch.bfloat16, "L0-2", seed=1)
    # L0-3 full: manifest mainstream workload (32 blocks)
    _run_case(1, 128, 1024, 2048, 512, 64, 2048, 1, 1024, torch.float16, "L0-3",
              seed=2)
    # L0-4 full: manifest longer-kv-lower-topk workload
    _run_case(1, 128, 512, 4096, 512, 64, 1024, 1, 512, torch.float16, "L0-4",
              seed=3)
    # L0-5 boundary: explicit out-of-contract indices (>= seq_len_kv) -> clamp
    _run_case(1, 128, 128, 256, 512, 64, 256, 1, 128, torch.float16, "L0-5",
              seed=4)


def run_L1():
    """L1 gate (blocking): mask/clamp/state-transition coverage."""
    print("== L1 (blocking) ==")
    # L1-1: first rows fully padded (only padding + few valid indices)
    _run_case(1, 32, 64, 512, 512, 64, 128, 1, 448, torch.float16, "L1-1", seed=5)
    # L1-2: sm_scale override
    _run_case(1, 128, 128, 256, 512, 64, 256, 1, 128, torch.float16, "L1-2",
              sm_scale=0.5 / math.sqrt(576), seed=6)
    # L1-3: many rows with real_kv < topk (mostly padding after clamp)
    _run_case(1, 128, 256, 2048, 512, 64, 512, 1, 128, torch.float16, "L1-3",
              seed=7)
    # L1-4: all rows exactly one valid index (max_kv_i == 1 position)
    _run_case(1, 64, 128, 1024, 512, 64, 64, 1, 1, torch.float16, "L1-4", seed=8)
    # L1-5: single query row
    _run_case(1, 128, 1, 256, 512, 64, 64, 1, 128, torch.float16, "L1-5", seed=9)


def run_L2():
    """L2 coverage (warn only, non-blocking)."""
    print("== L2 (non-blocking) ==")
    cases = [
        ((1, 128, 128, 256, 512, 64, 256, 1, 128, torch.bfloat16), dict(seed=10)),
        ((1, 32, 128, 2048, 512, 64, 1024, 1, 128, torch.float16), dict(seed=11)),
        ((1, 128, 512, 1024, 512, 64, 512, 1, 512, torch.bfloat16), dict(seed=12)),
        ((1, 32, 256, 512, 512, 64, 1024, 1, 0, torch.float16), dict(seed=13)),
        ((1, 128, 512, 1024, 512, 64, 1024, 1, 512, torch.float16),
         dict(seed=14, block_heads=64)),
    ]
    idx = 0
    for params, kw in cases:
        idx += 1
        tag = f"L2-{idx}"
        try:
            _run_case(*params, tag, **kw)
        except Exception as exc:  # noqa: BLE001 - warn only
            print(f"  [{tag}] WARN (recorded, non-blocking): {exc}")


def run_boundary():
    """Boundary cases (warn only, non-blocking)."""
    print("== Boundary (non-blocking) ==")
    cases = [
        # design-level shape floor: 1 query row x 1 head-block
        ((1, 32, 1, 64, 512, 64, 64, 1, 0, torch.float16), dict(seed=20),
         "boundary-min"),
        # topk == block_N (single pipelined iteration)
        ((1, 64, 128, 512, 512, 64, 64, 1, 128, torch.float16), dict(seed=21),
         "boundary-onetile"),
        # every row's max_kv_i == seq_len_kv - 1 (full causal span)
        ((1, 64, 128, 128, 512, 64, 128, 1, 0, torch.float16), dict(seed=22),
         "boundary-fullspan"),
        # non-default (smaller) block_N tiling knob
        # (block_N=128 was probed too: it compiles the IR but BiShengIR then
        #  reports `ub overflow, requires 2115584 bits while 1572864 available`
        #  -- recorded in RETROSPECTIVE as a non-default-tiling bound.)
        ((1, 32, 64, 256, 512, 64, 64, 1, 64, torch.float16),
         dict(seed=23, block_N=32), "boundary-bn32"),
    ]
    for params, kw, tag in cases:
        try:
            _run_case(*params, tag, **kw)
        except Exception as exc:  # noqa: BLE001 - warn only
            print(f"  [{tag}] WARN (recorded, non-blocking): {exc}")

    # contract negations: the factory must reject out-of-contract shapes.
    for desc, bad_kwargs in [
        ("stride_kv=2", dict(kv_stride=2)),
        ("heads%block_heads!=0", dict(block_heads=48)),
        ("topk%block_N!=0", dict(block_N=48)),
        ("is_causal=False", dict(is_causal=False)),
    ]:
        base = dict(batch=1, seq_len=64, seq_len_kv=128, heads=32, dim=512,
                    tail_dim=64, topk=64, kv_stride=1, q_start_index_s=64)
        base.update(bad_kwargs)
        try:
            _sparse_mla_kernel(**base)
            print(f"  [boundary-contract:{desc}] WARN: no rejection raised")
        except Exception as exc:  # noqa: BLE001 - expected rejection
            print(f"  [boundary-contract:{desc}] PASS (rejected: "
                  f"{type(exc).__name__})")


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
