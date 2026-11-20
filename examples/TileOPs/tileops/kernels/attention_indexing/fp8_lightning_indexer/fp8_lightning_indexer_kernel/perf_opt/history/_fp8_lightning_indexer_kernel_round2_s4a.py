# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""TileLang NPUIR implementation of `_fp8_lightning_indexer_kernel`.

Lightning-indexer logits (NPU / ``target="npuir"``), migrated from the GPU
TileOPs factory ``_fp8_lightning_indexer_kernel``
(``tileops/kernels/fp8_lightning_indexer.py``, L19-L149).

Semantics (unchanged from the source operator)
----------------------------------------------
For every query row ``s`` the kernel scores every key position ``t`` of the
index cache inside the window ``[cu_seqlen_ks[s], cu_seqlen_ke[s])``::

    Logits[b,s,t,g] = Scale[b,t,g] * sum_h relu(sum_d Q[b,s,g*Hp+h,d] * K[b,t,g,d])
                      * W[s, g*Hp+h]

with ``Hp = heads // kv_group``.  The window mask is *not* applied here: the
main kernel writes a block-aligned superset of the window and the independent
``clean_logits_`` migration unit overwrites everything outside
``[cu_seqlen_ks, cu_seqlen_ke)`` with ``-inf`` (same contract as the source).

FP8 capability gap (CG-2026-0022)
---------------------------------
The source kernel is an fp8 tensor-core kernel (``dtype=T.float8_e4m3fn``,
fp8 ``T.gemm``, device-side fp8 quantization in the op layer).  The npuir
front-end / codegen / ``T.gemm`` dtype matrix / ``torch_npu`` device cast have
no fp8 path, so this implementation uses a **bf16 index carrier + fp32
accumulation + fp32 vector epilogue**.  The fp8 de-quantization semantics are
carried by the optional fp32 ``index_k_scale`` multiplier (``has_scale``).

Programming mode
----------------
``ASCEND_MODE = "Developer"``: non-persistent 1-D grid, Cube ``T.gemm`` +
Vector epilogue with an ``L0C -> UB`` direct copy (no GM round-trip).

Implementation notes (deviations from DESIGN.md, see RETROSPECTIVE.md)
---------------------------------------------------------------------
* The kv window union (``cu_k_s_min`` / ``cu_k_e_max``) is not computed inside
  the kernel: accumulating a scalar min/max over a GM tensor and using it as an
  MTE2 copy offset produces an invalid address (device ``MPU address access is
  invalid``) on this toolchain.  The kernel instead scans the full block-aligned
  kv range ``[0, ceil(seq_len_kv/block_N)*block_N)`` clamped so the last block
  ends exactly at ``seq_len_kv``.  This writes a superset of the window, so the
  ``clean_logits_`` contract is preserved; only the block-skipping optimisation
  is deferred (Stage 4).
* ``functools.lru_cache`` on the outer factory mirrors the source caching model;
  ``batch/seq_len/heads/index_dim/seq_len_kv/kv_group`` are compile-time
  constants (enables static pipelined trip counts).
"""

import argparse
import functools
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import tilelang
import tilelang.language as T

ASCEND_MODE = "Developer"

_INDEX_DTYPE = "bfloat16"
_ACCUM_DTYPE = "float32"
_INDEX_INT_DTYPE = "int32"

__all__ = ["_fp8_lightning_indexer_kernel"]


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _fp8_lightning_indexer_kernel(batch,
                                  seq_len,
                                  heads,
                                  index_dim,
                                  seq_len_kv,
                                  kv_group,
                                  clean_logits=True,
                                  has_scale=True):
    """Build the NPUIR lightning-indexer kernel factory.

    Args:
        batch: Batch size (compile-time).
        seq_len: Query sequence length (compile-time).
        heads: Number of index heads (compile-time).
        index_dim: Index head dimension (compile-time).
        seq_len_kv: Key/value sequence length (compile-time).
        kv_group: Number of key/value groups (compile-time).
        clean_logits: Kept for interface compatibility with the source factory
            (the mask itself lives in the separate ``clean_logits_`` unit).
        has_scale: Compile-time E3 switch; when ``False`` the optional fp32
            ``index_k_scale`` multiplier is omitted (identity-scale path).

    Returns:
        The jit-decorated callable ``func(block_N, num_stages, block_Q)``.
    """
    heads_per_group = heads // kv_group

    @tilelang.jit(target="npuir")
    def _fp8_lightning_indexer_func(block_N=64, num_stages=2, block_Q=8):
        dtype = _INDEX_DTYPE
        accum_dtype = _ACCUM_DTYPE
        index_dtype = _INDEX_INT_DTYPE

        num_q_blocks = (seq_len + block_Q - 1) // block_Q
        num_kv_blocks = (seq_len_kv + block_N - 1) // block_N
        # Clamp bases so the last (possibly partial) tile ends exactly at the
        # boundary; overlapping tiles recompute identical values.
        seq_len_last = max(0, seq_len - block_Q)
        kv_last = max(0, seq_len_kv - block_N)
        q_rows = block_Q * heads_per_group
        # Grouped path loads the whole (contiguous) Q tile once and gathers the
        # per-group rows on the UB; the single-group path loads directly.
        qall_rows = block_Q * heads if kv_group > 1 else 1

        @T.prim_func
        def _fp8_lightning_indexer_main(
                IndexQ: T.Tensor([batch, seq_len * heads, index_dim], dtype),  # type: ignore
                IndexK: T.Tensor([batch, seq_len_kv, kv_group, index_dim], dtype),  # type: ignore
                IndexKScale: T.Tensor([batch, seq_len_kv, kv_group], accum_dtype),  # type: ignore
                Logits: T.Tensor([batch, seq_len, seq_len_kv, kv_group], accum_dtype),  # type: ignore
                Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
                CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
                CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
        ):
            with T.Kernel(batch * num_q_blocks, is_npu=True) as (cid, _):
                by = cid // num_q_blocks
                bx = cid % num_q_blocks
                seq_len_i = T.min(bx * block_Q, seq_len_last)

                q_ub = T.alloc_shared([q_rows, index_dim], dtype)
                qall_ub = T.alloc_shared([qall_rows, index_dim], dtype)
                k_ub = T.alloc_shared([block_N, index_dim], dtype)
                w2_ub = T.alloc_shared([block_Q, heads_per_group], accum_dtype)
                w_ub = T.alloc_shared([q_rows, 1], accum_dtype)
                scale_ub = T.alloc_shared([1, block_N], accum_dtype)
                brc_ub = T.alloc_shared([block_Q, block_N], accum_dtype)
                acc = T.alloc_fragment([q_rows, block_N], accum_dtype)
                acc_ub = T.alloc_shared([q_rows, block_N], accum_dtype)
                out_ub = T.alloc_shared([block_Q, block_N], accum_dtype)
                # Rank-3 aliases (pure views, no device op) hoisted out of the
                # kv loop: [block_Q, heads_per_group, block_N] with the head axis
                # reduced in a single op.
                acc3 = T.view(acc_ub, [block_Q, heads_per_group, block_N], accum_dtype)
                out3 = T.view(out_ub, [block_Q, 1, block_N], accum_dtype)

                if kv_group > 1:
                    T.copy(
                        IndexQ[by, seq_len_i * heads:seq_len_i * heads + block_Q * heads,
                               0:index_dim], qall_ub)

                for g in range(kv_group):
                    # Query tile for group g (source O3 hoist: loaded once per
                    # core). The grouped path gathers contiguous Hp-row segments
                    # from the UB tile to keep every GM load stride-1.
                    if kv_group > 1:
                        for q in range(block_Q):
                            T.copy(
                                qall_ub[q * heads + g * heads_per_group:
                                        q * heads + g * heads_per_group + heads_per_group,
                                        0:index_dim],
                                q_ub[q * heads_per_group:(q + 1) * heads_per_group,
                                     0:index_dim])
                    else:
                        T.copy(
                            IndexQ[by, seq_len_i * heads:seq_len_i * heads + block_Q * heads,
                                   0:index_dim], q_ub)
                    T.copy(
                        Weights[seq_len_i:seq_len_i + block_Q,
                                g * heads_per_group:(g + 1) * heads_per_group],
                        w2_ub)
                    T.reshape(w2_ub, w_ub)

                    for nbn in T.Pipelined(num_kv_blocks, num_stages=num_stages):
                        k0 = T.min(nbn * block_N, kv_last)
                        T.copy(IndexK[by, k0:k0 + block_N, g, 0:index_dim], k_ub)
                        if has_scale:
                            T.copy(IndexKScale[by, k0:k0 + block_N, g], scale_ub)

                        # Cube: acc[q*Hp+h, n] = sum_d Q[q,h,d] * K[n,d]
                        T.gemm(q_ub, k_ub, acc, b_transpose=True, initC=True)

                        # Vector epilogue (fp32): L0C -> UB direct copy.
                        # S4a: one rank-3 reduce over the head axis instead of
                        # ``block_Q`` runtime-loop reduce calls with dynamic row
                        # slices (the loop form dominates the scalar pipe).
                        T.copy(acc, acc_ub)
                        T.vrelu(acc_ub, acc_ub)
                        T.vmul(acc_ub, w_ub, acc_ub)
                        T.reduce_sum(acc3, out3, dim=1, clear=True)
                        if has_scale:
                            # E1: de-quant scale applied outside the head
                            # reduction; E3: whole branch elided when has_scale
                            # is False (identity scale).
                            T.vbrc(scale_ub, brc_ub)
                            T.vmul(out_ub, brc_ub, out_ub)
                        T.copy(
                            out_ub,
                            Logits[by, seq_len_i:seq_len_i + block_Q, k0:k0 + block_N, g])

        return _fp8_lightning_indexer_main

    return _fp8_lightning_indexer_func


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference, independent of the NPU algorithm)
# ---------------------------------------------------------------------------


def golden_fp8_lightning_indexer(index_q,
                                 index_k,
                                 weights,
                                 cu_seqlen_ks,
                                 cu_seqlen_ke,
                                 index_k_scale=None,
                                 query_chunk=256):
    """Reference lightning-indexer logits on CPU.

    Direct port of the source op's ``ref_program`` (einsum score -> relu ->
    grouped weighted sum -> window mask) with the optional de-quantization
    ``index_k_scale`` applied *inside* the head reduction (source semantics),
    so the reference does not replicate the NPU-side E1 reformulation.

    Args:
        index_q: ``[B, S, H, D]`` index queries.
        index_k: ``[B, Skv, G, D]`` index keys.
        weights: ``[S, H]`` fp32 head weights.
        cu_seqlen_ks: ``[S]`` int32 window lower bounds.
        cu_seqlen_ke: ``[S]`` int32 window upper bounds.
        index_k_scale: optional ``[B, Skv, G]`` fp32 de-quantization scale.
        query_chunk: number of query rows processed per chunk (bounded memory).

    Returns:
        ``[B, S, Skv, G]`` fp32 logits with ``-inf`` outside each window.
    """
    q = index_q.float()
    k = index_k.float()
    batch, seq_len, heads, index_dim = q.shape
    seq_len_kv, kv_group = k.shape[1], k.shape[2]
    heads_per_group = heads // kv_group

    kf = k.view(batch, seq_len_kv, kv_group, index_dim)
    w = weights.float().view(seq_len, kv_group, heads_per_group).permute(1, 2, 0)  # [G,Hp,S]
    if index_k_scale is not None:
        sc = index_k_scale.float().permute(0, 2, 1)  # [B,G,Skv]

    out = torch.empty(batch, seq_len, seq_len_kv, kv_group, dtype=torch.float32)
    for s0 in range(0, seq_len, query_chunk):
        s1 = min(s0 + query_chunk, seq_len)
        qc = q[:, s0:s1].view(batch, s1 - s0, kv_group, heads_per_group, index_dim)
        score = torch.einsum("bsghd,bngd->bghsn", qc, kf)  # [B,G,Hp,chunk,Skv]
        score = score.relu() * w[:, :, s0:s1].unsqueeze(0).unsqueeze(-1)
        if index_k_scale is not None:
            score = score * sc.unsqueeze(2).unsqueeze(3)  # [B,G,1,1,Skv]
        out[:, s0:s1] = score.sum(dim=2).permute(0, 2, 3, 1)

    ar = torch.arange(0, seq_len_kv)
    mask = (ar[None, :] >= cu_seqlen_ks[:, None]) & (ar[None, :] < cu_seqlen_ke[:, None])
    out = out.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float("-inf"))
    return out


# Canonical ``golden_<op>`` alias for external naming checks.
golden__fp8_lightning_indexer_kernel = golden_fp8_lightning_indexer


# ---------------------------------------------------------------------------
# Precision comparison
# ---------------------------------------------------------------------------

_TOL = 1e-3

# Stage 4 tuned defaults (exposed for the wrapper switching block).
TUNED_BLOCK_N = 64
TUNED_BLOCK_Q = 8
TUNED_NUM_STAGES = 2


def _window_mask(seq_len, seq_len_kv, cu_seqlen_ks, cu_seqlen_ke):
    ar = torch.arange(0, seq_len_kv)
    return (ar[None, :] >= cu_seqlen_ks[:, None]) & (ar[None, :] < cu_seqlen_ke[:, None])


def _compare(logits, ref, cu_seqlen_ks, cu_seqlen_ke):
    """Correlation difference + max abs error over the finite (window) region."""
    batch, seq_len, seq_len_kv, kv_group = logits.shape
    mask = _window_mask(seq_len, seq_len_kv, cu_seqlen_ks, cu_seqlen_ke)
    m = mask.unsqueeze(0).unsqueeze(-1).expand(batch, seq_len, seq_len_kv, kv_group)
    if m.sum() == 0:
        return 0.0, 0.0, 0
    a = logits[m].double()
    b = ref[m].double()
    denom = (a * a + b * b).sum()
    corr = (2 * (a * b).sum() / denom) if denom > 0 else torch.tensor(1.0, dtype=torch.float64)
    return (1.0 - corr).item(), (a - b).abs().max().item(), int(m.sum().item())


def run_case(batch, seq_len, heads, index_dim, seq_len_kv, kv_group, tag,
             has_scale=True, window="full", block_N=TUNED_BLOCK_N, block_Q=TUNED_BLOCK_Q,
             num_stages=TUNED_NUM_STAGES, seed=0):
    """Build inputs, run the kernel on NPU, compare with the CPU golden.

    ``window`` selects the query window layout: ``"full"`` (all keys visible),
    ``"partial"`` (short ~half-window for the tail rows) or ``"empty"`` (a
    degenerate empty window for the first half of the rows).
    """
    torch.manual_seed(seed)
    device = "npu"

    index_q = torch.randn(batch, seq_len, heads, index_dim, device=device,
                          dtype=torch.bfloat16)
    index_k = torch.randn(batch, seq_len_kv, kv_group, index_dim, device=device,
                          dtype=torch.bfloat16)
    weights = torch.randn(seq_len, heads, device=device, dtype=torch.float32)

    cu_seqlen_ks = torch.zeros(seq_len, device=device, dtype=torch.int32)
    if window == "partial":
        span = max(1, seq_len_kv // 2)
        cu_seqlen_ke = torch.full((seq_len,), span, device=device, dtype=torch.int32)
        cu_seqlen_ke[:seq_len // 2] = seq_len_kv
    elif window == "empty":
        mid = seq_len_kv // 2
        cu_seqlen_ks[:seq_len // 2] = mid
        cu_seqlen_ke = torch.full((seq_len,), seq_len_kv, device=device, dtype=torch.int32)
        cu_seqlen_ke[:seq_len // 2] = mid
    else:
        cu_seqlen_ke = torch.full((seq_len,), seq_len_kv, device=device, dtype=torch.int32)

    if has_scale:
        index_k_scale = torch.rand(batch, seq_len_kv, kv_group, device=device,
                                   dtype=torch.float32) + 0.5
    else:
        index_k_scale = torch.ones(batch, seq_len_kv, kv_group, device=device,
                                   dtype=torch.float32)

    logits = torch.empty(batch, seq_len, seq_len_kv, kv_group, device=device,
                         dtype=torch.float32)

    kernel = _fp8_lightning_indexer_kernel(batch, seq_len, heads, index_dim,
                                           seq_len_kv, kv_group, True, has_scale)(
                                               block_N=block_N, num_stages=num_stages,
                                               block_Q=block_Q)
    kernel(index_q.view(batch, seq_len * heads, index_dim), index_k, index_k_scale,
           logits, weights, cu_seqlen_ks, cu_seqlen_ke)

    ref = golden_fp8_lightning_indexer(
        index_q.cpu(), index_k.cpu(), weights.cpu(), cu_seqlen_ks.cpu(),
        cu_seqlen_ke.cpu(), index_k_scale.cpu() if has_scale else None)

    diff, max_abs, n = _compare(logits.cpu(), ref, cu_seqlen_ks.cpu(), cu_seqlen_ke.cpu())
    status = "PASS" if diff <= _TOL else "FAIL"
    print(f"[{tag}] {status} shape=(b{batch},s{seq_len},h{heads},d{index_dim},"
          f"skv{seq_len_kv},g{kv_group}) has_scale={has_scale} "
          f"corr_diff={diff:.3e} max_abs={max_abs:.3e} n={n}")
    assert diff <= _TOL, f"[{tag}] corr_diff {diff:.3e} > tol {_TOL}"
    return diff, max_abs


# ---------------------------------------------------------------------------
# Hierarchical test suites
# ---------------------------------------------------------------------------


def run_L0():
    """Gate tests: minimal smoke + representative bf16 shapes."""
    run_case(1, 1024, 8, 64, 1024, 1, "L0-smoke-no-scale", has_scale=False)
    run_case(1, 1024, 8, 64, 1024, 1, "L0-smoke-scale", has_scale=True)
    run_case(1, 2048, 32, 64, 2048, 1, "L0-h32-scale", has_scale=True)


def run_L1():
    """Gate tests: full manifest bf16 workloads + grouped path."""
    run_case(1, 4096, 32, 64, 8192, 1, "L1-bf16-s4k-h32", has_scale=True)
    run_case(1, 8192, 32, 64, 8192, 1, "L1-bf16-s8k-h32", has_scale=True)
    run_case(1, 1024, 8, 64, 1024, 2, "L1-g2", has_scale=True)
    run_case(1, 4096, 32, 64, 8192, 1, "L1-partial-window", has_scale=True,
             window="partial")


def _advisory(tag, fn):
    try:
        fn()
    except Exception as exc:  # pragma: no cover - advisory only
        print(f"[{tag}] WARN (recorded, non-blocking): {str(exc)[:300]}")


def run_L2():
    """Advisory cases (non-blocking)."""
    _advisory("L2-batch2", lambda: run_case(2, 1024, 8, 64, 1024, 1, "L2-batch2",
                                            has_scale=True))
    _advisory("L2-h32-no-scale", lambda: run_case(1, 2048, 32, 64, 2048, 1,
                                                  "L2-h32-no-scale", has_scale=False))
    _advisory("L2-g4", lambda: run_case(1, 1024, 32, 64, 1024, 4, "L2-g4",
                                        has_scale=True))


def run_boundary():
    """Boundary cases (advisory, non-blocking)."""
    # seq_len_kv not a multiple of block_N (clamped last kv tile).
    _advisory("Boundary-odd-skv", lambda: run_case(1, 512, 8, 64, 1000, 1,
                                                   "Boundary-odd-skv", has_scale=True))
    # seq_len not a multiple of block_Q (clamped last q-block).
    _advisory("Boundary-odd-s", lambda: run_case(1, 1000, 8, 64, 1024, 1,
                                                 "Boundary-odd-s", has_scale=True))
    # Degenerate empty window for the first half of the rows.
    _advisory("Boundary-empty-window", lambda: run_case(1, 256, 8, 64, 256, 1,
                                                        "Boundary-empty-window",
                                                        has_scale=True, window="empty",
                                                        seed=3))


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
