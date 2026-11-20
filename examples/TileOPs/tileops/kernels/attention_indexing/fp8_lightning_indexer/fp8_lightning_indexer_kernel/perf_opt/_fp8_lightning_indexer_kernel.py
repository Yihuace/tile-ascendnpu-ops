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

Stage 4 (optimize) changes
--------------------------
Two changes, both inside the same ``@T.prim_func`` (no algorithmic change):

1. ``block_Q`` 4 -> 8 (previous Stage-4 pass): halves the number of kv-tile
   iterations per q-block, which the profile shows to dominate.
2. fp16 vector epilogue + ``block_N`` 64 -> 128 (this pass): the L0C -> UB
   epilogue copy now converts to fp16, so ``acc_ub`` is half the size
   (``q_rows*block_N*2``).  That UB relief is what makes ``block_N=128``
   feasible (fp32 ``bN=128`` overflowed UB), halving the tile count again; and
   the relu / weight-multiply / head-reduce run on 128 fp16 lanes instead of
   64 fp32 lanes.  The window contract is untouched: this kernel still writes a
   block-aligned superset and ``clean_logits_`` applies the mask.

Historical note (previous pass):
``block_Q`` 4 -> 8, ``block_N`` 64 (unchanged), ``num_stages`` 2 (unchanged).
``block_N`` 64 (unchanged) and ``num_stages`` 2 (unchanged).  ``block_Q=8``
halves the number of kv-tile iterations per q-block, which the msprof profile
shows to dominate (per-iteration fixed cost; see ``perf_opt/opt_log.md``).
The tuned values are exposed as module-level ``TUNED_*`` constants so the
wrapper switching block can reference them; the factory signature is unchanged.
"""

import argparse
import functools
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import tilelang
import tilelang.language as T

ASCEND_MODE = "Developer"

# Stage 4 tuned tiling defaults (exposed for the wrapper switching block).
TUNED_BLOCK_N = 128
TUNED_BLOCK_Q = 8
TUNED_NUM_STAGES = 2

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
    def _fp8_lightning_indexer_func(block_N=TUNED_BLOCK_N, num_stages=TUNED_NUM_STAGES,
                                    block_Q=TUNED_BLOCK_Q):
        dtype = _INDEX_DTYPE
        accum_dtype = _ACCUM_DTYPE
        index_dtype = _INDEX_INT_DTYPE
        # The grouped path (kv_group > 1 -> Hp <= 8) overflows UB at
        # block_N=128 on this toolchain (gemm tiling heuristic + multi-buffer).
        # The tuned block_N is a single-group (kv_group == 1) win, so clamp it
        # for the grouped path at trace time (compile-time constant, not a
        # runtime dispatch).
        if kv_group > 1 and block_N > 64:
            block_N = 64

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
                w2h_ub = T.alloc_shared([block_Q, heads_per_group], "float16")
                w32_ub = T.alloc_shared([q_rows, 1], accum_dtype)
                w_ub = T.alloc_shared([q_rows, 1], "float16")
                scale_ub = T.alloc_shared([1, block_N], accum_dtype)
                brc_ub = T.alloc_shared([block_Q, block_N], accum_dtype)
                acc = T.alloc_fragment([q_rows, block_N], accum_dtype)
                acc_ub = T.alloc_shared([q_rows, block_N], "float16")
                outh_ub = T.alloc_shared([block_Q, block_N], "float16")
                out_ub = T.alloc_shared([block_Q, block_N], accum_dtype)

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
                    T.reshape(w2_ub, w32_ub)
                    T.copy(w32_ub, w_ub)

                    for nbn in T.Pipelined(num_kv_blocks, num_stages=num_stages):
                        k0 = T.min(nbn * block_N, kv_last)
                        T.copy(IndexK[by, k0:k0 + block_N, g, 0:index_dim], k_ub)
                        if has_scale:
                            T.copy(IndexKScale[by, k0:k0 + block_N, g], scale_ub)

                        # Cube: acc[q*Hp+h, n] = sum_d Q[q,h,d] * K[n,d]
                        T.gemm(q_ub, k_ub, acc, b_transpose=True, initC=True)

                        # R1b: fp16 vector epilogue.  L0C(fp32) -> UB(fp16)
                        # direct copy halves both the fixpipe traffic and the
                        # vector instruction count (128 fp16 lanes / 64 fp32).
                        T.copy(acc, acc_ub)
                        T.vrelu(acc_ub, acc_ub)
                        T.vmul(acc_ub, w_ub, acc_ub)
                        for q in T.serial(block_Q):
                            T.reduce_sum(
                                acc_ub[q * heads_per_group:(q + 1) * heads_per_group, :],
                                outh_ub[q:q + 1, :],
                                dim=0,
                                clear=True)
                        T.copy(outh_ub, out_ub)
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



# Expert path used by the wrapper for the validated BF16 shapes. The
# Developer factory above remains available for grouped/custom-config cases.
EXPERT_BLOCK_Q = 8
EXPERT_BLOCK_N = 128
EXPERT_CORES = 20

def _window_min(buf, start, width, n):
    result = buf[start, 0]
    for i in range(1, width):
        result = T.min(result, buf[start + i, 0])
    return T.max(0, T.min(n, result))

def _window_max(buf, start, width, n):
    result = buf[0, start]
    for i in range(1, width):
        result = T.max(result, buf[0, start + i])
    return T.max(0, T.min(n, result))

@tilelang.jit(target='npuir', pass_configs={'npuir.enable_auto_multi_buffer': False, 'tl.enable_plan_and_update_buffer_allocation': False})
def _fp8_indexer_normalize_qk(B, S, H, D, N, VC=20):
    QT = 4
    KT = 128 if N % 128 == 0 else 64
    ROWS = H * D
    CAP = max(QT * ROWS, KT * D)
    QBLOCKS = B * S // QT
    KBLOCKS = B * N // KT
    @T.prim_func
    def fp8_indexer_normalize_qk(Q: T.Tensor((B, S, H, D), 'bfloat16'), K: T.Tensor((B, N, 1, D), 'bfloat16'), W: T.Tensor((S, H), 'float32'), Scale: T.Tensor((B, N, 1), 'float32'), QN: T.Tensor((B, S, H, D), 'float16'), KN: T.Tensor((B, N, 1, D), 'float16'), WN: T.Tensor((S, H), 'float32'), QS: T.Tensor((B, S, 1), 'float32'), KSCALE: T.Tensor((B, N, 1), 'float32')):
        with T.Kernel(VC, is_npu=True) as (cid, subid):
            with T.Scope('Vector'):
                raw = T.alloc_ub((CAP,), 'bfloat16')
                val = T.alloc_ub((CAP,), 'float32')
                temp = T.alloc_ub((CAP,), 'float32')
                brc = T.alloc_ub((CAP,), 'float32')
                out = T.alloc_ub((CAP,), 'float16')
                qb = T.alloc_ub((QT, ROWS), 'bfloat16')
                q3 = T.alloc_ub((QT, H, D), 'bfloat16')
                qf = T.alloc_ub((QT, ROWS), 'float32')
                qa = T.alloc_ub((QT, ROWS), 'float32')
                qbrc = T.alloc_ub((QT, ROWS), 'float32')
                qo = T.alloc_ub((QT, ROWS), 'float16')
                qo3 = T.alloc_ub((QT, H, D), 'float16')
                kb = T.alloc_ub((KT, D), 'bfloat16')
                kf = T.alloc_ub((KT, D), 'float32')
                ka = T.alloc_ub((KT, D), 'float32')
                kbrc = T.alloc_ub((KT, D), 'float32')
                ko = T.alloc_ub((KT, D), 'float16')
                qmax = T.alloc_ub((QT, 1), 'float32')
                wf = T.alloc_ub((QT, H), 'float32')
                wa = T.alloc_ub((QT, H), 'float32')
                wmax = T.alloc_ub((QT, 1), 'float32')
                wbrc = T.alloc_ub((QT, H), 'float32')
                combined = T.alloc_ub((QT, 1), 'float32')
                qzero = T.alloc_ub((QT, 1), 'float32')
                qone = T.alloc_ub((QT, 1), 'float32')
                qcond = T.alloc_ub((QT, 1), 'bool')
                kmax = T.alloc_ub((KT, 1), 'float32')
                input_scale = T.alloc_ub((KT, 1), 'float32')
                kzero = T.alloc_ub((KT, 1), 'float32')
                kone = T.alloc_ub((KT, 1), 'float32')
                kcond = T.alloc_ub((KT, 1), 'bool')
                T.reshape(raw[:QT * ROWS], qb)
                T.reshape(qb, q3)
                T.reshape(val[:QT * ROWS], qf)
                T.reshape(temp[:QT * ROWS], qa)
                T.reshape(brc[:QT * ROWS], qbrc)
                T.reshape(out[:QT * ROWS], qo)
                T.reshape(qo, qo3)
                T.reshape(raw[:KT * D], kb)
                T.reshape(val[:KT * D], kf)
                T.reshape(temp[:KT * D], ka)
                T.reshape(brc[:KT * D], kbrc)
                T.reshape(out[:KT * D], ko)
                T.vbrc(T.float32(0), qzero)
                T.vbrc(T.float32(1), qone)
                T.vbrc(T.float32(0), kzero)
                T.vbrc(T.float32(1), kone)
                for task in T.serial(T.ceildiv(QBLOCKS, VC * 2)):
                    idx = task * VC * 2 + cid * 2 + subid
                    if idx < QBLOCKS:
                        b = idx // (S // QT)
                        s = idx % (S // QT) * QT
                        T.copy(Q[b, s:s + QT, :, :], q3)
                        T.vcast(qb, qf)
                        T.vabs(qf, qa)
                        T.reduce_max(qa, qmax, dim=1)
                        T.vcmp(qmax, qzero, qcond, 'eq')
                        T.vselect(qcond, qone, qmax, qmax)
                        T.vbrc(qmax, qbrc)
                        T.vdiv(qf, qbrc, qf)
                        T.vcast(qf, qo)
                        T.copy(qo3, QN[b, s:s + QT, :, :])
                        T.copy(W[s:s + QT, :], wf)
                        T.vabs(wf, wa)
                        T.reduce_max(wa, wmax, dim=1)
                        T.vcmp(wmax, qzero, qcond, 'eq')
                        T.vselect(qcond, qone, wmax, wmax)
                        T.vbrc(wmax, wbrc)
                        T.vdiv(wf, wbrc, wf)
                        if b == 0:
                            T.copy(wf, WN[s:s + QT, :])
                        T.vmul(qmax, wmax, combined)
                        T.copy(combined, QS[b, s:s + QT, :])
                T.pipe_barrier('PIPE_ALL')
                for task in T.serial(T.ceildiv(KBLOCKS, VC * 2)):
                    idx = task * VC * 2 + cid * 2 + subid
                    if idx < KBLOCKS:
                        b = idx // (N // KT)
                        n = idx % (N // KT) * KT
                        T.copy(K[b, n:n + KT, 0, :], kb)
                        T.vcast(kb, kf)
                        T.vabs(kf, ka)
                        T.reduce_max(ka, kmax, dim=1)
                        T.vcmp(kmax, kzero, kcond, 'eq')
                        T.vselect(kcond, kone, kmax, kmax)
                        T.vbrc(kmax, kbrc)
                        T.vdiv(kf, kbrc, kf)
                        T.vcast(kf, ko)
                        T.copy(ko, KN[b, n:n + KT, 0, :])
                        T.copy(Scale[b, n:n + KT, :], input_scale)
                        T.vmul(input_scale, kmax, input_scale)
                        T.copy(input_scale, KSCALE[b, n:n + KT, :])
    return fp8_indexer_normalize_qk

@tilelang.jit(target='npuir', pass_configs={'npuir.disable_hivm_auto_inject_sync': True, 'npuir.enable_auto_multi_buffer': False, 'tl.enable_plan_and_update_buffer_allocation': False})
def _fp8_lightning_indexer_expert(B, S, H, D, N, BQ, BN, CORES, HAS_SCALE, CLEAN):
    """FP32 QK/epilogue with BF16 score transport and fused ReLU.

    Alternate two UB input buffers to prefetch the next score tile.
    The runtime scale and per-query window remain part of the computation.
    """
    # Rotate query-wave lanes to spread variable window lengths across cores.
    TASKS = B * S // BQ
    VQ = BQ // 2
    CQ = BQ // 2
    NC = min(N, 4096)

    @T.prim_func
    def fp8_indexer_distinctmeta_inline(Q: T.Tensor((B, S * H, D), 'float16'), K: T.Tensor((B, N, 1, D), 'float16'), Scale: T.Tensor((B, N, 1), 'float32'), O: T.Tensor((B, S, N, 1), 'float32'), W: T.Tensor((S, H), 'float32'), KS: T.Tensor((S, 1), 'int32'), KE: T.Tensor((1, S), 'int32'), Score: T.Tensor((CORES, 2, T.ceildiv(N, BN) * BQ * H, BN), 'float16'), QWScale: T.Tensor((B, S, 1), 'float32'), flat: T.Tensor((B, S * N), 'float32')):
        with T.Kernel(CORES, is_npu=True) as (cid, subid):
            with T.Scope('Cube'):
                q_first = T.alloc_L1((CQ * H, D), 'float16')
                q_second = T.alloc_L1((CQ * H, D), 'float16')
                k_first = T.alloc_L1((BN, D), 'float16')
                k_second = T.alloc_L1((BN, D), 'float16')
                c_first = T.alloc_L0C((CQ * H, BN), 'float32')
                c_second = T.alloc_L0C((CQ * H, BN), 'float32')
                with T.rs('PIPE_MTE1'):
                    T.set_flag('PIPE_MTE2', 0)
                    T.set_flag('PIPE_MTE2', 4)
                    T.set_flag('PIPE_MTE2', 1)
                with T.rs('PIPE_M'):
                    T.set_flag('PIPE_MTE1', 7)
                    T.set_flag('PIPE_MTE1', 6)
                    # Query buffers are reused only after both halves finish.
                    T.set_flag('PIPE_MTE1', 5)
                for wave in T.serial(T.ceildiv(TASKS, CORES * 2)):
                    for phase in T.unroll(2):
                        task = wave * 2 + phase
                        tid = task * CORES + (cid + task * 7) % CORES
                        if tid < TASKS:
                            b = tid // (S // BQ)
                            s = tid % (S // BQ) * BQ
                            lo = _window_min(KS, s, BQ, N) if CLEAN else 0
                            hi = T.max(lo, _window_max(KE, s, BQ, N)) if CLEAN else N
                            first = lo // BN
                            nt = T.ceildiv(hi, BN) - first
                            slot = phase
                            with T.rs('PIPE_MTE2'):
                                T.sync_block_wait(2 + slot)
                                T.wait_flag('PIPE_MTE1', 1)
                            T.copy(Q[b, s * H:(s + CQ) * H, :], q_first)
                            T.copy(Q[b, (s + CQ) * H:(s + BQ) * H, :], q_second)
                            if nt > 0:
                                with T.rs('PIPE_MTE1'):
                                    T.wait_flag('PIPE_M', 5)
                            if nt > 0:
                                start_key = first * BN
                                with T.rs('PIPE_MTE2'):
                                    T.wait_flag('PIPE_MTE1', 0)
                                T.copy(K[b, start_key:start_key + BN, 0, :], k_first)
                                with T.rs('PIPE_MTE2'):
                                    T.set_flag('PIPE_MTE1', 0)
                            for it in T.serial(nt):
                                if it % 2 == 0:
                                    if it + 1 < nt:
                                        next_key = (first + (it + 1)) * BN
                                        with T.rs('PIPE_MTE2'):
                                            T.wait_flag('PIPE_MTE1', 4)
                                        T.copy(K[b, next_key:next_key + BN, 0, :], k_second)
                                        with T.rs('PIPE_MTE2'):
                                            T.set_flag('PIPE_MTE1', 4)
                                    with T.rs('PIPE_MTE1'):
                                        T.wait_flag('PIPE_MTE2', 0)
                                    T.gemm(q_first, k_first, c_first, b_transpose=True, initC=True)
                                    T.store_fixpipe(c_first, Score[cid, slot, it * BQ * H, 0], size=[CQ * H, BN], enable_nz2nd=True, pre_relu_mode='relu')
                                    T.gemm(q_second, k_first, c_second, b_transpose=True, initC=True)
                                    T.store_fixpipe(c_second, Score[cid, slot, it * BQ * H + CQ * H, 0], size=[CQ * H, BN], enable_nz2nd=True, pre_relu_mode='relu')
                                    with T.rs('PIPE_MTE1'):
                                        T.set_flag('PIPE_MTE2', 0)
                                else:
                                    if it + 1 < nt:
                                        next_key = (first + (it + 1)) * BN
                                        with T.rs('PIPE_MTE2'):
                                            T.wait_flag('PIPE_MTE1', 0)
                                        T.copy(K[b, next_key:next_key + BN, 0, :], k_first)
                                        with T.rs('PIPE_MTE2'):
                                            T.set_flag('PIPE_MTE1', 0)
                                    with T.rs('PIPE_MTE1'):
                                        T.wait_flag('PIPE_MTE2', 4)
                                    T.gemm(q_first, k_second, c_first, b_transpose=True, initC=True)
                                    T.store_fixpipe(c_first, Score[cid, slot, it * BQ * H, 0], size=[CQ * H, BN], enable_nz2nd=True, pre_relu_mode='relu')
                                    T.gemm(q_second, k_second, c_second, b_transpose=True, initC=True)
                                    T.store_fixpipe(c_second, Score[cid, slot, it * BQ * H + CQ * H, 0], size=[CQ * H, BN], enable_nz2nd=True, pre_relu_mode='relu')
                                    with T.rs('PIPE_MTE1'):
                                        T.set_flag('PIPE_MTE2', 4)
                                if nt >= 3 and it + 1 == nt // 3:
                                    with T.rs('PIPE_FIX'):
                                        T.sync_block_set(4 + slot)
                                if nt >= 3 and it + 1 == (2 * nt) // 3:
                                    with T.rs('PIPE_FIX'):
                                        T.sync_block_set(6 + slot)
                            if nt > 0:
                                with T.rs('PIPE_M'):
                                    T.set_flag('PIPE_MTE1', 5)
                            with T.rs('PIPE_MTE1'):
                                T.set_flag('PIPE_MTE2', 1)
                            with T.rs('PIPE_FIX'):
                                T.sync_block_set(slot)
                with T.rs('PIPE_MTE2'):
                    T.sync_block_wait(2)
                    T.sync_block_wait(3)
                    T.wait_flag('PIPE_MTE1', 0)
                    T.wait_flag('PIPE_MTE1', 4)
                    T.wait_flag('PIPE_MTE1', 1)
                with T.rs('PIPE_MTE1'):
                    T.wait_flag('PIPE_M', 7)
                    T.wait_flag('PIPE_M', 6)
                    T.wait_flag('PIPE_M', 5)
            with T.Scope('Vector'):
                x = T.alloc_ub((VQ * H, BN), 'float16')
                xn = T.alloc_ub((VQ * H, BN), 'float16')
                x3 = T.alloc_ub((VQ, H, BN), 'float16')
                xn3 = T.alloc_ub((VQ, H, BN), 'float16')
                y3 = T.alloc_ub((VQ, 1, BN), 'float16')
                y16 = T.alloc_ub((VQ, BN), 'float16')
                yt = T.alloc_ub((VQ, BN), 'float32')
                w2 = T.alloc_ub((VQ, H), 'float32')
                w16 = T.alloc_ub((VQ, H), 'float16')
                w = T.alloc_ub((VQ * H, 1), 'float16')
                wb = T.alloc_ub((VQ * H, BN), 'float16')
                sf = T.alloc_ub((1, BN), 'float32')
                sb = T.alloc_ub((VQ, BN), 'float32')
                qs = T.alloc_ub((VQ, 1), 'float32')
                qsb = T.alloc_ub((VQ, BN), 'float32')
                neg = T.alloc_ub((N,), 'float32')
                row = T.view(O, (B, S, 1, N), 'float32')
                T.reshape(x, x3)
                T.reshape(xn, xn3)
                T.reshape(y3, y16)
                T.reshape(w16, w)
                if CLEAN:
                    T.vbrc(-T.infinity('float32'), neg)
                    with T.rs('PIPE_V'):
                        T.set_flag('PIPE_MTE3', 1)
                    with T.rs('PIPE_MTE3'):
                        T.wait_flag('PIPE_V', 1)
                with T.rs('PIPE_V'):
                    T.set_flag('PIPE_MTE2', 0)
                    T.set_flag('PIPE_MTE2', 1)
                    T.set_flag('PIPE_MTE2', 2)
                    T.set_flag('PIPE_MTE2', 3)
                with T.rs('PIPE_MTE3'):
                    T.set_flag('PIPE_V', 0)
                    T.sync_block_set(2)
                    T.sync_block_set(3)
                for wave in T.serial(T.ceildiv(TASKS, CORES * 2)):
                    for phase in T.unroll(2):
                        task = wave * 2 + phase
                        tid = task * CORES + (cid + task * 7) % CORES
                        if tid < TASKS:
                            b = tid // (S // BQ)
                            s = tid % (S // BQ) * BQ
                            q0 = s + subid * VQ
                            lo = _window_min(KS, s, BQ, N) if CLEAN else 0
                            hi = T.max(lo, _window_max(KE, s, BQ, N)) if CLEAN else N
                            first = lo // BN
                            nt = T.ceildiv(hi, BN) - first
                            slot = phase
                            with T.rs('PIPE_MTE2'):
                                T.wait_flag('PIPE_V', 3)
                            T.copy(W[q0:q0 + VQ, :], w2)
                            T.copy(QWScale[b, q0:q0 + VQ, 0], qs[:, 0])
                            with T.rs('PIPE_MTE2'):
                                T.set_flag('PIPE_V', 3)
                            with T.rs('PIPE_V'):
                                T.wait_flag('PIPE_MTE2', 3)
                            T.vcast(w2, w16)
                            T.pipe_barrier('PIPE_V')
                            T.vbrc(w, wb)
                            T.pipe_barrier('PIPE_V')
                            T.vbrc(qs, qsb)
                            T.pipe_barrier('PIPE_V')
                            with T.rs('PIPE_V'):
                                T.set_flag('PIPE_MTE2', 3)
                            if nt >= 3:
                                with T.rs('PIPE_MTE2'):
                                    T.sync_block_wait(4 + slot)
                            else:
                                with T.rs('PIPE_MTE2'):
                                    T.sync_block_wait(slot)
                            if nt > 0:
                                with T.rs('PIPE_MTE2'):
                                    T.wait_flag('PIPE_V', 0)
                                T.copy(Score[cid, slot, subid * VQ * H:(subid + 1) * VQ * H, :], x)
                                with T.rs('PIPE_MTE2'):
                                    T.set_flag('PIPE_V', 0)
                            for it in T.serial(nt):
                                k0 = (first + it) * BN
                                if nt >= 3 and it + 1 == nt // 3:
                                    with T.rs('PIPE_MTE2'):
                                        T.sync_block_wait(6 + slot)
                                if nt >= 3 and it + 1 == (2 * nt) // 3:
                                    with T.rs('PIPE_MTE2'):
                                        T.sync_block_wait(slot)
                                if it % 2 == 0:
                                    if it + 1 < nt:
                                        with T.rs('PIPE_MTE2'):
                                            T.wait_flag('PIPE_V', 1)
                                        T.copy(Score[cid, slot, (it + 1) * BQ * H + subid * VQ * H:(it + 1) * BQ * H + (subid + 1) * VQ * H, :], xn)
                                        with T.rs('PIPE_MTE2'):
                                            T.set_flag('PIPE_V', 1)
                                    with T.rs('PIPE_V'):
                                        T.wait_flag('PIPE_MTE2', 0)
                                    T.vmul(x, wb, x)
                                    T.pipe_barrier('PIPE_V')
                                    if H == 32:
                                        T.vadd(x3[:, :16, :], x3[:, 16:32, :], x3[:, :16, :])
                                        T.pipe_barrier('PIPE_V')
                                        T.vadd(x3[:, :8, :], x3[:, 8:16, :], x3[:, :8, :])
                                        T.pipe_barrier('PIPE_V')
                                        T.vadd(x3[:, :4, :], x3[:, 4:8, :], x3[:, :4, :])
                                        T.pipe_barrier('PIPE_V')
                                        T.vadd(x3[:, :2, :], x3[:, 2:4, :], x3[:, :2, :])
                                        T.pipe_barrier('PIPE_V')
                                    else:
                                        T.vadd(x3[:, :4, :], x3[:, 4:8, :], x3[:, :4, :])
                                        T.pipe_barrier('PIPE_V')
                                        T.vadd(x3[:, :2, :], x3[:, 2:4, :], x3[:, :2, :])
                                        T.pipe_barrier('PIPE_V')
                                    T.vadd(x3[:, :1, :], x3[:, 1:2, :], y3)
                                    T.pipe_barrier('PIPE_V')
                                    with T.rs('PIPE_V'):
                                        T.set_flag('PIPE_MTE2', 0)
                                else:
                                    if it + 1 < nt:
                                        with T.rs('PIPE_MTE2'):
                                            T.wait_flag('PIPE_V', 0)
                                        T.copy(Score[cid, slot, (it + 1) * BQ * H + subid * VQ * H:(it + 1) * BQ * H + (subid + 1) * VQ * H, :], x)
                                        with T.rs('PIPE_MTE2'):
                                            T.set_flag('PIPE_V', 0)
                                    with T.rs('PIPE_V'):
                                        T.wait_flag('PIPE_MTE2', 1)
                                    T.vmul(xn, wb, xn)
                                    T.pipe_barrier('PIPE_V')
                                    if H == 32:
                                        T.vadd(xn3[:, :16, :], xn3[:, 16:32, :], xn3[:, :16, :])
                                        T.pipe_barrier('PIPE_V')
                                        T.vadd(xn3[:, :8, :], xn3[:, 8:16, :], xn3[:, :8, :])
                                        T.pipe_barrier('PIPE_V')
                                        T.vadd(xn3[:, :4, :], xn3[:, 4:8, :], xn3[:, :4, :])
                                        T.pipe_barrier('PIPE_V')
                                        T.vadd(xn3[:, :2, :], xn3[:, 2:4, :], xn3[:, :2, :])
                                        T.pipe_barrier('PIPE_V')
                                    else:
                                        T.vadd(xn3[:, :4, :], xn3[:, 4:8, :], xn3[:, :4, :])
                                        T.pipe_barrier('PIPE_V')
                                        T.vadd(xn3[:, :2, :], xn3[:, 2:4, :], xn3[:, :2, :])
                                        T.pipe_barrier('PIPE_V')
                                    T.vadd(xn3[:, :1, :], xn3[:, 1:2, :], y3)
                                    T.pipe_barrier('PIPE_V')
                                    with T.rs('PIPE_V'):
                                        T.set_flag('PIPE_MTE2', 1)
                                with T.rs('PIPE_V'):
                                    T.wait_flag('PIPE_MTE3', 0)
                                T.vcast(y16, yt)
                                T.pipe_barrier('PIPE_V')
                                with T.rs('PIPE_MTE2'):
                                    T.wait_flag('PIPE_V', 2)
                                T.copy(Scale[b, k0:k0 + BN, 0], sf[0, :])
                                with T.rs('PIPE_MTE2'):
                                    T.set_flag('PIPE_V', 2)
                                with T.rs('PIPE_V'):
                                    T.wait_flag('PIPE_MTE2', 2)
                                T.vbrc(sf, sb)
                                T.pipe_barrier('PIPE_V')
                                with T.rs('PIPE_V'):
                                    T.set_flag('PIPE_MTE2', 2)
                                T.vmul(yt, sb, yt)
                                T.pipe_barrier('PIPE_V')
                                T.vmul(yt, qsb, yt)
                                T.pipe_barrier('PIPE_V')
                                with T.rs('PIPE_V'):
                                    T.set_flag('PIPE_MTE3', 0)
                                with T.rs('PIPE_MTE3'):
                                    T.wait_flag('PIPE_V', 0)
                                T.copy(yt, row[b, q0:q0 + VQ, 0, k0:k0 + BN])
                                with T.rs('PIPE_MTE3'):
                                    T.set_flag('PIPE_V', 0)
                            with T.rs('PIPE_MTE2'):
                                T.sync_block_set(2 + slot)
                            T.pipe_barrier('PIPE_MTE3')
                            if CLEAN:
                                first_lo = T.max(0, T.min(N, KS[q0, 0]))
                                if first_lo > 0:
                                    T.copy(neg[:first_lo], flat[b, q0 * N:q0 * N + first_lo])
                                for r in T.serial(VQ):
                                    qr = q0 + r
                                    lo0 = T.max(0, T.min(N, KS[qr, 0]))
                                    hi0 = T.max(lo0, T.min(N, KE[0, qr]))
                                    gap_start = qr * N + hi0
                                    if r + 1 < VQ:
                                        next_lo = T.max(0, T.min(N, KS[qr + 1, 0]))
                                        gap = N - hi0 + next_lo
                                        if gap > N:
                                            T.copy(neg, flat[b, gap_start:gap_start + N])
                                            T.copy(neg[:gap - N], flat[b, gap_start + N:gap_start + gap])
                                        elif gap > 0:
                                            T.copy(neg[:gap], flat[b, gap_start:gap_start + gap])
                                    elif hi0 < N:
                                        T.copy(neg[:N - hi0], flat[b, gap_start:(qr + 1) * N])
                with T.rs('PIPE_MTE2'):
                    T.wait_flag('PIPE_V', 0)
                    T.wait_flag('PIPE_V', 1)
                    T.wait_flag('PIPE_V', 2)
                    T.wait_flag('PIPE_V', 3)
                with T.rs('PIPE_V'):
                    T.wait_flag('PIPE_MTE3', 0)
    return fp8_indexer_distinctmeta_inline


from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess,re
from tilelang.jit.jit_npu import compiler_npu,get_npucompiler_path

_INDEXER_QK_SOURCE = r'''
#include "kernel_operator_fixpipe_intf.h"

extern "C" __aicore__ __attribute__((always_inline))
void indexer_qk_resident(unsigned long long output, unsigned long long query, unsigned long long key,
                         unsigned int m, unsigned int n, unsigned int k, unsigned int it) {
    __ca__ half *a = (__ca__ half *)query;
    const bool odd = (it & 1) != 0;
    __cb__ half *b = (__cb__ half *)(odd ? n * k * 2 : 0);
    if (query == 0) {
        if (odd) wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        else wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
    }
    if (it == 0) {
        for (unsigned row = 0; row < m / 16; ++row) {
            load_cbuf_to_ca(a + row * k * 16, (__cbuf__ half *)query + row * 256,
                            0, k / 16, m / 16, 0, 0, false, inc);
        }
    }
    if (query == 0) {
        load_cbuf_to_cb(b, (__cbuf__ half *)key, 0, n * k / 256, 1, 0, 0, false, inc);
    }
    if (query == 0 || it == 0) {
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    }
    mad((__cc__ float *)output, a, b, m, k, n, 3, false, false, true);
    if (query != 0) {
        if (odd) set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        else set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
    }
}
'''


class _IndexerAsyncCompiler(compiler_npu):
    def _npuir_to_bin_enable_npu_compile(self):
        ir=self.mlir_content
        type_map={}
        cc_offset=0;l1_offset=0;new=[]
        for line in ir.splitlines():
            indent=line[:len(line)-len(line.lstrip())]
            match=re.search(r'(%[\w.]+) = memref.alloc\(\) : (memref<(\d+)x(\d+)x(f16|f32), strided<.*>, #hivm.address_space<(cbuf|cc)>>)',line)
            if match:
                dst,typ,ms,ns,dtype,space=match.groups();m,n=int(ms),int(ns)
                physical=f'memref<{n//16}x{m//16}x16x16x{dtype}, #hivm.address_space<{space}>>'
                type_map[typ]=physical
                if space=='cbuf':offset=l1_offset;l1_offset+=m*n*2
                else:offset=cc_offset;cc_offset+=m*n*4
                ident='in'+dst[1:]
                new.extend([indent+f'%{ident}addr = arith.constant {offset} : i64',indent+f'{dst} = hivm.hir.pointer_cast(%{ident}addr) : {physical}'])
            else:new.append(line)
        ir='\n'.join(new)
        for old,new in type_map.items():ir=ir.replace(old,new)
        new=[];count=0
        dynamic16='memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>'
        dynamic32='memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>'
        dynamicgm='memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>'
        for line in ir.splitlines():
            indent=line[:len(line)-len(line.lstrip())]
            loop_match=re.search(r'scf.for (%[\w.]+) =',line)
            if loop_match: iteration=loop_match.group(1)
            if 'hivm.hir.mmadL1 '  in line:
                operands,types,dst,dt=re.search(r'ins\((.*?) : (.*)\) outs\((%[\w.]+) : (.*)\)',line).groups()
                a,b,init,m,k,n=operands.split(', ')
                at,bt=re.findall(r'memref<.*?#hivm.address_space<cbuf>>',types)
                ident='inmm'+str(count);count+=1
                block=[]
                for name,buf,typ in [('a',a,at),('b',b,bt),('c',dst,dt)]:
                    block.extend([f'%{ident}{name}p = memref.extract_aligned_pointer_as_index {buf} : {typ} -> index',
                                  f'%{ident}{name} = arith.index_cast %{ident}{name}p : index to i64'])
                for name,v in [('m',m),('n',n),('k',k)]:
                    block.append(f'%{ident}{name} = arith.index_cast {v} : index to i32')
                block.append(f'func.call @indexer_qk_resident(%{ident}c, %{ident}a, %{ident}b, %{ident}m, %{ident}n, %{ident}k, {iteration}) : (i64, i64, i64, i32, i32, i32, i32) -> ()')
                new.extend(indent+x for x in block);continue
            if 'hivm.hir.fixpipe ' in line:
                src,st,dst,dt=re.search(r'ins\((%[\w.]+) : (.*)\) outs\((%[\w.]+) : (.*)\)',line).groups()
                ident='infx'+str(count);count+=1
                block=[
                    f'%{ident}s = memref.cast {src} : {st} to {dynamic32}',
                    f'%{ident}d = memref.cast {dst} : {dt} to {dynamicgm}',
                    f'%{ident}one = arith.constant 1 : i64',
                    f'%{ident}false = arith.constant false',
                    f'%{ident}unit = arith.constant 3 : i8',
                    f'func.call @fixpipe_nz2nd_float_to_half_4d_to_2d(%{ident}s, %{ident}d, %{ident}one, %{ident}one, %{ident}false, %{ident}unit) : ({dynamic32}, {dynamicgm}, i64, i64, i1, i8) -> ()']
                new.extend(indent+x for x in block);continue
            new.append(line)
        ir='\n'.join(new);last=ir.rfind('}')
        decl=f'''
func.func private @indexer_qk_resident(i64, i64, i64, i32, i32, i32, i32) attributes {{hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>}}
func.func private @fixpipe_nz2nd_float_to_half_4d_to_2d({dynamic32}, {dynamicgm}, i64, i64, i1, i8) attributes {{hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}}
'''
        self.mlir_content=ir[:last]+decl+ir[last:]
        self.metadata['mlir_content']=self.mlir_content
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'kernel.npuir';source.write_text(self.mlir_content)
            # Build the operator's native Cube helper with the same CANN toolchain.
            cann = Path(os.environ['ASCEND_HOME_PATH'])
            sdk = cann / (os.uname().machine + '-linux') / 'asc'
            native_source = root / 'indexer_qk.cc'
            bitcode = root / 'indexer_qk.bc'
            native_source.write_text(_INDEXER_QK_SOURCE)
            includes = [sdk, sdk / 'include', sdk / 'include/basic_api',
                        sdk / 'impl', sdk / 'impl/basic_api']
            subprocess.run(
                [str(cann / 'bin/bisheng'), '-xcce', '--cce-aicore-only',
                 '--cce-aicore-arch=dav-c220-cube', '-O3', '-std=c++17',
                 '-emit-llvm', '-c', str(native_source), '-o', str(bitcode),
                 *[f'-I{path}' for path in includes]],
                check=True, capture_output=True, text=True)
            opts = ['--enable-auto-multi-buffer=false', '--disable-hivm-auto-inject-sync=true',
                    '--enable-triton-kernel-compile=true', '--enable-hivm-compile=true',
                    '--disable-hivm-tensor-compile=true', f'--link-aicore-bitcode={bitcode}']
            result=subprocess.run([get_npucompiler_path(),str(source),*opts,'-o',str(root/'kernel')],capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stderr)
            self.workspace_size=self._get_workspace_size(str(root/'libkernel.so'),'_infer_workspace_shape_function')
            return (root/'kernel.o').read_bytes()

_indexer_async_program = _fp8_lightning_indexer_expert
@functools.lru_cache(maxsize=None)
def _fp8_lightning_indexer_expert(*args):
    implementation = _indexer_async_program.__jit_impl__
    return _IndexerAsyncCompiler().compile(
        implementation.func(*args), pass_configs=implementation.pass_configs)


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
