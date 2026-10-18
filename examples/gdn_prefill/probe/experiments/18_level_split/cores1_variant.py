# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""GDN prefill pipeline, function 2/6: ``prefill_blocksolve_A_bthd`` (NPU/Expert).

For every 64-row chunk of the packed sequence the delta rule needs

    S_sys[i, j] = beta_i * exp(g_i - g_j) * (k_i . k_j)     (i > j, strictly lower)
    A           = (I + S_sys)^{-1}                          (unit lower triangular)

NPU algorithm (DESIGN.md section 0.6 R1, log-doubling on the Cube):

    N = -S_sys        (strictly lower => N^64 = 0)
    A = prod_{i=0..5} (I + N^{2^i})       T <- T + P*T,  P <- P*P

i.e. 13 gemms per task (1 Gram + 1 identity seed + 12 chain gemms) instead of the
source's row-serial 16x16 block forward substitution.  One Cube scope runs the
Gram and the doubling chain, one Vector scope turns the Gram into ``N``; the two
engines are pipelined by six flags over GM slots (DESIGN.md section 7).

Tail chunk semantics (DESIGN.md section 0.6 R3): rows past the sequence end get
``beta = 0`` exactly (R3-b, per task) and the on-chip operands are zeroed once
per block (R3-a), so their row of ``N`` is *exactly* 0, hence their row of ``A``
is ``e_i`` -- the same answer the source's ``clamp + beta=0`` produces.  ``A`` is
written to a chunk-aligned ``A_pad`` and the wrapper returns the
``[:total_tokens]`` view, so the padded rows are invisible.

Programming mode: **Expert** (manual L1/L0C/UB allocation, manual flag sync).
Run: ``python prefill_blocksolve_A_bthd.py --level all``.
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # must assign: .bashrc exports Developer

import argparse

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
import tilelang
import tilelang.language as T

from tileops.utils.ascend_mode import scoped_ascend_mode

ASCEND_MODE = "Expert"

CHUNK = 64          # chunk rows (chunk_size is part of the operator contract)
DIM_K_CHOICES = (64, 128)
LOGICAL_CORES = 20  # NPUUtils.get().get_aicore_num() == 20 on Ascend910B3 (measured)
# Flag ids: f_g = F_G + slot, f_s = F_S + slot, f_a = F_A + slot  => 6 flags total.
F_G, F_S, F_A = 0, 2, 4


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference; fp64 direct solve, independent of the NPU algorithm)
# ---------------------------------------------------------------------------
def golden_blocksolve_A_bthd(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = CHUNK,
    uniform_seq_len: int = 0,
    num_sequences: int = 1,
) -> torch.Tensor:
    """``A = (I + S_sys)^{-1}`` per chunk, fp64, straight from the mathematical definition.

    ``k``: ``[1, T, key_head, dim_k]``; ``g``: ``[1, T, head]`` fp32;
    ``beta``: ``[1, T, head]``.  Returns ``[1, T, head, chunk_size]`` fp64.

    Rows past a sequence end repeat the last real row for ``k``/``g`` (the source's
    clamp) and take ``beta = 0``, so their ``S_sys`` row is zero.  Only the rows that
    exist are written; the padded rows of the caller's ``A_pad`` are not part of the
    mathematical answer.
    """
    total_tokens, key_head, dim_k = k.shape[1], k.shape[2], k.shape[3]
    head = g.shape[2]
    group = head // key_head
    seq_len = uniform_seq_len if uniform_seq_len else total_tokens // num_sequences
    per_seq = (seq_len + chunk_size - 1) // chunk_size
    eye_c = torch.eye(chunk_size, dtype=torch.float64)
    strict = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), -1)
    khids = torch.arange(head) // group
    A = torch.zeros(1, total_tokens, head, chunk_size, dtype=torch.float64)
    for seq in range(num_sequences):
        s0 = seq * seq_len
        end = s0 + seq_len
        for c in range(per_seq):
            base = s0 + c * chunk_size
            valid = min(chunk_size, end - base)
            rows = torch.clamp(torch.arange(base, base + chunk_size), max=end - 1)
            kk = k[0][rows][:, khids, :].permute(1, 0, 2).double()   # [head, 64, dim_k]
            gg = g[0][rows].transpose(0, 1).double()                 # [head, 64]
            bb = beta[0][rows].transpose(0, 1).double().clone()      # [head, 64]
            bb[:, valid:] = 0.0                                      # source: beta = 0
            G = torch.matmul(kk, kk.transpose(1, 2))                 # [head, 64, 64]
            S = torch.where(
                strict,
                bb[:, :, None] * torch.exp(gg[:, :, None] - gg[:, None, :]) * G,
                torch.zeros((), dtype=torch.float64),
            )                                                        # [head, 64, 64]
            sub = S[:, :valid, :valid]
            rhs = eye_c[:valid, :valid].expand(head, valid, valid)
            sol = torch.linalg.solve(eye_c[:valid, :valid] + sub, rhs)
            # Columns >= valid are 0: N's columns past the end are zero too, so the
            # 64-wide row of a short chunk carries the valid x valid inverse on the left.
            A[0, base:base + valid, :, 0:valid] = sol.permute(1, 0, 2)
    return A


def _blocked_inverse_fp64(s: torch.Tensor) -> torch.Tensor:
    """fp64 re-implementation of the *source* blocked forward substitution.

    4 diagonal 16x16 blocks solved row-serially (``x[r] = e_r + sum_{m<r} a[r,m] x[m]``
    with ``a = -S``), then the 16 off-diagonal block gemms
    ``X_ij = -X_ii * sum_{j<=k<i} S_ik X_kj``.  Used only to cross-check the golden.
    """
    n = s.shape[0]
    b = 16
    nb = n // b
    M = torch.eye(n, dtype=torch.float64) + s
    X = torch.zeros(n, n, dtype=torch.float64)
    for d in range(nb):
        lo = d * b
        sub = M[lo:lo + b, lo:lo + b]
        inv = torch.zeros(b, b, dtype=torch.float64)
        for r in range(b):
            inv[r, r] = 1.0
            for m in range(r):
                inv[r, :] = inv[r, :] - sub[r, m] * inv[m, :]
        X[lo:lo + b, lo:lo + b] = inv
    for i in range(nb):
        for j in range(i):
            acc = torch.zeros(b, b, dtype=torch.float64)
            for kk in range(j, i):
                acc = acc + M[i * b:(i + 1) * b, kk * b:(kk + 1) * b] @ X[kk * b:(kk + 1) * b, j * b:(j + 1) * b]
            X[i * b:(i + 1) * b, j * b:(j + 1) * b] = -X[i * b:(i + 1) * b, i * b:(i + 1) * b] @ acc
    return X


def golden_blocksolve_A_bthd_blocked(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = CHUNK,
    uniform_seq_len: int = 0,
    num_sequences: int = 1,
) -> torch.Tensor:
    """Same semantic as ``golden_blocksolve_A_bthd`` but via the source's block algorithm."""
    total_tokens, key_head = k.shape[1], k.shape[2]
    head = g.shape[2]
    group = head // key_head
    seq_len = uniform_seq_len if uniform_seq_len else total_tokens // num_sequences
    per_seq = (seq_len + chunk_size - 1) // chunk_size
    strict = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), -1)
    khids = torch.arange(head) // group
    A = torch.zeros(1, total_tokens, head, chunk_size, dtype=torch.float64)
    for seq in range(num_sequences):
        s0 = seq * seq_len
        end = s0 + seq_len
        for c in range(per_seq):
            base = s0 + c * chunk_size
            valid = min(chunk_size, end - base)
            rows = torch.clamp(torch.arange(base, base + chunk_size), max=end - 1)
            for hid in range(head):
                kk = k[0, rows, hid // group, :].double()
                gg = g[0, rows, hid].double()
                bb = beta[0, rows, hid].double().clone()
                bb[valid:] = 0.0
                S = torch.where(
                    strict,
                    bb[:, None] * torch.exp(gg[:, None] - gg[None, :]) * (kk @ kk.T),
                    torch.zeros((), dtype=torch.float64),
                )
                # The source always solves the full 64x64 system (tail rows carry beta=0,
                # so their rows/columns of S are zero and the padded block is the identity).
                X64 = _blocked_inverse_fp64(S)
                A[0, base:base + valid, hid, 0:valid] = X64[:valid, :valid]
    return A


def crosscheck_golden(k, g, beta, chunk_size=CHUNK, uniform_seq_len=0, num_sequences=1, tol=1e-10):
    """Both fp64 references must agree to ``tol`` (DESIGN.md section 8.1: measured 2.776e-17)."""
    a_solve = golden_blocksolve_A_bthd(k, g, beta, chunk_size, uniform_seq_len, num_sequences)
    a_block = golden_blocksolve_A_bthd_blocked(k, g, beta, chunk_size, uniform_seq_len, num_sequences)
    err = float((a_solve - a_block).abs().max())
    assert err <= tol, f"[golden-crosscheck] blocked vs solve max_abs_err={err:.3e} > {tol:.1e}"
    print(f"[golden-crosscheck] blocked_source vs linalg.solve max_abs_err={err:.3e} (tol {tol:.1e})")
    return err


# ---------------------------------------------------------------------------
# Host-side shape plan (one source for K / NPER / S_pad; never two copies)
# ---------------------------------------------------------------------------
def _shape_plan(total_tokens, num_sequences, head, chunk_size=CHUNK,
                uniform_seq_len=0, num_cores=LOGICAL_CORES):
    """Return ``(seq_len, per_seq, tasks, cores, nper, s_pad)`` for the equal-length branch."""
    seq_len = uniform_seq_len if uniform_seq_len else total_tokens // num_sequences
    per_seq = (seq_len + chunk_size - 1) // chunk_size
    tasks = num_sequences * per_seq * head
    cores = min(num_cores, tasks)
    nper = (tasks + cores - 1) // cores
    s_pad = ((total_tokens + chunk_size - 1) // chunk_size) * chunk_size
    return seq_len, per_seq, tasks, cores, nper, s_pad


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[], target="npuir")
def prefill_blocksolve_A_bthd_tl(
    total_tokens: int,
    num_sequences: int,
    head: int,
    key_head: int,
    chunk_size: int = CHUNK,
    dim_k: int = 64,
    in_dtype: str = "float16",
    offsets_dtype: str = "int32",
    uniform_seq_len: int = 0,
    num_cores: int = LOGICAL_CORES,
):
    """Build the kernel: one Cube scope + one Vector scope, one task loop each."""
    assert chunk_size == CHUNK, "prefill_blocksolve_A_bthd: chunk_size must be 64"
    assert dim_k in DIM_K_CHOICES, "prefill_blocksolve_A_bthd: dim_k in {64, 128}"
    assert head % key_head == 0, "head must be a multiple of key_head"
    if uniform_seq_len:
        assert num_sequences * uniform_seq_len == total_tokens

    group = head // key_head
    slen, per_seq, tasks, cores, nper, s_pad = _shape_plan(
        total_tokens, num_sequences, head, chunk_size, uniform_seq_len, num_cores
    )
    assert cores * nper >= tasks

    @T.prim_func
    def main(
        k: T.Tensor((1, total_tokens, key_head, dim_k), in_dtype),
        g: T.Tensor((1, total_tokens, head), "float32"),
        beta: T.Tensor((1, total_tokens, head), in_dtype),
        eye: T.Tensor((chunk_size, chunk_size), "float32"),
        mask01: T.Tensor((chunk_size, chunk_size), "float32"),
        kz: T.Tensor((chunk_size, dim_k), in_dtype),
        zero_g: T.Tensor((chunk_size, 1), "float32"),
        zero_b: T.Tensor((chunk_size, 1), in_dtype),
        gs_G: T.Tensor((cores, 2, chunk_size, chunk_size), "float32"),
        ws_S: T.Tensor((cores, 2, chunk_size, chunk_size), "float32"),
        ws_T: T.Tensor((cores, 2, chunk_size, chunk_size), "float32"),
        ws_P: T.Tensor((cores, 2, chunk_size, chunk_size), "float32"),
        y_blk: T.Tensor((cores, 2, chunk_size, chunk_size), in_dtype),
        A_pad: T.Tensor((s_pad, head, chunk_size), in_dtype),
    ):
        with T.Kernel(cores, is_npu=True) as (bid, _):
            # ================= Cube: Gram + 6-level log-doubling =================
            with T.Scope("Cube"):
                l1k = T.alloc_L1([chunk_size, dim_k], in_dtype)
                l1e = T.alloc_L1([chunk_size, chunk_size], "float32")
                l1p = T.alloc_L1([chunk_size, chunk_size], "float32")
                l1t = T.alloc_L1([chunk_size, chunk_size], "float32")
                l1kb = T.alloc_L1([chunk_size, dim_k], in_dtype)
                l0g = T.alloc_L0C([chunk_size, chunk_size], "float32")
                l0t = T.alloc_L0C([chunk_size, chunk_size], "float32")
                l0p = T.alloc_L0C([chunk_size, chunk_size], "float32")
                # Toolchain fact (measured 2026-10-17, CANN 9.0.0): feeding a *non-square*
                # [64, 128] L1 tile to BOTH gemm operands (b_transpose=True) makes
                # bishengir-compile die in codegen, while a square [64, 64] tile shared by
                # both operands is fine.  dim_k=128 therefore stages the B operand in its
                # own tile: identical bytes => identical Gram, +32 KB L1, +1 MTE2 copy.
                # dim_k=64 keeps the single shared operand tile of DESIGN.md section 4.3
                # (l1kb stays unused there).

                T.copy(eye[0:chunk_size, 0:chunk_size], l1e)
                # R3-a: deterministic zero fill of the k tile once per block (MTE2;
                # T.clear lowers to a Vector write and cannot reach L1).
                T.copy(kz[0:chunk_size, 0:dim_k], l1k)
                if dim_k == 128:
                    T.copy(kz[0:chunk_size, 0:dim_k], l1kb)

                # Prologue: Gram of this block's first task, slot 0.
                c0 = bid // head
                s0 = c0 // per_seq
                b0 = s0 * slen + (c0 % per_seq) * chunk_size
                v0 = T.min(chunk_size, s0 * slen + slen - b0)
                h0 = (bid % head) // group
                T.copy(k[0, b0:b0 + v0, h0, 0:dim_k], l1k[0:v0, 0:dim_k])
                if dim_k == 128:
                    T.copy(k[0, b0:b0 + v0, h0, 0:dim_k], l1kb[0:v0, 0:dim_k])
                    T.gemm(l1k, l1kb, l0g, size=[chunk_size, dim_k, chunk_size],
                           b_transpose=True, initC=True)
                else:
                    T.gemm(l1k, l1k, l0g, size=[chunk_size, dim_k, chunk_size],
                           b_transpose=True, initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0g, gs_G[bid, 0, 0:chunk_size, 0:chunk_size])
                    T.sync_block_set(F_G)

                for it in T.serial(nper):
                    t = bid + it * cores
                    slot = it % 2
                    if t < tasks:
                        # One-task-ahead Gram for the next task of this block.
                        if it + 1 < nper and t + cores < tasks:
                            tn = t + cores
                            cn = tn // head
                            sn = cn // per_seq
                            bn = sn * slen + (cn % per_seq) * chunk_size
                            vn = T.min(chunk_size, sn * slen + slen - bn)
                            hn = (tn % head) // group
                            T.copy(k[0, bn:bn + vn, hn, 0:dim_k], l1k[0:vn, 0:dim_k])
                            if dim_k == 128:
                                T.copy(k[0, bn:bn + vn, hn, 0:dim_k], l1kb[0:vn, 0:dim_k])
                                T.gemm(l1k, l1kb, l0g, size=[chunk_size, dim_k, chunk_size],
                                       b_transpose=True, initC=True)
                            else:
                                T.gemm(l1k, l1k, l0g, size=[chunk_size, dim_k, chunk_size],
                                       b_transpose=True, initC=True)
                            with T.rs("PIPE_FIX"):
                                T.copy(l0g, gs_G[bid, 1 - slot, 0:chunk_size, 0:chunk_size])
                                T.sync_block_set(F_G + 1 - slot)
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(F_S + slot)
                            T.copy(ws_S[bid, slot, 0:chunk_size, 0:chunk_size], l1p)
                        # T_0 = I: the only way an L0C takes a value is a gemm.
                        T.gemm(l1e, l1e, l0t, size=[chunk_size, chunk_size, chunk_size], initC=True)
                        # lev 0: T_1 = T_0 + P_0*T_0 = I + N   (P_0 = N in l1p)
                        T.gemm(l1p, l1e, l0t, size=[chunk_size, chunk_size, chunk_size], initC=False)
                        T.gemm(l1p, l1p, l0p, size=[chunk_size, chunk_size, chunk_size], initC=True)
                        with T.rs("PIPE_FIX"):
                            T.copy(l0t, ws_T[bid, 1, 0:chunk_size, 0:chunk_size])
                        with T.rs("PIPE_FIX"):
                            T.copy(l0p, ws_P[bid, 1, 0:chunk_size, 0:chunk_size])
                        T.copy(ws_P[bid, 1, 0:chunk_size, 0:chunk_size], l1p)
                        # lev 1: T_2 = T_1 + P_1*T_1 ; P_2 = P_1^2
                        T.copy(ws_T[bid, 1, 0:chunk_size, 0:chunk_size], l1t)
                        T.gemm(l1p, l1t, l0t, size=[chunk_size, chunk_size, chunk_size], initC=False)
                        T.gemm(l1p, l1p, l0p, size=[chunk_size, chunk_size, chunk_size], initC=True)
                        with T.rs("PIPE_FIX"):
                            T.copy(l0t, ws_T[bid, 0, 0:chunk_size, 0:chunk_size])
                        with T.rs("PIPE_FIX"):
                            T.copy(l0p, ws_P[bid, 0, 0:chunk_size, 0:chunk_size])
                        T.copy(ws_P[bid, 0, 0:chunk_size, 0:chunk_size], l1p)
                        # lev 2
                        T.copy(ws_T[bid, 0, 0:chunk_size, 0:chunk_size], l1t)
                        T.gemm(l1p, l1t, l0t, size=[chunk_size, chunk_size, chunk_size], initC=False)
                        T.gemm(l1p, l1p, l0p, size=[chunk_size, chunk_size, chunk_size], initC=True)
                        with T.rs("PIPE_FIX"):
                            T.copy(l0t, ws_T[bid, 1, 0:chunk_size, 0:chunk_size])
                        with T.rs("PIPE_FIX"):
                            T.copy(l0p, ws_P[bid, 1, 0:chunk_size, 0:chunk_size])
                        T.copy(ws_P[bid, 1, 0:chunk_size, 0:chunk_size], l1p)
                        # lev 3
                        T.copy(ws_T[bid, 1, 0:chunk_size, 0:chunk_size], l1t)
                        T.gemm(l1p, l1t, l0t, size=[chunk_size, chunk_size, chunk_size], initC=False)
                        T.gemm(l1p, l1p, l0p, size=[chunk_size, chunk_size, chunk_size], initC=True)
                        with T.rs("PIPE_FIX"):
                            T.copy(l0t, ws_T[bid, 0, 0:chunk_size, 0:chunk_size])
                        with T.rs("PIPE_FIX"):
                            T.copy(l0p, ws_P[bid, 0, 0:chunk_size, 0:chunk_size])
                        T.copy(ws_P[bid, 0, 0:chunk_size, 0:chunk_size], l1p)
                        # lev 4
                        T.copy(ws_T[bid, 0, 0:chunk_size, 0:chunk_size], l1t)
                        T.gemm(l1p, l1t, l0t, size=[chunk_size, chunk_size, chunk_size], initC=False)
                        T.gemm(l1p, l1p, l0p, size=[chunk_size, chunk_size, chunk_size], initC=True)
                        with T.rs("PIPE_FIX"):
                            T.copy(l0t, ws_T[bid, 1, 0:chunk_size, 0:chunk_size])
                        with T.rs("PIPE_FIX"):
                            T.copy(l0p, ws_P[bid, 1, 0:chunk_size, 0:chunk_size])
                        T.copy(ws_P[bid, 1, 0:chunk_size, 0:chunk_size], l1p)
                        # lev 5: T_6 = T_5 + P_5*T_5 = A  (P_5 = N^32 is used, never squared)
                        T.copy(ws_T[bid, 1, 0:chunk_size, 0:chunk_size], l1t)
                        T.gemm(l1p, l1t, l0t, size=[chunk_size, chunk_size, chunk_size], initC=False)
                        with T.rs("PIPE_FIX"):
                            # f32 -> input dtype quantization happens here, on a
                            # contiguous 64x64 scratch; the strided write is Vector's.
                            T.copy(l0t, y_blk[bid, slot, 0:chunk_size, 0:chunk_size])
                            T.sync_block_set(F_A + slot)

            # ============ Vector: gates/beta/mask -> N ; store the previous A ============
            with T.Scope("Vector"):
                MK = T.alloc_ub([chunk_size, chunk_size], "float32")     # static 0/1 strict-lower
                UB_G = T.alloc_ub([chunk_size, chunk_size], "float32")   # Gram
                UB_A = T.alloc_ub([chunk_size, chunk_size], "float32")   # g_i -> d
                UB_B = T.alloc_ub([chunk_size, chunk_size], "float32")   # g_j
                UB_E = T.alloc_ub([chunk_size, chunk_size], "float32")   # exp(d)
                UB_F = T.alloc_ub([chunk_size, chunk_size], "float32")   # -beta_i -> t1
                UB_N = T.alloc_ub([chunk_size, chunk_size], "float32")   # t2 -> N
                UB_Y = T.alloc_ub([chunk_size, chunk_size], in_dtype)    # A tile staging
                ub_gc = T.alloc_ub([chunk_size, 1], "float32")           # g_i source
                ub_gr = T.alloc_ub([1, chunk_size], "float32")           # g_j source
                ub_b16 = T.alloc_ub([chunk_size, 1], in_dtype)           # beta as stored
                ub_b32 = T.alloc_ub([chunk_size, 1], "float32")
                ub_bn = T.alloc_ub([chunk_size, 1], "float32")           # -beta

                T.copy(mask01[0:chunk_size, 0:chunk_size], MK)
                # R3-a: zero the gate sources once per block, so a block whose first
                # task is a tail chunk never reads uninitialized SRAM.
                T.copy(zero_g[0:chunk_size, 0:1], ub_gc)
                T.copy(zero_g[0:chunk_size, 0:1], ub_gr[0:1, 0:chunk_size])

                for it in T.serial(nper):
                    t = bid + it * cores
                    slot = it % 2
                    if t < tasks:
                        cq = t // head
                        sq = cq // per_seq
                        b = sq * slen + (cq % per_seq) * chunk_size
                        vv = T.min(chunk_size, sq * slen + slen - b)
                        hd = t % head
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(F_G + slot)
                            T.copy(gs_G[bid, slot, 0:chunk_size, 0:chunk_size], UB_G)
                            # R3-b: clear the whole beta column first, then overwrite
                            # only the valid rows => rows past the end are exactly 0.
                            T.copy(zero_b[0:chunk_size, 0:1], ub_b16)
                            T.copy(g[0, b:b + vv, hd:hd + 1], ub_gc[0:vv, 0:1])
                            T.copy(g[0, b:b + vv, hd:hd + 1], ub_gr[0:1, 0:vv])
                            T.copy(beta[0, b:b + vv, hd:hd + 1], ub_b16[0:vv, 0:1])
                        T.vcast(ub_b16, ub_b32, round_mode="rint")
                        T.vmul(ub_b32, T.float32(-1.0), ub_bn)   # the minus sign of N lands here
                        T.vbrc(ub_gc, UB_A)                      # UB_A[i, j] = g_i
                        T.vbrc(ub_gr, UB_B)                      # UB_B[i, j] = g_j
                        T.vsub(UB_A, UB_B, UB_A)                 # d = g_i - g_j
                        T.vmin(UB_A, T.float32(0.0), UB_A)       # bound the discarded region
                        T.vexp(UB_A, UB_E)                       # e = exp(min(d, 0))
                        T.vbrc(ub_bn, UB_F)                      # -beta_i
                        T.vmul(UB_F, UB_E, UB_F)                 # -beta_i * e
                        T.vmul(UB_F, UB_G, UB_N)                 # * G
                        T.vmul(UB_N, MK, UB_N)                   # * mask01 = N
                        with T.rs("PIPE_MTE3"):
                            T.copy(UB_N, ws_S[bid, slot, 0:chunk_size, 0:chunk_size])
                            T.sync_block_set(F_S + slot)
                    # One-task-lagged store. The guard binds the task being stored
                    # (t - K), not the current iteration (DESIGN.md section 7.4).
                    if it >= 1 and t - cores < tasks:
                        pt = t - cores
                        ps = (it - 1) % 2
                        pc = pt // head
                        pq = pc // per_seq
                        pb = pq * slen + (pc % per_seq) * chunk_size
                        ph = pt % head
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(F_A + ps)
                            T.copy(y_blk[bid, ps, 0:chunk_size, 0:chunk_size], UB_Y)
                        with T.rs("PIPE_MTE3"):
                            T.copy(UB_Y, A_pad[pb:pb + chunk_size, ph:ph + 1, 0:chunk_size])
                # Epilogue: this block's own last task, when the loop could not reach it.
                last_it = (tasks - 1 - bid) // cores
                if last_it == nper - 1:
                    lt = bid + last_it * cores
                    ls = last_it % 2
                    lc = lt // head
                    lq = lc // per_seq
                    lb = lq * slen + (lc % per_seq) * chunk_size
                    lh = lt % head
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(F_A + ls)
                        T.copy(y_blk[bid, ls, 0:chunk_size, 0:chunk_size], UB_Y)
                    with T.rs("PIPE_MTE3"):
                        T.copy(UB_Y, A_pad[lb:lb + chunk_size, lh:lh + 1, 0:chunk_size])

    return main


# ---------------------------------------------------------------------------
# Op wrapper (host side; no data transform beyond the padding view)
# ---------------------------------------------------------------------------
def prefill_blocksolve_A_bthd(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor = None,
    chunk_size: int = CHUNK,
    use_gate: bool = True,
    l2norm: bool = False,
    beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
    uniform_seq_len: int = 0,
    a_pad: torch.Tensor = None,
    num_cores: int = 0,
):
    """Per-chunk ``(I + S_sys)^{-1}`` for the delta rule (source contract kept).

    ``cu_seqlens`` is accepted for interface parity but not read on the equal-length
    branch (source: L257-L260 takes the arithmetic path when ``uniform_seq_len`` is set).
    Returns ``(A, k_rnorm)``: ``A`` is the ``[1, total_tokens, head, chunk_size]`` view of a
    chunk-aligned ``A_pad`` (contiguous), ``k_rnorm`` is the one-token fp32 placeholder the
    non-normalizing callers ignore.
    """
    if chunk_size != CHUNK:
        raise ValueError(f"chunk_size must be {CHUNK}, got {chunk_size}")
    dim_k = k.shape[-1]
    if dim_k not in DIM_K_CHOICES:
        raise ValueError(f"dim_k must be in {DIM_K_CHOICES}, got {dim_k}")
    if l2norm or beta_sigmoid or allow_neg_eigval or not use_gate:
        raise NotImplementedError(
            "prefill_blocksolve_A_bthd: in scope is use_gate=True, l2norm=False, "
            "beta_sigmoid=False, allow_neg_eigval=False"
        )
    if g.dtype != torch.float32:
        raise ValueError("g must be float32 (chunk-local cumulative log gate)")
    total_tokens, key_head = k.shape[1], k.shape[2]
    head = g.shape[-1]
    if head % key_head:
        raise ValueError(f"head ({head}) must be a multiple of key_head ({key_head})")
    num_sequences = 1 if cu_seqlens is None else cu_seqlens.shape[0] - 1
    seq_len, per_seq, tasks, cores, nper, s_pad = _shape_plan(
        total_tokens, num_sequences, head, chunk_size, uniform_seq_len,
        num_cores or LOGICAL_CORES
    )
    dev = k.device
    f32 = torch.float32
    eye = torch.eye(chunk_size, dtype=f32, device=dev)
    mask01 = torch.tril(torch.ones(chunk_size, chunk_size, dtype=f32, device=dev), -1)
    kz = torch.zeros(chunk_size, dim_k, dtype=k.dtype, device=dev)
    zero_g = torch.zeros(chunk_size, 1, dtype=f32, device=dev)
    zero_b = torch.zeros(chunk_size, 1, dtype=k.dtype, device=dev)
    ws_shape = (cores, 2, chunk_size, chunk_size)
    gs_G = torch.empty(ws_shape, dtype=f32, device=dev)
    ws_S = torch.empty(ws_shape, dtype=f32, device=dev)
    ws_T = torch.empty(ws_shape, dtype=f32, device=dev)
    ws_P = torch.empty(ws_shape, dtype=f32, device=dev)
    y_blk = torch.empty(ws_shape, dtype=k.dtype, device=dev)
    if a_pad is None:
        a_pad = torch.empty(s_pad, head, chunk_size, dtype=k.dtype, device=dev)
    in_dtype = str(k.dtype).split(".")[-1]
    with scoped_ascend_mode(ASCEND_MODE):
        kernel = prefill_blocksolve_A_bthd_tl(
            total_tokens, num_sequences, head, key_head, chunk_size, dim_k,
            in_dtype, "int32", uniform_seq_len, num_cores or LOGICAL_CORES,
        )
        kernel(k, g, beta, eye, mask01, kz, zero_g, zero_b,
               gs_G, ws_S, ws_T, ws_P, y_blk, a_pad)
    A = a_pad[:total_tokens].unsqueeze(0)
    k_rnorm = torch.empty(1, 1, key_head, dtype=f32, device=dev)
    return A, k_rnorm


# ---------------------------------------------------------------------------
# L0-h dump channel: the Vector gate/sign/mask chain, device side
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[], target="npuir")
def _gate_dump_tl(total_tokens: int, head: int, chunk_index: int = 0, dim_k: int = 128,
                  in_dtype: str = "float16"):
    """One task (``chunk_index``, head 0), Vector phase only, intermediates dumped.

    ``valid`` is a build-time constant here (L0-h is a numeric channel, not a schedule
    test); the op chain is the literal DESIGN.md section 3.3 sequence, R3-a/R3-b included.
    """
    base = chunk_index * CHUNK
    valid = min(CHUNK, total_tokens - base)
    assert valid > 0

    @T.prim_func
    def main(
        g: T.Tensor((1, total_tokens, head), "float32"),
        beta: T.Tensor((1, total_tokens, head), in_dtype),
        Gsrc: T.Tensor((CHUNK, CHUNK), "float32"),
        mask01: T.Tensor((CHUNK, CHUNK), "float32"),
        zero_g: T.Tensor((CHUNK, 1), "float32"),
        zero_b: T.Tensor((CHUNK, 1), in_dtype),
        o_e: T.Tensor((CHUNK, CHUNK), "float32"),
        o_n: T.Tensor((CHUNK, CHUNK), "float32"),
        o_gc: T.Tensor((CHUNK, 1), "float32"),
        o_gr: T.Tensor((1, CHUNK), "float32"),
        o_b16: T.Tensor((CHUNK, 1), in_dtype),
        o_bn: T.Tensor((CHUNK, 1), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (bid, _):
            with T.Scope("Vector"):
                MK = T.alloc_ub([CHUNK, CHUNK], "float32")
                UB_G = T.alloc_ub([CHUNK, CHUNK], "float32")
                UB_A = T.alloc_ub([CHUNK, CHUNK], "float32")
                UB_B = T.alloc_ub([CHUNK, CHUNK], "float32")
                UB_E = T.alloc_ub([CHUNK, CHUNK], "float32")
                UB_F = T.alloc_ub([CHUNK, CHUNK], "float32")
                UB_N = T.alloc_ub([CHUNK, CHUNK], "float32")
                ub_gc = T.alloc_ub([CHUNK, 1], "float32")
                ub_gr = T.alloc_ub([1, CHUNK], "float32")
                ub_b16 = T.alloc_ub([CHUNK, 1], in_dtype)
                ub_b32 = T.alloc_ub([CHUNK, 1], "float32")
                ub_bn = T.alloc_ub([CHUNK, 1], "float32")
                T.copy(mask01[0:CHUNK, 0:CHUNK], MK)
                T.copy(zero_g[0:CHUNK, 0:1], ub_gc)                     # R3-a
                T.copy(zero_g[0:CHUNK, 0:1], ub_gr[0:1, 0:CHUNK])       # R3-a
                with T.rs("PIPE_MTE2"):
                    T.copy(Gsrc[0:CHUNK, 0:CHUNK], UB_G)
                    T.copy(zero_b[0:CHUNK, 0:1], ub_b16)                # R3-b (first)
                    T.copy(g[0, base:base + valid, 0:1], ub_gc[0:valid, 0:1])
                    T.copy(g[0, base:base + valid, 0:1], ub_gr[0:1, 0:valid])
                    T.copy(beta[0, base:base + valid, 0:1], ub_b16[0:valid, 0:1])
                T.vcast(ub_b16, ub_b32, round_mode="rint")
                T.vmul(ub_b32, T.float32(-1.0), ub_bn)
                T.vbrc(ub_gc, UB_A)
                T.vbrc(ub_gr, UB_B)
                T.vsub(UB_A, UB_B, UB_A)
                T.vmin(UB_A, T.float32(0.0), UB_A)
                T.vexp(UB_A, UB_E)
                T.vbrc(ub_bn, UB_F)
                T.vmul(UB_F, UB_E, UB_F)
                T.vmul(UB_F, UB_G, UB_N)
                T.vmul(UB_N, MK, UB_N)
                with T.rs("PIPE_MTE3"):
                    T.copy(UB_E, o_e[0:CHUNK, 0:CHUNK])
                    T.copy(UB_N, o_n[0:CHUNK, 0:CHUNK])
                    T.copy(ub_gc, o_gc[0:CHUNK, 0:1])
                    T.copy(ub_gr, o_gr[0:1, 0:CHUNK])
                    T.copy(ub_b16, o_b16[0:CHUNK, 0:1])
                    T.copy(ub_bn, o_bn[0:CHUNK, 0:1])

    return main


# ---------------------------------------------------------------------------
# Layered tests
# ---------------------------------------------------------------------------
def _mk_inputs(total_tokens, head, key_head, dim_k, dtype, seed=0, beta_zero=False, g_step=1.0):
    """Deterministic inputs: normalized keys, chunk-local monotone gates, beta in (0, 0.9)."""
    torch.manual_seed(seed)
    kf = torch.randn(1, total_tokens, key_head, dim_k, dtype=torch.float32)
    kf = kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    k = kf.to(dtype)
    g = torch.zeros(1, total_tokens, head, dtype=torch.float32)
    for c in range((total_tokens + CHUNK - 1) // CHUNK):
        lo = c * CHUNK
        hi = min(lo + CHUNK, total_tokens)
        g[0, lo:hi] = torch.cumsum(-torch.rand(hi - lo, head) * g_step - 0.05, dim=0)
    bf = torch.rand(1, total_tokens, head, dtype=torch.float32) * 0.9
    if beta_zero:
        bf[:, ::7] = 0.0
    return k, g, bf.to(dtype)


def _check_padding(a_pad, total_tokens, tag):
    """Rows past ``total_tokens`` must be exactly ``e_i`` (i = row index inside its chunk)."""
    s_pad = a_pad.shape[0]
    assert s_pad >= total_tokens
    if s_pad == total_tokens:
        print(f"[{tag}] padding: none (S_pad == total_tokens == {total_tokens})")
        return 0.0
    pad = a_pad[total_tokens:].cpu().double()
    nrow = pad.shape[0]
    idx = torch.arange(total_tokens, s_pad) % CHUNK
    ref = torch.zeros(nrow, pad.shape[1], CHUNK, dtype=torch.float64)
    ref[torch.arange(nrow), :, idx] = 1.0
    err = float((pad - ref).abs().max())
    assert err == 0.0, f"[{tag}] padding rows != e_i: max_abs_err={err:.3e} (must be exactly 0)"
    print(f"[{tag}] padding: {nrow} rows == e_i exactly (max_abs_err=0.0)")
    return err


def _run_case(total_tokens, head, key_head, dim_k, dtype, tag, seed=0, beta_zero=False,
              g_step=1.0, a_pad_fill=1024.0):
    """One device case: golden compare + hard tail-chunk / store-coverage assertions.

    ``A_pad`` is pre-filled with a sentinel (1024.0, exact in fp16 and bf16) instead of
    ``torch.empty``: a missed store then shows up as a sentinel cell, which makes the
    exactly-once store coverage of DESIGN.md section 7.4 observable in every case.
    """
    k, g, beta = _mk_inputs(total_tokens, head, key_head, dim_k, dtype, seed, beta_zero, g_step)
    ref = golden_blocksolve_A_bthd(k, g, beta, CHUNK, total_tokens, 1)
    s_pad = ((total_tokens + CHUNK - 1) // CHUNK) * CHUNK
    a_pad = torch.full((s_pad, head, CHUNK), a_pad_fill, dtype=dtype, device="npu")
    a_dev, k_rnorm = prefill_blocksolve_A_bthd(
        k.npu(), g.npu(), beta.npu(), None, CHUNK, True, False, False, False,
        total_tokens, a_pad,
    )
    assert k_rnorm.dtype == torch.float32 and k_rnorm.shape == (1, 1, key_head)
    out = a_dev.cpu()
    assert out.shape == (1, total_tokens, head, CHUNK), f"[{tag}] unexpected A shape {tuple(out.shape)}"
    assert a_dev.is_contiguous(), f"[{tag}] A must be contiguous"
    nan = int(torch.isnan(out).sum())
    inf = int(torch.isinf(out).sum())
    assert nan == 0, f"[{tag}] {nan} NaN elements (tail-chunk zero-fill failed)"
    assert inf == 0, f"[{tag}] {inf} Inf elements"

    atol = 5e-3 if dtype == torch.float16 else 4e-2
    diff = (out.double() - ref).abs()
    max_abs = float(diff.max())
    denom = ref.abs()
    max_rel = float((diff / denom.clamp_min(1e-3)).max())
    scale = torch.where(ref.abs() > 4.0, ref.abs(), torch.ones_like(ref))
    viol = float((diff / (atol * scale)).max())
    assert viol <= 1.0, (
        f"[{tag}] FAIL tokens={total_tokens} head={head} dk={dim_k} {dtype} "
        f"max_abs_err={max_abs:.3e} max_rel_err={max_rel:.3e} threshold={atol:.1e}"
    )

    _check_padding(a_pad, total_tokens, tag)
    left = int((a_pad == a_pad_fill).sum())
    assert left == 0, f"[{tag}] {left} cells still hold the sentinel: store coverage is incomplete"
    print(f"[{tag}] store coverage: every cell of A_pad written (0 sentinels left)")

    print(f"[{tag}] PASS tokens={total_tokens} head={head} kh={key_head} dk={dim_k} "
          f"dtype={str(dtype).split('.')[-1]} max_abs_err={max_abs:.3e} "
          f"max_rel_err={max_rel:.3e} nan=0 (threshold {atol:.1e})")
    return max_abs


def _run_gate_dump(total_tokens, chunk_index, dim_k, tag, seed=0):
    """L0-h: verify ``d == g_i - g_j``, ``bn == -beta``, ``N == -beta*e*G*mask01`` on device."""
    head, key_head = 1, 1
    k, g, beta = _mk_inputs(total_tokens, head, key_head, dim_k, torch.float16, seed)
    base = chunk_index * CHUNK
    valid = min(CHUNK, total_tokens - base)
    rows = torch.clamp(torch.arange(base, base + CHUNK), max=total_tokens - 1)
    kk = k[0, rows, 0, :].double()
    G = kk @ kk.T
    G[valid:, :] = 0.0
    G[:, valid:] = 0.0                      # l1k is zero past the end (R3-a)
    gg = g[0, base:base + valid, 0].double()
    bb = beta[0, base:base + valid, 0].double()
    gc = torch.cat([gg, torch.zeros(CHUNK - valid, dtype=torch.float64)])
    gr = gc.clone()
    bp = torch.cat([bb, torch.zeros(CHUNK - valid, dtype=torch.float64)])
    strict = torch.tril(torch.ones(CHUNK, CHUNK, dtype=torch.bool), -1)
    e_ref = torch.exp(torch.minimum(gc[:, None] - gr[None, :], torch.zeros((), dtype=torch.float64)))
    n_ref = torch.where(strict, -bp[:, None] * e_ref * G, torch.zeros((), dtype=torch.float64))

    dev = "npu"
    o_e = torch.zeros(CHUNK, CHUNK, dtype=torch.float32, device=dev)
    o_n = torch.zeros(CHUNK, CHUNK, dtype=torch.float32, device=dev)
    o_gc = torch.zeros(CHUNK, 1, dtype=torch.float32, device=dev)
    o_gr = torch.zeros(1, CHUNK, dtype=torch.float32, device=dev)
    o_b16 = torch.zeros(CHUNK, 1, dtype=torch.float16, device=dev)
    o_bn = torch.zeros(CHUNK, 1, dtype=torch.float32, device=dev)
    with scoped_ascend_mode(ASCEND_MODE):
        kern = _gate_dump_tl(total_tokens, head, chunk_index, dim_k, "float16")
        kern(g.npu(), beta.npu(), G.float().npu(),
             torch.tril(torch.ones(CHUNK, CHUNK, dtype=torch.float32), -1).npu(),
             torch.zeros(CHUNK, 1, dtype=torch.float32).npu(),
             torch.zeros(CHUNK, 1, dtype=torch.float16).npu(),
             o_e, o_n, o_gc, o_gr, o_b16, o_bn)
    torch.npu.synchronize()

    e_err = float((o_e.cpu().double() - e_ref).abs().max())
    n_err = float((o_n.cpu().double() - n_ref).abs().max())
    gc_err = float((o_gc[0:valid, 0].cpu().double() - gg).abs().max())
    gr_err = float((o_gr[0, 0:valid].cpu().double() - gg).abs().max())
    b_err = float((o_b16[0:valid, 0].cpu().double() - bb).abs().max())
    bn_err = float((o_bn[0:valid, 0].cpu().double() + bb).abs().max())      # bn == -beta
    gc_tail = float(o_gc[valid:, 0].cpu().abs().max()) if valid < CHUNK else 0.0
    gr_tail = float(o_gr[0, valid:].cpu().abs().max()) if valid < CHUNK else 0.0
    b_tail = float(o_b16[valid:, 0].cpu().abs().max()) if valid < CHUNK else 0.0
    bn_tail = float(o_bn[valid:, 0].cpu().abs().max()) if valid < CHUNK else 0.0
    upper = float(o_n.cpu().double()[~strict].abs().max())
    print(f"[{tag}] dump valid={valid} dk={dim_k}: e_err={e_err:.3e} n_err={n_err:.3e} "
          f"gc_err={gc_err:.3e} gr_err={gr_err:.3e} b16_err={b_err:.3e} bn_err={bn_err:.3e}")
    print(f"[{tag}] tail zero-fill (R3-a/R3-b): gc={gc_tail:.1e} gr={gr_tail:.1e} "
          f"b16={b_tail:.1e} bn={bn_tail:.1e}; strictly-upper |N|max={upper:.1e}")
    assert e_err <= 1e-5, f"[{tag}] e != exp(min(g_i-g_j,0)): {e_err:.3e}"
    assert n_err <= 1e-5, f"[{tag}] N != -beta_i*e*G*mask01: {n_err:.3e}"
    assert max(gc_err, gr_err, b_err, bn_err) <= 1e-5, f"[{tag}] gate/beta mismatch"
    assert max(gc_tail, gr_tail, b_tail, bn_tail) == 0.0, f"[{tag}] tail rows not exactly zero"
    assert upper == 0.0, f"[{tag}] N is not exactly zero outside the strict lower triangle"
    print(f"[{tag}] PASS (gate direction, minus sign, mask, R3-a/R3-b zero fill)")
    return n_err


def run_L0():
    print("=== L0 (single/dual task, both dim_k, both dtypes, tail chunk, head stride) ===")
    k, g, beta = _mk_inputs(100, 1, 1, 64, torch.float16, seed=3)
    crosscheck_golden(k, g, beta, CHUNK, 100, 1)
    errs = []
    errs.append(_run_case(64, 1, 1, 64, torch.float16, "L0-a"))
    errs.append(_run_case(64, 1, 1, 128, torch.float16, "L0-b"))
    errs.append(_run_case(64, 1, 1, 64, torch.bfloat16, "L0-c"))
    errs.append(_run_case(100, 1, 1, 64, torch.float16, "L0-d", seed=1, a_pad_fill=1024.0))
    errs.append(_run_case(64, 2, 1, 64, torch.float16, "L0-e", seed=2))
    errs.append(_run_case(64, 1, 1, 64, torch.float16, "L0-f", seed=4,
                          beta_zero=True, g_step=1.3))
    errs.append(_run_case(64, 16, 16, 128, torch.float16, "L0-g", seed=5, a_pad_fill=1024.0))
    _run_gate_dump(64, 0, 128, "L0-h")
    _run_gate_dump(100, 1, 128, "L0-h-tail")
    errs.append(_run_case(1600, 1, 1, 64, torch.float16, "L0-i", seed=6))
    print(f"[L0] {len(errs)} golden cases + 2 dump cases passed, worst max_abs_err={max(errs):.3e}")
    return errs


def run_L1():
    print("=== L1 (multi-task / multi-core stride, flag reuse, real tail chunk) ===")
    errs = []
    errs.append(_run_case(1024, 4, 2, 128, torch.float16, "L1-a", seed=11))
    errs.append(_run_case(3000, 16, 1, 128, torch.float16, "L1-b", seed=12))
    print(f"[L1] {len(errs)} cases passed, worst max_abs_err={max(errs):.3e}")
    return errs


def run_L2():
    print("=== L2 (delivery shapes: S x key_head x dim_k x dtype) ===")
    errs = []
    for total_tokens in (4096, 12288, 32768):
        for key_head, dim_k in ((1, 64), (1, 128), (16, 64), (16, 128)):
            for dtype in (torch.float16, torch.bfloat16):
                tag = f"L2-S{total_tokens}-kh{key_head}-dk{dim_k}-{str(dtype).split('.')[-1]}"
                errs.append(_run_case(total_tokens, 16, key_head, dim_k, dtype, tag, seed=21))
        print(f"[L2] S={total_tokens} done, worst so far={max(errs):.3e}")
    print(f"[L2] {len(errs)} cases passed, worst max_abs_err={max(errs):.3e}")
    return errs


def test_boundary():
    """Chunk-boundary triple: valid < 64, == 64, == 1; NaN-free + padding == e_i are hard."""
    print("=== Boundary (S=56 / 64 / 65: valid<64, ==64, ==1) ===")
    errs = []
    errs.append(_run_case(56, 2, 1, 64, torch.float16, "boundary-S56", seed=31, a_pad_fill=1024.0))
    errs.append(_run_case(64, 2, 1, 64, torch.float16, "boundary-S64", seed=32))
    errs.append(_run_case(65, 2, 1, 64, torch.float16, "boundary-S65", seed=33, a_pad_fill=1024.0))
    print(f"[Boundary] {len(errs)} cases passed, worst max_abs_err={max(errs):.3e}")
    return errs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    elif args.level == "L2":
        run_L2()
    else:
        run_L0()
        run_L1()
        run_L2()
        test_boundary()
    print("All check passed!")


if __name__ == "__main__":
    main()
