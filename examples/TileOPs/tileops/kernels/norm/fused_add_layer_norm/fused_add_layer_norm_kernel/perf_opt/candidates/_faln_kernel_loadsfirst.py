# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Fused Add + LayerNorm forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``_fused_add_layer_norm_kernel`` (TileOPs
FusedAddLayerNormFwdOp, dual-output row reduction) to TileLang
``target="npuir"``.

Semantics (DESIGN.md section 0.1, frozen):
    s_ij          = x_ij + residual_ij              (fp32 add)
    residual_out  = round_dtype(s)                  (single rounding, 2nd output)
    mu_i          = sum_j(residual_out_ij) / N      (biased, ddof=0)
    var_i         = sum_j((residual_out_ij - mu_i)^2) / N
    rstd_i        = 1 / sqrt(var_i + eps)
    y_ij          = (residual_out_ij - mu_i) * rstd_i * w_j + b_j
    output        = round_dtype(y)                  (1st output)
    fp32 opmath throughout, single round-back to the input dtype.

The two outputs are ``output`` (out_idx=4, normalized y) and ``residual_out``
(out_idx=5, the pre-norm rounded sum). ``residual_out`` is the rounded value
re-entered into fp32 for the statistics (golden-aligned "round-first stats",
DESIGN.md section 1.6.1 O4 / discrimination segment B).

NPU redesign (DESIGN.md section 0.6):
    R1: persistent core split: ``num_kernels = min(num_logical,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop
        and an if-guard for out-of-range row blocks (static bounds).
    R2: block_m driven by the UB budget table (DESIGN.md section 4.5),
        not the GPU structural block_m=1.
    R3: tail handling via ``T.min`` + src/dst dual explicit slices +
        garbage-row drop (rows are independent; reduce dim=1 and the
        row-broadcast v-ops never mix rows; output slices copy valid
        rows only).
    R4: explicit fp32 transit chain for fp16/bf16:
        ``T.vcast(rint)`` up to fp32 -> fp32 v-prefix chain ->
        ``T.vcast(rint)`` single round-back (bf16 has no Vector
        arithmetic; fp16 transit matches the golden's fp32 opmath).
    R5: per-column affine factors (weight/bias (N,)) load into a (1, N)
        fp32 buffer and are expanded with ``T.vbrc`` (bm >= 2); bm == 1
        uses the same-shape vmul/vadd directly (avoids the empty-broadcast
        MLIR verify failure, TRAP-vbrc-same-shape-empty-broadcast).

Dropped GPU mechanisms (DESIGN.md section 0.5, intent handed over):
    256-alignment padding + N_padded (replaced by exact N + slice),
    threads/block two-level parallelism (Vector lanes),
    predicate masks ``T.if_then_else`` + ``T.And`` (replaced by T.min +
    dual explicit slices), implicit dtype truncation (explicit vcast rint).

Factory contract (source interface unchanged):
    _fused_add_layer_norm_kernel(M, N, eps, dtype) -> _func(block_m=default)
    ``main(x, residual, weight, bias, output, residual_out)`` returns
    ``(output, residual_out)`` (``out_idx=[4, 5]``, source contract).

Run the embedded hierarchical tests:
    python _fused_add_layer_norm_kernel.py --level L0
    python _fused_add_layer_norm_kernel.py --level all
"""

import os

# Developer mode: alloc_shared maps to UB, compiler auto-sync
# (DESIGN.md sections 2 and 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

# Programming mode of the verified kernel (Stage 3); kept in sync with the
# ambient TILELANG_ASCEND_MODE used when running this file standalone.
ASCEND_MODE = "Developer"

import argparse
import time

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["ASCEND_MODE", "_fused_add_layer_norm_kernel", "golden_fused_add_layer_norm"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported dtypes (factory validation).
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# UB capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196_608

# Measured multi-buffered UB cost per element of the (block_m, N) resident
# block (ada_layer_norm attempt-1 finding on this toolchain): when the
# persistent serial loop is active (num_local_tasks >= 2) the
# auto-multi-buffer pass materializes ~20 B/elem on both dtype paths
# (fp16/bf16 transit: 2 instances x (stage 2B + x_f32 4B + sq_f32 4B);
# fp32 direct: 2.5x of (x_f32 4B + sq_f32 4B)). The extra (1, N) fp32 w1
# buffer is small (4N B) and folded into the same conservative element cap.
# Budget expressed in resident elements: 192 KB / 20 B = 9830, kept at 9216
# (>= 12 KB margin).
_UB_BYTES_PER_ELEM = 20
_UB_MAX_ELEMS = 9_216

# block_m default cap (DESIGN.md section 4.5).
_BLOCK_M_CAP = 8

# Precision tolerances (DESIGN.md section 8.2, from the source repo
# tests/ops/test_fused_add_layer_norm.py ``_get_tolerances()``).
_TOLERANCE = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "bfloat16": (torch.bfloat16, 1.6e-2, 1.6e-2),
    "float32": (torch.float32, 1e-5, 1e-5),
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_fused_add_layer_norm(x, residual, weight, bias, eps=1e-5):
    """PyTorch reference: fused residual add + LayerNorm (two outputs).

    Port of the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_fused_add_layer_norm.py ``ref_program``): ``added = (x.float() +
    residual.float()).to(x.dtype)`` then ``F.layer_norm(added.float(), ...)``
    and a single round-back. Independent of the NPU algorithm (no vcast /
    fp32-transit / persistent structure / vbrc). Inputs: x/residual (M, N)
    same dtype; weight/bias (N,); output ``(y, residual_out)`` same shape /
    dtype as x. Runs on CPU and does not require an NPU device.
    """
    N = x.shape[-1]
    added = (x.float() + residual.float()).to(x.dtype)
    y = F.layer_norm(added.float(), (N,), weight.float(), bias.float(), eps).to(x.dtype)
    return y, added


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _default_block_m(N):
    """UB-budget default block_m (DESIGN.md section 4.5 formula, with the
    measured 20 B/elem multi-buffered budget; see Constants)."""
    return min(_BLOCK_M_CAP, max(1, _UB_MAX_ELEMS // N))


def _fused_add_layer_norm_kernel(M, N, eps, dtype, block_m=None):
    """Build the fused add + LayerNorm forward kernel factory.

    Mirrors the source signature ``(M, N, eps, dtype)``. Returns a JIT
    callable taking a single ``block_m`` (threads removed, DESIGN.md
    section 3.3 note), which compiles and returns ``main``;
    ``main(x, residual, weight, bias)`` returns ``(output, residual_out)``
    (``out_idx=[4, 5]``, source contract). ``block_m`` may be pre-bound at
    the factory level; otherwise the UB-budget default governs.
    """
    if M < 1 or N < 1:
        raise ValueError(f"M and N must be positive, got M={M}, N={N}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )

    # O1 (DESIGN.md section 1.6.1): compile-time reciprocal, used as the
    # scalar src2 of T.vmul (division -> multiplication by fl(1/N)).
    N_inv = 1.0 / float(N)

    # R4: fp32 transit for both half-precision dtypes (bf16 has no Vector
    # arithmetic; fp16 transit matches the golden's fp32 opmath rounding
    # path, TRAP-fp16-opmath-golden).
    use_fp32_transit = dtype in ("float16", "bfloat16")
    block_m_default = _default_block_m(N) if block_m is None else block_m

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op; measured 20 -> 40 on this device). Re-queried per
    # factory call so deployments on other devices adapt automatically.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    @tilelang.jit(out_idx=[4, 5], target="npuir")
    def _func(block_m=block_m_default):
        # Host-side UB budget guard (DESIGN.md sections 3.3 and 9.2, with
        # the measured 20 B/elem multi-buffered budget; see Constants).
        # ValueError (not a bare assert) so the guard survives `python -O`.
        if block_m < 1:
            raise ValueError(f"block_m must be positive, got block_m={block_m}")
        if block_m * N > _UB_MAX_ELEMS:
            raise ValueError(
                f"block_m={block_m} exceeds the UB budget for N={N} "
                f"(max {_UB_MAX_ELEMS} resident elements at "
                f"{_UB_BYTES_PER_ELEM} B/elem multi-buffered, "
                f"{_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # Persistent core split (DESIGN.md section 5.5). All quantities are
        # compile-time Python constants (fold to IntImm at trace time,
        # PL-1.5); the in-core serial bound is therefore static.
        num_logical = (M + block_m - 1) // block_m
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

        # Compile-time dtype dispatch outside the traced body (avoids the
        # TVM-script parser if-block variable-table scoping limitation,
        # pattern-library C11; ada/lerp same form).
        if use_fp32_transit:

            if block_m == 1:
                # block_m == 1 (N=8192, llama-70b-prefill): single
                # staging. loads-first adds a per-input buffer that
                # tips UB over (ub overflow by 256 bits) -- keep the
                # original vcast single-staging chain here.
                @T.prim_func
                def main(
                    x: T.Tensor[(M, N), dtype],
                    residual: T.Tensor[(M, N), dtype],
                    weight: T.Tensor[(N,), dtype],
                    bias: T.Tensor[(N,), dtype],
                    output: T.Tensor[(M, N), dtype],       # out_idx=4
                    residual_out: T.Tensor[(M, N), dtype],  # out_idx=5
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        # UB buffers (DESIGN.md section 4.3), allocated once per
                        # core and reused across serial tasks:
                        #   stage : x -> residual -> rounded sum -> rounded y
                        #           (4 roles; lifetimes disjoint)
                        #   x_f32 : x32 -> s -> s32 -> d (kept, O3) -> t result
                        #   sq_f32: r32 -> d^2 -> weight brc -> bias brc
                        #   wstage: (1, N) weight/bias load staging (its own row,
                        #           not a sub-row slice of stage -- a row slice of
                        #           a (block_m, N) buffer lowers to a transposing
                        #           vcast on this toolchain)
                        stage = T.alloc_shared((block_m, N), dtype)
                        x_f32 = T.alloc_shared((block_m, N), "float32")
                        sq_f32 = T.alloc_shared((block_m, N), "float32")
                        wstage = T.alloc_shared((1, N), dtype)
                        w1 = T.alloc_shared((1, N), "float32")
                        b1 = T.alloc_shared((1, N), "float32")
                        s1 = T.alloc_shared((block_m, 1), "float32")
                        mean = T.alloc_shared((block_m, 1), "float32")
                        s2 = T.alloc_shared((block_m, 1), "float32")
                        var = T.alloc_shared((block_m, 1), "float32")
                        rstd = T.alloc_shared((block_m, 1), "float32")

                        # A10: weight/bias are loop-invariant -- load + upcast once
                        # per core instead of once per tile (saves 2 GM copies + 2
                        # vcasts per iteration).
                        T.copy(weight[0:N], wstage[0:1, 0:N])
                        T.vcast(wstage, w1, round_mode="rint")
                        T.copy(bias[0:N], wstage[0:1, 0:N])
                        T.vcast(wstage, b1, round_mode="rint")

                        # Persistent in-core serial task loop (static bound,
                        # grid-stride mapping, if-guard masks out-of-range row
                        # blocks).
                        for i in T.serial(num_local_tasks):
                            block_id = i * num_kernels + cid
                            if block_id < num_logical:
                                off_m = block_id * block_m
                                # Tail row count. The v-ops below run on the full
                                # (block_m, N) buffers; garbage tail rows are
                                # row-isolated (reduce dim=1 / row broadcast) and
                                # never copied out.
                                real_m = T.min(block_m, M - off_m)

                                # --- 1. Load + upcast + residual add (R4) ---
                                # Dual explicit slices, dst zero-origin
                                # (TRAP-T-copy-region-semantics /
                                # TRAP-UB-dst-align recommended form).
                                T.copy(
                                    x[off_m : off_m + real_m, 0:N],
                                    stage[0:real_m, 0:N],
                                )
                                T.vcast(stage, x_f32, round_mode="rint")
                                T.copy(
                                    residual[off_m : off_m + real_m, 0:N],
                                    stage[0:real_m, 0:N],
                                )
                                T.vcast(stage, sq_f32, round_mode="rint")
                                T.vadd(x_f32, sq_f32, x_f32)  # s = x + residual

                                # --- 2. Rounded sum (residual_out) + re-enter fp32 (O4) ---
                                T.vcast(x_f32, stage, round_mode="rint")  # round_dtype(s)
                                T.copy(
                                    stage[0:real_m, 0:N],
                                    residual_out[off_m : off_m + real_m, 0:N],
                                )
                                T.vcast(stage, x_f32, round_mode="rint")  # s32

                                # --- 3. Mean (O1: S1 * fl(1/N)) ---
                                T.reduce_sum(x_f32, s1, dim=1)
                                T.vmul(s1, N_inv, mean)

                                # --- 4. Centered two-pass variance (d kept, O3) ---
                                T.vsub(x_f32, mean, x_f32)     # d = s - mean
                                T.vmul(x_f32, x_f32, sq_f32)   # sq = d^2
                                T.reduce_sum(sq_f32, s2, dim=1)
                                T.vmul(s2, N_inv, var)
                                T.vadd(var, eps, var)
                                # rstd = 1/sqrt(var): vsqrt+vdiv composition (the
                                # raw T.vrsqrt instruction is approximate on this
                                # toolchain; see module Implementation Notes).
                                T.vsqrt(var, s2)              # s2 reused: sqrt(var)
                                T.vdiv(s2, var, rstd)         # rstd = sqrt(var)/var

                                # --- 5. Normalize + affine (column factors, R5) ---
                                T.vmul(x_f32, rstd, x_f32)     # d * rstd
                                if block_m == 1:
                                    T.vmul(x_f32, w1, x_f32)
                                else:
                                    T.vbrc(w1, sq_f32)
                                    T.vmul(x_f32, sq_f32, x_f32)
                                if block_m == 1:
                                    T.vadd(x_f32, b1, x_f32)
                                else:
                                    T.vbrc(b1, sq_f32)
                                    T.vadd(x_f32, sq_f32, x_f32)

                                # --- 6. Single round-back + store (valid rows only) ---
                                T.vcast(x_f32, stage, round_mode="rint")
                                T.copy(
                                    stage[0:real_m, 0:N],
                                    output[off_m : off_m + real_m, 0:N],
                                )

            else:
                # block_m >= 2 (N=4096, llama-8b-prefill): loads-first
                # copy-cast (PL-1.10 decoupling; -18% measured).
                @T.prim_func
                def main(
                    x: T.Tensor[(M, N), dtype],
                    residual: T.Tensor[(M, N), dtype],
                    weight: T.Tensor[(N,), dtype],
                    bias: T.Tensor[(N,), dtype],
                    output: T.Tensor[(M, N), dtype],       # out_idx=4
                    residual_out: T.Tensor[(M, N), dtype],  # out_idx=5
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        # UB buffers (DESIGN.md section 4.3), allocated once per
                        # core and reused across serial tasks:
                        #   stage : rounded sum -> rounded y (2 roles, lifetimes
                        #           disjoint; round-trip + output staging only)
                        #   x_f32 : x (copy-cast load) -> s -> s32 -> d -> t
                        #   sq_f32: residual (copy-cast load) -> d^2 -> w/b brc
                        #   w1/b1 : (1, N) fp32 weight/bias (direct copy-cast load,
                        #           no (1,N) dtype staging -- upcast is exact)
                        stage = T.alloc_shared((block_m, N), dtype)
                        x_f32 = T.alloc_shared((block_m, N), "float32")
                        sq_f32 = T.alloc_shared((block_m, N), "float32")
                        w1 = T.alloc_shared((1, N), "float32")
                        b1 = T.alloc_shared((1, N), "float32")
                        s1 = T.alloc_shared((block_m, 1), "float32")
                        mean = T.alloc_shared((block_m, 1), "float32")
                        s2 = T.alloc_shared((block_m, 1), "float32")
                        var = T.alloc_shared((block_m, 1), "float32")
                        rstd = T.alloc_shared((block_m, 1), "float32")

                        # A10: weight/bias are loop-invariant -- load + upcast once
                        # per core instead of once per tile. The (N,)->(1,N) copy is
                        # a direct fp16->fp32 upcast (exact; no rounding), so the
                        # dtype staging buffer is dropped entirely.
                        T.copy(weight[0:N], w1[0:1, 0:N])
                        T.copy(bias[0:N], b1[0:1, 0:N])

                        # Persistent in-core serial task loop (static bound,
                        # grid-stride mapping, if-guard masks out-of-range row
                        # blocks).
                        for i in T.serial(num_local_tasks):
                            block_id = i * num_kernels + cid
                            if block_id < num_logical:
                                off_m = block_id * block_m
                                # Tail row count. The v-ops below run on the full
                                # (block_m, N) buffers; garbage tail rows are
                                # row-isolated (reduce dim=1 / row broadcast) and
                                # never copied out.
                                real_m = T.min(block_m, M - off_m)

                                # --- 1. Load + residual add (R4) ---
                                # PL-1.10 loads-first: both inputs front-loaded
                                # directly into their fp32 buffers via a copy-cast
                                # (fp16->fp32 upcast is exact; no rounding), so the
                                # single-staging reuse chain (MTE2(x)->VEC(vcast)->
                                # MTE2(residual)->VEC) is removed entirely -- the
                                # two loads form a pure MTE2 phase, then the vector
                                # chain runs uninterrupted.
                                T.copy(
                                    x[off_m : off_m + real_m, 0:N],
                                    x_f32[0:real_m, 0:N],
                                )
                                T.copy(
                                    residual[off_m : off_m + real_m, 0:N],
                                    sq_f32[0:real_m, 0:N],
                                )
                                T.vadd(x_f32, sq_f32, x_f32)  # s = x + residual

                                # --- 2. Rounded sum (residual_out) + re-enter fp32 (O4) ---
                                T.vcast(x_f32, stage, round_mode="rint")  # round_dtype(s)
                                T.copy(
                                    stage[0:real_m, 0:N],
                                    residual_out[off_m : off_m + real_m, 0:N],
                                )
                                T.vcast(stage, x_f32, round_mode="rint")  # s32

                                # --- 3. Mean (O1: S1 * fl(1/N)) ---
                                T.reduce_sum(x_f32, s1, dim=1)
                                T.vmul(s1, N_inv, mean)

                                # --- 4. Centered two-pass variance (d kept, O3) ---
                                T.vsub(x_f32, mean, x_f32)     # d = s - mean
                                T.vmul(x_f32, x_f32, sq_f32)   # sq = d^2
                                T.reduce_sum(sq_f32, s2, dim=1)
                                T.vmul(s2, N_inv, var)
                                T.vadd(var, eps, var)
                                # rstd = 1/sqrt(var): vsqrt+vdiv composition (the
                                # raw T.vrsqrt instruction is approximate on this
                                # toolchain; see module Implementation Notes).
                                T.vsqrt(var, s2)              # s2 reused: sqrt(var)
                                T.vdiv(s2, var, rstd)         # rstd = sqrt(var)/var

                                # --- 5. Normalize + affine (column factors, R5) ---
                                T.vmul(x_f32, rstd, x_f32)     # d * rstd
                                if block_m == 1:
                                    T.vmul(x_f32, w1, x_f32)
                                else:
                                    T.vbrc(w1, sq_f32)
                                    T.vmul(x_f32, sq_f32, x_f32)
                                if block_m == 1:
                                    T.vadd(x_f32, b1, x_f32)
                                else:
                                    T.vbrc(b1, sq_f32)
                                    T.vadd(x_f32, sq_f32, x_f32)

                                # --- 6. Single round-back + store (valid rows only) ---
                                T.vcast(x_f32, stage, round_mode="rint")
                                T.copy(
                                    stage[0:real_m, 0:N],
                                    output[off_m : off_m + real_m, 0:N],
                                )

        else:

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                residual: T.Tensor[(M, N), dtype],
                weight: T.Tensor[(N,), dtype],
                bias: T.Tensor[(N,), dtype],
                output: T.Tensor[(M, N), dtype],
                residual_out: T.Tensor[(M, N), dtype],
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # fp32 path (DESIGN.md section 4.3): no transit buffers,
                    # same-dtype direct copies, no vcast.
                    x_f32 = T.alloc_shared((block_m, N), "float32")   # s -> d -> t
                    sq_f32 = T.alloc_shared((block_m, N), "float32")  # r32 -> d^2 -> w/b brc
                    w1 = T.alloc_shared((1, N), "float32")
                    s1 = T.alloc_shared((block_m, 1), "float32")
                    mean = T.alloc_shared((block_m, 1), "float32")
                    s2 = T.alloc_shared((block_m, 1), "float32")
                    var = T.alloc_shared((block_m, 1), "float32")
                    rstd = T.alloc_shared((block_m, 1), "float32")

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)

                            T.copy(
                                x[off_m : off_m + real_m, 0:N],
                                x_f32[0:real_m, 0:N],
                            )
                            T.copy(
                                residual[off_m : off_m + real_m, 0:N],
                                sq_f32[0:real_m, 0:N],
                            )
                            T.vadd(x_f32, sq_f32, x_f32)  # s
                            T.copy(
                                x_f32[0:real_m, 0:N],
                                residual_out[off_m : off_m + real_m, 0:N],
                            )
                            T.reduce_sum(x_f32, s1, dim=1)
                            T.vmul(s1, N_inv, mean)
                            T.vsub(x_f32, mean, x_f32)
                            T.vmul(x_f32, x_f32, sq_f32)
                            T.reduce_sum(sq_f32, s2, dim=1)
                            T.vmul(s2, N_inv, var)
                            T.vadd(var, eps, var)
                            T.vsqrt(var, s2)
                            T.vdiv(s2, var, rstd)
                            T.vmul(x_f32, rstd, x_f32)
                            T.copy(weight[0:N], w1[0:1, 0:N])  # same-dtype direct
                            if block_m == 1:
                                T.vmul(x_f32, w1, x_f32)
                            else:
                                T.vbrc(w1, sq_f32)
                                T.vmul(x_f32, sq_f32, x_f32)
                            T.copy(bias[0:N], w1[0:1, 0:N])
                            if block_m == 1:
                                T.vadd(x_f32, w1, x_f32)
                            else:
                                T.vbrc(w1, sq_f32)
                                T.vadd(x_f32, sq_f32, x_f32)
                            T.copy(
                                x_f32[0:real_m, 0:N],
                                output[off_m : off_m + real_m, 0:N],
                            )

        return main

    return _func


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------

_FACTORY_CACHE = {}


def _get_factory(M, N, eps, dtype_str):
    """Cache factory instances (mirrors the wrapper's one-factory-per-op
    usage; the jit wrapper then caches compiled mains per block_m)."""
    key = (M, N, eps, dtype_str)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _fused_add_layer_norm_kernel(M, N, eps, dtype_str)
    return _FACTORY_CACHE[key]


# Per-dtype worst max_diff across the whole run (for the final summary).
_STATS_MAX_DIFF = {}


def _compare(out, ref, atol, rtol, equal_nan=False):
    """Return (max_diff, violation_count) between out and ref.

    NaN/Inf aware: positions where both are NaN count as equal when
    equal_nan is set; equal infinities count as equal.
    """
    eq = out == ref
    if equal_nan:
        eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    max_diff = diff.max().item()
    violations = int((diff > (atol + rtol * ref.float().abs())).sum().item())
    return max_diff, violations


def _run_case(
    dtype_str,
    M=None,
    N=None,
    block_m=None,
    tag="L0",
    x=None,
    residual=None,
    weight=None,
    bias=None,
    eps=1e-5,
    equal_nan=False,
):
    """Run one kernel case and compare both outputs against the golden.

    Tensors may be pre-built as CPU tensors (special-value cases); by
    default randn inputs are generated on CPU, cast, and moved to NPU
    (device-independent data generation). Raises AssertionError on
    tolerance violation (either output, independently).
    """
    torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
    if x is None:
        x = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
        residual = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
        weight = torch.randn(N, dtype=torch.float32).to(torch_dtype)
        bias = torch.randn(N, dtype=torch.float32).to(torch_dtype)
    else:
        assert residual is not None and weight is not None and bias is not None, (
            "need all four inputs"
        )
        M, N = x.shape[0], x.shape[1]
        assert x.dtype == torch_dtype, f"input dtype {x.dtype} != expected {torch_dtype}"

    if block_m is None:
        block_m = _default_block_m(N)

    x_npu = x.npu()
    residual_npu = residual.npu()
    weight_npu = weight.npu()
    bias_npu = bias.npu()

    kernel = _get_factory(M, N, eps, dtype_str)(block_m)
    y_out, residual_out = kernel(x_npu, residual_npu, weight_npu, bias_npu)
    ref_y, ref_res = golden_fused_add_layer_norm(x, residual, weight, bias, eps)

    max_diff_all = 0.0
    for name, out, ref in (
        ("output", y_out.cpu(), ref_y),
        ("residual_out", residual_out.cpu(), ref_res),
    ):
        max_diff, violations = _compare(out, ref, atol, rtol, equal_nan=equal_nan)
        if max_diff == max_diff:  # NaN guard (all-equal cases give 0.0)
            if max_diff > max_diff_all:
                max_diff_all = max_diff
            best = _STATS_MAX_DIFF.get(dtype_str, 0.0)
            if max_diff > best:
                _STATS_MAX_DIFF[dtype_str] = max_diff
        try:
            torch.testing.assert_close(
                out, ref, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
        except AssertionError:
            print(
                f"[{tag}] FAIL: shape=({M},{N}) dtype={dtype_str} "
                f"block_m={block_m} output={name} max_diff={max_diff:.3e} "
                f"violations={violations}/{out.numel()}"
            )
            raise
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} block_m={block_m} "
        f"max_diff={max_diff_all:.3e}"
    )
    return max_diff_all


def _try_case(failures, case_id, **kwargs):
    """Run _run_case, collecting (not re-raising) per-case failures.

    AssertionError is treated as a precision failure; any other exception
    is recorded as a runtime error (both block the gate).
    """
    try:
        _run_case(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


def _run_L0_3d_reshape(failures):
    """L0-11: 3D input (2, 512, 4096) fp16 via the Op-layer reshape path.

    The kernel is 2D (M, N); the Op layer reshapes any leading dims to
    (M, N) as a host metadata view (DESIGN.md sections 4.6 and 9.3). The
    golden runs on the original 3D tensors (last-dim LayerNorm).
    """
    case_id = "L0-11/3d-reshape"
    try:
        torch_dtype, atol, rtol = _TOLERANCE["float16"]
        B, S, H = 2, 512, 4096
        M = B * S
        x3 = torch.randn(B, S, H, dtype=torch.float32).to(torch_dtype)
        r3 = torch.randn(B, S, H, dtype=torch.float32).to(torch_dtype)
        w3 = torch.randn(H, dtype=torch.float32).to(torch_dtype)
        b3 = torch.randn(H, dtype=torch.float32).to(torch_dtype)
        # Host metadata view: contiguous reshape, no data movement.
        kernel = _get_factory(M, H, 1e-5, "float16")(2)
        y2, r2 = kernel(x3.reshape(M, H).npu(), r3.reshape(M, H).npu(),
                        w3.npu(), b3.npu())
        y = y2.cpu().reshape(B, S, H)
        rout = r2.cpu().reshape(B, S, H)
        ref_y, ref_res = golden_fused_add_layer_norm(x3, r3, w3, b3, 1e-5)
        max_diff = 0.0
        for name, out, ref in (("output", y, ref_y), ("residual_out", rout, ref_res)):
            md, violations = _compare(out, ref, atol, rtol)
            max_diff = max(max_diff, md)
            torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        best = _STATS_MAX_DIFF.get("float16", 0.0)
        if max_diff > best:
            _STATS_MAX_DIFF["float16"] = max_diff
        print(
            f"[L0-11] PASS: shape=(2,512,4096)->(1024,4096) dtype=float16 "
            f"block_m=2 max_diff={max_diff:.3e}"
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, all 13 items).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # (M, N, dtype, block_m, case_id) -- DESIGN.md section 8.3 table.
    # L0-1 note: the design table lists bm=3 for fp32 N=4096, which the
    # measured 20 B/elem multi-buffer budget rejects (3*4096 = 12288 >
    # 9216); per the DESIGN.md section 9.1 contingency the case runs at
    # bm=2 (same as the ada precedent).
    # L0-10 note: same for fp32 N=5120 (2*5120 = 10240 > 9216) -> bm=1.
    cases = [
        (64, 4096, "float32", 2, "L0-1"),  # aligned fp32 direct path
        (64, 4096, "float16", 2, "L0-2"),  # aligned fp32-transit path
        (64, 4096, "bfloat16", 2, "L0-3"),  # bf16 -> fp32 -> bf16 path
        (1025, 4096, "float16", 2, "L0-4"),  # M tail (1025 % 2 = 1)
        (1025, 4096, "bfloat16", 2, "L0-5"),  # M tail + bf16
        (2048, 4096, "float16", 2, "L0-6"),  # manifest llama-8b-prefill
        (1, 4096, "bfloat16", 2, "L0-7"),  # manifest llama-8b-decode (M=1)
        (2048, 8192, "float16", 1, "L0-8"),  # manifest llama-70b-prefill (bm=1)
        (1, 8192, "bfloat16", 1, "L0-9"),  # manifest llama-70b-decode
        (2048, 5120, "float32", 1, "L0-10"),  # manifest llama-13b-prefill (non-pow2 N)
        (16, 1152, "float16", 8, "L0-12"),  # naturally unaligned N (defensive)
        (1, 2, "float32", 1, "L0-13"),  # tiny shape (N=2, defensive)
    ]
    for M, N, dtype_str, block_m, case_id in cases:
        _try_case(
            failures,
            case_id,
            M=M,
            N=N,
            dtype_str=dtype_str,
            block_m=block_m,
            tag=case_id,
        )

    # L0-11: 3D input via the Op-layer reshape path.
    _run_L0_3d_reshape(failures)

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (13 items, 13 runs)")
    return failures


def _run_L1_contract(failures):
    """L1 contract checks (factory guards, no precision comparison)."""
    # UB budget guard rejects over-budget block_m values (measured
    # 20 B/elem multi-buffered budget, see module Constants).
    for M, N, dtype_str, bm in (
        (64, 4096, "float32", 3),  # 3*4096 = 12288 > 9216
        (64, 4096, "float16", 3),  # 12288 > 9216
        (64, 8192, "float32", 2),  # 16384 > 9216
    ):
        try:
            _fused_add_layer_norm_kernel(M, N, 1e-5, dtype_str)(bm)
        except ValueError as exc:
            print(f"[L1] PASS: UB guard rejects {dtype_str} bm={bm} N={N} ({exc})")
        else:
            failures.append(
                (
                    f"L1/UB-guard/{dtype_str}/bm={bm}",
                    AssertionError(f"{dtype_str} bm={bm} not rejected"),
                )
            )

    # Out-of-domain N (beyond the residency bound) is intercepted by the
    # same guard via the default block_m (DESIGN.md section 9.3). The guard
    # fires when _func(block_m) is invoked, not at factory build.
    try:
        _fused_add_layer_norm_kernel(4, 20000, 1e-5, "float16")()
    except ValueError as exc:
        print(f"[L1] PASS: domain guard rejects N=20000 fp16 ({exc})")
    else:
        failures.append(
            ("L1/domain-guard-N20000", AssertionError("N=20000 fp16 not rejected"))
        )

    # Non-positive M / N are rejected.
    for bad_m, bad_n in ((0, 1152), (64, 0)):
        try:
            _fused_add_layer_norm_kernel(bad_m, bad_n, 1e-5, "float16")
        except ValueError:
            print(f"[L1] PASS: M/N guard rejects M={bad_m}, N={bad_n}")
        else:
            failures.append(
                (
                    f"L1/mn-guard/{bad_m}x{bad_n}",
                    AssertionError(f"M={bad_m}, N={bad_n} not rejected"),
                )
            )

    # Unsupported dtype is rejected.
    try:
        _fused_add_layer_norm_kernel(64, 1152, 1e-5, "int32")
    except ValueError:
        print("[L1] PASS: dtype guard rejects int32")
    else:
        failures.append(("L1/dtype-guard", AssertionError("int32 not rejected")))


def run_L1():
    """L1: full functional coverage (dtype x shape x block_m + contracts).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # dtype x {aligned, M-tail, unaligned-N, tiny} grid.
    for dtype_str in ("float32", "float16", "bfloat16"):
        for M, N, bm in ((128, 512, 4), (130, 256, 8), (33, 1024, 2), (17, 514, 4)):
            _try_case(
                failures,
                f"L1/{dtype_str}/{M}x{N}/bm={bm}",
                M=M,
                N=N,
                dtype_str=dtype_str,
                block_m=bm,
                tag="L1",
            )

    # Manifest workloads: large M (persistent core split), the non-power-of-
    # two N and the B=1 decode shapes; bf16 + fp32 variants.
    for dtype_str, M, N, bm in (
        ("float16", 2048, 4096, 2),   # llama-8b-prefill
        ("bfloat16", 2048, 4096, 2),
        ("float32", 2048, 5120, 1),   # llama-13b-prefill (non-divisible N)
        ("float16", 2048, 5120, 1),   # non-divisible N + transit
        ("float16", 2048, 8192, 1),   # llama-70b-prefill
        ("bfloat16", 2048, 8192, 1),
        ("float32", 2048, 4096, 2),
        ("float16", 1, 4096, 2),      # B=1 decode (single row)
        ("bfloat16", 1, 8192, 1),     # B=1 decode large N
        ("float32", 1, 5120, 1),      # B=1 + non-divisible N
    ):
        _try_case(
            failures,
            f"L1/{dtype_str}/{M}x{N}/bm={bm}",
            M=M,
            N=N,
            dtype_str=dtype_str,
            block_m=bm,
            tag="L1",
        )

    # Persistent core-count switch boundary (num_logical around 40 cores;
    # 3*... shapes with block_m=8 -> 5 logical tasks, 1 core each).
    for M, bm in ((1, 8), (8, 8), (40, 8), (320, 8), (321, 8)):
        _try_case(
            failures,
            f"L1/persistent/M={M}/bm={bm}",
            M=M,
            N=1152,
            dtype_str="float16",
            block_m=bm,
            tag="L1",
        )

    # block_m sweep on one shape (wrapper config coverage; bm==1 exercises
    # the same-shape affine branch, bm>=2 the vbrc expand branch).
    for bm in (1, 2, 4, 8):
        _try_case(
            failures,
            f"L1/fp16/96x1152/bm={bm}",
            M=96,
            N=1152,
            dtype_str="float16",
            block_m=bm,
            tag="L1",
        )

    # Factory-default block_m path (UB-budget table value).
    _try_case(
        failures,
        "L1/fp16/50x3000/default-bm",
        M=50,
        N=3000,
        dtype_str="float16",
        tag="L1",
    )
    _try_case(
        failures,
        "L1/fp32/50x3000/default-bm",
        M=50,
        N=3000,
        dtype_str="float32",
        tag="L1",
    )

    # Non-default eps (exercises the eps plumbing: var + eps placement).
    _try_case(
        failures,
        "L1/fp16/64x1152/eps=1e-2",
        M=64,
        N=1152,
        dtype_str="float16",
        block_m=8,
        eps=1e-2,
        tag="L1",
    )

    # E2 factory contract: one factory callable serving two block_m values
    # (the jit wrapper caches the compiled main per block_m).
    try:
        dtype_str = "float16"
        torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
        M, N = 96, 1152
        x = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
        residual = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
        weight = torch.randn(N, dtype=torch.float32).to(torch_dtype)
        bias = torch.randn(N, dtype=torch.float32).to(torch_dtype)
        ref_y, ref_res = golden_fused_add_layer_norm(x, residual, weight, bias, 1e-5)
        fn = _get_factory(M, N, 1e-5, dtype_str)
        for bm in (8, 4):
            y, r = fn(bm)(x.npu(), residual.npu(), weight.npu(), bias.npu())
            torch.testing.assert_close(y.cpu(), ref_y, rtol=rtol, atol=atol)
            torch.testing.assert_close(r.cpu(), ref_res, rtol=rtol, atol=atol)
        print("[L1] PASS: E2 factory(block_m) dual-config reuse")
    except AssertionError as exc:
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse RUNTIME-ERROR: {exc!r}")

    # Factory contract guards (no NPU run).
    _run_L1_contract(failures)

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes (warn only, non-blocking)."""
    cases = [
        (1, 1, "float16", 1),  # minimal
        (2, 3, "bfloat16", 1),  # tiny odd
        (1, 4096, "float32", None),  # single row, default bm
        (3, 8192, "float16", 1),  # N beyond the manifest max
        (5000, 64, "float16", 8),  # large M, small N (deep serial)
        (1, 8192, "bfloat16", 1),  # single row bf16, large N
    ]
    for M, N, dtype_str, bm in cases:
        try:
            _run_case(
                M=M,
                N=N,
                dtype_str=dtype_str,
                block_m=bm,
                tag="L2",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record without blocking): shape=({M},{N}) {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    M, N = 32, 512

    def _inputs(x, dtype):
        residual = torch.randn(M, N, dtype=torch.float32).to(dtype)
        weight = torch.randn(N, dtype=torch.float32).to(dtype)
        bias = torch.randn(N, dtype=torch.float32).to(dtype)
        return dict(x=x, residual=residual, weight=weight, bias=bias)

    # Zeros: var = 0 -> rstd = 1/sqrt(eps).
    try:
        _run_case(dtype_str="float16", tag="Boundary-zeros",
                  **_inputs(torch.zeros(M, N, dtype=torch.float16), torch.float16))
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record without blocking): zeros: {e}")

    # Constant rows (var = 0, non-zero mean).
    try:
        _run_case(dtype_str="float32", tag="Boundary-const-rows",
                  **_inputs(torch.full((M, N), 3.7, dtype=torch.float32), torch.float32))
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record without blocking): const-rows: {e}")

    # Large fp16 magnitudes (x100 scale).
    try:
        x = (torch.randn(M, N, dtype=torch.float32) * 100.0).to(torch.float16)
        _run_case(dtype_str="float16", tag="Boundary-large", **_inputs(x, torch.float16))
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record without blocking): large: {e}")

    # Subnormal fp16 magnitudes (~1e-5 scale; exact fp32 upcast).
    try:
        x = (torch.randn(M, N, dtype=torch.float32) * 1e-5).to(torch.float16)
        _run_case(dtype_str="float16", tag="Boundary-subnormal", **_inputs(x, torch.float16))
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record without blocking): subnormal: {e}")

    # Extreme weight/bias magnitudes (fp16 max 65504).
    try:
        x = torch.randn(M, N, dtype=torch.float32).to(torch.float16)
        residual = torch.randn(M, N, dtype=torch.float32).to(torch.float16)
        weight = torch.where(
            torch.rand(N) > 0.5,
            torch.full((N,), 65504.0),
            torch.full((N,), -65504.0),
        ).to(torch.float16)
        bias = torch.where(
            torch.rand(N) > 0.5,
            torch.full((N,), 65504.0),
            torch.full((N,), -65504.0),
        ).to(torch.float16)
        _run_case(
            dtype_str="float16",
            tag="Boundary-extreme-affine",
            x=x,
            residual=residual,
            weight=weight,
            bias=bias,
            equal_nan=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record without blocking): extreme-affine: {e}")

    # inf/NaN natural propagation (DESIGN.md section 0.1: no special
    # handling; a full-inf row yields NaN output on both sides).
    try:
        x = torch.randn(M, N, dtype=torch.float32)
        x[0, :] = float("inf")
        x[1, ::4] = float("nan")
        x[2, ::4] = float("inf")
        _run_case(
            dtype_str="float16",
            tag="Boundary-inf-nan",
            equal_nan=True,
            **_inputs(x.to(torch.float16), torch.float16),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record without blocking): inf-nan: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    t0 = time.time()
    failures += run_L0()
    print(f"[L0] elapsed {time.time() - t0:.1f}s")

    if args.level == "all":
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    print(
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
