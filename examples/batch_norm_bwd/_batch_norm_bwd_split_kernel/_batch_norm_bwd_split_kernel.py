# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""BatchNorm backward split kernel for npuir (Expert / Vector-only).

Migrated from the GPU TileLang factory ``_batch_norm_bwd_split_kernel``
(``TileOPs-main/src/tileops/kernels/norm/batch_norm.py`` L1795) to
``target="npuir"`` with ``ascend_mode="Expert"`` and a single
``T.Scope("Vector")`` per stage -- no Cube, no ``T.gemm``, no L1/L0C.

Source semantics (DESIGN.md section 0.1 / 1.3), for ``grad_out, x : (N, C, S)``
and ``weight, mean, rstd : (C,)`` (fp32), with ``L = N * S`` and
``x_hat = (x - mean) * rstd`` (per channel ``c``)::

    grad_bias[c]   = sum_{l<L} grad_out[c, l]
    grad_weight[c] = sum_{l<L} grad_out[c, l] * x_hat[c, l]
    grad_x[c, l]   = w[c]*rstd[c]/L * ( L*grad_out[c, l]
                                         - grad_bias[c]
                                         - x_hat[c, l]*grad_weight[c] )

fp32 opmath throughout, a single round-back to the input dtype for grad_x;
grad_weight / grad_bias stay fp32.

NPU redesign (DESIGN.md section 0.6), three cooperating stages:

  stats    -- each channel is cut into ``splits`` balanced S-segments spread
              over a 1D persistent core grid (``min(C*splits, 40)``); each
              segment produces the two partial sums written to the GM
              workspace ``[C, splits]``.  A 2-D ``(bN, bS)`` UB tiling bounds
              the resident footprint for both very large N and very large S
              (DESIGN.md section 0.0 blocking item 3); the S-segment bounds use
              balanced (dist/rem) partition so every segment is non-empty
              (blocking item 2).
  finalize -- one persistent core pass merges the two workspace rows per
              channel with a hardware tree ``T.reduce(dims=[1])`` (no atomics)
              and pre-folds the per-channel epilogue coefficients
              ``scale / shift / centered_coef`` (O1/O2).
  apply    -- one persistent core pass maps ``grad_x`` per channel over the
              *full* S spatial axis with an independent ``n_tile_S_apply =
              ceil(S / bS)`` bound (blocking item 1).

Interface contract (source unchanged): the factory
``_batch_norm_bwd_split_kernel(N, C, S, dtype)`` returns a
``(stats, finalize, apply)`` triple of no-argument JIT factories; the
workspace ``partial_sum`` / ``partial_sum_xhat`` ``[C, splits]`` and the
coefficient planes are host-allocated.  ``apply`` keeps the source
``out_idx=[-1]`` (single output, returns ``grad_x``).

Run the embedded hierarchical tests:
    python _batch_norm_bwd_split_kernel.py --level L0
    python _batch_norm_bwd_split_kernel.py --level all
"""

import os

# Expert mode: explicit UB via T.alloc_ub, single T.Scope("Vector") per stage,
# compiler auto-sync (DESIGN.md sections 2 and 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

# Programming mode of the verified kernel (Stage 3); kept in sync with the
# ambient TILELANG_ASCEND_MODE used when running this file standalone.
ASCEND_MODE = "Expert"

import argparse
import time

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = [
    "ASCEND_MODE",
    "_batch_norm_bwd_split_kernel",
    "golden_batch_norm_bwd",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported dtypes (factory validation).
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# Source-repo split heuristic constants (wrapper batch_norm_bwd.py).
_SPLIT_TARGET_BLOCKS = 512

# UB budget (DESIGN.md sections 4.5 / 9.1).  The compiler auto multi-buffers
# the persistent serial loops, so the *resident* footprint per element exceeds
# the nominal buffer sizes.  Measured on this toolchain for the sibling wide
# kernel (same buffer set: stage + 4 fp32 planes):
#   fp16/bf16 transit path: ~22.0 B/elem  (4*2272 fp16 -> 200320 B)
#   fp32 path:              ~25.8 B/elem
# The safe budget is 160 KiB (below the 192 KiB physical capacity); the extra
# headroom covers small-bS shapes whose short trailing axis wastes vector
# lanes (e.g. S=5 with bS=5).  bS/bN are derived from this budget (DESIGN.md
# section 4.5 formula, refined multi-buffered cost).
_UB_SAFE_BYTES = 163_840
_UB_CAPACITY_BYTES = 196_608
_UB_PE_TRANSIT = 22
_UB_PE_FP32 = 26
_SMALL = 64            # (1,1) / (1,splits) small buffers margin
_ROW_OVERHEAD = 8      # red + red2 = 2 * (bN,1) fp32
_ACCUM = "float32"

# Precision tolerances (DESIGN.md section 8.2, from the source-repo
# batch_norm_bwd_tolerance).  Tuple: (torch dtype, atol, rtol, sum_atol),
# where sum_atol is the wider bound for the fp32 channel gradients.
_TOLERANCE = {
    "float32": (torch.float32, 1e-5, 1e-5, 1e-4),
    "float16": (torch.float16, 1e-2, 1e-2, 1e-2),
    "bfloat16": (torch.bfloat16, 2e-2, 2e-2, 2e-2),
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_batch_norm_bwd(grad_out, x, weight, mean, rstd):
    """PyTorch reference: BatchNorm backward (fp32 opmath + single round-back).

    Port of the source-repo independent reference
    (``examples/TileOPs/tileops/workloads/norm.py batch_norm_backward``).
    Inputs ``grad_out, x``: ``(N, C, S)`` same dtype; ``weight, mean, rstd``:
    ``(C,)`` fp32.  Returns ``(grad_x, grad_weight, grad_bias)`` with
    ``grad_x`` rounded to the input dtype and the channel grads in fp32.
    Independent of the NPU algorithm (no persistent core split / balanced
    segment partition / partial-sum workspace / finalize merge / two-step
    ``T.reduce`` / ``T.vbrc`` broadcast / fp32 transit chain).  Runs on CPU and
    does not require an NPU device.
    """
    N, C, S = grad_out.shape
    L = N * S
    shape = (1, C, 1)
    # fp32 opmath via direct torch.* module calls (S3-GOLDEN-TORCH):
    # promote both inputs with torch.zeros_like(..., dtype=torch.float32) +
    # torch.add so no tensor-method-only call remains in this body.
    go = torch.add(torch.zeros_like(grad_out, dtype=torch.float32), grad_out)
    xf = torch.add(torch.zeros_like(x, dtype=torch.float32), x)
    w = torch.reshape(weight, shape)
    m = torch.reshape(mean, shape)
    r = torch.reshape(rstd, shape)
    # x_hat = (x - mean) * rstd
    x_hat = torch.mul(torch.sub(xf, m), r)
    # grad_bias[c] = sum_{l<L} grad_out[c, l]
    grad_bias = torch.sum(go, dim=(0, 2))                      # (C,)
    # grad_weight[c] = sum_{l<L} grad_out[c, l] * x_hat[c, l]
    grad_weight = torch.sum(torch.mul(go, x_hat), dim=(0, 2))  # (C,)
    gb = torch.reshape(grad_bias, shape)
    gw = torch.reshape(grad_weight, shape)
    # scale = w * rstd / L
    scale = torch.div(torch.mul(w, r), L)
    # grad_x = scale * (L*grad_out - grad_bias - x_hat*grad_weight)
    grad_x = torch.mul(
        scale,
        torch.sub(
            torch.sub(torch.mul(go, float(L)), gb),
            torch.mul(x_hat, gw),
        ),
    )
    return grad_x.to(x.dtype), grad_weight, grad_bias


# ---------------------------------------------------------------------------
# Tiling plan helpers (single source of truth for factory and tests)
# ---------------------------------------------------------------------------
def _plan_splits(N, C, S):
    """Return the split count for this shape (== source ``_split_seed`` cap S).

    ``_split_seed = max(1, min(L, ceil(512 / C)))`` capped at ``S`` so that
    ``dist = S // splits >= 1`` and every balanced segment is non-empty
    (DESIGN.md section 5.4).
    """
    L = N * S
    splits = max(1, min(L, -(-_SPLIT_TARGET_BLOCKS // C)))
    return max(1, min(splits, S))


def _plan_tiling(N, S, splits, use_transit):
    """Return the 2-D ``(bN, bS)`` tiling and static loop bounds.

    ``bS`` is chosen so that a two-row ``(2, bS)`` UB plane fits the budget
    (``N > 1``; a single row when ``N == 1``); ``bN`` then fills the rest.
    ``n_tile_S_stats`` (segment bound) and ``n_tile_S_apply`` (full-S bound)
    are defined independently (DESIGN.md section 0.0 blocking item 1).
    """
    pe = _UB_PE_TRANSIT if use_transit else _UB_PE_FP32
    vw = 8 if use_transit else 4
    dist = S // splits
    rem = S % splits
    max_seg = dist + (1 if rem > 0 else 0)
    if N == 1:
        bN = 1
        bS = max(vw, (((_UB_SAFE_BYTES - _SMALL) // pe) // vw) * vw)
    else:
        bS = max(vw, (((_UB_SAFE_BYTES - _SMALL) // 2) // pe // vw) * vw)
        bS = min(bS, S)
        bN = min(
            N,
            max(2, (_UB_SAFE_BYTES - _SMALL) // (bS * pe + _ROW_OVERHEAD)),
        )
    bS = min(bS, S)
    n_tile_N = -(-N // bN)
    n_tile_S_stats = -(-max_seg // bS)
    n_tile_S_apply = -(-S // bS)
    return {
        "pe": pe,
        "dist": dist,
        "rem": rem,
        "max_seg": max_seg,
        "bN": bN,
        "bS": bS,
        "n_tile_N": n_tile_N,
        "n_tile_S_stats": n_tile_S_stats,
        "n_tile_S_apply": n_tile_S_apply,
    }


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _batch_norm_bwd_split_kernel(N, C, S, dtype="float16"):
    """Build the three-stage BatchNorm backward factories for an npuir target.

    Mirrors the source signature ``(N, C, S, dtype)`` and returns the
    ``(stats, finalize, apply)`` triple of no-argument JIT factories (the CUDA
    ``splits``/``threads``/``num_per_thread``/``blocks`` knobs are baked into
    the factory closures, DESIGN.md section 0.2 note).
    """
    if N < 1 or C < 1 or S < 1:
        raise ValueError(f"N, C, S must be positive, got {N}, {C}, {S}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )

    L = N * S
    L_inv = 1.0 / float(L)                                   # O1: compile-time constant
    use_transit = dtype in ("float16", "bfloat16")           # bf16 has no Vector arith

    splits = _plan_splits(N, C, S)
    plan = _plan_tiling(N, S, splits, use_transit)
    bN = plan["bN"]
    bS = plan["bS"]
    pe = plan["pe"]
    dist = plan["dist"]
    rem = plan["rem"]
    n_tile_N = plan["n_tile_N"]
    n_tile_S_stats = plan["n_tile_S_stats"]
    n_tile_S_apply = plan["n_tile_S_apply"]

    # Host-side UB budget guard (compile-time sanity; survives `python -O`).
    if bN * bS * pe + bN * _ROW_OVERHEAD + _SMALL > _UB_CAPACITY_BYTES:
        raise ValueError(
            f"planned UB footprint {bN}*{bS}*{pe}+{bN * _ROW_OVERHEAD} exceeds "
            f"capacity {_UB_CAPACITY_BYTES} B"
        )

    s_gt1 = S > 1            # dims=[1] reduce is legal
    n_gt1 = N > 1            # dims=[0] reduce is legal (bN >= 2)
    split_gt1 = splits > 1   # finalize merge reduces a real axis

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op; measured 20 -> 40 on this device).
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    n_tasks = C * splits
    KS = min(n_tasks, vector_cores)
    nlS = -(-n_tasks // KS)              # stats core-serial task count
    KC = min(C, vector_cores)
    nlC = -(-C // KC)                    # finalize/apply core-serial task count

    @tilelang.jit(target="npuir")
    def _stats():
        @T.prim_func
        def stats(
            grad_out: T.Tensor((N, C, S), dtype),
            x: T.Tensor((N, C, S), dtype),
            mean: T.Tensor((C,), "float32"),
            rstd: T.Tensor((C,), "float32"),
            partial_sum: T.Tensor((C, splits), "float32"),
            partial_sum_xhat: T.Tensor((C, splits), "float32"),
        ):
            with T.Kernel(KS, is_npu=True) as (cid, _):
                with T.Scope("Vector"):
                    stage = T.alloc_ub((bN, bS), dtype)       # g / x transit
                    g32 = T.alloc_ub((bN, bS), "float32")
                    xh32 = T.alloc_ub((bN, bS), "float32")    # x32 -> x_hat
                    prod = T.alloc_ub((bN, bS), "float32")
                    brc = T.alloc_ub((bN, bS), "float32")
                    red = T.alloc_ub((bN, 1), "float32")      # row sums (across S-tiles)
                    red2 = T.alloc_ub((bN, 1), "float32")
                    gb = T.alloc_ub((1, 1), "float32")        # segment partial (across N)
                    gw = T.alloc_ub((1, 1), "float32")
                    m1 = T.alloc_ub((1, 1), "float32")
                    r1 = T.alloc_ub((1, 1), "float32")

                    for i in T.serial(nlS):
                        t = i * KS + cid
                        if t < n_tasks:
                            c = t // splits
                            sp = t % splits
                            # Balanced segment [s0, s1): real = s1 - s0 >= 1.
                            s0 = sp * dist + T.min(sp, rem)
                            s1 = (sp + 1) * dist + T.min(sp + 1, rem)
                            T.copy(mean[c:c + 1], m1)
                            T.copy(rstd[c:c + 1], r1)
                            T.clear(gb)
                            T.clear(gw)
                            for nn in T.serial(n_tile_N):
                                n0 = nn * bN
                                rN = T.min(bN, N - n0)        # >= 1
                                T.clear(red)
                                T.clear(red2)
                                for tt in T.serial(n_tile_S_stats):
                                    ss = s0 + tt * bS
                                    r2 = T.min(bS, s1 - ss)   # tail tile may empty
                                    if r2 > 0:
                                        T.copy(
                                            grad_out[n0:n0 + rN, c, ss:ss + r2],
                                            stage[0:rN, 0:r2],
                                        )
                                        if use_transit:
                                            T.vcast(stage, g32, round_mode="rint")
                                        else:
                                            T.copy(stage[0:rN, 0:r2], g32[0:rN, 0:r2])
                                        T.copy(
                                            x[n0:n0 + rN, c, ss:ss + r2],
                                            stage[0:rN, 0:r2],
                                        )
                                        if use_transit:
                                            T.vcast(stage, xh32, round_mode="rint")
                                        else:
                                            T.copy(stage[0:rN, 0:r2], xh32[0:rN, 0:r2])
                                        # x_hat = (x - mean) * rstd
                                        T.vbrc(m1, brc)
                                        T.vsub(xh32, brc, xh32)
                                        T.vbrc(r1, brc)
                                        T.vmul(xh32, brc, xh32)
                                        # g * x_hat
                                        T.vmul(g32, xh32, prod)
                                        # size=[rN, r2] truncates the valid
                                        # extent; clear=False accumulates over
                                        # the S-tiles (accumulators pre-cleared).
                                        if s_gt1:
                                            T.reduce(
                                                g32, red, dims=[1],
                                                reduce_mode="sum", clear=False,
                                                size=[rN, r2],
                                            )
                                            T.reduce(
                                                prod, red2, dims=[1],
                                                reduce_mode="sum", clear=False,
                                                size=[rN, r2],
                                            )
                                        else:
                                            T.copy(g32[0:rN, 0:1], red[0:rN, 0:1])
                                            T.copy(prod[0:rN, 0:1], red2[0:rN, 0:1])
                                # Fold this N-block's row sums into (1,1).
                                if n_gt1:
                                    T.reduce(
                                        red, gb, dims=[0], reduce_mode="sum",
                                        clear=False, size=[rN, 1],
                                    )
                                    T.reduce(
                                        red2, gw, dims=[0], reduce_mode="sum",
                                        clear=False, size=[rN, 1],
                                    )
                                else:
                                    T.copy(red[0:1, 0:1], gb)
                                    T.copy(red2[0:1, 0:1], gw)
                            T.copy(gb, partial_sum[c:c + 1, sp:sp + 1])
                            T.copy(gw, partial_sum_xhat[c:c + 1, sp:sp + 1])

        return stats

    @tilelang.jit(target="npuir")
    def _finalize():
        @T.prim_func
        def finalize(
            partial_sum: T.Tensor((C, splits), "float32"),
            partial_sum_xhat: T.Tensor((C, splits), "float32"),
            weight: T.Tensor((C,), "float32"),
            rstd: T.Tensor((C,), "float32"),
            grad_weight: T.Tensor((C,), "float32"),
            grad_bias: T.Tensor((C,), "float32"),
            scale_out: T.Tensor((C,), "float32"),
            shift_out: T.Tensor((C,), "float32"),
            centered_coef_out: T.Tensor((C,), "float32"),
        ):
            with T.Kernel(KC, is_npu=True) as (cid, _):
                with T.Scope("Vector"):
                    pw = T.alloc_ub((1, splits), "float32")
                    pwx = T.alloc_ub((1, splits), "float32")
                    tot = T.alloc_ub((1, 1), "float32")
                    totx = T.alloc_ub((1, 1), "float32")
                    w1 = T.alloc_ub((1, 1), "float32")
                    r1 = T.alloc_ub((1, 1), "float32")
                    sc = T.alloc_ub((1, 1), "float32")
                    per = T.alloc_ub((1, 1), "float32")
                    sh = T.alloc_ub((1, 1), "float32")
                    ce = T.alloc_ub((1, 1), "float32")

                    for i in T.serial(nlC):
                        c = i * KC + cid
                        if c < C:
                            T.copy(partial_sum[c:c + 1, 0:splits], pw)
                            T.copy(partial_sum_xhat[c:c + 1, 0:splits], pwx)
                            if split_gt1:
                                T.reduce(pw, tot, dims=[1], reduce_mode="sum",
                                         clear=True)
                                T.reduce(pwx, totx, dims=[1], reduce_mode="sum",
                                         clear=True)
                            else:
                                T.copy(pw, tot)
                                T.copy(pwx, totx)
                            T.copy(tot, grad_bias[c:c + 1])
                            T.copy(totx, grad_weight[c:c + 1])
                            T.copy(weight[c:c + 1], w1)
                            T.copy(rstd[c:c + 1], r1)
                            T.vmul(w1, r1, sc)          # scale = w * rstd
                            T.vmul(sc, L_inv, per)      # per = scale * fl(1/L) (O1)
                            T.vmul(per, tot, sh)        #  per * gb
                            T.vmul(sh, -1.0, sh)        # shift = -per * gb
                            T.vmul(per, totx, ce)       # per * gw
                            T.vmul(ce, r1, ce)          # centered = per * gw * rstd
                            T.copy(sc, scale_out[c:c + 1])
                            T.copy(sh, shift_out[c:c + 1])
                            T.copy(ce, centered_coef_out[c:c + 1])

        return finalize

    @tilelang.jit(out_idx=[-1], target="npuir")
    def _apply():
        @T.prim_func
        def apply(
            grad_out: T.Tensor((N, C, S), dtype),
            x: T.Tensor((N, C, S), dtype),
            mean: T.Tensor((C,), "float32"),
            scale: T.Tensor((C,), "float32"),
            shift: T.Tensor((C,), "float32"),
            center: T.Tensor((C,), "float32"),
            grad_x: T.Tensor((N, C, S), dtype),
        ):
            with T.Kernel(KC, is_npu=True) as (cid, _):
                with T.Scope("Vector"):
                    stage = T.alloc_ub((bN, bS), dtype)
                    g32 = T.alloc_ub((bN, bS), "float32")
                    x32 = T.alloc_ub((bN, bS), "float32")
                    brc = T.alloc_ub((bN, bS), "float32")
                    m1 = T.alloc_ub((1, 1), "float32")
                    s1 = T.alloc_ub((1, 1), "float32")
                    h1 = T.alloc_ub((1, 1), "float32")
                    c1 = T.alloc_ub((1, 1), "float32")

                    for i in T.serial(nlC):
                        c = i * KC + cid
                        if c < C:
                            T.copy(mean[c:c + 1], m1)
                            T.copy(scale[c:c + 1], s1)
                            T.copy(shift[c:c + 1], h1)
                            T.copy(center[c:c + 1], c1)
                            for nn in T.serial(n_tile_N):
                                n0 = nn * bN
                                rN = T.min(bN, N - n0)       # >= 1
                                for tt in T.serial(n_tile_S_apply):
                                    s0 = tt * bS
                                    real = T.min(bS, S - s0)  # >= 1
                                    T.copy(
                                        grad_out[n0:n0 + rN, c, s0:s0 + real],
                                        stage[0:rN, 0:real],
                                    )
                                    if use_transit:
                                        T.vcast(stage, g32, round_mode="rint")
                                    else:
                                        T.copy(stage[0:rN, 0:real], g32[0:rN, 0:real])
                                    T.copy(
                                        x[n0:n0 + rN, c, s0:s0 + real],
                                        stage[0:rN, 0:real],
                                    )
                                    if use_transit:
                                        T.vcast(stage, x32, round_mode="rint")
                                    else:
                                        T.copy(stage[0:rN, 0:real], x32[0:rN, 0:real])
                                    # grad_x = scale*g + shift - center*(x - mean)
                                    T.vbrc(m1, brc)
                                    T.vsub(x32, brc, x32)
                                    T.vbrc(s1, brc)
                                    T.vmul(g32, brc, g32)
                                    T.vbrc(h1, brc)
                                    T.vadd(g32, brc, g32)
                                    T.vbrc(c1, brc)
                                    T.vmul(x32, brc, brc)
                                    T.vsub(g32, brc, g32)
                                    if use_transit:
                                        T.vcast(g32, stage, round_mode="rint")
                                    else:
                                        T.copy(g32[0:rN, 0:real], stage[0:rN, 0:real])
                                    T.copy(
                                        stage[0:rN, 0:real],
                                        grad_x[n0:n0 + rN, c, s0:s0 + real],
                                    )

        return apply

    return _stats, _finalize, _apply


# ---------------------------------------------------------------------------
# Precision comparison + hierarchical testing
# ---------------------------------------------------------------------------

_FACTORY_CACHE = {}

# Per-(layer, dtype) worst max_abs_err over the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_factory(N, C, S, dtype_str):
    key = (N, C, S, dtype_str)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _batch_norm_bwd_split_kernel(N, C, S, dtype_str)
    return _FACTORY_CACHE[key]


def _compare(out, ref, atol, rtol, equal_nan=False):
    """Return (max_abs_err, violation_count) between out and ref."""
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


def _run_split(grad_out, x, weight, mean, rstd):
    """Orchestrate the three stages exactly as the NPU wrapper does."""
    N, C, S = grad_out.shape
    stats, finalize, apply_ = _get_factory(
        N, C, S, str(grad_out.dtype).replace("torch.", "")
    )
    splits = _plan_splits(N, C, S)
    dev = grad_out.device
    empty = lambda *shape: torch.empty(*shape, device=dev, dtype=torch.float32)
    partial_sum = empty(C, splits)
    partial_sum_xhat = empty(C, splits)
    stats()(grad_out, x, mean, rstd, partial_sum, partial_sum_xhat)
    grad_weight, grad_bias = empty(C), empty(C)
    scale, shift, centered_coef = empty(C), empty(C), empty(C)
    finalize()(
        partial_sum,
        partial_sum_xhat,
        weight,
        rstd,
        grad_weight,
        grad_bias,
        scale,
        shift,
        centered_coef,
    )
    grad_x = apply_()(grad_out, x, mean, scale, shift, centered_coef)
    return grad_x, grad_weight, grad_bias


def _run_case(N, C, S, dtype_str, tag="L0", seed=0, equal_nan=False):
    """Run one split-kernel case and compare all three outputs to the golden.

    Raises AssertionError on tolerance violation (any output, independently).
    Returns the max abs err over the three outputs.
    """
    torch_dtype, atol, rtol, sum_atol = _TOLERANCE[dtype_str]
    torch.manual_seed(seed)
    grad_out = (torch.randn(N, C, S) * 0.5).to(torch_dtype)
    x = torch.randn(N, C, S).to(torch_dtype)
    weight = torch.randn(C)
    mean = torch.randn(C)
    rstd = torch.rand(C).abs() + 0.5

    gx, gw, gb = _run_split(
        grad_out.npu(), x.npu(), weight.npu(), mean.npu(), rstd.npu()
    )
    gx, gw, gb = gx.cpu(), gw.cpu(), gb.cpu()

    ref_gx, ref_gw, ref_gb = golden_batch_norm_bwd(grad_out, x, weight, mean, rstd)

    checks = (
        ("grad_x", gx, ref_gx, atol, rtol),
        ("grad_weight", gw, ref_gw, sum_atol, rtol),
        ("grad_bias", gb, ref_gb, sum_atol, rtol),
    )

    max_diff_all = 0.0
    for name, out, ref, a, r in checks:
        max_diff, violations = _compare(out, ref, a, r, equal_nan=equal_nan)
        if max_diff == max_diff and max_diff > max_diff_all:
            max_diff_all = max_diff
        try:
            torch.testing.assert_close(
                out.float(), ref.float(), rtol=r, atol=a, equal_nan=equal_nan
            )
        except AssertionError:
            print(
                f"[{tag}] FAIL: shape=({N},{C},{S}) dtype={dtype_str} "
                f"output={name} max_abs={max_diff:.3e} "
                f"violations={violations}/{out.numel()}"
            )
            raise

    best = _STATS_MAX_DIFF.get((tag.split("-")[0], dtype_str), 0.0)
    if max_diff_all > best:
        _STATS_MAX_DIFF[(tag.split("-")[0], dtype_str)] = max_diff_all
    print(
        f"[{tag}] PASS: shape=({N},{C},{S}) dtype={dtype_str} "
        f"max_abs_err={max_diff_all:.3e}"
    )
    return max_diff_all


def _try_case(failures, case_id, **kwargs):
    """Run a case, collecting (not re-raising) per-case failures."""
    try:
        _run_case(tag=case_id, **kwargs)
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


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3) -- blocking.

    Covers fp16/bf16/fp32, the split main load, N=1 / S=1 degenerate reduce
    branches, non-divisible / prime S tail segments, and the three v1
    regression points: balanced non-empty segments (L0-11), large-N 2-D UB
    tiling (L0-12) and full-S apply coverage (L0-13).
    """
    print("== L0 (blocking) ==")
    failures = []
    cases = [
        (4, 128, 262144, "float16"),     # L0-1  split main load (large-spatial)
        (4, 64, 4096, "float16"),        # L0-2  multi-segment middle size
        (8, 8, 65536, "float16"),        # L0-3  small C, high parallelism
        (4, 128, 262144, "bfloat16"),    # L0-4  fp32 transit chain
        (4, 256, 65536, "float32"),      # L0-5  fp32 direct path
        (1, 64, 131072, "float16"),      # L0-6  N=1 degenerate dims=[0] branch
        (4, 4, 65537, "float16"),        # L0-7  prime S, tail segment
        (2, 3, 65536, "float16"),        # L0-8  small C, small N
        (4, 128, 65536, "float16"),      # L0-9  splits steady (splits=4)
    ]
    for idx, (N, C, S, dtype_str) in enumerate(cases, start=1):
        _try_case(
            failures,
            f"L0-{idx}",
            N=N,
            C=C,
            S=S,
            dtype_str=dtype_str,
            seed=idx,
        )

    # L0-10: empty tensor (L=0) is intercepted at the factory boundary (the
    # kernel must never receive S=0).  Contract check, not a run.
    try:
        _batch_norm_bwd_split_kernel(2, 8, 0, "float16")
    except ValueError:
        print("[L0-10] PASS: empty-tensor (S=0) rejected at the factory")
    else:
        failures.append(("L0-10", AssertionError("S=0 not rejected")))
        print("[collected] L0-10 FAILED: S=0 not rejected")

    # L0-11: balanced non-empty segments (blocking item 2).  splits=4, S=5 ->
    # dist=1/rem=1 -> real=[2,1,1,1]; the v0 chunk partition produced real=-1
    # on sp=3 (illegal slice).
    _try_case(failures, "L0-11", N=13108, C=128, S=5, dtype_str="float16", seed=11)

    # L0-12: large-N 2-D UB tiling (blocking item 3).  v0 (N,bS) blew the UB
    # budget for this shape; v1 N-tiling (bN, n_tile_N) must compile and pass.
    _try_case(failures, "L0-12", N=8192, C=128, S=8, dtype_str="float16", seed=12)

    # L0-13: full-S apply coverage (blocking item 1).  splits>1 large-spatial
    # explicitly compares the *entire* grad_x (v0 apply stopped at
    # ceil(dist/bS)*bS columns).
    _try_case(failures, "L0-13", N=4, C=128, S=262144, dtype_str="float16", seed=13)

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print(f"[L0] ALL PASS ({len(cases) + 4} items)")
    return failures


def _run_L1_contract(failures):
    """L1 factory guards (no precision comparison)."""
    for bad in ((0, 64, 16, "float16"), (4, 0, 16, "float16"),
                (4, 64, 0, "float16"), (4, 64, 16, "int32")):
        try:
            _batch_norm_bwd_split_kernel(*bad)
        except ValueError:
            print(f"[L1] PASS: factory guard rejects {bad}")
        else:
            failures.append(
                (f"L1/factory-guard/{bad}", AssertionError(f"{bad} not rejected"))
            )


def run_L1():
    """L1 gate suite -- blocking.  dtype x shape grid + contracts."""
    print("== L1 (blocking) ==")
    failures = []

    grid = [
        (128, 512, 512, "float32"),      # held-ish, C > cores
        (128, 512, 512, "float16"),
        (128, 512, 512, "bfloat16"),
        (130, 256, 1024, "float16"),     # batch tail
        (33, 1024, 200, "float32"),      # C=1024 not divisible by 40
        (17, 514, 1000, "bfloat16"),     # odd C, odd spatial
        (4, 300, 4096, "float16"),       # C=300, tail cores
        (4, 64, 100000, "float16"),      # tiled, non-div tail segment
        (2, 40, 8192, "float32"),        # C == core count
        (1, 48, 20000, "float16"),       # N=1 tiled (dims=[0] skipped)
        (6, 8, 1, "float16"),            # S=1 degenerate dims=[1] branch
        (3, 5, 7, "bfloat16"),           # tiny odd everything
        (2, 3, 3, "float16"),            # very tiny
    ]
    for i, (N, C, S, dtype_str) in enumerate(grid):
        _try_case(failures, f"L1-{i}", N=N, C=C, S=S, dtype_str=dtype_str,
                  seed=1000 + i)

    _run_L1_contract(failures)

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes (warn only, non-blocking)."""
    print("== L2 (warn-only) ==")
    cases = [
        (1, 1, 1, "float16"),            # minimal (1-element plane)
        (2, 3, 3, "bfloat16"),           # tiny odd
        (1, 8, 65536, "float32"),        # single batch, tiled
        (30, 96, 33, "float16"),         # large N, small S
    ]
    for N, C, S, dtype_str in cases:
        try:
            _run_case(N=N, C=C, S=S, dtype_str=dtype_str, tag="L2", seed=3000)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record without blocking): shape=({N},{C},{S}) {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    print("== Boundary (warn-only) ==")
    N, C, S = 4, 16, 256

    def _run_special(name, dtype_str, gx, xx, w, m, r):
        try:
            gx_out, gw_out, gb_out = _run_split(
                gx.npu(), xx.npu(), w.npu(), m.npu(), r.npu()
            )
            gx_out, gw_out, gb_out = gx_out.cpu(), gw_out.cpu(), gb_out.cpu()
            ref_gx, ref_gw, ref_gb = golden_batch_norm_bwd(gx, xx, w, m, r)
            _, atol, rtol, sum_atol = _TOLERANCE[dtype_str]
            for out, ref, a in (
                (gx_out, ref_gx, atol),
                (gw_out, ref_gw, sum_atol),
                (gb_out, ref_gb, sum_atol),
            ):
                torch.testing.assert_close(
                    out.float(), ref.float(), rtol=rtol, atol=a, equal_nan=True
                )
            print(f"[Boundary] PASS: {name}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record without blocking): {name}: {e}")

    torch.manual_seed(0)
    base = lambda: (
        (torch.randn(N, C, S) * 0.5),
        torch.randn(N, C, S),
        torch.randn(C),
        torch.randn(C),
        (torch.rand(C).abs() + 0.5),
    )

    xx = torch.zeros(N, C, S)
    gx, _, w, m, r = base()
    _run_special("const-x", "float32", gx, xx, w, m, r)

    gx, xx, w, m, r = base()
    gx = gx * 1e3
    xx = xx * 1e-3
    _run_special("large-range", "float32", gx, xx, w, m, r)

    gx, xx, w, m, r = base()
    w = w * 1e4
    _run_special("extreme-weight", "float32", gx, xx, w, m, r)

    gx, xx, w, m, r = base()
    gx[0, 0, :] = float("inf")
    xx[1, 1, ::4] = float("nan")
    _run_special("inf-nan", "float32", gx, xx, w, m, r)


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
        "Summary max_abs_err per (layer, dtype): "
        + ", ".join(f"{k[0]}/{k[1]}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} blocking case(s) failed: "
            + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
