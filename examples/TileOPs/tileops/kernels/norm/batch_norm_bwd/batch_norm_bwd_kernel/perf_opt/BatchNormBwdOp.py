# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""BatchNorm backward kernels for npuir (Expert / Vector-only) -- merged Stage 4 module.

Single drop-in module aggregating the three tuned factory functions
``_batch_norm_bwd_kernel`` (general tiled),
``_batch_norm_bwd_wide_kernel`` (thread/block held) and
``_batch_norm_bwd_split_kernel`` (sum/merge/map), plus one shared golden and
one unified hierarchical test harness.

Round 3 (structure deep-dive) changes carried by this module:

  * wide kernel -- ``x_hat`` is eliminated from the elementwise chain: the
    per-channel (mean, rstd) factors are folded into the epilogue
    coefficients (``gw = rstd*sum(g*x) - (mean*rstd)*sum(g)`` and
    ``grad_x = A*g + B*x + C``), and both input planes get their own staging
    buffer (loads-first decoupling) with ``prod`` doubling as the vbrc
    scratch instead of a dedicated ``brc``.
  * split kernel -- per-input staging buffers in both ``stats`` and ``apply``,
    the apply epilogue is folded to ``scale*g + c3 - center*x`` (one constant
    broadcast per channel instead of subtracting ``mean`` from x every tile),
    and the apply stage gets its own tile width (it carries one extra
    in-flight staging buffer).

The three factory function bodies are byte-for-byte identical to the measured
standalone files ``perf_opt/_batch_norm_bwd_*_kernel.py``, except two
host-side renames that do NOT change the emitted NPU kernel:

  * the split kernel's tiling planner ``_plan_tiling`` is renamed
    ``_plan_tiling_split`` (name clash with the general kernel's 2-arg
    ``_plan_tiling``), and its private UB budget constant ``_UB_SAFE_BYTES``
    (163_840 B) is renamed ``_UB_SAFE_BYTES_SPLIT`` (the general/wide kernels
    use 184_320 B).  Both are host-side constant folds; the compiled kernel
    is unchanged, so the standalone ``msprof op`` measurements carry over
    without re-measurement (see opt_log.md).

Run the embedded hierarchical tests:
    python BatchNormBwdOp.py --level L0
    python BatchNormBwdOp.py --level all
"""

import os

# Expert mode: explicit UB via T.alloc_ub, single T.Scope("Vector"),
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
    "_batch_norm_bwd_kernel",
    "_batch_norm_bwd_wide_kernel",
    "_batch_norm_bwd_split_kernel",
    "golden_batch_norm_bwd",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported dtypes (factory validation).
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# UB budget (DESIGN.md sections 4.5 / 9.1).  The compiler auto multi-buffers
# the persistent serial task loop, so the *resident* footprint per element
# exceeds the nominal buffer sizes.  The general/wide kernels use a 180 KiB
# safe budget; the split kernel uses a tighter 160 KiB budget (extra headroom
# for short trailing axes that waste vector lanes).
_UB_SAFE_BYTES = 184_320
_UB_SAFE_BYTES_SPLIT = 163_840
_UB_CAPACITY_BYTES = 196_608
_UB_PE_TRANSIT = 22
_UB_PE_FP32 = 26
_ACCUM = "float32"

# Split-kernel specific constants (DESIGN.md section 5.4).
_SPLIT_TARGET_BLOCKS = 512
_SMALL = 64            # (1,1) / (1,splits) small buffers margin
# R11 stats planning budget: the conservative 163,840 left 3720 on the table;
# the backend accepts 2x22x4096 = 180,224 B, so plan stats up to capacity-8KB.
_UB_STATS_BYTES = _UB_CAPACITY_BYTES - 8_192
_ROW_OVERHEAD = 8      # red + red2 = 2 * (bN,1) fp32

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
    Independent of the NPU algorithm (no persistent core split / 2-D
    ``(bN, bS)`` tiling / two-step ``T.reduce`` / ``T.vbrc`` broadcast /
    fp32 transit chain / distributed epilogue).  Runs on CPU and does not
    require an NPU device.
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
# Tiling plan helpers
# ---------------------------------------------------------------------------
def _plan_tiling(N, S, use_transit):
    """Return ``(bpe, held, bN, bS)`` from the UB budget (DESIGN.md section 4.5).

    ``held`` (``N*S*bpe <= UB_SAFE``) keeps the whole channel plane resident
    (``bN = N``, ``bS = S``); otherwise the S axis is capped so a two-row
    ``(2, bS)`` plane fits, then ``bN`` fills the remaining budget (at least 2
    rows so the ``dims=[0]`` reduce is legal -- DESIGN.md section 0.0 blocking
    item 1).
    """
    bpe = _UB_PE_TRANSIT if use_transit else _UB_PE_FP32
    vec = 8 if use_transit else 4
    held = N * S * bpe <= _UB_SAFE_BYTES
    if held:
        bN, bS = N, S
    else:
        bS = min(S, ((_UB_SAFE_BYTES // (2 * bpe)) // vec) * vec)
        bS = max(vec, (bS // vec) * vec)
        bN = min(N, max(2, _UB_SAFE_BYTES // (bS * bpe)))
    return bpe, held, bN, bS


def _plan_bS(N, S, pe, use_transit):
    """Return the default S-tile width ``bS`` from the UB budget.

    ``bS = min(S, align_down(budget // (N * pe), vw))``: when the whole
    ``(N, S)`` plane fits (``bS == S``) the factory takes the single-scan
    (held) path; otherwise the channel is processed in ``bS``-wide tiles.
    """
    vw = 8 if use_transit else 4
    bS_cap = max(vw, ((_UB_SAFE_BYTES // (N * pe)) // vw) * vw)
    return min(S, bS_cap)


def _plan_splits(N, C, S):
    """Return the split count for this shape (== source ``_split_seed`` cap S).

    ``_split_seed = max(1, min(L, ceil(512 / C)))`` capped at ``S`` so that
    ``dist = S // splits >= 1`` and every balanced segment is non-empty
    (DESIGN.md section 5.4).
    """
    L = N * S
    splits = max(1, min(L, -(-_SPLIT_TARGET_BLOCKS // C)))
    _sp = int(os.environ.get("BN_SPLITS", "0") or "0")
    if _sp > 0:
        splits = _sp
    return max(1, min(splits, S))


def _largest_divisor_le(ub, target, vw):
    """Largest multiple of ``vw`` <= ub that divides ``target``.

    Falls back to the largest aligned value <= ub when no divisor is at least
    half of ub (avoids choosing a tiny tile just to remove a tail).
    """
    cand = (ub // vw) * vw
    if cand <= 0:
        return ub
    floor = max(vw, (cand + 1) // 2)
    for d in range(cand, floor - 1, -vw):
        if target % d == 0:
            return d
    return cand


def _plan_tiling_split(N, S, splits, use_transit):
    """Return the 2-D ``(bN, bS)`` tiling and static loop bounds.

    ``bS`` is chosen so that a two-row ``(2, bS)`` UB plane fits the budget
    (``N > 1``; a single row when ``N == 1``); ``bN`` then fills the rest.
    ``n_tile_S_stats`` (segment bound) and ``n_tile_S_apply`` (full-S bound)
    are defined independently (DESIGN.md section 0.0 blocking item 1).
    """
    pe = _UB_PE_TRANSIT if use_transit else _UB_PE_FP32
    # R3-S2 (loads-first decoupling, apply stage only): apply keeps a separate
    # GM staging buffer per input so both plane gathers can be in flight while
    # the vector chain of the previous tile runs.  The resident footprint grows
    # by one input dtype buffer (+2 B/elem for fp16/bf16, +4 for fp32), so only
    # the apply tile width shrinks -- stats keeps the wider tile.
    pe_apply = pe + (2 if use_transit else 4)
    vw = 8 if use_transit else 4
    dist = S // splits
    rem = S % splits
    max_seg = dist + (1 if rem > 0 else 0)
    if N == 1:
        bN = 1
        ub_bs = max(vw, (((_UB_STATS_BYTES - _SMALL) // pe) // vw) * vw)
        bS = _largest_divisor_le(ub_bs, max_seg, vw)
    else:
        ub_bs = max(vw, (((_UB_STATS_BYTES - _SMALL) // 2) // pe // vw) * vw)
        ub_bs = min(ub_bs, S)
        bS = _largest_divisor_le(ub_bs, max_seg, vw)
        bN = min(
            N,
            max(2, (_UB_STATS_BYTES - _SMALL) // (bS * pe + _ROW_OVERHEAD)),
        )
    bS = min(bS, S)
    # apply tile (extra in-flight staging buffer); bN follows the same rule.
    if N == 1:
        bS_apply = max(vw, (((_UB_SAFE_BYTES_SPLIT - _SMALL) // pe_apply) // vw) * vw)
        bN_apply = 1
    else:
        bS_apply = max(vw, (((_UB_SAFE_BYTES_SPLIT - _SMALL) // 2) // pe_apply // vw) * vw)
        bS_apply = min(bS_apply, S)
        bN_apply = min(
            N,
            max(2, (_UB_SAFE_BYTES_SPLIT - _SMALL) // (bS_apply * pe_apply + _ROW_OVERHEAD)),
        )
    bS_apply = min(bS_apply, S)
    n_tile_N = -(-N // bN)
    n_tile_S_stats = -(-max_seg // bS)
    n_tile_S_apply = -(-S // bS_apply)
    n_tile_N_apply = -(-N // bN_apply)
    # Tiling sweep hooks (2026-11-03): env overrides for bN/bS only; the UB
    # guard below still rejects anything that does not fit.
    _bs = int(os.environ.get("BN_BS", "0") or "0")
    _bn = int(os.environ.get("BN_BN", "0") or "0")
    if _bs > 0:
        bS = min(_bs, S)
        n_tile_S_stats = -(-max_seg // bS)
    if _bn > 0:
        bN = min(_bn, N)
        n_tile_N = -(-N // bN)
    return {
        "pe": pe,
        "pe_apply": pe_apply,
        "dist": dist,
        "rem": rem,
        "max_seg": max_seg,
        "bN": bN,
        "bS": bS,
        "bN_apply": bN_apply,
        "bS_apply": bS_apply,
        "n_tile_N": n_tile_N,
        "n_tile_N_apply": n_tile_N_apply,
        "n_tile_S_stats": n_tile_S_stats,
        "n_tile_S_apply": n_tile_S_apply,
    }


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _batch_norm_bwd_kernel(N, C, S, dtype="float16"):
    """Build the BatchNorm backward general kernel factory for an npuir target.

    Mirrors the source signature ``(N, C, S, dtype)``.  Returns a JIT callable
    taking the tile knobs ``(bN, bS)`` (the CUDA ``block_l``/``threads`` knobs
    are dropped, DESIGN.md section 0.2 note).  ``main`` keeps the source 8-arg
    order with ``out_idx=[-1]`` and returns the ``grad_x`` tensor.
    """
    if N < 1 or C < 1 or S < 1:
        raise ValueError(f"N, C, S must be positive, got {N}, {C}, {S}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )

    L = N * S
    L_inv = 1.0 / float(L)                                  # O1: compile-time constant

    # R6: fp32 transit for both half-precision dtypes (bf16 has no Vector
    # arithmetic; fp16 transit matches the golden's fp32 opmath path).
    use_transit = dtype in ("float16", "bfloat16")

    bpe, _held, bN_default, bS_default = _plan_tiling(N, S, use_transit)

    # OP1 (PL-1.13 subid-split): the Vector program is DUPLICATED across the two
    # AIV sub-blocks unless split by `subid`.  Launch `aicore_num` blocks and
    # assign each AIV its own channel slice.
    aicore_num = NPUUtils.get().get_aicore_num()   # 20 (Ascend910B3)
    num_kernels = min(C, aicore_num)                        # persistent core split (R1)
    total_slots = num_kernels * 2                            # 2 AIV per AI core
    num_local_tasks = (C + total_slots - 1) // total_slots   # static bound

    # 1-element plane (N == 1 and S == 1, L == 1): T.vbrc((1,1)->(1,1)) empty
    # broadcast / same-shape v-op is not vector-only -> MLIR verify fail
    # (sibling wide kernel Stage 3 evidence).  The kernel factory rejects it;
    # the op/wrapper routes this shape (DESIGN.md sections 9.1 / 9.3).
    scalar_plane = (N == 1 and S == 1)

    @tilelang.jit(out_idx=[-1], target="npuir")
    def _func(bN=bN_default, bS=bS_default):
        # Host-side UB budget / degenerate guard (survives `python -O`).
        if bN < 1 or bS < 1:
            raise ValueError(f"bN, bS must be positive, got bN={bN}, bS={bS}")
        if bN * bS * bpe > _UB_SAFE_BYTES:
            raise ValueError(
                f"bN={bN}, bS={bS} exceeds the UB budget for N={N}, dtype={dtype} "
                f"({bN * bS * bpe} B > {_UB_SAFE_BYTES} B at {bpe} B/elem "
                f"multi-buffered)"
            )
        if scalar_plane:
            raise ValueError(
                "1-element plane (N==1 and S==1): T.vbrc / same-shape v-op "
                "unsupported; op/wrapper must route this shape"
            )

        # Compile-time loop/structure constants (fold to IntImm at trace).
        n_tile_N = (N + bN - 1) // bN
        n_tile_S = (S + bS - 1) // bS
        n_gt1 = N > 1                        # dims=[0] reduce is legal
        s_gt1 = S > 1                        # dims=[1] reduce is legal
        single_tile = (n_tile_N == 1 and n_tile_S == 1)   # held: reuse pass-1 (O3)

        @T.prim_func
        def main(
            grad_out: T.Tensor((N, C, S), dtype),
            x: T.Tensor((N, C, S), dtype),
            weight: T.Tensor((C,), "float32"),
            mean: T.Tensor((C,), "float32"),
            rstd: T.Tensor((C,), "float32"),
            grad_weight: T.Tensor((C,), "float32"),           # host-preallocated GM
            grad_bias: T.Tensor((C,), "float32"),             # host-preallocated GM
            grad_x: T.Tensor((N, C, S), dtype),               # out_idx=-1
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, subid):
                with T.Scope("Vector"):
                    # UB buffers (DESIGN.md section 4.3), allocated once per
                    # core and reused across serial tasks:
                    #   stage : grad_out -> x -> grad_x (round-back transit)
                    #   g32   : grad_out plane in fp32 (held pass-2 reuse)
                    #   xh32  : x32 -> x_hat (held pass-2 reuse)
                    #   prod  : g*x_hat / scale*g / grad_x scratch
                    #   brc   : T.vbrc scalar-broadcast scratch
                    stage = T.alloc_ub((bN, bS), dtype)
                    g32 = T.alloc_ub((bN, bS), "float32")
                    xh32 = T.alloc_ub((bN, bS), "float32")
                    prod = T.alloc_ub((bN, bS), "float32")
                    brc = T.alloc_ub((bN, bS), "float32")
                    red = T.alloc_ub((bN, 1), "float32")
                    red2 = T.alloc_ub((bN, 1), "float32")
                    tred = T.alloc_ub((bN, 1), "float32")
                    tred2 = T.alloc_ub((bN, 1), "float32")
                    gb = T.alloc_ub((1, 1), "float32")
                    gw = T.alloc_ub((1, 1), "float32")
                    w1 = T.alloc_ub((1, 1), "float32")
                    m1 = T.alloc_ub((1, 1), "float32")
                    r1 = T.alloc_ub((1, 1), "float32")
                    scale = T.alloc_ub((1, 1), "float32")
                    per = T.alloc_ub((1, 1), "float32")
                    shift = T.alloc_ub((1, 1), "float32")
                    xc = T.alloc_ub((1, 1), "float32")

                    for i in T.serial(num_local_tasks):
                        c = i * total_slots + cid * 2 + subid
                        if c < C:
                            # ---- per-channel scalar parameters ----
                            T.copy(weight[c:c + 1], w1)
                            T.copy(mean[c:c + 1], m1)
                            T.copy(rstd[c:c + 1], r1)

                            # ---- pass 1: per (N,S) sub-block accumulate gb/gw ----
                            T.clear(gb)                       # accumulators across N-blocks
                            T.clear(gw)
                            for nn in T.serial(n_tile_N):
                                n0 = nn * bN
                                rN = T.min(bN, N - n0)
                                # Row-sum accumulators must be reset per N-block
                                # (v1 blocking item 1: otherwise previous blocks
                                # are double-counted).
                                T.clear(red)
                                T.clear(red2)
                                for tt in T.serial(n_tile_S):
                                    s0 = tt * bS
                                    rS = T.min(bS, S - s0)
                                    # plane gather (dst zero-origin dual slice)
                                    T.copy(
                                        grad_out[n0:n0 + rN, c, s0:s0 + rS],
                                        stage[0:rN, 0:rS],
                                    )
                                    if use_transit:
                                        T.vcast(stage, g32, round_mode="rint")
                                    else:
                                        T.copy(stage[0:rN, 0:rS], g32[0:rN, 0:rS])
                                    T.copy(
                                        x[n0:n0 + rN, c, s0:s0 + rS],
                                        stage[0:rN, 0:rS],
                                    )
                                    if use_transit:
                                        T.vcast(stage, xh32, round_mode="rint")
                                    else:
                                        T.copy(stage[0:rN, 0:rS], xh32[0:rN, 0:rS])
                                    # x_hat = (x - mean) * rstd
                                    T.vbrc(m1, brc)
                                    T.vsub(xh32, brc, xh32)
                                    T.vbrc(r1, brc)
                                    T.vmul(xh32, brc, xh32)
                                    # g * x_hat
                                    T.vmul(g32, xh32, prod)
                                    # Two-accumulator two-step reduce.  size=[rN,
                                    # rS] truncates the row sum to the valid
                                    # columns (tail-tile garbage is not summed);
                                    # clear=False accumulates across S-tiles.
                                    if s_gt1:
                                        T.reduce(
                                            g32, red, dims=[1], reduce_mode="sum",
                                            clear=False, size=[rN, rS],
                                        )
                                        T.reduce(
                                            prod, red2, dims=[1], reduce_mode="sum",
                                            clear=False, size=[rN, rS],
                                        )
                                    else:
                                        # S == 1: the (bN, 1) plane is already the
                                        # S-sum; skip the degenerate dims=[1] reduce.
                                        T.copy(g32[0:rN, 0:1], red[0:rN, 0:1])
                                        T.copy(prod[0:rN, 0:1], red2[0:rN, 0:1])
                                if n_gt1:
                                    # size=[rN, 1] truncates the tail N-block.
                                    T.reduce(
                                        red, gb, dims=[0], reduce_mode="sum",
                                        clear=False, size=[rN, 1],
                                    )
                                    T.reduce(
                                        red2, gw, dims=[0], reduce_mode="sum",
                                        clear=False, size=[rN, 1],
                                    )
                                else:
                                    # N == 1: (1,1) -> (1,1) is not a valid reduce.
                                    T.copy(red[0:1, 0:1], gb)
                                    T.copy(red2[0:1, 0:1], gw)
                            T.copy(gb, grad_bias[c:c + 1])
                            T.copy(gw, grad_weight[c:c + 1])

                            # ---- epilogue coefficients (O1/O2) ----
                            T.vmul(w1, r1, scale)          # scale = w * rstd
                            T.vmul(scale, L_inv, per)      # per = scale / L
                            T.vmul(per, gb, shift)         # shift = per * gb
                            T.vmul(per, gw, xc)            # xc = per * gw

                            # ---- pass 2: grad_x = scale*g - shift - xc*x_hat ----
                            if single_tile:
                                # Reuse pass-1 g32 / xh32 (single scan, O3).
                                T.vbrc(scale, brc)
                                T.vmul(g32, brc, prod)
                                T.vbrc(shift, brc)
                                T.vsub(prod, brc, prod)
                                T.vbrc(xc, brc)
                                T.vmul(xh32, brc, brc)
                                T.vsub(prod, brc, prod)
                                if use_transit:
                                    T.vcast(prod, stage, round_mode="rint")
                                else:
                                    T.copy(prod, stage)
                                T.copy(stage[0:N, 0:S], grad_x[0:N, c, 0:S])
                            else:
                                for nn in T.serial(n_tile_N):
                                    n0 = nn * bN
                                    rN = T.min(bN, N - n0)
                                    for tt in T.serial(n_tile_S):
                                        s0 = tt * bS
                                        rS = T.min(bS, S - s0)
                                        T.copy(
                                            grad_out[n0:n0 + rN, c, s0:s0 + rS],
                                            stage[0:rN, 0:rS],
                                        )
                                        if use_transit:
                                            T.vcast(stage, g32, round_mode="rint")
                                        else:
                                            T.copy(stage[0:rN, 0:rS], g32[0:rN, 0:rS])
                                        T.copy(
                                            x[n0:n0 + rN, c, s0:s0 + rS],
                                            stage[0:rN, 0:rS],
                                        )
                                        if use_transit:
                                            T.vcast(stage, xh32, round_mode="rint")
                                        else:
                                            T.copy(stage[0:rN, 0:rS], xh32[0:rN, 0:rS])
                                        T.vbrc(m1, brc)
                                        T.vsub(xh32, brc, xh32)
                                        T.vbrc(r1, brc)
                                        T.vmul(xh32, brc, xh32)
                                        T.vbrc(scale, brc)
                                        T.vmul(g32, brc, prod)
                                        T.vbrc(shift, brc)
                                        T.vsub(prod, brc, prod)
                                        T.vbrc(xc, brc)
                                        T.vmul(xh32, brc, brc)
                                        T.vsub(prod, brc, prod)
                                        if use_transit:
                                            T.vcast(prod, stage, round_mode="rint")
                                        else:
                                            T.copy(prod[0:rN, 0:rS], stage[0:rN, 0:rS])
                                        T.copy(
                                            stage[0:rN, 0:rS],
                                            grad_x[n0:n0 + rN, c, s0:s0 + rS],
                                        )

        return main

    return _func


def _ck_choose_ct(C):
    """Largest divisor of C nearest C/32 (32 AIV groups on Ascend910B3)."""
    target = max(1, C // 32)
    best = 1
    for ct in range(1, C + 1):
        if C % ct == 0 and abs(ct - target) < abs(best - target):
            best = ct
    return best


def _batch_norm_bwd_wide_ckernel(N, C, S, dtype="float16"):
    """P1 channel-coalesced wide kernel.

    For a fixed batch index ``n`` the layout ``(N, C, S)`` makes
    ``[n, c0:c0+Ct, 0:S]`` a single contiguous ``Ct*S`` block.  Processing
    ``Ct`` adjacent channels as the UB rows therefore turns every GM
    transfer into one large burst (stage3: 1.5KB -> 12.5KB) and replaces the
    per-channel scalar + ``vbrc`` epilogue with ``(Ct,1)`` row-broadcast
    vectors.
    """
    if N < 1 or C < 1 or S < 1:
        raise ValueError(f"N, C, S must be positive, got {N}, {C}, {S}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")

    L_inv = 1.0 / float(N * S)
    use_transit = dtype in ("float16", "bfloat16")
    native_apply = dtype == "float16"
    pe = _UB_PE_TRANSIT if use_transit else _UB_PE_FP32

    Ct = _ck_choose_ct(C)

    rows_elems = max(1, (_UB_SAFE_BYTES - 512) // pe)
    bS_cap = max(1, rows_elems // Ct)
    if bS_cap >= S:
        bS = S
    else:
        bS = (bS_cap // 64) * 64
        if bS < 1:
            bS = 1
        bS = min(bS, S)

    if Ct * bS * pe > _UB_SAFE_BYTES:
        raise ValueError(
            f"channel-coalesced UB footprint Ct={Ct} bS={bS} pe={pe} "
            f"exceeds {_UB_SAFE_BYTES} B"
        )

    groups = C // Ct
    aicore_num = NPUUtils.get().get_aicore_num()
    num_kernels = min(groups, aicore_num * 2)
    num_local_tasks = (groups + num_kernels - 1) // num_kernels
    n_tiles = (S + bS - 1) // bS
    s_gt1 = S > 1

    @tilelang.jit(out_idx=[7, 5, 6], target="npuir")
    def _func(bS=bS, Ct=Ct):
        @T.prim_func
        def main(
            grad_out: T.Tensor((N, C, S), dtype),
            x: T.Tensor((N, C, S), dtype),
            weight: T.Tensor((C,), "float32"),
            mean: T.Tensor((C,), "float32"),
            rstd: T.Tensor((C,), "float32"),
            grad_weight: T.Tensor((C,), "float32"),
            grad_bias: T.Tensor((C,), "float32"),
            grad_x: T.Tensor((N, C, S), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                g_stage = T.alloc_ub((Ct, bS), dtype)
                x_stage = T.alloc_ub((Ct, bS), dtype)
                g32 = T.alloc_ub((Ct, bS), "float32")
                xh32 = T.alloc_ub((Ct, bS), "float32")
                prod = T.alloc_ub((Ct, bS), "float32")
                pd = T.alloc_ub((Ct, bS), dtype)

                red = T.alloc_ub((Ct, 1), "float32")
                red2 = T.alloc_ub((Ct, 1), "float32")
                wv = T.alloc_ub((Ct, 1), "float32")
                mv = T.alloc_ub((Ct, 1), "float32")
                rv = T.alloc_ub((Ct, 1), "float32")
                gbv = T.alloc_ub((Ct, 1), "float32")
                gsv = T.alloc_ub((Ct, 1), "float32")
                gwv = T.alloc_ub((Ct, 1), "float32")
                coefv = T.alloc_ub((Ct, 1), "float32")
                perv = T.alloc_ub((Ct, 1), "float32")
                shiftv = T.alloc_ub((Ct, 1), "float32")
                xcv = T.alloc_ub((Ct, 1), "float32")
                bbv = T.alloc_ub((Ct, 1), "float32")
                ccv = T.alloc_ub((Ct, 1), "float32")
                t1v = T.alloc_ub((Ct, 1), "float32")
                t2v = T.alloc_ub((Ct, 1), "float32")
                coef16 = T.alloc_ub((Ct, 1), dtype)
                cc16 = T.alloc_ub((Ct, 1), dtype)
                bb16 = T.alloc_ub((Ct, 1), dtype)

                for i in T.serial(num_local_tasks):
                    g = i * num_kernels + cid
                    if g < groups:
                        c0 = g * Ct
                        # (Ct,1) per-channel vectors; Ct scalar copies are
                        # amortised over the whole channel group.
                        for j in T.serial(Ct):
                            T.copy(weight[c0 + j:c0 + j + 1], wv[j:j + 1, 0:1])
                            T.copy(mean[c0 + j:c0 + j + 1], mv[j:j + 1, 0:1])
                            T.copy(rstd[c0 + j:c0 + j + 1], rv[j:j + 1, 0:1])

                        T.clear(red)
                        T.clear(red2)
                        for n in T.serial(N):
                            for t in T.serial(n_tiles):
                                s0 = t * bS
                                real = T.min(bS, S - s0)
                                T.copy(
                                    grad_out[n, c0:c0 + Ct, s0:s0 + real],
                                    g_stage[0:Ct, 0:real],
                                )
                                T.copy(
                                    x[n, c0:c0 + Ct, s0:s0 + real],
                                    x_stage[0:Ct, 0:real],
                                )
                                if use_transit:
                                    T.vcast(g_stage, g32, round_mode="rint")
                                    T.vcast(x_stage, xh32, round_mode="rint")
                                else:
                                    T.copy(g_stage, g32)
                                    T.copy(x_stage, xh32)
                                T.vmul(g32, xh32, prod)
                                if s_gt1:
                                    T.reduce(
                                        g32, red, dims=[1], reduce_mode="sum",
                                        clear=False, size=[Ct, real],
                                    )
                                    T.reduce(
                                        prod, red2, dims=[1], reduce_mode="sum",
                                        clear=False, size=[Ct, real],
                                    )
                                else:
                                    T.vadd(red, g32, red)
                                    T.vadd(red2, prod, red2)

                        # gw = rstd*sum(g*x) - (mean*rstd)*sum(g)
                        T.copy(red, gbv)
                        T.copy(red2, gsv)
                        T.vmul(mv, rv, t1v)
                        T.vmul(t1v, -1.0, t1v)
                        T.vmul(rv, gsv, gwv)
                        T.vmul(t1v, gbv, t2v)
                        T.vadd(gwv, t2v, gwv)

                        T.vmul(wv, rv, coefv)
                        T.vmul(coefv, L_inv, perv)
                        T.vmul(perv, gbv, shiftv)
                        T.vmul(perv, gwv, xcv)
                        T.vmul(xcv, rv, bbv)
                        T.vmul(xcv, mv, t1v)
                        T.vmul(t1v, rv, t1v)
                        T.vmul(shiftv, -1.0, t2v)
                        T.vadd(t1v, t2v, ccv)

                        if native_apply:
                            T.vcast(coefv, coef16, round_mode="rint")
                            T.vcast(ccv, cc16, round_mode="rint")
                            T.vcast(bbv, bb16, round_mode="rint")

                        T.copy(gwv, grad_weight[c0:c0 + Ct])
                        T.copy(gbv, grad_bias[c0:c0 + Ct])

                        for n in T.serial(N):
                            for t in T.serial(n_tiles):
                                s0 = t * bS
                                real = T.min(bS, S - s0)
                                T.copy(
                                    grad_out[n, c0:c0 + Ct, s0:s0 + real],
                                    g_stage[0:Ct, 0:real],
                                )
                                T.copy(
                                    x[n, c0:c0 + Ct, s0:s0 + real],
                                    x_stage[0:Ct, 0:real],
                                )
                                if native_apply:
                                    T.vmul(g_stage, coef16, pd)
                                    T.vadd(pd, cc16, pd)
                                    T.vmul(x_stage, bb16, x_stage)
                                    T.vsub(pd, x_stage, pd)
                                    T.copy(
                                        pd[0:Ct, 0:real],
                                        grad_x[n, c0:c0 + Ct, s0:s0 + real],
                                    )
                                else:
                                    if use_transit:
                                        T.vcast(g_stage, g32, round_mode="rint")
                                        T.vcast(x_stage, xh32, round_mode="rint")
                                    else:
                                        T.copy(g_stage, g32)
                                        T.copy(x_stage, xh32)
                                    T.vmul(g32, coefv, g32)
                                    T.vadd(g32, ccv, g32)
                                    T.vmul(xh32, bbv, xh32)
                                    T.vsub(g32, xh32, g32)
                                    if use_transit:
                                        T.vcast(g32, g_stage, round_mode="rint")
                                    else:
                                        T.copy(g32, g_stage)
                                    T.copy(
                                        g_stage[0:Ct, 0:real],
                                        grad_x[n, c0:c0 + Ct, s0:s0 + real],
                                    )

        return main

    return _func

def _batch_norm_bwd_wide_kernel(N, C, S, dtype="float16"):
    """Build the BatchNorm backward wide kernel factory.

    Mirrors the source signature ``(N, C, S, dtype)``.  Returns a JIT
    callable taking a single ``bS`` (the CUDA ``threads``/``num_per_thread``
    knobs are dropped, DESIGN.md section 3.3 note).  ``main`` returns
    ``(grad_x, grad_weight, grad_bias)`` (``out_idx=[7, 5, 6]``).
    """
    if N < 1 or C < 1 or S < 1:
        raise ValueError(f"N, C, S must be positive, got {N}, {C}, {S}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )

    # R35 (P1 channel coalescing): C>=192 with short spatial (resnet-stage3
    # class) benefits from contiguous (Ct,S) bursts -- stage3 MTE3 6.1 ->
    # 52.0 GB/s, -24.5%.  C=64/128 keep the one-scan wide body (coalescing
    # doubles the read pass and regresses the small shapes); large-S already
    # has long bursts, so it also stays on the one-scan body.
    if C >= 192 and S <= 2048:
        return _batch_norm_bwd_wide_ckernel(N, C, S, dtype)

    L = N * S
    L_inv = 1.0 / float(L)                                  # O1: compile-time constant

    # R4: fp32 transit for both half-precision dtypes (bf16 has no Vector
    # arithmetic; fp16 transit matches the golden's fp32 opmath path).
    use_transit = dtype in ("float16", "bfloat16")
    # R19: fp16 apply in native dtype (same formula/proof as R12-b split apply).
    native_fp16_apply = dtype == "float16"
    pe = _UB_PE_TRANSIT if use_transit else _UB_PE_FP32   # multi-buffered B/elem

    bS_default = _plan_bS(N, S, pe, use_transit)
    # R21 pure-AIV: 2088-wide planes overflow UB after removing the AIC/subid
    # structure; 2048 keeps the same tile count for S=4096 and is aligned.
    if use_transit:
        bS_cap_pure = 2048
    else:
        # fp32 has 5 resident planes; only very large S needs a tighter cap.
        bS_cap_pure = 768 if S > 2048 else S
    bS_default = min(bS_default, bS_cap_pure)

    # OP1 (PL-1.13 subid-split): the Vector program of a `T.Kernel` block is
    # DUPLICATED across its two AIV sub-blocks (Mix Block Dim = 2 x Block Dim)
    # when the channel loop is not split by `subid`.  We launch
    # `num_kernels = min(C, aicore_num)` blocks and assign each AIV (`subid`)
    # its own channel slice so the 40 AIV process distinct channels in one wave
    # instead of 20 blocks x 2 redundant waves.
    aicore_num = NPUUtils.get().get_aicore_num()   # 20 (Ascend910B3)

    @tilelang.jit(out_idx=[7, 5, 6], target="npuir")
    def _func(bS=bS_default):
        # Host-side UB budget guard (DESIGN.md sections 3.3 and 9.2).
        # ValueError (not a bare assert) so the guard survives `python -O`.
        if bS < 1:
            raise ValueError(f"bS must be positive, got bS={bS}")
        if N * bS * pe > _UB_SAFE_BYTES:
            raise ValueError(
                f"bS={bS} exceeds the UB budget for N={N}, dtype={dtype} "
                f"({N * bS * pe} B > {_UB_SAFE_BYTES} B at {pe} B/elem "
                f"multi-buffered)"
            )

        # Compile-time loop/structure constants (fold to IntImm at trace).
        n_tiles = (S + bS - 1) // bS
        single_tile = (n_tiles == 1)          # held single-scan fast path (O3)
        s_gt1 = S > 1                         # dims=[1] reduce is legal
        n_gt1 = N > 1                         # dims=[0] reduce is legal
        # Degenerate (1, 1) plane: the (1,1) scalar buffers cannot be used as
        # vbrc sources into a same-shape (1,1) destination (empty broadcast dim
        # -> MLIR verify fail); use the same-shape ops directly instead.
        scalar_plane = (N == 1 and bS == 1)
        # P2: C=64/128 shapes divide 32 AIV exactly; 16 AIC avoids the
        # 40-slot partial wave and frees 4 AIC.  C>=192 keeps 20 AIC.
        # R14 (2026-11-06): C=128 also uses 16 AIC / 32 AIV (128/32=4 exact
        # tasks; standalone A/B 14.47 -> 13.77us, -4.8%). C=256 keeps 20 AIC.
        _core_cap = 16 if C in (64, 128) else aicore_num
        # R21: one block per AIV (no subid duplication, no AIC carrier).
        num_kernels = min(C, _core_cap * 2)
        total_slots = num_kernels
        num_local_tasks = (C + total_slots - 1) // total_slots

        @T.prim_func
        def main(
            grad_out: T.Tensor((N, C, S), dtype),
            x: T.Tensor((N, C, S), dtype),
            weight: T.Tensor((C,), "float32"),
            mean: T.Tensor((C,), "float32"),
            rstd: T.Tensor((C,), "float32"),
            grad_weight: T.Tensor((C,), "float32"),           # out_idx=5
            grad_bias: T.Tensor((C,), "float32"),             # out_idx=6
            grad_x: T.Tensor((N, C, S), dtype),               # out_idx=7
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # UB buffers (DESIGN.md section 4.3), allocated once per
                # core and reused across serial tasks:
                #   stage : grad_out -> x -> grad_x (round-back transit)
                #   g32   : grad_out plane in fp32
                #   xh32  : x32 -> x_hat
                #   prod  : g*x_hat / scale*g / grad_x scratch
                #   brc   : T.vbrc scalar-broadcast scratch
                # R3-W2 (loads-first decoupling): one staging buffer per
                # input so the two plane gathers are both in flight before
                # the vector chain consumes them (a single reused staging
                # buffer serialises MTE2 -> VEC per input).  `brc` is gone:
                # `prod` doubles as the broadcast scratch, which keeps the
                # resident footprint flat so the held single-scan path
                # (bS == S) survives.
                g_stage = T.alloc_ub((N, bS), dtype)
                x_stage = T.alloc_ub((N, bS), dtype)
                g32 = T.alloc_ub((N, bS), "float32")
                xh32 = T.alloc_ub((N, bS), "float32")
                prod = T.alloc_ub((N, bS), "float32")
                # R19: fp16-native apply plane + scalar coefficients
                pd = T.alloc_ub((N, bS), dtype)
                coef16 = T.alloc_ub((1, 1), dtype)
                cc16 = T.alloc_ub((1, 1), dtype)
                bb16 = T.alloc_ub((1, 1), dtype)
                red = T.alloc_ub((N, 1), "float32")
                red2 = T.alloc_ub((N, 1), "float32")
                gb = T.alloc_ub((1, 1), "float32")
                gw = T.alloc_ub((1, 1), "float32")
                w1 = T.alloc_ub((1, 1), "float32")
                m1 = T.alloc_ub((1, 1), "float32")
                r1 = T.alloc_ub((1, 1), "float32")
                coef = T.alloc_ub((1, 1), "float32")
                per = T.alloc_ub((1, 1), "float32")
                shift = T.alloc_ub((1, 1), "float32")
                xc = T.alloc_ub((1, 1), "float32")
                # R3-W1: grad_weight is folded algebraically --
                #   gw = sum(g*x_hat) = rstd*sum(g*x) - (mean*rstd)*sum(g)
                # so the in-loop x_hat chain disappears; the (mean, rstd)
                # factors move into the epilogue coefficients below.
                b1 = T.alloc_ub((1, 1), "float32")     # -mean*rstd
                gs = T.alloc_ub((1, 1), "float32")     # sum(g*x)
                t1 = T.alloc_ub((1, 1), "float32")     # scratch
                cc = T.alloc_ub((1, 1), "float32")     # constant term of grad_x
                t2 = T.alloc_ub((1, 1), "float32")     # scratch
                bb = T.alloc_ub((1, 1), "float32")     # x coefficient of grad_x

                for i in T.serial(num_local_tasks):
                    c = i * total_slots + cid
                    if c < C:
                        # ---- per-channel scalar parameters ----
                        T.copy(weight[c:c + 1], w1)
                        T.copy(mean[c:c + 1], m1)
                        T.copy(rstd[c:c + 1], r1)

                        # ---- pass 1: per-S-tile accumulate gb / gw ----
                        T.clear(red)
                        T.clear(red2)
                        for t in T.serial(n_tiles):
                            s0 = t * bS
                            real = T.min(bS, S - s0)
                            # plane gather (dst zero-origin dual slice)
                            T.copy(
                                grad_out[0:N, c, s0:s0 + real],
                                g_stage[0:N, 0:real],
                            )
                            T.copy(
                                x[0:N, c, s0:s0 + real],
                                x_stage[0:N, 0:real],
                            )
                            if use_transit:
                                T.vcast(g_stage, g32, round_mode="rint")
                            else:
                                T.copy(g_stage[0:N, 0:real], g32[0:N, 0:real])
                            if use_transit:
                                T.vcast(x_stage, xh32, round_mode="rint")
                            else:
                                T.copy(x_stage[0:N, 0:real], xh32[0:N, 0:real])
                            # R3-W1: x_hat is gone from the elementwise chain.
                            # Accumulate sum(g) -> red and sum(g*x) -> red2;
                            # the (mean, rstd) factors are folded into the
                            # epilogue coefficients (gw / grad_x).
                            T.vmul(g32, xh32, prod)
                            # Two-accumulator, two-step reduce.  size=[N,
                            # real] truncates the row sum to the valid
                            # columns (tail-tile garbage columns are not
                            # summed); clear=False accumulates across tiles
                            # (DESIGN.md section 0.0 blocking item 2).
                            if s_gt1:
                                T.reduce(
                                    g32, red, dims=[1], reduce_mode="sum",
                                    clear=False, size=[N, real],
                                )
                                T.reduce(
                                    prod, red2, dims=[1], reduce_mode="sum",
                                    clear=False, size=[N, real],
                                )
                            else:
                                # S == 1: the (N, 1) plane is already the
                                # S-sum; skip the degenerate dims=[1] reduce.
                                T.copy(g32, red)
                                T.copy(prod, red2)
                        if n_gt1:
                            T.reduce(
                                red, gb, dims=[0], reduce_mode="sum",
                                clear=True,
                            )
                            T.reduce(
                                red2, gs, dims=[0], reduce_mode="sum",
                                clear=True,
                            )
                        else:
                            # N == 1: (1, 1) -> (1, 1) is not a valid
                            # reduce; the partial is already the total.
                            T.copy(red, gb)
                            T.copy(red2, gs)
                        # R3-W1: gw = rstd*sum(g*x) - (mean*rstd)*sum(g)
                        T.vmul(m1, r1, b1)              # mean * rstd
                        T.vmul(b1, -1.0, b1)            # -(mean * rstd)
                        T.vmul(r1, gs, gw)              # rstd * sum(g*x)
                        T.vmul(b1, gb, t1)              # -(mean*rstd)*sum(g)
                        T.vadd(gw, t1, gw)
                        T.copy(gb, grad_bias[c:c + 1])
                        T.copy(gw, grad_weight[c:c + 1])

                        # ---- epilogue coefficients (O1/O2) ----
                        T.vmul(w1, r1, coef)            # scale = w * rstd
                        T.vmul(coef, L_inv, per)        # per = scale / L
                        T.vmul(per, gb, shift)          # shift = per * gb
                        T.vmul(per, gw, xc)             # xc = per * gw
                        # R3-W1: grad_x = A*g + B*x + C with
                        #   A = coef, B = -xc*rstd,
                        #   C = xc*mean*rstd - shift.
                        # Pass 2 subtracts bb*x, so bb = xc*rstd.
                        T.vmul(xc, r1, bb)              # xc * rstd
                        T.vmul(xc, m1, t1)              # xc * mean
                        T.vmul(t1, r1, t1)              # xc * mean * rstd
                        T.vmul(shift, -1.0, t2)         # -shift
                        T.vadd(t1, t2, cc)              # C = xc*mean*rstd - shift

                        # ---- pass 2: grad_x = scale*g - shift - xc*x_hat ----
                        if single_tile:
                            # Reuse pass-1 g32 / xh32 (single scan, O3).
                            if native_fp16_apply:
                                # R19: grad_x = coef*g + cc - bb*x in fp16.
                                T.vcast(coef, coef16, round_mode="rint")
                                T.vcast(cc, cc16, round_mode="rint")
                                T.vcast(bb, bb16, round_mode="rint")
                                T.vmul(g_stage, coef16, pd)
                                T.vadd(pd, cc16, pd)
                                T.vmul(x_stage, bb16, x_stage)
                                T.vsub(pd, x_stage, pd)
                                T.copy(
                                    pd[0:N, 0:S],
                                    grad_x[0:N, c, 0:S],
                                )
                            elif scalar_plane:
                                T.vmul(g32, coef, prod)
                                T.vadd(prod, cc, prod)
                                T.vmul(xh32, bb, xh32)
                                T.vsub(prod, xh32, prod)
                            else:
                                T.vbrc(coef, prod)
                                T.vmul(g32, prod, g32)
                                T.vbrc(cc, prod)
                                T.vadd(g32, prod, g32)
                                T.vbrc(bb, prod)
                                T.vmul(xh32, prod, xh32)
                                T.vsub(g32, xh32, g32)
                            if native_fp16_apply:
                                pass
                            elif use_transit:
                                T.vcast(g32, g_stage, round_mode="rint")
                            else:
                                T.copy(g32, g_stage)
                            if not native_fp16_apply:
                                T.copy(
                                    g_stage[0:N, 0:S],
                                    grad_x[0:N, c, 0:S],
                                )
                        else:
                            for t in T.serial(n_tiles):
                                s0 = t * bS
                                real = T.min(bS, S - s0)
                                T.copy(
                                    grad_out[0:N, c, s0:s0 + real],
                                    g_stage[0:N, 0:real],
                                )
                                T.copy(
                                    x[0:N, c, s0:s0 + real],
                                    x_stage[0:N, 0:real],
                                )
                                if use_transit:
                                    T.vcast(g_stage, g32, round_mode="rint")
                                else:
                                    T.copy(g_stage[0:N, 0:real], g32[0:N, 0:real])
                                if use_transit:
                                    T.vcast(x_stage, xh32, round_mode="rint")
                                else:
                                    T.copy(x_stage[0:N, 0:real], xh32[0:N, 0:real])
                                T.vbrc(coef, prod)
                                T.vmul(g32, prod, g32)
                                T.vbrc(cc, prod)
                                T.vadd(g32, prod, g32)
                                T.vbrc(bb, prod)
                                T.vmul(xh32, prod, xh32)
                                T.vsub(g32, xh32, g32)
                                if use_transit:
                                    T.vcast(g32, g_stage, round_mode="rint")
                                else:
                                    T.copy(g32[0:N, 0:real], g_stage[0:N, 0:real])
                                T.copy(
                                    g_stage[0:N, 0:real],
                                    grad_x[0:N, c, s0:s0 + real],
                                )

        return main

    return _func


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
    native_apply = dtype == "float16"

    splits = _plan_splits(N, C, S)
    plan = _plan_tiling_split(N, S, splits, use_transit)
    bN = plan["bN"]
    bS = plan["bS"]
    bN_apply = plan["bN_apply"]
    bS_apply = plan["bS_apply"]
    pe_apply = plan["pe_apply"]
    n_tile_N_apply = plan["n_tile_N_apply"]
    pe = plan["pe"]
    dist = plan["dist"]
    rem = plan["rem"]
    n_tile_N = plan["n_tile_N"]
    n_tile_S_stats = plan["n_tile_S_stats"]
    n_tile_S_apply = plan["n_tile_S_apply"]

    # R29 (memory hierarchy): fp16-native apply has only 3 dtype planes
    # (g/x/brc), so large-S can use one 16K-wide burst instead of 3.4K.
    # bN=1 makes the S copy contiguous; 895us -> 655us on large-spatial.
    if native_apply and S >= 8192:
        bS_apply = min(S, 45056)
        bN_apply = 1
        n_tile_S_apply = -(-S // bS_apply)
        n_tile_N_apply = N

    # Host-side UB budget guard (compile-time sanity; survives `python -O`).
    if bN * bS * pe + bN * _ROW_OVERHEAD + _SMALL > _UB_CAPACITY_BYTES:
        raise ValueError(
            f"planned UB footprint {bN}*{bS}*{pe}+{bN * _ROW_OVERHEAD} exceeds "
            f"capacity {_UB_CAPACITY_BYTES} B"
        )
    pe_apply_guard = 4 if native_apply else pe_apply
    if bN_apply * bS_apply * pe_apply_guard + bN_apply * _ROW_OVERHEAD + _SMALL > _UB_CAPACITY_BYTES:
        raise ValueError(
            f"planned apply UB footprint {bN_apply}*{bS_apply}*{pe_apply} exceeds "
            f"capacity {_UB_CAPACITY_BYTES} B"
        )

    s_gt1 = S > 1            # dims=[1] reduce is legal
    n_gt1 = N > 1            # dims=[0] reduce is legal (bN >= 2)
    split_gt1 = splits > 1   # finalize merge reduces a real axis
    # R36: exact-tile segments (large-spatial) accumulate g / g*x in `cat`
    # across S tiles and run one horizontal reduce per N-block.
    accum_stats = (rem == 0 and dist % bS == 0)

    # OP1 (PL-1.13 subid-split): the Vector program is DUPLICATED across the two
    # AIV sub-blocks unless split by `subid`.  Launch `aicore_num` blocks and
    # assign each AIV its own (c, sp) task / channel slice.
    aicore_num = NPUUtils.get().get_aicore_num()   # 20 (Ascend910B3)

    n_tasks = C * splits
    KS = min(n_tasks, aicore_num)
    nlS = -(-n_tasks // (KS * 2))         # stats AIV-serial task count
    KC = min(C, aicore_num)
    nlC = -(-C // (KC * 2))               # finalize/apply AIV-serial task count
    # R37: pure-AIV finalize grid (one block per AIV, no subid duplication).
    num_fin = min(C, aicore_num * 2)
    nl_fin = -(-C // num_fin)

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
            with T.Kernel(KS, is_npu=True) as (cid, subid):
                with T.Scope("Vector"):
                    # R3-S3 (loads-first decoupling, stats stage): one staging
                    # buffer per input, and `prod` doubles as the vbrc scratch
                    # instead of a dedicated `brc`, so the resident footprint
                    # stays flat and bS is unchanged.
                    g_stage = T.alloc_ub((bN, bS), dtype)
                    x_stage = T.alloc_ub((bN, bS), dtype)
                    g32 = T.alloc_ub((bN, bS), "float32")
                    xh32 = T.alloc_ub((bN, bS), "float32")
                    prod = T.alloc_ub((bN, bS), "float32")
                    # R20: pack g (rows 0..bN-1) and g*x (rows bN..2*bN-1)
                    # into one plane so a single T.reduce handles both.
                    cat = T.alloc_ub((2 * bN, bS), "float32")
                    redcat = T.alloc_ub((2 * bN, 1), "float32")
                    red = T.alloc_ub((bN, 1), "float32")
                    red2 = T.alloc_ub((bN, 1), "float32")
                    tred = T.alloc_ub((bN, 1), "float32")
                    tred2 = T.alloc_ub((bN, 1), "float32")
                    gb = T.alloc_ub((1, 1), "float32")        # segment partial (across N)
                    gw = T.alloc_ub((1, 1), "float32")
                    m1 = T.alloc_ub((1, 1), "float32")
                    r1 = T.alloc_ub((1, 1), "float32")

                    for i in T.serial(nlS):
                        t = i * (KS * 2) + cid * 2 + subid
                        if t < n_tasks:
                            c = t // splits
                            sp = t % splits
                            # Balanced segment [s0, s1): real = s1 - s0 >= 1.
                            s0 = sp * dist + T.min(sp, rem)
                            s1 = (sp + 1) * dist + T.min(sp + 1, rem)
                            T.clear(gb)
                            T.clear(gw)
                            for nn in T.serial(n_tile_N):
                                n0 = nn * bN
                                rN = bN if N % bN == 0 else T.min(bN, N - n0)  # static when divisible
                                if accum_stats:
                                    T.clear(cat)
                                    full_s = (s1 - s0) // bS
                                    for tt in T.Pipelined(full_s, num_stages=2):
                                        ss = s0 + tt * bS
                                        T.copy(
                                            grad_out[n0:n0 + rN, c, ss:ss + bS],
                                            g_stage[0:rN, 0:bS],
                                        )
                                        T.copy(
                                            x[n0:n0 + rN, c, ss:ss + bS],
                                            x_stage[0:rN, 0:bS],
                                        )
                                        if use_transit:
                                            T.vcast(g_stage, g32, round_mode="rint")
                                            T.vcast(x_stage, xh32, round_mode="rint")
                                        else:
                                            T.copy(g_stage[0:rN, 0:bS], g32[0:rN, 0:bS])
                                            T.copy(x_stage[0:rN, 0:bS], xh32[0:rN, 0:bS])
                                        T.vmul(g32, xh32, prod)
                                        T.vadd(cat[0:bN, 0:bS], g32,
                                               cat[0:bN, 0:bS])
                                        T.vadd(cat[bN:2 * bN, 0:bS], prod,
                                               cat[bN:2 * bN, 0:bS])
                                    if s_gt1:
                                        T.reduce(
                                            cat, redcat, dims=[1],
                                            reduce_mode="sum", clear=True,
                                            size=[2 * rN, bS],
                                        )
                                        T.copy(redcat[0:rN, 0:1], red[0:rN, 0:1])
                                        T.copy(redcat[bN:bN + rN, 0:1],
                                               red2[0:rN, 0:1])
                                    else:
                                        T.copy(cat[0:rN, 0:1], red[0:rN, 0:1])
                                        T.copy(cat[bN:bN + rN, 0:1], red2[0:rN, 0:1])
                                else:
                                    T.clear(redcat)
                                    # R8/P1: static full-tile S loop + one tail
                                    # tile, so the hot body has no T.min/guard and
                                    # reduce uses a static size.
                                    full_s = (s1 - s0) // bS
                                    for tt in T.Pipelined(full_s, num_stages=2):
                                        ss = s0 + tt * bS
                                        T.copy(
                                            grad_out[n0:n0 + rN, c, ss:ss + bS],
                                            g_stage[0:rN, 0:bS],
                                        )
                                        T.copy(
                                            x[n0:n0 + rN, c, ss:ss + bS],
                                            x_stage[0:rN, 0:bS],
                                        )
                                        if use_transit:
                                            T.vcast(g_stage, g32, round_mode="rint")
                                            T.vcast(x_stage, xh32, round_mode="rint")
                                        else:
                                            T.copy(g_stage[0:rN, 0:bS], g32[0:rN, 0:bS])
                                            T.copy(x_stage[0:rN, 0:bS], xh32[0:rN, 0:bS])
                                        T.vmul(g32, xh32, prod)
                                        T.copy(g32[0:rN, 0:bS], cat[0:rN, 0:bS])
                                        T.copy(prod[0:rN, 0:bS], cat[bN:bN + rN, 0:bS])
                                        if s_gt1:
                                            T.reduce(
                                                cat, redcat, dims=[1],
                                                reduce_mode="sum", clear=False,
                                                size=[2 * rN, bS],
                                            )
                                        else:
                                            T.copy(g32[0:rN, 0:1], red[0:rN, 0:1])
                                            T.copy(prod[0:rN, 0:1], red2[0:rN, 0:1])
                                    tail_s = (s1 - s0) - full_s * bS
                                    if tail_s > 0:
                                        ss = s0 + full_s * bS
                                        T.copy(
                                            grad_out[n0:n0 + rN, c, ss:ss + tail_s],
                                            g_stage[0:rN, 0:tail_s],
                                        )
                                        T.copy(
                                            x[n0:n0 + rN, c, ss:ss + tail_s],
                                            x_stage[0:rN, 0:tail_s],
                                        )
                                        if use_transit:
                                            T.vcast(g_stage, g32, round_mode="rint")
                                            T.vcast(x_stage, xh32, round_mode="rint")
                                        else:
                                            T.copy(g_stage[0:rN, 0:tail_s], g32[0:rN, 0:tail_s])
                                            T.copy(x_stage[0:rN, 0:tail_s], xh32[0:rN, 0:tail_s])
                                        T.vmul(g32, xh32, prod)
                                        T.copy(g32[0:rN, 0:tail_s], cat[0:rN, 0:tail_s])
                                        T.copy(prod[0:rN, 0:tail_s], cat[bN:bN + rN, 0:tail_s])
                                        if s_gt1:
                                            T.reduce(
                                                g32, tred, dims=[1],
                                                reduce_mode="sum", clear=True,
                                                size=[rN, tail_s],
                                            )
                                            T.reduce(
                                                prod, tred2, dims=[1],
                                                reduce_mode="sum", clear=True,
                                                size=[rN, tail_s],
                                            )
                                        else:
                                            T.copy(g32[0:rN, 0:1], red[0:rN, 0:1])
                                            T.copy(prod[0:rN, 0:1], red2[0:rN, 0:1])
                                    if s_gt1:
                                        T.copy(redcat[0:rN, 0:1], red[0:rN, 0:1])
                                        T.copy(redcat[bN:bN + rN, 0:1],
                                               red2[0:rN, 0:1])
                                        if tail_s > 0:
                                            T.vadd(red, tred, red)
                                            T.vadd(red2, tred2, red2)
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
            mean: T.Tensor((C,), "float32"),
            weight: T.Tensor((C,), "float32"),
            rstd: T.Tensor((C,), "float32"),
            grad_weight: T.Tensor((C,), "float32"),
            grad_bias: T.Tensor((C,), "float32"),
            scale_out: T.Tensor((C,), "float32"),
            shift_out: T.Tensor((C,), "float32"),
            centered_coef_out: T.Tensor((C,), "float32"),
        ):
            # R37: pure AIV (one block per AIV, no AIC carrier / subid
            # duplication) -- same treatment that gave wide R21 its launch win.
            with T.Kernel(num_fin, is_npu=True) as (cid, _):
                    pw = T.alloc_ub((1, splits), "float32")
                    pwx = T.alloc_ub((1, splits), "float32")
                    m1 = T.alloc_ub((1, 1), "float32")
                    mean_vec = T.alloc_ub((1, splits), "float32")
                    rstd_vec = T.alloc_ub((1, splits), "float32")
                    tmp = T.alloc_ub((1, splits), "float32")
                    tot = T.alloc_ub((1, 1), "float32")
                    totx = T.alloc_ub((1, 1), "float32")
                    w1 = T.alloc_ub((1, 1), "float32")
                    r1 = T.alloc_ub((1, 1), "float32")
                    sc = T.alloc_ub((1, 1), "float32")
                    per = T.alloc_ub((1, 1), "float32")
                    sh = T.alloc_ub((1, 1), "float32")
                    ce = T.alloc_ub((1, 1), "float32")

                    for i in T.serial(nl_fin):
                        c = i * num_fin + cid
                        if c < C:
                            T.copy(partial_sum[c:c + 1, 0:splits], pw)
                            T.copy(partial_sum_xhat[c:c + 1, 0:splits], pwx)
                            # R7: pwx is Σg*x; convert to Σg*x_hat on the
                            # (1,splits) vector (avoids the (1,1) scalar vsub
                            # verifier issue).
                            T.copy(mean[c:c + 1], m1)
                            T.copy(rstd[c:c + 1], r1)
                            T.vbrc(m1, mean_vec)
                            T.vbrc(r1, rstd_vec)
                            T.vmul(pw, mean_vec, tmp)   # mean * Σg
                            T.vsub(pwx, tmp, pwx)       # Σgx - mean*Σg
                            T.vmul(pwx, rstd_vec, pwx)  # * rstd
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
        if native_apply:
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
                with T.Kernel(KC, is_npu=True) as (cid, subid):
                    with T.Scope("Vector"):
                        g_stage = T.alloc_ub((bN_apply, bS_apply), dtype)
                        x_stage = T.alloc_ub((bN_apply, bS_apply), dtype)
                        m1 = T.alloc_ub((1, 1), "float32")
                        s1 = T.alloc_ub((1, 1), "float32")
                        h1 = T.alloc_ub((1, 1), "float32")
                        c1 = T.alloc_ub((1, 1), "float32")
                        c3 = T.alloc_ub((1, 1), "float32")
                        t1 = T.alloc_ub((1, 1), "float32")
                        sd = T.alloc_ub((1, 1), dtype)
                        hd = T.alloc_ub((1, 1), dtype)
                        cd = T.alloc_ub((1, 1), dtype)
                        c3d = T.alloc_ub((1, 1), dtype)
                        for i in T.serial(nlC):
                            c = i * (KC * 2) + cid * 2 + subid
                            if c < C:
                                T.copy(mean[c:c + 1], m1)
                                T.copy(scale[c:c + 1], s1)
                                T.copy(shift[c:c + 1], h1)
                                T.copy(center[c:c + 1], c1)
                                T.vmul(c1, m1, t1)
                                T.vadd(h1, t1, c3)
                                T.vcast(s1, sd, round_mode="rint")
                                T.vcast(h1, hd, round_mode="rint")
                                T.vcast(c1, cd, round_mode="rint")
                                T.vcast(c3, c3d, round_mode="rint")
                                for nn in T.serial(n_tile_N_apply):
                                    n0 = nn * bN_apply
                                    rN = T.min(bN_apply, N - n0)
                                    for tt in T.serial(n_tile_S_apply):
                                        s0 = tt * bS_apply
                                        real = T.min(bS_apply, S - s0)
                                        T.copy(
                                            grad_out[n0:n0 + rN, c, s0:s0 + real],
                                            g_stage[0:rN, 0:real],
                                        )
                                        T.copy(
                                            x[n0:n0 + rN, c, s0:s0 + real],
                                            x_stage[0:rN, 0:real],
                                        )
                                        T.vmul(g_stage, sd, g_stage)
                                        T.vadd(g_stage, c3d, g_stage)
                                        T.vmul(x_stage, cd, x_stage)
                                        T.vsub(g_stage, x_stage, g_stage)
                                        T.copy(
                                            g_stage[0:rN, 0:real],
                                            grad_x[n0:n0 + rN, c, s0:s0 + real],
                                        )
            return apply

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
            with T.Kernel(KC, is_npu=True) as (cid, subid):
                with T.Scope("Vector"):
                    g_stage = T.alloc_ub((bN_apply, bS_apply), dtype)
                    x_stage = T.alloc_ub((bN_apply, bS_apply), dtype)
                    g32 = T.alloc_ub((bN_apply, bS_apply), "float32")
                    x32 = T.alloc_ub((bN_apply, bS_apply), "float32")
                    brc = T.alloc_ub((bN_apply, bS_apply), "float32")
                    m1 = T.alloc_ub((1, 1), "float32")
                    s1 = T.alloc_ub((1, 1), "float32")
                    h1 = T.alloc_ub((1, 1), "float32")
                    c1 = T.alloc_ub((1, 1), "float32")
                    c3 = T.alloc_ub((1, 1), "float32")
                    t1 = T.alloc_ub((1, 1), "float32")
                    for i in T.serial(nlC):
                        c = i * (KC * 2) + cid * 2 + subid
                        if c < C:
                            T.copy(mean[c:c + 1], m1)
                            T.copy(scale[c:c + 1], s1)
                            T.copy(shift[c:c + 1], h1)
                            T.copy(center[c:c + 1], c1)
                            T.vmul(c1, m1, t1)
                            T.vadd(h1, t1, c3)
                            for nn in T.serial(n_tile_N_apply):
                                n0 = nn * bN_apply
                                rN = T.min(bN_apply, N - n0)
                                for tt in T.serial(n_tile_S_apply):
                                    s0 = tt * bS_apply
                                    real = T.min(bS_apply, S - s0)
                                    T.copy(
                                        grad_out[n0:n0 + rN, c, s0:s0 + real],
                                        g_stage[0:rN, 0:real],
                                    )
                                    T.copy(
                                        x[n0:n0 + rN, c, s0:s0 + real],
                                        x_stage[0:rN, 0:real],
                                    )
                                    if use_transit:
                                        T.vcast(g_stage, g32, round_mode="rint")
                                    else:
                                        T.copy(g_stage[0:rN, 0:real], g32[0:rN, 0:real])
                                    if use_transit:
                                        T.vcast(x_stage, x32, round_mode="rint")
                                    else:
                                        T.copy(x_stage[0:rN, 0:real], x32[0:rN, 0:real])
                                    T.vbrc(s1, brc)
                                    T.vmul(g32, brc, g32)
                                    T.vbrc(c3, brc)
                                    T.vadd(g32, brc, g32)
                                    T.vbrc(c1, brc)
                                    T.vmul(x32, brc, x32)
                                    T.vsub(g32, x32, g32)
                                    if use_transit:
                                        T.vcast(g32, g_stage, round_mode="rint")
                                    else:
                                        T.copy(g32[0:rN, 0:real], g_stage[0:rN, 0:real])
                                    T.copy(
                                        g_stage[0:rN, 0:real],
                                        grad_x[n0:n0 + rN, c, s0:s0 + real],
                                    )
        return apply


    @tilelang.jit(out_idx=[-1], target="npuir")
    def _apply_fused():
        """R34: finalize folded into the apply head (large-spatial fp16 only).

        Removes the standalone ``finalize`` launch (12.6us) by recomputing the
        per-channel coefficients from the stats partials inside each apply
        block; each channel is owned by exactly one (cid, subid), so the
        grad_weight / grad_bias writes stay race-free.
        """
        @T.prim_func
        def apply_fused(
            grad_out: T.Tensor((N, C, S), dtype),
            x: T.Tensor((N, C, S), dtype),
            partial_sum: T.Tensor((C, splits), "float32"),
            partial_sum_xhat: T.Tensor((C, splits), "float32"),
            mean: T.Tensor((C,), "float32"),
            weight: T.Tensor((C,), "float32"),
            rstd: T.Tensor((C,), "float32"),
            grad_weight: T.Tensor((C,), "float32"),
            grad_bias: T.Tensor((C,), "float32"),
            grad_x: T.Tensor((N, C, S), dtype),
        ):
            with T.Kernel(KC, is_npu=True) as (cid, subid):
                with T.Scope("Vector"):
                    g_stage = T.alloc_ub((bN_apply, bS_apply), dtype)
                    x_stage = T.alloc_ub((bN_apply, bS_apply), dtype)
                    pw = T.alloc_ub((1, splits), "float32")
                    pwx = T.alloc_ub((1, splits), "float32")
                    m1 = T.alloc_ub((1, 1), "float32")
                    r1 = T.alloc_ub((1, 1), "float32")
                    mvec = T.alloc_ub((1, splits), "float32")
                    rvec = T.alloc_ub((1, splits), "float32")
                    tmpv = T.alloc_ub((1, splits), "float32")
                    tot = T.alloc_ub((1, 1), "float32")
                    totx = T.alloc_ub((1, 1), "float32")
                    w1 = T.alloc_ub((1, 1), "float32")
                    sc = T.alloc_ub((1, 1), "float32")
                    per = T.alloc_ub((1, 1), "float32")
                    sh = T.alloc_ub((1, 1), "float32")
                    ce = T.alloc_ub((1, 1), "float32")
                    c3 = T.alloc_ub((1, 1), "float32")
                    t1 = T.alloc_ub((1, 1), "float32")
                    sd = T.alloc_ub((1, 1), dtype)
                    hd = T.alloc_ub((1, 1), dtype)
                    cd = T.alloc_ub((1, 1), dtype)
                    c3d = T.alloc_ub((1, 1), dtype)
                    for i in T.serial(nlC):
                        c = i * (KC * 2) + cid * 2 + subid
                        if c < C:
                            T.copy(partial_sum[c:c + 1, 0:splits], pw)
                            T.copy(partial_sum_xhat[c:c + 1, 0:splits], pwx)
                            T.copy(mean[c:c + 1], m1)
                            T.copy(rstd[c:c + 1], r1)
                            T.vbrc(m1, mvec)
                            T.vbrc(r1, rvec)
                            T.vmul(pw, mvec, tmpv)
                            T.vsub(pwx, tmpv, pwx)
                            T.vmul(pwx, rvec, pwx)
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
                            T.vmul(w1, r1, sc)
                            T.vmul(sc, L_inv, per)
                            T.vmul(per, tot, sh)
                            T.vmul(sh, -1.0, sh)
                            T.vmul(per, totx, ce)
                            T.vmul(ce, r1, ce)
                            T.vmul(ce, m1, t1)
                            T.vadd(sh, t1, c3)
                            T.vcast(sc, sd, round_mode="rint")
                            T.vcast(sh, hd, round_mode="rint")
                            T.vcast(ce, cd, round_mode="rint")
                            T.vcast(c3, c3d, round_mode="rint")
                            for nn in T.serial(n_tile_N_apply):
                                n0 = nn * bN_apply
                                rN = T.min(bN_apply, N - n0)
                                for tt in T.serial(n_tile_S_apply):
                                    s0 = tt * bS_apply
                                    real = T.min(bS_apply, S - s0)
                                    T.copy(
                                        grad_out[n0:n0 + rN, c, s0:s0 + real],
                                        g_stage[0:rN, 0:real],
                                    )
                                    T.copy(
                                        x[n0:n0 + rN, c, s0:s0 + real],
                                        x_stage[0:rN, 0:real],
                                    )
                                    T.vmul(g_stage, sd, g_stage)
                                    T.vadd(g_stage, c3d, g_stage)
                                    T.vmul(x_stage, cd, x_stage)
                                    T.vsub(g_stage, x_stage, g_stage)
                                    T.copy(
                                        g_stage[0:rN, 0:real],
                                        grad_x[n0:n0 + rN, c, s0:s0 + real],
                                    )
        return apply_fused

    _apply.fused = _apply_fused if (native_apply and S >= 8192) else None
    return _stats, _finalize, _apply


# ---------------------------------------------------------------------------
# Unified hierarchical test harness (all three kernels)
# ---------------------------------------------------------------------------
_FACTORY_CACHE = {}

# Per-(layer, dtype) worst max_abs_err over the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_factory(kernel, N, C, S, dtype_str):
    """Return the compiled factory for one of the three kernels."""
    key = (kernel, N, C, S, dtype_str)
    if key not in _FACTORY_CACHE:
        factory = {
            "general": _batch_norm_bwd_kernel,
            "wide": _batch_norm_bwd_wide_kernel,
            "split": _batch_norm_bwd_split_kernel,
        }[kernel]
        _FACTORY_CACHE[key] = factory(N, C, S, dtype_str)
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


def _infer(kernel, N, C, S, grad_out, x, weight, mean, rstd, dtype_str,
           bN=None, bS=None):
    """Run one kernel case on-device, returning (grad_x, grad_weight, grad_bias).

    Orchestrates the host contract of each kernel: the general kernel
    preallocates ``grad_weight``/``grad_bias`` (``out_idx=[-1]`` returns only
    ``grad_x``); the wide kernel returns all three; the split kernel runs the
    stats -> finalize -> apply stage triple.
    """
    dev = grad_out.device
    if kernel == "split":
        stats, finalize, apply_ = _get_factory(kernel, N, C, S, dtype_str)
        splits = _plan_splits(N, C, S)
        empty = lambda *shape: torch.empty(*shape, device=dev, dtype=torch.float32)
        partial_sum = empty(C, splits)
        partial_sum_xhat = empty(C, splits)
        stats()(grad_out, x, mean, rstd, partial_sum, partial_sum_xhat)
        fused_factory = getattr(apply_, "fused", None)
        if fused_factory is not None:
            grad_weight = empty(C)
            grad_bias = empty(C)
            grad_x = fused_factory()(
                grad_out, x, partial_sum, partial_sum_xhat, mean, weight, rstd,
                grad_weight, grad_bias,
            )
            return grad_x, grad_weight, grad_bias
        grad_weight = empty(C)
        grad_bias = empty(C)
        scale = empty(C)
        shift = empty(C)
        centered_coef = empty(C)
        finalize()(
            partial_sum, partial_sum_xhat, mean, weight, rstd,
            grad_weight, grad_bias, scale, shift, centered_coef,
        )
        grad_x = apply_()(grad_out, x, mean, scale, shift, centered_coef)
        return grad_x, grad_weight, grad_bias
    factory = _get_factory(kernel, N, C, S, dtype_str)
    if kernel == "general":
        prog = factory() if (bN is None and bS is None) else factory(bN, bS)
        grad_weight = torch.empty(C, device=dev, dtype=torch.float32)
        grad_bias = torch.empty(C, device=dev, dtype=torch.float32)
        out = prog(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out, grad_weight, grad_bias
    # wide
    prog = factory() if bS is None else factory(bS)
    gx, gw, gb = prog(grad_out, x, weight, mean, rstd)
    return gx, gw, gb


def _run_case(kernel, N, C, S, dtype_str, tag="L0", seed=0, bN=None, bS=None,
              equal_nan=False):
    """Run one kernel case and compare all three outputs against the golden.

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

    gx, gw, gb = _infer(
        kernel, N, C, S,
        grad_out.npu(), x.npu(), weight.npu(), mean.npu(), rstd.npu(),
        dtype_str, bN=bN, bS=bS,
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
                f"[{tag}] FAIL: kernel={kernel} shape=({N},{C},{S}) "
                f"dtype={dtype_str} output={name} max_abs={max_diff:.3e} "
                f"violations={violations}/{out.numel()}"
            )
            raise

    best = _STATS_MAX_DIFF.get((tag.split("-")[0], dtype_str), 0.0)
    if max_diff_all > best:
        _STATS_MAX_DIFF[(tag.split("-")[0], dtype_str)] = max_diff_all
    print(
        f"[{tag}] PASS: kernel={kernel} shape=({N},{C},{S}) dtype={dtype_str} "
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
    """L0 gate suite -- blocking.  Covers all three kernels on representative
    shapes (held / tiled / split, fp16/bf16/fp32, degenerate reduce branches).
    """
    print("== L0 (blocking) ==")
    failures = []
    # (kernel, N, C, S, dtype) -- DESIGN.md section 8.3 table.
    cases = [
        ("wide", 32, 64, 1, "float32"),          # resnet-fc fp32 (S=1)
        ("wide", 32, 64, 1, "float16"),          # resnet-fc fp16 (S=1)
        ("wide", 8, 64, 1024, "float16"),        # resnet-stage1 (held critical)
        ("wide", 8, 64, 1024, "bfloat16"),       # resnet-stage1 bf16
        ("wide", 4, 128, 1024, "float16"),       # resnet-stage2
        ("wide", 4, 256, 784, "float32"),        # resnet-stage3 fp32 (non-2^k S)
        ("wide", 4, 256, 784, "float16"),        # resnet-stage3 fp16
        ("general", 8, 64, 1024, "float16"),     # general held
        ("general", 4, 128, 1024, "float16"),    # general
        ("general", 4, 256, 784, "float16"),     # general non-2^k S
        ("general", 4, 64, 4097, "float16"),     # general tiled (tail)
        ("general", 2, 8, 3, "float16"),         # general tiny
        ("general", 1, 4, 16, "bfloat16"),       # general N=1 degenerate
        ("split", 4, 128, 262144, "float16"),    # split large-spatial
        ("split", 4, 128, 1048576, "float16"),   # split 251 S-tiles
        ("split", 8192, 128, 8, "float16"),      # split N-heavy
    ]
    for idx, (kernel, N, C, S, dtype_str) in enumerate(cases, start=1):
        _try_case(
            failures, f"L0-{idx}",
            kernel=kernel, N=N, C=C, S=S, dtype_str=dtype_str, seed=idx,
        )

    # Factory contract: empty tensor (S=0) rejected at the factory boundary.
    try:
        _batch_norm_bwd_kernel(2, 8, 0, "float16")
    except ValueError:
        print("[L0] PASS: empty-tensor (S=0) rejected at the factory")
    else:
        failures.append(("L0-empty", AssertionError("S=0 not rejected")))
        print("[collected] L0-empty FAILED: S=0 not rejected")

    # Factory contract: 1-element plane (N==1 and S==1) rejected.
    try:
        _batch_norm_bwd_kernel(1, 8, 1, "float16")()
    except ValueError:
        print("[L0] PASS: 1-element plane (N==1,S==1) rejected at the factory")
    else:
        failures.append(("L0-scalar", AssertionError("scalar_plane not rejected")))
        print("[collected] L0-scalar FAILED: scalar_plane not rejected")

    if failures:
        print(f"[L0] {len(failures)} failing case(s): "
              + ", ".join(cid for cid, _ in failures))
    else:
        print(f"[L0] ALL PASS ({len(cases) + 2} items)")
    return failures


def run_L1():
    """L1 gate suite -- blocking.  dtype x shape x path grid + contracts."""
    print("== L1 (blocking) ==")
    failures = []

    grid = [
        ("wide", 128, 512, 512, "float32"),
        ("wide", 128, 512, 512, "float16"),
        ("wide", 128, 512, 512, "bfloat16"),
        ("general", 130, 256, 1024, "float16"),
        ("general", 33, 1024, 200, "float32"),
        ("general", 17, 514, 1000, "bfloat16"),
        ("general", 4, 300, 4096, "float16"),
        ("general", 4, 64, 100000, "float16"),
        ("general", 2, 40, 8192, "float32"),
        ("general", 1, 48, 20000, "float16"),
        ("general", 6, 8, 1, "float16"),
        ("general", 3, 5, 7, "bfloat16"),
        ("split", 4, 64, 4096, "float16"),
        ("split", 4, 64, 4097, "float16"),
    ]
    for i, (kernel, N, C, S, dtype_str) in enumerate(grid):
        _try_case(
            failures, f"L1-{i}",
            kernel=kernel, N=N, C=C, S=S, dtype_str=dtype_str, seed=1000 + i,
        )

    # Factory guards (general kernel): UB budget + invalid inputs.
    for N, C, S, bN, bS, dtype_str in (
        (4, 64, 4097, 2, 8192, "float16"),
        (4, 64, 4097, 8, 4096, "float32"),
    ):
        try:
            _batch_norm_bwd_kernel(N, C, S, dtype_str)(bN, bS)
        except ValueError as exc:
            print(f"[L1] PASS: UB guard rejects {dtype_str} bN={bN} bS={bS} ({exc})")
        else:
            failures.append(
                (f"L1/UB-guard/{dtype_str}", AssertionError("oversize not rejected"))
            )
    for bad in ((0, 64, 16, "float16"), (4, 0, 16, "float16"), (4, 64, 16, "int32")):
        try:
            _batch_norm_bwd_kernel(*bad)
        except ValueError:
            print(f"[L1] PASS: factory guard rejects {bad}")
        else:
            failures.append((f"L1/factory-guard/{bad}", AssertionError(f"{bad} not rejected")))

    if failures:
        print(f"[L1] {len(failures)} failing case(s): "
              + ", ".join(cid for cid, _ in failures))
    else:
        print(f"[L1] ALL PASS ({len(grid)} grid + contracts)")
    return failures


def run_L2():
    """L2 extended suite -- warn only, non-blocking (record without blocking)."""
    print("== L2 (warn-only) ==")
    cases = [
        ("general", 32, 2048, 512, "float16"),
        ("general", 8, 64, 900, "float16"),
        ("wide", 4, 128, 1048576, "float16"),
        ("split", 13108, 128, 5, "float16"),
    ]
    for i, (kernel, N, C, S, dtype_str) in enumerate(cases):
        try:
            _run_case(kernel, N, C, S, dtype_str, tag="L2", seed=3000 + i)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record without blocking): "
                  f"kernel={kernel} shape=({N},{C},{S}) {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    print("== Boundary (warn-only) ==")
    N, C, S = 4, 16, 256
    kernel = "general"

    def _run_special(name, dtype_str, gx, xx, w, m, r):
        try:
            torch_dtype = _TOLERANCE[dtype_str][0]
            gx = gx.to(torch_dtype)
            xx = xx.to(torch_dtype)
            gx_out, gw_out, gb_out = _infer(
                kernel, N, C, S, gx.npu(), xx.npu(), w.npu(), m.npu(), r.npu(),
                dtype_str,
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
