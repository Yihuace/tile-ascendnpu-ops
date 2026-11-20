# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""BatchNorm backward general kernel for npuir (Expert / Vector-only).

Migrated from the GPU TileLang factory ``_batch_norm_bwd_kernel``
(``TileOPs-main/src/tileops/kernels/norm/batch_norm.py`` L1487 + host
``BatchNormBwdKernel`` L2146 + config source ``_TiledPath`` L66) to
``target="npuir"`` with ``ascend_mode="Expert"`` and a single
``T.Scope("Vector")`` -- no Cube, no ``T.gemm``, no L1/L0C.

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

NPU redesign (DESIGN.md section 0.6):
  R1: one-dimensional ``T.Kernel(num_kernels, is_npu=True)`` persistent
      core split -- one core per channel, ``num_kernels = min(C, 40)``
      (physical AI cores x 2 for a pure-Vector op), in-core ``T.serial``
      grid-stride task loop with an ``if c < C`` guard (static bound).
  R2: the per-channel reduction over the flattened ``N*S`` axis uses an
      in-core two-step ``T.reduce`` on a UB ``(bN, bS)`` plane:
      ``(bN, bS) --dims=[1]--> (bN, 1) --dims=[0]--> (1, 1)``.
  R3: N/S sub-block scheduling uses in-core ``T.serial(n_tile_N)`` x
      ``T.serial(n_tile_S)`` (static bounds); no ``T.Pipelined``.
  R4: address/layout: affine 2-D ``(bN, bS)`` plane (S is the contiguous
      inner axis) instead of the GPU flat ``l//S, l%S`` non-affine walk.
  R5: tail handling via ``T.min(bN, N-n0)`` / ``T.min(bS, S-s0)`` +
      src/dst explicit slices (I/O) + ``T.reduce(size=[rN, rS])`` /
      ``size=[rN, 1]`` (reduction extent).
  R6: explicit fp32 transit chain for fp16/bf16 via ``T.vcast(rint)``
      (bf16 has no Vector arithmetic); a single round-back on write.
  R7: oversized ``N*S`` uses in-core 2-D tiling (two scans per channel,
      bounded UB) -- no second block, no cross-core merge, no atomics.

held / tiled: the whole ``(N, S)`` channel plane is held in UB when it fits
the budget (``N*S*bpe <= UB_SAFE``; ``bN = N``, ``bS = S``, single scan);
otherwise the channel is processed in ``(bN, bS)`` sub-blocks with a serial
two-scan loop (bounded UB).  The budget uses the measured multi-buffered
per-element cost (``UB_PE_TRANSIT`` / ``UB_PE_FP32``).

Corner fixes carried over from DESIGN.md section 0.0 (v0 -> v1):
  * ``T.clear(red)`` / ``T.clear(red2)`` are INSIDE the ``for nn`` loop
    (per N-block reset of the row-sum accumulators); ``T.clear(gb)`` /
    ``T.clear(gw)`` are outside once.  This prevents the v0 cross-N-block
    double-count for every ``n_tile_N >= 2`` shape.
  * every v-op runs on the full static ``(bN, bS)`` buffers; runtime extents
    (``rN`` / ``rS``) are only passed to ``T.copy`` (I/O) and
    ``T.reduce(size=...)`` (reduction).

Interface contract (source unchanged): the factory
``_batch_norm_bwd_kernel(N, C, S, dtype)`` mirrors the source signature and
returns a JIT factory taking the tile knobs ``(bN, bS)``.  ``main`` keeps the
source parameter order
``(grad_out, x, weight, mean, rstd, grad_weight, grad_bias, grad_x)`` with
``out_idx=[-1]`` (only ``grad_x`` auto-allocated; ``grad_weight`` /
``grad_bias`` are host preallocated and passed in).  ``len(out_idx) == 1`` so
the JIT returns the ``grad_x`` tensor (not a list); the host calls
``program(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)``.

Run the embedded hierarchical tests:
    python _batch_norm_bwd_kernel.py --level L0
    python _batch_norm_bwd_kernel.py --level all
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
    "golden_batch_norm_bwd",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported dtypes (factory validation).
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# UB budget (DESIGN.md sections 4.5 / 9.1).  The compiler auto multi-buffers
# the persistent serial task loop (num_local_tasks >= 2, TRAP-UB-multibuffer-
# inflation), so the *resident* footprint per element exceeds the nominal
# buffer sizes.  Measured on this toolchain for the sibling wide kernel (same
# buffer set: stage + 4 fp32 planes):
#   fp16/bf16 transit path: ~22.0 B/elem  (4*2272 fp16 -> 200320 B)
#   fp32 path:              ~25.8 B/elem
# The safe budget is 180 KiB (below the 192 KiB physical capacity).  bN/bS are
# derived from this budget (DESIGN.md section 4.5 formula, refined cost).
_UB_SAFE_BYTES = 184_320
_UB_CAPACITY_BYTES = 196_608
_UB_PE_TRANSIT = 22
_UB_PE_FP32 = 26
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
# Tiling plan (single source of truth for factory and tests)
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


# ---------------------------------------------------------------------------
# Precision comparison + hierarchical testing
# ---------------------------------------------------------------------------

_FACTORY_CACHE = {}

# Per-(layer, dtype) worst max_abs_err over the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_factory(N, C, S, dtype_str):
    key = (N, C, S, dtype_str)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _batch_norm_bwd_kernel(N, C, S, dtype_str)
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


def _infer(N, C, S, grad_out, x, weight, mean, rstd, dtype_str, bN=None, bS=None):
    """Run the kernel, orchestrating the host-preallocated channel grads.

    ``grad_weight`` / ``grad_bias`` are not ``out_idx``; the host allocates
    them and passes them in (source host contract).  ``out_idx=[-1]`` makes the
    JIT return the ``grad_x`` tensor (unwrapped defensively).
    """
    factory = _get_factory(N, C, S, dtype_str)
    kernel = factory() if (bN is None and bS is None) else factory(bN, bS)
    dev = grad_out.device
    grad_weight = torch.empty(C, device=dev, dtype=torch.float32)
    grad_bias = torch.empty(C, device=dev, dtype=torch.float32)
    out = kernel(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out, grad_weight, grad_bias


def _run_case(N, C, S, dtype_str, tag="L0", seed=0, bN=None, bS=None, equal_nan=False):
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
        N, C, S,
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
                f"[{tag}] FAIL: shape=({N},{C},{S}) dtype={dtype_str} "
                f"bN={bN} bS={bS} output={name} max_abs={max_diff:.3e} "
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

    Covers fp16/bf16/fp32, held and tiled paths, the large-C ``n_tile_N == 2``
    v0 double-count regression, large-spatial 251 S-tiles, non-divisible dims,
    batch > 1, and the S=1 / N=1 degenerate reduce branches.
    """
    print("== L0 (blocking) ==")
    failures = []
    # (N, C, S, dtype) -- DESIGN.md section 8.3 table.
    cases = [
        (32, 64, 1, "float32"),          # L0-1  resnet-fc fp32 (S=1)
        (32, 64, 1, "float16"),          # L0-2  resnet-fc fp16 (S=1)
        (8, 64, 1024, "float16"),        # L0-3  resnet-stage1 (held critical)
        (8, 64, 1024, "bfloat16"),       # L0-4  resnet-stage1 bf16
        (4, 128, 1024, "float16"),       # L0-5  resnet-stage2
        (4, 256, 784, "float32"),        # L0-6  resnet-stage3 fp32 (non-2^k S)
        (4, 256, 784, "float16"),        # L0-7  resnet-stage3 fp16
        (4, 128, 1048576, "float16"),    # L0-8  large-spatial (tiled, S tail)
        (32, 2048, 512, "float16"),      # L0-9  large-C (n_tile_N=2, N tail)
        (8, 64, 900, "float16"),         # L0-10 test 30x30 (non-vector-width S)
        (2, 8, 3, "float16"),            # L0-11 tiny shape
        (1, 4, 16, "bfloat16"),          # L0-12 N=1 degenerate
        (4, 64, 4097, "float16"),        # L0-13 tiny non-div tiled (tail)
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

    # L0-14: empty tensor (L=0) is intercepted at the factory / op boundary
    # (the kernel must never receive S=0).  Contract check, not a run.
    try:
        _batch_norm_bwd_kernel(2, 8, 0, "float16")
    except ValueError:
        print("[L0-14] PASS: empty-tensor (S=0) rejected at the factory")
    else:
        failures.append(("L0-14", AssertionError("S=0 not rejected")))
        print("[collected] L0-14 FAILED: S=0 not rejected")

    # L0-15: 1-element plane (N==1 and S==1, L==1) is a factory-level
    # ``scalar_plane`` reject (T.vbrc / same-shape v-op unsupported); the
    # op/wrapper routes this shape.
    try:
        _batch_norm_bwd_kernel(1, 8, 1, "float16")()
    except ValueError:
        print("[L0-15] PASS: 1-element plane (N==1,S==1) rejected at the factory")
    else:
        failures.append(("L0-15", AssertionError("scalar_plane not rejected")))
        print("[collected] L0-15 FAILED: scalar_plane not rejected")

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print(f"[L0] ALL PASS ({len(cases) + 2} items)")
    return failures


def _run_L1_contract(failures):
    """L1 factory guards (no precision comparison)."""
    # UB guard: N=4, S=4097 fp16 -> tiled; force an oversize bS.
    for N, C, S, bN, bS, dtype_str in (
        (4, 64, 4097, 2, 8192, "float16"),   # 2*8192*22 = 360448 > 184320
        (4, 64, 4097, 8, 4096, "float32"),   # 8*4096*26 = 851968 > 184320
    ):
        try:
            _batch_norm_bwd_kernel(N, C, S, dtype_str)(bN, bS)
        except ValueError as exc:
            print(f"[L1] PASS: UB guard rejects {dtype_str} bN={bN} bS={bS} ({exc})")
        else:
            failures.append(
                (
                    f"L1/UB-guard/{dtype_str}/bN={bN},bS={bS}",
                    AssertionError(f"{dtype_str} bN={bN} bS={bS} not rejected"),
                )
            )

    # Non-positive / unsupported inputs are rejected at the factory.
    for bad in ((0, 64, 16, "float16"), (4, 0, 16, "float16"),
                (4, 64, 16, "int32")):
        try:
            _batch_norm_bwd_kernel(*bad)
        except ValueError:
            print(f"[L1] PASS: factory guard rejects {bad}")
        else:
            failures.append(
                (f"L1/factory-guard/{bad}", AssertionError(f"{bad} not rejected"))
            )


def run_L1():
    """L1 gate suite -- blocking.  dtype x shape x path grid + contracts."""
    print("== L1 (blocking) ==")
    failures = []

    # dtype x {held, tiled, non-div, degenerate} grid.
    grid = [
        (128, 512, 512, "float32"),      # held fp32
        (128, 512, 512, "float16"),      # held fp16
        (128, 512, 512, "bfloat16"),     # held bf16
        (130, 256, 1024, "float16"),     # batch tail (N even, C=256)
        (33, 1024, 200, "float32"),      # C=1024 not divisible by 40
        (17, 514, 1000, "bfloat16"),     # odd C, odd spatial
        (4, 300, 4096, "float16"),       # C=300 -> 40 cores x8 (tail cores)
        (4, 64, 100000, "float16"),      # tiled, non-div tail
        (2, 40, 8192, "float32"),        # C == core count exactly
        (1, 48, 20000, "float16"),       # N=1 tiled (dims=[0] skipped)
        (6, 8, 1, "float16"),            # S=1, C != core count
        (3, 5, 7, "bfloat16"),           # tiny odd everything
        (64, 1536, 256, "float16"),      # large-C, n_tile_N>=2 (2-D N tiling)
    ]
    for i, (N, C, S, dtype_str) in enumerate(grid):
        _try_case(
            failures,
            f"L1-{i}",
            N=N,
            C=C,
            S=S,
            dtype_str=dtype_str,
            seed=1000 + i,
        )

    # Explicit bN/bS override (tiled path) on a held-eligible shape.
    _try_case(
        failures,
        "L1-tile-override",
        N=4,
        C=64,
        S=4096,
        dtype_str="float16",
        bN=2,
        bS=2048,
        seed=2000,
    )

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
        (2, 3, 3, "bfloat16"),           # tiny odd
        (1, 8, 65536, "float32"),        # single batch, tiled
        (30, 96, 33, "float16"),         # large N, small S
        (1, 1, 1, "float16"),            # minimal (scalar_plane -> expected WARN)
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
            torch_dtype = _TOLERANCE[dtype_str][0]
            gx = gx.to(torch_dtype)
            xx = xx.to(torch_dtype)
            gx_out, gw_out, gb_out = _infer(
                N, C, S, gx.npu(), xx.npu(), w.npu(), m.npu(), r.npu(), dtype_str
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

    # Zero-variance channel (x constant).
    xx = torch.zeros(N, C, S)
    gx, _, w, m, r = base()
    _run_special("const-x", "float32", gx, xx, w, m, r)

    # Large dynamic range.
    gx, xx, w, m, r = base()
    gx = gx * 1e3
    xx = xx * 1e-3
    _run_special("large-range", "float32", gx, xx, w, m, r)

    # Extreme weight.
    gx, xx, w, m, r = base()
    w = w * 1e4
    _run_special("extreme-weight", "float32", gx, xx, w, m, r)

    # inf / NaN natural propagation.
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
