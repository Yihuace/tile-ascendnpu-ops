# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""GLA single-token decode kernel for npuir (Vector, Developer mode).

R40: bf16/DK=DV=128 serving shapes route to `_gla_decode_algo1_1head`
(1-head + fp16 + k_tile=128, decoupled o), measured 5-shape mean 12.25us.
fp16/fp32 and non-128 shapes keep the R32 routes.

R46 (2026-11-18): `ts` is allocated only on the cast path (the fp16 state
contract left a dead 32 KiB plane), which lets the next-head state land
directly in the idle ping-pong plane.  The 32 KiB UB->UB handoff copy of R44
is gone; same-session msprof: b8-h64 22.84 -> 21.14us (-7.4%), b8-h32
14.44 -> 13.48 (-6.6%), b1-* unchanged; `o`/`new_state` bitwise equal.

R32 (2026-11-12): H=64 now uses full-tile 2-head writeback by default
(`USE_HALFWB_H64=False`), based on a same-session 3-pair A/B; all other
routes are unchanged.

Stage-4 round-3 winner: **persistent core partitioning with compile-time
shape dispatch**.

Same semantics and tiling as the previous winner (k_tile=32; bf16/fp16 ->
fp32 transit path unchanged, fp32 accumulation is the GPU contract):

    state_new[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
    o[dv]             = scale * sum_dk q[dk] * state_new[dk, dv]

Change vs the flat kernel (``T.Kernel(B * H)``, one block per (b,h)): the
launch grid is capped at ``_plan_cores(B*H)`` physical AIV cores and each core
walks the remaining heads in a compile-time static grid-stride loop

    blk = s * ncore + cid            # s = 0 .. n_iter-1
    if blk < B*H: bid = blk // H; hid = blk % H

which removes the multi-wave block dispatch of the flat grid (b8-h64 = 512
blocks = 12.8 waves, b8-h32 = 256 = 6.4 waves).  Under-subscribed shapes
(``B*H <= ncore``) keep the flat body verbatim: the dispatch is a pure
compile-time integer choice inside the jitted factory, no runtime branch, no
Python-level op/factory split.

Exposed for the wrapper switch block: ``TUNED_K_TILE`` / ``TUNED_NCORE``.

Run: python GLADecodeOp_round3_dispatch.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch

_ACCUM = "float32"
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")
_TOLERANCE = {"float16": 2e-3, "bfloat16": 2e-2, "float32": 1e-4}

# Stage-4 tuned defaults (see opt_log.md Final Summary).
TUNED_K_TILE = 32
# Persistent grid cap: physical AIV cores (NPUUtils.get().get_aicore_num()*2).
TUNED_NCORE = 40
# 2026-11-12 same-session A/B (3 pairs, 910B3 / CANN 9.0.0):
#   b8-h64: full 2-head 38.08/38.70/38.74us vs halfwb 41.50/41.02/41.46us
#   b1-h64: full 2-head 8.58/8.52/8.30us     vs halfwb 9.30/9.04/9.32us
# Full-tile writeback wins 3/3 on both shapes; halfwb remains for A/B only.
USE_HALFWB_H64 = False
# R40: adopt algo1 v3 (1-head + fp16 + k_tile=128) for bf16 128x128.
USE_ALGO1 = True


def _ceil_ktile_fit(dk, dv):
    """Largest divisor of ``dk`` in {32,16,8,4,2,1} whose K-tile UB fits.

    UB = 192KB/AIV (hardware-context.md).  The manual budget counts the 9
    fp32 [kt,DV] working buffers + 2 dtype [kt,DV] transit buffers (plus a
    small ~2.6KB of (1,DV)/(kt,1) buffers not in this formula).  k_tile=32 at
    DK=DV=128 is ~162KB manual / ~182KB after the ~1.10–1.12x BishengIR
    inflation (constants.md) < 192KB and is empirically confirmed to compile;
    k_tile=64 (294912+32768=327KB) is rejected by the same bound.
    """
    for kt in (32, 16, 8, 4, 2, 1):
        if dk % kt:
            continue
        if 9 * kt * dv * 4 + 2 * kt * dv * 2 <= 192 * 1024:
            return kt
    return 1


def _plan_cores(total, ncore=TUNED_NCORE):
    """Compile-time ``(physical cores, heads per core)`` plan for ``B*H`` heads.

    ``ncore`` is the physical AIV count (Ascend910B3: 20 AIC x 2 = 40).  Each
    core walks ``n_iter`` heads in grid-stride order (``blk = s * nc + cid``),
    which replaces the flat grid's ``ceil(B*H / ncore)`` block waves.

    * ``total <= ncore``: one head per core (``n_iter=1``); the caller then
      emits the *flat* kernel body, byte-identical to the pre-persistent
      version (no loop, no bounds guard).
    * ``n_iter == 2`` (``ncore < total <= 2*ncore``): prefer the largest exact
      divisor of ``total`` in ``[ceil(total/2), ncore]`` so both iterations are
      full.  Measured on serving-b1-h64 (64 heads): 32 cores x 2 heads =
      12.51us vs 40 cores x (2,1) heads = 13.90us (-10%), because the tail
      iteration leaves 16 of 40 cores idle while the first iteration runs 40
      concurrent per-head DMA streams (higher latency per head).
    * otherwise: the full core count wins; the idle tail slots are amortised
      over >=3 iterations (measured b8-h32 256 heads: 40 cores/7 iters
      38.54us vs 32/8 iters 39.69us; b8-h64 512 heads: 68.87 vs 75.86us).
    """
    ncore = int(ncore) if ncore else TUNED_NCORE
    if total <= ncore:
        return total, 1
    n_iter = (total + ncore - 1) // ncore
    if n_iter == 2:
        for d in range(min(ncore, total // 2), total // 2 - 1, -1):
            if total % d == 0:
                return d, 2
    return ncore, n_iter


def _use_kt64_reuse(DK, DV, dtype):
    """R6 (2026-11-06): bf16/fp16 128x128 serving shapes use the four-plane
    in-place-reuse kernel with k_tile=64 (two K iterations instead of four).
    Measured -3%..-11% vs the delivered k_tile=32 kernel on all five serving
    workloads.  fp32 / non-128 shapes keep the validated original body."""
    return dtype != "float32" and DK == 128 and DV == 128


def _gla_decode_kernel_kt64reuse(B, H, DK, DV, dtype="float32", scale=None,
                       k_tile=None):
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if scale is None:
        scale = DK ** -0.5
    scale = float(scale)
    if k_tile is None:
        k_tile = 64
    k_tile = int(k_tile)
    if DK % k_tile:
        raise ValueError(f"DK ({DK}) must be divisible by k_tile ({k_tile})")
    acc = _ACCUM
    use_transit = dtype != acc
    total = B * H

    @tilelang.jit(out_idx=[5, 6], target="npuir")
    def _func(k_tile=k_tile, ncore=TUNED_NCORE):
        nc, n_iter = _plan_cores(total, ncore)

        if use_transit:
            @T.prim_func
            def _gla_decode_kt64_main(
                q: T.Tensor((B, H, DK), dtype),
                k: T.Tensor((B, H, DK), dtype),
                v: T.Tensor((B, H, DV), dtype),
                gk: T.Tensor((B, H, DK), dtype),
                state: T.Tensor((B, H, DK, DV), dtype),
                o: T.Tensor((B, H, DV), dtype),
                new_state: T.Tensor((B, H, DK, DV), dtype),
            ):
                with T.Kernel(nc, is_npu=True) as (cid, _):
                    v_ub = T.alloc_ub((1, DV), acc)
                    o2 = T.alloc_ub((1, DV), acc)
                    st = T.alloc_ub((k_tile, DV), acc)
                    out = T.alloc_ub((k_tile, DV), acc)
                    q_ub = T.alloc_ub((k_tile, 1), acc)
                    k_ub = T.alloc_ub((k_tile, 1), acc)
                    g_ub = T.alloc_ub((k_tile, 1), acc)
                    o2_t = T.alloc_ub((1, DV), acc)
                    t_v = T.alloc_ub((1, DV), dtype)
                    t_q = T.alloc_ub((k_tile, 1), dtype)
                    t_k = T.alloc_ub((k_tile, 1), dtype)
                    t_g = T.alloc_ub((k_tile, 1), dtype)
                    t_st = T.alloc_ub((k_tile, DV), dtype)
                    t_o = T.alloc_ub((1, DV), dtype)

                    for s in T.serial(n_iter):
                        blk = s * nc + cid
                        if blk < total:
                            bid = blk // H
                            hid = blk % H
                            T.copy(v[bid, hid, 0:DV], t_v)
                            T.vcast(t_v, v_ub, round_mode="rint")
                            T.clear(o2)

                            for kt in T.serial(DK // k_tile):
                                off = kt * k_tile
                                T.copy(q[bid, hid, off:off + k_tile], t_q)
                                T.vcast(t_q, q_ub, round_mode="rint")
                                T.copy(k[bid, hid, off:off + k_tile], t_k)
                                T.vcast(t_k, k_ub, round_mode="rint")
                                T.copy(gk[bid, hid, off:off + k_tile], t_g)
                                T.vcast(t_g, g_ub, round_mode="rint")
                                T.copy(state[bid, hid, off:off + k_tile, 0:DV],
                                       t_st)
                                T.vcast(t_st, st, round_mode="rint")

                                T.vexp(g_ub, g_ub)
                                # Row-broadcast vmul (no full-plane vbrc)
                                T.vbrc(v_ub, out)
                                T.vmul(out, k_ub, out)        # out = v * k
                                T.vmul(st, g_ub, st)          # state *= exp(g)
                                T.vadd(st, out, st)           # new_state
                                # write back before q destroys new_state
                                T.vcast(st, t_st, round_mode="rint")
                                T.copy(t_st,
                                       new_state[bid, hid, off:off + k_tile,
                                                 0:DV])
                                T.vmul(st, q_ub, st)          # q * new_state
                                T.reduce(st, o2_t, dims=[0],
                                         reduce_mode="sum", clear=True)
                                T.vadd(o2, o2_t, o2)

                            T.vmul(o2, scale, o2)
                            T.vcast(o2, t_o, round_mode="rint")
                            T.copy(t_o, o[bid, hid, 0:DV])

            return _gla_decode_kt64_main

        @T.prim_func
        def _gla_decode_kt64_main(
            q: T.Tensor((B, H, DK), dtype),
            k: T.Tensor((B, H, DK), dtype),
            v: T.Tensor((B, H, DV), dtype),
            gk: T.Tensor((B, H, DK), dtype),
            state: T.Tensor((B, H, DK, DV), dtype),
            o: T.Tensor((B, H, DV), dtype),
            new_state: T.Tensor((B, H, DK, DV), dtype),
        ):
            with T.Kernel(total, is_npu=True) as (cid, _):
                bid = cid // H
                hid = cid % H
                v_ub = T.alloc_ub((1, DV), acc)
                o2 = T.alloc_ub((1, DV), acc)
                st = T.alloc_ub((k_tile, DV), acc)
                out = T.alloc_ub((k_tile, DV), acc)
                dec = T.alloc_ub((k_tile, DV), acc)
                tmp = T.alloc_ub((k_tile, DV), acc)
                q_ub = T.alloc_ub((k_tile, 1), acc)
                k_ub = T.alloc_ub((k_tile, 1), acc)
                g_ub = T.alloc_ub((k_tile, 1), acc)
                o2_t = T.alloc_ub((1, DV), acc)
                T.copy(v[bid, hid, 0:DV], v_ub)
                T.clear(o2)
                for kt in T.serial(DK // k_tile):
                    off = kt * k_tile
                    T.copy(q[bid, hid, off:off + k_tile], q_ub)
                    T.copy(k[bid, hid, off:off + k_tile], k_ub)
                    T.copy(gk[bid, hid, off:off + k_tile], g_ub)
                    T.copy(state[bid, hid, off:off + k_tile, 0:DV], st)
                    T.vexp(g_ub, g_ub)
                    T.vbrc(v_ub, out)
                    T.vbrc(k_ub, dec)
                    T.vmul(out, dec, out)
                    T.vbrc(g_ub, dec)
                    T.vmul(st, dec, dec)
                    T.vadd(dec, out, st)
                    T.vbrc(q_ub, tmp)
                    T.vmul(st, tmp, tmp)
                    T.reduce(tmp, o2_t, dims=[0], reduce_mode="sum",
                             clear=True)
                    T.vadd(o2, o2_t, o2)
                    T.copy(st, new_state[bid, hid, off:off + k_tile, 0:DV])
                T.vmul(o2, scale, o2)
                T.copy(o2, o[bid, hid, 0:DV])

        return _gla_decode_kt64_main

    return _func


def _use_dvsplit_reuse(B, H, DK, DV, dtype):
    """R7 (2026-11-06): the smallest serving shape (B1 H8) is under-subscribed
    at 8 blocks; DV split=2 doubles the block count to 16 and k_tile=128 makes
    the state slab 32KB.  Measured 5.80us vs 6.24us for the k_tile=64 R6 path;
    B1 H32/H64 and B8 are NOT routed here (DV split loses there)."""
    return dtype != "float32" and DK == 128 and DV == 128 and B == 1 and H <= 8


def _gla_decode_kernel_dvsplit_reuse(B, H, DK, DV, dtype="float32", scale=None,
                       k_tile=None, dv_split=None):
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if scale is None:
        scale = DK ** -0.5
    scale = float(scale)
    if dv_split is None:
        dv_split = 2
    dv_split = int(dv_split)
    if DV % dv_split:
        raise ValueError(f"DV ({DV}) must be divisible by dv_split ({dv_split})")
    DVh = DV // dv_split
    if k_tile is None:
        k_tile = 128 if DVh <= 64 else 64
    k_tile = int(k_tile)
    if DK % k_tile:
        raise ValueError(f"DK ({DK}) must be divisible by k_tile ({k_tile})")
    acc = _ACCUM
    use_transit = dtype != acc
    total = B * H * dv_split
    DH = H * dv_split

    @tilelang.jit(out_idx=[5, 6], target="npuir")
    def _func(k_tile=k_tile, dv_split=dv_split, ncore=TUNED_NCORE):
        nc, n_iter = _plan_cores(total, ncore)

        if use_transit:
            @T.prim_func
            def _gla_decode_split_reuse_main(
                q: T.Tensor((B, H, DK), dtype),
                k: T.Tensor((B, H, DK), dtype),
                v: T.Tensor((B, H, DV), dtype),
                gk: T.Tensor((B, H, DK), dtype),
                state: T.Tensor((B, H, DK, DV), dtype),
                o: T.Tensor((B, H, DV), dtype),
                new_state: T.Tensor((B, H, DK, DV), dtype),
            ):
                with T.Kernel(nc, is_npu=True) as (cid, _):
                    v_ub = T.alloc_ub((1, DVh), acc)
                    o2 = T.alloc_ub((1, DVh), acc)
                    st = T.alloc_ub((k_tile, DVh), acc)
                    out = T.alloc_ub((k_tile, DVh), acc)
                    q_ub = T.alloc_ub((k_tile, 1), acc)
                    k_ub = T.alloc_ub((k_tile, 1), acc)
                    g_ub = T.alloc_ub((k_tile, 1), acc)
                    o2_t = T.alloc_ub((1, DVh), acc)
                    t_v = T.alloc_ub((1, DVh), dtype)
                    t_q = T.alloc_ub((k_tile, 1), dtype)
                    t_k = T.alloc_ub((k_tile, 1), dtype)
                    t_g = T.alloc_ub((k_tile, 1), dtype)
                    t_st = T.alloc_ub((k_tile, DVh), dtype)
                    t_o = T.alloc_ub((1, DVh), dtype)

                    for s in T.serial(n_iter):
                        blk = s * nc + cid
                        if blk < total:
                            bid = blk // DH
                            rem = blk % DH
                            hid = rem // dv_split
                            dvi = rem % dv_split
                            dv0 = dvi * DVh
                            T.copy(v[bid, hid, dv0:dv0 + DVh], t_v)
                            T.vcast(t_v, v_ub, round_mode="rint")
                            T.clear(o2)

                            for kt in T.serial(DK // k_tile):
                                off = kt * k_tile
                                T.copy(q[bid, hid, off:off + k_tile], t_q)
                                T.vcast(t_q, q_ub, round_mode="rint")
                                T.copy(k[bid, hid, off:off + k_tile], t_k)
                                T.vcast(t_k, k_ub, round_mode="rint")
                                T.copy(gk[bid, hid, off:off + k_tile], t_g)
                                T.vcast(t_g, g_ub, round_mode="rint")
                                T.copy(
                                    state[bid, hid, off:off + k_tile,
                                          dv0:dv0 + DVh],
                                    t_st,
                                )
                                T.vcast(t_st, st, round_mode="rint")
                                T.vexp(g_ub, g_ub)
                                T.vbrc(v_ub, out)
                                T.vmul(out, k_ub, out)        # out = v*k
                                T.vmul(st, g_ub, st)          # state *= exp(g)
                                T.vadd(st, out, st)           # new_state
                                T.vcast(st, t_st, round_mode="rint")
                                T.copy(
                                    t_st,
                                    new_state[bid, hid, off:off + k_tile,
                                              dv0:dv0 + DVh],
                                )
                                T.vmul(st, q_ub, st)          # q * new_state
                                T.reduce(st, o2_t, dims=[0],
                                         reduce_mode="sum", clear=True)
                                T.vadd(o2, o2_t, o2)
                            T.vmul(o2, scale, o2)
                            T.vcast(o2, t_o, round_mode="rint")
                            T.copy(t_o, o[bid, hid, dv0:dv0 + DVh])

            return _gla_decode_split_reuse_main

        @T.prim_func
        def _gla_decode_split_reuse_main(
            q: T.Tensor((B, H, DK), dtype),
            k: T.Tensor((B, H, DK), dtype),
            v: T.Tensor((B, H, DV), dtype),
            gk: T.Tensor((B, H, DK), dtype),
            state: T.Tensor((B, H, DK, DV), dtype),
            o: T.Tensor((B, H, DV), dtype),
            new_state: T.Tensor((B, H, DK, DV), dtype),
        ):
            with T.Kernel(total, is_npu=True) as (cid, _):
                bid = cid // DH
                rem = cid % DH
                hid = rem // dv_split
                dvi = rem % dv_split
                dv0 = dvi * DVh
                v_ub = T.alloc_ub((1, DVh), acc)
                o2 = T.alloc_ub((1, DVh), acc)
                st = T.alloc_ub((k_tile, DVh), acc)
                out = T.alloc_ub((k_tile, DVh), acc)
                dec = T.alloc_ub((k_tile, DVh), acc)
                tmp = T.alloc_ub((k_tile, DVh), acc)
                q_ub = T.alloc_ub((k_tile, 1), acc)
                k_ub = T.alloc_ub((k_tile, 1), acc)
                g_ub = T.alloc_ub((k_tile, 1), acc)
                o2_t = T.alloc_ub((1, DVh), acc)
                T.copy(v[bid, hid, dv0:dv0 + DVh], v_ub)
                T.clear(o2)
                for kt in T.serial(DK // k_tile):
                    off = kt * k_tile
                    T.copy(q[bid, hid, off:off + k_tile], q_ub)
                    T.copy(k[bid, hid, off:off + k_tile], k_ub)
                    T.copy(gk[bid, hid, off:off + k_tile], g_ub)
                    T.copy(state[bid, hid, off:off + k_tile, dv0:dv0 + DVh],
                           st)
                    T.vexp(g_ub, g_ub)
                    T.vbrc(v_ub, out)
                    T.vbrc(k_ub, dec)
                    T.vmul(out, dec, out)
                    T.vbrc(g_ub, dec)
                    T.vmul(st, dec, dec)
                    T.vadd(dec, out, st)
                    T.vbrc(q_ub, tmp)
                    T.vmul(st, tmp, tmp)
                    T.reduce(tmp, o2_t, dims=[0], reduce_mode="sum",
                             clear=True)
                    T.vadd(o2, o2_t, o2)
                    T.copy(st,
                           new_state[bid, hid, off:off + k_tile,
                                     dv0:dv0 + DVh])
                T.vmul(o2, scale, o2)
                T.copy(o2, o[bid, hid, dv0:dv0 + DVh])

        return _gla_decode_split_reuse_main

    return _func


def _use_2head(B, H, DK, DV, dtype):
    """R10 (2026-11-06): manual 2-head interleave at k_tile=32.  Re-tests the
    stale CASE-VEC-15 "<5%" claim under the R6 four-plane layout.
    Measured: B8 H64 60.85->54.58us (-10.3%), B1 H64 11.41->10.70us (-6.2%),
    B8 H32 34.50->32.87us (-4.7%); B1 H32 regresses, so it is excluded."""
    if dtype == "float32" or DK != 128 or DV != 128 or H % 2:
        return False
    if B == 1 and H == 64:
        return True
    if B >= 8 and H in (32, 64):
        return True
    return False


def _gla_decode_kernel_2head(B, H, DK, DV, dtype="float32", scale=None, k_tile=64):
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if H % 2:
        raise ValueError("2-head kernel requires even H")
    if scale is None:
        scale = DK ** -0.5
    scale = float(scale)
    k_tile = int(k_tile)
    if DK % k_tile:
        raise ValueError(f"DK ({DK}) must be divisible by k_tile ({k_tile})")
    acc = _ACCUM
    use_transit = dtype != acc
    total_pairs = (B * H) // 2
    HP = H // 2
    # B8H32: 128 pairs on 40 cores -> 3/4 pairs per core; 32 cores gives
    # exactly 4 pairs per core.  Paired A/B (3x): 32.90/33.17/32.85 us on 40
    # vs 32.11/31.81/32.06 us on 32 (-3.0%, 3/3 sign).  Other shapes keep 40.
    ncore_default = 32 if (B == 8 and H == 32) else TUNED_NCORE

    @tilelang.jit(out_idx=[5, 6], target="npuir",
                  pass_configs={"npuir.enable_auto_multi_buffer": False})
    def _func(k_tile=k_tile, ncore=ncore_default):
        nc, n_iter = _plan_cores(total_pairs, ncore)

        if use_transit:
            @T.prim_func
            def _gla_decode_2head_main(
                q: T.Tensor((B, H, DK), dtype),
                k: T.Tensor((B, H, DK), dtype),
                v: T.Tensor((B, H, DV), dtype),
                gk: T.Tensor((B, H, DK), dtype),
                state: T.Tensor((B, H, DK, DV), dtype),
                o: T.Tensor((B, H, DV), dtype),
                new_state: T.Tensor((B, H, DK, DV), dtype),
            ):
                with T.Kernel(nc, is_npu=True) as (cid, _):
                    v0 = T.alloc_ub((1, DV), acc); v1 = T.alloc_ub((1, DV), acc)
                    o20 = T.alloc_ub((1, DV), acc); o21 = T.alloc_ub((1, DV), acc)
                    st0 = T.alloc_ub((k_tile, DV), acc); st1 = T.alloc_ub((k_tile, DV), acc)
                    ou0 = T.alloc_ub((k_tile, DV), acc); ou1 = T.alloc_ub((k_tile, DV), acc)
                    q0 = T.alloc_ub((k_tile, 1), acc); q1 = T.alloc_ub((k_tile, 1), acc)
                    k0 = T.alloc_ub((k_tile, 1), acc); k1 = T.alloc_ub((k_tile, 1), acc)
                    g0 = T.alloc_ub((k_tile, 1), acc); g1 = T.alloc_ub((k_tile, 1), acc)
                    ot0 = T.alloc_ub((1, DV), acc); ot1 = T.alloc_ub((1, DV), acc)
                    tv0 = T.alloc_ub((1, DV), dtype); tv1 = T.alloc_ub((1, DV), dtype)
                    tq0 = T.alloc_ub((k_tile, 1), dtype); tq1 = T.alloc_ub((k_tile, 1), dtype)
                    tk0 = T.alloc_ub((k_tile, 1), dtype); tk1 = T.alloc_ub((k_tile, 1), dtype)
                    tg0 = T.alloc_ub((k_tile, 1), dtype); tg1 = T.alloc_ub((k_tile, 1), dtype)
                    ts0 = T.alloc_ub((k_tile, DV), dtype); ts1 = T.alloc_ub((k_tile, DV), dtype)
                    tw = T.alloc_ub((k_tile, DV), dtype)  # R17: writeback staging, decoupled from ts
                    to0 = T.alloc_ub((1, DV), dtype); to1 = T.alloc_ub((1, DV), dtype)

                    for s in T.serial(n_iter):
                        pair = s * nc + cid
                        if pair < total_pairs:
                            bid = pair // HP
                            hp = pair % HP
                            h0 = hp * 2
                            h1 = h0 + 1
                            T.copy(v[bid, h0, 0:DV], tv0)
                            T.vcast(tv0, v0, round_mode="rint")
                            T.copy(v[bid, h1, 0:DV], tv1)
                            T.vcast(tv1, v1, round_mode="rint")
                            T.clear(o20); T.clear(o21)

                            for kt in T.serial(DK // k_tile):
                                off = kt * k_tile
                                # both heads' MTE2 issued before either VEC chain
                                T.copy(q[bid, h0, off:off + k_tile], tq0)
                                T.copy(k[bid, h0, off:off + k_tile], tk0)
                                T.copy(gk[bid, h0, off:off + k_tile], tg0)
                                T.copy(state[bid, h0, off:off + k_tile, 0:DV], ts0)
                                T.copy(q[bid, h1, off:off + k_tile], tq1)
                                T.copy(k[bid, h1, off:off + k_tile], tk1)
                                T.copy(gk[bid, h1, off:off + k_tile], tg1)
                                T.copy(state[bid, h1, off:off + k_tile, 0:DV], ts1)
                                T.vcast(tq0, q0, round_mode="rint")
                                T.vcast(tk0, k0, round_mode="rint")
                                T.vcast(tg0, g0, round_mode="rint")
                                T.vcast(ts0, st0, round_mode="rint")
                                T.vcast(tq1, q1, round_mode="rint")
                                T.vcast(tk1, k1, round_mode="rint")
                                T.vcast(tg1, g1, round_mode="rint")
                                T.vcast(ts1, st1, round_mode="rint")

                                T.vexp(g0, g0)
                                # Row-broadcast vmul (no vbrc): [k,DV] * [k,1]
                                T.vbrc(v0, ou0)
                                T.vmul(ou0, k0, ou0)          # out = v * k
                                T.vmul(st0, g0, st0)          # state *= exp(g)
                                T.vadd(st0, ou0, st0)         # new_state
                                # write back before q destroys new_state
                                T.vcast(st0, tw, round_mode="rint")
                                T.copy(tw, new_state[bid, h0, off:off + k_tile, 0:DV])
                                T.vmul(st0, q0, st0)          # q * new_state
                                T.reduce(st0, ot0, dims=[0], reduce_mode="sum", clear=True)
                                T.vadd(o20, ot0, o20)

                                T.vexp(g1, g1)
                                T.vbrc(v1, ou1)
                                T.vmul(ou1, k1, ou1)
                                T.vmul(st1, g1, st1)
                                T.vadd(st1, ou1, st1)
                                T.vcast(st1, tw, round_mode="rint")
                                T.copy(tw, new_state[bid, h1, off:off + k_tile, 0:DV])
                                T.vmul(st1, q1, st1)
                                T.reduce(st1, ot1, dims=[0], reduce_mode="sum", clear=True)
                                T.vadd(o21, ot1, o21)

                            T.vmul(o20, scale, o20)
                            T.vcast(o20, to0, round_mode="rint")
                            T.copy(to0, o[bid, h0, 0:DV])
                            T.vmul(o21, scale, o21)
                            T.vcast(o21, to1, round_mode="rint")
                            T.copy(to1, o[bid, h1, 0:DV])

            return _gla_decode_2head_main

        raise NotImplementedError("2-head candidate only implements the transit dtype path")

    return _func


def _gla_decode_algo1_1head(B, H, DK, DV, dtype="float32", scale=None,
                           k_tile=128, state_dtype=None):
    """R41 algo1 v4: 1-head + fp16 + k_tile=128 + ping-pong state buffers.

    Plan A (state resident across the loop) does not apply: a core walks
    different heads every iteration (``blk = s * nc + cid``), so each round
    reads/writes a different ``state[b, h]`` slice, and 13 heads * 32 KiB
    would exceed the 192 KiB UB anyway.

    Plan B, minimal form: keep the *proven-correct* prefetch path (MTE2 lands
    the next head in ``ou``, then a ``memref.copy`` hands it to a state
    buffer) and only split the state buffer in two, unrolling the loop by two
    so the handoff of iteration i+1 targets the other buffer:

        i even : V(rw st0) -> MTE3(st0) ; MTE2(state -> ou) ; MTE2(ou -> st1)
        i odd  : V(rw st1) -> MTE3(st1) ; MTE2(state -> ou) ; MTE2(ou -> st0)

    ``ou -> st`` stays an ordinary MTE2 copy, so the backend keeps the same
    buffer-based dependency it already handled in the single-buffer form; the
    only change is that the WAR distance between "store reads stX" and "handoff
    writes stX" grows from 1 to 2 iterations.  Static UB (fp16): st0 32K +
    st1 32K + ou 32K + ts 32K + small ~4K = ~132 KiB < 192 KiB.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if scale is None:
        scale = DK ** -0.5
    scale = float(scale)
    k_tile = int(k_tile)
    if DK != k_tile:
        raise ValueError("1-head algo1 requires DK == k_tile (128)")
    acc = "float16"
    sdt = state_dtype or dtype
    # P0-0h: fp16 state skips both full-plane casts (fp16 == compute dtype).
    skip_state_cast = sdt == acc
    total = B * H
    # R47: see the prefetch branch in ``_step`` -- small totals keep the R45
    # landing copy, everything else uses R46's direct prefetch.
    use_landing_copy = total <= 32

    @tilelang.jit(out_idx=[5, 6], target="npuir",
                  pass_configs={"npuir.enable_auto_multi_buffer": False})
    def _func(k_tile=k_tile, ncore=TUNED_NCORE):
        nc, n_iter = _plan_cores(total, ncore)
        n_pairs = n_iter // 2
        n_tail = n_iter - 2 * n_pairs

        @T.macro
        def _step(q, k, v, gk, state, new_state, o,
                  tv, tq, tk, tg, v0, q0, k0, g0, qg, qkt, qk, o2, vtmp, to,
                  ou, ts, stb, stn, blk, nblk):
            bid = blk // H
            hid = blk % H
            if not skip_state_cast:
                T.copy(state[bid, hid, 0:DK, 0:DV], ts)
                T.vcast(ts, stb, round_mode="rint")
            T.copy(v[bid, hid, 0:DV], tv)
            T.copy(q[bid, hid, 0:DK], tq)
            T.copy(k[bid, hid, 0:DK], tk)
            T.copy(gk[bid, hid, 0:DK], tg)
            T.vcast(tv, v0, round_mode="rint")
            T.vcast(tq, q0, round_mode="rint")
            T.vcast(tk, k0, round_mode="rint")
            T.vcast(tg, g0, round_mode="rint")

            T.vexp(g0, g0)
            T.vmul(q0, g0, qg)
            T.vmul(stb, qg, ou)
            T.reduce(ou, o2, dims=[0], reduce_mode="sum", clear=True)
            T.vmul(q0, k0, qkt)
            T.reduce(qkt, qk, dims=[0], reduce_mode="sum", clear=True)
            T.vmul(v0, qk, vtmp)
            T.vadd(o2, vtmp, o2)
            # P0-0g: outer product as a single broadcast vmul.
            T.vmul(k0, v0, ou)
            T.vmul(stb, g0, stb)
            T.vadd(stb, ou, stb)
            if skip_state_cast:
                # R47: shape-dependent prefetch form.  R46 (direct into `stn`)
                # wins on the big serving shapes but loses ~0.4 us on the
                # launch-bound ones (b1-h8 3.42->3.50, b1-h32 4.78->5.24): with
                # few heads per core the extra plane pressure buys nothing and
                # the 3-in-flight form pays the deeper WAR.  Keep R46 for
                # total > 32 (>= 2 heads per core worthwhile) and R45's `ou`
                # landing copy below that.
                if use_landing_copy:
                    if nblk < total:
                        nb = nblk // H
                        nh = nblk % H
                        T.copy(state[nb, nh, 0:DK, 0:DV], ou)
                    T.copy(stb, new_state[bid, hid, 0:DK, 0:DV])
                    if nblk < total:
                        T.copy(ou, stn)
                else:
                    # R46: prefetch straight into the *other* ping-pong plane.
                    if nblk < total:
                        nb = nblk // H
                        nh = nblk % H
                        T.copy(state[nb, nh, 0:DK, 0:DV], stn)
                    T.copy(stb, new_state[bid, hid, 0:DK, 0:DV])
            else:
                T.vcast(stb, ts, round_mode="rint")
                T.copy(ts, new_state[bid, hid, 0:DK, 0:DV])

            T.vmul(o2, scale, o2)
            T.vcast(o2, to, round_mode="rint")
            T.copy(to, o[bid, hid, 0:DV])

        @T.prim_func
        def _gla_decode_algo1_1h_main(
            q: T.Tensor((B, H, DK), dtype),
            k: T.Tensor((B, H, DK), dtype),
            v: T.Tensor((B, H, DV), dtype),
            gk: T.Tensor((B, H, DK), dtype),
            state: T.Tensor((B, H, DK, DV), sdt),
            o: T.Tensor((B, H, DV), dtype),
            new_state: T.Tensor((B, H, DK, DV), sdt),
        ):
            with T.Kernel(nc, is_npu=True) as (cid, _):
                st0 = T.alloc_ub((k_tile, DV), acc)
                st1 = T.alloc_ub((k_tile, DV), acc)
                ou = T.alloc_ub((k_tile, DV), acc)
                # R46: `ts` is only referenced on the cast path.  On the fp16
                # state contract it is dead, but a full 32 KiB plane of the
                # 192 KiB UB budget; shrink it to one element so the allocator
                # cannot keep a zombie plane (frees the room used below).
                ts = T.alloc_ub((k_tile, DV) if not skip_state_cast else (1, 1), sdt)
                q0 = T.alloc_ub((k_tile, 1), acc)
                k0 = T.alloc_ub((k_tile, 1), acc)
                g0 = T.alloc_ub((k_tile, 1), acc)
                qg = T.alloc_ub((k_tile, 1), acc)
                qkt = T.alloc_ub((k_tile, 1), acc)
                qk = T.alloc_ub((1, 1), acc)
                v0 = T.alloc_ub((1, DV), acc)
                o2 = T.alloc_ub((1, DV), acc)
                vtmp = T.alloc_ub((1, DV), acc)
                tq = T.alloc_ub((k_tile, 1), dtype)
                tk = T.alloc_ub((k_tile, 1), dtype)
                tg = T.alloc_ub((k_tile, 1), dtype)
                tv = T.alloc_ub((1, DV), dtype)
                to = T.alloc_ub((1, DV), dtype)

                if skip_state_cast and cid < total:
                    T.copy(state[cid // H, cid % H, 0:DK, 0:DV], st0)

                for p in T.serial(n_pairs):
                    b0 = (2 * p) * nc + cid
                    b1 = (2 * p + 1) * nc + cid
                    if b0 < total:
                        _step(q, k, v, gk, state, new_state, o,
                              tv, tq, tk, tg, v0, q0, k0, g0, qg, qkt, qk,
                              o2, vtmp, to, ou, ts, st0, st1, b0, b1)
                    if b1 < total:
                        _step(q, k, v, gk, state, new_state, o,
                              tv, tq, tk, tg, v0, q0, k0, g0, qg, qkt, qk,
                              o2, vtmp, to, ou, ts, st1, st0, b1, b1 + nc)

                if n_tail:
                    bT = (2 * n_pairs) * nc + cid
                    if bT < total:
                        _step(q, k, v, gk, state, new_state, o,
                              tv, tq, tk, tg, v0, q0, k0, g0, qg, qkt, qk,
                              o2, vtmp, to, ou, ts, st0, st1, bT, bT + nc)

        return _gla_decode_algo1_1h_main

    def _entry(k_tile=None, ncore=None):
        # The Stage-5 wrapper passes its config ``k_tile`` positionally; for the
        # 1-head algo1 body DK is the only K tile (128), so that argument is
        # ignored.  ncore stays the tuning knob (default TUNED_NCORE=40).
        return _func(ncore=ncore) if ncore is not None else _func()

    return _entry


def _gla_decode_kernel_2head_halfwb(B, H, DK, DV, dtype="float32", scale=None, k_tile=64):
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if H % 2:
        raise ValueError("2-head kernel requires even H")
    if scale is None:
        scale = DK ** -0.5
    scale = float(scale)
    k_tile = int(k_tile)
    if DK % k_tile:
        raise ValueError(f"DK ({DK}) must be divisible by k_tile ({k_tile})")
    acc = _ACCUM
    use_transit = dtype != acc
    total_pairs = (B * H) // 2
    HP = H // 2
    # B8H32: 128 pairs on 40 cores -> 3/4 pairs per core; 32 cores gives
    # exactly 4 pairs per core.  Paired A/B (3x): 32.90/33.17/32.85 us on 40
    # vs 32.11/31.81/32.06 us on 32 (-3.0%, 3/3 sign).  Other shapes keep 40.
    ncore_default = 32 if (B == 8 and H == 32) else TUNED_NCORE

    @tilelang.jit(out_idx=[5, 6], target="npuir",
                  pass_configs={"npuir.enable_auto_multi_buffer": False})
    def _func(k_tile=k_tile, ncore=ncore_default):
        nc, n_iter = _plan_cores(total_pairs, ncore)

        if use_transit:
            @T.prim_func
            def _gla_decode_2head_main(
                q: T.Tensor((B, H, DK), dtype),
                k: T.Tensor((B, H, DK), dtype),
                v: T.Tensor((B, H, DV), dtype),
                gk: T.Tensor((B, H, DK), dtype),
                state: T.Tensor((B, H, DK, DV), dtype),
                o: T.Tensor((B, H, DV), dtype),
                new_state: T.Tensor((B, H, DK, DV), dtype),
            ):
                with T.Kernel(nc, is_npu=True) as (cid, _):
                    v0 = T.alloc_ub((1, DV), acc); v1 = T.alloc_ub((1, DV), acc)
                    o20 = T.alloc_ub((1, DV), acc); o21 = T.alloc_ub((1, DV), acc)
                    st0 = T.alloc_ub((k_tile, DV), acc); st1 = T.alloc_ub((k_tile, DV), acc)
                    ou0 = T.alloc_ub((k_tile, DV), acc); ou1 = T.alloc_ub((k_tile, DV), acc)
                    q0 = T.alloc_ub((k_tile, 1), acc); q1 = T.alloc_ub((k_tile, 1), acc)
                    k0 = T.alloc_ub((k_tile, 1), acc); k1 = T.alloc_ub((k_tile, 1), acc)
                    g0 = T.alloc_ub((k_tile, 1), acc); g1 = T.alloc_ub((k_tile, 1), acc)
                    ot0 = T.alloc_ub((1, DV), acc); ot1 = T.alloc_ub((1, DV), acc)
                    tv0 = T.alloc_ub((1, DV), dtype); tv1 = T.alloc_ub((1, DV), dtype)
                    tq0 = T.alloc_ub((k_tile, 1), dtype); tq1 = T.alloc_ub((k_tile, 1), dtype)
                    tk0 = T.alloc_ub((k_tile, 1), dtype); tk1 = T.alloc_ub((k_tile, 1), dtype)
                    tg0 = T.alloc_ub((k_tile, 1), dtype); tg1 = T.alloc_ub((k_tile, 1), dtype)
                    ts0 = T.alloc_ub((k_tile, DV), dtype); ts1 = T.alloc_ub((k_tile, DV), dtype)
                    tw0 = T.alloc_ub((k_tile // 2, DV), dtype)  # half-K writeback staging
                    tw1 = T.alloc_ub((k_tile // 2, DV), dtype)
                    to0 = T.alloc_ub((1, DV), dtype); to1 = T.alloc_ub((1, DV), dtype)

                    for s in T.serial(n_iter):
                        pair = s * nc + cid
                        if pair < total_pairs:
                            bid = pair // HP
                            hp = pair % HP
                            h0 = hp * 2
                            h1 = h0 + 1
                            T.copy(v[bid, h0, 0:DV], tv0)
                            T.vcast(tv0, v0, round_mode="rint")
                            T.copy(v[bid, h1, 0:DV], tv1)
                            T.vcast(tv1, v1, round_mode="rint")
                            T.clear(o20); T.clear(o21)

                            for kt in T.serial(DK // k_tile):
                                off = kt * k_tile
                                # both heads' MTE2 issued before either VEC chain
                                T.copy(q[bid, h0, off:off + k_tile], tq0)
                                T.copy(k[bid, h0, off:off + k_tile], tk0)
                                T.copy(gk[bid, h0, off:off + k_tile], tg0)
                                T.copy(state[bid, h0, off:off + k_tile, 0:DV], ts0)
                                T.copy(q[bid, h1, off:off + k_tile], tq1)
                                T.copy(k[bid, h1, off:off + k_tile], tk1)
                                T.copy(gk[bid, h1, off:off + k_tile], tg1)
                                T.copy(state[bid, h1, off:off + k_tile, 0:DV], ts1)
                                T.vcast(tq0, q0, round_mode="rint")
                                T.vcast(tk0, k0, round_mode="rint")
                                T.vcast(tg0, g0, round_mode="rint")
                                T.vcast(ts0, st0, round_mode="rint")
                                T.vcast(tq1, q1, round_mode="rint")
                                T.vcast(tk1, k1, round_mode="rint")
                                T.vcast(tg1, g1, round_mode="rint")
                                T.vcast(ts1, st1, round_mode="rint")

                                T.vexp(g0, g0)
                                # Row-broadcast vmul (no vbrc): [k,DV] * [k,1]
                                T.vbrc(v0, ou0)
                                T.vmul(ou0, k0, ou0)          # out = v * k
                                T.vmul(st0, g0, st0)          # state *= exp(g)
                                T.vadd(st0, ou0, st0)         # new_state
                                # write back before q destroys new_state
                                T.copy(st0[0:k_tile // 2, 0:DV], tw0)
                                T.copy(tw0, new_state[bid, h0, off:off + k_tile // 2, 0:DV])
                                T.copy(st0[k_tile // 2:k_tile, 0:DV], tw1)
                                T.copy(tw1, new_state[bid, h0, off + k_tile // 2:off + k_tile, 0:DV])
                                T.vmul(st0, q0, st0)          # q * new_state
                                T.reduce(st0, ot0, dims=[0], reduce_mode="sum", clear=True)
                                T.vadd(o20, ot0, o20)

                                T.vexp(g1, g1)
                                T.vbrc(v1, ou1)
                                T.vmul(ou1, k1, ou1)
                                T.vmul(st1, g1, st1)
                                T.vadd(st1, ou1, st1)
                                T.copy(st1[0:k_tile // 2, 0:DV], tw0)
                                T.copy(tw0, new_state[bid, h1, off:off + k_tile // 2, 0:DV])
                                T.copy(st1[k_tile // 2:k_tile, 0:DV], tw1)
                                T.copy(tw1, new_state[bid, h1, off + k_tile // 2:off + k_tile, 0:DV])
                                T.vmul(st1, q1, st1)
                                T.reduce(st1, ot1, dims=[0], reduce_mode="sum", clear=True)
                                T.vadd(o21, ot1, o21)

                            T.vmul(o20, scale, o20)
                            T.vcast(o20, to0, round_mode="rint")
                            T.copy(to0, o[bid, h0, 0:DV])
                            T.vmul(o21, scale, o21)
                            T.vcast(o21, to1, round_mode="rint")
                            T.copy(to1, o[bid, h1, 0:DV])

            return _gla_decode_2head_main

        raise NotImplementedError("2-head candidate only implements the transit dtype path")

    return _func


def _gla_decode_kernel(B, H, DK, DV, dtype="float32", scale=None, k_tile=None,
                       state_dtype=None):
    """NPU factory mirroring the GPU ``GLADecodeKernel`` builder."""
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if state_dtype is None:
        state_dtype = dtype
    if scale is None:
        scale = DK ** -0.5
    scale = float(scale)
    # R40: algo1 v3 (1-head + fp16 + k_tile=128) for bf16 128x128 serving shapes.
    if (k_tile is None and USE_ALGO1 and dtype == "bfloat16"
            and DK == 128 and DV == 128):
        return _gla_decode_algo1_1head(
            B, H, DK, DV, dtype=dtype, scale=scale, state_dtype=state_dtype)
    if state_dtype != dtype:
        raise ValueError(
            f"state_dtype {state_dtype!r} != dtype {dtype!r} is only supported "
            "on the bf16 128x128 algo1 route"
        )
    # R7: B1 H8 uses the DV-split reuse kernel (under-subscribed shape).
    if k_tile is None and _use_dvsplit_reuse(B, H, DK, DV, dtype):
        return _gla_decode_kernel_dvsplit_reuse(
            B, H, DK, DV, dtype=dtype, scale=scale, dv_split=2)
    # R10/R32: 2-head interleave.  H=64 defaults to full-tile writeback
    # (2026-11-12 A/B: 3/3 wins over halfwb on both b8-h64 and b1-h64).
    # Set USE_HALFWB_H64=True to reproduce the half-K staging variant.
    if k_tile is None and _use_2head(B, H, DK, DV, dtype):
        if H == 64 and USE_HALFWB_H64:
            return _gla_decode_kernel_2head_halfwb(
                B, H, DK, DV, dtype=dtype, scale=scale, k_tile=64)
        return _gla_decode_kernel_2head(
            B, H, DK, DV, dtype=dtype, scale=scale, k_tile=64)
    # R6: 128x128 bf16/fp16 serving shapes use the k_tile=64 reuse kernel.
    # Explicit k_tile keeps baseline / A-B callers on the original body.
    if k_tile is None and _use_kt64_reuse(DK, DV, dtype):
        return _gla_decode_kernel_kt64reuse(
            B, H, DK, DV, dtype=dtype, scale=scale, k_tile=128)
    if k_tile is None:
        k_tile = _ceil_ktile_fit(DK, DV)
    if DK % k_tile:
        raise ValueError(f"DK ({DK}) must be divisible by k_tile ({k_tile})")
    acc = _ACCUM
    use_transit = dtype != acc
    total = B * H

    @tilelang.jit(out_idx=[5, 6], target="npuir")
    def _func(k_tile, ncore=TUNED_NCORE):
        nc, n_iter = _plan_cores(total, ncore)
        # Persistent is scoped to the transit (sub-fp32) path on purpose: the
        # fp32 direct-copy path corrupted results when wrapped in the
        # grid-stride loop + bounds guard (measured max diff 5.1 / 43.6 on
        # B8-H32-D128 fp32), while copy->vcast->compute is correct.  All
        # serving workloads are bf16; fp32 keeps the validated flat body.
        persistent = bool(use_transit) and n_iter > 1
        if use_transit:

            if not persistent:
                @T.prim_func
                def _gla_decode_transit_main(
                    q: T.Tensor((B, H, DK), dtype),
                    k: T.Tensor((B, H, DK), dtype),
                    v: T.Tensor((B, H, DV), dtype),
                    gk: T.Tensor((B, H, DK), dtype),
                    state: T.Tensor((B, H, DK, DV), dtype),
                    o: T.Tensor((B, H, DV), dtype),
                    new_state: T.Tensor((B, H, DK, DV), dtype),
                ):
                    with T.Kernel(B * H, is_npu=True) as (cid, _):
                        bid = cid // H
                        hid = cid % H
                        v_ub = T.alloc_ub((1, DV), acc)
                        o2 = T.alloc_ub((1, DV), acc)
                        vb = T.alloc_ub((k_tile, DV), acc)
                        kb = T.alloc_ub((k_tile, DV), acc)
                        gb = T.alloc_ub((k_tile, DV), acc)
                        qb = T.alloc_ub((k_tile, DV), acc)
                        st_ub = T.alloc_ub((k_tile, DV), acc)
                        ns_ub = T.alloc_ub((k_tile, DV), acc)
                        dec_ub = T.alloc_ub((k_tile, DV), acc)
                        out_ub = T.alloc_ub((k_tile, DV), acc)
                        prod = T.alloc_ub((k_tile, DV), acc)
                        q_ub = T.alloc_ub((k_tile, 1), acc)
                        k_ub = T.alloc_ub((k_tile, 1), acc)
                        g_ub = T.alloc_ub((k_tile, 1), acc)
                        o2_t = T.alloc_ub((1, DV), acc)
                        t_v = T.alloc_ub((1, DV), dtype)
                        t_q = T.alloc_ub((k_tile, 1), dtype)
                        t_k = T.alloc_ub((k_tile, 1), dtype)
                        t_g = T.alloc_ub((k_tile, 1), dtype)
                        t_st = T.alloc_ub((k_tile, DV), dtype)
                        t_ns = T.alloc_ub((k_tile, DV), dtype)
                        t_o = T.alloc_ub((1, DV), dtype)

                        T.copy(v[bid, hid, 0:DV], t_v)
                        T.vcast(t_v, v_ub, round_mode="rint")
                        T.clear(o2)

                        for kt in T.serial(DK // k_tile):
                            off = kt * k_tile
                            T.copy(q[bid, hid, off:off + k_tile], t_q)
                            T.vcast(t_q, q_ub, round_mode="rint")
                            T.copy(k[bid, hid, off:off + k_tile], t_k)
                            T.vcast(t_k, k_ub, round_mode="rint")
                            T.copy(gk[bid, hid, off:off + k_tile], t_g)
                            T.vcast(t_g, g_ub, round_mode="rint")
                            T.copy(state[bid, hid, off:off + k_tile, 0:DV], t_st)
                            T.vcast(t_st, st_ub, round_mode="rint")

                            T.vexp(g_ub, g_ub)
                            T.vbrc(v_ub, vb)
                            T.vbrc(k_ub, kb)
                            T.vmul(vb, kb, out_ub)
                            T.vbrc(g_ub, gb)
                            T.vmul(st_ub, gb, dec_ub)
                            T.vadd(dec_ub, out_ub, ns_ub)
                            T.vbrc(q_ub, qb)
                            T.vmul(ns_ub, qb, prod)
                            T.reduce(prod, o2_t, dims=[0], reduce_mode="sum", clear=True)
                            T.vadd(o2, o2_t, o2)

                            T.vcast(ns_ub, t_ns, round_mode="rint")
                            T.copy(t_ns, new_state[bid, hid, off:off + k_tile, 0:DV])

                        T.vmul(o2, scale, o2)
                        T.vcast(o2, t_o, round_mode="rint")
                        T.copy(t_o, o[bid, hid, 0:DV])

                return _gla_decode_transit_main

            @T.prim_func
            def _gla_decode_transit_main(
                q: T.Tensor((B, H, DK), dtype),
                k: T.Tensor((B, H, DK), dtype),
                v: T.Tensor((B, H, DV), dtype),
                gk: T.Tensor((B, H, DK), dtype),
                state: T.Tensor((B, H, DK, DV), dtype),
                o: T.Tensor((B, H, DV), dtype),
                new_state: T.Tensor((B, H, DK, DV), dtype),
            ):
                with T.Kernel(nc, is_npu=True) as (cid, _):
                    v_ub = T.alloc_ub((1, DV), acc)
                    o2 = T.alloc_ub((1, DV), acc)
                    vb = T.alloc_ub((k_tile, DV), acc)
                    kb = T.alloc_ub((k_tile, DV), acc)
                    gb = T.alloc_ub((k_tile, DV), acc)
                    qb = T.alloc_ub((k_tile, DV), acc)
                    st_ub = T.alloc_ub((k_tile, DV), acc)
                    ns_ub = T.alloc_ub((k_tile, DV), acc)
                    dec_ub = T.alloc_ub((k_tile, DV), acc)
                    out_ub = T.alloc_ub((k_tile, DV), acc)
                    prod = T.alloc_ub((k_tile, DV), acc)
                    q_ub = T.alloc_ub((k_tile, 1), acc)
                    k_ub = T.alloc_ub((k_tile, 1), acc)
                    g_ub = T.alloc_ub((k_tile, 1), acc)
                    o2_t = T.alloc_ub((1, DV), acc)
                    t_v = T.alloc_ub((1, DV), dtype)
                    t_q = T.alloc_ub((k_tile, 1), dtype)
                    t_k = T.alloc_ub((k_tile, 1), dtype)
                    t_g = T.alloc_ub((k_tile, 1), dtype)
                    t_st = T.alloc_ub((k_tile, DV), dtype)
                    t_ns = T.alloc_ub((k_tile, DV), dtype)
                    t_o = T.alloc_ub((1, DV), dtype)

                    for s in T.serial(n_iter):
                        blk = s * nc + cid
                        if blk < total:
                            bid = blk // H
                            hid = blk % H
                            T.copy(v[bid, hid, 0:DV], t_v)
                            T.vcast(t_v, v_ub, round_mode="rint")
                            T.clear(o2)

                            for kt in T.serial(DK // k_tile):
                                off = kt * k_tile
                                T.copy(q[bid, hid, off:off + k_tile], t_q)
                                T.vcast(t_q, q_ub, round_mode="rint")
                                T.copy(k[bid, hid, off:off + k_tile], t_k)
                                T.vcast(t_k, k_ub, round_mode="rint")
                                T.copy(gk[bid, hid, off:off + k_tile], t_g)
                                T.vcast(t_g, g_ub, round_mode="rint")
                                T.copy(state[bid, hid, off:off + k_tile, 0:DV], t_st)
                                T.vcast(t_st, st_ub, round_mode="rint")

                                T.vexp(g_ub, g_ub)
                                T.vbrc(v_ub, vb)
                                T.vbrc(k_ub, kb)
                                T.vmul(vb, kb, out_ub)
                                T.vbrc(g_ub, gb)
                                T.vmul(st_ub, gb, dec_ub)
                                T.vadd(dec_ub, out_ub, ns_ub)
                                T.vbrc(q_ub, qb)
                                T.vmul(ns_ub, qb, prod)
                                T.reduce(prod, o2_t, dims=[0], reduce_mode="sum", clear=True)
                                T.vadd(o2, o2_t, o2)

                                T.vcast(ns_ub, t_ns, round_mode="rint")
                                T.copy(t_ns, new_state[bid, hid, off:off + k_tile, 0:DV])

                            T.vmul(o2, scale, o2)
                            T.vcast(o2, t_o, round_mode="rint")
                            T.copy(t_o, o[bid, hid, 0:DV])

            return _gla_decode_transit_main

        @T.prim_func
        def _gla_decode_main(
            q: T.Tensor((B, H, DK), dtype),
            k: T.Tensor((B, H, DK), dtype),
            v: T.Tensor((B, H, DV), dtype),
            gk: T.Tensor((B, H, DK), dtype),
            state: T.Tensor((B, H, DK, DV), dtype),
            o: T.Tensor((B, H, DV), dtype),
            new_state: T.Tensor((B, H, DK, DV), dtype),
        ):
            with T.Kernel(B * H, is_npu=True) as (cid, _):
                bid = cid // H
                hid = cid % H
                v_ub = T.alloc_ub((1, DV), acc)
                o2 = T.alloc_ub((1, DV), acc)
                vb = T.alloc_ub((k_tile, DV), acc)
                kb = T.alloc_ub((k_tile, DV), acc)
                gb = T.alloc_ub((k_tile, DV), acc)
                qb = T.alloc_ub((k_tile, DV), acc)
                st_ub = T.alloc_ub((k_tile, DV), acc)
                ns_ub = T.alloc_ub((k_tile, DV), acc)
                dec_ub = T.alloc_ub((k_tile, DV), acc)
                out_ub = T.alloc_ub((k_tile, DV), acc)
                prod = T.alloc_ub((k_tile, DV), acc)
                q_ub = T.alloc_ub((k_tile, 1), acc)
                k_ub = T.alloc_ub((k_tile, 1), acc)
                g_ub = T.alloc_ub((k_tile, 1), acc)
                o2_t = T.alloc_ub((1, DV), acc)

                T.copy(v[bid, hid, 0:DV], v_ub)
                T.clear(o2)

                for kt in T.serial(DK // k_tile):
                    off = kt * k_tile
                    T.copy(q[bid, hid, off:off + k_tile], q_ub)
                    T.copy(k[bid, hid, off:off + k_tile], k_ub)
                    T.copy(gk[bid, hid, off:off + k_tile], g_ub)
                    T.copy(state[bid, hid, off:off + k_tile, 0:DV], st_ub)

                    T.vexp(g_ub, g_ub)
                    T.vbrc(v_ub, vb)
                    T.vbrc(k_ub, kb)
                    T.vmul(vb, kb, out_ub)
                    T.vbrc(g_ub, gb)
                    T.vmul(st_ub, gb, dec_ub)
                    T.vadd(dec_ub, out_ub, ns_ub)
                    T.vbrc(q_ub, qb)
                    T.vmul(ns_ub, qb, prod)
                    T.reduce(prod, o2_t, dims=[0], reduce_mode="sum", clear=True)
                    T.vadd(o2, o2_t, o2)

                    T.copy(ns_ub, new_state[bid, hid, off:off + k_tile, 0:DV])

                T.vmul(o2, scale, o2)
                T.copy(o2, o[bid, hid, 0:DV])

        return _gla_decode_main

        @T.prim_func
        def _gla_decode_main(
            q: T.Tensor((B, H, DK), dtype),
            k: T.Tensor((B, H, DK), dtype),
            v: T.Tensor((B, H, DV), dtype),
            gk: T.Tensor((B, H, DK), dtype),
            state: T.Tensor((B, H, DK, DV), dtype),
            o: T.Tensor((B, H, DV), dtype),
            new_state: T.Tensor((B, H, DK, DV), dtype),
        ):
            with T.Kernel(nc, is_npu=True) as (cid, _):
                v_ub = T.alloc_ub((1, DV), acc)
                o2 = T.alloc_ub((1, DV), acc)
                vb = T.alloc_ub((k_tile, DV), acc)
                kb = T.alloc_ub((k_tile, DV), acc)
                gb = T.alloc_ub((k_tile, DV), acc)
                qb = T.alloc_ub((k_tile, DV), acc)
                st_ub = T.alloc_ub((k_tile, DV), acc)
                ns_ub = T.alloc_ub((k_tile, DV), acc)
                dec_ub = T.alloc_ub((k_tile, DV), acc)
                out_ub = T.alloc_ub((k_tile, DV), acc)
                prod = T.alloc_ub((k_tile, DV), acc)
                q_ub = T.alloc_ub((k_tile, 1), acc)
                k_ub = T.alloc_ub((k_tile, 1), acc)
                g_ub = T.alloc_ub((k_tile, 1), acc)
                o2_t = T.alloc_ub((1, DV), acc)

                for s in T.serial(n_iter):
                    blk = s * nc + cid
                    if blk < total:
                        bid = blk // H
                        hid = blk % H
                        T.copy(v[bid, hid, 0:DV], v_ub)
                        T.clear(o2)

                        for kt in T.serial(DK // k_tile):
                            off = kt * k_tile
                            T.copy(q[bid, hid, off:off + k_tile], q_ub)
                            T.copy(k[bid, hid, off:off + k_tile], k_ub)
                            T.copy(gk[bid, hid, off:off + k_tile], g_ub)
                            T.copy(state[bid, hid, off:off + k_tile, 0:DV], st_ub)

                            T.vexp(g_ub, g_ub)
                            T.vbrc(v_ub, vb)
                            T.vbrc(k_ub, kb)
                            T.vmul(vb, kb, out_ub)
                            T.vbrc(g_ub, gb)
                            T.vmul(st_ub, gb, dec_ub)
                            T.vadd(dec_ub, out_ub, ns_ub)
                            T.vbrc(q_ub, qb)
                            T.vmul(ns_ub, qb, prod)
                            T.reduce(prod, o2_t, dims=[0], reduce_mode="sum", clear=True)
                            T.vadd(o2, o2_t, o2)

                            T.copy(ns_ub, new_state[bid, hid, off:off + k_tile, 0:DV])

                        T.vmul(o2, scale, o2)
                        T.copy(o2, o[bid, hid, 0:DV])

        return _gla_decode_main

    return _func


def golden_gla_decode(q, k, v, gk, state, scale):
    """Independent pure-torch CPU golden (fp32): ``(o, new_state)``."""
    qf, kf, vf, gf, sf = (t.float() for t in (q, k, v, gk, state))
    e = torch.exp(gf)
    outer = kf.unsqueeze(-1) * vf.unsqueeze(-2)
    new_state = sf * e.unsqueeze(-1) + outer
    o = scale * torch.einsum("bhk,bhkv->bhv", qf, new_state)
    return o, new_state


golden = golden_gla_decode


# ---------------------------------------------------------------------------
# Embedded layered correctness tests (kernel + golden + main, `--level all`).
# ---------------------------------------------------------------------------

import torch_npu  # noqa: E402,F401  (registers the "npu" device)

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

_CASE_SMOKE = [
    (2, 4, 64, 64, "float16", -1.0),
    (2, 4, 64, 64, "bfloat16", -1.0),
]
_CASE_SERVING = [
    (1, 8, 128, 128, "bfloat16", -1.0),
    (1, 32, 128, 128, "bfloat16", -1.0),
    (1, 64, 128, 128, "bfloat16", -1.0),
    (8, 32, 128, 128, "bfloat16", -1.0),
    (8, 64, 128, 128, "bfloat16", -1.0),
]
_CASE_FULL = _CASE_SMOKE + _CASE_SERVING + [
    (1, 2, 32, 16, "float32", -1.0),
    (2, 3, 16, 16, "float16", -1.0),
    (1, 4, 128, 64, "float16", -1.0),
    (2, 3, 64, 64, "bfloat16", 0.25),
]
_CASE_BOUNDARY = [
    (1, 2, 32, 16, "float32", -1.0),   # untiled f32
    (2, 3, 16, 16, "float16", -1.0),   # small DK (k_tile fallback 16)
    (1, 4, 128, 64, "float16", -1.0),  # rectangular DV != DK
    (2, 3, 64, 64, "bfloat16", 0.25),  # custom readout scale
]


def _run_case(B, H, DK, DV, dtype_name, scale):
    dtype = _DTYPE_MAP[dtype_name]
    resolved = scale if scale > 0 else DK ** -0.5
    dev = "npu"
    q = (torch.randn(B, H, DK, device=dev, dtype=dtype) * 0.5).contiguous()
    k = (torch.randn(B, H, DK, device=dev, dtype=dtype) * 0.5).contiguous()
    v = (torch.randn(B, H, DV, device=dev, dtype=dtype) * 0.5).contiguous()
    gk = (-torch.rand(B, H, DK, device=dev, dtype=dtype)).contiguous()
    state = torch.randn(B, H, DK, DV, device=dev, dtype=dtype).contiguous()

    program = _gla_decode_kernel(B, H, DK, DV, dtype_name, resolved)
    prog = program(_ceil_ktile_fit(DK, DV))
    o, ns = prog(q, k, v, gk, state)

    ref_o, ref_ns = golden_gla_decode(q, k, v, gk, state, resolved)
    tol = _TOLERANCE[dtype_name]
    torch.testing.assert_close(o.cpu().float(), ref_o.cpu().float(), atol=tol, rtol=tol)
    torch.testing.assert_close(ns.cpu().float(), ref_ns.cpu().float(), atol=tol, rtol=tol)
    print(f"[PASS] {dtype_name} B{B} H{H} DK{DK} DV{DV} scale={scale} tol={tol:.1e}")


def _run_cases(cases):
    for case in cases:
        _run_case(*case)


def run_L0():
    _run_cases(_CASE_SMOKE)


def run_L1():
    _run_cases(_CASE_SERVING)


def run_L2():
    _run_cases(_CASE_FULL)


def run_boundary():
    _run_cases(_CASE_BOUNDARY)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="GLADecodeOp layered correctness test")
    ap.add_argument("--level", default="all", choices=["all", "L0", "L1", "L2", "Boundary"])
    args = ap.parse_args()
    levels = {"L0": run_L0, "L1": run_L1, "L2": run_L2, "Boundary": run_boundary}
    if args.level == "all":
        for fn in (run_L0, run_L1, run_L2, run_boundary):
            fn()
    else:
        levels[args.level]()
    print(f"[{args.level}] ALL PASS")


if __name__ == "__main__":
    main()
