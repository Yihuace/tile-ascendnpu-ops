# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""GLA single-token decode kernel for npuir (Vector, Developer mode).

Migrated from the GPU TileLang factory ``GLADecodeKernel``
(``TileOPs/src/tileops/kernels/linear_attention/gla/recurrent.py``) to
``target="npuir"``, per the verified feasibility probe
(``examples/gla_decode/probe/``).

Source semantics (kept verbatim):

    state_new[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
    o[dv]             = scale * sum_dk q[dk] * state_new[dk, dv]
    scale             = DK ** -0.5   (non-positive -> default)

Layout BHD: q,k,gk: ``[B,H,DK]``; v: ``[B,H,DV]``; state: ``[B,H,DK,DV]``;
outputs o: ``[B,H,DV]``, new_state: ``[B,H,DK,DV]``.  One NPU core per ``(b,h)``.

Structure (all Vector-unit work; no Cube):
  - the state is streamed in ``[k_tile, DV]`` K-tiles (the GPU kernel's
    ``dim_k // k_tile`` loop) so the working set fits UB at ``DK=DV=128``;
    each tile is decayed (``vexp`` gate + ``vbrc`` + ``vmul``), updated with the
    rank-1 ``k (x) v`` term (``vbrc`` + ``vmul`` + ``vadd``) and written back
    while ``o`` accumulates across tiles (``vmul`` + ``reduce`` sum over DK).
  - ``bf16`` has no Vector arithmetic, so fp16/bf16 inputs run in fp32 with
    ``vcast`` at the load/store boundary and fp32 accumulation (matching the
    GPU contract).

Run: python _gla_decode_kernel.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch

_ACCUM = "float32"
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")
_TOLERANCE = {"float16": 2e-3, "bfloat16": 2e-2, "float32": 1e-4}

# Stage-4 tuned default: k_tile=32 measured fastest on every serving workload
# (see opt_log.md Final Summary).  Exposed for the wrapper switch block.
TUNED_K_TILE = 32


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


def _gla_decode_kernel(B, H, DK, DV, dtype="float32", scale=None, k_tile=None):
    """NPU factory mirroring the GPU ``GLADecodeKernel`` builder.

    Returns an inner ``@tilelang.jit(target="npuir")`` factory taking ``k_tile``
    (the DK sub-tile width) as its only (compile-time) parameter.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if scale is None:
        scale = DK ** -0.5
    scale = float(scale)
    if k_tile is None:
        k_tile = _ceil_ktile_fit(DK, DV)
    if DK % k_tile:
        raise ValueError(f"DK ({DK}) must be divisible by k_tile ({k_tile})")
    acc = _ACCUM
    use_transit = dtype != acc

    @tilelang.jit(out_idx=[5, 6], target="npuir")
    def _func(k_tile):
        if use_transit:

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
# The reference is the pure-torch fp32 golden above (independent of the
# K-tiled NPU recurrence); the K-tiled kernel is compared per-case.
# ---------------------------------------------------------------------------

import torch_npu  # noqa: E402,F401  (registers the "npu" device)

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

# (B, H, DK, DV, dtype, scale) -- mirror tests/ops/test_gla_decode.py.
_CASE_SMOKE = [
    (2, 4, 64, 64, "float16", -1.0),
    (2, 4, 64, 64, "bfloat16", -1.0),
]
_CASE_SERVING = [
    (1, 8, 128, 128, "bfloat16", -1.0),
    (1, 32, 128, 128, "bfloat16", -1.0),
    (8, 32, 128, 128, "bfloat16", -1.0),
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
