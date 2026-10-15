# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Feasibility probe: GLA decode, ``target="npuir"`` (Ascend910B3, CANN 8.5.2).

Stage-0 pre-migration FEASIBILITY PROBE. This is NOT the migrated operator;
it is a minimal but *representative* kernel that exercises the hard
capabilities a migration of ``GLADecodeOp`` (baseline name
``GLARecurrentFwdOp``) depends on:

  * per-key-dim elementwise gate  ``exp(gk)``                  (Vector / T.vexp)
  * per-head state decay          ``state * exp(gk)``          (Vector / T.vbrc+T.vmul)
  * outer-product accumulate      ``+ k (x) v`` over (DK, DV)  (Vector / T.vbrc+T.vmul+T.vadd)
  * output contraction            ``o = scale * q . state_new``(Vector / T.vmul+T.reduce sum over DK)

The state is materialised (fp32 accumulation), so ``o`` is read directly off
``new_state`` -- mathematically identical to the GPU kernel's
``scale*(q*exp(gk))^T S + scale*(q.k) v`` rewrite, which avoids materialising
the state.

GPU source semantics (read as DATA; independent of this NPU algorithm),
``TileOPs/src/tileops/kernels/linear_attention/gla/recurrent.py``:

    state_new[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
    o[dv]             = scale * sum_dk q[dk] * state_new[dk, dv]
    scale             = DK ** -0.5   (non-positive -> default)

Layout BHD: q,k,v,gk: ``[B,H,DK]``/``[B,H,DV]``, state: ``[B,H,DK,DV]``;
outputs o: ``[B,H,DV]``, new_state: ``[B,H,DK,DV]``. One NPU core per (b,h).

Run: python gla_decode_probe.py
Prints per-case ``max_abs_err`` against a pure-torch CPU golden and a final
PROBE PASS/FAIL.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import traceback

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

__all__ = ["gla_decode_probe", "golden_gla_decode"]

_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")
_TOLERANCE = {
    "float16": 2e-3,
    "bfloat16": 2e-2,
    "float32": 1e-4,
}
_ACCUM = "float32"


# ---------------------------------------------------------------------------
# Independent pure-torch CPU golden (does NOT mirror the NPU algorithm)
# ---------------------------------------------------------------------------
def golden_gla_decode(q, k, v, gk, state, scale):
    """Reference GLA single-step decode (fp32 opmath on CPU).

    Inputs q,k,gk: ``(B,H,DK)``; v: ``(B,H,DV)``; state: ``(B,H,DK,DV)``.
    Returns ``(o, new_state)`` in fp32: ``o`` is ``(B,H,DV)`` and the returned
    state is the full-precision fp32 recurrence result.
    """
    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = gk.float()
    sf = state.float()
    e = torch.exp(gf)                                     # (B,H,DK)
    outer = kf.unsqueeze(-1) * vf.unsqueeze(-2)           # (B,H,DK,DV)
    new_state = sf * e.unsqueeze(-1) + outer              # (B,H,DK,DV)
    o = scale * torch.einsum("bhk,bhkv->bhv", qf, new_state)
    return o, new_state


# ---------------------------------------------------------------------------
# Probe kernel
# ---------------------------------------------------------------------------
def gla_decode_probe(B, H, DK, DV, dtype="float32", scale=None):
    """Build the probe kernel factory for a fixed ``(B,H,DK,DV,dtype)``.

    Returns a JIT callable compiling
    ``main(q,k,v,gk,state)`` -> ``(o, new_state)`` (``out_idx=[5, 6]``).
    """
    if min(B, H, DK, DV) < 1:
        raise ValueError(f"B,H,DK,DV must be positive, got {B},{H},{DK},{DV}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")

    acc = _ACCUM
    if scale is None:
        scale = DK ** -0.5
    scale = float(scale)
    use_transit = dtype != acc

    @tilelang.jit(out_idx=[5, 6], target="npuir")
    def _func():
        if use_transit:

            @T.prim_func
            def main(
                q: T.Tensor((B, H, DK), dtype),
                k: T.Tensor((B, H, DK), dtype),
                v: T.Tensor((B, H, DV), dtype),
                gk: T.Tensor((B, H, DK), dtype),
                state: T.Tensor((B, H, DK, DV), dtype),
                o: T.Tensor((B, H, DV), dtype),            # out_idx=5
                new_state: T.Tensor((B, H, DK, DV), dtype),  # out_idx=6
            ):
                with T.Kernel(B * H, is_npu=True) as (cid, _):
                    bid = cid // H
                    hid = cid % H
                    # fp32 compute buffers (rank-2, [DK,1]/[1,DV] for broadcast)
                    q_ub = T.alloc_ub((DK, 1), acc)
                    k_ub = T.alloc_ub((DK, 1), acc)
                    g_ub = T.alloc_ub((DK, 1), acc)
                    v_ub = T.alloc_ub((1, DV), acc)
                    st_ub = T.alloc_ub((DK, DV), acc)
                    ns_ub = T.alloc_ub((DK, DV), acc)
                    dec_ub = T.alloc_ub((DK, DV), acc)
                    out_ub = T.alloc_ub((DK, DV), acc)
                    vb = T.alloc_ub((DK, DV), acc)
                    kb = T.alloc_ub((DK, DV), acc)
                    gb = T.alloc_ub((DK, DV), acc)
                    qb = T.alloc_ub((DK, DV), acc)
                    prod = T.alloc_ub((DK, DV), acc)
                    o2 = T.alloc_ub((1, DV), acc)
                    # fp16/bf16 staging buffers (bf16 has no Vector arithmetic)
                    t_q = T.alloc_ub((DK, 1), dtype)
                    t_k = T.alloc_ub((DK, 1), dtype)
                    t_g = T.alloc_ub((DK, 1), dtype)
                    t_v = T.alloc_ub((1, DV), dtype)
                    t_st = T.alloc_ub((DK, DV), dtype)
                    t_ns = T.alloc_ub((DK, DV), dtype)
                    t_o = T.alloc_ub((1, DV), dtype)

                    T.copy(q[bid, hid, 0:DK], t_q)
                    T.vcast(t_q, q_ub, round_mode="rint")
                    T.copy(k[bid, hid, 0:DK], t_k)
                    T.vcast(t_k, k_ub, round_mode="rint")
                    T.copy(gk[bid, hid, 0:DK], t_g)
                    T.vcast(t_g, g_ub, round_mode="rint")
                    T.copy(v[bid, hid, 0:DV], t_v)
                    T.vcast(t_v, v_ub, round_mode="rint")
                    T.copy(state[bid, hid, 0:DK, 0:DV], t_st)
                    T.vcast(t_st, st_ub, round_mode="rint")

                    # ---- gate: exp(gk) ------------------------------------
                    T.vexp(g_ub, g_ub)
                    # ---- outer product  k (x) v ---------------------------
                    T.vbrc(v_ub, vb)
                    T.vbrc(k_ub, kb)
                    T.vmul(vb, kb, out_ub)
                    # ---- decay  state * exp(gk) ---------------------------
                    T.vbrc(g_ub, gb)
                    T.vmul(st_ub, gb, dec_ub)
                    # ---- new_state = decay + outer ------------------------
                    T.vadd(dec_ub, out_ub, ns_ub)
                    # ---- o = scale * q . new_state (reduce over DK) -------
                    T.vbrc(q_ub, qb)
                    T.vmul(ns_ub, qb, prod)
                    T.reduce(prod, o2, dims=[0], reduce_mode="sum", clear=True)
                    T.vmul(o2, scale, o2)

                    T.vcast(ns_ub, t_ns, round_mode="rint")
                    T.copy(t_ns, new_state[bid, hid, 0:DK, 0:DV])
                    T.vcast(o2, t_o, round_mode="rint")
                    T.copy(t_o, o[bid, hid, 0:DV])

        else:

            @T.prim_func
            def main(
                q: T.Tensor((B, H, DK), dtype),
                k: T.Tensor((B, H, DK), dtype),
                v: T.Tensor((B, H, DV), dtype),
                gk: T.Tensor((B, H, DK), dtype),
                state: T.Tensor((B, H, DK, DV), dtype),
                o: T.Tensor((B, H, DV), dtype),            # out_idx=5
                new_state: T.Tensor((B, H, DK, DV), dtype),  # out_idx=6
            ):
                with T.Kernel(B * H, is_npu=True) as (cid, _):
                    bid = cid // H
                    hid = cid % H
                    q_ub = T.alloc_ub((DK, 1), acc)
                    k_ub = T.alloc_ub((DK, 1), acc)
                    g_ub = T.alloc_ub((DK, 1), acc)
                    v_ub = T.alloc_ub((1, DV), acc)
                    st_ub = T.alloc_ub((DK, DV), acc)
                    ns_ub = T.alloc_ub((DK, DV), acc)
                    dec_ub = T.alloc_ub((DK, DV), acc)
                    out_ub = T.alloc_ub((DK, DV), acc)
                    vb = T.alloc_ub((DK, DV), acc)
                    kb = T.alloc_ub((DK, DV), acc)
                    gb = T.alloc_ub((DK, DV), acc)
                    qb = T.alloc_ub((DK, DV), acc)
                    prod = T.alloc_ub((DK, DV), acc)
                    o2 = T.alloc_ub((1, DV), acc)

                    T.copy(q[bid, hid, 0:DK], q_ub)
                    T.copy(k[bid, hid, 0:DK], k_ub)
                    T.copy(gk[bid, hid, 0:DK], g_ub)
                    T.copy(v[bid, hid, 0:DV], v_ub)
                    T.copy(state[bid, hid, 0:DK, 0:DV], st_ub)

                    T.vexp(g_ub, g_ub)
                    T.vbrc(v_ub, vb)
                    T.vbrc(k_ub, kb)
                    T.vmul(vb, kb, out_ub)
                    T.vbrc(g_ub, gb)
                    T.vmul(st_ub, gb, dec_ub)
                    T.vadd(dec_ub, out_ub, ns_ub)
                    T.vbrc(q_ub, qb)
                    T.vmul(ns_ub, qb, prod)
                    T.reduce(prod, o2, dims=[0], reduce_mode="sum", clear=True)
                    T.vmul(o2, scale, o2)

                    T.copy(ns_ub, new_state[bid, hid, 0:DK, 0:DV])
                    T.copy(o2, o[bid, hid, 0:DV])

        return main

    return _func


def gla_decode_tiled_probe(B, H, DK, DV, dtype="float32", scale=None, k_tile=16):
    """K-tiled variant: the full state never lives in UB at once.

    Mirrors the GPU kernel's ``dim_k // k_tile`` pipelined loop. Each K-tile of
    the state is decayed, updated with the rank-1 ``k (x) v`` term and written
    back; the output contraction accumulates across K-tiles. This is the shape
    that fits the manifest D=128 (the untiled variant overflows UB there).
    """
    if min(B, H, DK, DV) < 1 or DK % k_tile != 0:
        raise ValueError(f"need DK % k_tile == 0, got DK={DK}, k_tile={k_tile}")

    acc = _ACCUM
    if scale is None:
        scale = DK ** -0.5
    scale = float(scale)
    use_transit = dtype != acc

    @tilelang.jit(out_idx=[5, 6], target="npuir")
    def _func():
        if use_transit:

            @T.prim_func
            def main(
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
                        T.reduce(prod, o2_t, dims=[0], reduce_mode="sum",
                                 clear=True)
                        T.vadd(o2, o2_t, o2)

                        T.vcast(ns_ub, t_ns, round_mode="rint")
                        T.copy(t_ns, new_state[bid, hid, off:off + k_tile, 0:DV])

                    T.vmul(o2, scale, o2)
                    T.vcast(o2, t_o, round_mode="rint")
                    T.copy(t_o, o[bid, hid, 0:DV])

        else:

            @T.prim_func
            def main(
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
                        T.reduce(prod, o2_t, dims=[0], reduce_mode="sum",
                                 clear=True)
                        T.vadd(o2, o2_t, o2)

                        T.copy(ns_ub, new_state[bid, hid, off:off + k_tile, 0:DV])

                    T.vmul(o2, scale, o2)
                    T.copy(o2, o[bid, hid, 0:DV])

        return main

    return _func


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_case(B, H, DK, DV, dtype_str, k_tile=None):
    torch_dtype = getattr(torch, dtype_str)
    torch.manual_seed(0)
    scale = DK ** -0.5
    q = (torch.randn(B, H, DK) * 0.5).to(torch_dtype)
    k = (torch.randn(B, H, DK) * 0.5).to(torch_dtype)
    v = (torch.randn(B, H, DV) * 0.5).to(torch_dtype)
    gk = (-torch.rand(B, H, DK)).to(torch_dtype)          # log-space decay <= 0
    state = torch.randn(B, H, DK, DV).to(torch_dtype)

    if k_tile:
        kernel = gla_decode_tiled_probe(B, H, DK, DV, dtype_str, scale, k_tile)()
    else:
        kernel = gla_decode_probe(B, H, DK, DV, dtype_str, scale)()
    o, ns = kernel(q.npu(), k.npu(), v.npu(), gk.npu(), state.npu())
    o, ns = o.cpu(), ns.cpu()

    ref_o, ref_ns = golden_gla_decode(q, k, v, gk, state, scale)

    err_o = (o.float() - ref_o).abs().max().item()
    err_ns = (ns.float() - ref_ns).abs().max().item()
    err = max(err_o, err_ns)
    tol = _TOLERANCE[dtype_str]

    print(
        f"[PROBE] (B={B},H={H},DK={DK},DV={DV}) dtype={dtype_str} "
        f"{'tiled(k=' + str(k_tile) + ')' if k_tile else 'untiled'} "
        f"max_abs_err o={err_o:.3e} new_state={err_ns:.3e} "
        f"(rtol=atol={tol:.1e})"
    )

    ok = True
    for name, out, ref in (("o", o, ref_o), ("new_state", ns, ref_ns)):
        try:
            torch.testing.assert_close(
                out.float(), ref, rtol=tol, atol=tol,
                msg=lambda m, name=name: f"{name}: {m}",
            )
        except AssertionError as exc:
            ok = False
            print(f"  {name} FAILED: {str(exc).splitlines()[0]}")
    return ok, err


# (B, H, DK, DV, dtype, k_tile) -- k_tile=None selects the untiled variant.
CASES = [
    (2, 3, 16, 16, "float32", None),
    (1, 2, 32, 16, "float32", None),
    (2, 3, 16, 16, "float16", None),
    (1, 2, 32, 16, "float16", None),
    (2, 4, 32, 32, "bfloat16", None),
    # scaling toward the manifest workloads (H up to 64, DK=DV=128)
    (2, 8, 64, 64, "float16", None),
    # manifest D=128 requires K-tiling to fit UB
    (1, 4, 128, 128, "float16", 16),
    (1, 4, 128, 128, "float16", 32),
    (8, 32, 128, 128, "bfloat16", 16),
]


def main():
    print("device:", torch.npu.get_device_name(0))
    all_ok = True
    worst = 0.0
    failures = []
    for B, H, DK, DV, dtype_str, k_tile in CASES:
        print(f"\n===== case (B={B},H={H},DK={DK},DV={DV}) {dtype_str} "
              f"k_tile={k_tile} =====", flush=True)
        try:
            ok, err = _run_case(B, H, DK, DV, dtype_str, k_tile)
            worst = max(worst, err)
            all_ok = all_ok and ok
            if not ok:
                failures.append((B, H, DK, DV, dtype_str, k_tile, err))
        except Exception as e:  # noqa: BLE001
            all_ok = False
            failures.append((B, H, DK, DV, dtype_str, k_tile, repr(e)))
            print(f"[PROBE] EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\nworst max_abs_err across cases: {worst:.3e}")
    if all_ok:
        print("PROBE PASS")
    else:
        print(f"PROBE FAIL: {failures}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
