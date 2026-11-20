#!/usr/bin/env python3
"""Bounded feasibility probe: FALN row-reduce via Cube ones-vector GEMV.

Question (FALN 8b-prefill headroom): can the per-row reduction be moved off
the Vector dependency chain onto the Cube with ``T.gemm`` of shape
``(block_m, N) x (N, 1)``?

Two self-contained micro-kernels over the *same* input bytes:

  A  ``vec_reduce_main``  Expert ``T.Scope("Vector")``: fp16 GM -> fp16 UB ->
     ``T.vcast`` -> fp32 UB -> ``T.reduce_sum(dim=1)`` -> (M,1) fp32 GM.
     Semantically identical to the reduction step of the delivered FALN
     kernel (``perf_opt/FusedAddLayerNormFwdOp.py``, block_m=2 / block_m=1).

  B  ``cube_gemv_main``   Expert ``T.Scope("Cube")``: ``x(M,N)`` -> L1 via
     ``T.load_nd2nz``, ``ones`` -> L1, ``T.gemm`` accumulating in L0C,
     ``T.store_fixpipe`` -> GM.  ``b_transpose`` selects whether the ones
     vector is staged as a contiguous row ``(1,N)`` (bt=True) or as a
     column ``(N,1)`` (bt=False).

Modes
  compile  build every variant, print success / exception text only
  check    run on NPU once per shape, compare against a float64 torch ref
  bench    alternate A/B launches inside ONE process (msprof op driver)

Run (NPU-touching modes go through the workspace lock):
  .agents/tools/npu_lock.sh --wait 900 --wait-idle 60 --timeout 3600 -- \
      python perf_opt/probe_cube_gemv/probe_cube_gemv.py --mode check
"""

import argparse
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

DT = "float16"
ACC = "float32"

# Probe shapes: M is the FALN block_m, N the row length.
SHAPES = [(2, 4096), (1, 8192)]


# ---------------------------------------------------------------------------
# A: Vector baseline (T.reduce_sum(dim=1))
# ---------------------------------------------------------------------------
def vec_sum(M, N):
    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def vec_reduce_main(
            x: T.Tensor((M, N), DT),
            out: T.Tensor((M, 1), ACC),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _), T.Scope("Vector"):
                xh = T.alloc_ub([M, N], DT)
                xf = T.alloc_ub([M, N], ACC)
                s = T.alloc_ub([M, 1], ACC)
                T.copy(x[0:M, 0:N], xh[0:M, 0:N])
                T.vcast(xh, xf, round_mode="rint")
                T.reduce_sum(xf, s, dim=1)
                T.copy(s[0:M, 0:1], out[0:M, 0:1])

        return vec_reduce_main

    return _f


# ---------------------------------------------------------------------------
# B: Cube ones-vector GEMV  out(M,PN) = x(M,N) @ ones
# ---------------------------------------------------------------------------
def cube_gemv(M, N, PN=1, KC=0, bt=False):
    """K-chunked by KC (0 = single shot), PN = padded GEMV output width."""
    kc = N if KC == 0 else KC
    ksteps = (N + kc - 1) // kc
    ones_l1_shape = [PN, N] if bt else [N, PN]

    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def cube_gemv_main(
            x: T.Tensor((M, N), DT),
            ones: T.Tensor((PN, N) if bt else (N, PN), DT),
            out: T.Tensor((M, PN), ACC),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _), T.Scope("Cube"):
                x_l1 = T.alloc_L1([M, kc], DT)
                ones_l1 = T.alloc_L1(ones_l1_shape, DT)
                acc = T.alloc_L0C([M, PN], ACC)

                # ones is loop-invariant: one ND2NZ per kernel call.
                T.load_nd2nz(ones[0, 0], ones_l1, ones_l1_shape)
                for k in T.serial(ksteps):
                    T.load_nd2nz(x[0, k * kc], x_l1, [M, kc])
                    T.gemm(
                        x_l1,
                        ones_l1,
                        acc,
                        initC=(k == 0),
                        b_transpose=bt,
                        size=[M, kc, PN],
                    )
                T.store_fixpipe(acc, out[0, 0], size=[M, PN],
                                enable_nz2nd=True)

        return cube_gemv_main

    return _f


# ---------------------------------------------------------------------------
# Host drivers
# ---------------------------------------------------------------------------
def _inputs(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    ref = x.cpu().double().sum(dim=1)  # float64 reference row sums
    return x, ref


def _ones(N, PN, bt):
    return (torch.ones(PN, N, dtype=torch.float16, device="npu") if bt
            else torch.ones(N, PN, dtype=torch.float16, device="npu"))


def run_vec(M, N):
    x, ref = _inputs(M, N)
    k = vec_sum(M, N)()
    with torch.no_grad():
        out = k(x)
    torch.npu.synchronize()
    got = out.cpu().double().reshape(M)
    err = float((got - ref).abs().max().item())
    print(f"CHK vec  M={M} N={N}: max_abs_err={err:.3e} "
          f"(ref0={float(ref[0]):.5f} got0={float(got[0]):.5f})", flush=True)
    return err


def run_cube(M, N, PN=1, KC=0, bt=False):
    x, ref = _inputs(M, N)
    ones = _ones(N, PN, bt)
    k = cube_gemv(M, N, PN, KC, bt)()
    with torch.no_grad():
        out = k(x, ones)
    torch.npu.synchronize()
    got = out.cpu().double()[0:M, 0]
    err = float((got - ref).abs().max().item())
    spread = float((out.cpu().double()[:, 1:] - out.cpu().double()[:, 0:1]).abs().max()) \
        if PN > 1 else 0.0
    print(f"CHK cube M={M} N={N} PN={PN} KC={KC} bt={int(bt)}: "
          f"max_abs_err={err:.3e} (ref0={float(ref[0]):.5f} "
          f"got0={float(got[0]):.5f}) col_spread={spread:.3e}", flush=True)
    return err


def bench(M, N, iters, PN=1, KC=0, bt=False, do_vec=True, do_cube=True):
    """Alternate A/B inside ONE process so both see identical device state."""
    x, _ = _inputs(M, N)
    ones = _ones(N, PN, bt) if do_cube else None
    ka = vec_sum(M, N)() if do_vec else None
    kb = cube_gemv(M, N, PN, KC, bt)() if do_cube else None
    with torch.no_grad():
        for _ in range(iters):
            if ka is not None:
                ka(x)
            if kb is not None:
                kb(x, ones)
        torch.npu.synchronize()
    print(f"BENCH done M={M} N={N} iters={iters} PN={PN} KC={KC} bt={int(bt)} "
          f"vec={do_vec} cube={do_cube}", flush=True)


# ---------------------------------------------------------------------------
# Multi-tile (persistent-style) variants: amortize the ~4 us per-launch floor
# so the *marginal* per-tile reduce cost is measurable.  Single core, static
# in-core T.serial loop -- the shape the FALN integration would actually use.
# ---------------------------------------------------------------------------
def vec_sum_multi(bm, N, ntasks):
    M = bm * ntasks

    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def vec_reduce_multi_main(
            x: T.Tensor((M, N), DT),
            out: T.Tensor((M, 1), ACC),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _), T.Scope("Vector"):
                xh = T.alloc_ub([bm, N], DT)
                xf = T.alloc_ub([bm, N], ACC)
                s = T.alloc_ub([bm, 1], ACC)
                for i in T.serial(ntasks):
                    T.copy(x[i * bm:(i + 1) * bm, 0:N], xh[0:bm, 0:N])
                    T.vcast(xh, xf, round_mode="rint")
                    T.reduce_sum(xf, s, dim=1)
                    T.copy(s[0:bm, 0:1], out[i * bm:(i + 1) * bm, 0:1])

        return vec_reduce_multi_main

    return _f


def cube_gemv_multi(bm, N, ntasks, kc=0, bt=True):
    """Persistent Cube row-reduce; ``kc`` = K chunk (0 = whole row per tile).

    Whole-row tiles blow the L1 budget at N=8192: the NZ layout pads every
    tile to 16 rows, so one (bm,8192) fp16 tile is 256 KB and two of them
    (x + ones) plus auto multi-buffering exceed the 512 KB L1 ("cbuf
    overflow, requires 6291456 bits while 4194304 bits available").
    """
    M = bm * ntasks
    kc_eff = N if kc == 0 else kc
    ksteps = (N + kc_eff - 1) // kc_eff

    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def cube_gemv_multi_main(
            x: T.Tensor((M, N), DT),
            ones: T.Tensor((1, N) if bt else (N, 1), DT),
            out: T.Tensor((M, 1), ACC),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _), T.Scope("Cube"):
                x_l1 = T.alloc_L1([bm, kc_eff], DT)
                ones_l1 = T.alloc_L1([1, kc_eff] if bt else [kc_eff, 1], DT)
                acc = T.alloc_L0C([bm, 1], ACC)
                for i in T.serial(ntasks):
                    for k in T.serial(ksteps):
                        T.load_nd2nz(x[i * bm, k * kc_eff], x_l1,
                                     [bm, kc_eff])
                        T.load_nd2nz(ones[0, k * kc_eff] if bt else ones[k * kc_eff, 0],
                                     ones_l1,
                                     [1, kc_eff] if bt else [kc_eff, 1])
                        T.gemm(x_l1, ones_l1, acc, initC=(k == 0),
                               b_transpose=bt, size=[bm, kc_eff, 1])
                    T.store_fixpipe(acc, out[i * bm, 0], size=[bm, 1],
                                    enable_nz2nd=True)

        return cube_gemv_multi_main

    return _f


def run_multi(bm, N, ntasks, kc=0):
    """Correctness of both multi-tile paths over the whole M = bm*ntasks."""
    M = bm * ntasks
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    ref = x.cpu().double().sum(dim=1)
    ka = vec_sum_multi(bm, N, ntasks)() if bm <= 2 else None
    with torch.no_grad():
        oa = ka(x) if ka is not None else None
    ones = torch.ones(1, N, dtype=torch.float16, device="npu")
    kb = cube_gemv_multi(bm, N, ntasks, kc, True)()
    with torch.no_grad():
        ob = kb(x, ones)
    torch.npu.synchronize()
    ea = (float((oa.cpu().double().reshape(M) - ref).abs().max().item())
          if oa is not None else float("nan"))
    eb = float((ob.cpu().double().reshape(M) - ref).abs().max().item())
    print(f"CHK multi M={M} N={N} bm={bm} ntasks={ntasks} kc={kc}: "
          f"vec_err={ea:.3e} cube_err={eb:.3e}", flush=True)
    return ea, eb


def bench_multi(bm, N, ntasks, iters, kc=0, do_vec=1):
    M = bm * ntasks
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    ones = torch.ones(1, N, dtype=torch.float16, device="npu")
    ka = vec_sum_multi(bm, N, ntasks)() if do_vec else None
    kb = cube_gemv_multi(bm, N, ntasks, kc, True)()
    with torch.no_grad():
        for _ in range(iters):
            if ka is not None:
                ka(x)
            kb(x, ones)
        torch.npu.synchronize()
    print(f"BENCH multi done M={M} N={N} bm={bm} ntasks={ntasks} "
          f"kc={kc} vec={do_vec} iters={iters}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="check",
                    choices=["compile", "check", "bench", "multi"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--bm", type=int, default=2)
    ap.add_argument("--ntasks", type=int, default=128)
    ap.add_argument("--pn", type=int, default=1)
    ap.add_argument("--kc", type=int, default=0)
    ap.add_argument("--bt", type=int, default=0)
    ap.add_argument("--vec", type=int, default=1)
    ap.add_argument("--cube", type=int, default=1)
    ap.add_argument("--shapes", default="",
                    help="override, e.g. 2x4096,1x8192")
    args = ap.parse_args()

    shapes = SHAPES
    if args.shapes:
        shapes = [tuple(int(v) for v in s.split("x")) for s in
                  args.shapes.split(",")]

    rc = 0
    for M, N in shapes:
        try:
            if args.mode == "compile":
                if args.vec:
                    vec_sum(M, N)()
                    print(f"COMPILE vec  M={M} N={N}: OK", flush=True)
                if args.cube:
                    cube_gemv(M, N, args.pn, args.kc, bool(args.bt))()
                    print(f"COMPILE cube M={M} N={N} PN={args.pn} "
                          f"KC={args.kc} bt={args.bt}: OK", flush=True)
            elif args.mode == "check":
                if args.vec:
                    run_vec(M, N)
                if args.cube:
                    run_cube(M, N, args.pn, args.kc, bool(args.bt))
            elif args.mode == "multi":
                run_multi(args.bm, N, args.ntasks, args.kc)
                bench_multi(args.bm, N, args.ntasks, args.iters, args.kc,
                            args.vec)
            else:
                bench(M, N, args.iters, args.pn, args.kc, bool(args.bt),
                      bool(args.vec), bool(args.cube))
        except Exception as exc:  # noqa: BLE001
            rc = 1
            msg = str(exc).replace("\n", " | ")[:600]
            print(f"FAIL {args.mode} M={M} N={N}: {type(exc).__name__}: {msg}",
                  flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
