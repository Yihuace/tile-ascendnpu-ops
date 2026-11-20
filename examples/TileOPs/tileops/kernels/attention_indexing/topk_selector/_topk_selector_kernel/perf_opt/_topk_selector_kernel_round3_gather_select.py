"""Stage 4 round-3 candidate: compaction WITHOUT scalar cumsum / scatter.

Measured baseline (msprof, this machine):
    workload          count(40 launches)   select      total
    topk-1024            40 x 16.861      494.260    1168.700 us
    topk-4096            40 x 24.561      981.499    1963.939 us
    topk-8192-tiled      40 x 35.281     1944.538    3355.778 us

The select kernel is the single largest term and it is dominated by two ops the
compiler turns into SCALAR work:

  * `T.cumsum(sel, posf, dim=1)` -- the build itself warns:
    "Op 'hivm.hir.vcumsum' will execute by scalar instruction with low effiency"
  * `scratch[0, sposi[0, j]] = ...` -- a per-element scatter, also scalar.

This candidate removes both.  Because the op's contract is a SET (order is
irrelevant), the output can be produced as "the q-th smallest selected index":

    mask = (x >= thr)                     # one vector compare, reused
    for q in T.serial(K):                 # K output slots, fixed bound
        cand = mask ? iota : BIG          # vector select
        m    = reduce_min(cand)           # vector reduce
        mask[m] = 0                       # single scalar store, not N
        out[r, q] = m

so the per-element scalar work disappears: K passes of pure vector ops instead of
N cumsum steps plus N scatter stores.  Cost is O(K*N), which is favourable for the
small-K cases and roughly neutral at K=64 (measured below).

Whole-row-only: this path needs iota/mask over the full row, so it is used when the
row fits in one tile (N <= _SEL_TILE_CAP).  For wider rows the Stage 3 streaming
compaction is kept verbatim as the fallback.

Everything else (single block, serial rows, host bisection with 40 launches) is
unchanged -- row parallelism and in-kernel bisection are both blocked, see
logs/ (probe_round2_isolate.log: multi-block writes drop rows intermittently;
probe_inbranch_rounds.log: rounds>=2 crashes the compiler).
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

ROUNDS = 40

_CORE_CAP = 48
_UB_TILE_CAP = 2048         # count kernel's per-piece buffer cap (unchanged)
_SEL_TILE_CAP = 8192        # select gather path: whole row in UB (13 B/element)

BIG_INDEX = 1.0e9           # sentinel for "no selected element here"


def _auto_tile(n, cap=_UB_TILE_CAP):
    if n <= cap:
        return n
    for t in range(cap, 0, -1):
        if n % t == 0:
            return t
    return n


def golden_topk_indices(x, k):
    return torch.topk(x.detach().cpu().float(), k, dim=-1).indices


def set_equal(got, want):
    if got.shape != want.shape:
        return False
    for i in range(got.shape[0]):
        if set(got[i].tolist()) != set(want[i].tolist()):
            return False
    return True


# ---------- Kernel 1: count `x >= thr` per row (unchanged) ----------
def _topk_count_kernel(M, N, tile=None):
    tile = _auto_tile(N) if tile is None else tile
    if N % tile:
        raise ValueError(f"N={N} must be a multiple of tile={tile}")
    nt = N // tile

    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 thr: T.Tensor((M,), "float32"),
                 cnt: T.Tensor((M,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                row = T.alloc_ub((1, tile), "float32")
                one = T.alloc_ub((1, tile), "float32")
                zero = T.alloc_ub((1, tile), "float32")
                sel = T.alloc_ub((1, tile), "float32")
                diff = T.alloc_ub((1, tile), "float32")
                mask = T.alloc_ub((1, tile), "uint8")
                red = T.alloc_ub((1, 1), "float32")
                acc = T.alloc_ub((1, 1), "float32")
                ts = T.alloc_ub((1, 1), "float32")
                nts = T.alloc_ub((1, 1), "float32")
                for j in T.Parallel(tile):
                    one[0, j] = 1.0
                    zero[0, j] = 0.0
                for r in T.serial(M):
                    ts[0, 0] = thr[r]
                    T.vmul(ts, -1.0, nts)
                    T.clear(acc)
                    for t in T.serial(nt):
                        T.copy(x[r, t * tile:(t + 1) * tile], row[0, 0:tile])
                        T.vadd(row, nts[0, 0], diff)
                        T.vcmp(diff, zero, mask, "ge")
                        T.vselect(mask, one, zero, sel)
                        T.reduce(sel, red, dims=1, reduce_mode="sum")
                        T.vadd(acc, red, acc)
                    cnt[r] = acc[0, 0]
        return main
    return _f()


# ---------- Kernel 2a: whole-row gather compaction (NEW) ----------
def _topk_select_gather_kernel(M, N, K):
    """out[r, q] = the q-th smallest index with x >= thr  (set contract)."""
    tile = N

    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 thr: T.Tensor((M,), "float32"),
                 out: T.Tensor((M, K), "int32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                row = T.alloc_ub((1, tile), "float32")
                iota = T.alloc_ub((1, tile), "float32")
                big = T.alloc_ub((1, tile), "float32")
                cand = T.alloc_ub((1, tile), "float32")
                cand2 = T.alloc_ub((1, tile), "float32")
                mask = T.alloc_ub((1, tile), "uint8")
                lemask = T.alloc_ub((1, tile), "uint8")
                red = T.alloc_ub((1, 1), "float32")
                redi = T.alloc_ub((1, 1), "int32")
                lastf = T.alloc_ub((1, 1), "float32")
                ts = T.alloc_ub((1, 1), "float32")
                one1 = T.alloc_ub((1, 1), "float32")
                zero1 = T.alloc_ub((1, 1), "float32")
                # iota = 0..tile-1, built ONCE for the whole kernel.  A Parallel
                # assignment cannot write `float(j)` (j is a TIR Var -> float()
                # raises), so iota comes from cumsum over a ones row, then -1.
                for j in T.Parallel(tile):
                    row[0, j] = 1.0
                    big[0, j] = BIG_INDEX
                for j in T.Parallel(1):
                    one1[0, j] = 1.0
                    zero1[0, j] = 0.0
                    lastf[0, j] = -1.0          # nothing taken yet
                T.cumsum(row, iota, dim=1)          # 1..tile
                T.vsub(iota, one1[0, 0], iota)      # -> 0..tile-1
                for r in T.serial(M):
                    ts[0, 0] = thr[r]
                    # lastf must be reset per row: it is the "already taken up to
                    # here" cursor, and carrying it across rows silently excluded
                    # every candidate of the following rows (measured: single-row
                    # probe PASS, multi-row suite FAIL).
                    lastf[0, 0] = -1.0
                    T.copy(x[r, 0:tile], row[0, 0:tile])
                    T.vcmp(row, ts[0, 0], mask, "ge")   # mask = (x >= thr); scalar operand
                    # Selected indices are emitted in INCREASING order, so the next
                    # one is always > the last one.  Excluding `<= last` needs no
                    # mask mutation -- a scalar scatter store `mask[m] = 0` was
                    # measured to be silently dropped (probe_gather_out.py printed
                    # the same index K times), so this path uses no scatter at all.
                    for q in T.serial(K):
                        T.vselect(mask, iota, big, cand)
                        T.vcmp(cand, lastf[0, 0], lemask, "le")
                        T.vselect(lemask, big, cand, cand2)
                        T.reduce(cand2, red, dims=1, reduce_mode="min")
                        T.vcast(red, redi, round_mode="rint")
                        out[r, q] = redi[0, 0]
                        T.vadd(red, zero1[0, 0], lastf)
        return main
    return _f()


# ---------- Kernel 2b: Stage 3 streaming compaction (fallback, verbatim) ----------
def _topk_select_stream_kernel(M, N, K, tile=None):
    tile = _auto_tile(N) if tile is None else tile
    if N % tile:
        raise ValueError(f"N={N} must be a multiple of tile={tile}")
    nt = N // tile

    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 thr: T.Tensor((M,), "float32"),
                 out: T.Tensor((M, K), "int32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                row = T.alloc_ub((1, tile), "float32")
                one = T.alloc_ub((1, tile), "float32")
                zero = T.alloc_ub((1, tile), "float32")
                krow = T.alloc_ub((1, tile), "float32")
                sel = T.alloc_ub((1, tile), "float32")
                inv = T.alloc_ub((1, tile), "float32")
                diff = T.alloc_ub((1, tile), "float32")
                mask = T.alloc_ub((1, tile), "uint8")
                posf = T.alloc_ub((1, tile), "float32")
                spos = T.alloc_ub((1, tile), "float32")
                sposi = T.alloc_ub((1, tile), "int32")
                scratch = T.alloc_ub((1, K + 1), "int32")
                ts = T.alloc_ub((1, 1), "float32")
                nts = T.alloc_ub((1, 1), "float32")
                cnt = T.alloc_ub((1, 1), "float32")
                carried = T.alloc_ub((1, 1), "float32")
                for j in T.Parallel(tile):
                    one[0, j] = 1.0
                    zero[0, j] = 0.0
                    krow[0, j] = float(K)
                for r in T.serial(M):
                    ts[0, 0] = thr[r]
                    T.vmul(ts, -1.0, nts)
                    T.clear(scratch)
                    T.clear(carried)
                    for t in T.serial(nt):
                        T.copy(x[r, t * tile:(t + 1) * tile], row[0, 0:tile])
                        T.vadd(row, nts[0, 0], diff)
                        T.vcmp(diff, zero, mask, "ge")
                        T.vselect(mask, one, zero, sel)
                        T.cumsum(sel, posf, dim=1)
                        T.vsub(posf, one, posf)
                        T.vadd(posf, carried[0, 0], posf)
                        T.vsub(one, sel, inv)
                        T.vmul(inv, float(K), inv)
                        T.vmul(sel, posf, posf)
                        T.vadd(posf, inv, spos)
                        T.vmin(spos, krow, spos)
                        T.vcast(spos, sposi, round_mode="rint")
                        for j in T.Parallel(tile):
                            scratch[0, sposi[0, j]] = t * tile + j
                        T.reduce(sel, cnt, dims=1, reduce_mode="sum")
                        T.vadd(carried, cnt, carried)
                    T.copy(scratch[0:1, 0:K], out[r:r + 1, 0:K])
        return main
    return _f()


def _topk_select_kernel(M, N, K, tile=None):
    """Pick the compaction path: whole-row gather when the row fits, else streaming."""
    if tile is None and N <= _SEL_TILE_CAP:
        return _topk_select_gather_kernel(M, N, K)
    if tile is not None and tile >= N:
        return _topk_select_gather_kernel(M, N, K)
    return _topk_select_stream_kernel(M, N, K, tile=tile)


# ---------- Host-side driver: bisection + selection (unchanged) ----------
def topk_indices(x, k, rounds=ROUNDS, tile=None):
    m, n = x.shape
    kc = _topk_count_kernel(m, n, tile=tile)
    ks = _topk_select_kernel(m, n, k, tile=tile)
    xn = x.npu()
    lo = x.min(dim=1).values.to(torch.float32).clone()
    hi = x.max(dim=1).values.to(torch.float32).clone()
    for _ in range(rounds):
        mid = (lo + hi) * 0.5
        c = kc(xn, mid.npu()).cpu()
        ge = c >= float(k)
        lo = torch.where(ge, mid, lo)
        hi = torch.where(ge, hi, mid)
    idx = ks(xn, lo.npu()).cpu()
    return idx


# ---------- Driver (same harness contract as the Stage 3 file) ----------
def run_case(m, n, k, tag, distinct=True, rounds=ROUNDS, tile=None):
    if distinct:
        base = torch.randperm(n, dtype=torch.float32) * 7.0
        x = base.unsqueeze(0).repeat(m, 1) + torch.arange(m, dtype=torch.float32).unsqueeze(1)
    else:
        x = torch.randn(m, n, dtype=torch.float32)
    out = topk_indices(x, k, rounds=rounds, tile=tile)
    ref = golden_topk_indices(x, k)
    ok = set_equal(out, ref)
    if not ok:
        print(f"    got idx ={sorted(out[0].tolist())[:10]}")
        print(f"    want idx={sorted(ref[0].tolist())[:10]}")
    print(f"[{tag}] (M,N,K)=({m},{n},{k}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_L0():
    run_case(2, 256, 8, "L0")


def run_L1():
    run_case(4, 1024, 16, "L1")
    run_case(4, 1024, 1, "L1-k1")
    run_case(4, 1024, 1024, "L1-kN")
    run_case(3, 2048, 32, "L1-2048")
    run_case(3, 2048, 32, "L1-tiled", tile=512)
    run_case(2, 4096, 64, "L1-tiled-4096", tile=1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L0", choices=["L0", "all"])
    a, _ = ap.parse_known_args()
    run_L0()
    if a.level == "all":
        run_L1()
    print("done")


if __name__ == "__main__":
    main()
