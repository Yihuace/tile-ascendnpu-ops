"""Stage 4 round-2 candidate for TopkSelectorOp: ROW PARALLELISM.

Stage 3 delivered both kernels as ONE block with a serial row loop, because a
probe concluded that multi-block GM writes lose rows.  Re-measured on this op's
exact body (`probe_multiblock_rows.py`, grid1 / gridM / grid2), all three forms
now land every row, including per-row DIFFERENT thresholds (so a mis-targeted
write would be visible):

    want = [921, 804, 688, 571, 454, 337, 221, 104]
    grid1 (1 block + serial)  PASS
    gridM (T.Kernel(M), per-row block)  PASS
    grid2 (grid-stride, 2 blocks)       PASS

So the serial row loop is not required.  This candidate maps ROWS onto BLOCKS:

    grid = the largest divisor of M that is <= Vector cores (48)

An exact divisor keeps the per-block row loop tight (no tail, hence no runtime
guard on a loop variable -- the documented segfault trap).  For the manifest
workloads M is 8 or 16, i.e. grid = M, one row per block.

Everything else is the Stage 3 kernel unchanged: the 40 host-side bisection
launches stay (an in-kernel bisection is blocked: rounds>=2 crashes the compiler,
see probe_inbranch_rounds.py / logs/), and the count body keeps its 5-op form.

Expected effect: the per-row work (count pass, cumsum, compaction) now runs on M
cores instead of one; the 40 launches are unchanged, so only the per-launch
compute term shrinks.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

ROUNDS = 40                 # host-side bisection rounds (unchanged)

_CORE_CAP = 48              # Vector cores on this device
_UB_TILE_CAP = 2048         # 2048 fp32 -> ~40 KB of buffers, well inside UB


# Module-level tuned defaults, exposed for the Stage 5 wrapper switch block.
GRID_CAP = _CORE_CAP


def _grid_for(m, cap=GRID_CAP):
    """Largest divisor of ``m`` that is <= ``cap`` (falls back to 1)."""
    for g in range(min(m, cap), 0, -1):
        if m % g == 0:
            return g
    return 1


def _auto_tile(n, cap=_UB_TILE_CAP):
    """Largest divisor of ``n`` that is <= ``cap`` (fall back to ``n``)."""
    if n <= cap:
        return n
    for t in range(cap, 0, -1):
        if n % t == 0:
            return t
    return n


# ---------- Golden ----------
def golden_topk_indices(x, k):
    return torch.topk(x.detach().cpu().float(), k, dim=-1).indices


def set_equal(got, want):
    if got.shape != want.shape:
        return False
    for i in range(got.shape[0]):
        if set(got[i].tolist()) != set(want[i].tolist()):
            return False
    return True


# ---------- Kernel 1: count `x >= thr` per row (row-parallel) ----------
def _topk_count_kernel(M, N, tile=None):
    tile = _auto_tile(N) if tile is None else tile
    if N % tile:
        raise ValueError(f"N={N} must be a multiple of tile={tile}")
    nt = N // tile
    grid = _grid_for(M)
    per = M // grid

    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 thr: T.Tensor((M,), "float32"),
                 cnt: T.Tensor((M,), "float32")):
            with T.Kernel(grid, is_npu=True) as (pid, _):
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
                for q in T.serial(per):
                    ts[0, 0] = thr[pid + q * grid]
                    T.vmul(ts, -1.0, nts)               # -thr
                    T.clear(acc)
                    for t in T.serial(nt):
                        T.copy(x[pid + q * grid, t * tile:(t + 1) * tile], row[0, 0:tile])
                        T.vadd(row, nts[0, 0], diff)    # x - thr
                        T.vcmp(diff, zero, mask, "ge")  # x >= thr
                        T.vselect(mask, one, zero, sel)
                        T.reduce(sel, red, dims=1, reduce_mode="sum")
                        T.vadd(acc, red, acc)
                    cnt[pid + q * grid] = acc[0, 0]
        return main
    return _f()


# ---------- Kernel 2: select the indices >= thr, per row (row-parallel) ----------
def _topk_select_kernel(M, N, K, tile=None):
    tile = _auto_tile(N) if tile is None else tile
    if N % tile:
        raise ValueError(f"N={N} must be a multiple of tile={tile}")
    nt = N // tile
    grid = _grid_for(M)
    per = M // grid

    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 thr: T.Tensor((M,), "float32"),
                 out: T.Tensor((M, K), "int32")):
            with T.Kernel(grid, is_npu=True) as (pid, _):
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
                    krow[0, j] = float(K)               # clamp bound = sink slot
                for q in T.serial(per):
                    ts[0, 0] = thr[pid + q * grid]
                    T.vmul(ts, -1.0, nts)
                    T.clear(scratch)
                    T.clear(carried)
                    for t in T.serial(nt):
                        T.copy(x[pid + q * grid, t * tile:(t + 1) * tile], row[0, 0:tile])
                        T.vadd(row, nts[0, 0], diff)    # x - thr
                        T.vcmp(diff, zero, mask, "ge")  # x >= thr
                        T.vselect(mask, one, zero, sel)
                        T.cumsum(sel, posf, dim=1)      # 1-based rank in this piece
                        T.vsub(posf, one, posf)         # -> 0-based
                        T.vadd(posf, carried[0, 0], posf)
                        T.vsub(one, sel, inv)           # non-selected -> sink K
                        T.vmul(inv, float(K), inv)
                        T.vmul(sel, posf, posf)
                        T.vadd(posf, inv, spos)
                        T.vmin(spos, krow, spos)
                        T.vcast(spos, sposi, round_mode="rint")
                        for j in T.Parallel(tile):
                            scratch[0, sposi[0, j]] = t * tile + j
                        T.reduce(sel, cnt, dims=1, reduce_mode="sum")
                        T.vadd(carried, cnt, carried)
                    T.copy(scratch[0:1, 0:K], out[pid + q * grid:pid + q * grid + 1, 0:K])
        return main
    return _f()


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
