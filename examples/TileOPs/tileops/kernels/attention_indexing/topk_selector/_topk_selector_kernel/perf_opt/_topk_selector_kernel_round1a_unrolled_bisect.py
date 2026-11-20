"""Stage 4 round-1 candidate for TopkSelectorOp: bisection moved INTO the kernel.

Baseline (Stage 3 deliverable, `../_topk_selector_kernel.py`) runs the bisection on
the HOST: 40 launches of the count kernel, one per step, plus one select launch.
Each count launch handles a single threshold, so the op pays 41 kernel launches
(and 40 host round trips) for what is, per row, a 40-pass scan of the same data.

This candidate keeps the SAME algorithm and the SAME threshold arithmetic but
unrolls the 40 steps inside the count kernel:

    count kernel : row min/max -> 40 x (mid=(lo+hi)/2, count(x>=mid), update lo/hi)
    select kernel: unchanged, one launch with the converged threshold

Why the update is branch-free: the Stage 3 kernel's scalar compare is replaced by
arithmetic on integer-valued fp32 counts, which avoids `vcmp`/`vselect` on (1,1)
buffers:

    g   = clamp(count - (K-1), 0, 1)   # 1.0 iff count >= K
    lo' = lo + g * (mid - lo)
    hi' = mid + (1-g) * (hi - mid)

Compile-side constraint that shapes this file (measured in Stage 3, see
`../../../../../../topk_selector/_topk_selector_kernel/RETROSPECTIVE.md`):
">= 2 unrolled bisection bodies **plus** a compaction tail" segfaults the build,
while "no tail x 40 unrolled bodies" builds.  This candidate is exactly the
latter: the count kernel has no compaction tail, and compaction stays in the
separate one-launch select kernel.

What is NOT changed (must stay, all measured in Stage 3):
  * single block + serial row loop (`T.Kernel(M)` loses rows);
  * the row may be streamed in `tile`-wide pieces;
  * a GM scalar is copied into a (1,1) UB buffer and used as a v-op scalar operand.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

ROUNDS = 40                 # bisection steps, now unrolled inside the count kernel

# Tells the Stage 4 bench/runner that ONE count launch carries the whole bisection.
_COUNT_IN_KERNEL = True


# ---------- Golden ----------
def golden_topk_indices(x, k):
    """Set-contract golden: the top-k indices (order is irrelevant)."""
    return torch.topk(x.detach().cpu().float(), k, dim=-1).indices


def set_equal(got, want):
    """Set comparison, matching the in-repo `_set_compare` contract."""
    if got.shape != want.shape:
        return False
    for i in range(got.shape[0]):
        if set(got[i].tolist()) != set(want[i].tolist()):
            return False
    return True


_CORE_CAP = 48              # Vector cores on this device
_UB_TILE_CAP = 2048         # 2048 fp32 -> ~40 KB of buffers, well inside UB


def _auto_tile(n, cap=_UB_TILE_CAP):
    """Largest divisor of ``n`` that is <= ``cap`` (fall back to ``n``)."""
    if n <= cap:
        return n
    for t in range(cap, 0, -1):
        if n % t == 0:
            return t
    return n


# ---------- Kernel 1: in-kernel bisection -> per-row threshold ----------
def _topk_count_kernel(M, N, K, tile=None, rounds=ROUNDS):
    """Return the top-k threshold per row: the smallest t with `count(x >= t) >= K`.

    One launch, 40 unrolled bisection steps, no compaction tail (see module doc).
    """
    tile = _auto_tile(N) if tile is None else tile
    if N % tile:
        raise ValueError(f"N={N} must be a multiple of tile={tile}")
    nt = N // tile

    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 thr_out: T.Tensor((M,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                row = T.alloc_ub((1, tile), "float32")
                one = T.alloc_ub((1, tile), "float32")
                zero = T.alloc_ub((1, tile), "float32")
                sel = T.alloc_ub((1, tile), "float32")
                diff = T.alloc_ub((1, tile), "float32")
                mask = T.alloc_ub((1, tile), "uint8")
                red = T.alloc_ub((1, 1), "float32")
                acc = T.alloc_ub((1, 1), "float32")
                # Bisection state.  In-place (1,1) v-op updates are allowed (the
                # Stage 3 count kernel already does `T.vadd(acc, red, acc)`), so a
                # single lo/hi pair suffices and no Python ternary is needed --
                # a ternary inside a prim_func is parsed as a TIR Select and fails.
                lo_a = T.alloc_ub((1, 1), "float32")
                hi_a = T.alloc_ub((1, 1), "float32")
                mid = T.alloc_ub((1, 1), "float32")
                g = T.alloc_ub((1, 1), "float32")
                t1 = T.alloc_ub((1, 1), "float32")
                t3 = T.alloc_ub((1, 1), "float32")
                t4 = T.alloc_ub((1, 1), "float32")
                one1 = T.alloc_ub((1, 1), "float32")
                zero1 = T.alloc_ub((1, 1), "float32")
                km1 = T.alloc_ub((1, 1), "float32")
                for j in T.Parallel(tile):
                    one[0, j] = 1.0
                    zero[0, j] = 0.0
                for j in T.Parallel(1):
                    one1[0, j] = 1.0
                    zero1[0, j] = 0.0
                    km1[0, j] = float(K - 1)
                for r in T.serial(M):
                    # ---- seed: row min/max, one streamed pass ----
                    for j in T.Parallel(1):
                        lo_a[0, j] = 3.0e38
                        hi_a[0, j] = -3.0e38
                    for t in T.serial(nt):
                        T.copy(x[r, t * tile:(t + 1) * tile], row[0, 0:tile])
                        T.reduce(row, red, dims=1, reduce_mode="min")
                        T.vmin(lo_a, red, lo_a)
                        T.reduce(row, red, dims=1, reduce_mode="max")
                        T.vmax(hi_a, red, hi_a)
                    # ---- 40 unrolled bisection steps (Python-level unroll) ----
                    for _ in range(rounds):
                        T.vadd(lo_a, hi_a, mid)
                        T.vmul(mid, 0.5, mid)            # mid = (lo+hi)/2
                        T.clear(acc)
                        for t in T.serial(nt):
                            T.copy(x[r, t * tile:(t + 1) * tile], row[0, 0:tile])
                            T.vsub(row, mid[0, 0], diff)        # x - mid
                            T.vcmp(diff, zero, mask, "ge")      # x >= mid
                            T.vselect(mask, one, zero, sel)
                            T.reduce(sel, red, dims=1, reduce_mode="sum")
                            T.vadd(acc, red, acc)
                        # g = clamp(count - (K-1), 0, 1) -> 1.0 iff count >= K
                        T.vsub(acc, km1[0, 0], g)
                        T.vmax(g, zero1, g)
                        T.vmin(g, one1, g)
                        # lo += g*(mid-lo)      -- reads the OLD lo only
                        T.vsub(mid, lo_a, t1)
                        T.vmul(t1, g, t1)
                        T.vadd(lo_a, t1, lo_a)
                        # hi  = mid + (1-g)*(hi-mid)   -- reads the OLD hi only
                        T.vsub(hi_a, mid, t3)
                        T.vsub(one1, g, t4)
                        T.vmul(t3, t4, t3)
                        T.vadd(mid, t3, hi_a)
                    thr_out[r] = lo_a[0, 0]
        return main
    return _f()


# ---------- Kernel 2: select the indices >= thr, per row (UNCHANGED) ----------
def _topk_select_kernel(M, N, K, tile=None):
    """Emit the indices with `x >= thr`, up to K per row, into (M, K)."""
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
                    krow[0, j] = float(K)               # clamp bound = sink slot
                for r in T.serial(M):
                    ts[0, 0] = thr[r]
                    T.vmul(ts, -1.0, nts)
                    T.clear(scratch)
                    T.clear(carried)
                    for t in T.serial(nt):
                        T.copy(x[r, t * tile:(t + 1) * tile], row[0, 0:tile])
                        T.vadd(row, nts[0, 0], diff)    # x - thr
                        T.vcmp(diff, zero, mask, "ge")  # x >= thr
                        T.vselect(mask, one, zero, sel)
                        T.cumsum(sel, posf, dim=1)      # 1-based rank in this piece
                        T.vsub(posf, one, posf)         # -> 0-based
                        T.vadd(posf, carried[0, 0], posf)   # + carried offset
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
                    T.copy(scratch[0:1, 0:K], out[r:r + 1, 0:K])
        return main
    return _f()


# ---------- Host-side driver: ONE count launch + ONE select launch ----------
def topk_indices(x, k, rounds=ROUNDS, tile=None):
    """Return the (M, K) int32 index tensor.  Two launches, no host bisection."""
    m, n = x.shape
    kc = _topk_count_kernel(m, n, k, tile=tile, rounds=rounds)
    ks = _topk_select_kernel(m, n, k, tile=tile)
    xn = x.npu()
    thr = kc(xn).cpu()
    idx = ks(xn, thr.contiguous().npu()).cpu()
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
