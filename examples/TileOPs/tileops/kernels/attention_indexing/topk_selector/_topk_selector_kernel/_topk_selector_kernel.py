# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TopkSelectorOp -- TileLang-NPUIR kernels, golden and hierarchical tests.

Exact top-k along the last dimension, returning int32 INDICES.  The in-repo test
compares as a SET (`_set_compare`), and the golden below matches that contract.

Design: ./DESIGN.md (Stage 2 review passed, see ./REVIEW.md), with its blocker
section CORRECTED after re-measurement -- and with the ARCHITECTURE changed, for
the two measured reasons recorded below.

ARCHITECTURE: two simple kernels + a host-side bisection
-------------------------------------------------------
1. `_topk_count_kernel(x, thr) -> cnt`   -- per row, how many elements are >= thr
2. `_topk_select_kernel(x, thr) -> idx`  -- per row, the indices of those >= thr

The threshold bisection runs on the HOST (a loop there costs nothing).  Each
kernel is a single pass with NO unrolled rounds, which is what makes them build.

WHY NOT ONE KERNEL (measured, see repro/):
  * `PROBE-topk-min-repro.py`: a kernel with the compaction PLUS two or more
    unrolled copies of the bisection body SEGFAULTS the build.  `rounds=1` and
    all four write-form combinations build and run; `rounds>=2` always crash
    (3 segfaults), and it is not about any single instruction -- removing the
    scatter, the cumsum, the slice copy or the loop dependency changes nothing.
    `TILELANG_DUMP_IR=TRUE` shows the IR dying in TVM bufferization at
    `materialize_in_destination` for the K-length slice copy (stride mismatch),
    but that is the last thing printed, not the whole story.
  * `PROBE-rows-arch.py`: **multi-block GM writes are UNRELIABLE**.
        T.Kernel(M) (one block per row)   -> 4 rows, only 2 landed  (FAIL)
        T.Kernel(1) + T.serial over rows  -> 4 rows, all correct   (PASS)
    So every kernel here uses `T.Kernel(1)` with a serial loop over rows.

Other measured rules this file relies on (repro/PROBE-vop11.py,
PROBE-scatter.py, PROBE-topk-primitive-baseline.py):
  * a `T.Parallel` assignment loop's RHS must NOT read a (1,1) buffer;
  * a GUARDED scatter silently writes nothing -- fold the mask into the POSITION
    arithmetic and aim non-selected lanes at a sink slot instead;
  * counting must be in the fp32 domain (the 1-D int32 reduce is wrong);
  * a `T.prim_func` body must not contain a bare `return`;
  * nested `def` inside a `T.prim_func` is rejected by the script parser;
  * `TILELANG_ASCEND_MODE` must be an ENVIRONMENT VARIABLE.

Run:
    python _topk_selector_kernel.py --level L0
    python _topk_selector_kernel.py --level all
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

tilelang.cache.clear_cache()

ROUNDS = 40                 # host-side bisection rounds


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


# ---------- Kernel 1: count elements >= thr, per row ----------
_CORE_CAP = 48              # Vector cores on this device

# UB budget for the per-piece buffers, in fp32 elements.  `select` holds the most
# (about nine (1,tile) float buffers plus an int32 and a uint8 one), so the cap is
# chosen for it: ~9*4*tile + 4*tile + tile bytes must stay under the 192 KB UB.
_UB_TILE_CAP = 2048         # 2048 fp32 -> ~40 KB of buffers, well inside UB


def _auto_tile(n, cap=_UB_TILE_CAP):
    """Largest divisor of ``n`` that is <= ``cap`` (fall back to ``n``).

    A row wider than the cap MUST be streamed, otherwise the per-piece buffers
    overflow UB -- measured as a build failure:
        error: ub overflow, requires 1841920 bits while 1572864 bits available!
    Choosing a DIVISOR keeps the loop exact (no ragged tail to guard, and a
    runtime guard on a serial loop variable is the documented segfault trap).
    """
    if n <= cap:
        return n
    for t in range(cap, 0, -1):
        if n % t == 0:
            return t
    return n


def _topk_count_kernel(M, N, tile=None):
    """Count `x >= thr` per row.

    Architecture -- every choice here is measured, see repro/:
      * **grid = 1 block**, serial over rows.  One-block-per-row (`T.Kernel(M)`)
        LOSES rows (PROBE-rows-arch: 4 rows, only 2 landed); `T.Kernel(1)` +
        serial is the form that was correct.  (Capping the grid at
        `min(M, cores)` does NOT help: for small M that is one-block-per-row
        again, and it measured FAIL.)
      * the row may be streamed in `tile`-wide pieces, so N may exceed UB.
    """
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
                    # thr is a GM scalar: copy it into a (1,1) UB buffer and use it
                    # as an OPERAND of a row-wide v-op.  Reading a scalar inside a
                    # T.Parallel assignment loop is measured to fail.
                    ts[0, 0] = thr[r]
                    T.vmul(ts, -1.0, nts)               # -thr
                    T.clear(acc)
                    for t in T.serial(nt):
                        T.copy(x[r, t * tile:(t + 1) * tile], row[0, 0:tile])
                        T.vadd(row, nts[0, 0], diff)    # x - thr
                        T.vcmp(diff, zero, mask, "ge")  # x >= thr
                        T.vselect(mask, one, zero, sel)
                        T.reduce(sel, red, dims=1, reduce_mode="sum")
                        T.vadd(acc, red, acc)
                    cnt[r] = acc[0, 0]
        return main
    return _f()


# ---------- Kernel 2: select the indices >= thr, per row ----------
def _topk_select_kernel(M, N, K, tile=None):
    """Emit the indices with `x >= thr`, up to K per row, into (M, K).

    Single block, serial over rows, and the row may be streamed in `tile`-wide
    pieces.  Guard-free compaction: non-selected lanes are aimed at sink slot K
    (the scratch is (1, K+1); slot K is never read).  Across pieces the write
    position is carried in a (1,1) accumulator -- the same loop-carried form the
    count kernel already uses in a serial loop, so it is not the unrolled-body
    structure that segfaults the build.
    """
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
                        # CLAMP into the scratch: the carried offset can push a
                        # position past K, and the scratch is only (1, K+1).
                        T.vmin(spos, krow, spos)
                        T.vcast(spos, sposi, round_mode="rint")
                        for j in T.Parallel(tile):
                            scratch[0, sposi[0, j]] = t * tile + j
                        T.reduce(sel, cnt, dims=1, reduce_mode="sum")
                        T.vadd(carried, cnt, carried)
                    T.copy(scratch[0:1, 0:K], out[r:r + 1, 0:K])
        return main
    return _f()


# ---------- In-kernel bisection: the whole round loop inside ONE launch ----------
# WHY: the host bisection issues `rounds` launches, and a launch costs ~0.11-0.13 ms no
# matter how little the kernel does, so 41 launches = ~5.2 ms while the arithmetic is
# negligible.  msprof measures only 6.5-35 us of DEVICE time and therefore cannot see
# this at all.  Collapsing the round loop into the kernel takes 41 launches down to 2.
#
# TWO MEASURED CONSTRAINTS (see repro/PROBE-topk-inkernel-isolate.py):
#   * `T.if_then_else` on a runtime scalar SEGFAULTS; the branch must be built from
#     vector ops.  The flag comes from vcmp/vselect and lo/hi move with vadd/vsub/vmul
#     only -- a scalar read of a buffer just written by a vector op is a cross-pipe
#     dependency the compiler does not order (that version silently produced the round-1
#     midpoint 220.5 instead of 385.875 and then stopped advancing).
#   * everything must stay inside one UB-resident row, so no `T.copy` may appear in the
#     round loop (that is the "complex loop body" that raises ERR99999).
_UB_BUDGET = 192 * 1024
_BISECT_VEC_BUFS = 5          # row, diff, sel, one, zero  (N * 4 bytes each)
_BISECT_FLAG_BUFS = 1         # mask                       (N bytes)


def _row_fits_ub(n: int) -> bool:
    """Can one whole row plus its working buffers live in UB?"""
    return _BISECT_VEC_BUFS * n * 4 + _BISECT_FLAG_BUFS * n <= _UB_BUDGET


def _topk_bisect_kernel(M, N, K, rounds):
    """Per row: `rounds` bisection rounds, entirely on device.  Output = threshold.

    The invariant kept every round is `count(x >= lo) >= K`, so the returned `lo` has at
    least K elements at or above it -- exactly what the selection kernel needs.
    """
    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 thr: T.Tensor((M,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                row = T.alloc_ub((1, N), "float32")
                diff = T.alloc_ub((1, N), "float32")
                sel = T.alloc_ub((1, N), "float32")
                one = T.alloc_ub((1, N), "float32")
                zero = T.alloc_ub((1, N), "float32")
                mask = T.alloc_ub((1, N), "uint8")
                lo = T.alloc_ub((1, 1), "float32")
                hi = T.alloc_ub((1, 1), "float32")
                mid = T.alloc_ub((1, 1), "float32")
                nmid = T.alloc_ub((1, 1), "float32")
                s = T.alloc_ub((1, 1), "float32")
                mn = T.alloc_ub((1, 1), "float32")
                mx = T.alloc_ub((1, 1), "float32")
                one1 = T.alloc_ub((1, 1), "float32")
                zero1 = T.alloc_ub((1, 1), "float32")
                kc1 = T.alloc_ub((1, 1), "float32")
                m1 = T.alloc_ub((1, 1), "uint8")
                t1 = T.alloc_ub((1, 1), "float32")
                t2 = T.alloc_ub((1, 1), "float32")
                for j in T.Parallel(N):
                    one[0, j] = 1.0
                    zero[0, j] = 0.0
                one1[0, 0] = 1.0
                zero1[0, 0] = 0.0
                kc1[0, 0] = T.Cast("float32", K)
                for r in T.serial(M):
                    T.copy(x[r, 0:N], row[0, 0:N])
                    T.reduce(row, mn, dims=1, reduce_mode="min")
                    T.reduce(row, mx, dims=1, reduce_mode="max")
                    lo[0, 0] = mn[0, 0]
                    hi[0, 0] = mx[0, 0]
                    for _t in T.serial(rounds):
                        T.vadd(lo, hi, t1)                 # t1 = lo + hi
                        T.vmul(t1, 0.5, mid)               # mid = (lo + hi) / 2
                        T.vmul(mid, -1.0, nmid)            # -mid
                        T.vadd(row, nmid[0, 0], diff)      # x - mid
                        T.vcmp(diff, zero, mask, "ge")     # x >= mid
                        T.vselect(mask, one, zero, sel)
                        T.reduce(sel, s, dims=1, reduce_mode="sum")
                        T.vsub(s, kc1, s)                  # s = count - K
                        T.vcmp(s, zero1, m1, "ge")
                        T.vselect(m1, one1, zero1, s)      # s = 1 if count >= K else 0
                        T.vsub(mid, lo, t1)
                        T.vmul(t1, s[0, 0], t1)
                        T.vadd(lo, t1, lo)                 # lo = lo + (mid-lo)*flag
                        T.vsub(hi, mid, t2)
                        T.vmul(t2, s[0, 0], t2)
                        T.vadd(mid, t2, hi)                # hi = mid + (hi-mid)*flag
                    thr[r] = lo[0, 0]
        return main
    return _f()


# ---------- Fused kernel: bisection + selection in ONE launch ----------
# WHY: the harness measures one kernel per record and aggregates N samples of the
# SAME kernel with `durations.sort(); median = d[len//2]`.  Issuing two kernels (the
# old form) means msprof matched both `main` prim_funcs, so `sample_count` was 2 and
# the median became the SLOWER of two DIFFERENT kernels while the Ratio's roofline
# bin came from the first (faster) one -- latency and Ratio described different
# kernels and neither described the op.  One launch per call makes the harness's
# contract hold again.
#
# UB: row, diff, sel, one, zero (fp32) + mask (uint8) + sposi (int32)
#     = 25 B/element -> N<=4096 needs 100 KB (fits 192 KB), N=8192 needs 200 KB
#     (does not fit), so 8192 keeps the two-launch path.
_UB_FUSED_BYTES_PER_ELEM = 5 * 4 + 1 + 4


def _fused_fits_ub(n: int) -> bool:
    """Can the fused (single-launch) form fit UB at this row width?"""
    return _UB_FUSED_BYTES_PER_ELEM * n <= _UB_BUDGET


def _topk_fused_kernel(M, N, K, rounds):
    """Bisection followed by selection, one kernel, one launch per call.

    After the 40-round loop `diff`, `sel`, `one`, `zero` and `mask` are dead, so the
    selection runs over the same UB-resident row and reuses them; only one extra
    int32-wide buffer is needed for the cast scatter index.
    """
    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 idx: T.Tensor((M, K), "int32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                row = T.alloc_ub((1, N), "float32")
                diff = T.alloc_ub((1, N), "float32")
                sel = T.alloc_ub((1, N), "float32")
                one = T.alloc_ub((1, N), "float32")
                zero = T.alloc_ub((1, N), "float32")
                mask = T.alloc_ub((1, N), "uint8")
                sposi = T.alloc_ub((1, N), "int32")
                scratch = T.alloc_ub((1, K + 1), "int32")
                lo = T.alloc_ub((1, 1), "float32")
                hi = T.alloc_ub((1, 1), "float32")
                mid = T.alloc_ub((1, 1), "float32")
                nmid = T.alloc_ub((1, 1), "float32")
                s = T.alloc_ub((1, 1), "float32")
                mn = T.alloc_ub((1, 1), "float32")
                mx = T.alloc_ub((1, 1), "float32")
                one1 = T.alloc_ub((1, 1), "float32")
                zero1 = T.alloc_ub((1, 1), "float32")
                kc1 = T.alloc_ub((1, 1), "float32")
                m1 = T.alloc_ub((1, 1), "uint8")
                t1 = T.alloc_ub((1, 1), "float32")
                t2 = T.alloc_ub((1, 1), "float32")

                for j in T.Parallel(N):
                    one[0, j] = 1.0
                    zero[0, j] = 0.0
                one1[0, 0] = 1.0
                zero1[0, 0] = 0.0
                kc1[0, 0] = T.Cast("float32", K)

                for r in T.serial(M):
                    T.copy(x[r, 0:N], row[0, 0:N])
                    T.reduce(row, mn, dims=1, reduce_mode="min")
                    T.reduce(row, mx, dims=1, reduce_mode="max")
                    lo[0, 0] = mn[0, 0]
                    hi[0, 0] = mx[0, 0]

                    # ---- phase 1: bisection, all 40 rounds in-device ----
                    for _t in T.serial(rounds):
                        T.vadd(lo, hi, t1)
                        T.vmul(t1, 0.5, mid)
                        T.vmul(mid, -1.0, nmid)
                        T.vadd(row, nmid[0, 0], diff)
                        T.vcmp(diff, zero, mask, "ge")
                        T.vselect(mask, one, zero, sel)
                        T.reduce(sel, s, dims=1, reduce_mode="sum")
                        T.vsub(s, kc1, s)
                        T.vcmp(s, zero1, m1, "ge")
                        T.vselect(m1, one1, zero1, s)
                        T.vsub(mid, lo, t1)
                        T.vmul(t1, s[0, 0], t1)
                        T.vadd(lo, t1, lo)
                        T.vsub(hi, mid, t2)
                        T.vmul(t2, s[0, 0], t2)
                        T.vadd(mid, t2, hi)

                    # ---- phase 2: selection, reusing the dead buffers ----
                    T.vmul(lo, -1.0, t1)                  # t1 = -lo
                    T.vadd(row, t1[0, 0], diff)           # diff = x - lo
                    T.vcmp(diff, zero, mask, "ge")        # mask = x >= lo
                    T.vselect(mask, one, zero, sel)       # sel = 1/0
                    T.cumsum(sel, diff, dim=1)            # 1-based rank
                    T.vsub(diff, one, diff)               # 0-based rank
                    T.vmul(sel, diff, diff)               # selected keep rank
                    T.vmul(sel, one, sel)                 # recompute 1/0
                    T.vsub(one, sel, sel)                 # 0 for selected, else 1
                    T.vmul(sel, kc1[0, 0], sel)           # sink slot K
                    T.vadd(diff, sel, diff)               # final slot
                    T.vcast(diff, sposi, round_mode="rint")
                    T.clear(scratch)
                    for j in T.Parallel(N):
                        scratch[0, sposi[0, j]] = j
                    T.copy(scratch[0:1, 0:K], idx[r, 0:K])
        return main
    return _f()


# ---------- Host-side driver: bisection + selection ----------
# ---------- Kernel factory cache ----------
# Each `_topk_*_kernel(...)` call BUILDS a kernel: measured 46.5 ms per call after
# the first compile.  `topk_indices` used to rebuild on every invocation, so any
# caller of the raw function paid ~40-60 ms of pure factory cost per call while the
# actual device work is 17-53 us.  The Op wrapper already caches (self._kf/_kb/_kc),
# this brings the function path in line.
_KERNEL_CACHE: dict = {}


def _cached(builder, *args, **kwargs):
    """Memoise a kernel factory by (builder, args...) so it is never rebuilt."""
    key = (builder.__name__, args, tuple(sorted(kwargs.items())))
    hit = _KERNEL_CACHE.get(key)
    if hit is None:
        hit = builder(*args, **kwargs)
        _KERNEL_CACHE[key] = hit
    return hit


def topk_indices(x, k, rounds=ROUNDS, tile=None):
    """Return the (M, K) int32 index tensor.

    Fast path (a whole row fits in UB): ONE in-kernel bisection launch plus one
    selection launch.  Otherwise the original host-side bisection is used, which needs
    `rounds + 1` launches.

    `tile` streams each row in pieces of that width (needed when the row does not
    fit in UB); `None` means one whole-row piece.
    """
    m, n = x.shape
    xn = x.npu()

    if tile is None and _fused_fits_ub(n):
        # one launch: bisection and selection in the same kernel
        return _cached(_topk_fused_kernel, m, n, k, rounds)(xn).cpu()

    if tile is None and _row_fits_ub(n):
        kb = _cached(_topk_bisect_kernel, m, n, k, rounds)
        ks = _cached(_topk_select_kernel, m, n, k, tile=None)
        return ks(xn, kb(xn)).cpu()

    kc = _cached(_topk_count_kernel, m, n, tile=tile)
    ks = _cached(_topk_select_kernel, m, n, k, tile=tile)
    lo = x.min(dim=1).values.to(torch.float32).clone()
    hi = x.max(dim=1).values.to(torch.float32).clone()
    cnts = []
    for _ in range(rounds):
        mid = (lo + hi) * 0.5
        c = kc(xn, mid.npu()).cpu()
        cnts.append(c)
        ge = c >= float(k)
        lo = torch.where(ge, mid, lo)
        hi = torch.where(ge, hi, mid)
    idx = ks(xn, lo.npu()).cpu()
    return idx


# ---------- Driver ----------
def run_case(m, n, k, tag, distinct=True, rounds=ROUNDS, tile=None):
    if distinct:
        # Distinct values keep the SET comparison unambiguous: with ties AT the
        # boundary, which k of the tied elements are chosen is not defined by the
        # contract, so a tie test would test an undefined choice.
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
    # tiled path: the row is streamed in pieces smaller than its width
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
