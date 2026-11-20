# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""PROBE-topk-fused -- can bisection and selection live in ONE kernel?

WHY
---
The delivered op issues TWO tilelang kernels per call and both prim_funcs are
named `main`, so `msprof op --kernel-name=main` matches both:

    sample_count = 2   min_latency_us = 328.77   max_latency_us = 1943.96

`msprof.py` then does `durations.sort(); median = d[len//2]`, which with len=2 is
the SLOWER of two DIFFERENT kernels, and `_parse_msprof_roofline` takes
`candidates[0]` -- the FIRST op group, i.e. the bisect kernel. So the reported
latency is one kernel and the reported Ratio is the other; neither is the op.
Measured op total for 8x8192 is 328.77 + 1943.96 = 2272.7 us against a reported
1943.96 us.

The only way the harness's contract ("N samples of the SAME kernel") holds is for
the op to launch exactly ONE kernel. This probe checks the precondition: after the
40-round loop the bisect kernel's `diff`, `sel`, `one`, `zero` and `mask` are all
dead, so the selection can run in the same kernel over the same UB-resident row.

UB budget: 5 fp32-wide + 1 uint8 + 1 int32-wide = 25 B/element
    N<=4096 -> 100 KB (fits 192 KB)
    N= 8192 -> 200 KB (DOES NOT FIT -- 8192 must keep two launches)

Run:
    env -u PYTHONPATH <venv>/bin/python -u PROBE-topk-fused.py --N 256 --k 8
"""

import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

_TILEOPS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *([os.pardir] * 6)))
if _TILEOPS_ROOT not in sys.path:
    sys.path.insert(0, _TILEOPS_ROOT)

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401


def build_fused(M, N, K, rounds):
    """Bisection then selection, one kernel, one launch.

    After the round loop `lo` holds a threshold with >= K elements at or above it.
    The select phase reuses the dead buffers:
        diff  -> spos    sel -> posf    one -> krow    zero -> free
    and adds one int32-wide buffer for the cast scatter index.
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

                    # ---- phase 1: 40 rounds of bisection, all in-device ----
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

                    # ---- phase 2: select, reusing the dead buffers ----
                    # diff(t1) is free; use it as the negated threshold, then as spos
                    T.vmul(lo, -1.0, t1)                  # t1 = -lo
                    T.vadd(row, t1[0, 0], diff)           # diff = x - lo
                    T.vcmp(diff, zero, mask, "ge")        # mask = x >= lo
                    T.vselect(mask, one, zero, sel)       # sel = 1/0
                    T.cumsum(sel, diff, dim=1)            # diff = 1-based rank
                    T.vsub(diff, one, diff)               # 0-based rank
                    T.vmul(sel, diff, diff)               # selected keep rank, others 0
                    # unselected lanes aim at the sink slot K
                    T.vmul(sel, one, sel)                 # sel = 1/0 (recompute)
                    T.vsub(one, sel, sel)                 # sel = 0 for selected, 1 otherwise
                    T.vmul(sel, kc1[0, 0], sel)           # K for unselected
                    T.vadd(diff, sel, diff)               # diff = final slot
                    T.vcast(diff, sposi, round_mode="rint")
                    T.clear(scratch)          # T.clear takes the whole buffer, not a slice
                    for j in T.Parallel(N):
                        scratch[0, sposi[0, j]] = j
                    T.copy(scratch[0:1, 0:K], idx[r, 0:K])
        return main
    return _f()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=40)
    a = ap.parse_args()

    torch.manual_seed(0)
    x = (torch.randperm(a.N, dtype=torch.float32) * 7.0).unsqueeze(0).repeat(a.M, 1)
    ref = torch.topk(x, a.k, dim=-1).indices

    try:
        kf = build_fused(a.M, a.N, a.k, a.rounds)
        got = kf(x.npu()).cpu()
        torch.npu.synchronize()
        ok = all(set(got[i].tolist()) == set(ref[i].tolist()) for i in range(a.M))
        print(f"  FUSED COMPILED AND RAN  match={ok}", flush=True)
        if not ok:
            print(f"    got={got[0].tolist()[:8]}", flush=True)
            print(f"    ref={ref[0].tolist()[:8]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
        print(f"  FUSED FAILED {type(exc).__name__}", flush=True)
        for ln in lines[-6:]:
            print(f"    | {ln[:150]}", flush=True)


if __name__ == "__main__":
    main()
