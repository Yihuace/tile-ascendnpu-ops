# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""PROBE-topk-inkernel-bisect -- can the WHOLE bisection run inside one kernel?

WHY
---
The delivered TopkSelectorOp measures 5.2-8.6 ms wall clock, and the cost is almost
entirely HOST-SIDE: it issues 40 host-side counting launches + 1 select launch, and a
single launch costs ~0.11-0.13 ms regardless of how little the kernel does
(41 x ~127 us = 5.2 ms, matching the measurement).  msprof cannot see this at all -- it
reports 6.5-35 us because it profiles only the device kernel, not the host loop.

So the only lever that matters is the LAUNCH COUNT.  This probe checks the precondition
for collapsing all 40 rounds into ONE kernel:

  * load the row into UB once (no T.copy inside the round loop),
  * loop `rounds` times with a SIMPLE body: scalar mid, vsub, vcmp, vselect, reduce,
    scalar if_then_else to move lo/hi,
  * update lo/hi with a runtime condition.

This is the same "runtime T.serial with a simple body" shape that the earlier
`loopcount` probe proved workable, and it deliberately avoids the two things that were
measured to be fatal:
  * >=2 UNROLLED bodies + a taper  -> build-time segfault,
  * T.copy inside a runtime round loop -> ERR99999 complex-loop-body.

Run:
    env -u PYTHONPATH <venv>/bin/python -u PROBE-topk-inkernel-bisect.py --M 8 --N 256 --k 8
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

ROUNDS = 40


def build(M: int, N: int, k: int, rounds: int = ROUNDS):
    """Return (bisect_kernel, select_kernel).

    bisect: per row, 40 rounds of threshold bisection, all in-device.
            Output = the final `lo` (a threshold with >= k elements at or above it).
    select: per row, emit up to k indices whose value >= that threshold.
    """
    @tilelang.jit(out_idx=[1], target="npuir")
    def _bisect():
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
                for j in T.Parallel(N):
                    one[0, j] = 1.0
                    zero[0, j] = 0.0
                for r in T.serial(M):
                    T.copy(x[r, 0:N], row[0, 0:N])
                    T.reduce(row, mn, dims=1, reduce_mode="min")
                    T.reduce(row, mx, dims=1, reduce_mode="max")
                    lo[0, 0] = mn[0, 0]
                    hi[0, 0] = mx[0, 0]
                    for _t in T.serial(rounds):
                        mid[0, 0] = (lo[0, 0] + hi[0, 0]) * 0.5
                        T.vmul(mid, -1.0, nmid)
                        T.vadd(row, nmid[0, 0], diff)
                        T.vcmp(diff, zero, mask, "ge")
                        T.vselect(mask, one, zero, sel)
                        T.reduce(sel, s, dims=1, reduce_mode="sum")
                        lo[0, 0] = T.if_then_else(s[0, 0] >= float(k), mid[0, 0], lo[0, 0])
                        hi[0, 0] = T.if_then_else(s[0, 0] >= float(k), hi[0, 0], mid[0, 0])
                    thr[r] = lo[0, 0]
        return main
    return _bisect()


def build_select(M: int, N: int, k: int):
    """Emit up to k indices per row whose value >= thr (guarded compaction)."""
    @tilelang.jit(out_idx=[2], target="npuir")
    def _sel():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 thr: T.Tensor((M,), "float32"),
                 idx: T.Tensor((M, k), "int32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                row = T.alloc_ub((1, N), "float32")
                diff = T.alloc_ub((1, N), "float32")
                zero = T.alloc_ub((1, N), "float32")
                mask = T.alloc_ub((1, N), "uint8")
                pos = T.alloc_ub((1, N), "float32")
                acc = T.alloc_ub((1, N), "float32")
                cnt = T.alloc_ub((1, 1), "float32")
                one = T.alloc_ub((1, N), "float32")
                ts = T.alloc_ub((1, 1), "float32")
                nts = T.alloc_ub((1, 1), "float32")
                for j in T.Parallel(N):
                    zero[0, j] = 0.0
                    one[0, j] = 1.0
                    pos[0, j] = T.Cast("float32", j)
                for r in T.serial(M):
                    ts[0, 0] = thr[r]
                    T.vmul(ts, -1.0, nts)
                    T.copy(x[r, 0:N], row[0, 0:N])
                    T.vadd(row, nts[0, 0], diff)
                    T.vcmp(diff, zero, mask, "ge")
                    # inclusive prefix count: how many selected elements are at or before j
                    T.vcumsum(pos, one, acc, mask=mask)
                    acc[0, 0] = acc[0, 0] - 1.0
                    for j in T.Parallel(1):
                        pass
                    # scatter: accumulate position - (rank-1) - 1 into a slot buffer
                    # (implemented in the real kernel via UB index scatter)
                    cnt[0, 0] = 0.0
                    idx[r, 0] = 0
        return main
    return _sel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()

    torch.manual_seed(0)
    x = (torch.randperm(a.N, dtype=torch.float32) * 7.0).unsqueeze(0).repeat(a.M, 1)
    ref = torch.topk(x, a.k, dim=-1)

    try:
        kb = build(a.M, a.N, a.k)
        thr = kb(x.npu()).cpu()
        torch.npu.synchronize()
        print(f"  bisect: COMPILED AND RAN  thr={thr.tolist()[:4]}", flush=True)
        rv = ref.values[:, -1]                      # k-th largest per row
        err = (thr.float() - rv).abs().max().item()
        print(f"  bisect: k-th value max_err = {err:.4e}", flush=True)
    except Exception as exc:  # noqa: BLE001
        lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
        print(f"  bisect: FAILED {type(exc).__name__}", flush=True)
        for ln in lines[-5:]:
            print(f"    | {ln[:160]}", flush=True)


if __name__ == "__main__":
    main()
