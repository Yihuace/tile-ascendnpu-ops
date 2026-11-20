# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""PROBE-topk-inkernel-isolate -- which construct makes the in-kernel loop segfault?

The full in-kernel bisection probe dumped core.  The loop there contained a runtime
`T.if_then_else` on (1,1) UB buffers, which the earlier working `loopcount` probe did NOT
use (that one only did vcmp/vselect/reduce inside a runtime `T.serial`).  This probe
isolates the variable with a deliberately tiny case (M=2, N=64, 4 rounds):

    --case plain      runtime T.serial loop, accumulate only          (known-good shape)
    --case ifelse     same + T.if_then_else on a (1,1) UB buffer      (suspected culprit)
    --case vselect    same, but the branch is a vcmp/vselect on (1,1) (proposed fix)

If `plain` passes and `ifelse` segfaults while `vselect` passes, the fix is mechanical:
never branch on a runtime scalar, derive the 0/1 selector with a vector compare.

Run:
    env -u PYTHONPATH <venv>/bin/python -u PROBE-topk-inkernel-isolate.py --case plain
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

M, N, ROUNDS, K = 2, 64, 4, 8


def build(case: str):
    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 out: T.Tensor((M,), "float32")):
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
                m1 = T.alloc_ub((1, 1), "uint8")
                t1 = T.alloc_ub((1, 1), "float32")
                t2 = T.alloc_ub((1, 1), "float32")
                kc1 = T.alloc_ub((1, 1), "float32")
                for j in T.Parallel(N):
                    one[0, j] = 1.0
                    zero[0, j] = 0.0
                one1[0, 0] = 1.0
                zero1[0, 0] = 0.0
                kc1[0, 0] = float(K)
                for r in T.serial(M):
                    T.copy(x[r, 0:N], row[0, 0:N])
                    T.reduce(row, mn, dims=1, reduce_mode="min")
                    T.reduce(row, mx, dims=1, reduce_mode="max")
                    lo[0, 0] = mn[0, 0]
                    hi[0, 0] = mx[0, 0]
                    for _t in T.serial(ROUNDS):
                        mid[0, 0] = (lo[0, 0] + hi[0, 0]) * 0.5
                        T.vmul(mid, -1.0, nmid)
                        T.vadd(row, nmid[0, 0], diff)
                        T.vcmp(diff, zero, mask, "ge")
                        T.vselect(mask, one, zero, sel)
                        T.reduce(sel, s, dims=1, reduce_mode="sum")
                        if case == "plain":
                            # no branch at all: just fold the count into lo so the loop
                            # carries a real dependency on s (same shape as `loopcount`)
                            lo[0, 0] = lo[0, 0] + s[0, 0] * 0.0
                        elif case == "ifelse":
                            lo[0, 0] = T.if_then_else(s[0, 0] >= float(K), mid[0, 0], lo[0, 0])
                            hi[0, 0] = T.if_then_else(s[0, 0] >= float(K), hi[0, 0], mid[0, 0])
                        else:  # vselect: derive the 0/1 flag with a vector compare
                            # Everything stays in the VECTOR pipe: a scalar read of a
                            # buffer just written by a vector op is a cross-pipe
                            # dependency the compiler does not order, which is why the
                            # first version of this branch only took effect for round 1
                            # (it produced the round-1 midpoint 220.5 instead of 385.875).
                            T.vadd(lo, hi, t1)             # t1 = lo + hi
                            T.vmul(t1, 0.5, mid)           # mid = (lo + hi) / 2
                            T.vmul(mid, -1.0, nmid)
                            T.vadd(row, nmid[0, 0], diff)
                            T.vcmp(diff, zero, mask, "ge")
                            T.vselect(mask, one, zero, sel)
                            T.reduce(sel, s, dims=1, reduce_mode="sum")
                            T.vsub(s, kc1, s)              # s = count - k
                            T.vcmp(s, zero1, m1, "ge")
                            T.vselect(m1, one1, zero1, s)  # s := 1 if count>=k else 0
                            T.vsub(mid, lo, t1)            # t1 = mid - lo
                            T.vmul(t1, s[0, 0], t1)        # t1 = (mid-lo)*flag
                            T.vadd(lo, t1, lo)             # lo = lo + (mid-lo)*flag
                            T.vsub(hi, mid, t2)            # t2 = hi - mid
                            T.vmul(t2, s[0, 0], t2)        # t2 = (hi-mid)*flag
                            T.vadd(mid, t2, hi)            # hi = mid + (hi-mid)*flag
                    out[r] = lo[0, 0]
        return main
    return _f()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["plain", "ifelse", "vselect"])
    a = ap.parse_args()

    torch.manual_seed(0)
    x = (torch.randperm(N, dtype=torch.float32) * 7.0).unsqueeze(0).repeat(M, 1)
    try:
        k = build(a.case)
        out = k(x.npu()).cpu()
        torch.npu.synchronize()
        print(f"  RESULT {a.case:<8} COMPILED AND RAN  out={out.tolist()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
        print(f"  RESULT {a.case:<8} FAILED {type(exc).__name__}", flush=True)
        for ln in lines[-4:]:
            print(f"    | {ln[:150]}", flush=True)


if __name__ == "__main__":
    main()
