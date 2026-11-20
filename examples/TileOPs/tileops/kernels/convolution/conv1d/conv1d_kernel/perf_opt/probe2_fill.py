#!/usr/bin/env python3
"""Probe: how to materialise a constant row in an Expert Vector UB tile.

P-A  ``T.fill(ub, T.Cast(dtype, 1))``            (broadcast scalar)
P-B  ``T.vmul(ub_src, 0.0, ub); T.vadd(ub, 1.0, ub)`` (zero-then-add)

Both write the result to GM; the check is compile success + all-ones content.
Run: python3 probe2_fill.py {A|B}
"""
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

DT = "float16"
N = 1024


def build(mode):
    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(src: T.Tensor((1, N), DT), out: T.Tensor((1, N), DT)):
            with T.Kernel(1, is_npu=True) as (kid, _):
                with T.Scope("Vector"):
                    ub = T.alloc_ub([1, N], DT)
                    if mode == "A":
                        T.fill(ub, T.Cast(DT, 1))
                    else:
                        T.copy(src[0:1, 0:N], ub[0:1, 0:N])
                        T.vmul(ub, 0.0, ub)
                        T.vadd(ub, 1.0, ub)
                    T.copy(ub[0:1, 0:N], out[0:1, 0:N])

        return main

    return _f


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "A"
    torch.manual_seed(0)
    src = torch.randn(1, N, dtype=torch.float16, device="npu")
    try:
        k = build(mode)()
        torch.npu.synchronize()
        out = k(src)
        mn = float(out.min().item())
        mx = float(out.max().item())
        ok = abs(mn - 1.0) < 1e-3 and abs(mx - 1.0) < 1e-3
        print(f"PROBE fill-{mode}: {'PASS' if ok else 'FAIL'} min={mn} max={mx}")
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE fill-{mode}: EXC {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
