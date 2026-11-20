# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-parallel-scalar-broadcast-segv -- reading a (1,1) UB scalar inline
inside ``T.Parallel`` segfaults the npuir codegen; hoisting it into a Python
name works.

Entry:  traps-compiler.md / TRAP-parallel-scalar-broadcast-segv
Kind:   trap (reverse experience).  Assertions:
          * BAD form  -> the compile subprocess dies with SIGSEGV (-11);
          * GOOD form -> compiles, runs, and is bit-exact vs torch.
Why the BAD form is attractive: ``buf[0,0]`` in a scalar *statement* (e.g.
``sh[0,0] = sqrt(t)/sh[0,0]``) is legal and used by the shipped engram kernel,
so the same read inside a ``T.Parallel`` body looks equivalent -- it is not, and
it crashes at codegen time (no diagnostic, no traceback, just SIGSEGV).

Fix: bind the scalar to a plain Python name *outside* the loop
(``cv = buf[0,0]``) and use that name inside ``T.Parallel``.

First verified: 2026-10-31, tilelang 0.1.2+3ed7c91420e925ff3ec388fb981dd6ebf45ff4bf
+ CANN 9.0.0 + Ascend910B3, TILELANG_ASCEND_MODE=Developer.
origin_task: examples/engram_decode_mcore/engram_decode_mcore (Stage 3)
Re-verify: python repro/TRAP-parallel-scalar-broadcast-segv.py
"""

import os
import subprocess
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

F32 = "float32"
N = 64
BAD = "--bad-form"


def _build(hoist: bool):
    @tilelang.jit(target="npuir")
    def factory():
        @T.prim_func
        def main(
            x: T.Tensor((N,), F32),
            w: T.Tensor((N,), F32),
            y: T.Tensor((N,), F32),
        ):
            with T.Kernel(1, is_npu=True) as (bid, _):
                s = T.alloc_shared((1, 1), F32)
                s[0, 0] = 2.5                     # scalar in a statement: legal
                xs = T.alloc_shared((N,), F32)
                ws = T.alloc_shared((N,), F32)
                ys = T.alloc_shared((N,), F32)
                T.copy(x, xs)
                T.copy(w, ws)
                if hoist:
                    cval = s[0, 0]                # GOOD: hoisted Python name
                    for j in T.Parallel(N):
                        ys[j] = cval * xs[j] * ws[j]
                else:
                    for j in T.Parallel(N):       # BAD: (1,1) read inline
                        ys[j] = s[0, 0] * xs[j] * ws[j]
                T.copy(ys, y)

        return main

    return factory()


if __name__ == "__main__":
    if BAD in sys.argv:
        _build(hoist=False)                       # expected: SIGSEGV
        print("BAD form compiled -- trap no longer reproduces")
        sys.exit(0)

    # 1) the BAD form must crash the compiler subprocess
    proc = subprocess.run([sys.executable, __file__, BAD],
                          capture_output=True, text=True, timeout=1800)
    assert proc.returncode < 0, (
        f"BAD form did not crash (rc={proc.returncode}); trap changed")
    assert -proc.returncode == 11, f"expected SIGSEGV, got {-proc.returncode}"
    print(f"[ok] BAD form (inline s[0,0] in T.Parallel) -> SIGSEGV "
          f"(rc={proc.returncode})")

    # 2) the GOOD form must compile, run and be exact
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32).npu()
    w = torch.randn(N, dtype=torch.float32).npu()
    y = torch.zeros(N, dtype=torch.float32).npu()
    _build(hoist=True)(x, w, y)
    torch.npu.synchronize()
    err = (y.cpu() - 2.5 * x.cpu() * w.cpu()).abs().max().item()
    assert err < 1e-6, f"hoisted form wrong: {err}"
    print(f"[ok] GOOD form (hoisted cval) max_err={err:.1e}")
    print("OVERALL: TRAP_PASS")
